# Five-layer DFlash / DSpark with a layer-1 side loop

Branch: `research/dflash-loop-fusion`. The supported root shell entry points
are `train_dflash_side.sh`, `train_dspark_side.sh`, `eval_dflash_side.sh`, and
`eval_dspark_side.sh`. Settings are in `scripts/side/settings.sh`.

## Architecture and token alignment

Every experiment has five physical main layers and one main-stack pass.
The same layer-1 weights are used for recurrence. The first call is shared:
`SIDE_LOOPS=5` means five total layer-1 calls and one call each to layers 2–5.

| Variant | Total positions | Predicted tokens | Side path |
|---|---:|---:|---|
| baseline | 16 | 15 | none |
| A | 16 | 15 | four additional layer-1 calls, hidden fusion |
| B | 16 | 15 | five groups of three provisional tokens, hidden fusion |

All new launches set `sample_from_anchor=False`: input
`[anchor, MASK x 15]` has a given anchor at slot 0. Only slots `1:16` are
sampled or supervised. B groups are `1:4`, `4:7`, `7:10`, `10:13`, `13:16`;
each group feeds its predicted embeddings into those same positions. The
anchor and absolute position IDs stay fixed. Group states are saved **before**
feedback and concatenated at their original positions for final fusion.
B requires `BLOCK_SIZE - 1 = SIDE_LOOPS * GROUP_SIZE`.

Legacy DeepSpec uses shifted prediction when `sample_from_anchor` is absent
or true: parameter `block_size=7` produces seven candidates and the target
verifier reads eight positions including the given anchor. Its draft tensor
has seven positions because the anchor hidden also predicts the next token.
Legacy checkpoint behavior is preserved; new launches deliberately use the
anchor-excluded convention above.

The fusion is `main + up(silu(down(concat(main, side))))`, then final norm and
head. Default rank is 128, with a zero-initialized output projection. Recurrence
shares layer weights; fusion adds `3 * hidden_size * rank` parameters.

**B's branch tokens are provisional.** Final fusion samples again, so later
branch states may reflect a different prefix from the final proposal. Target
verification uses the probabilities of the final proposal, including the
Markov correction when DSpark is enabled.

## DFlash and DSpark

| Algorithm | Markov head | Confidence head | Final-output objective |
|---|---|---|---|
| DFlash | disabled | disabled | CE |
| DSpark | vanilla, rank 256 | enabled, with Markov embedding | 0.1 CE + 0.9 probability L1 + 1.0 confidence BCE |

DSpark defaults match `config/dspark/dspark_qwen3_4b.py`. Probability L1 is the
sum of absolute differences between draft and target vocabulary probabilities,
not an error between hidden states. Confidence BCE uses detached targets
`1 - 0.5 * probability_L1` per predicted position.

DSpark's final Markov head is teacher-forced with the preceding ground-truth
token during cached training, as in the original implementation. During
inference it uses the preceding sampled proposal token. In B, the **side
branch always uses its own provisional predictions** in both training and
inference. It receives actual given anchor IDs; subsequent groups condition
on the preceding provisional group. The vanilla Markov correction also
selects the three tokens within each group sequentially. DFlash's three
within-group predictions remain parallel.

A has no intermediate head or per-loop auxiliary loss. B adds
`SIDE_AUX_ALPHA * grouped_CE` (default `0.2`) to the final objective for both
algorithms. Five groups' logits are concatenated, and a **single normalized
CE over the 15 predictions** is computed, rather than summing five group means.
DSpark's auxiliary CE does not inherit confidence or target-probability terms.

All terms use valid-position weights `exp(-j / 4)`, normalized by their total.
The anchor, invalid positions and padding are excluded. Embedding and target
LM-head weights stay frozen. Main draft layers, fusion, Markov and confidence
modules (when present) train jointly. Discrete provisional selections detach;
logit losses and final fusion still backpropagate through the shared layer.

## Training

Defaults are eight GPUs, 3% data, three epochs, micro-batch 2/GPU, global batch
64, seed 42. Sampling reserves a fixed 32-example holdout before selecting the
training percentage. All six groups use the same selected examples and budget.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_PERCENT=3 EPOCHS=3
export BLOCK_SIZE=16 SIDE_LOOPS=5 GROUP_SIZE=3
export RUN_ROOT="$PWD/runs/side_p3_e3"

bash train_dflash_side.sh baseline
bash train_dflash_side.sh A
bash train_dflash_side.sh B

bash train_dspark_side.sh baseline
bash train_dspark_side.sh A
bash train_dspark_side.sh B
```

Override `SOURCE_CACHE` if necessary. DSpark requires the cached
`target_last_hidden_states` in addition to the selected target features;
the repository's target-cache format stores both.

Use `DRY_RUN=1 bash train_dspark_side.sh B` to inspect the launch without GPU
work. For a two-step smoke test, use a separate root:

```bash
SMOKE_STEPS=2 RUN_ROOT="$PWD/runs/side_smoke" bash train_dspark_side.sh B
```

The launcher prints `checkpoint_root`; final model weights are at its
`step_latest`. Only the final model is saved. For warm-starting, set
`INIT_CHECKPOINT=/path/to/step_latest` to a five-layer, single-main-pass
checkpoint of the **same algorithm**. Changing block size and prediction
convention is explicitly allowed, but label alignment changes and requires
training before making a quality comparison. Auto-resume rejects mismatched
block, side configuration, Markov or confidence heads.

For an auxiliary-loss ablation use a new run root:

```bash
SIDE_AUX_ALPHA=0 RUN_ROOT="$PWD/runs/side_aux0" bash train_dspark_side.sh B
```

## Evaluation and timing

```bash
export TARGET_MODEL=/home/jovyan/LMM/lmm_model/Qwen3-4B
export EVAL_GPU=0 TASKS=gsm8k MAX_SAMPLES=128 MAX_NEW_TOKENS=256
export WARMUP_SAMPLES=3 EVAL_REPEATS=3

bash eval_dflash_side.sh /path/to/checkpoint/step_latest both
bash eval_dspark_side.sh /path/to/checkpoint/step_latest both
```

For a baseline, `both` runs only serial. A/B compare serial and CUDA-stream
overlap on the same GPU. Algorithm-specific scripts validate checkpoint heads.
Acceptance, draft/verify timings and serial-request throughput are written to
JSON and `comparison.csv`. DSpark uses `CONFIDENCE_THRESHOLD=0` by default
(full 15-token proposal); a positive threshold enables confidence truncation.
Algorithm, heads, threshold and proposal length are recorded in results.

```bash
CONFIDENCE_THRESHOLD=0.5 bash eval_dspark_side.sh /path/to/checkpoint/step_latest both

CUDA_VISIBLE_DEVICES=0 python scripts/side/benchmark.py \
  --draft /path/to/A_or_B/step_latest --contexts 1024,4096 \
  --batches 1 --warmup 10 --repeats 50 --output runs/side_timing.json
```

Before training, adapt an existing same-algorithm five-layer checkpoint for
**timing only** with `--variant A --side-loops 5 --block-size 16`, or
`--variant B --side-loops 5 --group-size 3`. No quality claim is made for
untrained fusion.

The timing benchmark measures main-only, side-only, serial and overlap on
identical synthetic inputs. B includes intermediate vocabulary projection and
Markov selection when present. Final `output_head_ms` measures the vocabulary
head; `proposal_heads_ms` includes final Markov selection and confidence
projection. `main_stack_5l_ms` and `side_loopN_ms` exclude the common preparation
and final head. Both standalone paths include the first layer; combined
execution shares that first call, so do not add their standalone totals.

Use `overlap_overhead_vs_main_pct` to measure the actual added wall time;
`side_loopN_ms < main_stack_5l_ms` alone does not establish hidden overhead.
The post-fork window contains only layers 2–5 and concurrent streams compete
for GPU resources. Uninstrumented and component-event measurements are
separate. No synchronization occurs inside the fork/join region. Synthetic
cost measurements are separate from complete target verification timings.

## Debugging and validation

- `scripts/side/train.py`: experiment and loss options.
- `deepspec/trainer/dspark_trainer.py`, `run_batch`: final and auxiliary loss.
- `deepspec/modeling/dspark/side_loop.py`, `side_branch`: group hidden, provisional IDs and feedback.
- Same file, `run_paths`: main/side hidden fusion and fork/join scheduling.
- `deepspec/eval/dspark/draft_ops.py`, `build_dspark_proposal`: final proposal, Markov probabilities and confidence.

Both algorithms use the shared `dspark` model directory; the selected heads
and loss configuration determine which algorithm is active. For interactive
breakpoints use serial evaluation and a single request/sample.

```bash
python -m pytest -q
```

CPU tests exercise all six launch plans, actual DSpark objectives and gradients,
provisional-prefix alignment across blocks, checkpoint behavior and target AR
output equivalence. GPU tests check serial/overlap hidden and cache equality
for both algorithms and require a CUDA host.
