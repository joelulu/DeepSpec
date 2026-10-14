# Five-layer DFlash with a layer-1 side loop

Branch: `research/dflash-loop-fusion`, based on `research/loopd-flash` at
`29c7031a9991689f3121fe5f8b82e5614d71742b`.

## Experiments and exact token convention

All experiments retain **five physical main layers**, and `num_loops=1` for
the main stack. Side recurrence calls the **same layer-1 parameters**, not a
copy. The initial layer-1 forward is shared; `SIDE_LOOPS=5` means five total
calls to layer 1 and one call each to layers 2–5 (nine layer calls in total).

| Experiment | Default block | Side path | Final prediction |
|---|---:|---|---|
| baseline | 16 (anchor + 15 predictions) | none | main hidden → norm → head |
| A | 16 (anchor + 15 predictions) | layer 1, five total calls, no intermediate head | residual MLP fusion of final side/main hidden → norm → head |
| B | 16 (anchor + 15 predictions) | five calls, each predicts three provisional tokens and feeds back embeddings | fusion of each group's pre-feedback hidden/main hidden → norm → head |

These experiments explicitly set `sample_from_anchor=False`: input
`[anchor, MASK, …]` has 16 positions and only positions `1:16` predict tokens.
Position 0 is given, excluded from the loss, and never sampled. Group 1 uses
positions `1:4`; its predicted embeddings are fed back into those same positions.
The anchor stays at input slot 0. Positions do not slide.

Legacy DeepSpec checkpoints keep their shifted prediction convention when the
field is absent or true: the default parameter `block_size=7` produces seven
draft candidates, and the target verifier reads eight positions including the
given anchor. Its draft tensor itself has seven positions because slot 0 also
predicts the next token. Do not confuse that parameter with the total-position
convention used by the new side-loop launcher.

In B, the first shared call already emits group 1. Thus five calls × three
positions = 15 proposals. This differs from a schedule with an initialization
call followed by five decoding calls, which would require six total calls.
Groups decode internally in parallel; dependence is **between groups**.

**B's intermediate tokens are provisional.** Final fusion may select different
tokens, so this is a semi-autoregressive auxiliary branch, not a guarantee of
strict AR factorization over the final output tokens. Provisional choices are
greedy and deterministic even if final proposal sampling uses temperature > 0.
The verifier receives probabilities from the **final fused head**, never the
provisional group logits. Both schedules use the unchanged target verifier.

The fusion is `main + up(silu(down(concat(main, side))))`, with rank 128 by
default and a zero-initialized output projection. A/B therefore initially
preserve the corresponding main output. This MLP adds `3 * hidden_size * rank`
parameters; recurrence adds no additional layer parameters. Training uses the
same cached target data. B feeds back its **own predictions in training**, not
gold tokens, and adds a weighted CE on pre-feedback group logits (default 0.2).
There is no gold-token leakage through feedback. The early group loss helps
train layer-1 outputs to use the frozen target head.

### Training losses

All three variants use hard-token cross-entropy with the existing position
decay, `w[j] = exp(-j / 4)`, normalized by the sum of valid token weights.
Only predicted positions `1:16` participate; padding, invalid positions, and
the anchor are excluded.

| Variant | Objective |
|---|---|
| baseline | weighted CE of the final layer-5 output |
| A | weighted CE of the final fused output; no per-loop auxiliary CE |
| B | weighted CE of the final fused output + `SIDE_AUX_ALPHA` × weighted CE of the pre-feedback group predictions |

`SIDE_AUX_ALPHA` defaults to `0.2`. B concatenates each of the five groups'
three logits at its original position, then computes one normalized auxiliary
CE across the 15 predictions. It does not sum five separately averaged losses.
Feedback uses detached greedy predictions; gradients still flow through the
group logits and fused output to the shared layer. Target embedding/head
weights stay frozen, and there is no KL, probability L1, confidence, or router
loss in these experiments. A/B train the main layers and fusion jointly.
For the auxiliary-loss ablation use
`SIDE_AUX_ALPHA=0 RUN_ROOT="$PWD/runs/dflash_side_aux0" bash train_dflash_side.sh B`;
the intermediate heads and predicted-token feedback still run. Use a separate
run root because resuming requires the original auxiliary-loss weight.

## Quick training

Activate the same environment used for existing DeepSpec experiments, then:

```bash
git fetch origin
git switch research/dflash-loop-fusion

export SOURCE_CACHE=/path/to/existing_target_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RUN_ROOT="$PWD/runs/dflash_side_pilot"
export DATA_PERCENT=1 EPOCHS=1 LOCAL_BATCH=2 GLOBAL_BATCH=64
export SIDE_LOOPS=5 GROUP_SIZE=3 FUSION_RANK=128

# Inspect launch plans first (no GPU work).
DRY_RUN=1 bash train_dflash_side.sh baseline
DRY_RUN=1 bash train_dflash_side.sh A
DRY_RUN=1 bash train_dflash_side.sh B

# Check a real forward/backward on each group before the pilot.
SMOKE_STEPS=2 RUN_ROOT="$PWD/runs/dflash_side_smoke" bash train_dflash_side.sh baseline
SMOKE_STEPS=2 RUN_ROOT="$PWD/runs/dflash_side_smoke" bash train_dflash_side.sh A
SMOKE_STEPS=2 RUN_ROOT="$PWD/runs/dflash_side_smoke" bash train_dflash_side.sh B

# Fresh pilot root, identical training sample selection and step budget.
bash train_dflash_side.sh baseline
bash train_dflash_side.sh A
bash train_dflash_side.sh B
```

The launcher prints an experiment plan and the exact `checkpoint_root`;
`step_latest` under that directory is the saved model. It also stores the plan
and training log under `RUN_ROOT/logs`. Full pilot defaults are 1% data, one
epoch, four GPUs, batch 2/GPU, global batch 64. Increase these after checking
quality. `SMOKE_STEPS` caps a run; do not mix smoke and pilot in the same root.

For a quicker warm-start from an existing **5-layer, one-main-pass** checkpoint:

```bash
export INIT_CHECKPOINT=/path/to/trained_5layer_dflash/step_latest
```

The target, dimensions, target feature layers and mask token must match. This
launcher explicitly allows changing the block size (e.g. 7 → 16), which has
no associated learned weight shape, and adds only the missing fusion weights.
It also explicitly permits changing from the legacy shifted prediction
convention to the new anchor-excluded convention. This changes label alignment;
warm-started weights must be trained before making quality claims.
Warm-starting from a 1-layer or a whole-stack multi-loop checkpoint is rejected.
Auto-resume requires matching side architecture settings and its original run.

All three defaults already have matched prediction counts:

```bash
BLOCK_SIZE=16 bash train_dflash_side.sh baseline
BLOCK_SIZE=16 bash train_dflash_side.sh A
BLOCK_SIZE=16 bash train_dflash_side.sh B
```

Primary comparison: baseline16 vs A16 vs B16, each predicting 15 new tokens.
B requires `BLOCK_SIZE - 1 = SIDE_LOOPS * GROUP_SIZE` (default `16 - 1 = 5 * 3`).
Keep training data, optimizer, steps, seed, generation task and prompt set fixed.

## Quality and complete draft–verify timing

```bash
export TARGET_MODEL=/path/to/Qwen3-4B
export EVAL_GPU=0 TASKS=gsm8k MAX_SAMPLES=128 MAX_NEW_TOKENS=256
export WARMUP_SAMPLES=3 EVAL_REPEATS=3

bash eval_dflash_side.sh /path/to/baseline/step_latest serial
bash eval_dflash_side.sh /path/to/A/step_latest both
bash eval_dflash_side.sh /path/to/B/step_latest both
```

`both` runs the serial and real two-stream overlap schedules on the same GPU.
For a baseline, it runs only serial. Evaluation defaults to temperature zero,
reports acceptance by position, acceptance length, draft/verify time, and serial
request throughput. The saved JSON/CSV includes side mode, side loop count,
schedule and block size. `RESULT_ROOT` can override each checkpoint's output
directory. This evaluator is **not** a high-concurrency serving benchmark.
For AR speedup comparison, run `eval.py --autoregressive` with the same task,
seed, sample count, generation length and warmup, saving `eval_loop0.json` under
the common comparison root, then use `scripts/loop/summarize.py` on that root.

## Does the side loop fit inside the main path's time?

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/loop/benchmark_side.py \
  --draft /path/to/A/step_latest \
  --contexts 1024,4096,16384 --batches 1,4,16 \
  --warmup 10 --repeats 50 --output runs/dflash_side_pilot/timing_A.json

CUDA_VISIBLE_DEVICES=0 python scripts/loop/benchmark_side.py \
  --draft /path/to/B/step_latest \
  --contexts 1024,4096,16384 --batches 1,4,16 \
  --warmup 10 --repeats 50 --output runs/dflash_side_pilot/timing_B.json
```

Before training, a baseline checkpoint can be adapted **for timing only**:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/loop/benchmark_side.py \
  --draft /path/to/trained_5layer_dflash/step_latest --variant A --side-loops 5 \
  --block-size 16 --contexts 1024 --batches 1 \
  --output runs/dflash_side_pilot/timing_untrained_A.json

CUDA_VISIBLE_DEVICES=0 python scripts/loop/benchmark_side.py \
  --draft /path/to/trained_5layer_dflash/step_latest --variant B --side-loops 5 \
  --group-size 3 --contexts 1024 --batches 1 \
  --output runs/dflash_side_pilot/timing_untrained_B.json
```

These runs initialize an untrained fusion module, so they make **no quality
claim**. The main path keeps its original physical layer/whole-stack loop
requirements; unsupported checkpoints fail rather than silently changing depth.

The benchmark runs four modes on identical synthetic features:

| Mode/field | Meaning |
|---|---|
| `main` | five main layers + output head |
| `side` | N calls to shared layer 1; B includes its intermediate heads; standalone output head also measured separately |
| `serial` | shared layer 1 → main tail → side path → fusion → final head |
| `overlap` | shared layer 1 → fork main tail/side on two CUDA streams → join → fusion → final head |
| `main_stack_5l_ms` | shared layer 1 + main tail, excluding output head and common preparation |
| `side_loopN_ms` | shared layer 1 + side-after-fork, B includes intermediate heads/selection/feedback |
| `prepare_ms` | context fusion, RoPE, five physical-layer context KV projections |
| `intermediate_head_ms` | B's provisional vocabulary projections and greedy selection |
| `fusion_ms`, `output_head_ms` | final fusion and final vocabulary projection |
| `wall_total_ms`, `gpu_total_ms` | entire measured draft call; includes common preparation and final head |
| `overlap_overhead_vs_main_pct` | actual added wall time of overlap relative to main alone |
| `overlap_saving_vs_serial_pct` | actual wall-time saving from overlapping the branches |

Standalone side timing includes the first layer, although that first call is
shared in the combined schedule. **Do not add main and side standalone totals**
as if the first layer ran twice. In the overlap schedule, component durations
include resource contention and cannot be summed to obtain wall time.

Uninstrumented timings are measured separately from the detailed CUDA-event
pass. Trials warm up and randomize mode order, report median/mean/p90 and raw
samples. The script checks serial/overlap token equality before measuring.
No synchronization occurs inside the fork/join region. Context projections are
shared exactly, and provisional block KV never enters the persistent cache.

`side_loopN_ms < main_stack_5l_ms` **does not establish zero overhead**: after
the first layer, overlap has only the layers-2–5 window, and both streams compete
for GPU resources. Use `overlap_overhead_vs_main_pct`, repeated trials and the
real decode evaluation to decide. Loop/head weights may be reread from HBM;
parameter sharing does not guarantee bandwidth-free recurrence. Synthetic
features measure kernel/scheduling cost; actual target verification and serving
load must be measured separately. Reduce batches/contexts if the benchmark OOMs.

## Validation and debugging

```bash
python -m pytest tests/test_side_loop.py tests/test_loop_qwen3.py -q
```

CPU tests cover unchanged initial output, exact layer/head counts, legacy shifted and new same-position group
feedback across multiple blocks, gradients including activation checkpointing,
checkpoint roundtrip, incremental context reuse, future-context masking and
greedy target-output equivalence. Two additional tests require CUDA to check
serial/overlap hidden and cache equivalence. They are skipped on a CPU host.

Breakpoints: `side_loop.py:side_branch` at `group = ...` to inspect pre-feedback
states and at `token_inputs = ...` to inspect token feedback;
`run_paths` at `model.side_fusion(main, side)` to inspect the two hidden tensors.
Training always runs serial; overlap is explicitly CUDA inference-only.
