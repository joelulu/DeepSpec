# Frozen-checkpoint LoopCD experiment

LoopCD changes only the draft readout. It does not train the checkpoint, add a
gate, change the recurrent state, or reinject predicted tokens. The original
evaluation path remains the default. Supported models: looped Qwen3 DFlash with
both Markov and confidence heads disabled.

For a fixed final loop R and early loop k < R, use the same frozen LM head on
both normalized exits. Score each token by `z_R - lambda * z_k`. This is
equivalent, after renormalization, to `log p_R - lambda * log p_k`. Keep only
tokens with `p_R >= alpha * max(p_R)` using the unmodified final logits. See
[LoopCD](https://arxiv.org/html/2609.24196v1). Acceptance improvement on DFlash
is an experimental hypothesis, not an established result from this paper.

The backbone runs once per block. Context K/V are projected and cached once;
early exit normalization does not replace the recurrent state. Only the early
and final exits receive LM head calls. Sampling and rejection verification use
the same modified proposal distribution, including temperature and filtering.
The normal draft profiling scope includes exit collection, both heads, masking,
and sampling, so extra readout cost is counted.

## Quick comparison: 1L x 5loop, block size 7

Use your existing checkpoint: its configuration supplies block size and layer
count. Expose one GPU and set the actual paths:

```bash
export CUDA_VISIBLE_DEVICES=0
export TARGET_MODEL=/path/to/Qwen3-4B
MAX_SAMPLES=16 REPEATS=3 bash eval_dflash_loopcd.sh /path/to/1l_loop5/step_latest 5
```

This runs the unmodified loop5 baseline and early-loop1 LoopCD with lambda
0.1/0.2/0.3, alpha 0.1, temperature 0. No checkpoint weights are saved. Dataset
subset and sampling seeds are identical across runs; treatment order is shuffled
and every treatment has its own warmup. The baseline does not collect early
exits or pay for an extra head. Set `DRY_RUN=1` to inspect commands without a GPU.
Results go to a new directory under `runs/loopcd/`; set `RESULT_DIR` explicitly
if desired (the directory must not already exist).

For early-loop and lambda sweeps, use the Python entry point:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/loop/loopcd_sweep.py \
  --target "$TARGET_MODEL" --draft /path/to/1l_loop5/step_latest \
  --num-loops 5 --early-loops 1 2 3 --lambdas 0.1 0.2 0.3 \
  --tasks gsm8k --max-samples 128 --repeats 3 \
  --output-dir runs/loopcd/validation_128
```

Use a separate validation sample set/task or seed to select settings before a
held-out final comparison. Sixteen samples are a pilot, not strong evidence of
a throughput gain. Do not modify the checkpoint during a sweep.

`comparison.csv` contains per-repeat measurements and deltas against that
repeat's original baseline; `summary.csv` contains medians across repeats.
Check `accepted_draft_delta`, `decode_tps_gain_pct`, and
`generation_tps_gain_pct`. A higher acceptance length with lower throughput is
not a speed improvement. `draft_ms_per_round` measures average draft cost per
verification round, including LoopCD overhead. This is serial decoding, not
multi-request serving throughput.

The repository's `accept_length` metric usually includes the target bonus
token (except an accepted EOS termination). `accepted_draft_per_round` instead
sums cumulative products of conditional prefix acceptance rates and excludes
that bonus, including rounds terminated by an accepted EOS.

## Single evaluation

```bash
python eval.py --target_name_or_path "$TARGET_MODEL" \
  --draft_name_or_path /path/to/1l_loop5/step_latest \
  --tasks gsm8k --max-samples 16 --max-new-tokens 512 \
  --temperature 0 --seed 42 --num-loops 5 --confidence-threshold 0 \
  --loopcd --loopcd-early-loop 1 --loopcd-lambda 0.2 --loopcd-alpha 0.1 \
  --profile --warmup-samples 1 --output-json runs/loopcd/single/eval_loop5.json
```

Omit `--loopcd` for the original readout. Lambda zero with LoopCD enabled is an
overhead/control experiment: greedy proposals match the baseline, but a
positive-temperature proposal still changes because of the plausibility mask.

## Correctness checks

```bash
python -m pytest tests/test_loopcd.py tests/test_loop_qwen3.py -q
```

Tests cover readout math/masking, actual stochastic proposal probabilities,
context-cache reuse, unchanged final hidden states, disabled-path head count,
and target-equivalent greedy speculative generation with a tiny CPU model.
They cannot establish Qwen3-4B acceptance or GPU throughput gains.
