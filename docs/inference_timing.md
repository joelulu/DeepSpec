# Eight-way draft inference timing

This diagnostic uses a **trained A/fusion plain DFlash checkpoint** for all
groups. It does not train, save weights, or compare generation quality.

| Group | Mode | Backbone | Final projection |
| --- | --- | --- | --- |
| A | `main_only` | Layer1 to Layer5 | norm, head, greedy token selection |
| B | `side_only` | Layer1 called five times | norm, head, greedy token selection |
| C | `dual_stream` | first Layer1 shared, remaining four calls on each branch | fusion, norm, head, greedy token selection |
| D | `fusion_only` | Layer1 to Layer5, preserve H1 | fusion of H5 and H1, norm, head, greedy selection |

These group letters do not rename the earlier training variants. Each mode
runs with both `eager` and `cudagraph`, for eight groups. B has no intermediate
head or token feedback. `--side-loops 5` includes the first Layer1 call.

## Run

Activate the existing DeepSpec environment. Expose exactly one GPU. A quick
smoke checks the same eight groups with one warmup and two measured calls:

```bash
export CUDA_VISIBLE_DEVICES=0
export DRAFT="$PWD/runs/dflash_side_p3_e3si/deepspec/dflash_side_A_maskonly_b16_r5_g3_k128_p3_e2_seed42/step_latest"

python scripts/side/benchmark_inference.py \
  --target /home/jovyan/LMM/lmm_model/Qwen3-4B \
  --draft "$DRAFT" --task gsm8k \
  --warmup-calls 1 --measure-calls 2 \
  --output runs/inference_timing_smoke
```

Formal run (five warmup reference states, fifty measured reference states):

```bash
python scripts/side/benchmark_inference.py \
  --target /home/jovyan/LMM/lmm_model/Qwen3-4B \
  --draft "$DRAFT" --task gsm8k \
  --block-size 16 --side-loops 5 \
  --warmup-calls 5 --measure-calls 50 \
  --context-buckets 1024,4096 \
  --profile-trace \
  --output runs/inference_timing_8groups
```

Use a new output directory on each invocation; existing results are never
silently appended or overwritten. If graph capture fails, the command fails
with the mode/exact shape and original exception; it never substitutes eager for a
graph row. `--backends eager` is available for diagnosis. Raise the largest
context bucket if a real prefix is too long. Prefixes are never truncated.
`--max-samples 128` is only a safety cap, not a request to finish 128 samples.
If too few calls are collected, the command fails and retains partial results.

## Matched real inputs and timing scope

A normal speculative reference decode supplies target hidden features,
anchor IDs, positions, and the existing context-only draft cache. Reference
proposals always use main-only eager on the original unpadded cache. The
observer is called before that forward appends to the cache. All groups see
the same state; their predictions never change the reference acceptance or
next prefix. Each group is executed once per reference state, with randomized
group order. Each measured call is synchronized individually: 50 calls are
**50 repetitions at batch=1, not 50 concurrent requests**. At the final state,
the observer stops before another reference proposal/verification is needed.

Measured forward includes embedding, projection of newly available context
features, old/new KV concatenation, backbone, normalization, vocabulary head
and greedy selection. Target prefill/verification, input snapshot copies,
graph setup, validation, component diagnostics, and profiler runs are excluded.
This is not end-to-end generation throughput. B projects only the one physical
layer it needs; the other modes project all five.

Graphs require stable tensor addresses and shapes. Every snapshot now has
**exactly the original cached length and new-feature length**. No context or
feature padding and no extra attention mask are introduced. This preserves the
GEMM/SDPA shapes and the original first-call KV layout. Whenever the pair of
lengths changes, the current graph set is released and all selected graph modes
are recaptured. The diagnostic may therefore take longer to finish. Setup is
reported separately and is not part of measured forward latency. This measures
graph replay benefits, not the cost of deploying a graph for every decode step.

The legacy `--context-buckets` argument remains accepted as an upper prefix
limit and a grouping for profiler/component diagnostics. It does not pad any
input or affect compute shapes. Every state checks main-only snapshot hidden
and greedy tokens against the original incremental forward, using the original
elementwise hidden tolerance (`rtol=0.02, atol=0.02`) and exact token equality.
There is no BF16 outlier exception. Every mode also checks eager/graph tokens.
Validation results, including failures, are saved to `input_validation.jsonl`.

Graph setup includes extra uncaptured warmups and capture/replay per shape.
Those are outside the requested five warmup plus fifty measured observations.
Only the current shape's graph set is retained, limiting graph memory growth.

## Results

- `summary.csv`: mode/backend, actual calls, context range, mean/median/P90 of
  wall and GPU time, dispatch time, and overhead against **the same backend's**
  main-only baseline.
- `calls.jsonl`: each measured forward, matched reference-call ID and exact context sizes, wall/GPU/host-submit time, and separately measured snapshot
  copy time. One shared input copy time is repeated across rows for that state;
  do not sum it eight times as actual harness work.
- `metadata.json`: completion/error status, model/target/git revision, hardware,
  call budgets, scope, and graph setup costs. Check `status=complete`.
- `components.json`: separate instrumented eager calls, including shared L1,
  main tail, side tail, fork/join span, fusion and head. Overlapping branch
  times cannot be summed to obtain total time.
- `dual_stream_*.trace.json` with `--profile-trace`: separate CPU/CUDA traces
  for eager and graph on the first measured state in each configured context range.
  Open in a compatible trace viewer to inspect actual kernel overlap.

`host_submit_ms` includes Python/driver dispatch through end-event recording;
it is not pure CPU compute. CUDA Event spans may include gaps while the host
submits work. Neither field alone identifies a resource bottleneck.

The current C schedule remains side-first host submission, with correct
cross-stream fork/join dependencies. Graph versus eager is the controlled
change. Interleaved layer submission is a separate follow-up experiment.

## Checks

```bash
python -m pytest -q tests/test_draft_timing.py tests/test_side_loop.py
```

CPU checks cover cache isolation, exact-shape incremental equivalence in FP32/BF16,
layer counts, H1 fusion, stop-hook behavior, CLI validation and averaging.
Five added CUDA cases exercise graph replay with updated inputs and the exact
warmup/measurement budget. They are skipped on hosts without a CUDA GPU.
