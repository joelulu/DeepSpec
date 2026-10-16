# Matched draft inference timing and loop sweeps

This diagnostic uses a **trained A/fusion plain DFlash checkpoint** for all
groups. It does not train, save weights, or compare generation quality.

| Group | Mode | Backbone | Final projection |
| --- | --- | --- | --- |
| A | `main_only` | Layer1 to Layer5 | norm, head, greedy token selection |
| B | `side_only` | Layer1 called five times | norm, head, greedy token selection |
| C | `dual_stream` | first Layer1 shared, remaining four calls on each branch | fusion, norm, head, greedy token selection |
| D | `fusion_only` | Layer1 to Layer5, preserve H1 | fusion of H5 and H1, norm, head, greedy selection |
| E | `serial_dual` | first Layer1 shared; main tail then side tail on one stream | same fusion, norm, head, greedy selection as C |

These group letters do not rename the earlier training variants. Each mode
runs with both `eager` and `cudagraph`, for ten groups by default. B has no intermediate
head or token feedback. `--side-loops 5` includes the first Layer1 call.

## Run

Activate the existing DeepSpec environment. Expose exactly one GPU. A quick
smoke checks the same ten groups with one warmup and two measured calls:

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
  --output runs/inference_timing_10groups
```

## Focused loops=2 stage diagnosis

This test selects only `main_only`, `dual_stream`, and `side_only`, with total
Layer1 count fixed at two. The dual path is one shared L1, followed by main
L2–5 versus **one extra L1**, then fusion and output. Side-only performs L1 twice.
It retains the existing schedule so instrumentation diagnoses the current code.

```bash
export CUDA_VISIBLE_DEVICES=7
export DRAFT="$PWD/runs/dflash_side_p3_e3si/deepspec/dflash_side_A_maskonly_b16_r5_g3_k128_p3_e2_seed42/step_latest"
export TIMING_OUT="$PWD/runs/timing_loop2_stages_$(date +%Y%m%d_%H%M%S)"

python scripts/side/benchmark_inference.py \
  --target /home/jovyan/LMM/lmm_model/Qwen3-4B \
  --draft "$DRAFT" --task gsm8k \
  --block-size 16 --side-loops 2 \
  --modes main_only,dual_stream,side_only \
  --backends cudagraph --stage-timings modules \
  --warmup-calls 5 --measure-calls 50 \
  --context-buckets 1024,4096 \
  --no-component-timings --profile-trace \
  --output "$TIMING_OUT"
```

For a smoke run change the warmup/measurement budgets to 1/2, using a fresh
output directory. The formal run uses the same 50 measured reference states
for all three paths. Each state has one uninstrumented forward and one separate
instrumented forward per path; capture warmups/setup remain outside both budgets.

`--stage-timings` accepts `branches`, `layers`, and `modules` (default `none`).
Each level includes the preceding level. Branches measures prepare, shared L1,
main tail, side tail, fork/join, final norm, fusion, head and sampling. Layers
adds embedding, feature projection, RoPE, each context KV preparation and each
decoder-layer call. Modules adds context project/append, Q projection/normalization,
block KV projection/RoPE, KV concatenation, attention adapter/kernel, output
projection, input/post-attention norm and MLP. It does not split MLP's internal
linear operations. Module prefixes identify `shared_l1`, `main_l2` through
`main_l5`, and `side_repeat1_l1`.

Timing uses reusable `torch.cuda.Event(enable_timing=True, external=True)`
event nodes inside **a separate CUDA Graph**. Events are initialized during
warmup; timestamps are read only after replay completes. No stage synchronizes
the CPU or waits for the other branch. The existing fork/join dependencies are
preserved. External events are required to retain record nodes during graph
capture; see [PyTorch Event documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html).

Results added by this option:

| File | Content |
| --- | --- |
| `summary.csv` | Original uninstrumented total latency, three rows in this test |
| `stage_summary.csv` | Mean/median/P90 for every stage across the 50 diagnostic calls |
| `stage_comparison.csv` | Dual minus corresponding single-path stage; main stages compare with main-only, side-repeat stages compare with side-only |
| `stage_calls.jsonl` | Per-call durations and begin/end offsets from that forward's total start; matched reference-call ID |

Important rows to inspect:

- `main_tail` and `main_l2`–`main_l5`: do the same four layers slow down when
  the side branch is present? Inspect `.mlp`, `.attention_kernel`, etc. beneath
  the layers to locate the increase.
- `side_repeat1_l1`: compare the extra L1 in dual-stream with the second L1
  in side-only. This isolates the recurrence instead of comparing full forwards.
- `side_late_ms`: positive difference between side-end and main-end timestamps.
  Zero means the side ends before the main tail in the instrumented schedule.
- `join_wait`: main-stream elapsed time around the existing side-stream wait,
  after the main tail. Includes event/synchronization scheduling overhead and
  need not be exactly zero even if side-late is zero.
- `main_start_after_fork_ms`, `side_start_after_fork_ms`: delayed branch starts.
- `instrumentation_extra_gpu`: instrumented GPU span minus the matched
  uninstrumented GPU span. This is the perturbation check, not branch work.

Event nodes can change kernel scheduling. Judge original overhead using
`summary.csv`; stage results explain the instrumented replay. If instrumentation
inflation is material compared with the overhead under investigation, repeat
with `--stage-timings branches` in a new directory before attributing a cause.
Nested parent/child stages and overlapping branches **must not be summed**.
`branch_span_overlap_ms` is overlap of elapsed branch spans, including any idle
intervals; it is not simultaneous kernel-active time.

With stage timing enabled, `--profile-trace` saves **uninstrumented** traces for
all selected modes, enabling the three paths to be compared without event-node
perturbation. The older `components.json` remains a separate, one-state eager
diagnostic and can be disabled as in this command. It is not the source of the
new 50-call graph stage averages. Fusion timing now excludes the separately
reported `final_norm` stage.

Code locations:

- `scripts/side/benchmark_inference.py`: CLI and real-reference decoding.
- `deepspec/eval/dspark/draft_timing.py`: snapshots, graph capture, replay event
  timer, stage statistics and paired-path comparisons.
- `deepspec/modeling/dspark/side_loop.py`: shared L1, branch scheduling, join wait,
  individual layer scopes, fusion and final norm.
- `deepspec/modeling/dspark/qwen3/modeling.py`: attention/MLP module scopes.

## CUDA Graph loop sweep and serial control

Run both new tests together on one matched set of reference inputs:

```bash
python scripts/side/benchmark_inference.py \
  --target /home/jovyan/LMM/lmm_model/Qwen3-4B \
  --draft "$DRAFT" --task gsm8k \
  --block-size 16 --loop-sweep 1,2,3,4,5 \
  --modes main_only,side_only,dual_stream,fusion_only,serial_dual \
  --backends cudagraph \
  --warmup-calls 5 --measure-calls 50 \
  --context-buckets 1024,4096 --profile-trace \
  --output runs/inference_timing_graph_loop_sweep
```

This produces **25 rows**, each with 50 measured calls after five warmup states.
All loop counts and modes see the same 50 states in one reference decode;
measurement order is shuffled across all groups. `--loop-sweep` overrides
`--side-loops`. Loops count total Layer1 evaluations, including the first shared
call: loops=1 means zero extra Layer1 calls, loops=5 means four extra calls.
Main-only and fusion-only ignore the loop count; their baselines are measured
again for each count to expose variation rather than reusing one timing number.
The model's loop configuration is never changed between groups.

E executes the exact same physical layers, context projections, fusion and
output heads as C, sharing the first L1 only once. E finishes main L2–5, then
executes the side L1 repeats on the same stream. C forks the two tails onto two
streams. Both use the same requested backend. Serial/dual hidden states and
exact greedy tokens are checked during setup; greedy tokens are also checked
on every timed state. The setup record lists validated loop counts.

Compare dual-stream `overhead_wall_pct` with zero to assess how much additional
latency remains relative to main-only. Small differences require checking
median/P90 and repeatability. Compare `saved_vs_serial_wall_pct` to assess the
benefit of parallel scheduling for **equal work**: positive means C is faster
than E; negative means C is slower. This is distinct from overhead versus A.
At loops=1, E and D have identical math; C still exercises stream fork/join.

Use `--backends eager,cudagraph` for all 50 groups if an eager comparison is
also needed. To reproduce the original eight groups, select
`--modes main_only,side_only,dual_stream,fusion_only` with one loop count.

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

- `summary.csv`: loops/mode/backend, actual calls, context range, mean/median/P90
  of wall and GPU time, dispatch time, and overhead against the main-only
  baseline with **the same loops and backend**. Dual-stream rows also include
  `saved_vs_serial_wall_ms`, `saved_vs_serial_wall_pct`, and `speedup_vs_serial`.
  Saving is serial minus dual; percent saving divides by serial latency.
- `calls.jsonl`: each measured forward, loop count, matched reference-call ID and exact context sizes, wall/GPU/host-submit time, and separately measured snapshot
  copy time. One shared input copy time is repeated across rows for that state;
  do not sum it once per group as actual harness work.
- `metadata.json`: completion/error status, model/target/git revision, hardware,
  call budgets, scope, and graph setup costs. Check `status=complete`.
- `components.json`: separate instrumented eager calls, including shared L1,
  main tail, side tail, fork/join span, fusion and head. Overlapping branch
  times cannot be summed to obtain total time.
- `dual_stream_*_loops*.trace.json` and `serial_dual_*_loops*.trace.json` with
  `--profile-trace`: separate CPU/CUDA traces for each selected loop count and
  backend on the first measured state in each configured context range.
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
layer counts for loops 1–5, H1 fusion, loop override isolation, stop-hook
behavior, CLI validation, grouping and serial-saving calculations.
CUDA cases exercise graph replay with updated inputs at loops 1/3/5, serial
equivalence, matched loop-sweep states and the exact warmup/measurement budget. They are skipped on hosts without a CUDA GPU.
