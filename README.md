# DeepSpec side-loop experiments

This branch implements matched five-layer **DFlash and DSpark** experiments.
Every block has **one given anchor and 15 predicted tokens** (`BLOCK_SIZE=16`).
Old whole-stack-loop, Ouro, LoopCD and matrix shell launchers have been removed
from this branch. Their earlier versions remain in Git history and on
`research/loopd-flash`.

| Variant | Main path | Layer-1 side path |
|---|---|---|
| baseline | five layers | none |
| A | five layers | five shared calls; hidden fusion, no intermediate head |
| B | five layers | five shared calls; three provisional tokens per call, then hidden fusion |

The first layer-1 call is shared by both paths. DFlash disables Markov and
confidence heads; DSpark enables the original vanilla Markov head (rank 256)
and confidence head. Both algorithms keep the same anchor-excluded alignment.

## Training

Activate your existing DeepSpec environment. Default settings in
[`scripts/side/settings.sh`](scripts/side/settings.sh) are **eight GPUs, 3%
data, three epochs**, batch 2/GPU and global batch 64. `SOURCE_CACHE` points to
the existing target cache and can be overridden.

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

These are six independent commands; run the groups you need. Algorithm names
are included in output directories to keep DFlash and DSpark checkpoints
separate. Each launcher prints its exact `checkpoint_root`; the trained model
is at `checkpoint_root/step_latest`.

## Evaluation

```bash
export TARGET_MODEL=/home/jovyan/LMM/lmm_model/Qwen3-4B
export EVAL_GPU=0 TASKS=gsm8k
export MAX_SAMPLES=128 MAX_NEW_TOKENS=256
export WARMUP_SAMPLES=3 EVAL_REPEATS=3

bash eval_dflash_side.sh /path/to/dflash_baseline/step_latest serial
bash eval_dflash_side.sh /path/to/dflash_A/step_latest both
bash eval_dflash_side.sh /path/to/dflash_B/step_latest both

bash eval_dspark_side.sh /path/to/dspark_baseline/step_latest serial
bash eval_dspark_side.sh /path/to/dspark_A/step_latest both
bash eval_dspark_side.sh /path/to/dspark_B/step_latest both
```

`both` compares serial branches and actual CUDA-stream overlap. These are
serial-request benchmarks on one GPU. DSpark defaults to
`CONFIDENCE_THRESHOLD=0` to compare the full 15-token proposal; set a positive
threshold to test confidence-based truncation. Evaluation reads architecture
from the checkpoint and rejects an algorithm/checkpoint mismatch.

## Timing and implementation

For matched **real inference inputs**, five paths (including equal-work serial
and dual-stream branches) x eager/CUDA Graph, with 5 warmup and 50 measured
draft calls, see [inference timing and loops 1–5 sweeps](docs/inference_timing.md).
For loops=2 with only main/dual/side paths and per-module graph timing, see
[focused stage diagnosis](docs/inference_timing.md#focused-loops2-stage-diagnosis).

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/side/benchmark.py \
  --draft /path/to/A_or_B/step_latest \
  --contexts 1024,4096 --batches 1 --output runs/side_timing.json
```

This measures the five-layer main path, layer-1 recurrence, serial total and
actual overlapping total. DSpark timings include Markov selection and
confidence projection. Inspect `main_stack_5l_ms`, `side_loopN_ms`,
`proposal_heads_ms` and `overlap_overhead_vs_main_pct`.

See [exact token alignment, losses and debugging](docs/dflash_side_loop.md).
Training/evaluation launchers and timing utilities are under `scripts/side/`;
the common configuration is `config/side/qwen3_4b.py`. Both algorithms share
`deepspec/modeling/dspark/qwen3/modeling.py`; DFlash is its head-disabled
configuration. The actual side recurrence is in
`deepspec/modeling/dspark/side_loop.py`.

## Data preparation and environment

Use your existing target cache. If creating a new one, see
[scripts/data/README.md](scripts/data/README.md). The two data preparation shell
scripts are retained. Dependencies are listed in `requirements.txt`.

## License

DeepSpec is released under the [MIT License](./LICENSE). It includes code adapted
from third-party projects under their own licenses; see [NOTICE](./NOTICE) for the
full attribution.

## Acknowledgements

DeepSpec builds on the ideas and code of several excellent open-source projects:

- [SpecForge](https://github.com/sgl-project/SpecForge) (Apache-2.0) — the overall training framework and Eagle3 implementation; portions of the Eagle3 modeling, loss, optimizer, attention, and evaluation code are adapted from it. Adapted files carry an in-file attribution comment, and the full notice is recorded in [NOTICE](./NOTICE).
- [DFlash](https://github.com/z-lab/dflash) (MIT) — the DFlash draft-model design and training recipe.
- [Qwen3](https://github.com/QwenLM/Qwen3) and [Gemma](https://github.com/google-deepmind/gemma) — the target model families supported in this repo.

We thank the authors and maintainers of these projects. Contributions of new algorithms are welcome.

## Citation

```bibtex
@misc{cheng2026dsparkconfidencescheduledspeculativedecoding,
      title={DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation}, 
      author={Xin Cheng and Xingkai Yu and Chenze Shao and Jiashi Li and Yunfan Xiong and Yi Qian and Jiaqi Zhu and Shirong Ma and Xiaokang Zhang and Jiasheng Ye and Qinyu Chen and Chengqi Deng and Jiping Yu and Damai Dai and Zhengyan Zhang and Yixuan Wei and Yixuan Tan and Wenkai Yang and Runxin Xu and Yu Wu and Zhean Xu and Xuanyu Wang and Muyang Chen and Rui Tian and Xiao Bi and Zhewen Hao and Shaoyuan Chen and Huanqi Cao and Wentao Zhang and Anyi Xu and Huishuai Zhang and Dongyan Zhao and Wenfeng Liang},
      year={2026},
      eprint={2607.05147},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.05147}, 
}
```

