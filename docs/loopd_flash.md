# LoopDFlash / LoopDSpark 第一阶段实验

这一版提供 **Qwen3 的固定循环训练与推理**，用于验证深度收益和参数共享。
5L、15L 都需要训练；本分支不包含预训练权重。动态退出 router、Middle-Cycle、
15L 独立层多出口和 vLLM/SGLang 部署是后续阶段。

## 实验组

| 配置 | 独立层数 | 训练 loop 数 | 可评估 loop 数 | 执行深度 |
|---|---:|---:|---|---|
| `dflash_5l` / `dspark_5l` | 5 | 1 | 1 | 5 |
| `dflash_15l` / `dspark_15l` | 15 | 1 | 1 | 15 |
| `dflash_loop5x3` / `dspark_loop5x3` | 5 | 3 | 1、2、3 | 5、10、15 |

Loop 模型反复使用同一组五层参数，传递原始 block hidden；各出口以 Norm + LM head
读出。target 特征、token 位置和训练 attention mask 不随 loop 改变。
循环间不采样、不重新嵌入预测 token、不跨 block 保存循环状态。

固定三次循环使用完整反向传播和三个出口的监督，默认权重 `0.2/0.3/0.5`。
DSpark 的 Markov 和 confidence head 在出口间共享。DFlash 使用原有 CE；DSpark
保留原有 CE/L1/confidence 目标。TensorBoard 提供 `train/loop1/...` 至 `loop3/...`。

Context KV 每个物理层保存一份；每轮只追加新确认的 target hidden 一次。
block KV 随 loop 更新并在使用后丢弃。共享的是 **相同 target 特征的相同投影**，
不是把不同 loop 的 block KV 强行当作相同状态。训练投影共享保留梯度。

## 环境和数据

使用仓库 `requirements.txt` 对应的 CUDA 环境（PyTorch 2.9.1、Transformers 5.10.2）。
本机 CPU 测试使用对应版本的 CPU PyTorch；尚未验证真实 GPU flex-attention / FSDP。

`TARGET_CACHE` 必须指向已有 DeepSpec v2 cache：有 `manifest.json`、index 和 shard 文件。
仅有回答 JSONL 时，先按 `scripts/data/README.md` 生成 cache。改变 draft 深度不需要
重新生成 cache；脚本从 manifest 读取 target feature layer IDs。target 模型名称/路径
必须与 manifest 中 `target_model_name_or_path` 一致，模型维度必须匹配。

## 拉取和一次性配置

```bash
git fetch origin
git switch --track origin/research/loopd-flash

# 激活你现有的 DeepSpec Python 环境后执行。
export TARGET_CACHE=/实际路径/qwen3_4b_target_cache
export TARGET_MODEL=/实际路径/Qwen3-4B
export CUDA_VISIBLE_DEVICES=0,1
export GLOBAL_BATCH=16
export LOCAL_BATCH=1
```

如果 manifest 的 target 是 `Qwen/Qwen3-4B`，应使用该名称且确保模型在本地 HF cache
可用；也可以不设置 `TARGET_MODEL`，训练/矩阵脚本会使用 manifest 的名称。
这里的路径是占位符，必须换成你的路径。

## 先做 GPU smoke，再进行 pilot

```bash
# 独立目录，避免把两步 smoke checkpoint 用作正式实验的恢复点。
RUN_ROOT="$PWD/runs/loop_smoke" STEPS=2 MAX_SAMPLES=2 MAX_NEW_TOKENS=32 \
  bash scripts/loop/run_matrix.sh

# 默认：AR → 5L → 15L → Loop5x3（评估 1/2/3 loop）。
# 每组训练 500 个 optimizer steps；GSM8K/MBPP/Alpaca 各 64 样本，greedy。
bash scripts/loop/run_matrix.sh

# 扩展到 DSpark，保持同样数据和实验设置。
ALGORITHMS=dspark bash scripts/loop/run_matrix.sh
```

默认 `RUN_ROOT=$PWD/runs/loop`。结果：

```text
runs/loop/deepspec/dflash_5l_seed42/step_latest/
runs/loop/deepspec/dflash_15l_seed42/step_latest/
runs/loop/deepspec/dflash_loop5x3_seed42/step_latest/
runs/loop/comparison.csv
```

每个评估出口有 `eval_loopN.json`。CSV 包含接受长度、条件接受率、draft/verify GPU
事件耗时、整轮 wall time、串行 decode tokens/s 和相同设置 AR 对照的速度比。
接受长度沿用仓库定义：普通轮次含 target 推进 token；EOS 终止轮次可能无 bonus。
条件接受率为 `P(A > j | A >= j)`；默认禁止 confidence 截断。

计时在 warmup 后进行，decode wall time 排除 prefill，包含主机调度和处理开销。
这是 Transformers 单请求诊断，每张 GPU 串行运行请求；**不是服务引擎的高并发吞吐**。
多 GPU 的串行 TPS 用累计 token / 累计请求活跃时间统计，不能当作聚合吞吐。

## 单独训练和评估

```bash
bash scripts/loop/train.sh dflash 5l
bash scripts/loop/train.sh dflash 15l
bash scripts/loop/train.sh dflash loop5x3

checkpoint="$PWD/runs/loop/deepspec/dflash_loop5x3_seed42/step_latest"
bash scripts/loop/eval.sh "$checkpoint" 1
bash scripts/loop/eval.sh "$checkpoint" 2
bash scripts/loop/eval.sh "$checkpoint" 3

# 可选：随机最大循环次数，所有 rank 每个 optimizer step 采用相同深度。
EXP_NAME=dflash_loop_random_seed42 bash scripts/loop/train.sh dflash loop5x3 \
  --opts model.sample_loop_count=True
```

常用环境变量：`PYTHON`、`STEPS`、`ANCHORS`、`GLOBAL_BATCH`、`LOCAL_BATCH`、`LR`、
`SHARDING`、`SAVE_EVERY`、`SEED`、`RUN_ROOT`、`TASKS`、`MAX_SAMPLES`、`MAX_NEW_TOKENS`。
训练发生 OOM 时先降低 `ANCHORS`（例如 8），或采用 `SHARDING=full_shard`；各组保持
相同数据/anchors/批量/采样模式。`data.max_length` 不会自动截断已有 cache。
同一个实验名会自动恢复已有 checkpoint；改变实验设置时使用新 `EXP_NAME`/`RUN_ROOT`。
矩阵脚本实验名固定为算法、变体、seed，因此不同 pilot 设置请用不同 `RUN_ROOT`。

## 验证与后续决策

```bash
python -m pip install pytest
python -m pytest tests/test_loop_qwen3.py -q
```

测试覆盖：单 loop 与原实现一致；上下文增量缓存等价完整重算；缓存不存 block KV；
多出口训练和 checkpoint 恢复；训练时未来 target 特征不可见；greedy 投机输出等价 AR。
CPU 的训练测试使用与生产 mask 等价的 dense SDPA mask；GPU smoke 验证真实 flex 路径。

500 步是趋势筛查，不代表收敛。确认 15L 的深度收益后，比较同一个 Loop checkpoint
在 1/2/3 loop 的接受长度和成本。再做 15L 的 5/10/15 层多出口对照，构造每个 block
的最优深度标签，训练基于接受前缀增益/耗时的 router。暂不以任务类型或低 confidence
直接指定更多 loop；更深循环不保证每个 block 都更好。
