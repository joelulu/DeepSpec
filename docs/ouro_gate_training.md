# Looped DFlash：baseline、简化 Ouro 与两阶段 Ouro

这套实现是 Ouro 训练目标的 **block 级适配**。每个 block 共用一个退出深度，默认在 loop3/4/5 中选择；门只读取当前 normalized hidden 的位置平均。raw/rmsnorm 的循环传递方式不变。不包含 LoopCD、DSpark confidence 或 token 级路由。

## 公共设置

在仓库根目录执行，先激活已有 DeepSpec 环境：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SOURCE_CACHE="/你的/target_cache"
export DATA_PERCENT=10 EPOCHS=10 SEED=42
export LOCAL_BATCH=8 GLOBAL_BATCH=256
export LOOP_TRANSFER=raw LOOP_LOSS=late
export RUN_ROOT="$PWD/runs/dflash_ouro"
export MIN_EXIT_LOOP=3
export GATE_EPOCHS=1 GATE_LR=1e-3
export GATE_GAIN_SLOPE=50 GATE_GAIN_THRESHOLD=0.005
export OURO_BETA=0.1
```

`SOURCE_CACHE` 必须是已有 DeepSpec cache（含 manifest.json），target 路径从 manifest 读取。训练与 holdout 的划分沿用 `scripts/loop/settings.sh`，阶段间保持相同数据比例和 seed。`GATE_EPOCHS/GATE_LR/OURO_BETA` 是初始试验值。

## 三种训练入口

```bash
# 无门控；保留 LOOP_LOSS=late/uniform/final 和原初始化方式。
bash train_dflash_loop.sh baseline 1 5

# A：使用已经训练好的模型；自动读取其结构与循环传递方式。
BASELINE_CKPT="/你的/已训练模型/step_latest"
bash train_dflash_loop.sh A "$BASELINE_CKPT"

# B：新建 draft+gate，联合训练完成后自动执行专项训练。
bash train_dflash_loop.sh B 1 5

# 仅执行 B 的专项训练阶段，或恢复该阶段。
JOINT_CKPT="/你的/B_joint/实验名/step_latest"
bash train_dflash_loop.sh B-gate "$JOINT_CKPT"
```

原来的 `bash train_dflash_loop.sh 1 5`、`5l/15l/loop5x3` 用法保留。

所有训练都执行完整 R 轮、使用同一批 anchors。最小退出深度必须小于最大 loop 数；对于 R=3 等模型，需要显式设置 `MIN_EXIT_LOOP=1` 或 `2`。

### baseline

没有 gate 参数、没有 gate 前向，仍按固定 `loop_loss_weights` 训练。

### A / B 的专项训练

冻结整个 draft（包括 norm、fc、embedding 与 LM head），只更新共享 Linear gate。在每个 block 的有效标签位置平均 CE，不使用位置衰减；dummy blocks 不参与 loss。用当前轮到下一轮的改善监督当前轮：

```python
gain = (ce_r - ce_next).detach().clamp_min(0)
stop_target = 1 - sigmoid(k * (gain - tau))
loss = BCEWithLogits(gate_logit_r, stop_target)
```

只训练允许退出的非最后轮。默认监督 loop3 和 loop4，loop5 强制结束。CE 与 backbone 无梯度，gate 有梯度。默认 k=50/tau=0.005 下，零改善的停止软标签约为 0.562，而不是 1；必须结合验证集检查 q 的实际效果。

A 加载模型权重并初始化新门。B 专项训练保留联合阶段已经学到的门。两者都新建 optimizer 和调度，不继承输入 checkpoint 的 optimizer；相同输出任务的重启才恢复自身状态。

### B 的联合训练

draft 初始化与原训练一致，embedding/LM head 从 target 复制并冻结；联合更新原本可训练的 draft 参数和 gate。退出分布只定义在允许的深度上：

```python
p3 = g3
p4 = (1 - g3) * g4
p5 = (1 - g3) * (1 - g4)
loss = mean(p3 * ce3 + p4 * ce4 + p5 * ce5 - beta * entropy(p))
```

最后一轮吸收全部剩余概率。训练使用概率加权 **各轮 CE**，不是混合 logits 后再计算 CE。概率、hidden 和 CE 均保留梯度；`LOOP_LOSS` 固定权重不参与此目标。默认 CE-only、完整深度、关闭 torch.compile，不增加计算成本惩罚。

阶段一成功保存后，才会启动阶段二。输出分别位于 `RUN_ROOT/baseline`、`RUN_ROOT/A_gate`、`RUN_ROOT/B_joint`、`RUN_ROOT/B_gate` 下，日志打印准确的 `step_latest` 路径。模型结构、来源、数据与调度写入 run identity；不同设置的任务不会自动混用状态。

新入口默认 `SAVE_TRAINING_STATE=1`，仍只保存最终模型，同时保留 optimizer 以支持恢复。设为 `0` 可只保留最终模型及完成标记。阶段中断恢复需要自己开启非 final-only 保存；只保存最终模型时，阶段中途没有恢复点。输入来源目录只读。

## 两种推理及统一评测

```bash
export EVAL_GPU=0
export TASKS=gsm8k MAX_SAMPLES=16 MAX_NEW_TOKENS=256 EVAL_REPEATS=3
export EXIT_Q_LIST="0.5 0.8 0.95"
CKPT="/你的/模型/step_latest"

bash eval_dflash_loop.sh fixed "$CKPT"   # loop1 到 loopR，强制关闭门
bash eval_dflash_loop.sh gate "$CKPT"    # 对各 q 正常运行动态门
bash eval_dflash_loop.sh both "$CKPT"    # 两种评测顺序执行
```

`EVAL_GPU` 是一次评测使用的单张 GPU，脚本默认 0。`TARGET_MODEL` 可显式指定，否则从 SOURCE_CACHE manifest 读取。`RESULT_ROOT` 可指定输出目录。各模式必须使用相同输入、seed、温度和重复次数。

固定评测是多个深度分别执行完整生成；原有单参数 `fixed/prefix` 历史实验入口保留，同前缀分析仍用 prefix 工具。带门模型也可以运行 fixed。无门或没有训练标记的 checkpoint 不能启用 gate。

动态推理每个 block 重置 remaining=1/cdf=0；从训练时的 min_exit_loop 开始累积条件停止概率，达到 q 后在 backbone 内真正 break，最后一轮强制退出。退出后才调用 LM head，proposal 分布继续由现有 target 验证器处理。当前支持一个请求、一个 block；不能同时启用 LoopCD。q 不参与训练目标。

结果目录每个 repeat 保存 `eval_loop*.json`、逐样本 `*.responses.jsonl`，并生成 `comparison.csv`。记录接受长度、draft/verify/总耗时、串行吞吐、平均执行 loop 数及退出计数。固定模型的最大深度与动态模型的实际平均深度分开记录；门控开销包含在时间内。这是单请求串行评测，不是高并发 serving benchmark。

## 检查与 smoke

不启动 CUDA 时先查看实际启动计划：

```bash
DRY_RUN=1 bash train_dflash_loop.sh B 1 5
DRY_RUN=1 bash train_dflash_loop.sh A "$BASELINE_CKPT"
DRY_RUN=1 bash eval_dflash_loop.sh both "$CKPT"
```

GPU 上的小规模端到端检查：

```bash
bash smoke_dflash_ouro.sh
```

CPU 检查：`python -m pytest tests -q`。新增覆盖真实 layer 调用计数、固定深度等价、冻结参数不变、联合训练梯度、CE mask、A/B2 权重加载、保存重载、跨 block cache 与贪心 AR 一致性、阶段串联和评测规划。CPU 验证不能替代真实 Qwen GPU 训练与吞吐实验。

参考：Ouro 论文 https://arxiv.org/abs/2510.25741，第 3.2–3.4 节。这里采用 block 级门和受限退出范围，并保留 DFlash 的条件输入、初始化与 raw 传递，所以是目标适配而非完整复现。
