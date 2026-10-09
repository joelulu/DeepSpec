# DFlash 3% pilot 与同前缀深度诊断

基于 `research/loopd-flash` 分支。保持现有五层循环结构、原始 hidden 传递和
固定多出口 loss 权重 0.2/0.3/0.5。本次不训练 gate、不改归一化、不接服务引擎。

## 数据和设置

`scripts/loop/settings_3pct.sh` 写死所有设置，不需要额外 export。
源数据为用户的 `qwen3_4b_target_cache_130w_20260713_041400`，路径前缀
`/home/jovyan/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main/cache/`。
若该路径不同，只改这个 Bash 设置文件。GPU 为 0,1,2,3；global batch 256，local
batch 1；anchors 32；1000 个 optimizer steps；每 500 步保存；seed 42。

按 manifest 中实际样本数乘 0.03 后向下取整，不依据目录名估计。若恰有 130 万条，
选出 39,000 条。同一确定性随机子集用于全部训练组。只重建索引/manifest，并通过
软链接引用原始 tensor shards；不能删除或移动源缓存。原始文件不修改。
保存 `selected_source_ids.json` 和 `subset.json`，重复启动复用相同子集。
子集来源、种子或索引被改变时拒绝复用；改变设置应使用新的子集目录及 RUN_ROOT。
1000 步会重复遍历子集，不代表一次 epoch；这是趋势筛查，不代表已经收敛。

## 启动

在仓库根目录激活原有 CUDA 环境，不需要重新安装或更换 CUDA PyTorch。

```bash
# 快速验证三组训练、保存、五种固定出口和同前缀诊断；独立目录保存。
bash smoke_dflash_loop.sh

# 正式 pilot：以下三个命令可以分别运行。
bash train_dflash_loop.sh 5l
bash train_dflash_loop.sh 15l
bash train_dflash_loop.sh loop5x3

# 三组训练完成后：AR、5L、15L、Loop 的 1/2/3 出口。
bash eval_dflash_loop.sh fixed

# Loop 训练完成后即可运行，不依赖独立 5L/15L 权重。
bash eval_dflash_loop.sh prefix
```

固定评估使用 GPU 0、GSM8K/MBPP/Alpaca 各 64 条、最多 512 新 token、greedy。
新实验名是 `dflash_4b_{5l|15l|loop5x3}_3pct_seed42`，不恢复先前 50k 实验。
同名实验仍会自动恢复当前实验已有 checkpoint；改训练设置请同时改实验名或目录。
smoke 使用每组 2 步、batch 4、anchors 2，**不能用于性能结论**。

## 输出与判读

`runs/dflash_loop_3pct/comparison.csv`：真实固定档位串行 decode TPS、相对 AR 速度比、
接受长度和阶段计时。`draft_ms`/`verify_ms` 是整个数据集的累计 GPU 时间；整轮
`round_wall_ms` 是平均每轮时间。接受长度沿用仓库定义，普通轮次含 target bonus。

`same_prefix.jsonl`：每任务 16 个样本、每样本最多 40 轮、每个出口重复 3 次取中位数。
1/2/3-loop 候选从同一 target/draft cache 状态启动；每次 target 验证前回滚缓存；
只有最后一个 3-loop trial 的已验证前缀被保留，按真实 3-loop greedy 轨迹继续。
不同循环不填回预测 token。记录 prefix SHA256、候选、接受长度和耗时。

`same_prefix_summary.json`：每任务的固定深度效率代理、最佳深度分布、接受前缀
修正/破坏次数，以及离线 oracle 效率代理。全局 oracle 最大化总推进 token / 总时间，
不能把每 block TPS 简单取平均。EOS 或输出预算边界的整轮从 utility 标签中排除。

同前缀计时包含 draft/proposal、target 验证和 context/cache 更新；排除 prefill、
诊断缓存重置、trace 写盘和尚未实现的 gate。重复测量可能有缓存/硬件顺序偏差。
oracle 使用未来信息并受计时噪声影响，**不是动态退出实际速度或吞吐承诺**。
固定 TPS 表才是当前真实速度比较；本版仍不属于高并发服务吞吐测试。

先检查 Loop 1 是否损失浅出口、Loop 3 与独立 15L 的接受长度差距、额外循环是否
抵消成本。再看同前缀 oracle 相比最佳固定深度是否有足够收益，决定是否训练 gate。
3% 是训练数据比例；评估使用已有评估集，不从训练子集抽样。

## 检查

```bash
python -m pytest tests/test_loop_qwen3.py tests/test_loop_pilot.py -q
```

CPU 测试验证实际 subset tensor 数据一致性、重复抽样、不同源拒绝复用、三个出口
缓存隔离与 AR 一致性、oracle 与穷举最优解一致；GPU smoke 检查真实训练链路。
