# Shared flex experiments. Edit this file; train CLI only takes LAYERS LOOPS.
export SOURCE_CACHE=${SOURCE_CACHE:-/home/jovyan/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main/cache/qwen3_4b_target_cache_130w_20260713_041400}
export DATA_PERCENT=${DATA_PERCENT:-3}
# Percentage applies to the pool after reserving these fixed validation samples.
export HOLDOUT_SAMPLES=${HOLDOUT_SAMPLES:-32}
export EPOCHS=${EPOCHS:-3}
export LOCAL_BATCH=${LOCAL_BATCH:-4}
export GLOBAL_BATCH=${GLOBAL_BATCH:-256}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29510}
export RANK=0 WORLD_SIZE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export SEED=${SEED:-42}
export ANCHORS=${ANCHORS:-32}
export LR=${LR:-6e-4}
export SHARDING=${SHARDING:-no_shard}
export RUN_ROOT=${RUN_ROOT:-$PWD/runs/dflash_flex}
# raw: pass unnormalized hidden; rmsnorm: pass the normalized exit hidden.
export LOOP_TRANSFER=${LOOP_TRANSFER:-raw}
export LOOP_LOSS=${LOOP_LOSS:-final}  # final | uniform | late
export LOG_EVERY=${LOG_EVERY:-5}
export DIAGNOSTIC_EVERY=${DIAGNOSTIC_EVERY:-20}
export TASKS=${TASKS:-gsm8k}
export MAX_SAMPLES=${MAX_SAMPLES:-16}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
export WARMUP_SAMPLES=${WARMUP_SAMPLES:-1}
export EVAL_REPEATS=${EVAL_REPEATS:-1}
export PROBE_SAMPLES=${PROBE_SAMPLES:-16}
export PROBE_ROUNDS=${PROBE_ROUNDS:-40}
export PROBE_REPEATS=${PROBE_REPEATS:-3}
