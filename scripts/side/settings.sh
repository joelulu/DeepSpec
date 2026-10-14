# DFlash / DSpark side-loop experiments: eight GPUs, 3% data, three epochs.
export SOURCE_CACHE=${SOURCE_CACHE:-/home/jovyan/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main/cache/qwen3_4b_target_cache_130w_20260713_041400}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DATA_PERCENT=${DATA_PERCENT:-3}
export EPOCHS=${EPOCHS:-3}
export LOCAL_BATCH=${LOCAL_BATCH:-2}
export GLOBAL_BATCH=${GLOBAL_BATCH:-64}
export BLOCK_SIZE=${BLOCK_SIZE:-16}
export SIDE_LOOPS=${SIDE_LOOPS:-5}
export GROUP_SIZE=${GROUP_SIZE:-3}
export FUSION_RANK=${FUSION_RANK:-128}
export SIDE_AUX_ALPHA=${SIDE_AUX_ALPHA:-0.2}
export HOLDOUT_SAMPLES=${HOLDOUT_SAMPLES:-32}
export ANCHORS=${ANCHORS:-32}
export SEED=${SEED:-42}
export LR=${LR:-6e-4}
export SHARDING=${SHARDING:-no_shard}
export LOG_EVERY=${LOG_EVERY:-5}
export DIAGNOSTIC_EVERY=0
export RUN_ROOT=${RUN_ROOT:-$PWD/runs/side_p3_e3}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29510}
export RANK=0 WORLD_SIZE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export EVAL_GPU=${EVAL_GPU:-0}
export TASKS=${TASKS:-gsm8k}
export MAX_SAMPLES=${MAX_SAMPLES:-128}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
export WARMUP_SAMPLES=${WARMUP_SAMPLES:-3}
export EVAL_REPEATS=${EVAL_REPEATS:-3}
# 0 tests a fixed 15-token proposal. Increase to test confidence truncation.
export CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0}
