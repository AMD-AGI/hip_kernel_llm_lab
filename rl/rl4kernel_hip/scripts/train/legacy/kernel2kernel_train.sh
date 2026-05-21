#!/bin/bash
set -euo pipefail
set -x

#############################################
# Environment Setup
#############################################
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
set +x
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before launching training.}"
set -x
export HYDRA_FULL_ERROR=1
export HIP_VISIBLE_DEVICES="0,1,2,3"
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000

ulimit -n 1048576

# Start the automatic cleanup daemon.
CLEANUP_SCRIPT="$(dirname "$0")/auto_cleanup_stuck_processes.sh"
if [[ -f "$CLEANUP_SCRIPT" ]]; then
    # Avoid launching duplicate daemon processes.
    if ! pgrep -f "auto_cleanup_stuck_processes.sh.*daemon" > /dev/null; then
        echo "[INFO] Starting auto cleanup daemon..."
        nohup "$CLEANUP_SCRIPT" --daemon > /tmp/hip_cleanup_daemon.log 2>&1 &
        CLEANUP_PID=$!
        echo "[INFO] Auto cleanup daemon started with PID: $CLEANUP_PID"
        # Save the PID for later cleanup.
        echo $CLEANUP_PID > /tmp/hip_cleanup_daemon.pid
    else
        echo "[INFO] Auto cleanup daemon already running"
    fi
else
    echo "[WARN] Auto cleanup script not found at: $CLEANUP_SCRIPT"
fi

#############################################
# Key Training Parameters (Easy to Modify)
#############################################
# Data & Model
TRAIN_FILE="./dataset/kernel2kernel/train_kernel2kernel_gpumode_hard_samples.parquet"
VAL_FILE="./dataset/hip2hip/val_gpumode_rl_v2_hip2hip.parquet"
MODEL_PATH="./models/Qwen3-8B"

RES_LENGTH=4096
PROMPT_LENGTH=2048

# Batch Sizes
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=8
PPO_MINI_BATCH_SIZE=4
PPO_MICRO_BATCH_SIZE=4

# Rollout & Sampling
ROLLOUT_N=8              # Increase samples per prompt for exploration diversity.
TEMPERATURE=1.0          # Raise sampling temperature to increase output diversity.
VAL_N=4
VAL_DO_SAMPLE=True
TOP_P=0.95               # Enable nucleus sampling.

# Entropy & KL Regularization
ENTROPY_COEFF=0.0       # Entropy bonus encourages output diversity.
USE_KL_LOSS=False       # Enable KL loss.
KL_LOSS_COEF=0.0        # KL coefficient prevents the policy from drifting too quickly from the reference model.

# Training Control
TOTAL_EPOCHS=10
SAVE_FREQ=10
TEST_FREQ=1000

# Parallelism
TENSOR_MODEL_PARALLEL_SIZE=1
ULYSSES_SEQUENCE_PARALLEL_SIZE=1

# vLLM & Memory
GPU_MEMORY_UTILIZATION=0.6  # Leave more memory headroom for evaluation.
ENABLE_CHUNKED_PREFILL=True
MAX_NUM_BATCHED_TOKENS=20000

# PPO Clipping
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.4
ADVANTAGE_SHAPING=add

# Dynamic Sampling (Filter Groups)
ENABLE_FILTER_GROUPS=True
FILTER_GROUPS_METRIC=acc
MAX_NUM_GEN_BATCHES=10

# Reward Function
REWARD_PY="./reward/reward_batch.py"
REWARD_NAME="compute_score"           # Single-sample fallback function.
REWARD_NAME_BATCH="compute_score_batch"
REWARD_MANAGER="batch_parallel"       ##dapo or batch_parallel
REWARD_MODE="legacy_default"

# Server URL configuration:
# - single-sample mode: use http://localhost:8080/run_code directly
# - batch mode: reward_batch.py converts it to /run_code_batch
export SF_URL="http://localhost:8080/run_code"
if [[ "${SF_URL}" != *"/run_code" ]]; then
  echo "FATAL: --sf-url must end with /run_code ; got: ${SF_URL}" >&2; exit 1
fi


SF_MAX_CONCURRENT=64
SF_MEMORY_LIMIT_MB=2048

# Experiment Tracking
PROJECT_NAME="hip_agent"
EXPERIMENT_NAME="8b-4k-grpo-add-batch-gpumode-kernel2kernel-Exp03"

#############################################
# Parse Command Line Arguments
#############################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL_PATH="$2"; shift 2 ;;
    --sf-url)      SF_URL="$2"; shift 2 ;;
    --train)       TRAIN_FILE="$2"; shift 2 ;;
    --val)         VAL_FILE="$2"; shift 2 ;;
    --reward-py)   REWARD_PY="$2"; shift 2 ;;
    --reward-name) REWARD_NAME="$2"; shift 2 ;;
    --reward-manager) REWARD_MANAGER="$2"; shift 2 ;;
    --reward-mode) REWARD_MODE="$2"; shift 2 ;;
    --epochs)      TOTAL_EPOCHS="$2"; shift 2 ;;
    *) break ;;
  esac
done

#############################################
# Sanity Checks
#############################################
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "FATAL: train_file not found: $TRAIN_FILE" >&2; exit 1
fi
if [[ ! -f "$VAL_FILE" ]]; then
  echo "FATAL: val_file not found: $VAL_FILE" >&2; exit 1
fi
if [[ ! -f "$REWARD_PY" ]]; then
  echo "FATAL: reward file not found: $REWARD_PY" >&2; exit 1
fi
if [[ "${SF_URL}" != *"/run_code" ]]; then
  echo "FATAL: --sf-url must end with /run_code ; got: ${SF_URL}" >&2; exit 1
fi

echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] TRAIN=${TRAIN_FILE}"
echo "[INFO] VAL=${VAL_FILE}"
echo "[INFO] SF_URL=${SF_URL}"
echo "[INFO] REWARD_FN=${REWARD_PY}:${REWARD_NAME}"
echo "[INFO] TOTAL_EPOCHS=${TOTAL_EPOCHS}"
echo "[INFO] REWARD_MANAGER=${REWARD_MANAGER}"
export REWARD_MODE="${REWARD_MODE}"
echo "[INFO] REWARD_MODE=${REWARD_MODE}"

#############################################
# Launch Training
#############################################
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=$VAL_BATCH_SIZE \
  data.max_prompt_length=$PROMPT_LENGTH \
  data.max_response_length=$RES_LENGTH \
  data.truncation=left \
  \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((RES_LENGTH + PROMPT_LENGTH)) \
  actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
  actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
  actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
  +actor_rollout_ref.actor.advantage_shaping=$ADVANTAGE_SHAPING \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.top_p=$TOP_P \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.load_format=auto \
  actor_rollout_ref.rollout.enable_chunked_prefill=$ENABLE_CHUNKED_PREFILL \
  actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
  actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
  actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
  actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P \
  \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE \
  \
  algorithm.kl_ctrl.kl_coef=0.001 \
  +algorithm.filter_groups.enable=$ENABLE_FILTER_GROUPS \
  +algorithm.filter_groups.metric=$FILTER_GROUPS_METRIC \
  +algorithm.filter_groups.max_num_gen_batches=$MAX_NUM_GEN_BATCHES \
  \
  custom_reward_function.path="$REWARD_PY" \
  custom_reward_function.name="$REWARD_NAME" \
  \
  reward_model.reward_manager="$REWARD_MANAGER" \
  +reward_model.compute_score_batch.path="$REWARD_PY" \
  +reward_model.compute_score_batch.name="$REWARD_NAME_BATCH" \
  reward_model.sandbox_fusion.url="$SF_URL" \
  reward_model.sandbox_fusion.max_concurrent=$SF_MAX_CONCURRENT \
  reward_model.sandbox_fusion.memory_limit_mb=$SF_MEMORY_LIMIT_MB \
  reward_model.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE \
  \
  critic.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE \
  \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.default_hdfs_dir=null \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.resume_mode=auto \
  "${@:1}"
