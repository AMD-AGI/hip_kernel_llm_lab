#!/bin/bash
set -euo pipefail
set -x

# Disable host/GPU core dump artifacts (core.*, gpucore.*) in cwd.
ulimit -c 0 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

#############################################
# Multi-turn Specific Parameters
#############################################
# Standalone multi-turn launcher. Keep the full effective config visible here
# instead of wrapping react_single_turn.sh, so async rollout/tool-use settings
# are explicit and do not drift behind a thin shell wrapper.
TOOL_CONFIG_PATH="${HIP_KERNEL_TOOL_CONFIG_PATH:-${SCRIPT_DIR}/tool_config/hip_kernel_eval_tool_config.yaml}"
TOOLS_KWARGS_BUILDER_PATH="${HIP_KERNEL_TOOLS_KWARGS_BUILDER_PATH:-${PROJECT_ROOT}/dataset/multi_turn_tools.py}"
TOOLS_KWARGS_BUILDER_NAME="${HIP_KERNEL_TOOLS_KWARGS_BUILDER_NAME:-build_kernel_eval_tools_kwargs}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-8}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-4}"
MAX_TOOL_WALLCLOCK_S="${MAX_TOOL_WALLCLOCK_S:-600}"
TOOL_SERVER_URL="${HIP_KERNEL_TOOL_SERVER_URL:-}"

#############################################
# Environment Setup
#############################################
# On ROCm, Ray/vLLM interop is more reliable when training launches only expose
# CUDA_VISIBLE_DEVICES. Setting HIP_VISIBLE_DEVICES at the same time can trip
# Ray's AMD GPU env detection and break async vLLM startup.
DEFAULT_TRAIN_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${DEFAULT_TRAIN_VISIBLE_DEVICES}"
unset HIP_VISIBLE_DEVICES || true
unset ROCR_VISIBLE_DEVICES || true
unset RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES || true
unset RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES || true
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES || true
set +x
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before launching training.}"
set -x
export HYDRA_FULL_ERROR=1
if [[ -n "${N_GPUS_PER_NODE:-}" ]]; then
  export N_GPUS_PER_NODE
else
  IFS=',' read -r -a __cuda_visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
  export N_GPUS_PER_NODE="${#__cuda_visible_devices[@]}"
  unset __cuda_visible_devices
fi
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
# Kernel Novelty Tracking
export KERNEL_NOVELTY_LOG_DIR="./kernel_novelty_logs/${EXP_NAME:-default}"
export KERNEL_NOVELTY_LOG_ALL="1"  # "1"=record all, "0"=only copy/exception cases

ulimit -n 1048576

# Start auto cleanup daemon
CLEANUP_SCRIPT="$(dirname "$0")/auto_cleanup_stuck_processes.sh"
if [[ -f "$CLEANUP_SCRIPT" ]]; then
    if ! pgrep -f "auto_cleanup_stuck_processes.sh.*daemon" > /dev/null; then
        echo "[INFO] Starting auto cleanup daemon..."
        nohup "$CLEANUP_SCRIPT" --daemon > /tmp/hip_cleanup_daemon.log 2>&1 &
        CLEANUP_PID=$!
        echo "[INFO] Auto cleanup daemon started with PID: $CLEANUP_PID"
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
TRAIN_FILE="./dataset/kernel-agent-single-sft-1125/rl_data_v01_mi325x_react_verl.parquet"
VAL_FILE="./dataset/kernel-agent-single-sft-1125/rl_data_v01_mi325x_react_verl.parquet"
MODEL_PATH="./models/Qwen3-32B"

RES_LENGTH=24576
PROMPT_LENGTH=4096
ENABLE_THINKING=True

# Batch Sizes
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=8
PPO_MINI_BATCH_SIZE=8
PPO_MICRO_BATCH_SIZE=8

# Rollout & Sampling
ROLLOUT_N=16
TEMPERATURE=1.0
VAL_N=4
VAL_DO_SAMPLE=True
TOP_P=0.95

# Entropy & KL Regularization
ENTROPY_COEFF=0.015
USE_KL_LOSS=False
KL_LOSS_COEF=0.0

# Training Control
TOTAL_EPOCHS=10
SAVE_FREQ=10
TEST_FREQ=1000

# Resume Load Tuning
LOAD_USE_MMAP=True
LOAD_OPTIMIZER_LOCAL_CONCURRENCY=1
LOAD_LOG_CPU_RSS=True

# Parallelism
TENSOR_MODEL_PARALLEL_SIZE=2
ULYSSES_SEQUENCE_PARALLEL_SIZE=1
USE_TORCH_COMPILE=True

# vLLM / Rollout Init Tuning
ROLLOUT_LOAD_FORMAT="dummy_dtensor"
ACTOR_OPTIMIZER_OFFLOAD=True

# vLLM & Memory
GPU_MEMORY_UTILIZATION=0.85
ENABLE_CHUNKED_PREFILL=True
MAX_NUM_BATCHED_TOKENS=$((PROMPT_LENGTH + RES_LENGTH))

# Ray / ROCm Memory Monitor
RAY_MEMORY_USAGE_THRESHOLD="${RAY_MEMORY_USAGE_THRESHOLD:-0.99}"
RAY_MEMORY_MONITOR_REFRESH_MS="${RAY_MEMORY_MONITOR_REFRESH_MS:-}"

# PPO Clipping
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.4
ADVANTAGE_SHAPING=add
PG_LOSS_AGG_MODE="seq-mean-token-sum-norm"

# Dynamic Sampling (Filter Groups)
ENABLE_FILTER_GROUPS=True
FILTER_GROUPS_METRIC=acc
MAX_NUM_GEN_BATCHES=10

# Reward Function
REWARD_PY="./reward/reward_batch.py"
REWARD_NAME="compute_score"
REWARD_NAME_BATCH="compute_score_batch"
REWARD_MANAGER="batch_parallel"
REWARD_MODE="correct_speedup_copy_penalty"
REWARD_CORRECT_SPEEDUP_R_OK=0.3
REWARD_CORRECT_SPEEDUP_CAP=10
REWARD_CORRECT_SPEEDUP_COPY_REWARD=0.0

# Server URL configuration
export SF_URL="http://172.17.54.86:8080/run_code"
if [[ "${SF_URL}" != *"/run_code" ]]; then
  echo "FATAL: --sf-url must end with /run_code ; got: ${SF_URL}" >&2; exit 1
fi

SF_MAX_CONCURRENT=64
SF_MEMORY_LIMIT_MB=2048

# Experiment Tracking
PROJECT_NAME="hip_agent"
EXPERIMENT_NAME="32b-24k-grpo-react-multi-turn-correct_speedup_copy_penalty-Exp02"
ARCHIVE_BASE_DIR="${PROJECT_ROOT}/hip_kernel_evaluation_server/runtime/kernel_archives"
export REWARD_EVAL_ARCHIVE_DIR="${REWARD_EVAL_ARCHIVE_DIR:-${ARCHIVE_BASE_DIR}/${EXPERIMENT_NAME}}"
export REWARD_EVAL_EXPERIMENT_NAME="${REWARD_EVAL_EXPERIMENT_NAME:-${EXPERIMENT_NAME}}"
export REWARD_EVAL_RUN_ID="${REWARD_EVAL_RUN_ID:-${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)_$$}"
export REWARD_EVAL_ARCHIVE_INCLUDE_RAW_RESPONSE="${REWARD_EVAL_ARCHIVE_INCLUDE_RAW_RESPONSE:-0}"

#############################################
# Parse Command Line Arguments
#############################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL_PATH="$2"; shift 2 ;;
    --sf-url)      SF_URL="$2"; shift 2 ;;
    --tool-server-url) TOOL_SERVER_URL="$2"; shift 2 ;;
    --tool-config) TOOL_CONFIG_PATH="$2"; shift 2 ;;
    --tools-kwargs-builder-path) TOOLS_KWARGS_BUILDER_PATH="$2"; shift 2 ;;
    --tools-kwargs-builder-name) TOOLS_KWARGS_BUILDER_NAME="$2"; shift 2 ;;
    --max-assistant-turns) MAX_ASSISTANT_TURNS="$2"; shift 2 ;;
    --max-tool-calls) MAX_TOOL_CALLS="$2"; shift 2 ;;
    --max-tool-wallclock-s) MAX_TOOL_WALLCLOCK_S="$2"; shift 2 ;;
    --train)       TRAIN_FILE="$2"; shift 2 ;;
    --val)         VAL_FILE="$2"; shift 2 ;;
    --reward-py)   REWARD_PY="$2"; shift 2 ;;
    --reward-name) REWARD_NAME="$2"; shift 2 ;;
    --reward-manager) REWARD_MANAGER="$2"; shift 2 ;;
    --reward-mode) REWARD_MODE="$2"; shift 2 ;;
    --reward-correct-speedup-r-ok) REWARD_CORRECT_SPEEDUP_R_OK="$2"; shift 2 ;;
    --reward-correct-speedup-cap) REWARD_CORRECT_SPEEDUP_CAP="$2"; shift 2 ;;
    --reward-correct-speedup-copy-reward) REWARD_CORRECT_SPEEDUP_COPY_REWARD="$2"; shift 2 ;;
    --tp)          TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
    --gpu-mem-util) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --ray-memory-threshold) RAY_MEMORY_USAGE_THRESHOLD="$2"; shift 2 ;;
    --ray-memory-monitor-refresh-ms) RAY_MEMORY_MONITOR_REFRESH_MS="$2"; shift 2 ;;
    --use-torch-compile) USE_TORCH_COMPILE="$2"; shift 2 ;;
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
if [[ ! -f "$TOOL_CONFIG_PATH" ]]; then
  echo "FATAL: tool config not found: $TOOL_CONFIG_PATH" >&2; exit 1
fi
if [[ ! -f "$TOOLS_KWARGS_BUILDER_PATH" ]]; then
  echo "FATAL: tools kwargs builder not found: $TOOLS_KWARGS_BUILDER_PATH" >&2; exit 1
fi
if [[ "${SF_URL}" != *"/run_code" ]]; then
  echo "FATAL: --sf-url must end with /run_code ; got: ${SF_URL}" >&2; exit 1
fi

if [[ -z "${TOOL_SERVER_URL}" ]]; then
  TOOL_SERVER_URL="${SF_URL%/run_code}"
fi
TOOL_SERVER_URL="${TOOL_SERVER_URL%/}"
if [[ "${TOOL_SERVER_URL}" == *"/run_code_batch" ]]; then
  TOOL_SERVER_URL="${TOOL_SERVER_URL%/run_code_batch}"
fi
if [[ "${TOOL_SERVER_URL}" == *"/run_code" ]]; then
  TOOL_SERVER_URL="${TOOL_SERVER_URL%/run_code}"
fi
if [[ -z "${TOOL_SERVER_URL}" ]]; then
  echo "FATAL: resolved tool server url is empty" >&2; exit 1
fi
export HIP_KERNEL_TOOL_SERVER_URL="${TOOL_SERVER_URL}"

export RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}"
if [[ -n "${RAY_MEMORY_MONITOR_REFRESH_MS}" ]]; then
  export RAY_memory_monitor_refresh_ms="${RAY_MEMORY_MONITOR_REFRESH_MS}"
fi

echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] TRAIN=${TRAIN_FILE}"
echo "[INFO] VAL=${VAL_FILE}"
echo "[INFO] SF_URL=${SF_URL}"
echo "[INFO] HIP_KERNEL_TOOL_SERVER_URL=${HIP_KERNEL_TOOL_SERVER_URL}"
echo "[INFO] TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH}"
echo "[INFO] TOOLS_KWARGS_BUILDER=${TOOLS_KWARGS_BUILDER_PATH}:${TOOLS_KWARGS_BUILDER_NAME}"
echo "[INFO] MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS}"
echo "[INFO] MAX_TOOL_CALLS=${MAX_TOOL_CALLS}"
echo "[INFO] MAX_TOOL_WALLCLOCK_S=${MAX_TOOL_WALLCLOCK_S}"
echo "[INFO] REWARD_FN=${REWARD_PY}:${REWARD_NAME}"
echo "[INFO] TOTAL_EPOCHS=${TOTAL_EPOCHS}"
echo "[INFO] REWARD_MANAGER=${REWARD_MANAGER}"
echo "[INFO] TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE}"
echo "[INFO] GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "[INFO] RAY_memory_usage_threshold=${RAY_memory_usage_threshold}"
if [[ -n "${RAY_MEMORY_MONITOR_REFRESH_MS}" ]]; then
  echo "[INFO] RAY_memory_monitor_refresh_ms=${RAY_memory_monitor_refresh_ms}"
fi
echo "[INFO] USE_TORCH_COMPILE=${USE_TORCH_COMPILE}"

export REWARD_MODE="${REWARD_MODE}"
export REWARD_CORRECT_SPEEDUP_R_OK="${REWARD_CORRECT_SPEEDUP_R_OK}"
export REWARD_CORRECT_SPEEDUP_CAP="${REWARD_CORRECT_SPEEDUP_CAP}"
export REWARD_CORRECT_SPEEDUP_COPY_REWARD="${REWARD_CORRECT_SPEEDUP_COPY_REWARD}"
echo "[INFO] REWARD_MODE=${REWARD_MODE}"
echo "[INFO] REWARD_CORRECT_SPEEDUP_R_OK=${REWARD_CORRECT_SPEEDUP_R_OK}"
echo "[INFO] REWARD_CORRECT_SPEEDUP_CAP=${REWARD_CORRECT_SPEEDUP_CAP}"
echo "[INFO] REWARD_CORRECT_SPEEDUP_COPY_REWARD=${REWARD_CORRECT_SPEEDUP_COPY_REWARD}"
echo "[INFO] REWARD_EVAL_ARCHIVE_DIR=${REWARD_EVAL_ARCHIVE_DIR}"
echo "[INFO] REWARD_EVAL_EXPERIMENT_NAME=${REWARD_EVAL_EXPERIMENT_NAME}"
echo "[INFO] REWARD_EVAL_RUN_ID=${REWARD_EVAL_RUN_ID}"
echo "[INFO] REWARD_EVAL_ARCHIVE_INCLUDE_RAW_RESPONSE=${REWARD_EVAL_ARCHIVE_INCLUDE_RAW_RESPONSE}"

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
  data.enable_thinking=$ENABLE_THINKING \
  data.truncation=left \
  data.return_raw_chat=True \
  data.need_tools_kwargs=True \
  data.tools_kwargs_builder.path="$TOOLS_KWARGS_BUILDER_PATH" \
  data.tools_kwargs_builder.name="$TOOLS_KWARGS_BUILDER_NAME" \
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
  actor_rollout_ref.actor.use_torch_compile=$USE_TORCH_COMPILE \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((RES_LENGTH + PROMPT_LENGTH)) \
  actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
  actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
  actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
  actor_rollout_ref.actor.policy_loss.loss_agg_mode=$PG_LOSS_AGG_MODE \
  +actor_rollout_ref.actor.advantage_shaping=$ADVANTAGE_SHAPING \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
  actor_rollout_ref.actor.checkpoint.load_use_mmap=$LOAD_USE_MMAP \
  actor_rollout_ref.actor.checkpoint.load_optimizer_local_concurrency=$LOAD_OPTIMIZER_LOCAL_CONCURRENCY \
  actor_rollout_ref.actor.checkpoint.load_log_cpu_rss=$LOAD_LOG_CPU_RSS \
  \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.top_p=$TOP_P \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.load_format=$ROLLOUT_LOAD_FORMAT \
  actor_rollout_ref.rollout.enable_chunked_prefill=$ENABLE_CHUNKED_PREFILL \
  actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.enable_thinking=$ENABLE_THINKING \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSISTANT_TURNS \
  actor_rollout_ref.rollout.multi_turn.max_tool_calls=$MAX_TOOL_CALLS \
  actor_rollout_ref.rollout.multi_turn.max_tool_wallclock_s=$MAX_TOOL_WALLCLOCK_S \
  actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
  actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
  actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P \
  \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE \
  actor_rollout_ref.ref.use_torch_compile=$USE_TORCH_COMPILE \
  \
  algorithm.kl_ctrl.kl_coef=0.001 \
  +algorithm.filter_groups.enable=$ENABLE_FILTER_GROUPS \
  +algorithm.filter_groups.metric=$FILTER_GROUPS_METRIC \
  +algorithm.filter_groups.max_num_gen_batches=$MAX_NUM_GEN_BATCHES \
  \
  custom_reward_function.path="$REWARD_PY" \
  custom_reward_function.name="$REWARD_NAME" \
  +custom_reward_function.reward_kwargs.reward_mode="$REWARD_MODE" \
  +custom_reward_function.reward_kwargs.reward_correct_speedup_r_ok=$REWARD_CORRECT_SPEEDUP_R_OK \
  +custom_reward_function.reward_kwargs.reward_correct_speedup_cap=$REWARD_CORRECT_SPEEDUP_CAP \
  +custom_reward_function.reward_kwargs.reward_correct_speedup_copy_reward=$REWARD_CORRECT_SPEEDUP_COPY_REWARD \
  \
  reward_model.reward_manager="$REWARD_MANAGER" \
  +reward_model.compute_score_batch.path="$REWARD_PY" \
  +reward_model.compute_score_batch.name="$REWARD_NAME_BATCH" \
  +reward_model.compute_score_batch.reward_kwargs.reward_mode="$REWARD_MODE" \
  +reward_model.compute_score_batch.reward_kwargs.reward_correct_speedup_r_ok=$REWARD_CORRECT_SPEEDUP_R_OK \
  +reward_model.compute_score_batch.reward_kwargs.reward_correct_speedup_cap=$REWARD_CORRECT_SPEEDUP_CAP \
  +reward_model.compute_score_batch.reward_kwargs.reward_correct_speedup_copy_reward=$REWARD_CORRECT_SPEEDUP_COPY_REWARD \
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
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=1 \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.default_hdfs_dir=null \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.resume_mode=auto \
  "${@:1}"
