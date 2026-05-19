#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-/wekafs/zepingl/rl4kernel_hip/checkpoints/converted_hf_models/8b-16k-grpo-kde-react-single-turn-Exp02/global_step_300}
# MODEL=${MODEL:-/wekafs/zepingl/rl4kernel_hip/models/rl_trained/14b-react-w-domain-knowledge-Exp01/global_step_220}
DATASET_ROOT=${DATASET_ROOT:-/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_neurlps/hip_eval_dataset_kernelbench_25_tasks}
TURN_MODE=${TURN_MODE:-multi}
TURNS=${TURNS:-4}
LEVEL=${LEVEL:-level-3}
SAMPLE_COUNT=${SAMPLE_COUNT:-8}
ROLLOUT_N=${ROLLOUT_N:-1}
PERF_ITERATIONS=${PERF_ITERATIONS:-10}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
OPTIMIZATION_PARADIGM=${OPTIMIZATION_PARADIGM:-kernel2kernel_splice}
DATASET_TAG=${DATASET_TAG:-$(basename "$DATASET_ROOT")}
OUTPUT_ROOT=${OUTPUT_ROOT:-/wekafs/zepingl/rl4kernel_hip/outputs/HIP_benchmark_kit/multiturn_profile/kernel2kernel_global_step_300-try02_${DATASET_TAG}_${LEVEL}_${SAMPLE_COUNT}_${TURN_MODE}_turns${TURNS}}

# Keep compatibility with older callers that export DATA_ROOT directly.
# If DATA_ROOT is not provided, auto-adapt flat datasets (hip_code at root)
# into the level-structured layout expected by stage-subset.
DATA_ROOT=${DATA_ROOT:-}
if [[ -z "$DATA_ROOT" ]]; then
  if [[ -d "$DATASET_ROOT/hip_code" && ! -d "$DATASET_ROOT/$LEVEL/hip_code" ]]; then
    DATA_ROOT="$OUTPUT_ROOT/dataset_view"
    mkdir -p "$DATA_ROOT"
    ln -sfn "$DATASET_ROOT" "$DATA_ROOT/$LEVEL"
    echo "[dataset-adapter] Using flat dataset via view: $DATASET_ROOT -> $DATA_ROOT/$LEVEL"
  else
    DATA_ROOT="$DATASET_ROOT"
  fi
fi

# By default, reruns reuse complete turn artifacts under OUTPUT_ROOT and resume
# from the first incomplete stage. Pass --overwrite_outputs to clear OUTPUT_ROOT
# and rerun from turn_01.
python -m HIP_benchmark_kit.orchestration multiturn-profile-run \
  --model "$MODEL" \
  --kernelbench_hip_root "$DATA_ROOT" \
  --optimization_paradigm "$OPTIMIZATION_PARADIGM" \
  --turn_mode "$TURN_MODE" \
  --turns "$TURNS" \
  --level "$LEVEL" \
  --sample_count "$SAMPLE_COUNT" \
  --rollout_n "$ROLLOUT_N" \
  --perf_iterations "$PERF_ITERATIONS" \
  --gpu_ids "$GPU_IDS" \
  --output_root "$OUTPUT_ROOT" \
  "$@"
