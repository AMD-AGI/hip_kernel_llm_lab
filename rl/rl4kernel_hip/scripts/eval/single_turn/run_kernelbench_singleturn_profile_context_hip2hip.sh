#!/usr/bin/env bash
set -euo pipefail

# Disable host/GPU core dump artifacts (core.*, gpucore.*) in cwd.
ulimit -c 0 || true

MODEL=${MODEL:-/wekafs/zepingl/rl4kernel_hip/checkpoints/converted_hf_models/8b-16k-hip2hip-grpo-vanilla-react-single-turn-Exp01/global_step_110}
DATASET_ROOT=${DATASET_ROOT:-/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_evaluation_pub/kernelbench_hip/kernelbench_hip_100_l1_35_l2_35_l3_30}
TURN_MODE=single  #options: multi, single
TURNS=1
LEVEL=all
SAMPLE_COUNT=100
ROLLOUT_NS=${ROLLOUT_NS:-1,4,16}
PERF_ITERATIONS=${PERF_ITERATIONS:-1000}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
OPTIMIZATION_PARADIGM=${OPTIMIZATION_PARADIGM:-hip2hip_full_file}
DATASET_TAG=${DATASET_TAG:-$(basename "$DATASET_ROOT")}
OUTPUT_ROOT_BASE=${OUTPUT_ROOT_BASE:-/wekafs/zepingl/rl4kernel_hip/outputs/kernelbench_100/8b-16k-hip2hip-grpo-vanilla-react-single-turn-Exp01_${DATASET_TAG}_${TURN_MODE}_origin_profile_context_rollout_scaling/w_origin_profile_context}
PROFILE_GENERATED=${PROFILE_GENERATED:-never}
ORIGIN_PROFILE_CONTEXT=${ORIGIN_PROFILE_CONTEXT:-ensure_and_use}
ORIGIN_PROFILE_MISSING_POLICY=${ORIGIN_PROFILE_MISSING_POLICY:-fail}
PROFILE_METADATA_MODE=${PROFILE_METADATA_MODE:-deferred}
ORIGIN_PROFILE_ARTIFACT_ROOT=${ORIGIN_PROFILE_ARTIFACT_ROOT:-$OUTPUT_ROOT_BASE/shared/origin_profiling/artifacts}
ORIGIN_PROFILE_PROMPT_ROOT=${ORIGIN_PROFILE_PROMPT_ROOT:-$OUTPUT_ROOT_BASE/shared/origin_profiling/prompt_maps}
ORIGIN_BASELINE_EVAL_ROOT=${ORIGIN_BASELINE_EVAL_ROOT:-$OUTPUT_ROOT_BASE/shared/origin_baseline/eval}
SHARED_COMPILE_CACHE_ROOT=${SHARED_COMPILE_CACHE_ROOT:-$OUTPUT_ROOT_BASE/shared/reference_cache}

echo "=== KernelBench HIP2HIP single-turn with origin profile context ==="
echo "ORIGIN_PROFILE_CONTEXT=$ORIGIN_PROFILE_CONTEXT"
echo "ORIGIN_PROFILE_MISSING_POLICY=$ORIGIN_PROFILE_MISSING_POLICY"
echo "PROFILE_METADATA_MODE=$PROFILE_METADATA_MODE"
echo "ORIGIN_PROFILE_ARTIFACT_ROOT=$ORIGIN_PROFILE_ARTIFACT_ROOT"
echo "ORIGIN_PROFILE_PROMPT_ROOT=$ORIGIN_PROFILE_PROMPT_ROOT"
echo "OUTPUT_ROOT_BASE=$OUTPUT_ROOT_BASE"

# Keep compatibility with older callers that export DATA_ROOT directly.
# If DATA_ROOT is not provided, auto-adapt flat datasets (hip_code at root)
# into the level-structured layout expected by stage-subset.
DATA_ROOT=${DATA_ROOT:-}
if [[ -z "$DATA_ROOT" ]]; then
  if [[ -d "$DATASET_ROOT/hip_code" && ! -d "$DATASET_ROOT/$LEVEL/hip_code" ]]; then
    DATA_ROOT="$OUTPUT_ROOT_BASE/dataset_view"
    mkdir -p "$DATA_ROOT"
    ln -sfn "$DATASET_ROOT" "$DATA_ROOT/$LEVEL"
    echo "[dataset-adapter] Using flat dataset via view: $DATASET_ROOT -> $DATA_ROOT/$LEVEL"
  else
    DATA_ROOT="$DATASET_ROOT"
  fi
fi

IFS=',' read -r -a ROLLOUT_ARRAY <<< "$ROLLOUT_NS"
PREVIOUS_OUTPUT_ROOT=""
for ROLLOUT_N in "${ROLLOUT_ARRAY[@]}"; do
  ROLLOUT_N=${ROLLOUT_N//[[:space:]]/}
  [[ -z "$ROLLOUT_N" ]] && continue

  OUTPUT_ROOT="$OUTPUT_ROOT_BASE/rollout_n_${ROLLOUT_N}"
  EXTRA_ARGS=()
  if [[ -n "$PREVIOUS_OUTPUT_ROOT" && "${DISABLE_ROLLOUT_REUSE:-0}" != "1" ]]; then
    EXTRA_ARGS+=(--reuse_generation_from_root "$PREVIOUS_OUTPUT_ROOT")
  fi
  if [[ "${DISABLE_ROLLOUT_REUSE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--disable_rollout_reuse)
  fi
  EXTRA_ARGS+=(
    --origin_profile_context "$ORIGIN_PROFILE_CONTEXT"
    --profile_artifact_root "$ORIGIN_PROFILE_ARTIFACT_ROOT"
    --profile_prompt_root "$ORIGIN_PROFILE_PROMPT_ROOT"
    --profile_missing_policy "$ORIGIN_PROFILE_MISSING_POLICY"
    --profile_metadata_mode "$PROFILE_METADATA_MODE"
  )

  echo "=== KernelBench HIP2HIP single-turn profile-context rollout_n=${ROLLOUT_N} ==="
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
    --profile_generated "$PROFILE_GENERATED" \
    --origin_baseline_eval_root "$ORIGIN_BASELINE_EVAL_ROOT" \
    --shared_compile_cache_root "$SHARED_COMPILE_CACHE_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    "${EXTRA_ARGS[@]}" \
    "$@"

  PREVIOUS_OUTPUT_ROOT="$OUTPUT_ROOT"
done
