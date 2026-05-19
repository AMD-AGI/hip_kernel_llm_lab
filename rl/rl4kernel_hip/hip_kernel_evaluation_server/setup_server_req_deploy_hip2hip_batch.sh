#!/bin/bash
# Start the batch-parallel HIP kernel evaluation server.
#
# Override any setting below by exporting it before running this script, e.g.
#   HIP_VISIBLE_DEVICES=0,1 HIP_DEFAULT_BATCH_SIZE=64 ./setup_server_req_deploy_hip2hip_batch.sh

pkill -9 -f gunicorn || true
ulimit -c 0

# Export NAME with DEFAULT only when NAME is unset or empty.
default_env() {
    local name="$1"
    local default="$2"

    export "$name=${!name:-$default}"
}

default_env HCC_AMDGPU_TARGET "gfx942"
default_env AMDGPU_TARGETS "$HCC_AMDGPU_TARGET"
default_env HIP_EVAL_ARCH "$HCC_AMDGPU_TARGET"
default_env HIP_VISIBLE_DEVICES "0,1,2,3,4,5,6,7"

default_env HIP_ERROR_LOG_DIR "./runtime/error_log/8b-16k-hip2hip-grpo-vanilla-react-single-turn-Exp02"
default_env HIP_REFERENCE_CACHE_DIR "./runtime/reference_cache"
default_env HIP_REFERENCE_CACHE_MODE "golden+compile"
default_env HIP_REF_PERF_CACHE_TTL_S "3600"

default_env HIP_DEFAULT_BATCH_SIZE "64"
default_env HIP_PERF_ITERATIONS "1000"
default_env HIP_COMPILE_TIMEOUT_S "600"
default_env HIP_RUN_TIMEOUT_S "600"

default_env HIP_CONFIRM_SPEEDUP_ENABLED "0"
default_env HIP_CONFIRM_SPEEDUP_THRESHOLD "1.05"
default_env HIP_CONFIRM_SPEEDUP_BAND "0.02"
default_env HIP_CONFIRM_PERF_ITERATIONS "3000"

default_env HIP_COMPILE_CPU_SLOTS "16"
default_env HIP_COMPILE_INNER_JOBS "4"
default_env HIP_ENABLE_TWO_STAGE_BATCH "1"
default_env HIP_TOOL_GPU_IDS "$HIP_VISIBLE_DEVICES"
default_env HIP_TOOL_MAX_CPU_JOBS "$HIP_COMPILE_CPU_SLOTS"
default_env HIP_TOOL_QUICK_PERF_ITERATIONS "5"
default_env HIP_TOOL_PROFILE_PERF_ITERATIONS "20"
default_env HIP_TOOL_SESSION_TTL_S "7200"

default_env OMP_NUM_THREADS "1"
default_env MKL_NUM_THREADS "1"
default_env OPENBLAS_NUM_THREADS "1"

export HYDRA_FULL_ERROR=1
export MAX_JOBS="$HIP_COMPILE_INNER_JOBS"

case "$HIP_REFERENCE_CACHE_MODE" in
    golden-only)
        export HIP_ENABLE_REF_COMPILE_CACHE=0
        export HIP_ENABLE_REF_GOLDEN_CACHE=1
        export HIP_ENABLE_REF_PERF_CACHE=0
        ;;
    golden+compile)
        export HIP_ENABLE_REF_COMPILE_CACHE=1
        export HIP_ENABLE_REF_GOLDEN_CACHE=1
        export HIP_ENABLE_REF_PERF_CACHE=0
        ;;
    golden+compile+perf)
        export HIP_ENABLE_REF_COMPILE_CACHE=1
        export HIP_ENABLE_REF_GOLDEN_CACHE=1
        export HIP_ENABLE_REF_PERF_CACHE=1
        ;;
    *)
        echo "Unsupported HIP_REFERENCE_CACHE_MODE: $HIP_REFERENCE_CACHE_MODE"
        echo "Supported values: golden-only, golden+compile, golden+compile+perf"
        exit 1
        ;;
esac

mkdir -p "$HIP_ERROR_LOG_DIR"
mkdir -p "$HIP_REFERENCE_CACHE_DIR"

cat <<EOF
========================================
Starting Batch Parallel HIP Server
========================================
Configuration:
  GPUs: $HIP_VISIBLE_DEVICES
  APIs: /run_code_batch, /run_code, /tool/*
  Error Log Dir: $HIP_ERROR_LOG_DIR
  Reference Cache Dir: $HIP_REFERENCE_CACHE_DIR
  Reference Cache Mode: $HIP_REFERENCE_CACHE_MODE
  Recommended Batch Size: $HIP_DEFAULT_BATCH_SIZE
  Perf Iterations: $HIP_PERF_ITERATIONS
  Confirm Speedup: enabled=$HIP_CONFIRM_SPEEDUP_ENABLED threshold=$HIP_CONFIRM_SPEEDUP_THRESHOLD band=$HIP_CONFIRM_SPEEDUP_BAND iterations=$HIP_CONFIRM_PERF_ITERATIONS
  Two-Stage Batch Enabled: $HIP_ENABLE_TWO_STAGE_BATCH
  Compile Jobs: cpu_slots=$HIP_COMPILE_CPU_SLOTS inner_jobs=$HIP_COMPILE_INNER_JOBS MAX_JOBS=$MAX_JOBS
  BLAS/OMP Threads: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS
  Timeouts: compile=${HIP_COMPILE_TIMEOUT_S}s run=${HIP_RUN_TIMEOUT_S}s
  Effective Arch: $HIP_EVAL_ARCH
  Reference Cache Flags: compile=$HIP_ENABLE_REF_COMPILE_CACHE golden=$HIP_ENABLE_REF_GOLDEN_CACHE perf=$HIP_ENABLE_REF_PERF_CACHE ttl=${HIP_REF_PERF_CACHE_TTL_S}s
  Tool Eval: gpus=$HIP_TOOL_GPU_IDS cpu_slots=$HIP_TOOL_MAX_CPU_JOBS quick_iters=$HIP_TOOL_QUICK_PERF_ITERATIONS profile_iters=$HIP_TOOL_PROFILE_PERF_ITERATIONS
========================================
EOF

gunicorn server_req_deploy_hip2hip_batch:app \
    -k uvicorn.workers.UvicornWorker \
    --workers 1 \
    --bind 0.0.0.0:8080 \
    --timeout 1000 \
    --log-level info
