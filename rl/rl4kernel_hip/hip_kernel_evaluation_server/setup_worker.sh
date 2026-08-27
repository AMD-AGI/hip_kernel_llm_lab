#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# ============================================================================
# HIP Kernel Evaluation Worker Server Startup Script
# ============================================================================
# This script starts a worker server on a compute node. The worker receives
# kernel evaluation tasks from the master and executes them on local GPUs.
#
# Usage:
#   ./setup_worker.sh                # Start with default settings
#   ./setup_worker.sh --port 8080    # Use custom port
#   ./setup_worker.sh --help         # Show help
#
# Note: Run this script on each worker node in your cluster.
# ============================================================================

set -e

#############################################
# Parse Arguments
#############################################
PORT=8080
PERF_ITERATIONS=1000

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --perf-iterations)
            PERF_ITERATIONS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT              Port to bind (default: 8080)"
            echo "  --perf-iterations N      Performance test iterations (default: 1000)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

#############################################
# Environment Setup
#############################################
export HCC_AMDGPU_TARGET="${HCC_AMDGPU_TARGET:-gfx942}"
export AMDGPU_TARGETS="${AMDGPU_TARGETS:-$HCC_AMDGPU_TARGET}"
export HIP_EVAL_ARCH="${HIP_EVAL_ARCH:-$HCC_AMDGPU_TARGET}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HYDRA_FULL_ERROR=1

# Node identifier for logging
export NODE_ID=$(hostname)

export HIP_PERF_ITERATIONS="$PERF_ITERATIONS"
export HIP_CONFIRM_SPEEDUP_ENABLED="${HIP_CONFIRM_SPEEDUP_ENABLED:-0}"
export HIP_CONFIRM_SPEEDUP_THRESHOLD="${HIP_CONFIRM_SPEEDUP_THRESHOLD:-1.05}"
export HIP_CONFIRM_SPEEDUP_BAND="${HIP_CONFIRM_SPEEDUP_BAND:-0.02}"
export HIP_CONFIRM_PERF_ITERATIONS="${HIP_CONFIRM_PERF_ITERATIONS:-3000}"
export HIP_COMPILE_TIMEOUT_S="${HIP_COMPILE_TIMEOUT_S:-600}"
export HIP_RUN_TIMEOUT_S="${HIP_RUN_TIMEOUT_S:-600}"
export HIP_HANDLER_TIMEOUT_S="${HIP_HANDLER_TIMEOUT_S:-1200}"
export HIP_ERROR_LOG_DIR="${HIP_ERROR_LOG_DIR:-./runtime/error_log/worker}"
export HIP_REFERENCE_CACHE_DIR="${HIP_REFERENCE_CACHE_DIR:-./runtime/reference_cache/worker}"
export HIP_ENABLE_REF_COMPILE_CACHE="${HIP_ENABLE_REF_COMPILE_CACHE:-0}"
export HIP_ENABLE_REF_GOLDEN_CACHE="${HIP_ENABLE_REF_GOLDEN_CACHE:-0}"
export HIP_ENABLE_REF_PERF_CACHE="${HIP_ENABLE_REF_PERF_CACHE:-0}"
export HIP_REF_PERF_CACHE_TTL_S="${HIP_REF_PERF_CACHE_TTL_S:-3600}"
mkdir -p "$HIP_ERROR_LOG_DIR"
mkdir -p "$HIP_REFERENCE_CACHE_DIR"

#############################################
# Server Startup
#############################################
echo "========================================"
echo "Starting HIP Kernel Worker Server"
echo "========================================"
echo "Configuration:"
echo "  Node ID: $NODE_ID"
echo "  Port: $PORT"
echo "  GPUs: $HIP_VISIBLE_DEVICES"
echo "  Effective Arch: $HIP_EVAL_ARCH"
echo "  Perf Iterations: $PERF_ITERATIONS"
echo "  Confirm Speedup Enabled: $HIP_CONFIRM_SPEEDUP_ENABLED"
echo "  Confirm Threshold: $HIP_CONFIRM_SPEEDUP_THRESHOLD"
echo "  Confirm Band: $HIP_CONFIRM_SPEEDUP_BAND"
echo "  Confirm Perf Iterations: $HIP_CONFIRM_PERF_ITERATIONS"
echo "  Compile Timeout: $HIP_COMPILE_TIMEOUT_S"
echo "  Run Timeout: $HIP_RUN_TIMEOUT_S"
echo "  Error Log Dir: $HIP_ERROR_LOG_DIR"
echo "  Reference Cache Dir: $HIP_REFERENCE_CACHE_DIR"
echo "  Compile Cache Enabled: $HIP_ENABLE_REF_COMPILE_CACHE"
echo "  Golden Cache Enabled: $HIP_ENABLE_REF_GOLDEN_CACHE"
echo "  Perf Cache Enabled: $HIP_ENABLE_REF_PERF_CACHE"
echo "  Perf Cache TTL (s): $HIP_REF_PERF_CACHE_TTL_S"
echo "  Role: Worker (Evaluator)"
echo "========================================"

# Start the worker server
gunicorn server_req_deploy_hip2hip_batch:app \
    -k uvicorn.workers.UvicornWorker \
    --workers 1 \
    --bind 0.0.0.0:$PORT \
    --timeout 1000 \
    --log-level info

