#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# ============================================================================
# HIP Kernel Evaluation Master Server Startup Script
# ============================================================================
# This script starts the master server which coordinates kernel evaluation
# across multiple worker nodes. The master also uses its local GPUs for
# evaluation.
#
# Usage:
#   ./setup_master.sh                    # Use default workers.yaml
#   ./setup_master.sh --config my.yaml   # Use custom config file
#   ./setup_master.sh --help             # Show help
# ============================================================================

set -e

#############################################
# Parse Arguments
#############################################
CONFIG_FILE="workers.yaml"
PORT=8080

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE   Path to workers.yaml config file (default: workers.yaml)"
            echo "  --port PORT     Port to bind (default: 8080)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

#############################################
# Validate Configuration
#############################################
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    echo "Please create it or specify a valid path with --config"
    exit 1
fi

echo "Using configuration: $CONFIG_FILE"

#############################################
# Environment Setup
#############################################
export HCC_AMDGPU_TARGET="${HCC_AMDGPU_TARGET:-gfx942}"
export AMDGPU_TARGETS="${AMDGPU_TARGETS:-$HCC_AMDGPU_TARGET}"
export HIP_EVAL_ARCH="${HIP_EVAL_ARCH:-$HCC_AMDGPU_TARGET}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HYDRA_FULL_ERROR=1
export WORKER_CONFIG="$CONFIG_FILE"

export HIP_PERF_ITERATIONS="${HIP_PERF_ITERATIONS:-1000}"
export HIP_CONFIRM_SPEEDUP_ENABLED="${HIP_CONFIRM_SPEEDUP_ENABLED:-0}"
export HIP_CONFIRM_SPEEDUP_THRESHOLD="${HIP_CONFIRM_SPEEDUP_THRESHOLD:-1.05}"
export HIP_CONFIRM_SPEEDUP_BAND="${HIP_CONFIRM_SPEEDUP_BAND:-0.02}"
export HIP_CONFIRM_PERF_ITERATIONS="${HIP_CONFIRM_PERF_ITERATIONS:-3000}"
export HIP_COMPILE_TIMEOUT_S="${HIP_COMPILE_TIMEOUT_S:-600}"
export HIP_RUN_TIMEOUT_S="${HIP_RUN_TIMEOUT_S:-600}"
export HIP_HANDLER_TIMEOUT_S="${HIP_HANDLER_TIMEOUT_S:-1200}"
export HIP_ERROR_LOG_DIR="${HIP_ERROR_LOG_DIR:-./runtime/error_log/master}"
export HIP_REFERENCE_CACHE_DIR="${HIP_REFERENCE_CACHE_DIR:-./runtime/reference_cache/master}"
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
echo "Starting HIP Kernel Master Server"
echo "========================================"
echo "Configuration:"
echo "  Config File: $CONFIG_FILE"
echo "  Port: $PORT"
echo "  Local GPUs: $HIP_VISIBLE_DEVICES"
echo "  Role: Master (Coordinator + Evaluator)"
echo "  Effective Arch: $HIP_EVAL_ARCH"
echo "  Perf Iterations: $HIP_PERF_ITERATIONS"
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
echo "========================================"

# Read and display worker count from config
if command -v python3 &> /dev/null; then
    WORKER_COUNT=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(len(cfg.get('workers', [])))" 2>/dev/null || echo "unknown")
    echo "  Remote Workers: $WORKER_COUNT"
fi
echo "========================================"

# Start the master server
gunicorn master_server:app \
    -k uvicorn.workers.UvicornWorker \
    --workers 1 \
    --bind 0.0.0.0:$PORT \
    --timeout 1000 \
    --log-level info

