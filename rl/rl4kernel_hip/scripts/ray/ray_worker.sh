#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
# Preserve HIP device visibility when Ray launches worker processes
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
# WandB credentials must be provided by the caller, never stored in this repo.
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before launching Ray.}"

# Disable host/GPU core dump artifacts (core.*, gpucore.*) in cwd.
ulimit -c 0 || true

# 1. 停止防火墙 (确保连接畅通)
echo "=== Stopping firewalls for connectivity ==="
systemctl stop firewalld 2>/dev/null || true
systemctl stop ufw 2>/dev/null || true

# 2. 清理旧的 Ray 进程
echo "=== Cleaning up old Ray processes ==="
ray stop 2>/dev/null || true

# 3. 启动 Ray Worker
# 指定与 Head 相同的端口范围 10002-19999
echo "=== Starting Ray Worker Node ==="
ray start \
  --address=10.254.6.41:6379 \
  --node-ip-address=10.254.6.40 \
  --num-gpus=8 \
  --min-worker-port=10002 \
  --max-worker-port=19999 \
  --disable-usage-stats

echo "Ray Worker started and connecting to Head (10.254.6.41)..."
