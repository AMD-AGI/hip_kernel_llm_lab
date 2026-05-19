#!/bin/bash
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
# 如果未来需要开启防火墙，请确保开放端口: 6379, 8265, 10000-19999
echo "=== Stopping firewalls for connectivity ==="
systemctl stop firewalld 2>/dev/null || true
systemctl stop ufw 2>/dev/null || true

# 2. 清理旧的 Ray 进程
echo "=== Cleaning up old Ray processes ==="
ray stop 2>/dev/null || true

# 3. 启动 Ray Head
# 指定端口范围 10002-19999，避免随机使用高位端口和与client_server端口冲突
echo "=== Starting Ray Head Node ==="
ray start --head \
  --node-ip-address=10.254.6.41 \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --num-gpus=8 \
  --min-worker-port=10002 \
  --max-worker-port=19999 \
  --disable-usage-stats

echo "Ray Head started. Dashboard: http://10.254.6.41:8265"
