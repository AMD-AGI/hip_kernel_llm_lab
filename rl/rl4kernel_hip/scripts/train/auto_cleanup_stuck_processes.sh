#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# 自动清理卡住的HIP编译和评估进程
# 用法: 
#   1. 在tmux中后台运行: ./auto_cleanup_stuck_processes.sh &
#   2. 或者通过crontab定期运行: */5 * * * * /path/to/auto_cleanup_stuck_processes.sh

set -euo pipefail

LOG_FILE="/tmp/hip_cleanup.log"
MAX_PROCESS_AGE_SECONDS=1200  # 20分钟

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 清理超过N秒的hip_comp进程
cleanup_hip_comp() {
    log "Checking hip_comp processes..."
    local killed=0
    
    # 查找所有hip_comp进程
    while read -r pid elapsed cmd; do
        if [[ -n "$pid" ]] && [[ "$elapsed" =~ ^[0-9]+$ ]]; then
            if (( elapsed > MAX_PROCESS_AGE_SECONDS )); then
                log "Killing stuck hip_comp process: PID=$pid, AGE=${elapsed}s, CMD=$cmd"
                kill -9 "$pid" 2>/dev/null || true
                ((killed++))
            fi
        fi
    done < <(ps -eo pid,etimes,cmd | grep -E "python.*hip_comp_" | grep -v grep | awk '{print $1, $2, $3}')
    
    log "Killed $killed hip_comp processes"
}

# 清理超过N秒的ninja编译进程
cleanup_ninja() {
    log "Checking ninja processes..."
    local killed=0
    
    while read -r pid elapsed; do
        if [[ -n "$pid" ]] && [[ "$elapsed" =~ ^[0-9]+$ ]]; then
            if (( elapsed > MAX_PROCESS_AGE_SECONDS )); then
                log "Killing stuck ninja process: PID=$pid, AGE=${elapsed}s"
                kill -9 "$pid" 2>/dev/null || true
                ((killed++))
            fi
        fi
    done < <(ps -eo pid,etimes,cmd | grep -E "ninja.*hip_hip_" | grep -v grep | awk '{print $1, $2}')
    
    log "Killed $killed ninja processes"
}

# 清理超过N秒的hipcc/clang++进程
cleanup_compilers() {
    log "Checking hipcc/clang++ processes..."
    local killed=0
    
    while read -r pid elapsed; do
        if [[ -n "$pid" ]] && [[ "$elapsed" =~ ^[0-9]+$ ]]; then
            if (( elapsed > MAX_PROCESS_AGE_SECONDS )); then
                log "Killing stuck compiler process: PID=$pid, AGE=${elapsed}s"
                kill -9 "$pid" 2>/dev/null || true
                ((killed++))
            fi
        fi
    done < <(ps -eo pid,etimes,cmd | grep -E "(hipcc|clang\+\+).*hip_hip_" | grep -v grep | awk '{print $1, $2}')
    
    log "Killed $killed compiler processes"
}

# 清理旧的临时文件（超过1小时）
cleanup_temp_files() {
    log "Cleaning old temp files..."
    local cleaned=0
    
    # 清理编译缓存
    if [[ -d "/tmp/torch_extensions" ]]; then
        find /tmp/torch_extensions -name "hip_hip_*" -type d -mmin +60 -exec rm -rf {} + 2>/dev/null || true
        cleaned=1
    fi
    
    # 清理评估临时文件
    find /tmp -maxdepth 1 -name "hip_eval_*" -type d -mmin +60 -exec rm -rf {} + 2>/dev/null || true
    
    log "Cleaned temp files: $cleaned"
}

# 检查内存使用并报警
check_memory() {
    local mem_available_gb=$(free -g | awk '/^Mem:/ {print $7}')
    local mem_total_gb=$(free -g | awk '/^Mem:/ {print $2}')
    local mem_usage_percent=$((100 - (mem_available_gb * 100 / mem_total_gb)))
    
    log "Memory usage: ${mem_usage_percent}% (Available: ${mem_available_gb}GB / Total: ${mem_total_gb}GB)"
    
    if (( mem_usage_percent > 85 )); then
        log "WARNING: High memory usage detected! (${mem_usage_percent}%)"
        # 如果内存压力大，更激进地清理
        MAX_PROCESS_AGE_SECONDS=180  # 降到3分钟
    fi
}

# 主循环
main() {
    log "========== Auto Cleanup Started =========="
    log "MAX_PROCESS_AGE_SECONDS=$MAX_PROCESS_AGE_SECONDS"
    
    if [[ "${1:-}" == "--daemon" ]]; then
        # 守护进程模式：每分钟运行一次
        log "Running in daemon mode (checking every 60 seconds)..."
        while true; do
            check_memory
            cleanup_hip_comp
            cleanup_ninja
            cleanup_compilers
            cleanup_temp_files
            sleep 60
        done
    else
        # 单次运行模式
        check_memory
        cleanup_hip_comp
        cleanup_ninja
        cleanup_compilers
        cleanup_temp_files
        log "========== Cleanup Completed =========="
    fi
}

# 捕获退出信号
trap 'log "Cleanup script stopped"; exit 0' SIGINT SIGTERM

main "$@"

