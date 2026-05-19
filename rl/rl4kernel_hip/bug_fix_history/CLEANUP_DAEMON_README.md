# 自动清理守护进程使用说明

## 功能

自动清理卡住的HIP编译和评估进程，防止OOM。

## 清理内容

1. **卡住的进程** (默认超过5分钟)
   - `hip_comp_*` Python进程
   - `ninja` 编译进程（编译HIP kernel）
   - `hipcc` / `clang++` 编译器进程

2. **临时文件** (超过1小时)
   - `/tmp/torch_extensions/hip_hip_*`
   - `/tmp/hip_eval_*`

3. **内存监控**
   - 当内存使用率 > 85% 时，清理更激进（3分钟超时）

## 使用方法

### 自动启动（推荐）

训练脚本 `hip2hip_train.sh` 已集成自动启动：

```bash
./hip2hip_train.sh
```

守护进程会在训练开始时自动启动，每60秒检查一次。

### 手动启动

```bash
# 后台运行守护进程
./auto_cleanup_stuck_processes.sh --daemon &

# 或使用nohup（训练脚本的方式）
nohup ./auto_cleanup_stuck_processes.sh --daemon > /tmp/hip_cleanup_daemon.log 2>&1 &
```

### 停止守护进程

```bash
./stop_cleanup_daemon.sh
```

### 单次运行（不守护）

```bash
# 立即执行一次清理
./auto_cleanup_stuck_processes.sh
```

## 查看日志

```bash
# 实时查看守护进程日志
tail -f /tmp/hip_cleanup.log

# 查看守护进程输出（如果用nohup启动）
tail -f /tmp/hip_cleanup_daemon.log
```

## 配置参数

编辑 `auto_cleanup_stuck_processes.sh` 调整参数：

```bash
MAX_PROCESS_AGE_SECONDS=300  # 进程超时时间（秒），默认5分钟
```

## 典型日志输出

```
[2025-11-11 06:45:00] ========== Auto Cleanup Started ==========
[2025-11-11 06:45:00] MAX_PROCESS_AGE_SECONDS=300
[2025-11-11 06:45:00] Running in daemon mode (checking every 60 seconds)...
[2025-11-11 06:45:00] Memory usage: 72% (Available: 35GB / Total: 125GB)
[2025-11-11 06:45:00] Checking hip_comp processes...
[2025-11-11 06:45:00] Killing stuck hip_comp process: PID=12345, AGE=320s, CMD=python
[2025-11-11 06:45:00] Killed 1 hip_comp processes
[2025-11-11 06:45:00] Checking ninja processes...
[2025-11-11 06:45:00] Killed 0 ninja processes
[2025-11-11 06:45:00] Checking hipcc/clang++ processes...
[2025-11-11 06:45:00] Killed 0 compiler processes
[2025-11-11 06:45:00] Cleaning old temp files...
```

## Crontab定期运行（可选）

如果不想用守护进程，可以设置crontab：

```bash
# 编辑crontab
crontab -e

# 添加：每5分钟运行一次
*/5 * * * * /home/zeping.li@amd.com/work/HIP_Kernel_LLM_RL/auto_cleanup_stuck_processes.sh >> /tmp/hip_cleanup_cron.log 2>&1
```

## 监控脚本状态

```bash
# 检查守护进程是否运行
ps aux | grep auto_cleanup_stuck_processes.sh

# 检查PID文件
cat /tmp/hip_cleanup_daemon.pid

# 查看最近清理了什么
grep "Killing stuck" /tmp/hip_cleanup.log | tail -20
```

## 注意事项

1. **进程年龄判断**: 只清理运行时间超过阈值的进程，避免误杀正常进程
2. **内存自适应**: 内存压力大时（>85%），自动降低超时阈值到3分钟
3. **日志记录**: 所有清理操作都会记录到 `/tmp/hip_cleanup.log`
4. **重复启动保护**: 如果守护进程已在运行，不会重复启动

## 故障排查

### 守护进程没有启动

```bash
# 手动运行检查错误
./auto_cleanup_stuck_processes.sh --daemon

# 查看日志
cat /tmp/hip_cleanup_daemon.log
```

### 进程没有被清理

检查进程年龄：
```bash
ps -eo pid,etimes,cmd | grep hip_comp
```

如果 `etimes` (运行时间秒数) < 300，不会被清理。可以降低 `MAX_PROCESS_AGE_SECONDS`。

### OOM仍然发生

考虑：
1. 降低 `MAX_PROCESS_AGE_SECONDS` 到 180 (3分钟)
2. 降低训练配置中的 `SF_MAX_CONCURRENT`
3. 减小 `BATCH_SIZE` 和 `ROLLOUT_N`

