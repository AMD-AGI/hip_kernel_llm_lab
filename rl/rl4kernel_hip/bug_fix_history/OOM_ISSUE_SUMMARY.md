# OOM 问题分析与修复

## 问题

单个HIP编译进程占用 **2141GB 内存**，导致节点 OOM。

**原因**：编译进程无超时控制 + 高并发（128）→ 内存泄漏爆炸

## 已修复

1. **编译超时保护** (`hip_kernel_check_utils_hip2hip.py`)
   - 编译：120秒超时
   - 测试：60秒超时

2. **清理守护进程加速** (`auto_cleanup_stuck_processes.sh`)
   - 进程存活上限：300秒 → 150秒

## 推荐配置

```bash
SF_MAX_CONCURRENT=64    # 从128降到64
RES_LENGTH=2048         # 从4096降回2048
GPU_MEMORY_UTILIZATION=0.6
```

## 启动检查

```bash
# 评估服务器
curl http://localhost:8080/health

# 清理守护进程
ps aux | grep auto_cleanup

# 内存状态
free -h

# 卡住的进程
ps aux | grep hip_comp
```

## 故障排查

```bash
# OOM时立即执行
ps aux --sort=-%mem | head -10
ps aux | awk '/hip_comp/ && $6 > 1000000 {print $2}' | xargs -r kill -9
```
