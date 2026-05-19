# PyTorch JIT缓存Bug修复验证

本目录包含PyTorch JIT编译缓存bug的所有验证测试和文档。

## 🐛 问题概述

PyTorch使用kernel name缓存JIT编译结果。在veRL训练中，同一kernel_name但不同代码会错误复用缓存，导致：
- ❌ 新生成的代码未被评估
- ❌ Reward信号完全错误
- ❌ RL训练失效

## ✅ 修复方案

在kernel_name中添加代码hash作为唯一标识：

```python
# 修复前
kernel_name = "172_coalesced_tiling_kernel_base"  # 固定

# 修复后
code_hash = hashlib.md5(hip_src.encode()).hexdigest()[:8]
kernel_name = f"172_coalesced_tiling_kernel_base_{code_hash}"  # 唯一
```

**修复位置**：
- `../reward/reward.py` (第120-125行)
- `../reward/reward_batch.py` (第144-147行)

## 📁 文件说明

### 测试脚本

| 文件 | 说明 | 运行方式 |
|------|------|---------|
| `test_cache_uniqueness.py` | Level 1: 单元测试 - Hash唯一性验证 | `python test_cache_uniqueness.py` |
| `test_cache_fix_integration.py` | Level 2: 集成测试 - 端到端逻辑验证 | `python test_cache_fix_integration.py` |
| `test_verl_simulation.py` | Level 3: veRL模拟 - 训练场景模拟 | `python test_verl_simulation.py` |
| `run_all_cache_tests.sh` | 一键运行所有测试 | `bash run_all_cache_tests.sh` |

### 文档

| 文件 | 说明 |
|------|------|
| `HOW_TO_VERIFY_CACHE_FIX.md` | 完整验证指南（推荐阅读） |
| `VERIFICATION_GUIDE.md` | 详细验证步骤 |
| `README.md` | 本文档 |

## 🚀 快速开始

### 一键验证

```bash
cd pytorch_jit_cache_fix
bash run_all_cache_tests.sh
```

**预期输出**：

```
✅ Level 1: 单元测试 - PASS
✅ Level 2: 集成测试 - PASS
✅ Level 3: 模拟测试 - PASS

🎉 所有测试通过！缓存bug修复已验证。
```

### 单独运行

```bash
# Level 1: 单元测试（1秒）
python test_cache_uniqueness.py

# Level 2: 集成测试（2秒）
python test_cache_fix_integration.py

# Level 3: 模拟测试（3秒）
python test_verl_simulation.py
```

## 📊 测试覆盖

| 测试层次 | 验证内容 | 状态 |
|---------|---------|------|
| Level 1 | Hash函数唯一性 | ✅ 通过 |
| Level 2 | 端到端逻辑正确性 | ✅ 通过 |
| Level 3 | veRL训练场景模拟 | ✅ 通过 |
| Level 4 | 实际veRL训练 | 待验证 |

## 🔍 实际训练验证

在真实veRL训练中添加监控：

```python
# 训练循环中
for episode in range(num_episodes):
    reward = compute_score(...)
    print(f"Episode {episode}: kernel={kernel_name}, reward={reward}")
```

**检查**：
- ✅ 每个episode的kernel_name后缀都不同
- ✅ Reward有波动（说明评估了不同代码）

## ⚠️ 重要提示

1. **修复前的训练实验应作废**
   - 基于错误reward训练的模型不可信
   - 需要使用修复后的代码重新训练

2. **清理旧缓存**（可选）
   ```bash
   rm -rf ~/.cache/torch_extensions/py310_cu126/hip_*
   ```

3. **监控训练**
   - 检查kernel_name唯一性
   - 检查reward波动
   - 对比修复前后训练曲线

## 📖 详细文档

- **快速验证**: 查看本README
- **详细步骤**: 阅读 `HOW_TO_VERIFY_CACHE_FIX.md`
- **完整指南**: 阅读 `VERIFICATION_GUIDE.md`

## 🎯 总结

- **问题严重性**: 🔴 P0 - 导致veRL训练完全失效
- **修复状态**: ✅ 已完成并验证
- **测试状态**: ✅ 所有测试通过
- **可用性**: ✅ 可安全用于生产

---

**最后更新**: 2025-11-18

