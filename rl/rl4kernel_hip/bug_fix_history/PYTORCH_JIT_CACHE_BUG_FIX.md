# PyTorch JIT缓存Bug修复

## 🐛 问题发现

在veRL训练中发现**严重的PyTorch JIT缓存bug**：

- PyTorch根据kernel_name缓存编译结果
- 同一kernel_name但不同代码会错误复用缓存
- 导致新生成的代码未被评估，RL训练完全失效

## ✅ 修复状态

**状态**: ✅ 已修复并验证  
**优先级**: 🔴 P0 - 关键性bug  
**影响**: veRL训练从失效 → 正常

## 📁 相关文件

### 核心修复代码

修复已应用到以下文件：

- ✅ `reward/reward.py` (第120-125行)
- ✅ `reward/reward_batch.py` (第144-147行)

**修复内容**：添加代码hash到kernel_name

```python
# 修复前（Bug）
kernel_name = "172_coalesced_tiling_kernel_base"  # 固定名称

# 修复后（正确）
import hashlib
code_hash = hashlib.md5(hip_src.encode()).hexdigest()[:8]
kernel_name = f"172_coalesced_tiling_kernel_base_{code_hash}"  # 唯一名称
```

### 验证测试

所有验证测试和文档已整理到：

**📂 `pytorch_jit_cache_fix/`**

包含：
- ✅ 3个测试脚本（单元测试、集成测试、veRL模拟）
- ✅ 一键验证脚本
- ✅ 完整验证文档
- ✅ README说明

## 🚀 快速验证

```bash
# 运行所有验证测试
cd pytorch_jit_cache_fix
bash run_all_cache_tests.sh
```

**预期结果**：

```
✅ Level 1: 单元测试 - PASS
✅ Level 2: 集成测试 - PASS
✅ Level 3: 模拟测试 - PASS

🎉 所有测试通过！
```

## 📖 详细文档

进入 `pytorch_jit_cache_fix/` 目录查看：

- **`README.md`** - 快速开始指南
- **`HOW_TO_VERIFY_CACHE_FIX.md`** - 完整验证步骤
- **`VERIFICATION_GUIDE.md`** - 详细验证指南

## ⚠️ 重要提示

1. **所有修复前的训练实验应作废**
   - 基于错误reward训练的模型不可信
   - 需要重新训练

2. **清理旧缓存**（可选）
   ```bash
   rm -rf ~/.cache/torch_extensions/py310_cu126/hip_*
   ```

3. **训练监控**
   - 检查每个episode的kernel_name都不同
   - 检查reward有波动
   - 对比修复前后的训练曲线

## 📊 修复效果

### 修复前（Bug存在）

```
Episode 1: kernel="172_base", reward=0.85
Episode 2: kernel="172_base", reward=0.85  ← 相同（错误！）
Episode 3: kernel="172_base", reward=0.85  ← 相同（错误！）
→ RL训练失效
```

### 修复后（正确）

```
Episode 1: kernel="172_base_abc123", reward=0.85
Episode 2: kernel="172_base_def456", reward=0.92  ← 不同（正确）
Episode 3: kernel="172_base_ghi789", reward=0.88  ← 不同（正确）
→ RL训练正常
```

## 🎯 总结

| 项目 | 状态 |
|------|------|
| Bug识别 | ✅ 完成 |
| 修复实现 | ✅ 完成 |
| 测试验证 | ✅ 通过（3层测试） |
| 文档编写 | ✅ 完成 |
| 可用性 | ✅ 可安全用于生产 |

---

**详细信息**: 查看 `pytorch_jit_cache_fix/README.md`  
**最后更新**: 2025-11-18

