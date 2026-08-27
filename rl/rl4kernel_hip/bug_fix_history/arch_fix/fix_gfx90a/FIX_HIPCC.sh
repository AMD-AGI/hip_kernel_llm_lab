#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# 修复 hipcc 的 amdgpu-target 参数问题
# 这个脚本需要 root 权限运行

set -e

echo "============================================================"
echo "修复 hipcc amdgpu-target 参数问题"
echo "============================================================"

# 检查是否有 root 权限
if [ "$EUID" -ne 0 ]; then 
    echo "❌ 请使用 root 权限运行此脚本："
    echo "   sudo bash FIX_HIPCC.sh"
    exit 1
fi

# 备份原始 hipcc
if [ ! -f /opt/rocm/bin/hipcc.original ]; then
    echo "1. 备份原始 hipcc..."
    cp /opt/rocm/bin/hipcc /opt/rocm/bin/hipcc.original
    echo "   ✓ 已备份到 /opt/rocm/bin/hipcc.original"
else
    echo "1. 检测到已存在备份，跳过备份步骤"
fi

# 创建 wrapper 脚本
echo "2. 创建 hipcc wrapper..."
cat > /opt/rocm/bin/hipcc << 'EOF'
#!/bin/bash

# hipcc wrapper - 修复 amdgpu-target 参数问题
# 强制只使用 gfx942，避免分号分隔符导致的问题

# 设置环境变量
export HCC_AMDGPU_TARGET="gfx942"
export AMDGPU_TARGETS="gfx942"

# 调用原始 hipcc
exec /opt/rocm/bin/hipcc.original "$@"
EOF

chmod +x /opt/rocm/bin/hipcc
echo "   ✓ Wrapper 已创建"

# 测试
echo ""
echo "3. 测试 hipcc..."
/opt/rocm/bin/hipcc --version | head -3

echo ""
echo "============================================================"
echo "✅ 修复完成！"
echo "============================================================"
echo ""
echo "修复内容:"
echo "  - 原始 hipcc 已备份到: /opt/rocm/bin/hipcc.original"
echo "  - 新的 wrapper 强制使用 HCC_AMDGPU_TARGET=gfx942"
echo ""
echo "如需恢复原始版本:"
echo "  mv /opt/rocm/bin/hipcc.original /opt/rocm/bin/hipcc"
echo ""

