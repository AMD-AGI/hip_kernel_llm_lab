#!/bin/bash
# 一键修复并测试 HIP 编译问题

echo "============================================================"
echo "一键修复 HIP 编译问题"
echo "============================================================"

cd /home/zeping.li@amd.com/work/HIP_Kernel_LLM_RL

# 步骤 1: 检查权限
if [ "$EUID" -ne 0 ]; then 
    echo "❌ 此脚本需要 root 权限"
    echo ""
    echo "请使用以下命令运行:"
    echo "  sudo bash quick_fix.sh"
    exit 1
fi

# 步骤 2: 恢复之前的修改（如果有）
echo ""
echo "步骤 1/4: 清理之前的修改..."
if [ -f /opt/rocm/bin/hipcc.original ]; then
    mv /opt/rocm/bin/hipcc.original /opt/rocm/bin/hipcc
    echo "   ✓ 已恢复 hipcc"
fi

# 步骤 3: 运行修复
echo ""
echo "步骤 2/4: 应用修复..."
bash FIX_HIPCC_V2.sh

# 步骤 4: 清除编译缓存
echo ""
echo "步骤 3/4: 清除编译缓存..."
rm -rf /root/.cache/torch_extensions/*
rm -rf ~/.cache/torch_extensions/* 2>/dev/null || true
echo "   ✓ 缓存已清除"

# 步骤 5: 运行测试
echo ""
echo "步骤 4/4: 运行测试..."
echo "============================================================"
python3  ./fix_bugs/fix_gfx90a/test_hip_compile_v2.py

exit_code=$?

echo ""
echo "============================================================"
if [ $exit_code -eq 0 ]; then
    echo "🎉 修复成功！HIP 编译环境已就绪"
    echo ""
    echo "现在可以运行你的程序了："
    echo "  python3 reward/reward.py"
else
    echo "❌ 测试失败，请查看上面的错误信息"
    echo ""
    echo "请尝试："
    echo "  1. 查看详细日志"
    echo "  2. 阅读 SOLUTION_V2.md 获取更多信息"
fi
echo "============================================================"

exit $exit_code

