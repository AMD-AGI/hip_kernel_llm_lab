#!/bin/bash
# 极简 HIP 环境检查脚本

echo "============================================================"
echo "HIP 编译环境检查"
echo "============================================================"

echo -e "\n1. 检查可能冲突的环境变量..."
conflict_vars=("AMDGPU_TARGETS" "HCC_AMDGPU_TARGET" "GPU_ARCHS")
has_conflict=0

for var in "${conflict_vars[@]}"; do
    if [ ! -z "${!var}" ]; then
        echo "   ⚠️  发现冲突变量: $var=${!var}"
        has_conflict=1
    else
        echo "   ✓ $var 未设置"
    fi
done

echo -e "\n2. 检查 ROCm 环境..."
echo "   ROCm 路径: ${ROCM_HOME:-未设置}"
echo "   HIP 平台: ${HIP_PLATFORM:-未设置}"

echo -e "\n3. 检查编译器..."
if command -v hipcc &> /dev/null; then
    echo "   ✓ hipcc 可用"
    hipcc --version | head -3
else
    echo "   ❌ hipcc 不可用"
fi

echo -e "\n4. 检查 Python 和 PyTorch..."
python3 -c "
import torch
print(f'   PyTorch 版本: {torch.__version__}')
print(f'   CUDA 可用: {torch.cuda.is_available()}')
if hasattr(torch.version, 'hip'):
    print(f'   HIP 版本: {torch.version.hip}')
else:
    print('   HIP 版本: N/A (可能是 CUDA 版本的 PyTorch)')
" 2>/dev/null || echo "   ❌ PyTorch 检查失败"

echo -e "\n============================================================"
if [ $has_conflict -eq 1 ]; then
    echo "⚠️  发现可能导致编译冲突的环境变量"
    echo "建议在 Python 脚本中清除这些变量后再编译"
else
    echo "✓ 环境变量检查通过"
fi
echo "============================================================"

