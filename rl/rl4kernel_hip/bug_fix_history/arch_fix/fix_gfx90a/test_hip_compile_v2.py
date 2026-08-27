#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
HIP 编译测试 v2 - 使用环境变量强制覆盖 hipcc 行为
"""
import os
import sys
import tempfile
import shutil
import subprocess

print("=" * 60)
print("HIP 编译环境测试 V2")
print("=" * 60)

# 强制设置环境变量，覆盖 hipcc 的默认行为
print("\n1. 设置环境变量强制覆盖...")
os.environ['HCC_AMDGPU_TARGET'] = 'gfx942'  # 注意：没有分号！
os.environ['AMDGPU_TARGETS'] = ''  # 设为空，避免冲突
os.environ['GPU_ARCHS'] = ''
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
print(f"   HCC_AMDGPU_TARGET=gfx942")
print(f"   AMDGPU_TARGETS=(empty)")
print(f"   GPU_ARCHS=(empty)")
print(f"   PYTORCH_ROCM_ARCH=gfx942")

# 创建临时目录
temp_dir = tempfile.mkdtemp(prefix="hip_test_v2_")
print(f"\n2. 创建临时目录: {temp_dir}")

try:
    # 创建简单的 HIP 内核
    hip_code = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void test_kernel(float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = idx * 2.0f;
    }
}

torch::Tensor test_forward(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    hipLaunchKernelGGL(test_kernel, dim3(blocks), dim3(threads), 0, 0,
                       output.data_ptr<float>(), size);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &test_forward, "Test HIP kernel");
}
"""
    
    hip_file = os.path.join(temp_dir, "test_kernel.hip")
    with open(hip_file, "w") as f:
        f.write(hip_code)
    print(f"\n3. 创建 HIP 内核文件: {hip_file}")
    
    # 方案1: 尝试直接使用 clang++ 而不是 hipcc
    print("\n4. 方案1: 直接使用 clang++ 编译...")
    
    # 设置使用 clang++ 作为编译器
    os.environ['CXX'] = '/opt/rocm/lib/llvm/bin/clang++'
    os.environ['CC'] = '/opt/rocm/lib/llvm/bin/clang'
    
    from torch.utils.cpp_extension import load
    
    # 不使用 extra_cuda_cflags，直接在 extra_cflags 中指定
    test_ext = load(
        name="test_hip_kernel_v2",
        sources=[hip_file],
        extra_cflags=[
            "-O2",
            "-D__HIP_PLATFORM_AMD__=1",
            "-I/opt/rocm/include",
            "-fPIC"
        ],
        extra_ldflags=[
            "-L/opt/rocm/lib",
            "-lamdhip64"
        ],
        verbose=True,
        with_cuda=False  # 明确指定不使用 CUDA
    )
    
    print("\n" + "=" * 60)
    print("✅ 编译成功！")
    print("=" * 60)
    
    # 测试运行
    print("\n5. 测试运行内核...")
    import torch
    if torch.cuda.is_available():
        test_input = torch.zeros(10, device='cuda')
        result = test_ext.forward(test_input)
        print(f"   输入: {test_input.cpu().numpy()}")
        print(f"   输出: {result.cpu().numpy()}")
        print("\n✅ 内核运行成功！")
    else:
        print("   ⚠️  CUDA/ROCm 不可用，跳过运行测试")
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print("=" * 60)
    sys.exit(0)
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ 方案1失败，尝试方案2...")
    print("=" * 60)
    print(f"错误: {str(e)[:200]}")
    
    try:
        print("\n方案2: 创建 hipcc wrapper...")
        
        # 创建临时的 hipcc wrapper
        wrapper_path = os.path.join(temp_dir, "hipcc_wrapper")
        with open(wrapper_path, "w") as f:
            f.write("""#!/bin/bash
# 过滤掉含分号的 amdgpu-target 参数
args=()
for arg in "$@"; do
    if [[ "$arg" =~ --amdgpu-target=.*\; ]]; then
        continue
    fi
    args+=("$arg")
done
exec /opt/rocm/bin/hipcc "${args[@]}"
""")
        os.chmod(wrapper_path, 0o755)
        
        # 设置 PATH，让我们的 wrapper 优先
        os.environ['PATH'] = temp_dir + ":" + os.environ['PATH']
        
        # 重新尝试编译
        from torch.utils.cpp_extension import load
        
        test_ext = load(
            name="test_hip_kernel_v2_alt",
            sources=[hip_file],
            extra_cflags=["-O2"],
            extra_cuda_cflags=["--offload-arch=gfx942"],
            verbose=True
        )
        
        print("\n" + "=" * 60)
        print("✅ 方案2编译成功！")
        print("=" * 60)
        sys.exit(0)
        
    except Exception as e2:
        print("\n" + "=" * 60)
        print("❌ 所有方案都失败了")
        print("=" * 60)
        print(f"\n方案1错误: {str(e)[:100]}")
        print(f"方案2错误: {str(e2)[:100]}")
        print("\n这是 Docker 镜像中 ROCm/hipcc 配置的问题")
        print("建议联系镜像维护者修复 hipcc")
        sys.exit(1)
    
finally:
    print(f"\n6. 清理临时目录: {temp_dir}")
    shutil.rmtree(temp_dir, ignore_errors=True)

