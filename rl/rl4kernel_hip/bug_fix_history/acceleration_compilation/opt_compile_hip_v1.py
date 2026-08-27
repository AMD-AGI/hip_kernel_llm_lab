# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import subprocess
import sys
import torch
import sysconfig


def get_torch_paths():
    """获取 PyTorch 的 include 与 library 路径"""
    include_paths = []
    lib_paths = []
    try:
        from torch.utils.cpp_extension import include_paths as incs, library_paths as libs
        include_paths = incs()
        lib_paths = libs()
    except Exception:
        torch_dir = os.path.dirname(torch.__file__)
        include_paths = [
            os.path.join(torch_dir, "include"),
            os.path.join(torch_dir, "include", "torch", "csrc", "api", "include"),
        ]
        lib_paths = [os.path.join(torch_dir, "lib")]

    return include_paths, lib_paths


def get_python_include():
    """获取 Python 头文件路径"""
    try:
        out = subprocess.check_output(["python3-config", "--includes"], text=True)
        for token in out.strip().split():
            if token.startswith("-I"):
                return token[2:]
    except Exception:
        return sysconfig.get_paths()["include"]
    return None


def build_hip_extension():
    include_paths, lib_paths = get_torch_paths()
    python_include = get_python_include()

    hip_file = "hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"
    output_so = "EDMLoss.so"

    cmd = [
        "hipcc", "-O2", "-std=c++17", "-fPIC", "-shared",
        hip_file, "-o", output_so,
        "-L", "/opt/rocm/lib", "-lhipblas", "-lrocblas",
        "-I", python_include,
    ]

    # 添加 PyTorch include
    for inc in include_paths:
        cmd.extend(["-I", inc])

    # 添加 PyTorch lib
    for lib in lib_paths:
        cmd.extend(["-L", lib])

    # 显式链接 PyTorch 核心库（包含 pybind）
    cmd.extend([
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
        "-ltorch_python",  # ✅ 关键：修复 undefined symbol (pybind11::type_caster)
        "-DTORCH_EXTENSION_NAME=EDMLoss",
        "--offload-arch=gfx942",
    ])

    print("🔧 编译命令：")
    print(" ".join(cmd))
    print("\n⏳ 正在编译...")

    # 使用干净的环境
    env = os.environ.copy()
    for var in ['HIPCC_COMPILE_FLAGS_APPEND', 'HIPCC_LINK_FLAGS_APPEND',
                'AMDGPU_TARGETS', 'HIP_TARGETS']:
        env.pop(var, None)

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    if result.returncode == 0:
        print("✅ 编译成功！生成文件:", output_so)
    else:
        print("❌ 编译失败！")
        print(result.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # build_hip_extension()

    import time
    s = time.time()

    build_hip_extension()
     
    e = time.time()
    print(f'compile time: {e - s}')

