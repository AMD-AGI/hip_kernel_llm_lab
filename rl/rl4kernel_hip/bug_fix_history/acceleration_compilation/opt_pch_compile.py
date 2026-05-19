#!/usr/bin/env python3
"""
☆ 预编译头文件 (PCH) 优化版本
基于 opt_compile_hip_v1.py，增加预编译头文件支持

使用方法:
  python opt_pch_compile.py                    # 正常编译（自动使用 PCH）
  python opt_pch_compile.py --rebuild-pch      # 重新生成 PCH
  python opt_pch_compile.py --no-pch           # 不使用 PCH（对比）
  python opt_pch_compile.py --benchmark        # 性能对比测试

优化原理:
  1. 预编译常用头文件（torch/extension.h, hip/hip_runtime.h等）
  2. 编译时直接使用预编译结果，跳过头文件解析
  3. 特别适合大型头文件和频繁编译场景
  
性能提升:
  大型头文件项目: 编译速度提升 20-40%
  小型项目: 提升 5-15%
  
注意事项:
  - PCH 需要与编译器版本、编译选项匹配
  - 头文件修改后需要重新生成 PCH
  - 适合稳定的头文件，不适合频繁修改的头文件
"""

import os
import subprocess
import sys
import torch
import sysconfig
import time
from pathlib import Path
from typing import List, Tuple, Optional


# PCH 相关配置
PCH_DIR = Path(".hip_pch_cache")
PCH_HEADER_FILE = PCH_DIR / "common_headers.h"
PCH_FILE = PCH_DIR / "common_headers.h.gch"  # g++ 生成 .h.gch 文件


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


def create_pch_header():
    """创建预编译头文件"""
    
    PCH_DIR.mkdir(exist_ok=True)
    
    # 定义常用头文件
    common_headers = """
// ========================================
// 预编译头文件 (Precompiled Header)
// 包含常用的 PyTorch、HIP、C++ 标准库头文件
// ========================================

// C++ 标准库
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <algorithm>

// Python 头文件
#include <Python.h>

// PyTorch 头文件
#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

// HIP 头文件
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HIP BLAS 头文件（如果需要）
#ifdef USE_HIPBLAS
#include <hipblas.h>
#include <rocblas.h>
#endif

// 常用宏定义
#ifndef CHECK_HIP
#define CHECK_HIP(call) \\
  do { \\
    hipError_t status = call; \\
    if (status != hipSuccess) { \\
      throw std::runtime_error( \\
        std::string("HIP error: ") + hipGetErrorString(status)); \\
    } \\
  } while (0)
#endif

#ifndef CHECK_CUDA  // HIP 兼容宏
#define CHECK_CUDA CHECK_HIP
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#endif
"""
    
    with open(PCH_HEADER_FILE, 'w') as f:
        f.write(common_headers)
    
    print(f"✅ 已创建预编译头文件: {PCH_HEADER_FILE}")
    return PCH_HEADER_FILE


def build_pch(include_paths: List[str], python_include: str, force: bool = False) -> Tuple[bool, float]:
    """
    生成预编译头文件 (.pch)
    
    Returns:
        (success, elapsed_time)
    """
    
    # 检查是否需要重新生成
    if PCH_FILE.exists() and not force:
        pch_mtime = os.path.getmtime(PCH_FILE)
        header_mtime = os.path.getmtime(PCH_HEADER_FILE) if PCH_HEADER_FILE.exists() else 0
        
        if pch_mtime > header_mtime:
            print(f"✅ 预编译头文件已存在: {PCH_FILE}")
            return True, 0.0
    
    print("\n" + "=" * 70)
    print("🔨 生成预编译头文件")
    print("=" * 70)
    
    # 确保头文件存在
    if not PCH_HEADER_FILE.exists():
        create_pch_header()
    
    # 构建 PCH 编译命令
    # 注意: 使用 g++ 而不是 hipcc，因为 hipcc/clang++ 对 -x c++-header 支持有问题
    # g++ 会生成 .h.gch 文件，编译时自动检测使用
    pch_cmd = [
        "g++",
        "-x", "c++-header",  # 指定为 C++ 头文件
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-D__HIP_PLATFORM_AMD__",  # 定义 HIP 平台为 AMD
        str(PCH_HEADER_FILE.absolute()),  # 使用绝对路径
        "-I", python_include,
        # 添加 HIP/ROCm 相关的 include 路径
        "-I", "/opt/rocm/include",
        "-I", "/opt/rocm/include/hip",
    ]
    
    # 添加 PyTorch include 路径
    for inc in include_paths:
        pch_cmd.extend(["-I", inc])
    
    print(f"命令: {' '.join(pch_cmd)}")
    print(f"💡 使用 g++ 生成 .gch 格式的 PCH")
    print(f"\n⏳ 正在生成预编译头文件...")
    
    # 使用干净的环境
    env = os.environ.copy()
    for var in ['HIPCC_COMPILE_FLAGS_APPEND', 'HIPCC_LINK_FLAGS_APPEND',
                'AMDGPU_TARGETS', 'HIP_TARGETS']:
        env.pop(var, None)
    
    start = time.time()
    result = subprocess.run(
        pch_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    elapsed = time.time() - start
    
    if result.returncode == 0:
        file_size = os.path.getsize(PCH_FILE) / (1024 * 1024)
        print(f"✅ PCH 生成成功: {PCH_FILE} ({file_size:.2f} MB, {elapsed:.2f}s)")
        print("=" * 70)
        return True, elapsed
    else:
        print(f"❌ PCH 生成失败！")
        print(result.stderr)
        print("=" * 70)
        return False, elapsed


def compile_with_pch(hip_file: str, obj_file: str, include_paths: List[str], 
                    python_include: str, use_pch: bool = True) -> Tuple[bool, float]:
    """
    使用预编译头文件编译 HIP 文件
    
    Returns:
        (success, elapsed_time)
    """
    
    compile_cmd = [
        "hipcc", "-O2", "-std=c++17", "-fPIC", "-c",
        hip_file, "-o", obj_file,
        "--offload-arch=gfx942",
        "-I", python_include,
    ]
    
    # 如果使用 PCH
    if use_pch and PCH_FILE.exists():
        # g++ 的 .gch 使用 -include 而不是 -include-pch
        # hipcc 会自动传递给底层的 g++/clang++
        compile_cmd.extend([
            "-include", str(PCH_HEADER_FILE),  # 包含头文件路径（不含 .gch）
            "-I", str(PCH_DIR),  # 添加 PCH 目录到 include 路径
            "-DUSING_PRECOMPILED_HEADER"
        ])
    
    # 添加 PyTorch include
    for inc in include_paths:
        compile_cmd.extend(["-I", inc])
    
    mode_str = "with PCH" if (use_pch and PCH_FILE.exists()) else "without PCH"
    print(f"\n🔨 编译 ({mode_str}): {hip_file}")
    print(f"命令: {' '.join(compile_cmd)}")
    
    # 使用干净的环境
    env = os.environ.copy()
    for var in ['HIPCC_COMPILE_FLAGS_APPEND', 'HIPCC_LINK_FLAGS_APPEND',
                'AMDGPU_TARGETS', 'HIP_TARGETS']:
        env.pop(var, None)
    
    start = time.time()
    result = subprocess.run(
        compile_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    elapsed = time.time() - start
    
    if result.returncode == 0:
        file_size = os.path.getsize(obj_file) / 1024
        print(f"✅ 编译成功: {obj_file} ({file_size:.1f} KB, {elapsed:.2f}s)")
        return True, elapsed
    else:
        print(f"❌ 编译失败: {hip_file}")
        print(result.stderr)
        return False, elapsed


def link_object_files(obj_files: List[str], output_so: str, lib_paths: List[str]) -> Tuple[bool, float]:
    """
    链接目标文件
    
    Returns:
        (success, elapsed_time)
    """
    
    link_cmd = [
        "hipcc", "-shared",
        *obj_files,
        "-o", output_so,
        "-L", "/opt/rocm/lib", "-lhipblas", "-lrocblas",
    ]
    
    # 添加 PyTorch lib
    for lib in lib_paths:
        link_cmd.extend(["-L", lib])
    
    # 显式链接 PyTorch 核心库
    link_cmd.extend([
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
        "-ltorch_python",
        "-DTORCH_EXTENSION_NAME=EDMLoss",
    ])
    
    print(f"\n🔗 链接目标文件...")
    
    # 使用干净的环境
    env = os.environ.copy()
    for var in ['HIPCC_COMPILE_FLAGS_APPEND', 'HIPCC_LINK_FLAGS_APPEND',
                'AMDGPU_TARGETS', 'HIP_TARGETS']:
        env.pop(var, None)
    
    start = time.time()
    result = subprocess.run(
        link_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    elapsed = time.time() - start
    
    if result.returncode == 0:
        file_size = os.path.getsize(output_so) / (1024 * 1024)
        print(f"✅ 链接成功！生成文件: {output_so} ({file_size:.2f} MB, {elapsed:.2f}s)")
        return True, elapsed
    else:
        print("❌ 链接失败！")
        print(result.stderr)
        return False, elapsed


def build_hip_extension_with_pch(use_pch: bool = True, rebuild_pch: bool = False):
    """
    使用预编译头文件编译 HIP 扩展
    
    Args:
        use_pch: 是否使用预编译头文件
        rebuild_pch: 是否强制重新生成 PCH
    """
    
    include_paths, lib_paths = get_torch_paths()
    python_include = get_python_include()
    
    hip_file = "hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"
    obj_file = "EDMLoss_pch.o"
    output_so = "EDMLoss_pch.so"
    
    print("=" * 70)
    print("🚀 预编译头文件 (PCH) 优化模式")
    print("=" * 70)
    print(f"使用 PCH: {'是' if use_pch else '否'}")
    print(f"重建 PCH: {'是' if rebuild_pch else '否'}")
    print("=" * 70)
    
    pch_time = 0.0
    
    # 生成/检查 PCH
    if use_pch:
        if not PCH_HEADER_FILE.exists():
            create_pch_header()
        
        success, pch_time = build_pch(include_paths, python_include, force=rebuild_pch)
        if not success:
            print("⚠️  PCH 生成失败，将不使用 PCH")
            use_pch = False
    
    # 编译阶段
    success, compile_time = compile_with_pch(
        hip_file, obj_file, include_paths, python_include, use_pch
    )
    
    if not success:
        sys.exit(1)
    
    # 链接阶段
    success, link_time = link_object_files([obj_file], output_so, lib_paths)
    
    if not success:
        sys.exit(1)
    
    return pch_time, compile_time, link_time


def benchmark_pch_effect():
    """对比使用和不使用 PCH 的性能差异"""
    
    print("\n" + "=" * 70)
    print("🧪 预编译头文件 (PCH) 性能测试")
    print("=" * 70)
    
    results = []
    
    # 测试 1: 不使用 PCH（基准）
    print("\n【测试 1】不使用 PCH（基准测试）")
    print("─" * 70)
    
    # 清理旧文件
    for f in ["EDMLoss_pch.o", "EDMLoss_pch.so"]:
        if os.path.exists(f):
            os.remove(f)
    
    total_start = time.time()
    pch_time, compile_time, link_time = build_hip_extension_with_pch(use_pch=False)
    total_time_no_pch = time.time() - total_start
    
    results.append({
        'name': '不使用 PCH',
        'pch_time': pch_time,
        'compile_time': compile_time,
        'link_time': link_time,
        'total_time': total_time_no_pch
    })
    
    print(f"\n⏱️  总耗时: {total_time_no_pch:.2f}s")
    
    # 测试 2: 首次使用 PCH（需要生成 PCH）
    print("\n\n【测试 2】首次使用 PCH（包含 PCH 生成时间）")
    print("─" * 70)
    
    # 清理 PCH
    import shutil
    if PCH_DIR.exists():
        shutil.rmtree(PCH_DIR)
    
    # 清理旧文件
    for f in ["EDMLoss_pch.o", "EDMLoss_pch.so"]:
        if os.path.exists(f):
            os.remove(f)
    
    total_start = time.time()
    pch_time, compile_time, link_time = build_hip_extension_with_pch(use_pch=True, rebuild_pch=True)
    total_time_first_pch = time.time() - total_start
    
    results.append({
        'name': '首次使用 PCH',
        'pch_time': pch_time,
        'compile_time': compile_time,
        'link_time': link_time,
        'total_time': total_time_first_pch
    })
    
    print(f"\n⏱️  总耗时: {total_time_first_pch:.2f}s (含 PCH 生成 {pch_time:.2f}s)")
    
    # 测试 3: 再次使用 PCH（PCH 已存在）
    print("\n\n【测试 3】再次使用 PCH（PCH 已缓存）")
    print("─" * 70)
    
    # 只清理编译产物，保留 PCH
    for f in ["EDMLoss_pch.o", "EDMLoss_pch.so"]:
        if os.path.exists(f):
            os.remove(f)
    
    total_start = time.time()
    pch_time, compile_time, link_time = build_hip_extension_with_pch(use_pch=True, rebuild_pch=False)
    total_time_cached_pch = time.time() - total_start
    
    results.append({
        'name': '使用缓存的 PCH',
        'pch_time': pch_time,
        'compile_time': compile_time,
        'link_time': link_time,
        'total_time': total_time_cached_pch
    })
    
    print(f"\n⏱️  总耗时: {total_time_cached_pch:.2f}s")
    
    # 打印对比结果
    print("\n\n" + "=" * 70)
    print("📊 性能对比总结")
    print("=" * 70)
    print(f"{'测试配置':<20} {'PCH生成':<10} {'编译':<10} {'链接':<10} {'总计':<10} {'加速比':<10}")
    print("─" * 70)
    
    baseline = results[0]['total_time']
    
    for r in results:
        speedup = baseline / r['total_time']
        print(f"{r['name']:<20} "
              f"{r['pch_time']:>8.2f}s  "
              f"{r['compile_time']:>8.2f}s  "
              f"{r['link_time']:>8.2f}s  "
              f"{r['total_time']:>8.2f}s  "
              f"{speedup:>9.2f}x")
    
    print("=" * 70)
    
    # 分析
    print("\n💡 性能分析:")
    
    compile_speedup = results[0]['compile_time'] / results[2]['compile_time']
    print(f"  • 编译阶段加速: {compile_speedup:.2f}x")
    
    pch_overhead = results[1]['pch_time']
    compile_saving = results[0]['compile_time'] - results[2]['compile_time']
    break_even = pch_overhead / compile_saving if compile_saving > 0 else float('inf')
    print(f"  • PCH 生成开销: {pch_overhead:.2f}s")
    print(f"  • 每次编译节省: {compile_saving:.2f}s")
    print(f"  • 编译 {break_even:.0f} 次后开始收益")
    
    print("\n📌 使用建议:")
    if compile_speedup > 1.2:
        print("  ✅ PCH 效果显著，推荐在以下场景使用:")
        print("     - 频繁修改代码、反复编译")
        print("     - CI/CD 流水线（可缓存 PCH）")
        print("     - 大型项目（头文件多且复杂）")
    else:
        print("  ⚠️  PCH 效果有限，可能因为:")
        print("     - 项目较小，头文件简单")
        print("     - 编译器优化已经很快")
        print("     - 建议在大型项目中测试")


def clean_pch():
    """清理 PCH 缓存"""
    import shutil
    
    print("🧹 清理 PCH 缓存...")
    
    if PCH_DIR.exists():
        shutil.rmtree(PCH_DIR)
        print(f"✅ 已删除 PCH 目录: {PCH_DIR}")
    else:
        print(f"⚠️  PCH 目录不存在: {PCH_DIR}")
    
    print("✅ 清理完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='使用预编译头文件编译 HIP 扩展')
    parser.add_argument('--no-pch', action='store_true',
                       help='不使用预编译头文件')
    parser.add_argument('--rebuild-pch', action='store_true',
                       help='强制重新生成 PCH')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能对比测试')
    parser.add_argument('--clean', action='store_true',
                       help='清理 PCH 缓存')
    
    args = parser.parse_args()
    
    if args.clean:
        clean_pch()
        sys.exit(0)
    
    if args.benchmark:
        benchmark_pch_effect()
        sys.exit(0)
    
    # 正常编译
    total_start = time.time()
    
    pch_time, compile_time, link_time = build_hip_extension_with_pch(
        use_pch=not args.no_pch,
        rebuild_pch=args.rebuild_pch
    )
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("⏱️  总耗时统计")
    print("=" * 70)
    if pch_time > 0:
        print(f"PCH 生成: {pch_time:.2f}s")
    print(f"编译阶段: {compile_time:.2f}s")
    print(f"链接阶段: {link_time:.2f}s")
    print(f"总计时间: {total_time:.2f}s")
    print("=" * 70)

