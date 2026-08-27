#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
单文件编译优化策略综合测试
测试多种编译优化策略的性能对比：
- 基准编译 (v1 单步、v2 分离)
- 增量编译
- 并行编译
- 预编译头文件 (PCH)
- 编译缓存 (ccache)
"""

import os
import subprocess
import sys
import time
import shutil
from pathlib import Path

# 导入优化模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opt_incremental_compile
import opt_parallel_compile
import opt_pch_compile


def check_ccache_available():
    """检查 ccache 是否可用"""
    result = subprocess.run(
        ["which", "ccache"],
        capture_output=True
    )
    return result.returncode == 0


def clear_ccache():
    """清空 ccache 缓存"""
    subprocess.run(["ccache", "-C"], capture_output=True)
    print("✅ ccache 缓存已清空")


def test_baseline_v1():
    """测试 1: 基准 - v1 单步编译"""
    print("\n" + "=" * 70)
    print("【测试 1】基准: v1 单步编译")
    print("=" * 70)
    
    hip_file = "hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"
    output_so = "test_baseline_v1.so"
    
    # 清理旧文件
    for f in [output_so, hip_file.replace('.hip', '.o')]:
        if os.path.exists(f):
            os.remove(f)
    
    include_paths, lib_paths = opt_incremental_compile.get_torch_paths()
    python_include = opt_incremental_compile.get_python_include()
    
    # 构建编译命令
    cmd = [
        "hipcc", "-O2", "-std=c++17", "-fPIC", "-shared",
        hip_file, "-o", output_so,
        "--offload-arch=gfx942",
        "-I", python_include,
    ]
    
    for inc in include_paths:
        cmd.extend(["-I", inc])
    
    for lib in lib_paths:
        cmd.extend(["-L", lib])
    
    cmd.extend([
        "-L", "/opt/rocm/lib",
        "-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10",
        "-lhipblas", "-lrocblas",
        "-DTORCH_EXTENSION_NAME=EDMLoss",
    ])
    
    # 使用干净的环境
    env = os.environ.copy()
    for var in ['HIPCC_COMPILE_FLAGS_APPEND', 'HIPCC_LINK_FLAGS_APPEND',
                'AMDGPU_TARGETS', 'HIP_TARGETS']:
        env.pop(var, None)
    
    print(f"🔨 编译: {hip_file}")
    
    start = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, text=True, env=env)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print("❌ 编译失败！")
        print(result.stderr)
        return None
    
    print(f"✅ 编译成功！耗时: {elapsed:.2f}s")
    
    return {
        "name": "v1: 单步编译 (基准)",
        "compile_time": elapsed,
        "link_time": 0,  # 单步编译不分离
        "total_time": elapsed,
        "speedup": 1.0
    }


def test_baseline_v2():
    """测试 2: 基准 - v2 分离编译"""
    print("\n" + "=" * 70)
    print("【测试 2】基准: v2 分离编译")
    print("=" * 70)
    
    hip_file = "hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"
    obj_file = hip_file.replace('.hip', '.o')
    output_so = "test_baseline_v2.so"
    
    # 清理旧文件
    for f in [obj_file, output_so]:
        if os.path.exists(f):
            os.remove(f)
    
    include_paths, lib_paths = opt_incremental_compile.get_torch_paths()
    python_include = opt_incremental_compile.get_python_include()
    
    # 使用干净的环境
    env = os.environ.copy()
    for var in ['HIPCC_COMPILE_FLAGS_APPEND', 'HIPCC_LINK_FLAGS_APPEND',
                'AMDGPU_TARGETS', 'HIP_TARGETS']:
        env.pop(var, None)
    
    # 编译阶段
    compile_cmd = [
        "hipcc", "-O2", "-std=c++17", "-fPIC", "-c",
        hip_file, "-o", obj_file,
        "--offload-arch=gfx942",
        "-I", python_include,
    ]
    
    for inc in include_paths:
        compile_cmd.extend(["-I", inc])
    
    print(f"🔨 编译: {hip_file}")
    
    compile_start = time.time()
    result = subprocess.run(compile_cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, env=env)
    compile_time = time.time() - compile_start
    
    if result.returncode != 0:
        print("❌ 编译失败！")
        print(result.stderr)
        return None
    
    print(f"✅ 编译完成！耗时: {compile_time:.2f}s")
    
    # 链接阶段
    link_cmd = [
        "hipcc", "-shared", obj_file, "-o", output_so,
        "-L", "/opt/rocm/lib",
    ]
    
    for lib in lib_paths:
        link_cmd.extend(["-L", lib])
    
    link_cmd.extend([
        "-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10",
        "-lhipblas", "-lrocblas",
        "-DTORCH_EXTENSION_NAME=EDMLoss",
    ])
    
    print(f"🔗 链接: {obj_file}")
    
    link_start = time.time()
    result = subprocess.run(link_cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, env=env)
    link_time = time.time() - link_start
    
    if result.returncode != 0:
        print("❌ 链接失败！")
        print(result.stderr)
        return None
    
    print(f"✅ 链接完成！耗时: {link_time:.2f}s")
    
    total_time = compile_time + link_time
    
    return {
        "name": "v2: 分离编译",
        "compile_time": compile_time,
        "link_time": link_time,
        "total_time": total_time,
        "speedup": 1.0  # 稍后计算
    }


def test_incremental_compile(baseline_time):
    """测试 3: 增量编译"""
    print("\n" + "=" * 70)
    print("【测试 3】增量编译 - 智能跳过未修改文件")
    print("=" * 70)
    
    hip_file = "hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"
    
    # 清理缓存
    if opt_incremental_compile.CACHE_DIR.exists():
        shutil.rmtree(opt_incremental_compile.CACHE_DIR)
    
    # 清理旧文件
    obj_file = hip_file.replace('.hip', '.o')
    if os.path.exists(obj_file):
        os.remove(obj_file)
    if os.path.exists("EDMLoss_incremental.so"):
        os.remove("EDMLoss_incremental.so")
    
    # 首次编译
    print("\n📝 首次编译（建立缓存）...")
    start = time.time()
    compile_time, link_time, compile_count, skip_count = \
        opt_incremental_compile.build_hip_extension_incremental([hip_file], force=True)
    first_time = time.time() - start
    
    print(f"✅ 首次编译完成: {first_time:.2f}s")
    
    # 再次编译（无修改）
    print("\n⚡ 再次编译（无修改，应跳过）...")
    start = time.time()
    compile_time2, link_time2, compile_count2, skip_count2 = \
        opt_incremental_compile.build_hip_extension_incremental([hip_file], force=False)
    second_time = time.time() - start
    
    print(f"✅ 增量编译完成: {second_time:.2f}s")
    print(f"   跳过文件: {skip_count2} 个")
    
    speedup = baseline_time / second_time if second_time > 0 else float('inf')
    
    return {
        "name": "增量编译 (无修改)",
        "compile_time": compile_time2,
        "link_time": link_time2,
        "total_time": second_time,
        "speedup": speedup,
        "extra_info": f"跳过 {skip_count2} 个文件"
    }


def test_parallel_compile(baseline_time):
    """测试 4: 并行编译（单文件，主要测试框架）"""
    print("\n" + "=" * 70)
    print("【测试 4】并行编译 - 单文件测试")
    print("=" * 70)
    
    hip_file = "hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"
    
    # 清理旧文件
    obj_file = hip_file.replace('.hip', '.o')
    if os.path.exists(obj_file):
        os.remove(obj_file)
    if os.path.exists("EDMLoss_parallel.so"):
        os.remove("EDMLoss_parallel.so")
    
    start = time.time()
    compile_time, link_time, speedup = \
        opt_parallel_compile.build_hip_extension_parallel([hip_file], max_workers=1)
    total_time = time.time() - start
    
    actual_speedup = baseline_time / total_time if total_time > 0 else 1.0
    
    return {
        "name": "并行编译 (单文件)",
        "compile_time": compile_time,
        "link_time": link_time,
        "total_time": total_time,
        "speedup": actual_speedup,
        "extra_info": "单文件并行效果有限"
    }


def test_pch_compile(baseline_time):
    """测试 5: 预编译头文件 (PCH)"""
    print("\n" + "=" * 70)
    print("【测试 5】预编译头文件 (PCH)")
    print("=" * 70)
    
    # 清理 PCH 缓存
    if opt_pch_compile.PCH_DIR.exists():
        shutil.rmtree(opt_pch_compile.PCH_DIR)
    
    # 清理旧文件
    for f in ["EDMLoss_pch.o", "EDMLoss_pch.so"]:
        if os.path.exists(f):
            os.remove(f)
    
    # 首次使用 PCH（包含生成时间）
    print("\n📝 首次使用 PCH（生成 PCH）...")
    start = time.time()
    pch_time1, compile_time1, link_time1 = \
        opt_pch_compile.build_hip_extension_with_pch(use_pch=True, rebuild_pch=True)
    first_time = time.time() - start
    
    print(f"✅ 首次完成: {first_time:.2f}s (含 PCH 生成 {pch_time1:.2f}s)")
    
    # 清理编译产物，保留 PCH
    for f in ["EDMLoss_pch.o", "EDMLoss_pch.so"]:
        if os.path.exists(f):
            os.remove(f)
    
    # 再次使用 PCH（已缓存）
    print("\n⚡ 再次使用 PCH（已缓存）...")
    start = time.time()
    pch_time2, compile_time2, link_time2 = \
        opt_pch_compile.build_hip_extension_with_pch(use_pch=True, rebuild_pch=False)
    second_time = time.time() - start
    
    print(f"✅ 缓存 PCH 完成: {second_time:.2f}s")
    
    speedup = baseline_time / second_time if second_time > 0 else 1.0
    
    return {
        "name": "PCH (已缓存)",
        "compile_time": compile_time2,
        "link_time": link_time2,
        "total_time": second_time,
        "speedup": speedup,
        "extra_info": f"PCH 生成耗时 {pch_time1:.2f}s"
    }


def test_ccache_compile(baseline_time):
    """测试 6: ccache 编译缓存"""
    print("\n" + "=" * 70)
    print("【测试 6】ccache - 编译缓存")
    print("=" * 70)
    
    if not check_ccache_available():
        print("⚠️  ccache 未安装，跳过测试")
        print("   安装方法: sudo apt install ccache")
        return None
    
    hip_file = "hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"
    obj_file = hip_file.replace('.hip', '.o')
    output_so = "test_ccache.so"
    
    include_paths, lib_paths = opt_incremental_compile.get_torch_paths()
    python_include = opt_incremental_compile.get_python_include()
    
    # 使用干净的环境
    env = os.environ.copy()
    for var in ['HIPCC_COMPILE_FLAGS_APPEND', 'HIPCC_LINK_FLAGS_APPEND',
                'AMDGPU_TARGETS', 'HIP_TARGETS']:
        env.pop(var, None)
    
    # 清空 ccache
    clear_ccache()
    
    # 清理旧文件
    for f in [obj_file, output_so]:
        if os.path.exists(f):
            os.remove(f)
    
    # 编译命令（使用 ccache）
    compile_cmd = [
        "ccache", "hipcc", "-O2", "-std=c++17", "-fPIC", "-c",
        hip_file, "-o", obj_file,
        "--offload-arch=gfx942",
        "-I", python_include,
    ]
    
    for inc in include_paths:
        compile_cmd.extend(["-I", inc])
    
    # 首次编译（缓存未命中）
    print("\n📝 首次编译（缓存未命中）...")
    start = time.time()
    result = subprocess.run(compile_cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, env=env)
    first_compile = time.time() - start
    
    if result.returncode != 0:
        print("❌ 编译失败！")
        print(result.stderr)
        return None
    
    print(f"✅ 首次编译完成: {first_compile:.2f}s")
    
    # 链接
    link_cmd = [
        "hipcc", "-shared", obj_file, "-o", output_so,
        "-L", "/opt/rocm/lib",
    ]
    
    for lib in lib_paths:
        link_cmd.extend(["-L", lib])
    
    link_cmd.extend([
        "-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10",
        "-lhipblas", "-lrocblas",
        "-DTORCH_EXTENSION_NAME=EDMLoss",
    ])
    
    result = subprocess.run(link_cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, env=env)
    
    # 清理 .o 文件，保留 ccache
    if os.path.exists(obj_file):
        os.remove(obj_file)
    
    # 再次编译（缓存命中）
    print("\n⚡ 再次编译（缓存命中）...")
    start = time.time()
    result = subprocess.run(compile_cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, env=env)
    second_compile = time.time() - start
    
    if result.returncode != 0:
        print("❌ 编译失败！")
        return None
    
    print(f"✅ 缓存命中编译: {second_compile:.2f}s")
    
    # 链接
    link_start = time.time()
    result = subprocess.run(link_cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, env=env)
    link_time = time.time() - link_start
    
    total_time = second_compile + link_time
    speedup = baseline_time / total_time if total_time > 0 else 1.0
    
    # 显示 ccache 统计
    print("\n📊 ccache 统计:")
    subprocess.run(["ccache", "-s"], stdout=subprocess.PIPE)
    
    return {
        "name": "ccache (缓存命中)",
        "compile_time": second_compile,
        "link_time": link_time,
        "total_time": total_time,
        "speedup": speedup,
        "extra_info": f"首次 {first_compile:.2f}s → 缓存 {second_compile:.2f}s ({first_compile/second_compile:.1f}x)"
    }


def run_comprehensive_benchmark():
    """运行综合性能测试"""
    
    print("=" * 70)
    print("🚀 HIP 编译优化策略综合性能测试")
    print("=" * 70)
    
    hip_file = "hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"
    
    if not os.path.exists(hip_file):
        print(f"❌ 源文件不存在: {hip_file}")
        sys.exit(1)
    
    print(f"\n📁 测试文件: {hip_file}")
    print(f"💻 CPU 核心数: {os.cpu_count()}")
    print(f"🔧 GPU 架构: gfx942")
    
    results = []
    
    # 测试 1: 基准 v1
    result = test_baseline_v1()
    if result:
        results.append(result)
        baseline_time = result['total_time']
    else:
        print("❌ 基准测试失败，终止")
        return
    
    # 测试 2: 基准 v2
    result = test_baseline_v2()
    if result:
        result['speedup'] = baseline_time / result['total_time']
        results.append(result)
    
    # 测试 3: 增量编译
    result = test_incremental_compile(baseline_time)
    if result:
        results.append(result)
    
    # 测试 4: 并行编译
    result = test_parallel_compile(baseline_time)
    if result:
        results.append(result)
    
    # 测试 5: PCH
    result = test_pch_compile(baseline_time)
    if result:
        results.append(result)
    
    # 测试 6: ccache
    result = test_ccache_compile(baseline_time)
    if result:
        results.append(result)
    
    # 打印综合对比
    print("\n\n" + "=" * 80)
    print("📊 编译优化策略性能对比总结")
    print("=" * 80)
    print(f"\n{'策略':<25} {'编译':<10} {'链接':<10} {'总计':<10} {'加速比':<10} {'说明':<20}")
    print("-" * 80)
    
    for r in results:
        extra = r.get('extra_info', '')
        print(f"{r['name']:<25} "
              f"{r['compile_time']:>8.2f}s  "
              f"{r['link_time']:>8.2f}s  "
              f"{r['total_time']:>8.2f}s  "
              f"{r['speedup']:>9.2f}x  "
              f"{extra:<20}")
    
    print("=" * 80)
    
    # 找出最快的方法
    fastest = min(results, key=lambda x: x['total_time'])
    print(f"\n🏆 最快方法: {fastest['name']}")
    print(f"   总耗时: {fastest['total_time']:.2f}s")
    print(f"   加速比: {fastest['speedup']:.2f}x")
    
    # 给出建议
    print("\n💡 使用建议:")
    print("\n  1️⃣  单文件开发：")
    print("     - 推荐: 增量编译 + ccache")
    print("     - 首次编译慢，后续编译极快")
    
    print("\n  2️⃣  多文件项目：")
    print("     - 推荐: 并行编译 + 增量编译")
    print("     - 充分利用多核 CPU")
    
    print("\n  3️⃣  频繁编译场景：")
    print("     - 推荐: PCH + ccache + 增量编译")
    print("     - 头文件预编译 + 结果缓存")
    
    print("\n  4️⃣  CI/CD 流水线：")
    print("     - 推荐: ccache (持久化缓存)")
    print("     - 跨构建复用编译结果")
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    run_comprehensive_benchmark()
