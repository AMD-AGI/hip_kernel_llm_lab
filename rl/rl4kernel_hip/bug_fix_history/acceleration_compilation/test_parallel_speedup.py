#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
多文件并行编译提速效果测试
基于 hip_ref_hip_8825_EDMLoss.hip 和 hip_ref_hip_969_PITF_Loss.hip
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 导入现有模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opt_parallel_compile


def clean_obj_files():
    """清理 .o 文件"""
    for obj_file in Path("hip_code_ex").glob("*.o"):
        obj_file.unlink()
    print("✅ 已清理 .o 文件")


def test_serial_compile(hip_files):
    """测试串行编译"""
    print("\n" + "=" * 70)
    print("【测试 1】串行编译 - 逐个编译文件")
    print("=" * 70)
    
    include_paths, lib_paths = opt_parallel_compile.get_torch_paths()
    python_include = opt_parallel_compile.get_python_include()
    
    total_start = time.time()
    compile_times = []
    
    for idx, hip_file in enumerate(hip_files, 1):
        obj_file = hip_file.replace('.hip', '.o')
        
        compile_cmd = [
            "hipcc", "-O2", "-std=c++17", "-fPIC", "-c",
            hip_file, "-o", obj_file,
            "--offload-arch=gfx942",
            "-I", python_include,
        ]
        
        for inc in include_paths:
            compile_cmd.extend(["-I", inc])
        
        # 使用干净的环境
        env = os.environ.copy()
        for var in ['HIPCC_COMPILE_FLAGS_APPEND', 'HIPCC_LINK_FLAGS_APPEND',
                    'AMDGPU_TARGETS', 'HIP_TARGETS']:
            env.pop(var, None)
        
        print(f"\n[{idx}/{len(hip_files)}] 🔨 编译: {hip_file}")
        
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
            compile_times.append(elapsed)
            print(f"[{idx}/{len(hip_files)}] ✅ 完成: {hip_file} ({elapsed:.2f}s)")
        else:
            print(f"[{idx}/{len(hip_files)}] ❌ 失败: {hip_file}")
            print(result.stderr)
            return None
    
    total_time = time.time() - total_start
    
    print(f"\n" + "=" * 70)
    print(f"📊 串行编译统计")
    print(f"=" * 70)
    print(f"文件数量: {len(hip_files)}")
    print(f"单个耗时: {', '.join([f'{t:.2f}s' for t in compile_times])}")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均每文件: {sum(compile_times) / len(compile_times):.2f}s")
    print(f"=" * 70)
    
    return {
        "method": "串行编译",
        "files": len(hip_files),
        "total_time": total_time,
        "compile_times": compile_times,
        "sum_times": sum(compile_times)
    }


def test_parallel_compile(hip_files, workers):
    """测试并行编译"""
    print("\n" + "=" * 70)
    print(f"【测试 2】并行编译 - {workers} 个进程同时编译")
    print("=" * 70)
    
    include_paths, lib_paths = opt_parallel_compile.get_torch_paths()
    python_include = opt_parallel_compile.get_python_include()
    
    # 准备编译任务
    compile_tasks = [
        (hip_file, include_paths, python_include, idx + 1, len(hip_files))
        for idx, hip_file in enumerate(hip_files)
    ]
    
    total_start = time.time()
    compile_times = []
    
    print(f"\n🔄 开始并行编译（{workers} 个进程）...\n")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(opt_parallel_compile.compile_single_hip_file, task): task[0]
            for task in compile_tasks
        }
        
        for future in as_completed(futures):
            success, hip_file, obj_file, elapsed = future.result()
            if success:
                compile_times.append(elapsed)
            else:
                print(f"❌ 编译失败: {hip_file}")
                return None
    
    total_time = time.time() - total_start
    
    print(f"\n" + "=" * 70)
    print(f"📊 并行编译统计")
    print(f"=" * 70)
    print(f"文件数量: {len(hip_files)}")
    print(f"并行进程: {workers}")
    print(f"单个耗时: {', '.join([f'{t:.2f}s' for t in sorted(compile_times)])}")
    print(f"实际总耗时: {total_time:.2f}s")
    print(f"串行总时间: {sum(compile_times):.2f}s")
    print(f"并行加速比: {sum(compile_times) / total_time:.2f}x")
    print(f"=" * 70)
    
    return {
        "method": f"并行编译 ({workers} 进程)",
        "files": len(hip_files),
        "total_time": total_time,
        "compile_times": compile_times,
        "sum_times": sum(compile_times),
        "workers": workers
    }


def test_different_parallel_levels(hip_files):
    """测试不同并行度的效果"""
    print("\n" + "=" * 70)
    print("【测试 3】不同并行度对比")
    print("=" * 70)
    
    results = []
    
    # 测试 1 个进程（伪并行，相当于串行）
    clean_obj_files()
    result = test_parallel_compile(hip_files, workers=1)
    if result:
        results.append(result)
    
    # 测试 2 个进程（文件数相同）
    clean_obj_files()
    result = test_parallel_compile(hip_files, workers=2)
    if result:
        results.append(result)
    
    # 测试 4 个进程（超出文件数，但不会更快）
    clean_obj_files()
    result = test_parallel_compile(hip_files, workers=4)
    if result:
        results.append(result)
    
    return results


def run_comprehensive_test():
    """运行综合测试"""
    print("=" * 70)
    print("🚀 多文件并行编译提速效果测试")
    print("=" * 70)
    
    hip_files = [
        "hip_code_ex/hip_ref_hip_969_PITF_Loss.hip",
        "hip_code_ex/hip_ref_hip_8825_EDMLoss.hip",
    ]
    
    # 检查文件是否存在
    for hip_file in hip_files:
        if not os.path.exists(hip_file):
            print(f"❌ 文件不存在: {hip_file}")
            sys.exit(1)
    
    print(f"\n📁 测试文件:")
    for i, f in enumerate(hip_files, 1):
        size = os.path.getsize(f)
        print(f"  {i}. {f} ({size} bytes)")
    
    print(f"\n💻 系统信息:")
    print(f"  CPU 核心数: {os.cpu_count()}")
    print(f"  测试文件数: {len(hip_files)}")
    
    results = []
    
    # 测试 1: 串行编译
    clean_obj_files()
    serial_result = test_serial_compile(hip_files)
    if serial_result:
        results.append(serial_result)
    else:
        print("❌ 串行编译失败，终止测试")
        return
    
    # 等待一下
    time.sleep(2)
    
    # 测试 2: 并行编译 (2 进程)
    clean_obj_files()
    parallel_result = test_parallel_compile(hip_files, workers=2)
    if parallel_result:
        results.append(parallel_result)
    
    # 测试 3: 不同并行度（可选，注释掉以节省时间）
    # parallel_results = test_different_parallel_levels(hip_files)
    # results.extend(parallel_results)
    
    # 打印总结对比
    print("\n\n" + "=" * 70)
    print("📊 性能对比总结")
    print("=" * 70)
    print(f"\n{'方法':<20} {'总耗时':<12} {'串行总时':<12} {'加速比':<10} {'效率':<10}")
    print("-" * 70)
    
    baseline = results[0]["total_time"]
    
    for r in results:
        speedup = r["sum_times"] / r["total_time"]
        efficiency = speedup / r.get("workers", 1) * 100 if "workers" in r else 100
        
        print(f"{r['method']:<20} "
              f"{r['total_time']:>10.2f}s  "
              f"{r['sum_times']:>10.2f}s  "
              f"{speedup:>9.2f}x  "
              f"{efficiency:>8.1f}%")
    
    print("=" * 70)
    
    # 关键发现
    if len(results) >= 2:
        serial = results[0]
        parallel = results[1]
        speedup = serial["total_time"] / parallel["total_time"]
        time_saved = serial["total_time"] - parallel["total_time"]
        
        print(f"\n💡 关键发现:")
        print(f"   - 文件数: {len(hip_files)} 个")
        print(f"   - 串行编译: {serial['total_time']:.2f}s")
        print(f"   - 并行编译: {parallel['total_time']:.2f}s")
        print(f"   - 实际加速比: {speedup:.2f}x")
        print(f"   - 节省时间: {time_saved:.2f}s ({time_saved/serial['total_time']*100:.1f}%)")
        
        print(f"\n📈 理论 vs 实际:")
        print(f"   - 理论最大加速: {len(hip_files):.0f}x (完美并行)")
        print(f"   - 实际加速比: {speedup:.2f}x")
        print(f"   - 并行效率: {speedup/len(hip_files)*100:.1f}%")
        
        if speedup >= 1.8:
            print(f"\n🎉 并行编译效果显著！")
        elif speedup >= 1.5:
            print(f"\n✅ 并行编译有明显提升！")
        else:
            print(f"\n⚠️  并行效果受限（可能是 I/O 瓶颈或系统负载）")
    
    print(f"\n✅ 测试完成！")
    
    # 清理
    clean_obj_files()


if __name__ == "__main__":
    run_comprehensive_test()

