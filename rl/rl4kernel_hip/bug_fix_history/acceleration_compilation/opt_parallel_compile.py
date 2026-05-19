#!/usr/bin/env python3
"""
☆ 并行编译优化版本（独立 .so 模式）
基于 opt_compile_hip_v1.py，增加多文件并行编译支持

使用方法:
  python opt_parallel_compile.py              # 编译所有 HIP 文件
  python opt_parallel_compile.py -j 4         # 使用 4 个并行进程
  python opt_parallel_compile.py --benchmark  # 运行性能测试

优化原理:
  1. 利用多核CPU同时编译多个 .hip 文件
  2. 每个文件生成独立的 .so 模块（避免 PYBIND11_MODULE 符号冲突）
  3. 编译和链接过程完全并行化
  
性能提升:
  多文件项目可获得 1.5-3x 加速（取决于CPU核心数和文件数量）

注意事项:
  - 每个 .hip 文件必须包含独立的 PYBIND11_MODULE 定义
  - 生成的 .so 文件名从源文件名自动提取
"""

import os
import subprocess
import sys
import torch
import sysconfig
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


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


def compile_and_link_single_hip(args: Tuple[str, List[str], List[str], str, int, int]) -> Tuple[bool, str, str, float, float]:
    """
    编译并链接单个 HIP 文件成独立的 .so（用于并行编译）
    
    Args:
        args: (hip_file, include_paths, lib_paths, python_include, file_index, total_files)
    
    Returns:
        (success, hip_file, output_so, compile_time, link_time)
    """
    hip_file, include_paths, lib_paths, python_include, idx, total = args
    
    # 从文件名提取模块名
    base_name = os.path.splitext(os.path.basename(hip_file))[0]
    # 将 .o 文件输出到当前目录，避免权限问题
    obj_file = f"{base_name}.o"
    
    # 从文件名中提取合适的模块名（去除 hip_ref_hip_ 前缀和数字）
    if "PITF_Loss" in base_name:
        module_name = "PITF_Loss"
    elif "EDMLoss" in base_name:
        module_name = "EDMLoss"
    else:
        # 默认使用基础文件名
        module_name = base_name.replace("hip_ref_hip_", "").split("_", 1)[-1] if "hip_ref_hip_" in base_name else base_name
    
    output_so = f"{module_name}.so"
    
    print(f"[{idx}/{total}] 🔨 编译: {hip_file} -> {output_so}")
    
    # 使用干净的环境
    env = os.environ.copy()
    for var in ['HIPCC_COMPILE_FLAGS_APPEND', 'HIPCC_LINK_FLAGS_APPEND', 
                'AMDGPU_TARGETS', 'HIP_TARGETS']:
        env.pop(var, None)
    
    # 1️⃣ 编译 .hip -> .o
    compile_cmd = [
        "hipcc", "-O2", "-std=c++17", "-fPIC", "-c",
        hip_file, "-o", obj_file,
        "--offload-arch=gfx942",
        "-I", python_include,
    ]
    
    for inc in include_paths:
        compile_cmd.extend(["-I", inc])
    
    compile_start = time.time()
    result = subprocess.run(
        compile_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    compile_time = time.time() - compile_start
    
    if result.returncode != 0:
        print(f"[{idx}/{total}] ❌ 编译失败: {hip_file}")
        print(result.stderr)
        return False, hip_file, output_so, compile_time, 0.0
    
    print(f"[{idx}/{total}] ✅ 编译完成: {obj_file} ({compile_time:.2f}s)")
    
    # 2️⃣ 链接 .o -> .so
    link_cmd = [
        "hipcc", "-shared", obj_file, "-o", output_so,
        "-L", "/opt/rocm/lib", "-lhipblas", "-lrocblas",
    ]
    
    for lib in lib_paths:
        link_cmd.extend(["-L", lib])
    
    link_cmd.extend([
        "-ltorch",
        "-ltorch_cpu",
        "-ltorch_python",
        "-lc10",
        f"-DTORCH_EXTENSION_NAME={module_name}",
    ])
    
    link_start = time.time()
    result = subprocess.run(
        link_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    link_time = time.time() - link_start
    
    # 清理临时 .o 文件
    if os.path.exists(obj_file):
        os.remove(obj_file)
    
    if result.returncode == 0:
        file_size = os.path.getsize(output_so) / 1024  # KB
        print(f"[{idx}/{total}] ✅ 链接成功: {output_so} ({file_size:.1f} KB, {link_time:.2f}s)")
        return True, hip_file, output_so, compile_time, link_time
    else:
        print(f"[{idx}/{total}] ❌ 链接失败: {output_so}")
        print(result.stderr)
        return False, hip_file, output_so, compile_time, link_time


def build_hip_extension_parallel(hip_files: List[str] = None, max_workers: int = None):
    """
    并行编译 HIP 扩展（每个文件生成独立的 .so）
    
    Args:
        hip_files: HIP 源文件列表，如果为 None 则自动搜索
        max_workers: 最大并行进程数，默认为 CPU 核心数
    """
    
    include_paths, lib_paths = get_torch_paths()
    python_include = get_python_include()
    
    # 自动搜索 HIP 文件（如果未指定）
    if hip_files is None:
        hip_dir = Path("hip_code_ex")
        if hip_dir.exists():
            hip_files = [str(f) for f in hip_dir.glob("*.hip")]
        else:
            hip_files = ["hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"]
    
    if not hip_files:
        print("❌ 未找到 HIP 源文件！")
        sys.exit(1)
    
    # 确定并行进程数
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(hip_files))
    
    print("=" * 70)
    print("🚀 并行编译优化模式（独立 .so 文件）")
    print("=" * 70)
    print(f"📁 源文件数量: {len(hip_files)}")
    print(f"🔄 并行进程数: {max_workers}")
    print(f"💻 CPU 核心数: {multiprocessing.cpu_count()}")
    print("=" * 70)
    
    # 准备编译任务
    compile_tasks = [
        (hip_file, include_paths, lib_paths, python_include, idx + 1, len(hip_files))
        for idx, hip_file in enumerate(hip_files)
    ]
    
    # 并行编译和链接
    output_files = []
    compile_times = []
    link_times = []
    
    total_start = time.time()
    
    if len(hip_files) == 1:
        # 单文件：直接编译
        print("\n⚠️  只有一个文件，使用单线程编译")
        success, _, output_so, c_time, l_time = compile_and_link_single_hip(compile_tasks[0])
        if success:
            output_files.append(output_so)
            compile_times.append(c_time)
            link_times.append(l_time)
    else:
        # 多文件：并行编译
        print(f"\n🔄 开始并行编译和链接（{max_workers} 个进程）...\n")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(compile_and_link_single_hip, task): task[0]
                for task in compile_tasks
            }
            
            for future in as_completed(futures):
                success, hip_file, output_so, c_time, l_time = future.result()
                if success:
                    output_files.append(output_so)
                    compile_times.append(c_time)
                    link_times.append(l_time)
                else:
                    print(f"❌ 处理 {hip_file} 失败，继续其他文件...")
    
    total_elapsed = time.time() - total_start
    
    if not output_files:
        print("❌ 没有成功生成的 .so 文件！")
        sys.exit(1)
    
    # 统计时间
    total_compile_time = sum(compile_times)
    total_link_time = sum(link_times)
    total_sequential_time = total_compile_time + total_link_time
    speedup = total_sequential_time / total_elapsed if total_elapsed > 0 else 1.0
    
    print("\n" + "=" * 70)
    print("📊 编译统计")
    print("=" * 70)
    print(f"成功编译文件数: {len(output_files)}/{len(hip_files)}")
    print(f"生成的共享库:")
    for so_file in output_files:
        file_size = os.path.getsize(so_file) / 1024  # KB
        print(f"  ✅ {so_file} ({file_size:.1f} KB)")
    print()
    print(f"总编译时间: {total_compile_time:.2f}s")
    print(f"总链接时间: {total_link_time:.2f}s")
    print(f"实际并行时间: {total_elapsed:.2f}s")
    print(f"串行总时间: {total_sequential_time:.2f}s")
    print(f"并行加速比: {speedup:.2f}x")
    print(f"时间节省: {total_sequential_time - total_elapsed:.2f}s")
    print("=" * 70)
    
    return total_compile_time, total_link_time, speedup


def benchmark_compile_only():
    """测试并行编译性能（包含编译和链接，每个文件独立 .so）"""
    
    print("\n" + "=" * 70)
    print("🧪 并行编译性能测试（独立 .so 模式）")
    print("=" * 70)
    
    # 获取所有 HIP 文件
    hip_dir = Path("hip_code_ex")
    if hip_dir.exists():
        hip_files = [str(f) for f in hip_dir.glob("*.hip")]
    else:
        hip_files = ["hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"]
    
    if len(hip_files) < 2:
        print("⚠️  只有一个 HIP 文件，并行优化效果有限")
        print("提示: 可以复制文件进行测试")
    
    print(f"📁 测试文件: {len(hip_files)} 个")
    for f in hip_files:
        print(f"   - {f}")
    print("=" * 70)
    
    include_paths, lib_paths = get_torch_paths()
    python_include = get_python_include()
    
    results = []
    
    # 测试不同的并行度
    test_configs = [
        (1, "串行编译（基准）"),
        (2, "2 个进程"),
        (min(multiprocessing.cpu_count(), len(hip_files)), f"{min(multiprocessing.cpu_count(), len(hip_files))} 个进程"),
    ]
    
    for workers, desc in test_configs:
        if workers > len(hip_files) and workers > 1:
            continue
            
        print(f"\n{'─' * 70}")
        print(f"测试配置: {desc}")
        print(f"{'─' * 70}")
        
        # 准备编译任务
        compile_tasks = [
            (hip_file, include_paths, lib_paths, python_include, idx + 1, len(hip_files))
            for idx, hip_file in enumerate(hip_files)
        ]
        
        # 清理旧的 .o 和 .so 文件
        for hip_file in hip_files:
            obj_file = hip_file.replace('.hip', '.o')
            if os.path.exists(obj_file):
                os.remove(obj_file)
        
        # 并行编译和链接
        output_files = []
        compile_times = []
        link_times = []
        
        total_start = time.time()
        
        if workers == 1:
            # 串行编译
            for task in compile_tasks:
                success, _, output_so, c_time, l_time = compile_and_link_single_hip(task)
                if success:
                    output_files.append(output_so)
                    compile_times.append(c_time)
                    link_times.append(l_time)
        else:
            # 并行编译
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(compile_and_link_single_hip, task): task[0]
                    for task in compile_tasks
                }
                
                for future in as_completed(futures):
                    success, hip_file, output_so, c_time, l_time = future.result()
                    if success:
                        output_files.append(output_so)
                        compile_times.append(c_time)
                        link_times.append(l_time)
        
        total_elapsed = time.time() - total_start
        
        # 统计时间
        total_compile_time = sum(compile_times)
        total_link_time = sum(link_times)
        total_sequential_time = total_compile_time + total_link_time
        speedup = total_sequential_time / total_elapsed if total_elapsed > 0 else 1.0
        
        print(f"\n📊 编译完成: {len(output_files)}/{len(hip_files)} 个文件")
        print(f"   实际时间: {total_elapsed:.2f}s")
        print(f"   串行总时间: {total_sequential_time:.2f}s")
        print(f"   并行加速比: {speedup:.2f}x")
        
        results.append({
            'workers': workers,
            'desc': desc,
            'total_time': total_elapsed,
            'speedup': speedup
        })
    
    # 打印对比结果
    print("\n" + "=" * 70)
    print("📊 性能对比总结")
    print("=" * 70)
    print(f"{'配置':<30} {'总时间':<15} {'相对加速比':<15}")
    print("─" * 70)
    
    baseline_time = results[0]['total_time']
    
    for r in results:
        overall_speedup = baseline_time / r['total_time']
        print(f"{r['desc']:<30} "
              f"{r['total_time']:>12.2f}s  "
              f"{overall_speedup:>12.2f}x")
    
    print("=" * 70)
    
    # 最佳配置
    best = min(results, key=lambda x: x['total_time'])
    print(f"\n🏆 最佳配置: {best['desc']}")
    print(f"   总时间: {best['total_time']:.2f}s")
    print(f"   相对基准加速比: {baseline_time / best['total_time']:.2f}x")
    print(f"   时间节省: {baseline_time - best['total_time']:.2f}s ({(1 - best['total_time']/baseline_time)*100:.1f}%)")


def benchmark_parallel_vs_sequential():
    """对比并行编译和串行编译的性能（别名，调用 benchmark_compile_only）"""
    benchmark_compile_only()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='并行编译 HIP 扩展')
    parser.add_argument('-j', '--jobs', type=int, default=None,
                       help='并行进程数（默认: CPU核心数）')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能对比测试（完整编译+链接）')
    parser.add_argument('--benchmark-compile', action='store_true',
                       help='运行编译阶段性能对比测试（不链接）')
    
    args = parser.parse_args()
    
    total_start = time.time()
    
    if args.benchmark_compile:
        benchmark_compile_only()
    elif args.benchmark:
        benchmark_parallel_vs_sequential()
    else:
        compile_time, link_time, speedup = build_hip_extension_parallel(
            max_workers=args.jobs
        )
        total_elapsed = time.time() - total_start
        
        print("\n" + "=" * 70)
        print("⏱️  总耗时统计")
        print("=" * 70)
        print(f"编译阶段: {compile_time:.2f}s")
        print(f"链接阶段: {link_time:.2f}s")
        print(f"总计时间: {total_elapsed:.2f}s")
        print(f"并行加速比: {speedup:.2f}x")
        print("=" * 70)

