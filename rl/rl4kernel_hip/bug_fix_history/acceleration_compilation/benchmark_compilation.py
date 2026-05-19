#!/usr/bin/env python3
"""
HIP 编译优化策略性能对比测试
测试不同优化方案的编译时间、文件大小、运行性能
"""

import subprocess
import time
import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple


class CompilationBenchmark:
    def __init__(self, hip_file: str, work_dir: str = "."):
        self.hip_file = hip_file
        self.work_dir = Path(work_dir)
        self.results = []
        
    def clean_build_artifacts(self):
        """清理编译产物"""
        for pattern in ["*.o", "*.so", "build/*"]:
            subprocess.run(f"rm -rf {pattern}", shell=True, cwd=self.work_dir)
        
    def clear_ccache(self):
        """清理 ccache"""
        subprocess.run(["ccache", "-C"], capture_output=True)
        
    def get_file_size(self, filepath: str) -> float:
        """获取文件大小（MB）"""
        if os.path.exists(filepath):
            return os.path.getsize(filepath) / (1024 * 1024)
        return 0.0
    
    def run_compilation(self, 
                       name: str, 
                       compile_cmd: List[str],
                       link_cmd: List[str] = None,
                       clean_before: bool = True,
                       repeat: int = 3) -> Dict:
        """运行编译测试并记录结果"""
        
        if clean_before:
            self.clean_build_artifacts()
            
        print(f"\n{'='*60}")
        print(f"🧪 测试: {name}")
        print(f"{'='*60}")
        
        times = []
        
        for i in range(repeat):
            if i > 0 and clean_before:
                self.clean_build_artifacts()
                
            print(f"\n第 {i+1}/{repeat} 轮...")
            
            start = time.time()
            
            # 编译阶段
            print(f"命令: {' '.join(compile_cmd)}")
            result = subprocess.run(
                compile_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.work_dir
            )
            
            if result.returncode != 0:
                print(f"❌ 编译失败!")
                print(result.stderr)
                return None
            
            # 链接阶段（如果有）
            if link_cmd:
                print(f"链接: {' '.join(link_cmd)}")
                result = subprocess.run(
                    link_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.work_dir
                )
                
                if result.returncode != 0:
                    print(f"❌ 链接失败!")
                    print(result.stderr)
                    return None
            
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"⏱️  耗时: {elapsed:.2f}s")
        
        # 统计结果
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # 获取输出文件大小
        output_file = self._extract_output_file(compile_cmd if not link_cmd else link_cmd)
        file_size = self.get_file_size(output_file) if output_file else 0.0
        
        result = {
            "name": name,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "file_size_mb": file_size,
            "compile_cmd": " ".join(compile_cmd),
            "link_cmd": " ".join(link_cmd) if link_cmd else None
        }
        
        self.results.append(result)
        
        print(f"\n✅ 平均耗时: {avg_time:.2f}s")
        print(f"📦 文件大小: {file_size:.2f} MB")
        
        return result
    
    def _extract_output_file(self, cmd: List[str]) -> str:
        """从命令中提取输出文件名"""
        try:
            idx = cmd.index("-o")
            return cmd[idx + 1]
        except (ValueError, IndexError):
            return None
    
    def print_summary(self):
        """打印汇总对比表"""
        if not self.results:
            print("没有测试结果")
            return
        
        print(f"\n\n{'='*80}")
        print("📊 编译性能对比总结")
        print(f"{'='*80}\n")
        
        # 表头
        print(f"{'策略':<30} {'平均耗时':<12} {'最快':<12} {'文件大小':<12} {'相对v1':<10}")
        print("-" * 80)
        
        baseline = self.results[0]["avg_time"]
        
        for r in self.results:
            speedup = baseline / r["avg_time"]
            print(f"{r['name']:<30} "
                  f"{r['avg_time']:>10.2f}s  "
                  f"{r['min_time']:>10.2f}s  "
                  f"{r['file_size_mb']:>10.2f}MB  "
                  f"{speedup:>9.2f}x")
        
        print(f"{'='*80}\n")
        
        # 最佳策略
        fastest = min(self.results, key=lambda x: x["avg_time"])
        smallest = min(self.results, key=lambda x: x["file_size_mb"])
        
        print(f"🏆 最快编译: {fastest['name']} ({fastest['avg_time']:.2f}s)")
        print(f"📦 最小文件: {smallest['name']} ({smallest['file_size_mb']:.2f} MB)")
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """保存结果到文件"""
        output_path = self.work_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 结果已保存到: {output_path}")


def get_torch_and_python_paths():
    """获取 PyTorch 和 Python 路径"""
    import torch
    torch_dir = os.path.dirname(torch.__file__)
    torch_include = os.path.join(torch_dir, "include")
    torch_lib = os.path.join(torch_dir, "lib")
    
    python_include = subprocess.check_output(
        ["python3-config", "--includes"], 
        text=True
    ).strip().split()[0][2:]
    
    return torch_include, torch_lib, python_include


def main():
    # 配置
    hip_file = "hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"
    
    if not os.path.exists(hip_file):
        print(f"❌ 源文件不存在: {hip_file}")
        print("请调整 hip_file 变量指向实际的 .hip 文件")
        sys.exit(1)
    
    torch_include, torch_lib, python_include = get_torch_and_python_paths()
    
    # 检查 ccache
    has_ccache = subprocess.run(
        ["which", "ccache"], 
        capture_output=True
    ).returncode == 0
    
    if not has_ccache:
        print("⚠️  未检测到 ccache，部分测试将跳过")
        print("安装: sudo apt install ccache")
    
    # 创建测试实例
    benchmark = CompilationBenchmark(hip_file)
    
    print("🚀 开始编译性能对比测试")
    print(f"源文件: {hip_file}")
    print(f"测试次数: 每个策略3次")
    
    # 基础参数
    base_flags = ["-std=c++17", "-fPIC", "--offload-arch=gfx942",
                  "-I", python_include, "-I", torch_include]
    
    lib_flags = ["-L", torch_lib, "-L", "/opt/rocm/lib",
                 "-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10",
                 "-lhipblas", "-lrocblas"]
    
    # ========== 测试 1: v1 单步编译（基准） ==========
    benchmark.run_compilation(
        name="v1: 单步编译 -O2",
        compile_cmd=["hipcc", "-O2", "-shared", hip_file, "-o", "v1_O2.so"] + base_flags + lib_flags,
        repeat=3
    )
    
    # ========== 测试 2: v2 分离编译 ==========
    benchmark.run_compilation(
        name="v2: 分离编译 -O2",
        compile_cmd=["hipcc", "-O2", "-c", hip_file, "-o", "v2_O2.o"] + base_flags,
        link_cmd=["hipcc", "-shared", "v2_O2.o", "-o", "v2_O2.so"] + lib_flags,
        repeat=3
    )
    
    # ========== 测试 3: v2 + 增量编译（第二次不清理）==========
    # 首次编译
    benchmark.run_compilation(
        name="v2: 首次编译",
        compile_cmd=["hipcc", "-O2", "-c", hip_file, "-o", "v2_incr.o"] + base_flags,
        link_cmd=["hipcc", "-shared", "v2_incr.o", "-o", "v2_incr.so"] + lib_flags,
        clean_before=True,
        repeat=1
    )
    
    # 增量编译（不清理，模拟未修改）
    benchmark.run_compilation(
        name="v2: 增量编译（未修改）",
        compile_cmd=["hipcc", "-O2", "-c", hip_file, "-o", "v2_incr.o"] + base_flags,
        link_cmd=["hipcc", "-shared", "v2_incr.o", "-o", "v2_incr.so"] + lib_flags,
        clean_before=False,  # 关键：不清理
        repeat=1
    )
    
    # ========== 测试 4: 不同优化级别 ==========
    for opt_level in ["-O0", "-O3", "-Ofast"]:
        benchmark.run_compilation(
            name=f"v2: {opt_level}",
            compile_cmd=["hipcc", opt_level, "-c", hip_file, "-o", f"v2{opt_level}.o"] + base_flags,
            link_cmd=["hipcc", "-shared", f"v2{opt_level}.o", "-o", f"v2{opt_level}.so"] + lib_flags,
            repeat=2
        )
    
    # ========== 测试 5: ccache（如果可用）==========
    if has_ccache:
        # 清空缓存
        benchmark.clear_ccache()
        
        # 首次编译（缓存未命中）
        benchmark.run_compilation(
            name="ccache: 首次（缓存未命中）",
            compile_cmd=["ccache", "hipcc", "-O2", "-c", hip_file, "-o", "ccache.o"] + base_flags,
            link_cmd=["hipcc", "-shared", "ccache.o", "-o", "ccache.so"] + lib_flags,
            clean_before=True,
            repeat=1
        )
        
        # 第二次编译（缓存命中）
        benchmark.run_compilation(
            name="ccache: 缓存命中",
            compile_cmd=["ccache", "hipcc", "-O2", "-c", hip_file, "-o", "ccache.o"] + base_flags,
            link_cmd=["hipcc", "-shared", "ccache.o", "-o", "ccache.so"] + lib_flags,
            clean_before=True,  # 清理 .o 文件但保留 ccache
            repeat=1
        )
        
        # 显示 ccache 统计
        print("\n📊 ccache 统计:")
        subprocess.run(["ccache", "-s"])
    
    # ========== 测试 6: 高级优化 ==========
    advanced_flags = base_flags + [
        "-march=native",
        "-mtune=native", 
        "-ffast-math",
        "-funroll-loops"
    ]
    
    benchmark.run_compilation(
        name="v2: -O3 + 高级优化",
        compile_cmd=["hipcc", "-O3", "-c", hip_file, "-o", "v2_adv.o"] + advanced_flags,
        link_cmd=["hipcc", "-shared", "v2_adv.o", "-o", "v2_adv.so", 
                  "-Wl,--strip-all", "-Wl,--gc-sections"] + lib_flags,
        repeat=2
    )
    
    # 打印总结
    benchmark.print_summary()
    benchmark.save_results()
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()

