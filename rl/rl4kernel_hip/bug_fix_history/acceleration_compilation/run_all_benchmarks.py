#!/usr/bin/env python3
"""
统一性能对比测试脚本
一键对比所有编译优化策略的效果

使用方法:
  python run_all_benchmarks.py                # 完整测试
  python run_all_benchmarks.py --quick        # 快速测试（减少重复次数）
  python run_all_benchmarks.py --single-file  # 仅测试单文件场景

测试内容:
  1. v1 单步编译（基准）
  2. v2 分离编译
  3. 增量编译（首次 vs 再次）
  4. 并行编译（多核）
  5. 预编译头文件（PCH）
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List
import shutil


class BenchmarkRunner:
    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.results = []
        self.repeat_count = 1 if quick_mode else 2
        
    def clean_all(self):
        """清理所有编译产物和缓存"""
        print("🧹 清理所有编译产物和缓存...")
        
        # 清理 .o 和 .so 文件
        for pattern in ["*.o", "*.so"]:
            for f in Path(".").glob(pattern):
                f.unlink()
                
        # 清理缓存目录
        for cache_dir in [".hip_incremental_cache", ".hip_pch_cache", ".hip_build_cache"]:
            if Path(cache_dir).exists():
                shutil.rmtree(cache_dir)
        
        print("✅ 清理完成")
    
    def run_command(self, cmd: List[str], description: str) -> Dict:
        """运行命令并测量时间"""
        print(f"\n{'='*70}")
        print(f"🧪 {description}")
        print(f"{'='*70}")
        print(f"命令: {' '.join(cmd)}")
        
        times = []
        
        for i in range(self.repeat_count):
            if i > 0:
                print(f"\n第 {i+1}/{self.repeat_count} 轮...")
                
            start = time.time()
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            elapsed = time.time() - start
            
            if result.returncode != 0:
                print(f"❌ 执行失败！")
                print(result.stderr)
                return None
            
            times.append(elapsed)
            print(f"⏱️  耗时: {elapsed:.2f}s")
            
            # 打印输出的关键信息
            if "compile time:" in result.stdout:
                for line in result.stdout.split('\n'):
                    if "compile time:" in line or "总耗时" in line or "✅" in line:
                        print(line)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        
        return {
            'description': description,
            'avg_time': avg_time,
            'min_time': min_time,
            'times': times
        }
    
    def test_v1_baseline(self):
        """测试 v1 单步编译（基准）"""
        self.clean_all()
        
        result = self.run_command(
            ["python", "opt_compile_hip_v1.py"],
            "v1: 单步编译（基准）"
        )
        
        if result:
            self.results.append(result)
    
    def test_v2_separate(self):
        """测试 v2 分离编译"""
        self.clean_all()
        
        result = self.run_command(
            ["python", "opt_compile_hip_v2.py"],
            "v2: 分离编译"
        )
        
        if result:
            self.results.append(result)
    
    def test_incremental_first(self):
        """测试增量编译（首次）"""
        self.clean_all()
        
        result = self.run_command(
            ["python", "opt_incremental_compile.py"],
            "增量编译: 首次编译"
        )
        
        if result:
            self.results.append(result)
    
    def test_incremental_cached(self):
        """测试增量编译（缓存）"""
        # 不清理，使用上次的缓存
        
        result = self.run_command(
            ["python", "opt_incremental_compile.py"],
            "增量编译: 使用缓存（未修改）"
        )
        
        if result:
            self.results.append(result)
    
    def test_parallel(self):
        """测试并行编译"""
        self.clean_all()
        
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        
        result = self.run_command(
            ["python", "opt_parallel_compile.py", "-j", str(min(num_cores, 8))],
            f"并行编译: {min(num_cores, 8)} 个进程"
        )
        
        if result:
            self.results.append(result)
    
    def test_pch_first(self):
        """测试 PCH（首次，需要生成 PCH）"""
        self.clean_all()
        
        result = self.run_command(
            ["python", "opt_pch_compile.py", "--rebuild-pch"],
            "PCH: 首次编译（含 PCH 生成）"
        )
        
        if result:
            self.results.append(result)
    
    def test_pch_cached(self):
        """测试 PCH（PCH 已缓存）"""
        # 清理编译产物但保留 PCH
        for pattern in ["*.o", "*.so"]:
            for f in Path(".").glob(pattern):
                f.unlink()
        
        result = self.run_command(
            ["python", "opt_pch_compile.py"],
            "PCH: 使用缓存的 PCH"
        )
        
        if result:
            self.results.append(result)
    
    def test_no_pch(self):
        """测试不使用 PCH"""
        for pattern in ["*.o", "*.so"]:
            for f in Path(".").glob(pattern):
                f.unlink()
        
        result = self.run_command(
            ["python", "opt_pch_compile.py", "--no-pch"],
            "PCH: 不使用 PCH（对比）"
        )
        
        if result:
            self.results.append(result)
    
    def print_summary(self):
        """打印汇总结果"""
        if not self.results:
            print("❌ 没有测试结果")
            return
        
        print("\n\n" + "="*80)
        print("📊 所有优化策略性能对比总结")
        print("="*80)
        
        # 表头
        print(f"{'编译策略':<40} {'平均时间':<12} {'最快时间':<12} {'相对加速':<10}")
        print("-"*80)
        
        baseline = self.results[0]['avg_time']
        
        for r in self.results:
            speedup = baseline / r['avg_time']
            
            # 颜色标记（使用 emoji）
            if speedup >= 2.0:
                emoji = "🚀🚀"
            elif speedup >= 1.5:
                emoji = "🚀"
            elif speedup >= 1.2:
                emoji = "⚡"
            elif speedup >= 1.0:
                emoji = "✅"
            else:
                emoji = "⚠️"
            
            print(f"{r['description']:<40} "
                  f"{r['avg_time']:>10.2f}s  "
                  f"{r['min_time']:>10.2f}s  "
                  f"{speedup:>9.2f}x {emoji}")
        
        print("="*80)
        
        # 最佳策略
        fastest = min(self.results, key=lambda x: x['avg_time'])
        max_speedup_result = max(self.results, key=lambda x: baseline / x['avg_time'])
        
        print(f"\n🏆 最快策略: {fastest['description']}")
        print(f"   时间: {fastest['avg_time']:.2f}s")
        print(f"   加速: {baseline / fastest['avg_time']:.2f}x")
        
        if max_speedup_result != fastest:
            print(f"\n🚀 最大加速比: {max_speedup_result['description']}")
            print(f"   加速: {baseline / max_speedup_result['avg_time']:.2f}x")
        
        # 保存结果
        self.save_results()
    
    def save_results(self, filename: str = "all_benchmarks_results.json"):
        """保存结果到 JSON 文件"""
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'quick_mode': self.quick_mode,
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 结果已保存到: {filename}")


def check_dependencies():
    """检查必需的脚本文件"""
    required_files = [
        "opt_compile_hip_v1.py",
        "opt_compile_hip_v2.py",
        "opt_incremental_compile.py",
        "opt_parallel_compile.py",
        "opt_pch_compile.py",
    ]
    
    missing = []
    for f in required_files:
        if not Path(f).exists():
            missing.append(f)
    
    if missing:
        print("❌ 缺少以下脚本文件:")
        for f in missing:
            print(f"   - {f}")
        return False
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='运行所有编译优化策略的性能对比测试')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（减少重复次数）')
    parser.add_argument('--single-file', action='store_true',
                       help='仅测试单文件场景')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查源文件
    hip_file = "hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"
    if not Path(hip_file).exists():
        print(f"❌ 源文件不存在: {hip_file}")
        print("请确保源文件路径正确")
        sys.exit(1)
    
    print("="*80)
    print("🚀 HIP 编译优化策略 - 完整性能对比测试")
    print("="*80)
    print(f"快速模式: {'是' if args.quick else '否'}")
    print(f"源文件: {hip_file}")
    print("="*80)
    
    runner = BenchmarkRunner(quick_mode=args.quick)
    
    total_start = time.time()
    
    # 运行所有测试
    print("\n【阶段 1】基准测试")
    runner.test_v1_baseline()
    
    print("\n【阶段 2】分离编译")
    runner.test_v2_separate()
    
    print("\n【阶段 3】增量编译测试")
    runner.test_incremental_first()
    runner.test_incremental_cached()
    
    if not args.single_file:
        print("\n【阶段 4】并行编译测试")
        runner.test_parallel()
    
    print("\n【阶段 5】预编译头文件测试")
    runner.test_pch_first()
    runner.test_pch_cached()
    runner.test_no_pch()
    
    total_elapsed = time.time() - total_start
    
    # 打印汇总
    runner.print_summary()
    
    print(f"\n⏱️  测试总耗时: {total_elapsed:.2f}s")
    
    print("\n" + "="*80)
    print("✅ 所有测试完成！")
    print("="*80)
    
    print("\n💡 使用建议:")
    print("  1. 日常开发: 使用增量编译（缓存未修改文件）")
    print("  2. 多文件项目: 使用并行编译（充分利用多核）")
    print("  3. 频繁编译: 使用 PCH（减少头文件解析时间）")
    print("  4. CI/CD: 组合使用 ccache + 并行编译")
    print("  5. 生产环境: 使用 -O3 优化编译（提升运行性能）")


if __name__ == "__main__":
    main()

