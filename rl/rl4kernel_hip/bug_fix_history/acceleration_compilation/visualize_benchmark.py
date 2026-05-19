#!/usr/bin/env python3
"""
编译性能对比可视化工具
读取 benchmark_results.json 并生成对比图表
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import sys

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_benchmark_results(filename: str = "benchmark_results.json"):
    """加载测试结果"""
    if not Path(filename).exists():
        print(f"❌ 未找到结果文件: {filename}")
        print("请先运行: python benchmark_compilation.py")
        sys.exit(1)
    
    with open(filename, 'r') as f:
        return json.load(f)


def plot_compilation_time_comparison(results, output_file="compilation_time_comparison.png"):
    """绘制编译时间对比图"""
    names = [r['name'] for r in results]
    avg_times = [r['avg_time'] for r in results]
    min_times = [r['min_time'] for r in results]
    max_times = [r['max_time'] for r in results]
    
    # 计算误差条
    errors = [[avg - min_val for avg, min_val in zip(avg_times, min_times)],
              [max_val - avg for avg, max_val in zip(avg_times, max_times)]]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(names))
    bars = ax.barh(x, avg_times, xerr=errors, capsize=5, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(names))),
                   edgecolor='black', linewidth=1.2)
    
    # 添加数值标签
    for i, (bar, time) in enumerate(zip(bars, avg_times)):
        ax.text(time + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{time:.2f}s', 
                va='center', fontsize=10, fontweight='bold')
    
    # 设置标签
    ax.set_yticks(x)
    ax.set_yticklabels(names)
    ax.set_xlabel('Compilation Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('HIP Kernel Compilation Time Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {output_file}")
    plt.close()


def plot_speedup_comparison(results, output_file="speedup_comparison.png"):
    """绘制加速比对比图"""
    baseline = results[0]['avg_time']
    names = [r['name'] for r in results]
    speedups = [baseline / r['avg_time'] for r in results]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(names))
    colors = ['red' if s < 1 else 'green' if s > 1.5 else 'orange' for s in speedups]
    bars = ax.barh(x, speedups, color=colors, edgecolor='black', linewidth=1.2)
    
    # 添加基准线
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (v1)')
    
    # 添加数值标签
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        label = f'{speedup:.2f}x'
        ax.text(speedup + 0.1, bar.get_y() + bar.get_height()/2, 
                label, va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(x)
    ax.set_yticklabels(names)
    ax.set_xlabel('Speedup (relative to v1)', fontsize=12, fontweight='bold')
    ax.set_title('Compilation Speedup Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {output_file}")
    plt.close()


def plot_file_size_comparison(results, output_file="file_size_comparison.png"):
    """绘制文件大小对比图"""
    names = [r['name'] for r in results]
    sizes = [r['file_size_mb'] for r in results]
    
    # 过滤掉大小为 0 的结果
    valid_data = [(n, s) for n, s in zip(names, sizes) if s > 0]
    if not valid_data:
        print("⚠️  没有有效的文件大小数据，跳过文件大小对比图")
        return
    
    names, sizes = zip(*valid_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(names))
    bars = ax.bar(x, sizes, color=plt.cm.plasma(np.linspace(0, 1, len(names))),
                  edgecolor='black', linewidth=1.2)
    
    # 添加数值标签
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                f'{size:.2f} MB',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('File Size (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Output File Size Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {output_file}")
    plt.close()


def plot_optimization_summary(results, output_file="optimization_summary.png"):
    """绘制优化策略综合对比"""
    baseline_time = results[0]['avg_time']
    
    # 选择关键策略对比
    key_strategies = {
        'v1': None,
        'v2': None,
        'v2_incr': None,
        'ccache_hit': None,
        'O3': None
    }
    
    for r in results:
        name = r['name'].lower()
        if 'v1' in name and 'v2' not in name and key_strategies['v1'] is None:
            key_strategies['v1'] = r
        elif 'v2' in name and '增量' not in name and 'ccache' not in name and '-o' not in name and key_strategies['v2'] is None:
            key_strategies['v2'] = r
        elif '增量' in name:
            key_strategies['v2_incr'] = r
        elif 'ccache' in name and '命中' in name:
            key_strategies['ccache_hit'] = r
        elif '-o3' in name:
            key_strategies['O3'] = r
    
    # 过滤出有效数据
    valid_strategies = {k: v for k, v in key_strategies.items() if v is not None}
    
    if len(valid_strategies) < 2:
        print("⚠️  数据不足，跳过综合对比图")
        return
    
    # 准备数据
    labels = []
    times = []
    speedups = []
    
    label_map = {
        'v1': 'v1 单步编译',
        'v2': 'v2 分离编译',
        'v2_incr': 'v2 增量编译',
        'ccache_hit': 'ccache 缓存命中',
        'O3': '-O3 优化'
    }
    
    for key in ['v1', 'v2', 'v2_incr', 'ccache_hit', 'O3']:
        if key in valid_strategies:
            labels.append(label_map[key])
            times.append(valid_strategies[key]['avg_time'])
            speedups.append(baseline_time / valid_strategies[key]['avg_time'])
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1: 编译时间
    x = np.arange(len(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    bars1 = ax1.bar(x, times, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                f'{time:.2f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right')
    ax1.set_ylabel('Compilation Time (s)', fontsize=12, fontweight='bold')
    ax1.set_title('Compilation Time', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 子图2: 加速比
    bars2 = ax2.bar(x, speedups, color=colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
    
    for bar, speedup in zip(bars2, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.2,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right')
    ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax2.set_title('Speedup Relative to v1', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend()
    
    plt.suptitle('HIP Compilation Optimization Summary', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {output_file}")
    plt.close()


def generate_report(results):
    """生成文本报告"""
    print("\n" + "="*80)
    print("📊 HIP COMPILATION OPTIMIZATION BENCHMARK REPORT")
    print("="*80 + "\n")
    
    baseline = results[0]
    baseline_time = baseline['avg_time']
    
    print(f"基准方案: {baseline['name']}")
    print(f"基准时间: {baseline_time:.2f}s\n")
    
    print("-" * 80)
    print(f"{'策略':<35} {'平均时间':<12} {'加速比':<10} {'文件大小':<12}")
    print("-" * 80)
    
    for r in results:
        speedup = baseline_time / r['avg_time']
        speedup_str = f"{speedup:.2f}x"
        if speedup > 1.5:
            speedup_str = f"🚀 {speedup_str}"
        elif speedup > 1.0:
            speedup_str = f"⚡ {speedup_str}"
        
        size_str = f"{r['file_size_mb']:.2f} MB" if r['file_size_mb'] > 0 else "N/A"
        
        print(f"{r['name']:<35} {r['avg_time']:>10.2f}s  {speedup_str:<10} {size_str:<12}")
    
    print("-" * 80)
    
    # 最佳策略
    fastest = min(results, key=lambda x: x['avg_time'])
    max_speedup = max(results, key=lambda x: baseline_time / x['avg_time'])
    
    print(f"\n🏆 最快编译: {fastest['name']} ({fastest['avg_time']:.2f}s)")
    print(f"🚀 最大加速比: {max_speedup['name']} ({baseline_time / max_speedup['avg_time']:.2f}x)")
    
    if any(r['file_size_mb'] > 0 for r in results):
        smallest = min([r for r in results if r['file_size_mb'] > 0], 
                      key=lambda x: x['file_size_mb'])
        print(f"📦 最小文件: {smallest['name']} ({smallest['file_size_mb']:.2f} MB)")
    
    print("\n" + "="*80 + "\n")


def main():
    # 加载数据
    results = load_benchmark_results()
    
    print("🎨 生成可视化图表...")
    
    # 生成报告
    generate_report(results)
    
    # 生成图表
    plot_compilation_time_comparison(results)
    plot_speedup_comparison(results)
    plot_file_size_comparison(results)
    plot_optimization_summary(results)
    
    print("\n✅ 所有图表已生成!")
    print("\n生成的文件:")
    print("  - compilation_time_comparison.png")
    print("  - speedup_comparison.png")
    print("  - file_size_comparison.png")
    print("  - optimization_summary.png")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        if 'matplotlib' in str(e):
            print("❌ 缺少 matplotlib 库")
            print("安装: pip install matplotlib")
            sys.exit(1)
        else:
            raise

