#!/usr/bin/env python3
"""
☆ 增量编译与依赖追踪优化版本
基于 opt_compile_hip_v1.py，增加智能增量编译支持

使用方法:
  python opt_incremental_compile.py           # 正常编译
  python opt_incremental_compile.py --clean   # 清理缓存
  python opt_incremental_compile.py --force   # 强制重新编译
  python opt_incremental_compile.py --demo    # 演示增量编译效果

优化原理:
  1. 基于文件哈希值判断源文件是否修改
  2. 基于时间戳判断依赖关系
  3. 只重新编译修改过的文件
  4. 缓存依赖关系到 JSON 文件
  
性能提升:
  未修改文件: 跳过编译，接近 0s
  修改单个文件: 只编译该文件，提升 80-95%
"""

import os
import subprocess
import sys
import torch
import sysconfig
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# 缓存目录
CACHE_DIR = Path(".hip_incremental_cache")
DEPENDENCY_FILE = CACHE_DIR / "dependencies.json"


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


def compute_file_hash(filepath: str) -> str:
    """计算文件的 MD5 哈希值"""
    if not os.path.exists(filepath):
        return ""
    
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_dependencies() -> Dict:
    """加载依赖关系缓存"""
    if DEPENDENCY_FILE.exists():
        try:
            with open(DEPENDENCY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  加载依赖缓存失败: {e}")
            return {}
    return {}


def save_dependencies(dependencies: Dict):
    """保存依赖关系缓存"""
    CACHE_DIR.mkdir(exist_ok=True)
    with open(DEPENDENCY_FILE, 'w') as f:
        json.dump(dependencies, f, indent=2)


def needs_recompile(hip_file: str, obj_file: str, dependencies: Dict, 
                   force: bool = False) -> Tuple[bool, str]:
    """
    判断是否需要重新编译
    
    Returns:
        (need_compile, reason)
    """
    if force:
        return True, "强制重新编译"
    
    # 检查目标文件是否存在
    if not os.path.exists(obj_file):
        return True, "目标文件不存在"
    
    # 检查源文件是否存在
    if not os.path.exists(hip_file):
        return False, "源文件不存在"
    
    # 计算当前哈希值
    current_hash = compute_file_hash(hip_file)
    
    # 检查缓存中的哈希值
    cached_info = dependencies.get(hip_file, {})
    cached_hash = cached_info.get('hash', '')
    
    if current_hash != cached_hash:
        return True, "文件内容已修改"
    
    # 检查时间戳
    src_mtime = os.path.getmtime(hip_file)
    obj_mtime = os.path.getmtime(obj_file)
    
    if src_mtime > obj_mtime:
        return True, "源文件时间戳更新"
    
    return False, "无需重新编译"


def update_dependency(hip_file: str, obj_file: str, dependencies: Dict, 
                     compile_time: float):
    """更新依赖信息"""
    dependencies[hip_file] = {
        'hash': compute_file_hash(hip_file),
        'mtime': os.path.getmtime(hip_file),
        'obj_file': obj_file,
        'obj_mtime': os.path.getmtime(obj_file) if os.path.exists(obj_file) else 0,
        'last_compile_time': compile_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    save_dependencies(dependencies)


def compile_hip_file(hip_file: str, obj_file: str, include_paths: List[str], 
                    python_include: str) -> Tuple[bool, float]:
    """
    编译单个 HIP 文件
    
    Returns:
        (success, elapsed_time)
    """
    compile_cmd = [
        "hipcc", "-O2", "-std=c++17", "-fPIC", "-c",
        hip_file, "-o", obj_file,
        "--offload-arch=gfx942",
        "-I", python_include,
    ]
    
    # 添加 PyTorch include
    for inc in include_paths:
        compile_cmd.extend(["-I", inc])
    
    print(f"\n🔨 编译: {hip_file}")
    print(f"命令: {' '.join(compile_cmd)}")
    
    # 创建干净的环境（移除可能导致问题的环境变量）
    env = os.environ.copy()
    # 移除可能引起 GPU 架构冲突的环境变量
    problematic_vars = [
        'HIPCC_COMPILE_FLAGS_APPEND',
        'HIPCC_LINK_FLAGS_APPEND',
        'AMDGPU_TARGETS',
        'HIP_TARGETS',
    ]
    for var in problematic_vars:
        if var in env:
            print(f"⚠️  移除环境变量: {var}={env[var]}")
            env.pop(var)
    
    start = time.time()
    result = subprocess.run(
        compile_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env  # 使用清理后的环境
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
    
    # 使用干净的环境（与编译时相同）
    env = os.environ.copy()
    problematic_vars = [
        'HIPCC_COMPILE_FLAGS_APPEND',
        'HIPCC_LINK_FLAGS_APPEND',
        'AMDGPU_TARGETS',
        'HIP_TARGETS',
    ]
    for var in problematic_vars:
        env.pop(var, None)
    
    start = time.time()
    result = subprocess.run(
        link_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env  # 使用清理后的环境
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


def build_hip_extension_incremental(hip_files: List[str] = None, force: bool = False):
    """
    增量编译 HIP 扩展
    
    Args:
        hip_files: HIP 源文件列表，如果为 None 则使用默认文件
        force: 是否强制重新编译所有文件
    """
    
    include_paths, lib_paths = get_torch_paths()
    python_include = get_python_include()
    
    # 默认源文件
    if hip_files is None:
        hip_files = ["hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"]
    
    # 加载依赖缓存
    dependencies = load_dependencies()
    
    print("=" * 70)
    print("🔄 增量编译优化模式")
    print("=" * 70)
    print(f"📁 源文件数量: {len(hip_files)}")
    print(f"💾 依赖缓存: {DEPENDENCY_FILE}")
    print(f"🔧 强制重编译: {'是' if force else '否'}")
    print("=" * 70)
    
    # 检查每个文件是否需要编译
    obj_files = []
    compile_count = 0
    skip_count = 0
    total_compile_time = 0.0
    
    for hip_file in hip_files:
        obj_file = hip_file.replace('.hip', '.o')
        
        need_compile, reason = needs_recompile(hip_file, obj_file, dependencies, force)
        
        if need_compile:
            print(f"\n📝 {hip_file}: {reason}")
            success, elapsed = compile_hip_file(hip_file, obj_file, include_paths, python_include)
            
            if success:
                update_dependency(hip_file, obj_file, dependencies, elapsed)
                obj_files.append(obj_file)
                compile_count += 1
                total_compile_time += elapsed
            else:
                print("❌ 编译失败，终止！")
                sys.exit(1)
        else:
            print(f"\n⚡ {hip_file}: {reason} (跳过)")
            obj_files.append(obj_file)
            skip_count += 1
    
    print("\n" + "=" * 70)
    print("📊 编译统计")
    print("=" * 70)
    print(f"总文件数: {len(hip_files)}")
    print(f"重新编译: {compile_count}")
    print(f"跳过编译: {skip_count}")
    print(f"编译耗时: {total_compile_time:.2f}s")
    if compile_count > 0:
        print(f"平均每文件: {total_compile_time / compile_count:.2f}s")
    print("=" * 70)
    
    # 链接阶段
    output_so = "EDMLoss_incremental.so"
    success, link_time = link_object_files(obj_files, output_so, lib_paths)
    
    if not success:
        sys.exit(1)
    
    return total_compile_time, link_time, compile_count, skip_count


def clean_cache():
    """清理编译缓存"""
    import shutil
    
    print("🧹 清理编译缓存...")
    
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"✅ 已删除缓存目录: {CACHE_DIR}")
    else:
        print(f"⚠️  缓存目录不存在: {CACHE_DIR}")
    
    # 清理 .o 文件
    obj_count = 0
    for obj_file in Path(".").rglob("*.o"):
        obj_file.unlink()
        obj_count += 1
    
    if obj_count > 0:
        print(f"✅ 已删除 {obj_count} 个 .o 文件")
    
    print("✅ 清理完成！")


def demo_incremental_compile():
    """演示增量编译效果"""
    
    print("\n" + "=" * 70)
    print("🎬 增量编译效果演示")
    print("=" * 70)
    
    hip_file = "hip_code_ex/hip_ref_hip_8825_EDMLoss.hip"
    
    if not os.path.exists(hip_file):
        print(f"❌ 源文件不存在: {hip_file}")
        return
    
    print("\n【步骤 1】首次编译（完整编译）")
    print("─" * 70)
    clean_cache()
    start = time.time()
    build_hip_extension_incremental([hip_file], force=False)
    first_time = time.time() - start
    print(f"\n⏱️  首次编译总耗时: {first_time:.2f}s")
    
    print("\n\n【步骤 2】再次编译（无修改，应该全部跳过）")
    print("─" * 70)
    input("按 Enter 继续...")
    start = time.time()
    build_hip_extension_incremental([hip_file], force=False)
    second_time = time.time() - start
    speedup = first_time / second_time if second_time > 0 else float('inf')
    print(f"\n⏱️  增量编译总耗时: {second_time:.2f}s")
    print(f"🚀 加速比: {speedup:.2f}x")
    
    print("\n\n【步骤 3】模拟文件修改后编译")
    print("─" * 70)
    print("提示: 修改源文件后，只会重新编译修改的文件")
    print(f"你可以手动修改 {hip_file}，然后重新运行此脚本")
    
    print("\n\n【步骤 4】强制重新编译")
    print("─" * 70)
    input("按 Enter 继续...")
    start = time.time()
    build_hip_extension_incremental([hip_file], force=True)
    force_time = time.time() - start
    print(f"\n⏱️  强制编译总耗时: {force_time:.2f}s")
    
    print("\n" + "=" * 70)
    print("📊 演示总结")
    print("=" * 70)
    print(f"首次编译: {first_time:.2f}s")
    print(f"增量编译（无修改）: {second_time:.2f}s ({first_time / second_time:.2f}x 加速)")
    print(f"强制重编译: {force_time:.2f}s")
    print("=" * 70)
    
    print("\n💡 增量编译优势:")
    print("  • 未修改文件: 跳过编译，几乎瞬间完成")
    print("  • 修改少量文件: 只编译修改的文件，节省 80-95% 时间")
    print("  • 大型项目: 效果更显著")


def show_dependency_info():
    """显示依赖关系信息"""
    dependencies = load_dependencies()
    
    if not dependencies:
        print("⚠️  没有依赖信息（缓存为空）")
        return
    
    print("\n" + "=" * 70)
    print("📋 依赖关系缓存信息")
    print("=" * 70)
    
    for hip_file, info in dependencies.items():
        print(f"\n源文件: {hip_file}")
        print(f"  哈希值: {info.get('hash', 'N/A')[:16]}...")
        print(f"  目标文件: {info.get('obj_file', 'N/A')}")
        print(f"  编译时间: {info.get('last_compile_time', 0):.2f}s")
        print(f"  最后编译: {info.get('timestamp', 'N/A')}")
    
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='增量编译 HIP 扩展')
    parser.add_argument('--force', '-f', action='store_true',
                       help='强制重新编译所有文件')
    parser.add_argument('--clean', '-c', action='store_true',
                       help='清理编译缓存')
    parser.add_argument('--demo', '-d', action='store_true',
                       help='演示增量编译效果')
    parser.add_argument('--info', '-i', action='store_true',
                       help='显示依赖信息')
    
    args = parser.parse_args()
    
    if args.clean:
        clean_cache()
        sys.exit(0)
    
    if args.info:
        show_dependency_info()
        sys.exit(0)
    
    if args.demo:
        demo_incremental_compile()
        sys.exit(0)
    
    # 正常编译
    total_start = time.time()
    
    compile_time, link_time, compile_count, skip_count = build_hip_extension_incremental(
        force=args.force
    )
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("⏱️  总耗时统计")
    print("=" * 70)
    print(f"编译阶段: {compile_time:.2f}s ({compile_count} 个文件)")
    print(f"跳过文件: {skip_count} 个")
    print(f"链接阶段: {link_time:.2f}s")
    print(f"总计时间: {total_time:.2f}s")
    
    if skip_count > 0:
        print(f"\n💡 增量编译节省了 {skip_count} 个文件的编译时间！")
    
    print("=" * 70)

