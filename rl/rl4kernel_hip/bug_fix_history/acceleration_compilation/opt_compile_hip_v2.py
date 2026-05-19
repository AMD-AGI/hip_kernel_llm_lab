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


def build_single_hip_extension(hip_file, module_name, include_paths, lib_paths, python_include):
    """编译单个 HIP 文件成独立的 .so"""
    
    # 从 hip_file 路径中提取基础文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(hip_file))[0]
    obj_file = f"{base_name}.o"
    output_so = f"{module_name}.so"

    print(f"\n{'='*60}")
    print(f"📦 编译模块: {module_name}")
    print(f"   源文件: {hip_file}")
    print(f"   目标文件: {output_so}")
    print(f"{'='*60}")

    # 1️⃣ 编译 .hip -> .o
    compile_cmd = [
        "hipcc", "-O2", "-std=c++17", "-fPIC", "-c", hip_file, "-o", obj_file,
        "--offload-arch=gfx942",
        "-I", python_include,
    ]
    for inc in include_paths:
        compile_cmd.extend(["-I", inc])

    print("\n🔧 编译对象文件...")
    print("命令:", " ".join(compile_cmd))

    # 使用干净的环境
    env = os.environ.copy()
    for var in ['HIPCC_COMPILE_FLAGS_APPEND', 'HIPCC_LINK_FLAGS_APPEND',
                'AMDGPU_TARGETS', 'HIP_TARGETS']:
        env.pop(var, None)

    result = subprocess.run(compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if result.returncode != 0:
        print(f"❌ 编译失败: {hip_file}")
        print(result.stderr)
        return False

    print(f"✅ 对象文件生成: {obj_file}")

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

    print("\n🔗 链接共享库...")
    print("命令:", " ".join(link_cmd))

    result = subprocess.run(link_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if result.returncode == 0:
        print(f"✅ 编译成功: {output_so}")
        # 清理临时对象文件
        if os.path.exists(obj_file):
            os.remove(obj_file)
            print(f"🧹 清理临时文件: {obj_file}")
        return True
    else:
        print(f"❌ 链接失败: {output_so}")
        print(result.stderr)
        return False


def build_hip_extension():
    """编译所有 HIP 扩展"""
    include_paths, lib_paths = get_torch_paths()
    python_include = get_python_include()

    # 定义要编译的 HIP 文件及其对应的模块名
    hip_extensions = [
        {
            "hip_file": "hip_code_ex/hip_ref_hip_969_PITF_Loss.hip",
            "module_name": "PITF_Loss"
        },
        {
            "hip_file": "hip_code_ex/hip_ref_hip_8825_EDMLoss.hip",
            "module_name": "EDMLoss"
        }
    ]

    print("\n" + "="*60)
    print("🚀 开始编译 HIP 扩展模块（方案一：独立编译）")
    print("="*60)

    success_count = 0
    failed_modules = []

    for ext in hip_extensions:
        success = build_single_hip_extension(
            ext["hip_file"],
            ext["module_name"],
            include_paths,
            lib_paths,
            python_include
        )
        if success:
            success_count += 1
        else:
            failed_modules.append(ext["module_name"])

    # 总结
    print("\n" + "="*60)
    print("📊 编译总结")
    print("="*60)
    print(f"✅ 成功: {success_count}/{len(hip_extensions)}")
    if failed_modules:
        print(f"❌ 失败: {', '.join(failed_modules)}")
        sys.exit(1)
    else:
        print("🎉 所有模块编译成功！")
    print("="*60)


if __name__ == "__main__":
    import time
    s = time.time()

    build_hip_extension()
     
    e = time.time()
    print(f'compile time: {e - s}')
