#!/bin/bash
# 修复 hipcc/clang++ 的 amdgpu-target 参数问题 V2
# 
# 问题：hipcc.original 内部使用 system() 执行命令，生成畸形参数
#       --amdgpu-target=gfx90a;gfx942，分号被 shell 解析导致错误
#
# 解决方案：
# 1. hipcc wrapper 完全绕过 hipcc.original，直接调用 clang++
# 2. clang++ wrapper 过滤所有 --amdgpu-target 参数
# 3. 完善 hipcc wrapper，添加必要的 include 和 library 路径

set -e

echo "============================================================"
echo "修复 HIP 编译环境 (hipcc & clang++) V2"
echo "============================================================"

# 检查是否有 root 权限
if [ "$EUID" -ne 0 ]; then 
    echo "❌ 请使用 root 权限运行此脚本："
    echo "   sudo bash FIX_HIPCC_V2.sh"
    exit 1
fi

#=============================================
# Part 1: 修复 hipcc
#=============================================
echo ""
echo "=== Part 1: 修复 hipcc ==="

HIPCC_PATH="/opt/rocm/bin/hipcc"
HIPCC_BACKUP="/opt/rocm/bin/hipcc.original"

echo "1. 备份原始 hipcc..."

# 检查是否需要备份（只有当 hipcc 是 ELF 文件时才备份）
if file "$HIPCC_PATH" 2>/dev/null | grep -q "ELF"; then
    if [ ! -f "$HIPCC_BACKUP" ]; then
        cp "$HIPCC_PATH" "$HIPCC_BACKUP"
        echo "   ✓ 已备份到 $HIPCC_BACKUP"
    else
        echo "   ✓ 备份已存在，跳过"
    fi
else
    echo "   ✓ hipcc 不是 ELF 文件，跳过备份"
fi

echo "2. 创建 hipcc wrapper (完整功能版)..."

# 强制删除并重新创建
rm -f "$HIPCC_PATH"

# 新的 hipcc wrapper - 完整模拟 hipcc 功能
cat > "$HIPCC_PATH" << 'HIPCC_WRAPPER'
#!/bin/bash
# hipcc wrapper - 完整功能版
# 直接调用 clang++，绕过有问题的 hipcc.original
# 同时添加必要的 HIP include 路径和库链接

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
HIP_CLANG_PATH="$ROCM_PATH/lib/llvm/bin"
GPU_ARCH="${GPU_TARGETS:-gfx942}"

# 收集参数
args=()
hip_files=()
output_file=""
next_is_output=false
is_compile_only=false
is_link_only=false
has_std_flag=false
verbose=false

for arg in "$@"; do
    if $next_is_output; then
        output_file="$arg"
        next_is_output=false
        continue
    fi
    
    case "$arg" in
        -o)
            next_is_output=true
            ;;
        -c)
            is_compile_only=true
            args+=("$arg")
            ;;
        --version)
            echo "HIP version: 6.3 (wrapper)"
            exec "$HIP_CLANG_PATH/clang" --version
            ;;
        --help|-h)
            echo "hipcc wrapper - 调用 clang++ 编译 HIP 代码"
            echo ""
            echo "Usage: hipcc [options] <source files>"
            echo ""
            echo "This is a wrapper that bypasses hipcc.original to avoid"
            echo "the malformed --amdgpu-target argument issue."
            echo ""
            exec "$HIP_CLANG_PATH/clang" --help
            ;;
        -v|--verbose)
            verbose=true
            args+=("$arg")
            ;;
        *.hip|*.cu)
            hip_files+=("$arg")
            ;;
        *.cpp|*.cxx|*.cc)
            # C++ 文件也可能包含 HIP 代码
            hip_files+=("$arg")
            ;;
        *.o|*.a|*.so)
            # 目标文件/库文件
            args+=("$arg")
            ;;
        --amdgpu-target=*)
            # 完全忽略这个参数
            if $verbose; then
                echo "hipcc wrapper: Ignoring deprecated --amdgpu-target" >&2
            fi
            ;;
        --offload-arch=*)
            # 使用用户指定的架构
            GPU_ARCH="${arg#--offload-arch=}"
            ;;
        -std=*)
            has_std_flag=true
            args+=("$arg")
            ;;
        -I*|-L*|-l*|-D*|-W*|-O*|-g*|-f*|-m*|-pthread)
            args+=("$arg")
            ;;
        -x)
            # 跳过 -x，我们会自己处理
            ;;
        hip|cuda)
            # 跳过语言类型参数
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

# 构建 clang++ 命令
clang_args=()

# 添加 HIP 编译选项
clang_args+=("-x" "hip")
clang_args+=("--offload-arch=$GPU_ARCH")
clang_args+=("-D__HIP_PLATFORM_AMD__")

# 添加 HIP include 路径
clang_args+=("-I$ROCM_PATH/include")
clang_args+=("-I$ROCM_PATH/include/hip")
clang_args+=("-I$HIP_PATH/include")

# 如果不是只编译模式，添加链接选项
if ! $is_compile_only; then
    clang_args+=("--hip-link")
    clang_args+=("-L$ROCM_PATH/lib")
    clang_args+=("-lamdhip64")
    # 链接 C++ 标准库和数学库（关键！）
    clang_args+=("-lstdc++")
    clang_args+=("-lm")
    # 添加 rpath 以便运行时能找到库
    clang_args+=("-Wl,-rpath,$ROCM_PATH/lib")
fi

# 如果没有指定 C++ 标准，使用 C++17
if ! $has_std_flag; then
    clang_args+=("-std=c++17")
fi

# 添加用户参数
clang_args+=("${args[@]}")

# 添加输出文件
if [ -n "$output_file" ]; then
    clang_args+=("-o" "$output_file")
fi

# 添加源文件
clang_args+=("${hip_files[@]}")

# 如果 verbose 模式，打印完整命令
if $verbose; then
    echo "hipcc wrapper: $HIP_CLANG_PATH/clang++ ${clang_args[*]}" >&2
fi

# 调用 clang++
exec "$HIP_CLANG_PATH/clang++" "${clang_args[@]}"
HIPCC_WRAPPER

chmod +x "$HIPCC_PATH"
echo "   ✓ hipcc wrapper 已创建 (完整功能版)"

#=============================================
# Part 2: 修复 clang++
#=============================================
echo ""
echo "=== Part 2: 修复 clang++ ==="

CLANG_PATH="/opt/rocm/lib/llvm/bin/clang++"
CLANG_BACKUP="/opt/rocm/lib/llvm/bin/clang++.original"
CLANG_REAL="/opt/rocm/lib/llvm/bin/clang"

echo "3. 处理 clang++..."

# 如果是符号链接，需要特殊处理
if [ -L "$CLANG_PATH" ]; then
    echo "   检测到 clang++ 是符号链接 -> $(readlink "$CLANG_PATH")"
    rm -f "$CLANG_PATH"
    echo "   ✓ 已删除符号链接"
elif file "$CLANG_PATH" 2>/dev/null | grep -q "ELF"; then
    if [ ! -f "$CLANG_BACKUP" ]; then
        cp "$CLANG_PATH" "$CLANG_BACKUP"
        echo "   ✓ 已备份到 $CLANG_BACKUP"
    fi
    rm -f "$CLANG_PATH"
else
    rm -f "$CLANG_PATH"
    echo "   ✓ 已删除旧文件"
fi

echo "4. 创建 clang++ wrapper..."

cat > "$CLANG_PATH" << 'CLANG_WRAPPER'
#!/bin/bash
# clang++ wrapper - 过滤错误的 --amdgpu-target 参数

args=()
for arg in "$@"; do
    # 跳过所有 --amdgpu-target 参数
    if [[ "$arg" =~ ^--amdgpu-target= ]]; then
        echo "Warning: Filtered deprecated argument: $arg" >&2
        continue
    fi
    args+=("$arg")
done

# 调用真正的 clang 编译器
exec /opt/rocm/lib/llvm/bin/clang "${args[@]}"
CLANG_WRAPPER

chmod +x "$CLANG_PATH"
echo "   ✓ clang++ wrapper 已创建"

#=============================================
# Part 3: 验证
#=============================================
echo ""
echo "=== Part 3: 验证修复结果 ==="

echo "5. 验证 hipcc..."
HIPCC_TYPE=$(file "$HIPCC_PATH")
if echo "$HIPCC_TYPE" | grep -q -E "(shell script|ASCII text)"; then
    echo "   ✓ hipcc: shell script"
else
    echo "   ❌ hipcc: 验证失败 - $HIPCC_TYPE"
    exit 1
fi

echo "6. 验证 clang++..."
CLANG_TYPE=$(file "$CLANG_PATH")
if echo "$CLANG_TYPE" | grep -q -E "(shell script|ASCII text)"; then
    echo "   ✓ clang++: shell script"
else
    echo "   ❌ clang++: 验证失败 - $CLANG_TYPE"
    exit 1
fi

echo ""
echo "7. 测试 hipcc..."
"$HIPCC_PATH" --version 2>&1 | head -3

echo ""
echo "============================================================"
echo "✅ 修复完成！"
echo "============================================================"
echo ""
echo "修复内容:"
echo "  - hipcc wrapper: 直接调用 clang++，绕过 hipcc.original"
echo "  - clang++ wrapper: 过滤所有 --amdgpu-target 参数"
echo ""
echo "hipcc wrapper 功能:"
echo "  - 自动添加 HIP include 路径 (-I$ROCM_PATH/include)"
echo "  - 自动链接 HIP 运行时库 (-lamdhip64)"
echo "  - 自动链接 C++ 标准库 (-lstdc++) 和数学库 (-lm)"
echo "  - 使用 --offload-arch=gfx942 指定 GPU 架构"
echo "  - 支持 -v 选项查看完整编译命令"
echo ""
echo "备份文件:"
echo "  - $HIPCC_BACKUP (如果存在)"
echo ""
echo "如需恢复原始版本:"
echo "  mv $HIPCC_BACKUP $HIPCC_PATH"
echo "  rm $CLANG_PATH && ln -s clang $CLANG_PATH"
echo ""
echo "现在可以运行测试:"
echo "  rm -rf ~/.cache/torch_extensions/*"
echo "  # 重新运行你的评估脚本"
echo ""
