#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Kernel utility functions for HIP code processing.

Functions for extracting, finding, and replacing kernel functions in HIP code.
"""

import re
from typing import Optional, Tuple


def extract_kernel_name(kernel_code: str) -> Optional[str]:
    """从 kernel 代码中提取 kernel 函数名
    
    支持格式：
    - __global__ void kernel_name(...)
    - __global__ void __launch_bounds__(...) kernel_name(...)
    - template<...> __global__ void kernel_name(...)
    - extern "C" __global__ void kernel_name(...)
    - static __global__ void kernel_name(...)
    
    Args:
        kernel_code: HIP kernel 代码
        
    Returns:
        kernel 函数名，或 None（未找到）
    """
    # 更鲁棒的正则：匹配 __global__ 后的函数名
    # 支持各种修饰符：__launch_bounds__, __attribute__, const, static 等
    patterns = [
        # 标准格式：__global__ [modifiers] return_type kernel_name(
        r'__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?(?:__attribute__\s*\(\([^)]*\)\)\s*)?(?:void|int|float|double|unsigned|signed|long|short|char|bool|\w+)\s+(\w+)\s*\(',
        # 简化格式：直接找 __global__ 后面的函数名
        r'__global__\s+\w+\s+(\w+)\s*\(',
        # 更宽松：__global__ 到 ( 之间的最后一个标识符
        r'__global__[^(]+\b(\w+)\s*\(',
    ]
    for pattern in patterns:
        match = re.search(pattern, kernel_code)
        if match:
            return match.group(1)
    return None


def find_kernel_in_code(code: str, kernel_name: str) -> Optional[Tuple[int, int]]:
    """在代码中查找指定 kernel 函数的位置
    
    Args:
        code: HIP 代码
        kernel_name: kernel 函数名
        
    Returns:
        (start, end) 位置元组，或 None（未找到）
    """
    # 尝试多种模式匹配 kernel 函数签名
    patterns = [
        # 标准格式
        rf'__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?(?:__attribute__\s*\(\([^)]*\)\)\s*)?(?:void|int|float|double|unsigned|signed|long|short|char|bool|\w+)\s+{re.escape(kernel_name)}\s*\(',
        # 简化格式
        rf'__global__\s+\w+\s+{re.escape(kernel_name)}\s*\(',
        # 宽松格式：只要 __global__ 后有该函数名
        rf'__global__[^{{;]*\b{re.escape(kernel_name)}\s*\(',
    ]
    
    match = None
    for pattern in patterns:
        match = re.search(pattern, code)
        if match:
            break
    
    if not match:
        return None
    
    kernel_start = match.start()
    
    # 找到函数体的开始 '{'
    brace_start = code.find('{', match.end())
    if brace_start == -1:
        return None
    
    # 配对花括号找到函数体结束
    brace_count = 1
    pos = brace_start + 1
    while pos < len(code) and brace_count > 0:
        if code[pos] == '{':
            brace_count += 1
        elif code[pos] == '}':
            brace_count -= 1
        pos += 1
    
    if brace_count != 0:
        return None
    
    return (kernel_start, pos)


def extract_kernel_from_hip_code(hip_code: str) -> Tuple[Optional[str], Optional[str]]:
    """从完整的HIP代码中提取kernel函数
    
    Args:
        hip_code: 完整的HIP代码（包含includes、辅助函数、kernel等）
        
    Returns:
        (kernel_code, kernel_name) 元组，如果未找到则返回 (None, None)
    """
    # 首先找到kernel函数名
    kernel_name = extract_kernel_name(hip_code)
    if not kernel_name:
        return None, None
    
    # 找到kernel函数的位置
    kernel_range = find_kernel_in_code(hip_code, kernel_name)
    if not kernel_range:
        return None, None
    
    kernel_start, kernel_end = kernel_range
    kernel_code = hip_code[kernel_start:kernel_end]
    
    return kernel_code, kernel_name


def replace_kernel_in_hip_code(hip_ref: str, new_kernel: str, kernel_name: Optional[str] = None) -> str:
    """将 hip_ref 中的对应 kernel 函数替换为 new_kernel
    
    Args:
        hip_ref: 完整的 HIP 代码（包含 includes、辅助函数、kernel 等）
        new_kernel: 新的 kernel 函数代码（以 __global__ 开头）
        kernel_name: 已知的 kernel 函数名（可选，若提供则优先使用）
        
    Returns:
        替换后的完整 HIP 代码
    """
    # 确定原始 kernel 名（从 hip_ref 或参数获取）
    original_kernel_name = kernel_name
    if not original_kernel_name:
        original_kernel_name = extract_kernel_name(hip_ref)
    
    if not original_kernel_name:
        print(f"[WARN] Cannot determine original kernel name, fallback to raw kernel")
        return new_kernel
    
    # 提取生成的 kernel 名
    generated_kernel_name = extract_kernel_name(new_kernel)
    
    # 如果模型改变了 kernel 名字，需要替换回原始名字以保持与调用处一致
    if generated_kernel_name and generated_kernel_name != original_kernel_name:
        print(f"[INFO] Model changed kernel name from '{original_kernel_name}' to '{generated_kernel_name}', reverting to original name")
        new_kernel = new_kernel.replace(generated_kernel_name, original_kernel_name)
    
    # 在 hip_ref 中查找 kernel 函数位置
    kernel_range = find_kernel_in_code(hip_ref, original_kernel_name)
    
    if not kernel_range:
        print(f"[WARN] Cannot find kernel '{original_kernel_name}' in hip_ref, fallback to raw kernel")
        return new_kernel
    
    kernel_start, kernel_end = kernel_range
    
    # 替换 kernel
    result = hip_ref[:kernel_start] + new_kernel + hip_ref[kernel_end:]
    return result


def strip_code_fences(s: Optional[str]) -> str:
    """去掉 Markdown 代码围栏
    
    支持的格式：
    1. 标准 markdown: ```lang ... ```
    2. kernel2kernel 格式: ```hip``` ... ``````
    
    Args:
        s: 可能包含代码围栏的字符串
        
    Returns:
        去掉代码围栏后的字符串
    """
    if not s:
        return ""
    s = s.strip()
    
    # 处理开始标签
    if s.startswith("```"):
        # kernel2kernel 格式: ```hip``` 或标准格式: ```lang
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        else:
            # 没有换行符，尝试去掉开头的 ``` 部分
            if s.startswith("```hip```"):
                s = s[9:]  # len("```hip```") = 9
            elif s.startswith("```"):
                end_of_tag = s.find("```", 3)
                if end_of_tag != -1:
                    s = s[end_of_tag + 3:]
                else:
                    s = s[3:]
    
    s = s.strip()
    
    # 处理结束标签
    # kernel2kernel 格式使用 `````` (6个反引号)
    if s.endswith("``````"):
        s = s[:-6]
    elif s.endswith("```"):
        s = s[:-3]
    
    return s.strip()

