# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Utility functions for reward computation
HIP kernel 代码处理工具函数
"""
from __future__ import annotations
import os
import json
import re
import ast
from typing import Any, Optional, List, Tuple


AUTO_OUTPUT_CONTRACT = "auto"
SAMPLE_JSON_OUTPUT_CONTRACT = "sample_json_v1"
LEGACY_HIP_FENCE_OUTPUT_CONTRACT = "legacy_hip_fence_v1"
KERNEL_FUNCTION_CODE_UNIT = "kernel_function"
HIP_TRANSLATION_UNIT_CODE_UNIT = "hip_translation_unit"
KNOWN_OUTPUT_CONTRACTS = {
    AUTO_OUTPUT_CONTRACT,
    SAMPLE_JSON_OUTPUT_CONTRACT,
    LEGACY_HIP_FENCE_OUTPUT_CONTRACT,
}
KNOWN_CODE_UNITS = {
    KERNEL_FUNCTION_CODE_UNIT,
    HIP_TRANSLATION_UNIT_CODE_UNIT,
}


def strip_code_fences(s: Optional[str]) -> str:
    """去掉 Markdown 代码围栏
    
    支持的格式：
    1. 标准 markdown: ```lang ... ```
    2. kernel2kernel 格式: ```hip``` ... ``````
    """
    if not s:
        return ""
    s = s.strip()
    
    # 处理开始标签
    if s.startswith("```"):
        # kernel2kernel 格式: ```hip``` 或标准格式: ```lang
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        else:
            # 没有换行符，尝试去掉开头的 ``` 部分
            # 处理 ```hip``` 这种格式（可能在同一行）
            if s.startswith("```hip```"):
                s = s[9:]  # len("```hip```") = 9
            elif s.startswith("```"):
                # 找到第一个 ``` 后面的内容
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


def strip_think_blocks(response: Any) -> str:
    """移除响应里的 `<think>...</think>` 块及残留标签。"""
    if response is None:
        return ""

    text = str(response).strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?think>", "\n", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_output_contract(output_contract: Optional[str]) -> str:
    normalized = str(output_contract or "").strip().lower()
    if not normalized or normalized in {"auto", "none"}:
        return AUTO_OUTPUT_CONTRACT
    if normalized in {
        SAMPLE_JSON_OUTPUT_CONTRACT,
        "sample_json",
        "sample-json",
        "json",
    }:
        return SAMPLE_JSON_OUTPUT_CONTRACT
    if normalized in {
        LEGACY_HIP_FENCE_OUTPUT_CONTRACT,
        "legacy_hip_fence",
        "legacy-hip-fence",
        "hip_fence",
        "fenced_hip",
    }:
        return LEGACY_HIP_FENCE_OUTPUT_CONTRACT
    return normalized


def normalize_expected_code_unit(expected_code_unit: Optional[str]) -> str:
    normalized = str(expected_code_unit or KERNEL_FUNCTION_CODE_UNIT).strip().lower()
    aliases = {
        "kernel": KERNEL_FUNCTION_CODE_UNIT,
        "kernel_snippet": KERNEL_FUNCTION_CODE_UNIT,
        "kernel-function": KERNEL_FUNCTION_CODE_UNIT,
        "function": KERNEL_FUNCTION_CODE_UNIT,
        "hip": HIP_TRANSLATION_UNIT_CODE_UNIT,
        "hip_file": HIP_TRANSLATION_UNIT_CODE_UNIT,
        "hip-file": HIP_TRANSLATION_UNIT_CODE_UNIT,
        "full_file": HIP_TRANSLATION_UNIT_CODE_UNIT,
        "full-file": HIP_TRANSLATION_UNIT_CODE_UNIT,
        "translation_unit": HIP_TRANSLATION_UNIT_CODE_UNIT,
        "translation-unit": HIP_TRANSLATION_UNIT_CODE_UNIT,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in KNOWN_CODE_UNITS:
        supported = ", ".join(sorted(KNOWN_CODE_UNITS))
        raise ValueError(f"Unsupported expected_code_unit={expected_code_unit!r}; supported values: {supported}")
    return normalized


def _find_balanced_brace_end(text: str, start_idx: int) -> Optional[int]:
    depth = 0
    in_single = False
    in_double = False
    escape = False

    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and (in_single or in_double):
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx + 1
    return None


def _iter_json_object_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    for match in re.finditer(r"\{", text):
        start_idx = match.start()
        end_idx = _find_balanced_brace_end(text, start_idx)
        if end_idx is None:
            continue
        candidates.append(text[start_idx:end_idx])
    return candidates


def try_extract_code_from_json_response(
    response: Any,
    *,
    strip_think: bool = False,
) -> Tuple[str, Optional[str]]:
    """严格提取 JSON response 里的 `code` 字段。"""
    if response is None or response == "":
        return ("", "empty response")

    if isinstance(response, dict):
        parsed = response
    else:
        text = strip_think_blocks(response) if strip_think else str(response)
        parsed = None
        parse_errors: List[str] = []
        for candidate in _iter_json_object_candidates(text):
            try:
                parsed = json.loads(candidate)
                break
            except Exception as exc:
                parse_errors.append(f"json.loads: {exc}")
            try:
                parsed = ast.literal_eval(candidate)
                break
            except Exception as exc:
                parse_errors.append(f"ast.literal_eval: {exc}")
        if parsed is None:
            if parse_errors:
                return ("", "; ".join(parse_errors[:4]))
            return ("", "no JSON object found in response")

    if not isinstance(parsed, dict):
        return ("", f"parsed payload is not a dict: {type(parsed).__name__}")

    code = parsed.get("code")
    if not isinstance(code, str) or not code.strip():
        return ("", "response JSON missing non-empty string `code` field")

    return (code, None)


def extract_code_from_json_response(response: str) -> str:
    """从 JSON 格式的 response 中提取 code 字段。

    保留旧接口语义：解析失败时返回原始字符串。
    """
    code, error = try_extract_code_from_json_response(response)
    if error:
        print(f"[WARN] Failed to extract code from JSON response: {error}")
        return "" if not response else str(response)
    return code


def _match_brace_range(code: str, start_pos: int) -> Optional[tuple]:
    brace_start = code.find("{", start_pos)
    if brace_start == -1:
        return None

    brace_count = 1
    pos = brace_start + 1
    while pos < len(code) and brace_count > 0:
        if code[pos] == "{":
            brace_count += 1
        elif code[pos] == "}":
            brace_count -= 1
        pos += 1

    if brace_count != 0:
        return None
    return (start_pos, pos)


def extract_kernel_snippet_from_code(
    code: str,
    kernel_name: Optional[str] = None,
    hip_ref: Optional[str] = None,
) -> str:
    normalized = strip_code_fences(code).strip()
    if not normalized:
        return ""

    target_kernel_name = kernel_name
    if not target_kernel_name and hip_ref:
        target_kernel_name = extract_kernel_name(hip_ref)

    if target_kernel_name:
        kernel_range = find_kernel_in_code(normalized, target_kernel_name)
        if kernel_range:
            return normalized[kernel_range[0] : kernel_range[1]].strip()

    auto_kernel_name = extract_kernel_name(normalized)
    if auto_kernel_name:
        kernel_range = find_kernel_in_code(normalized, auto_kernel_name)
        if kernel_range:
            return normalized[kernel_range[0] : kernel_range[1]].strip()

    generic_match = re.search(r"\b__global__\b", normalized)
    if generic_match:
        kernel_range = _match_brace_range(normalized, generic_match.start())
        if kernel_range:
            return normalized[kernel_range[0] : kernel_range[1]].strip()

    return ""


def extract_hip_kernel_code_from_response(
    response: str,
    kernel_name: Optional[str] = None,
    hip_ref: Optional[str] = None,
) -> str:
    """Extract HIP kernel code from reasoning + model response.

    Pipeline:
        response
          -> remove balanced <think>...</think>
          -> scan only ```hip ... ``` fenced code blocks
          -> extract a kernel snippet from each hip block
          -> return the best hip-fenced kernel block
          -> return "" if no hip-fenced kernel-like code exists

    Notes:
    - Only code inside a hip fence is considered valid input.
    - Never fall back to raw reasoning text or non-hip fences.
    """
    if not response:
        return ""

    text = strip_think_blocks(response)
    target_kernel_name = kernel_name
    if not target_kernel_name and hip_ref:
        target_kernel_name = extract_kernel_name(hip_ref)

    hip_blocks = [
        match.group(0)
        for match in re.finditer(r"```[ \t]*hip[ \t]*\n[\s\S]*?```", text, re.IGNORECASE)
    ]
    fenced_candidates = []
    for raw_block in hip_blocks:
        extracted = extract_kernel_snippet_from_code(
            raw_block,
            kernel_name=kernel_name,
            hip_ref=hip_ref,
        )
        if extracted:
            fenced_candidates.append(extracted)

    if fenced_candidates:
        if target_kernel_name:
            for code in fenced_candidates:
                if re.search(rf"\b{re.escape(target_kernel_name)}\b", code):
                    return code
        return fenced_candidates[0]

    return ""


def _has_hip_fence(text: str) -> bool:
    return bool(re.search(r"```[ \t]*hip[ \t]*\n", text, flags=re.IGNORECASE))


def _looks_like_json_code_payload(text: str) -> bool:
    return bool(re.search(r'["\']code["\']\s*:', text))


def _resolve_parse_modes(
    output_contract: Optional[str],
    data_source: Optional[str],
    stripped_response: str,
) -> Tuple[str, List[str], List[str]]:
    requested_contract = normalize_output_contract(output_contract)
    warnings: List[str] = []
    resolved_contract = requested_contract

    if requested_contract not in KNOWN_OUTPUT_CONTRACTS:
        warnings.append(
            f"unknown output_contract={requested_contract!r}, fallback to auto detection"
        )
        resolved_contract = AUTO_OUTPUT_CONTRACT

    has_hip_fence = _has_hip_fence(stripped_response)
    looks_json = _looks_like_json_code_payload(stripped_response)

    if resolved_contract == SAMPLE_JSON_OUTPUT_CONTRACT:
        return (resolved_contract, [SAMPLE_JSON_OUTPUT_CONTRACT], warnings)
    if resolved_contract == LEGACY_HIP_FENCE_OUTPUT_CONTRACT:
        return (resolved_contract, [LEGACY_HIP_FENCE_OUTPUT_CONTRACT], warnings)
    if looks_json and not has_hip_fence:
        return (resolved_contract, [SAMPLE_JSON_OUTPUT_CONTRACT, LEGACY_HIP_FENCE_OUTPUT_CONTRACT], warnings)
    if has_hip_fence and not looks_json:
        return (resolved_contract, [LEGACY_HIP_FENCE_OUTPUT_CONTRACT, SAMPLE_JSON_OUTPUT_CONTRACT], warnings)
    if data_source == "kernel-agent-single-sft-train":
        return (resolved_contract, [SAMPLE_JSON_OUTPUT_CONTRACT, LEGACY_HIP_FENCE_OUTPUT_CONTRACT], warnings)
    return (resolved_contract, [LEGACY_HIP_FENCE_OUTPUT_CONTRACT, SAMPLE_JSON_OUTPUT_CONTRACT], warnings)


def _build_parse_attempt_chain(parse_modes: List[str], fallback: str = "") -> str:
    compact_modes = [mode for mode in parse_modes if mode]
    if compact_modes:
        return "->".join(compact_modes)
    return fallback


def parse_kernel_generation_response(
    response: Any,
    *,
    data_source: Optional[str] = None,
    kernel_name: Optional[str] = None,
    hip_ref: Optional[str] = None,
    output_contract: Optional[str] = None,
) -> dict[str, Any]:
    stripped_response = strip_think_blocks(response)
    resolved_contract, parse_modes, warnings = _resolve_parse_modes(
        output_contract=output_contract,
        data_source=data_source,
        stripped_response=stripped_response,
    )
    parse_attempt_chain = _build_parse_attempt_chain(parse_modes, fallback=resolved_contract)
    errors = list(warnings)

    for parse_mode in parse_modes:
        if parse_mode == SAMPLE_JSON_OUTPUT_CONTRACT:
            code, error = try_extract_code_from_json_response(
                response,
                strip_think=True,
            )
            if error:
                errors.append(f"{parse_mode}: {error}")
                continue
            hip_src = extract_kernel_snippet_from_code(
                code,
                kernel_name=kernel_name,
                hip_ref=hip_ref,
            )
            if not hip_src or "__global__" not in hip_src:
                errors.append(f"{parse_mode}: extracted `code` field does not contain a valid __global__ kernel")
                continue
            return {
                "hip_src": hip_src,
                "parse_mode": parse_mode,
                "parse_ok": True,
                "parse_error": "",
                "output_contract": resolved_contract,
                "attempted_parse_modes": parse_modes,
                "parse_attempt_chain": parse_attempt_chain,
            }

        hip_src = extract_hip_kernel_code_from_response(
            response,
            kernel_name=kernel_name,
            hip_ref=hip_ref,
        )
        if not hip_src or "__global__" not in hip_src:
            errors.append(f"{parse_mode}: no valid ```hip fenced kernel found")
            continue
        return {
            "hip_src": hip_src,
            "parse_mode": parse_mode,
            "parse_ok": True,
            "parse_error": "",
            "output_contract": resolved_contract,
            "attempted_parse_modes": parse_modes,
            "parse_attempt_chain": parse_attempt_chain,
        }

    return {
        "hip_src": "",
        "parse_mode": parse_attempt_chain,
        "parse_ok": False,
        "parse_error": " | ".join(errors) if errors else "failed to parse kernel response",
        "output_contract": resolved_contract,
        "attempted_parse_modes": parse_modes,
        "parse_attempt_chain": parse_attempt_chain,
    }


def _has_balanced_curly_braces(code: str) -> bool:
    depth = 0
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False
    escape = False
    idx = 0
    while idx < len(code):
        ch = code[idx]
        nxt = code[idx + 1] if idx + 1 < len(code) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            idx += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                idx += 2
                continue
            idx += 1
            continue
        if escape:
            escape = False
            idx += 1
            continue
        if ch == "\\" and (in_single or in_double):
            escape = True
            idx += 1
            continue
        if not in_single and not in_double and ch == "/" and nxt == "/":
            in_line_comment = True
            idx += 2
            continue
        if not in_single and not in_double and ch == "/" and nxt == "*":
            in_block_comment = True
            idx += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            idx += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            idx += 1
            continue
        if in_single or in_double:
            idx += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
        idx += 1
    return depth == 0 and not in_single and not in_double and not in_block_comment


def _validate_hip_translation_unit(hip_src: str) -> Optional[str]:
    if not hip_src or not hip_src.strip():
        return "empty HIP translation unit"
    if "```" in hip_src:
        return "HIP translation unit contains markdown fence markers"
    if "__global__" not in hip_src and "hipLaunchKernelGGL" not in hip_src:
        return "HIP translation unit does not contain a visible HIP kernel or launch"
    if not _has_balanced_curly_braces(hip_src):
        return "HIP translation unit has unbalanced braces or unterminated strings/comments"
    return None


def _extract_first_hip_fenced_translation_unit(response: Any) -> Tuple[str, Optional[str]]:
    text = strip_think_blocks(response)
    matches = re.findall(r"```[ \t]*hip[ \t]*\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if not matches:
        return ("", "no ```hip fenced HIP translation unit found")
    hip_src = matches[0].strip()
    error = _validate_hip_translation_unit(hip_src)
    if error:
        return ("", error)
    return (hip_src, None)


def _extract_json_translation_unit(response: Any) -> Tuple[str, Optional[str]]:
    code, error = try_extract_code_from_json_response(response, strip_think=True)
    if error:
        return ("", error)
    if "```" in code:
        return ("", "JSON `code` field must not contain markdown fence markers")
    hip_src = code.strip()
    error = _validate_hip_translation_unit(hip_src)
    if error:
        return ("", error)
    return (hip_src, None)


def parse_hip_translation_unit_response(
    response: Any,
    *,
    data_source: Optional[str] = None,
    output_contract: Optional[str] = None,
) -> dict[str, Any]:
    stripped_response = strip_think_blocks(response)
    resolved_contract, parse_modes, warnings = _resolve_parse_modes(
        output_contract=output_contract,
        data_source=data_source,
        stripped_response=stripped_response,
    )
    parse_attempt_chain = _build_parse_attempt_chain(parse_modes, fallback=resolved_contract)
    errors = list(warnings)

    for parse_mode in parse_modes:
        if parse_mode == SAMPLE_JSON_OUTPUT_CONTRACT:
            hip_src, error = _extract_json_translation_unit(response)
        else:
            hip_src, error = _extract_first_hip_fenced_translation_unit(response)
        if error:
            errors.append(f"{parse_mode}: {error}")
            continue
        return {
            "hip_src": hip_src,
            "parse_mode": parse_mode,
            "parse_ok": True,
            "parse_error": "",
            "output_contract": resolved_contract,
            "attempted_parse_modes": parse_modes,
            "parse_attempt_chain": parse_attempt_chain,
        }

    if resolved_contract == AUTO_OUTPUT_CONTRACT:
        hip_src = strip_code_fences(stripped_response).strip()
        error = _validate_hip_translation_unit(hip_src)
        if not error:
            return {
                "hip_src": hip_src,
                "parse_mode": "raw_strip_fence",
                "parse_ok": True,
                "parse_error": "",
                "output_contract": resolved_contract,
                "attempted_parse_modes": [*parse_modes, "raw_strip_fence"],
                "parse_attempt_chain": _build_parse_attempt_chain([*parse_modes, "raw_strip_fence"]),
            }
        errors.append(f"raw_strip_fence: {error}")

    return {
        "hip_src": "",
        "parse_mode": parse_attempt_chain,
        "parse_ok": False,
        "parse_error": " | ".join(errors) if errors else "failed to parse HIP translation unit response",
        "output_contract": resolved_contract,
        "attempted_parse_modes": parse_modes,
        "parse_attempt_chain": parse_attempt_chain,
    }


def parse_generation_response(
    response: Any,
    *,
    data_source: Optional[str] = None,
    kernel_name: Optional[str] = None,
    hip_ref: Optional[str] = None,
    output_contract: Optional[str] = None,
    expected_code_unit: Optional[str] = None,
) -> dict[str, Any]:
    code_unit = normalize_expected_code_unit(expected_code_unit)
    if code_unit == HIP_TRANSLATION_UNIT_CODE_UNIT:
        return parse_hip_translation_unit_response(
            response,
            data_source=data_source,
            output_contract=output_contract,
        )
    return parse_kernel_generation_response(
        response,
        data_source=data_source,
        kernel_name=kernel_name,
        hip_ref=hip_ref,
        output_contract=output_contract,
    )


def extract_kernel_name(kernel_code: str) -> Optional[str]:
    """从 kernel 代码中提取 kernel 函数名
    
    支持格式：
    - __global__ void kernel_name(...)
    - __global__ void __launch_bounds__(...) kernel_name(...)
    - template<...> __global__ void kernel_name(...)
    - extern "C" __global__ void kernel_name(...)
    - static __global__ void kernel_name(...)
    """
    if not kernel_code:
        return None

    global_match = re.search(r"\b__global__\b", kernel_code)
    if not global_match:
        return None

    # Normalize common decorators so the final identifier before '(' is the
    # actual kernel name rather than an attribute helper like __launch_bounds__.
    window = kernel_code[global_match.start() : global_match.start() + 1024]
    window = re.sub(r"__launch_bounds__\s*\([^)]*\)", " ", window)
    window = re.sub(r"__attribute__\s*\(\([^)]*\)\)", " ", window)

    patterns = [
        r'__global__[^(){};]*\b([A-Za-z_]\w*)\s*\(',
        r'__global__[\s\S]{0,256}?\b([A-Za-z_]\w*)\s*\(',
    ]
    for pattern in patterns:
        match = re.search(pattern, window)
        if match:
            return match.group(1)
    return None


def find_kernel_in_code(code: str, kernel_name: str) -> Optional[tuple]:
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


def maybe_read_text(val: Optional[str], code_root: Optional[str] = None) -> str:
    """读取文件或返回字符串"""
    if not val:
        return ""
    paths_to_try = []
    if code_root:
        paths_to_try.append(os.path.join(code_root, val))
    paths_to_try.append(val)
    for p in paths_to_try:
        try:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception:
            pass
    return val


# ============================================================================
# DTW (Dynamic Time Warping) 相似度计算
# 用于衡量生成代码与参考代码的结构相似度
# ============================================================================

def tokenize_code(code: str) -> List[str]:
    """
    将代码分词为 token 列表
    
    使用简单的词法分析：
    - 标识符、关键字
    - 运算符、标点符号
    - 数字字面量
    - 字符串字面量
    
    Args:
        code: 源代码字符串
        
    Returns:
        token 列表
    """
    if not code:
        return []
    
    # 移除注释
    # 单行注释 //
    code = re.sub(r'//[^\n]*', '', code)
    # 多行注释 /* */
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    
    # 词法分析正则
    # 匹配：标识符、数字（含浮点）、运算符、标点、字符串
    token_pattern = r'''
        (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')  |  # 字符串字面量
        (?P<number>\d+\.?\d*(?:[eE][+-]?\d+)?[fFlLuU]*)  |  # 数字（含科学计数法和后缀）
        (?P<ident>[a-zA-Z_]\w*)                          |  # 标识符
        (?P<op><<|>>|<=|>=|==|!=|\+=|-=|\*=|/=|&&|\|\||->|\+\+|--)  |  # 双字符运算符
        (?P<punct>[{}()\[\];,.<>+\-*/%&|^~!=?:])         # 单字符运算符和标点
    '''
    
    tokens = []
    for match in re.finditer(token_pattern, code, re.VERBOSE):
        token = match.group().strip()
        if token:
            tokens.append(token)
    
    return tokens


def extract_kernel_body(code: str, kernel_name: Optional[str] = None) -> str:
    """
    从 HIP 代码中提取 kernel 函数体
    
    注意：此函数只提取 __global__ kernel 函数体，不包括：
    - #include 语句
    - 辅助函数（非 __global__）
    - 宏定义
    
    Args:
        code: 完整的 HIP 代码
        kernel_name: kernel 函数名（可选，若不提供则自动检测）
        
    Returns:
        kernel 函数体代码（不含签名），如果找不到则返回空字符串
    """
    if not code:
        return ""
    
    # 确定 kernel 名
    if not kernel_name:
        kernel_name = extract_kernel_name(code)
    
    if not kernel_name:
        # 无法确定 kernel 名，尝试提取所有 __global__ 函数体
        # 而不是返回整个代码（可能包含 includes 等无关内容）
        print(f"[WARN] extract_kernel_body: cannot determine kernel name, trying to extract first __global__ function")
        kernel_name = extract_kernel_name(code)
        if not kernel_name:
            # 仍然找不到，返回空字符串而不是整个代码
            print(f"[WARN] extract_kernel_body: no __global__ function found, returning empty string")
            return ""
    
    # 查找 kernel 函数位置
    kernel_range = find_kernel_in_code(code, kernel_name)
    if not kernel_range:
        # 找不到指定 kernel，打印警告并返回空字符串
        print(f"[WARN] extract_kernel_body: kernel '{kernel_name}' not found in code, returning empty string")
        return ""
    
    kernel_start, kernel_end = kernel_range
    kernel_code = code[kernel_start:kernel_end]
    
    # 提取函数体（花括号内的部分）
    brace_start = kernel_code.find('{')
    if brace_start == -1:
        return kernel_code
    
    # 函数体从第一个 { 到最后一个 }
    body = kernel_code[brace_start + 1:-1].strip()
    return body


def dtw_distance(seq1: List[str], seq2: List[str]) -> float:
    """
    计算两个序列的 DTW (Dynamic Time Warping) 距离
    
    DTW 允许序列在时间轴上进行弹性匹配，适合比较不同长度的代码序列
    
    Args:
        seq1: 第一个 token 序列
        seq2: 第二个 token 序列
        
    Returns:
        DTW 距离（非负浮点数）
    """
    n, m = len(seq1), len(seq2)
    
    # 处理空序列
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return float(max(n, m))
    
    # 动态规划表
    # dtw[i][j] = 将 seq1[0:i] 与 seq2[0:j] 对齐的最小累积距离
    # 使用滚动数组优化内存
    prev = [float('inf')] * (m + 1)
    curr = [float('inf')] * (m + 1)
    prev[0] = 0.0
    
    for i in range(1, n + 1):
        curr[0] = float('inf')
        for j in range(1, m + 1):
            # 局部距离：token 相同为 0，不同为 1
            cost = 0.0 if seq1[i - 1] == seq2[j - 1] else 1.0
            # DTW 递推：取三个方向的最小值
            curr[j] = cost + min(
                prev[j],      # 插入（seq2 多出一个）
                curr[j - 1],  # 删除（seq1 多出一个）
                prev[j - 1]   # 匹配
            )
        prev, curr = curr, prev
    
    return prev[m]


def normalized_dtw_distance(code1: str, code2: str, 
                            kernel_name: Optional[str] = None) -> Tuple[float, int, int]:
    """
    计算两段代码之间的归一化 DTW 距离（仅比较 kernel function body）
    
    归一化方式：DTW距离 / max(len1, len2)
    这使得距离值在 [0, 1] 范围内，便于设置阈值
    
    注意：只比较 kernel function 级别的代码，不包括 includes、辅助函数等
    
    Args:
        code1: 第一段代码（通常是参考代码）
        code2: 第二段代码（通常是生成代码）
        kernel_name: kernel 函数名（用于提取 kernel body）
        
    Returns:
        (normalized_dist, len1, len2):
        - normalized_dist: 归一化 DTW 距离 ∈ [0, 1]
        - len1: code1 的 token 数量
        - len2: code2 的 token 数量
    """
    # 提取 kernel body（只提取 __global__ 函数体）
    body1 = extract_kernel_body(code1, kernel_name)
    body2 = extract_kernel_body(code2, kernel_name)
    
    # 如果无法提取 kernel body，记录警告
    if not body1:
        print(f"[WARN] normalized_dtw_distance: failed to extract kernel body from code1 (kernel_name={kernel_name})")
    if not body2:
        print(f"[WARN] normalized_dtw_distance: failed to extract kernel body from code2 (kernel_name={kernel_name})")
    
    # 分词
    tokens1 = tokenize_code(body1)
    tokens2 = tokenize_code(body2)
    
    len1, len2 = len(tokens1), len(tokens2)
    
    # 处理空序列（可能是 kernel 提取失败）
    if len1 == 0 and len2 == 0:
        # 两个都为空，无法比较，返回 0 距离（保守策略）
        return (0.0, 0, 0)
    if len1 == 0 or len2 == 0:
        # 一个为空一个不为空，视为完全不同
        return (1.0, len1, len2)
    
    # 计算 DTW 距离
    raw_dist = dtw_distance(tokens1, tokens2)
    
    # 归一化
    max_len = max(len1, len2)
    normalized = raw_dist / max_len
    
    # 确保在 [0, 1] 范围内
    normalized = min(max(normalized, 0.0), 1.0)
    
    return (normalized, len1, len2)


def compute_dtw_to_ref(ref_code: str, gen_code: str, 
                       kernel_name: Optional[str] = None) -> Tuple[float, int]:
    """
    计算生成代码与参考代码的 DTW 相似度距离
    
    这是 reward 计算中使用的主要接口
    
    Args:
        ref_code: 参考代码（reference kernel）
        gen_code: 生成代码（generated kernel）
        kernel_name: kernel 函数名
        
    Returns:
        (d_ref, max_token_len):
        - d_ref: 归一化 DTW 距离 ∈ [0, 1]
          - 接近 0：生成代码与参考代码非常相似（可能是复制）
          - 接近 1：生成代码与参考代码差异较大（有创新）
        - max_token_len: 两段代码中较长的 token 数量（用于自适应阈值）
    """
    normalized, len1, len2 = normalized_dtw_distance(ref_code, gen_code, kernel_name)
    max_len = max(len1, len2)
    return (normalized, max_len)


def get_adaptive_thresholds(token_len: int) -> Tuple[float, float, float]:
    """
    根据 kernel 的 token 长度计算自适应阈值
    
    设计思路：
    - 短 kernel（<50 tokens）：阈值放宽，避免小改动被过度奖励
    - 中等 kernel（50-200 tokens）：使用标准阈值
    - 长 kernel（>200 tokens）：阈值收紧，避免大量改动被忽视
    
    基准阈值（针对 100 tokens 的 kernel）：
    - d_copy = 0.08 → 8 tokens
    - d_low = 0.10 → 10 tokens  
    - d_high = 0.20 → 20 tokens
    
    自适应策略：按 "绝对 token 差异" 换算回 "归一化距离"
    
    Args:
        token_len: kernel 的 token 数量
        
    Returns:
        (d_copy, d_low, d_high) 自适应阈值元组
    """
    # 基准值（针对 100 tokens）
    BASE_LEN = 100
    BASE_COPY_TOKENS = 8    # 完全复制：改动 < 8 tokens
    BASE_LOW_TOKENS = 10    # 低差异：改动 < 10 tokens
    BASE_HIGH_TOKENS = 20   # 高差异：改动 > 20 tokens
    
    # 最小/最大 token 长度限制，避免极端情况
    MIN_LEN = 20
    MAX_LEN = 500
    effective_len = max(MIN_LEN, min(token_len, MAX_LEN))
    
    # 短 kernel 特殊处理：使用更宽松的绝对 token 数
    if effective_len < 50:
        # 短 kernel：至少需要 6 tokens 差异才不算复制
        copy_tokens = 6
        low_tokens = 8
        high_tokens = 15
    elif effective_len > 200:
        # 长 kernel：按比例缩放，但有上限
        scale = effective_len / BASE_LEN
        copy_tokens = min(BASE_COPY_TOKENS * scale, 30)   # 最多 30 tokens
        low_tokens = min(BASE_LOW_TOKENS * scale, 50)     # 最多 50 tokens
        high_tokens = min(BASE_HIGH_TOKENS * scale, 80)   # 最多 80 tokens
    else:
        # 中等 kernel：标准阈值
        copy_tokens = BASE_COPY_TOKENS
        low_tokens = BASE_LOW_TOKENS
        high_tokens = BASE_HIGH_TOKENS
    
    # 转换回归一化距离
    d_copy = copy_tokens / effective_len
    d_low = low_tokens / effective_len
    d_high = high_tokens / effective_len
    
    # 确保阈值在合理范围内
    d_copy = min(max(d_copy, 0.05), 0.15)   # [0.05, 0.15]
    d_low = min(max(d_low, 0.08), 0.20)     # [0.08, 0.20]
    d_high = min(max(d_high, 0.10), 0.40)   # [0.10, 0.40]
    
    return (d_copy, d_low, d_high)

