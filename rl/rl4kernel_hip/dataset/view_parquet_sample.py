#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Script to view samples from a parquet file with human-readable formatting.
Usage: python view_parquet_sample.py [parquet_file] [num_samples]
"""

import pandas as pd
import sys
import json
import textwrap
import numpy as np

# ANSI color codes
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    BG_GRAY = "\033[48;5;236m"

TERM_WIDTH = 100
THIN_LINE = "─" * TERM_WIDTH
THICK_LINE = "━" * TERM_WIDTH
DOUBLE_LINE = "═" * TERM_WIDTH


def _is_code_string(s: str) -> bool:
    """Heuristic: does this string look like source code?"""
    indicators = ["#include", "__global__", "def ", "class ", "import ", "void ", "return ", "torch.", "PYBIND11"]
    return isinstance(s, str) and len(s) > 80 and any(ind in s for ind in indicators)


def _format_code_block(code: str, lang: str = "", max_lines: int = 40) -> str:
    """Format a code string as a visually distinct block."""
    lines = code.strip().split("\n")
    truncated = len(lines) > max_lines
    if truncated:
        display_lines = lines[:max_lines]
    else:
        display_lines = lines

    header = f"  {C.DIM}┌{'─' * 90} {lang}{C.RESET}"
    footer_mark = f"  {C.DIM}└{'─' * 90}{C.RESET}"
    if truncated:
        footer_mark = f"  {C.DIM}│ ... ({len(lines) - max_lines} more lines, {len(lines)} total){C.RESET}\n" + footer_mark

    body = "\n".join(f"  {C.DIM}│{C.RESET} {line}" for line in display_lines)
    return f"{header}\n{body}\n{footer_mark}"


def _format_dict_compact(d: dict, indent: int = 4, code_max_lines: int = 30) -> str:
    """Pretty-print a dict, rendering code-like values as code blocks."""
    parts = []
    prefix = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            parts.append(f"{prefix}{C.CYAN}{k}{C.RESET}:")
            parts.append(_format_dict_compact(v, indent + 4, code_max_lines))
        elif isinstance(v, (list, np.ndarray)):
            arr_str = str(v)
            if len(arr_str) > 200:
                arr_str = arr_str[:200] + f" ... (truncated)"
            parts.append(f"{prefix}{C.CYAN}{k}{C.RESET}: {arr_str}")
        elif _is_code_string(v):
            lang = "hip" if "#include" in v or "__global__" in v else "python"
            parts.append(f"{prefix}{C.CYAN}{k}{C.RESET}: {C.DIM}[{lang} code, {len(v)} chars]{C.RESET}")
            parts.append(_format_code_block(v, lang=lang, max_lines=code_max_lines))
        elif isinstance(v, str) and len(v) > 200:
            parts.append(f"{prefix}{C.CYAN}{k}{C.RESET}: {v[:200]}... {C.DIM}({len(v)} chars){C.RESET}")
        else:
            parts.append(f"{prefix}{C.CYAN}{k}{C.RESET}: {v}")
    return "\n".join(parts)


def _format_chat_messages(messages: list) -> str:
    """Format a list of chat messages [{role, content}, ...] for readability."""
    parts = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            parts.append(str(msg))
            continue

        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        role_colors = {"system": C.RED, "user": C.GREEN, "assistant": C.BLUE}
        rc = role_colors.get(role, C.YELLOW)
        parts.append(f"  {rc}{C.BOLD}[{role.upper()}]{C.RESET}")
        parts.append(f"  {THIN_LINE[:60]}")

        # Split content into text vs code blocks (``` ... ```)
        segments = content.split("```")
        for j, seg in enumerate(segments):
            if j % 2 == 0:
                # Regular text — wrap and indent
                text = seg.strip()
                if not text:
                    continue
                # Preserve paragraph structure
                paragraphs = text.split("\n\n")
                for para in paragraphs:
                    lines = para.strip().split("\n")
                    for line in lines:
                        wrapped = textwrap.fill(line, width=TERM_WIDTH - 6, initial_indent="    ", subsequent_indent="    ")
                        parts.append(wrapped)
                    parts.append("")
            else:
                # Code block — first line may be language tag
                code_lines = seg.split("\n")
                lang = code_lines[0].strip() if code_lines else ""
                code_body = "\n".join(code_lines[1:]) if len(code_lines) > 1 else ""
                if not code_body.strip():
                    code_body = lang
                    lang = ""
                parts.append(_format_code_block(code_body, lang=lang, max_lines=50))
                parts.append("")

    return "\n".join(parts)


def _to_native(value):
    """Convert numpy/pandas wrappers to native Python types for formatting."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    return value


def format_field(col_name: str, value) -> str:
    """Format a single field value based on column name and content type."""
    value = _to_native(value)

    # prompt: list of chat messages [{role, content}, ...]
    if col_name == "prompt" and isinstance(value, list) and value and isinstance(value[0], dict):
        return _format_chat_messages(value)

    # Nested dicts (reward_model, extra_info, etc.)
    if isinstance(value, dict):
        return _format_dict_compact(value)

    # Lists that aren't chat messages
    if isinstance(value, list):
        formatted = json.dumps(value, indent=4, default=str, ensure_ascii=False)
        if len(formatted) > 500:
            return f"    {formatted[:500]}\n    {C.DIM}... (truncated){C.RESET}"
        return textwrap.indent(formatted, "    ")

    # Long strings
    if isinstance(value, str):
        if _is_code_string(value):
            lang = "hip" if "__global__" in value else "python"
            return _format_code_block(value, lang=lang)
        if len(value) > 500:
            return f"    {value[:500]}\n    {C.DIM}... (truncated, {len(value)} chars total){C.RESET}"
        return f"    {value}"

    return f"    {value}"


def view_parquet_sample(file_path, num_samples=3):
    """View samples from a parquet file with structured pretty-printing."""
    print(f"\n{C.BOLD}Reading:{C.RESET} {file_path}\n")
    print(DOUBLE_LINE)

    df = pd.read_parquet(file_path)

    # Dataset overview
    print(f"\n  {C.BOLD}Dataset Info{C.RESET}")
    print(f"    Rows:    {C.YELLOW}{len(df)}{C.RESET}")
    print(f"    Columns: {C.YELLOW}{len(df.columns)}{C.RESET}")
    print()
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        print(f"    {i}. {C.CYAN}{col}{C.RESET}  {C.DIM}({dtype}){C.RESET}")

    print(f"\n{DOUBLE_LINE}")
    print(f"  {C.BOLD}Showing {min(num_samples, len(df))} sample(s){C.RESET}")
    print(DOUBLE_LINE)

    for idx in range(min(num_samples, len(df))):
        print(f"\n{C.BOLD}{C.MAGENTA}{'▸ Sample #' + str(idx + 1)}{C.RESET}")
        print(THICK_LINE)

        for col in df.columns:
            value = df.iloc[idx][col]

            print(f"\n  {C.BOLD}{C.YELLOW}◆ {col}{C.RESET}")
            print(f"  {THIN_LINE[:60]}")
            print(format_field(col, value))

        print(f"\n{THICK_LINE}\n")

    # Null check (compact)
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print(f"  {C.GREEN}No null values{C.RESET}")
    else:
        print(f"  {C.RED}Null values found:{C.RESET}")
        for col, cnt in null_counts[null_counts > 0].items():
            print(f"    {col}: {cnt}")

    print(DOUBLE_LINE + "\n")


if __name__ == "__main__":
    parquet_file = "/wekafs/zepingl/rl4kernel_hip/dataset/hip2hip_parquet/rl_data_hard3k_normal10k_mixed_hip2hip_mi300x_react_sample_json_v1_verl.parquet"

    if len(sys.argv) > 1:
        parquet_file = sys.argv[1]

    num_samples = 10
    if len(sys.argv) > 2:
        num_samples = int(sys.argv[2])

    try:
        view_parquet_sample(parquet_file, num_samples)
    except FileNotFoundError:
        print(f"{C.RED}Error: File not found: {parquet_file}{C.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{C.RED}Error: {e}{C.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
