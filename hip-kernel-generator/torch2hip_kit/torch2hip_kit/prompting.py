# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .config import AttemptRecord

CODE_FENCE_PATTERN = re.compile(r"```(?:[\w.+-]+)?\s*(.*?)```", re.DOTALL)


def format_response(response: str) -> str:
    cleaned = response.strip()
    if "<think>" in cleaned and "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()

    fenced_match = CODE_FENCE_PATTERN.search(cleaned)
    if fenced_match:
        return fenced_match.group(1).strip()
    return cleaned


def _truncate_block(content: str, limit: int) -> str:
    stripped = content.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rstrip() + "\n... [truncated]"


def _render_attempt_history(
    attempt_records: Iterable[AttemptRecord],
    *,
    code_char_limit: int,
    feedback_char_limit: int,
) -> str:
    sections: list[str] = []
    for record in attempt_records:
        candidate_code = ""
        candidate_path = Path(record.candidate_path)
        if candidate_path.exists():
            candidate_code = candidate_path.read_text(encoding="utf-8")

        status_parts = [f"Status: {record.status}."]
        status_parts.append(f"Compile={'passed' if record.compile_success else 'failed'}.")
        if record.correctness_success:
            status_parts.append("Correctness=passed.")
        elif record.compile_success:
            status_parts.append("Correctness=failed.")
        if record.speedup is not None:
            status_parts.append(f"Speedup={record.speedup:.4f}x.")
        if record.module_latency_ms is not None:
            status_parts.append(f"PyTorch latency={record.module_latency_ms:.4f}ms.")
        if record.hip_latency_ms is not None:
            status_parts.append(f"HIP latency={record.hip_latency_ms:.4f}ms.")

        feedback = record.feedback or record.error or record.mismatch or "No feedback captured."
        sections.extend(
            [
                f"Attempt {record.attempt}:",
                " ".join(status_parts),
                "Candidate HIP code:",
                "```cpp",
                _truncate_block(candidate_code or "// Candidate code was not written.", code_char_limit),
                "```",
                "Verifier feedback, error trace, or performance note:",
                "```text",
                _truncate_block(feedback, feedback_char_limit),
                "```",
                "",
            ]
        )
    return "\n".join(sections).strip()


def _best_speedup(attempt_records: Iterable[AttemptRecord]) -> float | None:
    speedups = [record.speedup for record in attempt_records if record.speedup is not None]
    if not speedups:
        return None
    return max(speedups)


def build_prompt(
    module_code: str,
    functional_code: str,
    relative_path: Path,
    attempt_records: list[AttemptRecord],
    *,
    system_instruction: str,
    few_shot_examples: str,
    code_char_limit: int,
    feedback_char_limit: int,
) -> str:
    sections = [
        system_instruction.strip(),
        "",
        "Few-shot examples:",
        few_shot_examples.strip(),
        "",
        "Task requirements:",
        "- Produce exactly one HIP source file.",
        "- Expose a `forward` function through `PYBIND11_MODULE`.",
        "- Keep the `forward` signature callable from the paired functional PyTorch file.",
        "- Preserve the semantics of the original PyTorch module.",
        "- Optimize for speed only after correctness is preserved.",
        "- Reuse any previously correct candidate as a semantic baseline when possible.",
        "- If a previous attempt failed, fix the exact issue described in the verifier feedback or traceback.",
        "- If a previous attempt was correct but slower, preserve its interface and improve kernel performance.",
        "",
        f"Target sample: {relative_path.as_posix()}",
        "",
        "Original PyTorch module code:",
        module_code.strip(),
        "",
        "Paired PyTorch functional code:",
        functional_code.strip(),
    ]

    best_speedup = _best_speedup(attempt_records)
    if best_speedup is not None:
        sections.extend(
            [
                "",
                f"Current best validated speedup: {best_speedup:.4f}x. Try to beat it without breaking correctness.",
            ]
        )

    history_section = _render_attempt_history(
        attempt_records,
        code_char_limit=code_char_limit,
        feedback_char_limit=feedback_char_limit,
    )
    if history_section:
        sections.extend(
            [
                "",
                "Previous attempts are included below. Use the candidate code together with the verifier feedback, traceback, and speedup numbers to improve both correctness and performance:",
                history_section,
            ]
        )

    return "\n".join(sections) + "\n"
