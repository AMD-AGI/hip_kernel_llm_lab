# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .config import AttemptRecord
from .hip_parser import GPUFunction

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


def _best_speedup_vs_baseline(attempt_records: Iterable[AttemptRecord]) -> float | None:
    speedups = [record.speedup_vs_baseline for record in attempt_records if record.speedup_vs_baseline is not None]
    if not speedups:
        return None
    return max(speedups)


def _render_attempt_history(
    attempt_records: Iterable[AttemptRecord],
    *,
    code_char_limit: int,
    feedback_char_limit: int,
) -> str:
    sections: list[str] = []
    for record in attempt_records:
        candidate_code = ""
        candidate_path = Path(record.function_candidate_path)
        if candidate_path.exists():
            candidate_code = candidate_path.read_text(encoding="utf-8")

        status_parts = [f"Status: {record.status}."]
        status_parts.append(f"Compile={'passed' if record.compile_success else 'failed'}.")
        if record.correctness_success:
            status_parts.append("Correctness=passed.")
        elif record.compile_success:
            status_parts.append("Correctness=failed.")
        if record.speedup_vs_baseline is not None:
            status_parts.append(f"Speedup vs baseline={record.speedup_vs_baseline:.4f}x.")
        if record.speedup_vs_module is not None:
            status_parts.append(f"Speedup vs PyTorch={record.speedup_vs_module:.4f}x.")
        if record.candidate_latency_ms is not None:
            status_parts.append(f"Candidate latency={record.candidate_latency_ms:.4f}ms.")
        if record.baseline_latency_ms is not None:
            status_parts.append(f"Baseline HIP latency={record.baseline_latency_ms:.4f}ms.")

        feedback = record.feedback or record.error or record.mismatch or "No feedback captured."
        sections.extend(
            [
                f"Attempt {record.attempt}:",
                " ".join(status_parts),
                "Candidate GPU function:",
                "```cpp",
                _truncate_block(candidate_code or "// Candidate function was not written.", code_char_limit),
                "```",
                "Verifier feedback, error trace, or performance note:",
                "```text",
                _truncate_block(feedback, feedback_char_limit),
                "```",
                "",
            ]
        )
    return "\n".join(sections).strip()


def build_prompt(
    module_code: str,
    functional_code: str,
    hip_code: str,
    relative_path: Path,
    target_function: GPUFunction,
    attempt_records: list[AttemptRecord],
    *,
    system_instruction: str,
    few_shot_examples: str,
    code_char_limit: int,
    feedback_char_limit: int,
    baseline_message: str | None,
    module_latency_ms: float | None,
    baseline_latency_ms: float | None,
) -> str:
    sections = [
        system_instruction.strip(),
        "",
        "Few-shot examples:",
        few_shot_examples.strip(),
        "",
        "Task requirements:",
        "- Optimize exactly one GPU function from the baseline HIP file.",
        "- Return exactly one complete function definition for the selected target function.",
        "- Keep the same public interface so the function body can be transplanted back into the baseline source.",
        "- Do not emit a full HIP file.",
        "- Preserve correctness before pursuing speed.",
        "",
        f"Target sample: {relative_path.as_posix()}",
        f"Selected function: {target_function.name}",
        f"Selected function kind: {target_function.qualifier}",
        f"Selected function body length: {target_function.body_char_length}",
        "",
    ]

    if baseline_message:
        sections.extend(
            [
                "Baseline validation summary:",
                baseline_message.strip(),
                "",
            ]
        )
    if module_latency_ms is not None or baseline_latency_ms is not None:
        latency_parts = []
        if module_latency_ms is not None:
            latency_parts.append(f"PyTorch latency={module_latency_ms:.4f}ms")
        if baseline_latency_ms is not None:
            latency_parts.append(f"Baseline HIP latency={baseline_latency_ms:.4f}ms")
        sections.extend(
            [
                "Baseline performance context:",
                ". ".join(latency_parts) + ".",
                "",
            ]
        )

    sections.extend(
        [
            "Original PyTorch module code:",
            module_code.strip(),
            "",
            "Paired PyTorch functional code:",
            functional_code.strip(),
            "",
            "Full baseline HIP source:",
            hip_code.strip(),
            "",
            "Target GPU function to optimize:",
            target_function.full_text.strip(),
        ]
    )

    best_speedup = _best_speedup_vs_baseline(attempt_records)
    if best_speedup is not None:
        sections.extend(
            [
                "",
                f"Current best validated speedup vs baseline HIP: {best_speedup:.4f}x. Try to beat it without breaking correctness.",
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
                "Previous attempts are included below. Use the candidate function together with compiler errors, correctness failures, and latency feedback to improve the next version:",
                history_section,
            ]
        )

    return "\n".join(sections) + "\n"
