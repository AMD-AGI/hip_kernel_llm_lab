# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import AttemptRecord
from .prompt_assets import FEW_SHOT_EXAMPLES, SYSTEM_INSTRUCTION


def format_response(response: str) -> str:
    cleaned = response.strip()
    if "<think>" in cleaned and "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()

    for fence in ("```python", "```py", "```"):
        if fence in cleaned:
            cleaned = cleaned.split(fence, 1)[1]
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]
            cleaned = cleaned.strip()
            break
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
        if record.status == "success":
            continue

        candidate_code = ""
        candidate_path = Path(record.candidate_path)
        if candidate_path.exists():
            candidate_code = candidate_path.read_text(encoding="utf-8")

        feedback = record.feedback or record.error or record.mismatch or "No feedback captured."
        sections.extend(
            [
                f"Attempt {record.attempt}:",
                "Candidate code:",
                "```python",
                _truncate_block(candidate_code or "# Candidate code was not written.", code_char_limit),
                "```",
                "Observed failure:",
                "```text",
                _truncate_block(feedback, feedback_char_limit),
                "```",
                "",
            ]
        )
    return "\n".join(sections).strip()


def build_prompt(
    module_code: str,
    relative_path: Path,
    attempt_records: list[AttemptRecord],
    *,
    code_char_limit: int,
    feedback_char_limit: int,
) -> str:
    sections = [
        SYSTEM_INSTRUCTION.strip(),
        "",
        "Few-shot examples:",
        FEW_SHOT_EXAMPLES.strip(),
        "",
        f"Target file: {relative_path.as_posix()}",
        "",
        "Original PyTorch module code:",
        module_code.strip(),
    ]
    history_section = _render_attempt_history(
        attempt_records,
        code_char_limit=code_char_limit,
        feedback_char_limit=feedback_char_limit,
    )
    if history_section:
        sections.extend(
            [
                "",
                "Previous failed attempts are included below. Reuse what is correct, fix what is wrong, and avoid repeating the same mistakes:",
                history_section,
            ]
        )
    return "\n".join(sections) + "\n"
