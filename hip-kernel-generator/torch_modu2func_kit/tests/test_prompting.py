# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

from torch_modu2func_kit.config import AttemptRecord
from torch_modu2func_kit.prompting import build_prompt


def test_build_prompt_includes_failed_attempt_history(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text("def module_fn(x):\n    return x - 1\n", encoding="utf-8")

    attempt_records = [
        AttemptRecord(
            attempt=1,
            prompt_path=(tmp_path / "prompt_1.txt").as_posix(),
            candidate_path=candidate_path.as_posix(),
            status="failed",
            feedback="AssertionError: output tensor mismatch",
            mismatch="AssertionError: output tensor mismatch",
        )
    ]

    prompt = build_prompt(
        "class Model:\n    pass\n",
        Path("level_1/sample.py"),
        attempt_records,
        code_char_limit=1000,
        feedback_char_limit=1000,
    )

    assert "Previous failed attempts are included below" in prompt
    assert "Attempt 1:" in prompt
    assert "return x - 1" in prompt
    assert "AssertionError: output tensor mismatch" in prompt


def test_build_prompt_truncates_long_history_blocks(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text("x" * 200, encoding="utf-8")

    attempt_records = [
        AttemptRecord(
            attempt=1,
            prompt_path=(tmp_path / "prompt_1.txt").as_posix(),
            candidate_path=candidate_path.as_posix(),
            status="failed",
            feedback="y" * 200,
        )
    ]

    prompt = build_prompt(
        "class Model:\n    pass\n",
        Path("level_1/sample.py"),
        attempt_records,
        code_char_limit=50,
        feedback_char_limit=50,
    )

    assert "... [truncated]" in prompt
