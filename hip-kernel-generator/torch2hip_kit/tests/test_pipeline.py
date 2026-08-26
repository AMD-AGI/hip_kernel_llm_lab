# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from pathlib import Path

import torch2hip_kit.pipeline as pipeline_module
from torch2hip_kit.config import PipelineConfig
from torch2hip_kit.verifier import VerificationResult


MODULE_SOURCE = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        return x + self.bias


bias_value = 2.5


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return [bias_value]
"""


FUNCTIONAL_SOURCE = """
import torch
import torch.nn as nn


def module_fn(x, bias):
    return x + bias


class Model(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x, fn=module_fn):
        return fn(x, self.bias)


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return [2.5]
"""


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, messages, **kwargs):
        return self._responses.pop(0)


def test_convert_single_file_selects_fastest_success(monkeypatch, tmp_path: Path) -> None:
    module_dir = tmp_path / "module"
    functional_dir = tmp_path / "functional"
    output_dir = tmp_path / "output"
    artifacts_dir = tmp_path / "artifacts"
    source_path = module_dir / "level_1" / "sample.py"
    functional_path = functional_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    functional_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(MODULE_SOURCE, encoding="utf-8")
    functional_path.write_text(FUNCTIONAL_SOURCE, encoding="utf-8")

    verification_results = [
        VerificationResult(True, "attempt1", compile_success=True, correctness_success=True, speedup=1.10),
        VerificationResult(False, "attempt2 failed", compile_success=True, correctness_success=False),
        VerificationResult(True, "attempt3", compile_success=True, correctness_success=True, speedup=1.75),
    ]

    def fake_verify_candidate(*args, **kwargs):
        return verification_results.pop(0)

    monkeypatch.setattr(pipeline_module, "verify_candidate", fake_verify_candidate)

    config = PipelineConfig(
        module_dir=module_dir,
        functional_dir=functional_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        max_attempts=3,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    client = FakeClient(
        [
            "```cpp\n// attempt1\n```",
            "```cpp\n// attempt2\n```",
            "```cpp\n// attempt3\n```",
        ]
    )
    record = pipeline_module.convert_single_file(source_path, client, config)

    assert record.status == "success"
    assert record.best_attempt == 3
    assert record.best_speedup == 1.75
    assert record.attempts_used == 3
    assert (output_dir / "level_1" / "sample.hip").read_text(encoding="utf-8") == "// attempt3"

    third_prompt = (artifacts_dir / "prompts" / "level_1" / "sample.attempt_3.txt").read_text(encoding="utf-8")
    assert "Attempt 1:" in third_prompt
    assert "Current best validated speedup: 1.1000x." in third_prompt


def test_run_conversion_pipeline_writes_json_records(monkeypatch, tmp_path: Path) -> None:
    module_dir = tmp_path / "module"
    functional_dir = tmp_path / "functional"
    output_dir = tmp_path / "output"
    artifacts_dir = tmp_path / "artifacts"
    source_path = module_dir / "level_1" / "sample.py"
    functional_path = functional_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    functional_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(MODULE_SOURCE, encoding="utf-8")
    functional_path.write_text(FUNCTIONAL_SOURCE, encoding="utf-8")

    monkeypatch.setattr(
        pipeline_module,
        "verify_candidate",
        lambda *args, **kwargs: VerificationResult(
            True,
            "ok",
            compile_success=True,
            correctness_success=True,
            speedup=1.5,
        ),
    )

    config = PipelineConfig(
        module_dir=module_dir,
        functional_dir=functional_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        max_attempts=1,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    client = FakeClient(["```cpp\n// best\n```"])
    summary = pipeline_module.run_conversion_pipeline(client, config)

    assert summary == {"total": 1, "success": 1, "failed": 0, "skipped": 0}
    records = json.loads(config.records_file.read_text(encoding="utf-8"))
    successes = json.loads(config.success_file.read_text(encoding="utf-8"))
    failures = json.loads(config.failure_file.read_text(encoding="utf-8"))

    assert len(records) == 1
    assert len(successes) == 1
    assert failures == []
    assert successes[0]["best_attempt"] == 1
    assert successes[0]["module_source_path"] == source_path.as_posix()
    assert successes[0]["functional_source_path"] == functional_path.as_posix()
    assert successes[0]["relative_path"] == Path("level_1/sample.py").as_posix()
    assert successes[0]["output_path"] == (output_dir / "level_1" / "sample.hip").as_posix()
    assert successes[0]["attempts"][0]["prompt_path"] == (
        artifacts_dir / "prompts" / "level_1" / "sample.attempt_1.txt"
    ).as_posix()
    assert successes[0]["attempts"][0]["candidate_path"] == (
        artifacts_dir / "candidates" / "level_1" / "sample.attempt_1.hip"
    ).as_posix()


def test_convert_single_file_fails_when_functional_pair_missing(tmp_path: Path) -> None:
    module_dir = tmp_path / "module"
    source_path = module_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(MODULE_SOURCE, encoding="utf-8")

    config = PipelineConfig(
        module_dir=module_dir,
        functional_dir=tmp_path / "functional",
        output_dir=tmp_path / "output",
        artifacts_dir=tmp_path / "artifacts",
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    record = pipeline_module.convert_single_file(source_path, FakeClient([]), config)

    assert record.status == "failed"
    assert "Paired functional file was not found" in (record.final_error or "")
    assert "\\" not in (record.final_error or "")
