# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from pathlib import Path

from torch_modu2func_kit.config import PipelineConfig
from torch_modu2func_kit.pipeline import convert_single_file, run_conversion_pipeline


SOURCE_CODE = """
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


BAD_GENERATION = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        return x - self.bias


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return [2.5]
"""


GOOD_GENERATION = """
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


bias_value = 2.5


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return [bias_value]
"""


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, messages, **kwargs):
        return self._responses.pop(0)


def test_convert_single_file_retries_until_success(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    artifacts_dir = tmp_path / "artifacts"
    source_path = input_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(SOURCE_CODE, encoding="utf-8")

    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        max_attempts=2,
    ).with_defaults()

    client = FakeClient([BAD_GENERATION, GOOD_GENERATION])
    record = convert_single_file(source_path, client, config)

    assert record.status == "success"
    assert record.attempts_used == 2
    assert (output_dir / "level_1" / "sample.py").exists()
    assert len(record.attempts) == 2
    assert record.attempts[0].status == "failed"
    assert record.attempts[1].status == "success"
    assert record.attempts[0].feedback is not None

    second_prompt = (artifacts_dir / "prompts" / "level_1" / "sample.attempt_2.txt").read_text(encoding="utf-8")
    assert "Attempt 1:" in second_prompt
    assert "return x - self.bias" in second_prompt
    assert "Observed failure:" in second_prompt
    assert record.attempts[0].feedback in second_prompt


def test_run_conversion_pipeline_writes_json_records(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    artifacts_dir = tmp_path / "artifacts"
    source_path = input_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(SOURCE_CODE, encoding="utf-8")

    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        max_attempts=1,
    ).with_defaults()

    client = FakeClient([GOOD_GENERATION])
    summary = run_conversion_pipeline(client, config)

    assert summary == {"total": 1, "success": 1, "failed": 0, "skipped": 0}

    records = json.loads(config.records_file.read_text(encoding="utf-8"))
    successes = json.loads(config.success_file.read_text(encoding="utf-8"))
    failures = json.loads(config.failure_file.read_text(encoding="utf-8"))

    assert len(records) == 1
    assert len(successes) == 1
    assert failures == []
    assert successes[0]["attempts"][0]["status"] == "success"
    assert "feedback" in successes[0]["attempts"][0]
    assert successes[0]["source_path"] == source_path.as_posix()
    assert successes[0]["relative_path"] == Path("level_1/sample.py").as_posix()
    assert successes[0]["output_path"] == (output_dir / "level_1" / "sample.py").as_posix()
    assert successes[0]["attempts"][0]["prompt_path"] == (
        artifacts_dir / "prompts" / "level_1" / "sample.attempt_1.txt"
    ).as_posix()
    assert successes[0]["attempts"][0]["candidate_path"] == (
        artifacts_dir / "candidates" / "level_1" / "sample.attempt_1.py"
    ).as_posix()
