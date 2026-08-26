# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

from torch_modu2func_kit.verifier import verify_candidate


ORIGINAL_CODE = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        return x + self.bias


bias_value = 3.0


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return [bias_value]
"""


GENERATED_CODE = """
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


bias_value = 3.0


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return [bias_value]
"""


def test_verify_candidate_accepts_equivalent_code(tmp_path: Path) -> None:
    original_path = tmp_path / "original.py"
    generated_path = tmp_path / "generated.py"
    original_path.write_text(ORIGINAL_CODE, encoding="utf-8")
    generated_path.write_text(GENERATED_CODE, encoding="utf-8")

    result = verify_candidate(
        original_path,
        generated_path,
        seed=1234,
        rtol=1e-4,
        atol=1e-4,
    )

    assert result.success is True
