# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TensorStat(BaseModel):
    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float


class EvalRequest(BaseModel):
    kernel_name: str
    hip_code: str
    hip_ref_code: str
    pytorch_module_code: str = ""
    pytorch_functional_code: str = ""
    atol: Optional[float] = 1e-4
    rtol: Optional[float] = 1e-3
    compile_timeout_s: Optional[int] = None
    run_timeout_s: Optional[int] = None


class SingleGPUEvalRequest(EvalRequest):
    gpu_id: int


class EvalResponse(BaseModel):
    kernel_name: str
    compile_ok: bool
    run_ok: bool
    match_ok: bool
    speedup: float
    reason: Optional[str] = None
    stats: Optional[List[TensorStat]] = None
    timing: Optional[Dict[str, Any]] = None


class BatchEvalRequest(BaseModel):
    requests: List[EvalRequest] = Field(default_factory=list)


class BatchEvalResponse(BaseModel):
    responses: List[EvalResponse] = Field(default_factory=list)
    total_time: float
    batch_size: int
