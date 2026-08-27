# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# client_adapter.py
from __future__ import annotations
import json, requests, textwrap
from pydantic import BaseModel
from typing import Optional, List



# ======== 你的数据类 ========
class EvalRequest(BaseModel):
    kernel_name: str
    hip_code: str
    pytorch_module_code: str
    pytorch_functional_code: str
    atol: Optional[float] = 1e-4
    rtol: Optional[float] = 1e-3
    compile_timeout_s: Optional[int] = 10000
    run_timeout_s: Optional[int] = 10000


class TensorStat(BaseModel):
    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float


class EvalResponse(BaseModel):
    kernel_name: str
    compile_ok: bool
    run_ok: bool
    match_ok: bool
    speedup: float
    reason: Optional[str] = None
    stats: Optional[List[TensorStat]] = None


def call_run_code(sf_url: str, req: EvalRequest, timeout_s: int = 1000) -> EvalResponse:
    if not sf_url.endswith("/run_code"):
        raise ValueError(f"Sandbox Fusion URL must end with /run_code: {sf_url}")
    # Sandbox Fusion /run_code 只接受 {"code": "<python>"} 的 JSON。:contentReference[oaicite:1]{index=1}
    response = requests.post(sf_url, json=req, timeout=timeout_s)
    # r = requests.post(sf_url, json=req)
    # print(f'Evalution results:{response}.')
    return response
