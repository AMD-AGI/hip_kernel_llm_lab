# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Compatibility wrappers for parallel HIP-to-HIP evaluation utilities.

The implementation now lives in `eval_core.py`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sandbox_core.config import load_eval_settings
from sandbox_core.eval import evaluate_requests_parallel
from sandbox_core.protocol import EvalRequest
from .hip2hip import (
    _compare_results,
    clear_pts,
    construct_pytorch_functional_unittest,
    construct_pytorch_module_unittest,
    extract_module_name,
    perf_call_and_exec_hip2hip,
    perf_compile,
)


def perf_call_and_exec_hip2hip_parallel(
    kernel_tasks: List[Dict[str, Any]],
    max_workers: int = None,
    base_tmp_dir: str = "hip_eval_parallel",
    rtol: float = 1e-4,
    atol: float = 1e-5,
    gpu_ids: List[int] = None,
    error_log_dir: str = "./error_log",
    perf_iterations: int = 100,
) -> Dict[str, Any]:
    settings = load_eval_settings(
        perf_iterations=perf_iterations,
        error_log_dir=error_log_dir,
        gpu_ids=gpu_ids,
    )
    requests = [
        EvalRequest(
            kernel_name=task["kernel_name"],
            hip_code=task["hip_code"],
            hip_ref_code=task.get("hip_ref_code", ""),
            pytorch_module_code=task.get("pytorch_module_code", ""),
            pytorch_functional_code=task.get("pytorch_functional_code", ""),
            rtol=float(task.get("rtol", rtol)),
            atol=float(task.get("atol", atol)),
            compile_timeout_s=task.get("compile_timeout_s"),
            run_timeout_s=task.get("run_timeout_s"),
        )
        for task in kernel_tasks
    ]
    results = evaluate_requests_parallel(
        requests,
        max_workers=max_workers,
        base_tmp_dir=base_tmp_dir,
        gpu_ids=gpu_ids,
        error_log_dir=error_log_dir,
        settings=settings,
    )
    return {
        request.kernel_name: result
        for request, result in zip(requests, results)
    }
