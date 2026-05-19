"""
Compatibility wrappers for HIP-to-HIP evaluation utilities.

The implementation now lives in `eval_core.py`.
"""

from __future__ import annotations

from sandbox_core.config import load_eval_settings
from sandbox_core.eval import (
    _compare_results,
    clear_pts,
    construct_pytorch_functional_unittest,
    construct_pytorch_module_unittest,
    extract_module_name,
    perf_compile,
    run_eval_request,
)
from sandbox_core.protocol import EvalRequest


def perf_call_and_exec_hip2hip(
    kernel_name,
    hip_code,
    hip_ref_code,
    pytorch_functional_code,
    tmp_dir=None,
    rtol=1e-4,
    atol=1e-5,
    gpu_id=None,
    error_log_file=None,
    perf_iterations=100,
    pytorch_module_code="",
    compile_timeout_s=None,
    run_timeout_s=None,
):
    request = EvalRequest(
        kernel_name=kernel_name,
        hip_code=hip_code,
        hip_ref_code=hip_ref_code,
        pytorch_module_code=pytorch_module_code or "",
        pytorch_functional_code=pytorch_functional_code,
        rtol=rtol,
        atol=atol,
        compile_timeout_s=compile_timeout_s,
        run_timeout_s=run_timeout_s,
    )
    settings = load_eval_settings(perf_iterations=perf_iterations)
    return run_eval_request(
        request,
        tmp_dir=tmp_dir,
        gpu_id=gpu_id,
        error_log_file=error_log_file,
        settings=settings,
    )
