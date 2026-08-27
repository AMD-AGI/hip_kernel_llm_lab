# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import pytest

from sandbox_core.config import EvalSettings
from sandbox_core.result import EvalRunResult
from sandbox_core.tool_protocol import (
    KernelReferenceBundle,
    KernelToolActionRequest,
    KernelToolBudget,
    KernelToolCreateSessionRequest,
    KernelToolUpdateCandidateRequest,
)
from sandbox_core.tool_runtime import KernelToolRuntime


def _make_settings(tmp_path):
    return EvalSettings(
        gpu_ids=[0],
        node_id="test-node",
        error_log_dir=str(tmp_path / "error_log"),
        perf_iterations=100,
        speedup_confirm_enabled=False,
        speedup_confirm_threshold=1.05,
        speedup_confirm_band=0.02,
        speedup_confirm_iterations=3000,
        compile_timeout_s=600,
        run_timeout_s=600,
        handler_timeout_s=1200,
        effective_arch="gfx942",
        cache_root=str(tmp_path / "reference_cache"),
        enable_ref_compile_cache=False,
        enable_ref_golden_cache=False,
        enable_ref_perf_cache=False,
        ref_perf_cache_ttl_s=3600,
        cache_golden_on_cpu=True,
        cleanup_tmp_on_success=False,
        retain_tmp_on_failure=True,
    )


@pytest.mark.asyncio
async def test_kernel_tool_runtime_caches_compile_observation(monkeypatch, tmp_path):
    monkeypatch.setenv("HIP_TOOL_GPU_IDS", "0")
    settings = _make_settings(tmp_path)
    runtime = KernelToolRuntime(settings)
    runtime.runtime_root = tmp_path / "tool_sessions"
    runtime.runtime_root.mkdir(parents=True, exist_ok=True)

    async def run_cpu(fn):
        return fn(), {"resource_kind": "cpu", "queue_wait_s": 0.0}

    monkeypatch.setattr(runtime.scheduler, "run_cpu", run_cpu)
    monkeypatch.setattr(
        "sandbox_core.tool_runtime.run_compile_request",
        lambda request, tmp_dir=None, settings=None: EvalRunResult(True, False, False, 0.0, {"total": 0.1}),
    )

    create_resp = await runtime.create_session(
        KernelToolCreateSessionRequest(
            session_id="req-1",
            reference=KernelReferenceBundle(
                problem_id="sample-1",
                kernel_name="test_kernel",
                hip_ref_code="__global__ void test_kernel() {}",
                pytorch_functional_code="def get_inputs():\n    return []",
            ),
            budget=KernelToolBudget(max_tool_calls=2, max_wallclock_s=300),
        )
    )
    assert create_resp.session_id == "req-1"

    update_resp = await runtime.update_candidate(
        KernelToolUpdateCandidateRequest(
            session_id="req-1",
            hip_code="__global__ void test_kernel() {}",
        )
    )
    assert update_resp.updated
    assert os.path.exists(runtime.runtime_root / "req-1" / "artifacts" / update_resp.artifact_id / "candidate.hip")

    first = await runtime.compile_check(KernelToolActionRequest(session_id="req-1"))
    second = await runtime.compile_check(KernelToolActionRequest(session_id="req-1"))

    assert first.compile_ok is True
    assert first.cached is False
    assert second.compile_ok is True
    assert second.cached is True
    assert second.budget["tool_calls_used"] == 2


@pytest.mark.asyncio
async def test_kernel_tool_runtime_enforces_tool_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("HIP_TOOL_GPU_IDS", "0")
    settings = _make_settings(tmp_path)
    runtime = KernelToolRuntime(settings)
    runtime.runtime_root = tmp_path / "tool_sessions"
    runtime.runtime_root.mkdir(parents=True, exist_ok=True)

    async def run_cpu(fn):
        return fn(), {"resource_kind": "cpu", "queue_wait_s": 0.0}

    monkeypatch.setattr(runtime.scheduler, "run_cpu", run_cpu)
    monkeypatch.setattr(
        "sandbox_core.tool_runtime.run_compile_request",
        lambda request, tmp_dir=None, settings=None: EvalRunResult(True, False, False, 0.0, {"total": 0.1}),
    )

    await runtime.create_session(
        KernelToolCreateSessionRequest(
            session_id="req-budget",
            reference=KernelReferenceBundle(
                problem_id="sample-1",
                kernel_name="test_kernel",
                hip_ref_code="__global__ void test_kernel() {}",
                pytorch_functional_code="def get_inputs():\n    return []",
            ),
            budget=KernelToolBudget(max_tool_calls=1, max_wallclock_s=300),
        )
    )
    await runtime.update_candidate(
        KernelToolUpdateCandidateRequest(
            session_id="req-budget",
            hip_code="__global__ void test_kernel() {}",
        )
    )

    ok = await runtime.compile_check(KernelToolActionRequest(session_id="req-budget"))
    rejected = await runtime.compile_check(KernelToolActionRequest(session_id="req-budget"))

    assert ok.status == "ok"
    assert rejected.status == "rejected"
    assert "budget" in rejected.reason.lower()
