# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import py_hip_kernel2kernel_kit.verifier as verifier_module
from py_hip_kernel2kernel_kit.verifier import VerificationContext, VerificationTimeoutError


def _make_context() -> VerificationContext:
    def functional_model(*args, **kwargs):
        return 1

    return VerificationContext(
        functional_model=functional_model,
        original_model=None,
        expected_output=1,
        forward_args=[],
        forward_kwargs={},
        seed=1234,
        rtol=1e-4,
        atol=1e-4,
        perf_warmup=1,
        perf_iterations=2,
        module_latency_ms=3.0,
        baseline_latency_ms=2.0,
        offload_arch=None,
        hip_compile_timeout_seconds=1.0,
        execution_timeout_seconds=1.0,
        benchmark_timeout_seconds=1.0,
    )


def test_verify_candidate_reports_compile_timeout(monkeypatch, tmp_path: Path) -> None:
    context = _make_context()

    monkeypatch.setattr(
        verifier_module,
        "load_hip_forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            VerificationTimeoutError("compile candidate HIP extension", 1.0)
        ),
    )

    result = context.verify_candidate(
        tmp_path / "candidate.hip",
        build_dir=tmp_path / "build",
    )

    assert result.success is False
    assert result.compile_success is False
    assert result.correctness_success is False
    assert "compile candidate HIP extension timed out" in result.message


def test_verify_candidate_reports_benchmark_timeout_after_correctness(monkeypatch, tmp_path: Path) -> None:
    context = _make_context()

    monkeypatch.setattr(verifier_module, "load_hip_forward", lambda *args, **kwargs: lambda *a, **k: None)
    monkeypatch.setattr(
        verifier_module,
        "_measure_latency_ms",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            VerificationTimeoutError("benchmark candidate HIP function", 1.0)
        ),
    )

    result = context.verify_candidate(
        tmp_path / "candidate.hip",
        build_dir=tmp_path / "build",
    )

    assert result.success is False
    assert result.compile_success is True
    assert result.correctness_success is True
    assert "benchmark candidate HIP function timed out" in result.message
