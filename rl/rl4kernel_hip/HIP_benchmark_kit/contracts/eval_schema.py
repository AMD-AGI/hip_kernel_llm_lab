# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Evaluation record and backend schema constants."""

from __future__ import annotations

from typing import Iterable, Mapping

SERVER_INPROCESS_BACKEND = "server-inprocess"
SANDBOX_INPROCESS_BACKEND = "sandbox-inprocess"
SUPPORTED_SERVER_BACKENDS = {SERVER_INPROCESS_BACKEND, SANDBOX_INPROCESS_BACKEND}

SUPPORTED_REFERENCE_CACHE_MODES = {"golden-only", "golden+compile", "golden+compile+perf"}

BASELINE_RESULTS_JSON = "baseline_hip_results.json"
BASELINE_RESULTS_CSV = "baseline_hip_results.csv"
COMPARISON_RESULTS_JSON = "origin_vs_optimized_results.json"
COMPARISON_RESULTS_CSV = "origin_vs_optimized_results.csv"
COMPARISON_PERF_TRACE_CSV = "origin_vs_optimized_perf_trace.csv"

LEGACY_EVAL_REQUIRED_FIELDS = (
    "hip_file",
    "base_name",
    "gen_idx",
    "compile_ok",
    "run_ok",
    "match_ok",
    "pytorch_time_ms",
    "hip_time_ms",
    "speedup",
    "error_message",
    "artifact_side",
    "eval_backend",
)


def normalize_eval_backend(value: str) -> str:
    """Normalize compatibility aliases to the only supported execution path."""
    backend = (value or "").strip()
    if backend == SANDBOX_INPROCESS_BACKEND:
        return SERVER_INPROCESS_BACKEND
    return backend


def validate_eval_records(records: Iterable[Mapping[str, object]]) -> None:
    """Fail early when an eval row no longer satisfies the shared JSON contract."""
    missing_by_index: list[str] = []
    for index, record in enumerate(records):
        missing = [field for field in LEGACY_EVAL_REQUIRED_FIELDS if field not in record]
        if missing:
            missing_by_index.append(f"row {index}: {', '.join(missing)}")
    if missing_by_index:
        details = "; ".join(missing_by_index[:5])
        raise ValueError(f"Eval record schema violation: {details}")
