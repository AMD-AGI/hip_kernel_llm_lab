# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import datetime
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SERVER_ROOT = os.path.join(REPO_ROOT, "hip_kernel_evaluation_server")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from HIP_benchmark_kit.contracts.eval_schema import (
    LEGACY_EVAL_REQUIRED_FIELDS,
    SANDBOX_INPROCESS_BACKEND,
    SERVER_INPROCESS_BACKEND,
    SUPPORTED_REFERENCE_CACHE_MODES,
    SUPPORTED_SERVER_BACKENDS,
    normalize_eval_backend,
    validate_eval_records,
)


def _ensure_server_path() -> None:
    if SERVER_ROOT not in sys.path:
        sys.path.insert(0, SERVER_ROOT)


def parse_hip_filename(hip_file: str) -> Tuple[str, Optional[int]]:
    match = re.match(r"^(.+)_gen(\d+)_hip\.hip$", hip_file)
    if match:
        return match.group(1), int(match.group(2))

    match = re.match(r"^(.+)_gen(\d+)\.hip$", hip_file)
    if match:
        base_name = match.group(1)
        if base_name.endswith("_hip"):
            base_name = base_name[:-4]
        return base_name, int(match.group(2))

    if hip_file.endswith(".hip"):
        base_name = hip_file[:-4]
        if base_name.endswith("_hip"):
            base_name = base_name[:-4]
        return base_name, None
    return hip_file, None


def reference_python_file(code_dir: str, base_name: str) -> str:
    return os.path.join(code_dir, base_name.replace("hip_", "py_") + ".py")


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def has_valid_perf_value(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0


def _default_env(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value


def _cache_flags_from_mode(mode: str) -> Tuple[bool, bool, bool]:
    if mode not in SUPPORTED_REFERENCE_CACHE_MODES:
        raise ValueError(
            f"Unsupported reference cache mode: {mode}. "
            f"Supported values: {', '.join(sorted(SUPPORTED_REFERENCE_CACHE_MODES))}"
        )
    return {
        "golden-only": (False, True, False),
        "golden+compile": (True, True, False),
        "golden+compile+perf": (True, True, True),
    }[mode]


def build_sandbox_settings(
    *,
    gpu_ids: Sequence[int],
    error_log_dir: str,
    cache_root: str,
    perf_iterations: int,
    reference_cache_mode: str = "golden+compile",
    disable_compile_cache: bool = False,
):
    """Build EvalSettings using the same operational defaults as the batch server launcher."""
    _ensure_server_path()
    from sandbox_core.config import load_eval_settings

    gpu_text = ",".join(str(gpu) for gpu in gpu_ids)
    _default_env("HCC_AMDGPU_TARGET", "gfx942")
    _default_env("AMDGPU_TARGETS", os.environ["HCC_AMDGPU_TARGET"])
    _default_env("HIP_EVAL_ARCH", os.environ["HCC_AMDGPU_TARGET"])
    os.environ["HIP_VISIBLE_DEVICES"] = gpu_text

    os.environ["HIP_ERROR_LOG_DIR"] = os.path.abspath(error_log_dir)
    os.environ["HIP_REFERENCE_CACHE_DIR"] = os.path.abspath(cache_root)
    os.environ["HIP_REFERENCE_CACHE_MODE"] = reference_cache_mode
    _default_env("HIP_REF_PERF_CACHE_TTL_S", "3600")

    os.environ["HIP_PERF_ITERATIONS"] = str(int(perf_iterations))
    _default_env("HIP_COMPILE_TIMEOUT_S", "600")
    _default_env("HIP_RUN_TIMEOUT_S", "600")
    _default_env("HIP_CONFIRM_SPEEDUP_ENABLED", "0")
    _default_env("HIP_CONFIRM_SPEEDUP_THRESHOLD", "1.05")
    _default_env("HIP_CONFIRM_SPEEDUP_BAND", "0.02")
    _default_env("HIP_CONFIRM_PERF_ITERATIONS", "3000")

    _default_env("HIP_COMPILE_CPU_SLOTS", "16")
    _default_env("HIP_COMPILE_INNER_JOBS", "4")
    _default_env("HIP_ENABLE_TWO_STAGE_BATCH", "1")
    _default_env("OMP_NUM_THREADS", "1")
    _default_env("MKL_NUM_THREADS", "1")
    _default_env("OPENBLAS_NUM_THREADS", "1")
    os.environ["MAX_JOBS"] = os.environ["HIP_COMPILE_INNER_JOBS"]
    os.environ["HIP_CACHE_GOLDEN_ON_CPU"] = "1"

    enable_compile, enable_golden, enable_perf = _cache_flags_from_mode(reference_cache_mode)
    if disable_compile_cache:
        enable_compile = False
    os.environ["HIP_ENABLE_REF_COMPILE_CACHE"] = "1" if enable_compile else "0"
    os.environ["HIP_ENABLE_REF_GOLDEN_CACHE"] = "1" if enable_golden else "0"
    os.environ["HIP_ENABLE_REF_PERF_CACHE"] = "1" if enable_perf else "0"

    return load_eval_settings(
        perf_iterations=int(perf_iterations),
        error_log_dir=os.path.abspath(error_log_dir),
        gpu_ids=list(gpu_ids),
        enable_ref_compile_cache=enable_compile,
        enable_ref_golden_cache=enable_golden,
        enable_ref_perf_cache=enable_perf,
    )


@dataclass(frozen=True)
class EvalInputManifest:
    hip_file: str
    base_name: str
    gen_idx: Optional[int]
    candidate_path: str
    reference_path: str
    pytorch_functional_path: str
    pytorch_module_path: str


def build_input_manifest(
    *,
    hip_file: str,
    hip_code_dir: str,
    reference_hip_code_dir: str,
    pytorch_func_dir: str,
    pytorch_modu_dir: str,
) -> EvalInputManifest:
    base_name, gen_idx = parse_hip_filename(hip_file)
    manifest = EvalInputManifest(
        hip_file=hip_file,
        base_name=base_name,
        gen_idx=gen_idx,
        candidate_path=os.path.abspath(os.path.join(hip_code_dir, hip_file)),
        reference_path=os.path.abspath(os.path.join(reference_hip_code_dir, f"{base_name}.hip")),
        pytorch_functional_path=os.path.abspath(reference_python_file(pytorch_func_dir, base_name)),
        pytorch_module_path=os.path.abspath(reference_python_file(pytorch_modu_dir, base_name)),
    )
    for path in (
        manifest.candidate_path,
        manifest.reference_path,
        manifest.pytorch_functional_path,
        manifest.pytorch_module_path,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required server eval input missing: {path}")
    return manifest


def build_eval_request(manifest: EvalInputManifest, *, rtol: float, atol: float):
    _ensure_server_path()
    from sandbox_core.protocol import EvalRequest

    return EvalRequest(
        kernel_name=manifest.base_name,
        hip_code=read_text_file(manifest.candidate_path),
        hip_ref_code=read_text_file(manifest.reference_path),
        pytorch_module_code=read_text_file(manifest.pytorch_module_path),
        pytorch_functional_code=read_text_file(manifest.pytorch_functional_path),
        atol=float(atol),
        rtol=float(rtol),
    )


def _failure_message(timing: Dict[str, Any]) -> Optional[str]:
    reason = timing.get("failure_reason") or timing.get("failure_stage")
    detail = timing.get("failure_detail")
    if reason and detail:
        return f"{reason}: {detail}"
    if reason:
        return str(reason)
    return None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=True)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)


def map_result_to_legacy_record(
    *,
    manifest: EvalInputManifest,
    result: Any,
    artifact_side: str,
    eval_backend: str,
) -> Dict[str, Any]:
    timing = dict(result.timing or {})
    candidate_perf = timing.get("candidate_perf_ms")
    reference_perf = timing.get("reference_perf_ms")
    speedup = result.speedup if result.speedup else None
    if speedup is None and has_valid_perf_value(reference_perf) and has_valid_perf_value(candidate_perf):
        speedup = float(reference_perf) / float(candidate_perf)

    row: Dict[str, Any] = {
        "hip_file": manifest.hip_file,
        "base_name": manifest.base_name,
        "gen_idx": manifest.gen_idx,
        "compile_ok": bool(result.compile_ok),
        "run_ok": bool(result.run_ok),
        "match_ok": bool(result.match_ok),
        "pytorch_time_ms": float(reference_perf) if has_valid_perf_value(reference_perf) else None,
        "hip_time_ms": float(candidate_perf) if has_valid_perf_value(candidate_perf) else None,
        "speedup": float(speedup) if has_valid_perf_value(speedup) else None,
        "module_perf_enabled": False,
        "hip_perf_enabled": True,
        "compile_cache_key": timing.get("reference_compile_cache_key"),
        "compile_cache_hit": timing.get("reference_compile_cache_hit"),
        "compile_cache_enabled": timing.get("reference_compile_cache_enabled"),
        "compile_artifact_path": timing.get("candidate_build_directory"),
        "compiled_library_path": timing.get("candidate_library_path"),
        "compiled_module_name": timing.get("candidate_module_name"),
        "artifact_side": artifact_side,
        "perf_gpu_id": timing.get("assigned_gpu_id"),
        "perf_started_at": timing.get("perf_started_at"),
        "perf_finished_at": timing.get("perf_finished_at"),
        "error_message": _failure_message(timing),
        "eval_backend": eval_backend,
        "sandbox_timing": _json_safe(timing),
    }
    validate_legacy_eval_record(row)
    return row


def validate_legacy_eval_record(row: Dict[str, Any]) -> None:
    validate_eval_records([row])
    for field in ("hip_file", "base_name", "artifact_side", "eval_backend"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"Legacy eval row field {field!r} must be a non-empty string")
    if row["gen_idx"] is not None and not isinstance(row["gen_idx"], int):
        raise ValueError("Legacy eval row gen_idx must be int or None")
    for field in ("compile_ok", "run_ok", "match_ok"):
        if not isinstance(row[field], bool):
            raise ValueError(f"Legacy eval row field {field!r} must be bool")
    for field in ("pytorch_time_ms", "hip_time_ms", "speedup"):
        value = row[field]
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"Legacy eval row field {field!r} must be numeric or None")
    json.dumps(row, ensure_ascii=False)


def evaluate_hip_files_with_server(
    *,
    hip_file_list: Sequence[str],
    hip_code_dir: str,
    reference_hip_code_dir: str,
    pytorch_func_dir: str,
    pytorch_modu_dir: str,
    rtol: float,
    atol: float,
    perf_iterations: int,
    gpu_ids: Sequence[int],
    error_log_dir: str,
    runtime_dir: str,
    max_workers: Optional[int],
    artifact_side: str,
    cache_root: str,
    reference_cache_mode: str,
    disable_compile_cache: bool,
    eval_backend: str = SERVER_INPROCESS_BACKEND,
) -> List[Dict[str, Any]]:
    _ensure_server_path()
    from sandbox_core.eval import evaluate_requests_parallel

    manifests = [
        build_input_manifest(
            hip_file=hip_file,
            hip_code_dir=os.path.abspath(hip_code_dir),
            reference_hip_code_dir=os.path.abspath(reference_hip_code_dir),
            pytorch_func_dir=os.path.abspath(pytorch_func_dir),
            pytorch_modu_dir=os.path.abspath(pytorch_modu_dir),
        )
        for hip_file in hip_file_list
    ]
    requests = [build_eval_request(manifest, rtol=rtol, atol=atol) for manifest in manifests]
    settings = build_sandbox_settings(
        gpu_ids=[int(gpu) for gpu in gpu_ids if gpu is not None],
        error_log_dir=os.path.abspath(error_log_dir),
        cache_root=os.path.abspath(cache_root),
        perf_iterations=int(perf_iterations),
        reference_cache_mode=reference_cache_mode,
        disable_compile_cache=disable_compile_cache,
    )
    results = evaluate_requests_parallel(
        requests,
        max_workers=max_workers,
        base_tmp_dir=os.path.join(os.path.abspath(runtime_dir), "server_inprocess"),
        gpu_ids=settings.gpu_ids,
        error_log_dir=os.path.abspath(error_log_dir),
        settings=settings,
    )
    rows = [
        map_result_to_legacy_record(
            manifest=manifest,
            result=result,
            artifact_side=artifact_side,
            eval_backend=eval_backend,
        )
        for manifest, result in zip(manifests, results)
    ]
    return rows


def validate_legacy_eval_records(rows: Iterable[Dict[str, Any]]) -> None:
    for row in rows:
        validate_legacy_eval_record(row)
