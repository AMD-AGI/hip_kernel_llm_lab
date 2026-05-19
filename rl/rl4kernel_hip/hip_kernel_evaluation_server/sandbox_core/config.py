from __future__ import annotations

import os
import platform
import socket
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch

BASE_DIR = Path(__file__).resolve().parent.parent


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def parse_gpu_ids(raw: Optional[str] = None) -> List[int]:
    source = raw if raw is not None else os.environ.get("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    return [int(x.strip()) for x in source.split(",") if x.strip()]


def resolve_effective_arch(explicit_arch: Optional[str] = None) -> str:
    for candidate in (
        explicit_arch,
        os.environ.get("HIP_EVAL_ARCH"),
        os.environ.get("HCC_AMDGPU_TARGET"),
        os.environ.get("AMDGPU_TARGETS"),
        os.environ.get("GPU_ARCHS"),
        os.environ.get("PYTORCH_ROCM_ARCH"),
        "gfx942",
    ):
        if not candidate:
            continue
        if "," in candidate:
            first = next((part.strip() for part in candidate.split(",") if part.strip()), "")
            if first:
                return first
            continue
        return candidate.strip()
    return "gfx942"


def build_software_stack_fingerprint(effective_arch: str) -> Dict[str, Any]:
    return {
        "arch": effective_arch,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_hip_version": getattr(torch.version, "hip", None),
    }


def build_compile_identity(effective_arch: str) -> Dict[str, Any]:
    return {
        **build_software_stack_fingerprint(effective_arch),
        "hostname": socket.gethostname(),
        "node_id": os.environ.get("NODE_ID", socket.gethostname()),
    }


def build_runtime_fingerprint(gpu_id: Optional[int], effective_arch: str) -> Dict[str, Any]:
    fingerprint: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "node_id": os.environ.get("NODE_ID", socket.gethostname()),
        "arch": effective_arch,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_hip_version": getattr(torch.version, "hip", None),
        "gpu_id": gpu_id,
    }
    try:
        if torch.cuda.is_available():
            visible_index = 0
            if gpu_id is not None and torch.cuda.device_count() > 1:
                visible_index = min(gpu_id, torch.cuda.device_count() - 1)
            fingerprint["gpu_name"] = torch.cuda.get_device_name(visible_index)
            props = torch.cuda.get_device_properties(visible_index)
            fingerprint["gpu_total_memory"] = int(props.total_memory)
            fingerprint["gpu_multiprocessor_count"] = int(getattr(props, "multi_processor_count", 0))
    except Exception as exc:  # pragma: no cover - best effort only
        fingerprint["gpu_probe_error"] = str(exc)
    return fingerprint


@dataclass(frozen=True)
class EvalSettings:
    gpu_ids: List[int]
    node_id: str
    error_log_dir: str
    perf_iterations: int
    speedup_confirm_enabled: bool
    speedup_confirm_threshold: float
    speedup_confirm_band: float
    speedup_confirm_iterations: int
    compile_timeout_s: int
    run_timeout_s: int
    handler_timeout_s: int
    effective_arch: str
    cache_root: str
    enable_ref_compile_cache: bool
    enable_ref_golden_cache: bool
    enable_ref_perf_cache: bool
    ref_perf_cache_ttl_s: int
    cache_golden_on_cpu: bool
    cleanup_tmp_on_success: bool
    retain_tmp_on_failure: bool
    compile_cpu_slots: int = 16
    compile_inner_jobs: int = 4
    omp_num_threads: int = 1
    mkl_num_threads: int = 1
    openblas_num_threads: int = 1
    enable_two_stage_batch: bool = True


def load_eval_settings(
    *,
    perf_iterations: Optional[int] = None,
    speedup_confirm_enabled: Optional[bool] = None,
    speedup_confirm_threshold: Optional[float] = None,
    speedup_confirm_band: Optional[float] = None,
    speedup_confirm_iterations: Optional[int] = None,
    error_log_dir: Optional[str] = None,
    explicit_arch: Optional[str] = None,
    gpu_ids: Optional[List[int]] = None,
    enable_ref_compile_cache: Optional[bool] = None,
    enable_ref_golden_cache: Optional[bool] = None,
    enable_ref_perf_cache: Optional[bool] = None,
    ref_perf_cache_ttl_s: Optional[int] = None,
) -> EvalSettings:
    resolved_gpu_ids = gpu_ids if gpu_ids is not None else parse_gpu_ids()
    effective_arch = resolve_effective_arch(explicit_arch)
    resolved_perf_iterations = perf_iterations if perf_iterations is not None else _parse_int_env("HIP_PERF_ITERATIONS", 1000)
    resolved_confirm_iterations = (
        speedup_confirm_iterations
        if speedup_confirm_iterations is not None
        else _parse_int_env("HIP_CONFIRM_PERF_ITERATIONS", max(3000, resolved_perf_iterations))
    )
    cpu_count = os.cpu_count() or 1
    default_compile_slots = max(1, min(16, cpu_count))
    return EvalSettings(
        gpu_ids=resolved_gpu_ids,
        node_id=os.environ.get("NODE_ID", socket.gethostname()),
        error_log_dir=os.path.abspath(error_log_dir or os.environ.get("HIP_ERROR_LOG_DIR", str(BASE_DIR / "runtime" / "error_log"))),
        perf_iterations=resolved_perf_iterations,
        speedup_confirm_enabled=(
            speedup_confirm_enabled
            if speedup_confirm_enabled is not None
            else _parse_bool_env("HIP_CONFIRM_SPEEDUP_ENABLED", False)
        ),
        speedup_confirm_threshold=(
            max(0.0, float(speedup_confirm_threshold))
            if speedup_confirm_threshold is not None
            else max(0.0, _parse_float_env("HIP_CONFIRM_SPEEDUP_THRESHOLD", 1.05))
        ),
        speedup_confirm_band=(
            max(0.0, float(speedup_confirm_band))
            if speedup_confirm_band is not None
            else max(0.0, _parse_float_env("HIP_CONFIRM_SPEEDUP_BAND", 0.02))
        ),
        speedup_confirm_iterations=max(resolved_perf_iterations, int(resolved_confirm_iterations)),
        compile_timeout_s=_parse_int_env("HIP_COMPILE_TIMEOUT_S", 600),
        run_timeout_s=_parse_int_env("HIP_RUN_TIMEOUT_S", 600),
        handler_timeout_s=_parse_int_env("HIP_HANDLER_TIMEOUT_S", 1200),
        effective_arch=effective_arch,
        cache_root=os.path.abspath(os.environ.get("HIP_REFERENCE_CACHE_DIR", str(BASE_DIR / "runtime" / "reference_cache"))),
        enable_ref_compile_cache=(
            enable_ref_compile_cache
            if enable_ref_compile_cache is not None
            else _parse_bool_env("HIP_ENABLE_REF_COMPILE_CACHE", False)
        ),
        enable_ref_golden_cache=(
            enable_ref_golden_cache
            if enable_ref_golden_cache is not None
            else _parse_bool_env("HIP_ENABLE_REF_GOLDEN_CACHE", False)
        ),
        enable_ref_perf_cache=(
            enable_ref_perf_cache
            if enable_ref_perf_cache is not None
            else _parse_bool_env("HIP_ENABLE_REF_PERF_CACHE", False)
        ),
        ref_perf_cache_ttl_s=(
            max(1, int(ref_perf_cache_ttl_s))
            if ref_perf_cache_ttl_s is not None
            else max(1, _parse_int_env("HIP_REF_PERF_CACHE_TTL_S", 3600))
        ),
        cache_golden_on_cpu=_parse_bool_env("HIP_CACHE_GOLDEN_ON_CPU", True),
        cleanup_tmp_on_success=_parse_bool_env("HIP_CLEANUP_TMP_ON_SUCCESS", False),
        retain_tmp_on_failure=_parse_bool_env("HIP_RETAIN_TMP_ON_FAILURE", True),
        compile_cpu_slots=max(1, _parse_int_env("HIP_COMPILE_CPU_SLOTS", default_compile_slots)),
        compile_inner_jobs=max(1, _parse_int_env("HIP_COMPILE_INNER_JOBS", 4)),
        omp_num_threads=max(1, _parse_int_env("OMP_NUM_THREADS", 1)),
        mkl_num_threads=max(1, _parse_int_env("MKL_NUM_THREADS", 1)),
        openblas_num_threads=max(1, _parse_int_env("OPENBLAS_NUM_THREADS", 1)),
        enable_two_stage_batch=_parse_bool_env("HIP_ENABLE_TWO_STAGE_BATCH", True),
    )


def eval_settings_from_payload(payload: Optional[Dict[str, Any]]) -> EvalSettings:
    defaults = asdict(load_eval_settings())
    merged = {**defaults, **(payload or {})}
    return EvalSettings(**merged)
