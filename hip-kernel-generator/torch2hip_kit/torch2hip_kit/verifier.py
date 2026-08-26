# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import copy
import math
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch

from .hip_runtime import cleanup_dir, load_hip_forward, load_python_module

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

REQUIRED_ORIGINAL_EXPORTS = ("Model", "get_inputs", "get_init_inputs")
REQUIRED_FUNCTIONAL_EXPORTS = ("module_fn", "Model", "get_inputs", "get_init_inputs")


def _path_text(path: Path) -> str:
    return path.as_posix()


@dataclass(slots=True)
class VerificationResult:
    success: bool
    message: str
    compile_success: bool = False
    correctness_success: bool = False
    speedup: float | None = None
    module_latency_ms: float | None = None
    hip_latency_ms: float | None = None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _ensure_required_exports(module: ModuleType, module_path: Path, required_exports: tuple[str, ...]) -> None:
    for export_name in required_exports:
        if not hasattr(module, export_name):
            raise AttributeError(f"{_path_text(module_path)} is missing required export `{export_name}`.")


def _normalize_call_args(value: Any) -> tuple[list[Any], dict[str, Any]]:
    if value is None:
        return [], {}
    if isinstance(value, dict):
        return [], _clone_value(value)
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], list) and isinstance(value[1], dict):
        return _clone_value(value[0]), _clone_value(value[1])
    if isinstance(value, tuple):
        return list(_clone_value(value)), {}
    if isinstance(value, list):
        return _clone_value(value), {}
    return [_clone_value(value)], {}


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _compare_scalars(expected: Any, actual: Any, rtol: float, atol: float) -> tuple[bool, str]:
    if isinstance(expected, bool) and isinstance(actual, bool):
        return expected == actual, f"Boolean mismatch: expected {expected}, got {actual}."
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isclose(float(expected), float(actual), rel_tol=rtol, abs_tol=atol):
            return True, ""
        return False, f"Scalar mismatch: expected {expected}, got {actual}."
    if expected == actual:
        return True, ""
    return False, f"Value mismatch: expected {expected!r}, got {actual!r}."


def _compare_outputs(expected: Any, actual: Any, rtol: float, atol: float, path: str = "output") -> tuple[bool, str]:
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        expected_cpu = expected.detach().cpu()
        actual_cpu = actual.detach().cpu()
        matches = torch.allclose(expected_cpu, actual_cpu, rtol=rtol, atol=atol, equal_nan=True)
        if matches:
            return True, ""
        max_abs = torch.max(torch.abs(expected_cpu - actual_cpu)).item()
        return False, f"{path} tensor mismatch: max abs diff={max_abs:.6g}."

    if type(expected) is not type(actual):
        return False, f"{path} type mismatch: expected {type(expected).__name__}, got {type(actual).__name__}."

    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return False, f"{path} length mismatch: expected {len(expected)}, got {len(actual)}."
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            matches, detail = _compare_outputs(expected_item, actual_item, rtol, atol, f"{path}[{index}]")
            if not matches:
                return False, detail
        return True, ""

    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return False, f"{path} dict keys mismatch: expected {sorted(expected.keys())}, got {sorted(actual.keys())}."
        for key in expected:
            matches, detail = _compare_outputs(expected[key], actual[key], rtol, atol, f"{path}[{key!r}]")
            if not matches:
                return False, detail
        return True, ""

    matches, detail = _compare_scalars(expected, actual, rtol, atol)
    if matches:
        return True, ""
    return False, f"{path} {detail}"


def _measure_latency_ms(fn: Callable[[], Any], *, warmup: int, iterations: int) -> float:
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")

    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    with torch.no_grad():
        for _ in range(iterations):
            fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def verify_candidate(
    original_module_path: Path,
    functional_module_path: Path,
    candidate_hip_path: Path,
    *,
    build_dir: Path,
    seed: int,
    rtol: float,
    atol: float,
    perf_warmup: int,
    perf_iterations: int,
    keep_build_dir: bool = False,
) -> VerificationResult:
    if not torch.cuda.is_available():
        return VerificationResult(False, "CUDA/HIP device is not available.", compile_success=False)

    try:
        original_module = load_python_module(
            original_module_path,
            f"torch2hip_original_{original_module_path.stem}_{seed}",
        )
        functional_module = load_python_module(
            functional_module_path,
            f"torch2hip_functional_{functional_module_path.stem}_{seed}",
        )
        _ensure_required_exports(original_module, original_module_path, REQUIRED_ORIGINAL_EXPORTS)
        _ensure_required_exports(functional_module, functional_module_path, REQUIRED_FUNCTIONAL_EXPORTS)

        _set_seed(seed)
        init_args = getattr(original_module, "get_init_inputs")()
        init_call_args, init_call_kwargs = _normalize_call_args(init_args)

        _set_seed(seed)
        original_model = getattr(original_module, "Model")(*_clone_value(init_call_args), **_clone_value(init_call_kwargs))
        _set_seed(seed)
        functional_model = getattr(functional_module, "Model")(*_clone_value(init_call_args), **_clone_value(init_call_kwargs))

        device = torch.device("cuda")
        original_model = original_model.to(device).eval()
        functional_model = functional_model.to(device).eval()

        _set_seed(seed)
        forward_inputs = getattr(original_module, "get_inputs")()
        forward_args, forward_kwargs = _normalize_call_args(forward_inputs)
        module_args = _move_to_device(_clone_value(forward_args), device)
        module_kwargs = _move_to_device(_clone_value(forward_kwargs), device)
        hip_args = _move_to_device(_clone_value(forward_args), device)
        hip_kwargs = _move_to_device(_clone_value(forward_kwargs), device)

        hip_fn = load_hip_forward(candidate_hip_path, build_dir, verbose=False)

        with torch.no_grad():
            _set_seed(seed)
            expected = original_model(*_clone_value(module_args), **_clone_value(module_kwargs))
            _set_seed(seed)
            actual = functional_model(*_clone_value(hip_args), **_clone_value(hip_kwargs), fn=hip_fn)

        matches, detail = _compare_outputs(expected, actual, rtol, atol)
        if not matches:
            return VerificationResult(
                False,
                detail,
                compile_success=True,
                correctness_success=False,
            )

        module_latency_ms = _measure_latency_ms(
            lambda: original_model(*module_args, **module_kwargs),
            warmup=perf_warmup,
            iterations=perf_iterations,
        )
        hip_latency_ms = _measure_latency_ms(
            lambda: functional_model(*hip_args, **hip_kwargs, fn=hip_fn),
            warmup=perf_warmup,
            iterations=perf_iterations,
        )
        speedup = None
        if hip_latency_ms > 0:
            speedup = module_latency_ms / hip_latency_ms

        summary = "Correctness passed."
        if speedup is not None:
            summary += f" Speedup={speedup:.4f}x (PyTorch={module_latency_ms:.4f}ms, HIP={hip_latency_ms:.4f}ms)."

        return VerificationResult(
            True,
            summary,
            compile_success=True,
            correctness_success=True,
            speedup=speedup,
            module_latency_ms=module_latency_ms,
            hip_latency_ms=hip_latency_ms,
        )
    except Exception as exc:
        return VerificationResult(
            False,
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            compile_success=False,
            correctness_success=False,
        )
    finally:
        if not keep_build_dir:
            cleanup_dir(build_dir)
