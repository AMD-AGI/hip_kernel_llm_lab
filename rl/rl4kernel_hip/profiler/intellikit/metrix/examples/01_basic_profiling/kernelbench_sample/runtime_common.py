#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Shared helpers for profiling KernelBench HIP extensions with Metrix."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


THIS_DIR = Path(__file__).resolve().parent


def _discover_repo_root() -> Path:
    for parent in THIS_DIR.parents:
        if (parent / "HIP_benchmark_kit").exists() and (parent / "profiler").exists():
            return parent
    raise RuntimeError(f"Unable to locate repo root from {THIS_DIR}")


REPO_ROOT = _discover_repo_root()
EXAMPLES_DIR = THIS_DIR.parent
SERVER_SANDBOX_DIR = REPO_ROOT / "hip_kernel_evaluation_server" / "sandbox_core"
DATASET_ROOT = (
    REPO_ROOT / "HIP_benchmark_kit" / "data" / "hip_eval_dataset_kernelbench_25_tasks"
)
HIP_CODE_DIR = DATASET_ROOT / "hip_code"
FUNCTIONAL_CODE_DIR = DATASET_ROOT / "pytorch_code_functional"
OUTPUT_DIR = THIS_DIR / "output"
STAGING_ROOT = OUTPUT_DIR / "staging"
BUILD_ROOT = OUTPUT_DIR / "torch_extensions"

# Shared eval helpers now live in the server sandbox package; keep the profiler
# wired to that canonical implementation instead of the removed legacy eval copy.
for path in (EXAMPLES_DIR, SERVER_SANDBOX_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fix_seed import set_seed
from profile_local import (
    ARCH_ENV_VARS,
    CONFLICTING_FLAG_ENV_VARS,
    _build_clean_hip_env as _profile_build_clean_hip_env,
    _describe_arch_hints as _profile_describe_arch_hints,
    _resolve_effective_arch as _profile_resolve_effective_arch,
)
from safe_call_helper import SAFE_CALL_HELPER


_SAFE_CALL_NAMESPACE: Dict[str, Any] = {}
exec(SAFE_CALL_HELPER, _SAFE_CALL_NAMESPACE)
safe_call_model = _SAFE_CALL_NAMESPACE["_safe_call"]

DEFAULT_METRICS = [
    "memory.hbm_bandwidth_utilization",
    "memory.l2_hit_rate",
    "memory.coalescing_efficiency",
    "compute.total_flops",
]

DEFAULT_SAMPLE_CONFIGS = (
    (
        "8189_matmul_swish_scaling_2d_base.hip",
        "GEMM + elementwise post-op",
        "Linear addmm followed by a custom swish-and-scale kernel.",
    ),
    (
        "6190_coalesced_memory_access_kernel_base.hip",
        "Memory-heavy conv3d fusion",
        "Conv3d/max_pool pipeline followed by a custom logsumexp + ReLU kernel.",
    ),
    (
        "172_coalesced_tiling_kernel_base.hip",
        "Conv transpose post-process",
        "ConvTranspose2d pipeline followed by a custom coalesced post-process kernel.",
    ),
)

DEFAULT_SAMPLE_METADATA = {
    file_name: {"category": category, "note": note}
    for file_name, category, note in DEFAULT_SAMPLE_CONFIGS
}


@dataclass(frozen=True)
class SampleSpec:
    """Paths and metadata for one profiled sample."""

    hip_file: Path
    functional_file: Path
    category: str
    note: str
    custom_kernel_names: tuple[str, ...]

    @property
    def stem(self) -> str:
        return self.hip_file.stem


def ensure_runtime_dirs() -> None:
    for path in (OUTPUT_DIR, STAGING_ROOT, BUILD_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def resolve_effective_arch(requested_arch: Optional[str] = None):
    if requested_arch:
        return requested_arch, "cli override", {}
    return _profile_resolve_effective_arch()


def build_clean_hip_env(effective_arch: str) -> Dict[str, str]:
    return _profile_build_clean_hip_env(effective_arch)


def describe_arch_state(effective_arch: str, env_hints: Dict[str, List[str]]) -> Dict[str, Any]:
    mismatches, ambiguous = _profile_describe_arch_hints(effective_arch, env_hints)
    cleared_flags = [
        var for var in CONFLICTING_FLAG_ENV_VARS if os.environ.get(var)
    ]
    return {
        "effective_arch": effective_arch,
        "mismatches": mismatches,
        "ambiguous": ambiguous,
        "cleared_flags": cleared_flags,
    }


def normalize_current_process_env(effective_arch: str, gpu_id: Optional[int] = None) -> None:
    for var in CONFLICTING_FLAG_ENV_VARS:
        os.environ.pop(var, None)
    for var in ARCH_ENV_VARS:
        os.environ[var] = effective_arch
    if gpu_id is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def module_name_for_sample(hip_file: Path, effective_arch: str) -> str:
    safe_name = re.sub(r"[^0-9A-Za-z_]", "_", hip_file.stem)
    return f"kernelbench_{safe_name}_{effective_arch}"


def stage_hip_code(hip_file: Path) -> tuple[Path, Path]:
    ensure_runtime_dirs()
    stage_dir = STAGING_ROOT / hip_file.stem
    stage_dir.mkdir(parents=True, exist_ok=True)

    staged_hip = stage_dir / hip_file.name
    shutil.copy2(hip_file, staged_hip)

    src_include_dir = hip_file.parent / "include"
    dst_include_dir = stage_dir / "include"
    if src_include_dir.exists():
        if dst_include_dir.exists():
            shutil.rmtree(dst_include_dir)
        shutil.copytree(src_include_dir, dst_include_dir)
    elif dst_include_dir.exists():
        shutil.rmtree(dst_include_dir)

    return stage_dir, staged_hip


def build_directory_for_sample(hip_file: Path, effective_arch: str) -> Path:
    ensure_runtime_dirs()
    build_dir = BUILD_ROOT / effective_arch / hip_file.stem
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir


def load_module_from_path(module_path: Path, module_name: Optional[str] = None):
    module_name = module_name or f"module_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def derive_functional_file(
    hip_file: Path, functional_dir: Optional[Path] = None
) -> Path:
    source_dir = functional_dir or FUNCTIONAL_CODE_DIR
    return source_dir / f"{hip_file.stem}.py"


def extract_custom_kernel_names(hip_file: Path) -> tuple[str, ...]:
    source = hip_file.read_text(encoding="utf-8")
    matches = re.findall(r"__global__\s+void\s+([A-Za-z_]\w*)", source)
    seen = set()
    ordered = []
    for match in matches:
        if match not in seen:
            ordered.append(match)
            seen.add(match)
    return tuple(ordered)


def build_sample_spec(
    hip_file: Path,
    functional_file: Optional[Path] = None,
    category: str = "",
    note: str = "",
) -> SampleSpec:
    functional_file = functional_file or derive_functional_file(hip_file)
    return SampleSpec(
        hip_file=hip_file,
        functional_file=functional_file,
        category=category,
        note=note,
        custom_kernel_names=extract_custom_kernel_names(hip_file),
    )


def get_default_sample_specs() -> List[SampleSpec]:
    return [
        build_sample_spec(HIP_CODE_DIR / file_name, category=category, note=note)
        for file_name, category, note in DEFAULT_SAMPLE_CONFIGS
    ]


def infer_sample_metadata(hip_file: Path) -> Dict[str, str]:
    metadata = DEFAULT_SAMPLE_METADATA.get(hip_file.name)
    if metadata is not None:
        return metadata
    return {
        "category": "Dataset sample",
        "note": f"Auto-selected dataset sample {hip_file.stem}.",
    }


def list_dataset_sample_specs(
    hip_dir: Path = HIP_CODE_DIR, functional_dir: Path = FUNCTIONAL_CODE_DIR
) -> List[SampleSpec]:
    hip_dir = hip_dir.resolve()
    functional_dir = functional_dir.resolve()
    if not hip_dir.is_dir():
        raise FileNotFoundError(f"HIP source directory not found: {hip_dir}")
    if not functional_dir.is_dir():
        raise FileNotFoundError(f"Functional reference directory not found: {functional_dir}")

    specs = []
    for hip_file in sorted(hip_dir.glob("*.hip")):
        functional_file = derive_functional_file(hip_file, functional_dir)
        if not functional_file.exists():
            continue
        metadata = infer_sample_metadata(hip_file)
        specs.append(
            build_sample_spec(
                hip_file,
                functional_file=functional_file,
                category=metadata["category"],
                note=metadata["note"],
            )
        )
    return specs


def resolve_named_sample_specs(
    sample_names: Sequence[str],
    hip_dir: Path = HIP_CODE_DIR,
    functional_dir: Path = FUNCTIONAL_CODE_DIR,
) -> List[SampleSpec]:
    all_specs = list_dataset_sample_specs(hip_dir=hip_dir, functional_dir=functional_dir)
    by_name = {spec.hip_file.name: spec for spec in all_specs}
    by_stem = {spec.hip_file.stem: spec for spec in all_specs}

    resolved = []
    for raw_name in sample_names:
        sample_name = raw_name.strip()
        if not sample_name:
            continue
        spec = by_name.get(sample_name) or by_stem.get(sample_name)
        if spec is None:
            raise FileNotFoundError(
                f"Unable to resolve sample '{sample_name}' in {hip_dir.resolve()}"
            )
        resolved.append(spec)
    return resolved


def resolve_sample_specs(
    sample_names: Optional[Sequence[str]] = None,
    sample_regex: Optional[str] = None,
    all_samples: bool = False,
    max_samples: Optional[int] = None,
    hip_dir: Optional[Path] = None,
    functional_dir: Optional[Path] = None,
) -> List[SampleSpec]:
    resolved_hip_dir = (hip_dir or HIP_CODE_DIR).resolve()
    resolved_functional_dir = (functional_dir or FUNCTIONAL_CODE_DIR).resolve()

    if sample_names:
        specs = resolve_named_sample_specs(
            sample_names,
            hip_dir=resolved_hip_dir,
            functional_dir=resolved_functional_dir,
        )
    elif all_samples or sample_regex:
        specs = list_dataset_sample_specs(
            hip_dir=resolved_hip_dir,
            functional_dir=resolved_functional_dir,
        )
    else:
        if hip_dir is not None or functional_dir is not None:
            specs = list_dataset_sample_specs(
                hip_dir=resolved_hip_dir,
                functional_dir=resolved_functional_dir,
            )
        else:
            specs = get_default_sample_specs()

    if sample_regex:
        pattern = re.compile(sample_regex)
        specs = [spec for spec in specs if pattern.search(spec.stem)]

    if max_samples is not None and max_samples >= 0:
        specs = specs[:max_samples]

    if not specs:
        raise ValueError("Sample selection resolved to an empty set.")

    return specs


def resolve_model_class(functional_module, base_name: str):
    model_class = getattr(functional_module, "Model", None)
    if model_class is not None:
        return model_class

    fallback_name = base_name.split("_", 2)[-1]
    model_class = getattr(functional_module, fallback_name, None)
    if model_class is None:
        raise AttributeError(
            f"No Model class found in {functional_module.__file__} for {base_name}"
        )
    return model_class


def instantiate_model(functional_module, base_name: str):
    model_class = resolve_model_class(functional_module, base_name)
    init_inputs = list(functional_module.get_init_inputs())
    if len(init_inputs) == 0:
        return model_class()
    if (
        len(init_inputs) == 2
        and isinstance(init_inputs[0], list)
        and isinstance(init_inputs[1], dict)
    ):
        kwargs = init_inputs[1]
        return model_class() if len(kwargs) == 0 else model_class(**kwargs)
    return model_class(*init_inputs)


def move_inputs_to_cuda(inputs: Sequence[Any]) -> List[Any]:
    moved = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to("cuda"))
        else:
            moved.append(value)
    return moved


def summarize_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        return [summarize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): summarize_value(item) for key, item in value.items()}
    return value


def base_sample_metadata(sample_spec: SampleSpec) -> Dict[str, Any]:
    return {
        "sample": sample_spec.stem,
        "hip_file": str(sample_spec.hip_file),
        "functional_file": str(sample_spec.functional_file),
        "category": sample_spec.category,
        "note": sample_spec.note,
        "custom_kernel_names": list(sample_spec.custom_kernel_names),
    }


def build_deferred_sample_metadata(
    sample_spec: SampleSpec, *, reason: str
) -> Dict[str, Any]:
    metadata = base_sample_metadata(sample_spec)
    metadata["init_inputs"] = []
    metadata["inputs"] = []
    metadata["metadata_status"] = "deferred"
    metadata["metadata_reason"] = reason
    return metadata


def collect_sample_metadata(sample_spec: SampleSpec) -> Dict[str, Any]:
    set_seed(42)
    module = load_module_from_path(
        sample_spec.functional_file, f"functional_{sample_spec.stem}"
    )
    init_inputs = list(module.get_init_inputs())
    inputs = list(module.get_inputs())
    metadata = base_sample_metadata(sample_spec)
    metadata["init_inputs"] = summarize_value(init_inputs)
    metadata["inputs"] = summarize_value(inputs)
    metadata["metadata_status"] = "collected"
    metadata["metadata_reason"] = ""
    return metadata


def stats_to_dict(stats) -> Dict[str, Any]:
    if stats is None:
        return {}
    return {
        "min": stats.min,
        "max": stats.max,
        "avg": stats.avg,
        "count": stats.count,
        "unit": getattr(stats, "unit", ""),
    }


def profiling_results_to_dict(
    results,
    sample_spec: SampleSpec,
    effective_arch: str,
    run_config: Dict[str, Any],
    metadata: Dict[str, Any],
    phase: str,
    kernel_filter: Optional[str] = None,
) -> Dict[str, Any]:
    kernels = []
    for kernel in results.kernels:
        kernels.append(
            {
                "name": kernel.name,
                "dispatch_count": kernel.dispatch_count,
                "duration_us": stats_to_dict(kernel.duration_us),
                "metrics": {
                    name: stats_to_dict(stats)
                    for name, stats in kernel.metrics.items()
                },
            }
        )

    return {
        "phase": phase,
        "effective_arch": effective_arch,
        "command": results.command,
        "kernel_filter": kernel_filter,
        "total_kernels": results.total_kernels,
        "run_config": run_config,
        "sample": metadata,
        "kernels": kernels,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
