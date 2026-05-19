#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Profile KernelBench HIP samples with Metrix."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import shlex
import subprocess
import sys
import time
import traceback
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from metrix import Metrix

from runtime_common import (
    DEFAULT_METRICS,
    OUTPUT_DIR,
    SampleSpec,
    build_deferred_sample_metadata,
    collect_sample_metadata,
    describe_arch_state,
    normalize_current_process_env,
    profiling_results_to_dict,
    resolve_effective_arch,
    resolve_sample_specs,
    write_json,
    write_text,
)


THIS_DIR = Path(__file__).resolve().parent
RUNNER_PATH = THIS_DIR / "runner.py"
MAX_PROFILE_GPUS = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile arbitrary KernelBench samples with Metrix."
    )
    parser.add_argument("--arch", type=str, default=None, help="Override GPU arch.")
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help=(
            "Comma-separated sample stems or filenames. "
            "If omitted, the representative default seed set is used."
        ),
    )
    parser.add_argument(
        "--sample-regex",
        type=str,
        default=None,
        help="Regex used to select dataset samples by stem.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Select all KernelBench dataset samples that have paired functional code.",
    )
    parser.add_argument(
        "--hip-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing HIP sources to profile. When set without --samples, "
            "all paired samples in the directory are selected."
        ),
    )
    parser.add_argument(
        "--functional-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing paired functional PyTorch reference files. "
            "Required with --hip-dir for staged KernelBench subsets."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap after sample selection.",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=None, help="Optional GPU id for all runs."
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU ids for sample-level parallel profiling. "
            "Ignored when --gpu-id is set."
        ),
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help=(
            "Sample-level worker count. 0 means auto: spread samples across distinct "
            "visible GPUs when possible. 1 forces sequential execution."
        ),
    )
    parser.add_argument(
        "--compile-workers",
        type=int,
        default=0,
        help=(
            "Compile-stage worker count. 0 means auto: match the GPU worker count, "
            "which enables compile/profile overlap without oversubscribing too hard."
        ),
    )
    parser.add_argument(
        "--prewarm-iters",
        type=int,
        default=2,
        help="Warmup iterations executed outside Metrix before profiling.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=5,
        help="Iterations executed inside each profiled runner invocation.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Timeout passed to Metrix.profile and the prewarm command.",
    )
    parser.add_argument(
        "--profile-retries",
        type=int,
        default=1,
        help=(
            "Retry count for a per-sample profiling failure. rocprofv3 can occasionally "
            "return successfully without writing CSV output under concurrent runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for profiling artifacts.",
    )
    parser.add_argument(
        "--metadata-mode",
        choices=("deferred", "full"),
        default="deferred",
        help=(
            "Metadata collection strategy for scheduling and reporting. "
            "'deferred' avoids calling get_inputs() during sample planning."
        ),
    )
    return parser.parse_args()


def parse_sample_names(raw_value: Optional[str]) -> List[str]:
    if raw_value is None or raw_value.strip() == "":
        return []
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def parse_gpu_ids(raw_value: Optional[str]) -> List[int]:
    if raw_value is None or raw_value.strip() == "":
        return []
    gpu_ids = []
    seen = set()
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        gpu_id = int(part)
        if gpu_id < 0:
            raise ValueError(f"GPU id must be non-negative, got {gpu_id}")
        if gpu_id in seen:
            raise ValueError(
                f"Duplicate GPU id {gpu_id} is not allowed for parallel profiling."
            )
        seen.add(gpu_id)
        gpu_ids.append(gpu_id)
    return gpu_ids


def validate_sample_dirs(args) -> None:
    if (args.hip_dir is None) != (args.functional_dir is None):
        raise ValueError("--hip-dir and --functional-dir must be provided together.")


def visible_gpu_ids() -> List[int]:
    return list(range(min(torch.cuda.device_count(), MAX_PROFILE_GPUS)))


def resolve_execution_plan(args, sample_specs: List[SampleSpec]) -> Dict[str, Any]:
    num_samples = len(sample_specs)

    if args.gpu_id is not None:
        if args.parallel_workers not in (0, 1):
            raise ValueError(
                "--parallel-workers > 1 with --gpu-id would run multiple profiling jobs on the "
                "same GPU, which would pollute the measurements."
            )
        return {
            "mode": "single_gpu",
            "gpu_pool": [args.gpu_id],
            "assigned_gpu_id": args.gpu_id,
            "worker_count": 1,
        }

    candidate_gpu_ids = parse_gpu_ids(args.gpu_ids) or visible_gpu_ids()
    if not candidate_gpu_ids:
        return {
            "mode": "sequential",
            "gpu_pool": [None],
            "assigned_gpu_id": None,
            "worker_count": 1,
        }

    if args.parallel_workers == 1:
        return {
            "mode": "sequential",
            "gpu_pool": [candidate_gpu_ids[0]],
            "assigned_gpu_id": candidate_gpu_ids[0],
            "worker_count": 1,
        }

    if args.parallel_workers > 1:
        gpu_pool = candidate_gpu_ids[: min(args.parallel_workers, len(candidate_gpu_ids))]
    else:
        gpu_pool = candidate_gpu_ids[: min(MAX_PROFILE_GPUS, len(candidate_gpu_ids))]

    if len(gpu_pool) <= 1:
        return {
            "mode": "sequential",
            "gpu_pool": gpu_pool or [None],
            "assigned_gpu_id": gpu_pool[0] if gpu_pool else None,
            "worker_count": 1,
        }

    return {
        "mode": "parallel",
        "gpu_pool": gpu_pool,
        "assigned_gpu_id": None,
        "worker_count": len(gpu_pool),
    }


def build_runner_command(
    sample_spec: SampleSpec,
    effective_arch: str,
    warmup_iters: int,
    run_iters: int,
    gpu_id: Optional[int],
    compile_only: bool = False,
) -> List[str]:
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--hip-file",
        str(sample_spec.hip_file),
        "--functional-file",
        str(sample_spec.functional_file),
        "--arch",
        effective_arch,
        "--warmup-iters",
        str(warmup_iters),
        "--run-iters",
        str(run_iters),
    ]
    if gpu_id is not None:
        command.extend(["--gpu-id", str(gpu_id)])
    if compile_only:
        command.append("--compile-only")
    return command


def resolve_compile_worker_count(
    args, sample_count: int, execution_plan: Dict[str, Any]
) -> int:
    if args.compile_workers > 0:
        return min(args.compile_workers, sample_count)
    return max(1, min(sample_count, execution_plan["worker_count"]))


def run_compile_stage(
    sample_spec: SampleSpec,
    effective_arch: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    command = build_runner_command(
        sample_spec=sample_spec,
        effective_arch=effective_arch,
        warmup_iters=0,
        run_iters=0,
        gpu_id=None,
        compile_only=True,
    )
    print(f"[compile] {sample_spec.stem}: compiling extension")
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=str(THIS_DIR),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(
            "Compile failed for "
            f"{sample_spec.stem}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    print(f"[compile] {sample_spec.stem}: ready in {elapsed:.2f}s")
    return {
        "compile_seconds": elapsed,
        "compile_command": shlex.join(command),
    }


def run_prewarm(
    sample_spec: SampleSpec,
    effective_arch: str,
    prewarm_iters: int,
    gpu_id: Optional[int],
    timeout_seconds: int,
) -> None:
    command = build_runner_command(
        sample_spec=sample_spec,
        effective_arch=effective_arch,
        warmup_iters=prewarm_iters,
        run_iters=0,
        gpu_id=gpu_id,
    )
    result = subprocess.run(
        command,
        cwd=str(THIS_DIR),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Prewarm failed for "
            f"{sample_spec.stem}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def build_kernel_filter(sample_spec: SampleSpec, inventory_results) -> Optional[str]:
    if sample_spec.custom_kernel_names:
        return "|".join(re.escape(name) for name in sample_spec.custom_kernel_names)

    if inventory_results.kernels:
        return "|".join(re.escape(kernel.name) for kernel in inventory_results.kernels)

    return None


def metric_avg(kernel, metric_name: str) -> Optional[float]:
    stats = kernel.metrics.get(metric_name)
    return None if stats is None else stats.avg


def select_primary_kernel(filtered_results, sample_spec: SampleSpec):
    candidates = [
        kernel
        for kernel in filtered_results.kernels
        if any(name in kernel.name for name in sample_spec.custom_kernel_names)
    ]
    if not candidates:
        candidates = list(filtered_results.kernels)
    if not candidates:
        return None
    return max(candidates, key=lambda kernel: kernel.duration_us.avg)


def classify_kernel(
    primary_kernel, sample_spec: SampleSpec, inventory_kernel_names: List[str]
) -> str:
    if primary_kernel is None:
        return "No profiled custom-kernel result was captured."

    library_kernels = [
        name
        for name in inventory_kernel_names
        if not any(custom in name for custom in sample_spec.custom_kernel_names)
    ]
    bandwidth = metric_avg(primary_kernel, "memory.hbm_bandwidth_utilization")
    coalescing = metric_avg(primary_kernel, "memory.coalescing_efficiency")
    flops = metric_avg(primary_kernel, "compute.total_flops")

    if sample_spec.category == "GEMM + elementwise post-op" and library_kernels:
        return (
            "The custom swish/scale kernel is lightweight and well-coalesced; the inventory "
            "still shows a large GEMM kernel, so end-to-end latency is likely GEMM-dominated."
        )
    if sample_spec.category == "Memory-heavy conv3d fusion" and library_kernels:
        return (
            "The filtered custom kernel itself looks efficient, but the full inventory still "
            "contains conv3d and pooling work, so this path is pipeline-dominated rather than "
            "limited by the custom epilogue alone."
        )
    if sample_spec.category == "Conv transpose post-process" and library_kernels:
        return (
            "The custom post-process is measurable, but the inventory still contains a MIOpen "
            "conv-transpose kernel; this makes the custom kernel a secondary cost in the full path."
        )
    if bandwidth is not None and bandwidth >= 30:
        return "Memory-bound leaning: the filtered kernel is driving noticeable HBM traffic."
    if coalescing is not None and coalescing < 85:
        return (
            "Memory-access inefficiency: coalescing is weaker than expected, "
            "so access pattern quality likely matters more than raw FLOPs."
        )
    if flops is not None and flops > 0 and (bandwidth is None or bandwidth < 10):
        return (
            "Mixed but math-heavier: arithmetic work is visible while direct HBM pressure "
            "stays comparatively low."
        )
    return (
        "Mixed pipeline: the custom kernel looks lightweight relative to the surrounding "
        f"{sample_spec.category.lower()} path."
    )


def results_entry(
    sample_spec: SampleSpec,
    metadata: Dict[str, Any],
    inventory_results,
    filtered_results,
    kernel_filter: Optional[str],
    gpu_id: Optional[int],
    stage_timings_s: Dict[str, float],
) -> Dict[str, Any]:
    primary_kernel = select_primary_kernel(filtered_results, sample_spec)
    inventory_kernel_names = [kernel.name for kernel in inventory_results.kernels]
    return {
        "sample": sample_spec.stem,
        "category": sample_spec.category,
        "note": sample_spec.note,
        "gpu_id": gpu_id,
        "profile_ok": True,
        "metadata": metadata,
        "inventory_kernel_names": inventory_kernel_names,
        "kernel_filter": kernel_filter,
        "stage_timings_s": stage_timings_s,
        "primary_kernel": None
        if primary_kernel is None
        else {
            "name": primary_kernel.name,
            "dispatch_count": primary_kernel.dispatch_count,
            "duration_us": primary_kernel.duration_us.avg,
            "memory.hbm_bandwidth_utilization": metric_avg(
                primary_kernel, "memory.hbm_bandwidth_utilization"
            ),
            "memory.l2_hit_rate": metric_avg(primary_kernel, "memory.l2_hit_rate"),
            "memory.coalescing_efficiency": metric_avg(
                primary_kernel, "memory.coalescing_efficiency"
            ),
            "compute.total_flops": metric_avg(primary_kernel, "compute.total_flops"),
        },
        "interpretation": classify_kernel(
            primary_kernel, sample_spec, inventory_kernel_names
        ),
    }


def profile_failure_entry(
    sample_spec: SampleSpec,
    metadata: Dict[str, Any],
    gpu_id: Optional[int],
    compile_info: Optional[Dict[str, Any]],
    elapsed_seconds: float,
    attempts: int,
    exc: BaseException,
) -> Dict[str, Any]:
    compile_seconds = 0.0 if compile_info is None else float(compile_info["compile_seconds"])
    stage_timings_s = {
        "compile_seconds": compile_seconds,
        "prewarm_seconds": 0.0,
        "inventory_profile_seconds": 0.0,
        "filtered_profile_seconds": 0.0,
        "post_compile_total_seconds": elapsed_seconds,
        "serial_equivalent_total_seconds": compile_seconds + elapsed_seconds,
    }
    error = {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "attempts": attempts,
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    return {
        "sample": sample_spec.stem,
        "category": sample_spec.category,
        "note": sample_spec.note,
        "gpu_id": gpu_id,
        "profile_ok": False,
        "profile_error": error,
        "metadata": metadata,
        "inventory_kernel_names": [],
        "kernel_filter": None,
        "stage_timings_s": stage_timings_s,
        "primary_kernel": None,
        "interpretation": f"Profiling failed after {attempts} attempt(s): {error['message']}",
    }


def compile_failure_entry(
    sample_spec: SampleSpec,
    metadata: Dict[str, Any],
    exc: BaseException,
) -> Dict[str, Any]:
    stage_timings_s = {
        "compile_seconds": 0.0,
        "prewarm_seconds": 0.0,
        "inventory_profile_seconds": 0.0,
        "filtered_profile_seconds": 0.0,
        "post_compile_total_seconds": 0.0,
        "serial_equivalent_total_seconds": 0.0,
    }
    error = {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "attempts": 1,
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "stage": "compile",
    }
    return {
        "sample": sample_spec.stem,
        "category": sample_spec.category,
        "note": sample_spec.note,
        "gpu_id": None,
        "profile_ok": False,
        "profile_error": error,
        "metadata": metadata,
        "inventory_kernel_names": [],
        "kernel_filter": None,
        "stage_timings_s": stage_timings_s,
        "primary_kernel": None,
        "interpretation": f"Compile stage failed: {error['message']}",
    }


def build_timing_table_lines(entries: List[Dict[str, Any]]) -> List[str]:
    sorted_entries = sorted(
        entries,
        key=lambda entry: entry["stage_timings_s"]["serial_equivalent_total_seconds"],
        reverse=True,
    )
    lines = [
        "| Kernel | GPU | Compile (s) | Prewarm (s) | Inventory (s) | Filtered (s) | Post-Compile Total (s) | Serial Equivalent Total (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in sorted_entries:
        timings = entry["stage_timings_s"]
        lines.append(
            "| "
            f"`{entry['sample']}` | "
            f"{entry['gpu_id']} | "
            f"{timings['compile_seconds']:.4f} | "
            f"{timings['prewarm_seconds']:.4f} | "
            f"{timings['inventory_profile_seconds']:.4f} | "
            f"{timings['filtered_profile_seconds']:.4f} | "
            f"{timings['post_compile_total_seconds']:.4f} | "
            f"{timings['serial_equivalent_total_seconds']:.4f} |"
        )
    return lines


def render_summary(
    entries: List[Dict[str, Any]],
    effective_arch: str,
    arch_report: Dict[str, Any],
    run_config: Dict[str, Any],
) -> str:
    lines = [
        "# KernelBench Profiling Summary",
        "",
    ]
    lines.extend(build_timing_table_lines(entries))
    lines.extend(
        [
            "",
            "- `Post-Compile Total (s)` = `Prewarm + Inventory + Filtered`",
            "- `Serial Equivalent Total (s)` = `Compile + Post-Compile Total`",
            "- `Serial Equivalent Total (s)` is a per-sample serial estimate, not the parallel end-to-end wall time.",
            "",
        ]
    )
    lines.extend(
        [
        f"- Effective arch: `{effective_arch}`",
        f"- Sample count: `{run_config['sample_count']}`",
        f"- Metrics: `{', '.join(DEFAULT_METRICS)}`",
        f"- Execution mode: `{run_config['execution_mode']}`",
        f"- GPU pool: `{run_config['gpu_pool']}`",
        f"- Compile workers: `{run_config['compile_workers']}`",
        f"- Prewarm iterations: `{run_config['prewarm_iters']}`",
        f"- Profile iterations per run: `{run_config['profile_iters']}`",
        f"- Forced GPU id: `{run_config['gpu_id']}`",
        f"- Metadata mode: `{run_config['metadata_mode']}`",
        f"- Wall time (s): `{run_config['wall_time_seconds']:.2f}`",
        "",
        ]
    )

    if arch_report["mismatches"]:
        lines.append(
            f"- Ignored conflicting arch hints: `{', '.join(arch_report['mismatches'])}`"
        )
    if arch_report["ambiguous"]:
        lines.append(
            f"- Ignored multi-arch hints: `{', '.join(arch_report['ambiguous'])}`"
        )
    if arch_report["cleared_flags"]:
        lines.append(
            f"- Cleared conflicting compile flags: `{', '.join(arch_report['cleared_flags'])}`"
        )
    lines.append("")

    for entry in entries:
        lines.extend(
            [
                f"## {entry['sample']}",
                "",
                f"- Category: `{entry['category']}`",
                f"- Note: {entry['note']}",
                f"- Assigned GPU: `{entry['gpu_id']}`",
                f"- Compile seconds: `{entry['stage_timings_s']['compile_seconds']:.4f}`",
                f"- Prewarm seconds: `{entry['stage_timings_s']['prewarm_seconds']:.4f}`",
                f"- Inventory profile seconds: `{entry['stage_timings_s']['inventory_profile_seconds']:.4f}`",
                f"- Filtered profile seconds: `{entry['stage_timings_s']['filtered_profile_seconds']:.4f}`",
                f"- Post-compile total seconds: `{entry['stage_timings_s']['post_compile_total_seconds']:.4f}`",
                f"- Serial equivalent total seconds: `{entry['stage_timings_s']['serial_equivalent_total_seconds']:.4f}`",
                f"- HIP source: `{entry['metadata']['hip_file']}`",
                f"- Functional pair: `{entry['metadata']['functional_file']}`",
                f"- Custom kernels: `{', '.join(entry['metadata']['custom_kernel_names'])}`",
                f"- Metadata status: `{entry['metadata'].get('metadata_status', 'unknown')}`",
                f"- Metadata note: `{entry['metadata'].get('metadata_reason', '')}`",
                f"- Input summary: `{json.dumps(entry['metadata']['inputs'])}`",
                f"- Init summary: `{json.dumps(entry['metadata']['init_inputs'])}`",
                f"- Inventory kernels: `{', '.join(entry['inventory_kernel_names'])}`",
                f"- Applied kernel filter: `{entry['kernel_filter']}`",
            ]
        )

        primary = entry["primary_kernel"]
        if primary is None:
            lines.append("- Primary kernel: `none`")
        else:
            lines.extend(
                [
                    f"- Primary kernel: `{primary['name']}`",
                    f"- Avg duration (us): `{primary['duration_us']:.4f}`",
                    f"- HBM bandwidth utilization: `{primary['memory.hbm_bandwidth_utilization']}`",
                    f"- L2 hit rate: `{primary['memory.l2_hit_rate']}`",
                    f"- Coalescing efficiency: `{primary['memory.coalescing_efficiency']}`",
                    f"- Total FLOPs: `{primary['compute.total_flops']}`",
                ]
            )
        if entry.get("profile_ok") is False:
            error = entry.get("profile_error") or {}
            lines.append(
                f"- Profiling status: `failed` ({error.get('type', 'error')}: {error.get('message', '')})"
            )
        lines.append(f"- Interpretation: {entry['interpretation']}")
        lines.append("")

    return "\n".join(lines)


def profile_one_sample(
    profiler: Metrix,
    sample_spec: SampleSpec,
    effective_arch: str,
    arch_report: Dict[str, Any],
    args,
    compile_info: Optional[Dict[str, Any]] = None,
    metadata_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    gpu_label = args.gpu_id if args.gpu_id is not None else "default"
    sample_started = time.perf_counter()

    if metadata_override is not None:
        print(
            f"[sample][gpu {gpu_label}] {sample_spec.stem}: using planned metadata "
            f"(status={metadata_override.get('metadata_status', 'unknown')})"
        )
        metadata = metadata_override
    else:
        print(f"[sample][gpu {gpu_label}] {sample_spec.stem}: collecting metadata")
        metadata = collect_sample_metadata(sample_spec)
    print(f"[sample][gpu {gpu_label}] {sample_spec.stem}: prewarming extension")
    prewarm_started = time.perf_counter()
    run_prewarm(
        sample_spec=sample_spec,
        effective_arch=effective_arch,
        prewarm_iters=args.prewarm_iters,
        gpu_id=args.gpu_id,
        timeout_seconds=args.timeout_seconds,
    )
    prewarm_seconds = time.perf_counter() - prewarm_started

    runner_command = build_runner_command(
        sample_spec=sample_spec,
        effective_arch=effective_arch,
        warmup_iters=0,
        run_iters=args.profile_iters,
        gpu_id=args.gpu_id,
    )
    command_str = shlex.join(runner_command)
    print(f"[sample][gpu {gpu_label}] {sample_spec.stem}: inventory profile")
    run_config = {
        "prewarm_iters": args.prewarm_iters,
        "profile_iters": args.profile_iters,
        "gpu_id": args.gpu_id,
        "timeout_seconds": args.timeout_seconds,
        "arch_report": arch_report,
        "metrics": DEFAULT_METRICS,
    }

    inventory_started = time.perf_counter()
    inventory_results = profiler.profile(
        command=command_str,
        time_only=True,
        num_replays=1,
        cwd=str(THIS_DIR),
        timeout_seconds=args.timeout_seconds,
    )
    inventory_seconds = time.perf_counter() - inventory_started
    kernel_filter = build_kernel_filter(sample_spec, inventory_results)

    print(
        f"[sample][gpu {gpu_label}] {sample_spec.stem}: filtered profile "
        f"({kernel_filter if kernel_filter else 'no filter'})"
    )
    filtered_started = time.perf_counter()
    try:
        filtered_results = profiler.profile(
            command=command_str,
            metrics=DEFAULT_METRICS,
            kernel_filter=kernel_filter,
            num_replays=1,
            cwd=str(THIS_DIR),
            timeout_seconds=args.timeout_seconds,
        )
    except RuntimeError as exc:
        error_text = str(exc).lower()
        filter_failed = (
            kernel_filter
            and (
                ("kernel" in error_text and "unrecognized" in error_text)
                or "no output csv" in error_text
            )
        )
        if filter_failed:
            kernel_filter = None
            filtered_results = profiler.profile(
                command=command_str,
                metrics=DEFAULT_METRICS,
                num_replays=1,
                cwd=str(THIS_DIR),
                timeout_seconds=args.timeout_seconds,
            )
        else:
            raise
    filtered_seconds = time.perf_counter() - filtered_started

    compile_seconds = 0.0 if compile_info is None else compile_info["compile_seconds"]
    post_compile_total_seconds = time.perf_counter() - sample_started
    stage_timings_s = {
        "compile_seconds": compile_seconds,
        "prewarm_seconds": prewarm_seconds,
        "inventory_profile_seconds": inventory_seconds,
        "filtered_profile_seconds": filtered_seconds,
        "post_compile_total_seconds": post_compile_total_seconds,
        "serial_equivalent_total_seconds": compile_seconds + post_compile_total_seconds,
    }
    run_config["stage_timings_s"] = stage_timings_s
    if compile_info is not None:
        run_config["compile_command"] = compile_info["compile_command"]

    inventory_payload = profiling_results_to_dict(
        results=inventory_results,
        sample_spec=sample_spec,
        effective_arch=effective_arch,
        run_config=run_config,
        metadata=metadata,
        phase="inventory",
        kernel_filter=None,
    )
    filtered_payload = profiling_results_to_dict(
        results=filtered_results,
        sample_spec=sample_spec,
        effective_arch=effective_arch,
        run_config=run_config,
        metadata=metadata,
        phase="filtered",
        kernel_filter=kernel_filter,
    )

    write_json(args.output_dir / f"{sample_spec.stem}_inventory.json", inventory_payload)
    write_json(args.output_dir / f"{sample_spec.stem}_filtered.json", filtered_payload)
    print(f"[sample][gpu {gpu_label}] {sample_spec.stem}: wrote JSON artifacts")

    return results_entry(
        sample_spec=sample_spec,
        metadata=metadata,
        inventory_results=inventory_results,
        filtered_results=filtered_results,
        kernel_filter=kernel_filter,
        gpu_id=args.gpu_id,
        stage_timings_s=stage_timings_s,
    )


def build_worker_args(args, gpu_id: Optional[int]) -> Dict[str, Any]:
    return {
        "prewarm_iters": args.prewarm_iters,
        "profile_iters": args.profile_iters,
        "profile_retries": args.profile_retries,
        "timeout_seconds": args.timeout_seconds,
        "output_dir": args.output_dir,
        "gpu_id": gpu_id,
    }


def iter_tensor_summaries(value: Any):
    if isinstance(value, dict):
        if value.get("kind") == "tensor":
            yield value
            return
        for item in value.values():
            yield from iter_tensor_summaries(item)
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_tensor_summaries(item)


def product(values: List[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def estimate_compile_score(sample_spec: SampleSpec) -> float:
    source_bytes = sample_spec.hip_file.stat().st_size
    include_dir = sample_spec.hip_file.parent / "include"
    include_bytes = 0
    if include_dir.exists():
        for path in include_dir.rglob("*"):
            if path.is_file():
                include_bytes += path.stat().st_size
    return float(source_bytes + include_bytes)


def estimate_profile_score(metadata: Dict[str, Any], args) -> float:
    tensors = list(iter_tensor_summaries(metadata.get("inputs", [])))
    total_elements = 0
    total_rank = 0
    for tensor in tensors:
        shape = [int(dim) for dim in tensor.get("shape", [])]
        total_elements += product(shape)
        total_rank += len(shape)

    tensor_bonus = len(tensors) * 1_000_000
    rank_bonus = total_rank * 250_000
    kernel_bonus = len(metadata.get("custom_kernel_names", [])) * 2_000_000
    iteration_factor = max(1, args.prewarm_iters + args.profile_iters)

    return float(total_elements * iteration_factor + tensor_bonus + rank_bonus + kernel_bonus)


def resolve_plan_metadata(sample_spec: SampleSpec, args) -> Dict[str, Any]:
    if args.metadata_mode == "full":
        try:
            return collect_sample_metadata(sample_spec)
        except Exception as exc:
            print(
                f"[metadata] {sample_spec.stem}: full collection failed "
                f"({exc.__class__.__name__}: {exc}); falling back to deferred mode",
                flush=True,
            )
            return build_deferred_sample_metadata(
                sample_spec,
                reason=(
                    "fallback_from_full_after_collection_error:"
                    f"{exc.__class__.__name__}"
                ),
            )
    return build_deferred_sample_metadata(
        sample_spec,
        reason=(
            "deferred_by_default_to_avoid_large_cpu_tensor_materialization_in_planning"
        ),
    )


def build_sample_plan(sample_spec: SampleSpec, args) -> Dict[str, Any]:
    metadata = resolve_plan_metadata(sample_spec, args)
    estimated_compile_score = estimate_compile_score(sample_spec)
    estimated_profile_score = estimate_profile_score(metadata, args)
    return {
        "sample_spec": sample_spec,
        "metadata": metadata,
        "estimated_compile_score": estimated_compile_score,
        "estimated_profile_score": estimated_profile_score,
        "estimated_total_score": estimated_compile_score + estimated_profile_score,
    }


def profile_one_sample_worker(
    sample_spec: SampleSpec,
    effective_arch: str,
    arch_report: Dict[str, Any],
    worker_args: Dict[str, Any],
    compile_info: Optional[Dict[str, Any]] = None,
    metadata_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalize_current_process_env(
        effective_arch, gpu_id=worker_args.get("gpu_id")
    )
    args = SimpleNamespace(**worker_args)
    attempts = max(1, int(getattr(args, "profile_retries", 0)) + 1)
    gpu_label = args.gpu_id if args.gpu_id is not None else "default"
    started = time.perf_counter()
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                print(
                    f"[sample][gpu {gpu_label}] {sample_spec.stem}: retrying profile "
                    f"({attempt}/{attempts})",
                    flush=True,
                )
            profiler = Metrix(arch=effective_arch)
            entry = profile_one_sample(
                profiler=profiler,
                sample_spec=sample_spec,
                effective_arch=effective_arch,
                arch_report=arch_report,
                args=args,
                compile_info=compile_info,
                metadata_override=metadata_override,
            )
            entry["profile_attempts"] = attempt
            return entry
        except Exception as exc:
            last_exc = exc
            print(
                f"[sample][gpu {gpu_label}] {sample_spec.stem}: profile attempt "
                f"{attempt}/{attempts} failed: {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(min(30, 2 * attempt))

    assert last_exc is not None
    metadata = (
        metadata_override
        if metadata_override is not None
        else build_deferred_sample_metadata(
            sample_spec,
            reason="profile_worker_failure_without_prefetched_metadata",
        )
    )
    entry = profile_failure_entry(
        sample_spec=sample_spec,
        metadata=metadata,
        gpu_id=args.gpu_id,
        compile_info=compile_info,
        elapsed_seconds=time.perf_counter() - started,
        attempts=attempts,
        exc=last_exc,
    )
    write_json(args.output_dir / f"{sample_spec.stem}_profile_error.json", entry)
    return entry


def run_pipelined_samples(
    sample_plans: List[Dict[str, Any]],
    effective_arch: str,
    arch_report: Dict[str, Any],
    args,
    execution_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    compile_workers = resolve_compile_worker_count(
        args, len(sample_plans), execution_plan
    )
    gpu_pool = execution_plan["gpu_pool"]
    results: List[Optional[Dict[str, Any]]] = [None] * len(sample_plans)
    pending_indices = set(range(len(sample_plans)))
    active_gpu: Dict[Any, tuple[int, Optional[int]]] = {}
    ctx = multiprocessing.get_context("spawn")
    compile_futures: Dict[int, Any] = {}
    compile_results: Dict[int, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=compile_workers) as compile_executor, ProcessPoolExecutor(
        max_workers=execution_plan["worker_count"], mp_context=ctx
    ) as gpu_executor:
        ordered_plans = sorted(
            enumerate(sample_plans),
            key=lambda item: (
                item[1]["estimated_total_score"],
                item[1]["estimated_profile_score"],
                item[1]["estimated_compile_score"],
            ),
            reverse=True,
        )
        for sample_index, sample_plan in ordered_plans:
            future = compile_executor.submit(
                run_compile_stage,
                sample_plan["sample_spec"],
                effective_arch,
                args.timeout_seconds,
            )
            compile_futures[sample_index] = future

        available_gpus = list(gpu_pool)

        while pending_indices or active_gpu:
            made_progress = False
            for sample_index in list(pending_indices):
                future = compile_futures[sample_index]
                if not future.done() or sample_index in compile_results:
                    continue
                compile_exc = future.exception()
                if compile_exc is not None:
                    results[sample_index] = compile_failure_entry(
                        sample_spec=sample_plans[sample_index]["sample_spec"],
                        metadata=sample_plans[sample_index]["metadata"],
                        exc=compile_exc,
                    )
                    pending_indices.remove(sample_index)
                    made_progress = True
                    continue
                compile_results[sample_index] = future.result()

            for gpu_id in list(available_gpus):
                ready_indices = [
                    index for index in pending_indices if index in compile_results
                ]
                if not ready_indices:
                    break
                ready_index = max(
                    ready_indices,
                    key=lambda index: (
                        sample_plans[index]["estimated_profile_score"],
                        compile_results[index]["compile_seconds"],
                        sample_plans[index]["estimated_compile_score"],
                    ),
                )
                future = gpu_executor.submit(
                    profile_one_sample_worker,
                    sample_plans[ready_index]["sample_spec"],
                    effective_arch,
                    arch_report,
                    build_worker_args(args, gpu_id),
                    compile_results[ready_index],
                    sample_plans[ready_index]["metadata"],
                )
                active_gpu[future] = (ready_index, gpu_id)
                available_gpus.remove(gpu_id)
                pending_indices.remove(ready_index)
                made_progress = True

            for future in list(active_gpu):
                if not future.done():
                    continue
                sample_index, gpu_id = active_gpu.pop(future)
                compile_info = compile_results.get(sample_index)
                try:
                    results[sample_index] = future.result()
                except Exception as exc:
                    results[sample_index] = profile_failure_entry(
                        sample_spec=sample_plans[sample_index]["sample_spec"],
                        metadata=sample_plans[sample_index]["metadata"],
                        gpu_id=gpu_id,
                        compile_info=compile_info,
                        elapsed_seconds=0.0,
                        attempts=1,
                        exc=exc,
                    )
                available_gpus.append(gpu_id)
                made_progress = True

            if not pending_indices and not active_gpu:
                break

            if made_progress:
                continue

            wait_set = set(active_gpu)
            wait_set.update(
                future
                for index, future in compile_futures.items()
                if index in pending_indices and index not in compile_results and not future.done()
            )
            if not wait_set:
                continue
            wait(wait_set, return_when=FIRST_COMPLETED)

    return [entry for entry in results if entry is not None]


def main():
    args = parse_args()
    validate_sample_dirs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    effective_arch, _, env_hints = resolve_effective_arch(args.arch)
    arch_report = describe_arch_state(effective_arch, env_hints)
    sample_specs = resolve_sample_specs(
        sample_names=parse_sample_names(args.samples),
        sample_regex=args.sample_regex,
        all_samples=args.all_samples,
        max_samples=args.max_samples,
        hip_dir=args.hip_dir,
        functional_dir=args.functional_dir,
    )
    sample_plans = [build_sample_plan(sample_spec, args) for sample_spec in sample_specs]
    execution_plan = resolve_execution_plan(args, sample_specs)
    compile_workers = resolve_compile_worker_count(
        args, len(sample_specs), execution_plan
    )

    print(
        "Execution plan: "
        f"mode={execution_plan['mode']}, gpu_pool={execution_plan['gpu_pool']}, "
        f"workers={execution_plan['worker_count']}, compile_workers={compile_workers}, "
        f"samples={len(sample_specs)}, metadata_mode={args.metadata_mode}"
    )

    wall_started = time.perf_counter()
    entries = run_pipelined_samples(
        sample_plans=sample_plans,
        effective_arch=effective_arch,
        arch_report=arch_report,
        args=args,
        execution_plan=execution_plan,
    )
    wall_time_seconds = time.perf_counter() - wall_started

    summary = render_summary(
        entries=entries,
        effective_arch=effective_arch,
        arch_report=arch_report,
        run_config={
            "execution_mode": execution_plan["mode"],
            "gpu_pool": execution_plan["gpu_pool"],
            "compile_workers": compile_workers,
            "sample_count": len(sample_specs),
            "prewarm_iters": args.prewarm_iters,
            "profile_iters": args.profile_iters,
            "gpu_id": args.gpu_id,
            "metadata_mode": args.metadata_mode,
            "wall_time_seconds": wall_time_seconds,
        },
    )
    write_text(args.output_dir / "summary.md", summary)
    failed_entries = [entry for entry in entries if entry.get("profile_ok") is False]
    if failed_entries:
        write_json(args.output_dir / "profile_failures.json", failed_entries)
        print(f"Profiling completed with {len(failed_entries)} failed sample(s).")
    print(f"Wrote profiling artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
