#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Execute one KernelBench HIP sample through its PyTorch extension path."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from runtime_common import (
    build_directory_for_sample,
    build_sample_spec,
    collect_sample_metadata,
    instantiate_model,
    load_module_from_path,
    module_name_for_sample,
    move_inputs_to_cuda,
    normalize_current_process_env,
    resolve_effective_arch,
    safe_call_model,
    set_seed,
    stage_hip_code,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile and execute one KernelBench HIP sample."
    )
    parser.add_argument("--hip-file", type=Path, required=True, help="Path to the HIP source.")
    parser.add_argument(
        "--functional-file",
        type=Path,
        required=True,
        help="Path to the paired functional PyTorch reference.",
    )
    parser.add_argument("--arch", type=str, default=None, help="Override GPU arch.")
    parser.add_argument(
        "--gpu-id", type=int, default=None, help="Optional GPU id for execution."
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        help="Warmup iterations to execute before the profiled loop.",
    )
    parser.add_argument(
        "--run-iters",
        type=int,
        default=1,
        help="Iterations to execute for the main workload loop.",
    )
    parser.add_argument(
        "--emit-metadata",
        action="store_true",
        help="Print resolved sample metadata before running.",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile and load the extension, but do not execute warmup or workload.",
    )
    return parser.parse_args()


def load_hip_extension(sample_spec, effective_arch: str):
    stage_dir, staged_hip = stage_hip_code(sample_spec.hip_file)
    build_dir = build_directory_for_sample(sample_spec.hip_file, effective_arch)
    include_dir = stage_dir / "include"
    include_paths = [str(include_dir)] if include_dir.exists() else []
    module_name = module_name_for_sample(sample_spec.hip_file, effective_arch)
    return load(
        name=module_name,
        extra_include_paths=include_paths,
        sources=[str(staged_hip)],
        build_directory=str(build_dir),
        extra_cflags=["-O2", f"--offload-arch={effective_arch}"],
        verbose=False,
    )


def run_sample(sample_spec, effective_arch: str, warmup_iters: int, run_iters: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available for runner execution.")

    set_seed(42)
    functional_module = load_module_from_path(
        sample_spec.functional_file, f"functional_runtime_{sample_spec.stem}"
    )
    model = instantiate_model(functional_module, sample_spec.stem).to("cuda")
    inputs = move_inputs_to_cuda(functional_module.get_inputs())
    hip_extension = load_hip_extension(sample_spec, effective_arch)
    hip_fn = hip_extension.forward
    model.eval()

    torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup_iters):
            safe_call_model(model, inputs, hip_fn)
        torch.cuda.synchronize()

        result = None
        for _ in range(run_iters):
            result = safe_call_model(model, inputs, hip_fn)
    torch.cuda.synchronize()
    return result


def compile_sample(sample_spec, effective_arch: str):
    load_hip_extension(sample_spec, effective_arch)


def main():
    args = parse_args()
    sample_spec = build_sample_spec(args.hip_file, args.functional_file)
    effective_arch, _, _ = resolve_effective_arch(args.arch)

    # load() consumes process env, so normalize the current process as well.
    normalize_current_process_env(effective_arch, gpu_id=args.gpu_id)

    if args.emit_metadata:
        metadata = collect_sample_metadata(sample_spec)
        print(metadata)

    if args.compile_only:
        compile_sample(sample_spec=sample_spec, effective_arch=effective_arch)
        return 0

    run_sample(
        sample_spec=sample_spec,
        effective_arch=effective_arch,
        warmup_iters=max(args.warmup_iters, 0),
        run_iters=max(args.run_iters, 0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
