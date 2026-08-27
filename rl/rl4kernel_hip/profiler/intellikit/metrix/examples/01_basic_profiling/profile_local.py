#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Metrix Example: Basic GPU Kernel Profiling

This example writes a simple vector addition kernel to /tmp,
compiles it, and uses Metrix to profile its performance.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_ARCH = "gfx942"
ARCH_ENV_VARS = (
    "HIP_EVAL_ARCH",
    "HCC_AMDGPU_TARGET",
    "AMDGPU_TARGETS",
    "GPU_ARCHS",
    "PYTORCH_ROCM_ARCH",
    "ROCM_ARCH",
)
CONFLICTING_FLAG_ENV_VARS = (
    "HIPCC_COMPILE_FLAGS_APPEND",
    "HIPCC_LINK_FLAGS_APPEND",
    "HIP_TARGETS",
)

# Simple vector addition kernel
VECTOR_ADD_KERNEL = """
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vector_add(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024 * 1024;  // 1M elements
    const size_t bytes = N * sizeof(float);

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(vector_add, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_a, d_b, d_c, N);
    hipDeviceSynchronize();

    printf("Vector add completed\\n");

    free(h_a);
    free(h_b);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    return 0;
}
"""


def _extract_arch_candidates(raw_value):
    if not raw_value:
        return []
    return re.findall(r"gfx[0-9a-z]+", raw_value)


def _detect_gpu_arch():
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    match = re.search(r"Name:\s+(gfx\w+)", result.stdout)
    return match.group(1) if match else None


def _resolve_effective_arch():
    detected_arch = _detect_gpu_arch()
    env_hints = {}

    for var in ARCH_ENV_VARS:
        candidates = _extract_arch_candidates(os.environ.get(var))
        if candidates:
            env_hints[var] = candidates

    if detected_arch:
        return detected_arch, "rocminfo", env_hints

    for var in ARCH_ENV_VARS:
        candidates = env_hints.get(var, [])
        if len(candidates) == 1:
            return candidates[0], var, env_hints

    return DEFAULT_ARCH, "default fallback", env_hints


def _build_clean_hip_env(effective_arch):
    env = os.environ.copy()

    for var in CONFLICTING_FLAG_ENV_VARS:
        env.pop(var, None)

    for var in ARCH_ENV_VARS:
        env[var] = effective_arch

    return env


def _describe_arch_hints(effective_arch, env_hints):
    mismatches = []
    ambiguous = []

    for var, candidates in env_hints.items():
        if len(candidates) > 1:
            ambiguous.append(f"{var}={','.join(candidates)}")
        elif candidates[0] != effective_arch:
            mismatches.append(f"{var}={candidates[0]}")

    return mismatches, ambiguous


def main():
    print("=" * 80)
    print("Metrix Example: Basic GPU Kernel Profiling")
    print("=" * 80)
    print()

    effective_arch, arch_source, env_hints = _resolve_effective_arch()
    clean_env = _build_clean_hip_env(effective_arch)
    mismatches, ambiguous = _describe_arch_hints(effective_arch, env_hints)
    cleared_flags = [
        var for var in CONFLICTING_FLAG_ENV_VARS if os.environ.get(var)
    ]

    print("Step 0: Resolving GPU architecture...")
    print(f"  Using architecture: {effective_arch} ({arch_source})")
    if mismatches:
        print(f"  Ignoring conflicting arch hints: {', '.join(mismatches)}")
    if ambiguous:
        print(f"  Ignoring multi-arch hints: {', '.join(ambiguous)}")
    if cleared_flags:
        print(f"  Clearing conflicting compile flags: {', '.join(cleared_flags)}")
    print()

    # Keep child processes on a single architecture and strip bad appended flags.
    for var in CONFLICTING_FLAG_ENV_VARS:
        os.environ.pop(var, None)
    for var in ARCH_ENV_VARS:
        os.environ[var] = effective_arch

    with tempfile.TemporaryDirectory(prefix="metrix_example_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Temp directory: {tmp_dir}")
        print()

        # Write kernel to file
        print("Step 1: Writing kernel to /tmp...")
        kernel_file = tmp_path / "vector_add.hip"
        kernel_file.write_text(VECTOR_ADD_KERNEL)
        print(f"  Wrote: {kernel_file}")
        print()

        # Compile kernel
        print("Step 2: Compiling kernel...")
        binary_file = tmp_path / "vector_add"
        cmd = [
            "hipcc",
            str(kernel_file),
            "-o",
            str(binary_file),
            "-O2",
            f"--offload-arch={effective_arch}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env=clean_env)

        if result.returncode != 0:
            print(f"  Compilation failed:\n{result.stderr}")
            return 1
        print(f"  Compiled: {binary_file}")
        print()

        # Profile with metrix
        print("Step 3: Profiling with Metrix...")

        try:
            from metrix import Metrix

            # Use the same architecture for compile and profiling.
            profiler = Metrix(arch=effective_arch)

            # Select a few key metrics to display
            metrics_to_collect = [
                "memory.hbm_bandwidth_utilization",
                "memory.l2_hit_rate",
                "memory.coalescing_efficiency",
                "compute.total_flops",
            ]

            print(f"  Running: {binary_file}")
            results = profiler.profile(
                command=str(binary_file), metrics=metrics_to_collect, cwd=str(tmp_path)
            )

            print()
            print("=" * 80)
            print("GPU PERFORMANCE METRICS")
            print("=" * 80)

            for kernel in results.kernels:
                print(f"\nKernel: {kernel.name}")
                print(f"  Duration: {kernel.duration_us.avg:.2f} μs")

                # Display metrics
                for metric_name, stats in kernel.metrics.items():
                    print(f"  {metric_name}: {stats.avg:.2f}")

            print("=" * 80)

        except Exception as e:
            print(f"  Metrix profiling failed: {e}")
            print("  Running kernel directly to verify compilation...")
            result = subprocess.run(
                [str(binary_file)], capture_output=True, text=True, env=clean_env
            )
            print(result.stdout)
            if result.returncode == 0:
                print("  Kernel executed successfully")

        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
