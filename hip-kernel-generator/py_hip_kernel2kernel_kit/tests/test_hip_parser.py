# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from py_hip_kernel2kernel_kit.hip_parser import (
    extract_gpu_functions,
    replace_function_body,
    select_optimization_target,
)


HIP_SOURCE = """
// __global__ void ignored_in_comment() {}
__device__ float helper(const float* x, int idx) {
    float value = x[idx];
    if (value > 0.0f) {
        value *= 2.0f;
    }
    return value;
}

__global__ void sample_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = helper(x, idx);
        out[idx] = value + 1.0f;
    }
}
"""


HIP_VARIANTS_SOURCE = """
__global__ void proto_only(float* out);

__global__ __launch_bounds__(256) void bounded_kernel(float* out) {
    out[threadIdx.x] = 1.0f;
}

__global__ __launch_bounds__(256, 4) void bounded_kernel_2(float* out) {
    out[threadIdx.x] = 2.0f;
}

__device__ __forceinline__ int forced_inline(int x) {
    return x + 1;
}

__device__ __noinline__ int noinline_helper(int x) {
    int y = x + 2;
    return y;
}

__host__ __device__ constexpr int host_device_fn(int x) {
    return x + 3;
}

template<typename T>
__global__ void templated_kernel(T* out) {
    out[threadIdx.x] = T{};
}

template<int N>
__global__ __launch_bounds__(N) void templated_launch_bounds(float* out) {
    out[threadIdx.x] = static_cast<float>(N);
}

__global__
void multiline_kernel(float* __restrict__ out) {
    out[threadIdx.x] = 3.0f;
}
"""


HIP_HEADER_TEMPLATE_SOURCE = """
#include "hip/hip_runtime.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cstdint>

// HIP kernel: generic row-wise scaling
template <typename scalar_t>
__global__ void row_scale_rows_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int64_t N,
    const int64_t M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * M) {
        C[idx] = A[idx / M] * B[idx];
    }
}
"""


def test_extract_gpu_functions_and_select_target() -> None:
    functions = extract_gpu_functions(HIP_SOURCE)

    assert [function.name for function in functions] == ["helper", "sample_kernel"]
    assert functions[0].qualifier == "__device__"
    assert functions[1].qualifier == "__global__"

    target = select_optimization_target(functions)
    assert target is not None
    assert target.name == "sample_kernel"
    assert "helper(x, idx)" in target.body


def test_extract_gpu_functions_supports_launch_bounds_templates_and_host_device() -> None:
    functions = extract_gpu_functions(HIP_VARIANTS_SOURCE)

    assert [function.name for function in functions] == [
        "bounded_kernel",
        "bounded_kernel_2",
        "forced_inline",
        "noinline_helper",
        "host_device_fn",
        "templated_kernel",
        "templated_launch_bounds",
        "multiline_kernel",
    ]
    assert "__global__ __launch_bounds__(256) void bounded_kernel" in functions[0].full_text
    assert "__global__ __launch_bounds__(256, 4) void bounded_kernel_2" in functions[1].full_text
    assert "__device__ __forceinline__ int forced_inline" in functions[2].full_text
    assert "__device__ __noinline__ int noinline_helper" in functions[3].full_text
    assert "__host__ __device__ constexpr int host_device_fn" in functions[4].full_text
    assert "template<typename T>" in functions[5].full_text
    assert "template<int N>" in functions[6].full_text
    assert "__global__\nvoid multiline_kernel" in functions[7].full_text
    assert all(function.name != "proto_only" for function in functions)
    assert functions[4].qualifier == "__device__"


def test_select_optimization_target_respects_device_only_mode() -> None:
    functions = extract_gpu_functions(HIP_VARIANTS_SOURCE)

    device_target = select_optimization_target(functions, mode="device")
    assert device_target is not None
    assert device_target.qualifier == "__device__"
    assert device_target.name == "noinline_helper"

    global_target = select_optimization_target(functions, mode="global")
    assert global_target is not None
    assert global_target.qualifier == "__global__"


def test_extract_gpu_functions_handles_file_header_before_first_template_kernel() -> None:
    functions = extract_gpu_functions(HIP_HEADER_TEMPLATE_SOURCE)

    assert len(functions) == 1
    assert functions[0].name == "row_scale_rows_kernel"
    assert functions[0].qualifier == "__global__"
    assert functions[0].full_text.startswith("template <typename scalar_t>\n__global__ void row_scale_rows_kernel")
    assert "#include" not in functions[0].signature
    assert "HIP kernel" not in functions[0].signature


def test_replace_function_body_preserves_signature() -> None:
    functions = extract_gpu_functions(HIP_SOURCE)
    target = select_optimization_target(functions)
    assert target is not None

    replaced = replace_function_body(
        HIP_SOURCE,
        target,
        """
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
    out[idx] = x[idx];
}
""",
    )

    assert "__global__ void sample_kernel(const float* x, float* out, int n) {" in replaced
    assert "out[idx] = x[idx];" in replaced
    assert "out[idx] = value + 1.0f;" not in replaced
