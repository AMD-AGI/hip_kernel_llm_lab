// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "hip/hip_runtime.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

// Kernel computes the sum of squared absolute differences between CDFs along channel dimension
// For each (n, h, w) line, compute running cumsum over C, accumulate |cdf_est - cdf_tgt|^2, and atomicAdd to global sum
__global__ void cdf_diff_sumsq_kernel(
    const float* __restrict__ p_target,
    const float* __restrict__ p_estimate,
    float* __restrict__ global_sum,
    const int N,
    const int C,
    const int H,
    const int W) {

    const int lines = N * H * W;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= lines) return;

    // Recover (n, h, w) from flattened line index
    const int w = tid % W;
    const int tmp1 = tid / W;
    const int h = tmp1 % H;
    const int n = tmp1 / H;

    const int HW = H * W;
    const int CHW = C * HW;
    const int base_n = n * CHW;
    const int base_hw = h * W + w;

    float cumsum_t = 0.0f;
    float cumsum_e = 0.0f;
    float sumsq = 0.0f;

    // Loop over channels to build CDFs and accumulate squared abs differences
    for (int c = 0; c < C; ++c) {
        const int idx = base_n + c * HW + base_hw;
        cumsum_t += p_target[idx];
        cumsum_e += p_estimate[idx];
        const float diff = cumsum_e - cumsum_t;
        const float adiff = fabsf(diff);
        sumsq = fmaf(adiff, adiff, sumsq); // sumsq += adiff * adiff
    }

    // Accumulate per-line sum of squares into global sum
    atomicAdd(global_sum, sumsq);
}

// Launcher: computes sqrt(mean(|CDF_est - CDF_tgt|^2)) over all elements, matching the given PyTorch logic
at::Tensor forward(
    at::Tensor p_target,
    at::Tensor p_estimate) {

    TORCH_CHECK(p_target.sizes() == p_estimate.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(p_target.dim() == 4, "Expected 4D tensors (N, C, H, W)");

    // Ensure contiguous and float tensors
    at::Tensor tgt = p_target.contiguous().to(torch::kFloat);
    at::Tensor est = p_estimate.contiguous().to(torch::kFloat);

    const int64_t N = tgt.size(0);
    const int64_t C = tgt.size(1);
    const int64_t H = tgt.size(2);
    const int64_t W = tgt.size(3);

    const int64_t total_elements = N * C * H * W;
    const int64_t lines = N * H * W;

    auto options = torch::TensorOptions().device(tgt.device()).dtype(torch::kFloat);
    at::Tensor global_sum_tensor = torch::zeros({1}, options);

    const int blockSize = 256;
    const int gridSize = (static_cast<int>(lines) + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(
        cdf_diff_sumsq_kernel,
        dim3(gridSize),
        dim3(blockSize),
        0, 0,
        tgt.data_ptr<float>(),
        est.data_ptr<float>(),
        global_sum_tensor.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W)
    );

    hipDeviceSynchronize();

    // Compute sqrt(mean(squared_abs_diff)) over all elements
    const float sumsq = global_sum_tensor.item<float>();
    const float mean_val = sumsq / static_cast<float>(total_elements);
    const float result = sqrtf(mean_val);

    // The original PyTorch applies .mean() again on a scalar; returning the scalar as a 1-element tensor
    return torch::tensor({result}, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "EDMLoss-like forward (CDF diff sqrt(mean(square)))");
}