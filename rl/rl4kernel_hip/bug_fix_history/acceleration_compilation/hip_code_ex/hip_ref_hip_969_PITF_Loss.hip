// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

// HIP kernel for PITF Loss: out = -log(sigmoid(r_p - r_ne))
template <typename scalar_t>
__global__ void pitf_loss_kernel(
    const scalar_t* __restrict__ r_p,
    const scalar_t* __restrict__ r_ne,
    scalar_t* __restrict__ out,
    int64_t numel) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t diff = r_p[idx] - r_ne[idx];
        scalar_t sigm = 1.0f / (1.0f + expf(-diff));
        scalar_t log_sigm = logf(sigm);
        out[idx] = -log_sigm;
    }
}

// Kernel launcher
torch::Tensor pitf_loss_forward(torch::Tensor r_p, torch::Tensor r_ne) {
    TORCH_CHECK(r_p.is_contiguous(), "r_p must be contiguous");
    TORCH_CHECK(r_ne.is_contiguous(), "r_ne must be contiguous");
    TORCH_CHECK(r_p.sizes() == r_ne.sizes(), "Input sizes must match");
    TORCH_CHECK(r_p.device().is_cuda(), "Input must be on CUDA/HIP device");
    auto numel = r_p.numel();
    auto out = torch::empty_like(r_p);

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(r_p.scalar_type(), "pitf_loss_kernel", ([&] {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(pitf_loss_kernel<scalar_t>),
            dim3(blocks), dim3(threads), 0, 0,
            r_p.data_ptr<scalar_t>(),
            r_ne.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel
        );
    }));

    hipDeviceSynchronize();
    return out;
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pitf_loss_forward, "PITF Loss forward (HIP)");
}
