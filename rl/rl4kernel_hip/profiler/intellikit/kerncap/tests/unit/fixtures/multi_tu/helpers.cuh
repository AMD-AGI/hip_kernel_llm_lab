// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

template <typename T>
__device__ T fma_helper(T a, T b) {
    return a + b;
}
