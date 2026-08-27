#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel-agent generation package for HIP_benchmark_kit.

Heavy vLLM imports intentionally stay behind ``backend_vllm`` so lightweight
tests can import helper modules without requiring the serving stack.
"""

__all__: list[str] = []

