# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM backend adapter for generation."""

from __future__ import annotations

from .vllm_generator import VLLMGenerator


def build_generator(config) -> VLLMGenerator:
    return VLLMGenerator(config)
