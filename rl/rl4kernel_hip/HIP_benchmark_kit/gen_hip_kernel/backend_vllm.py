"""vLLM backend adapter for generation."""

from __future__ import annotations

from .vllm_generator import VLLMGenerator


def build_generator(config) -> VLLMGenerator:
    return VLLMGenerator(config)
