# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generation paradigm policies for HIP optimization outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from dataset.prompts import (
    DEFAULT_OPTIMIZATION_PARADIGM,
    HIP2HIP_FULL_FILE_PARADIGM,
    KERNEL2KERNEL_SPLICE_PARADIGM,
    normalize_optimization_paradigm,
)

from .kernel_utils import extract_kernel_from_hip_code, replace_kernel_in_hip_code

KERNEL_FUNCTION_CODE_UNIT = "kernel_function"
HIP_TRANSLATION_UNIT_CODE_UNIT = "hip_translation_unit"
SPLICE_PERSISTENCE_MODE = "splice_kernel"
FULL_FILE_PERSISTENCE_MODE = "direct_full_file"


@dataclass(frozen=True)
class PromptSource:
    starter_code: str
    kernel_name: Optional[str]
    starter_code_kind: str
    expected_code_unit: str
    persistence_mode: str


@dataclass(frozen=True)
class GenerationParadigmPolicy:
    optimization_paradigm: str
    expected_code_unit: str
    persistence_mode: str

    def build_prompt_source(self, hip_code: str) -> PromptSource:
        if self.optimization_paradigm == HIP2HIP_FULL_FILE_PARADIGM:
            kernel_code, kernel_name = extract_kernel_from_hip_code(hip_code)
            return PromptSource(
                starter_code=hip_code,
                kernel_name=kernel_name,
                starter_code_kind="full_file",
                expected_code_unit=self.expected_code_unit,
                persistence_mode=self.persistence_mode,
            )

        kernel_code, kernel_name = extract_kernel_from_hip_code(hip_code)
        return PromptSource(
            starter_code=kernel_code or hip_code,
            kernel_name=kernel_name,
            starter_code_kind="kernel" if kernel_code is not None else "full_file_fallback",
            expected_code_unit=self.expected_code_unit,
            persistence_mode=self.persistence_mode,
        )

    def extract_previous_generated_code(self, hip_code: str) -> str:
        if self.optimization_paradigm == HIP2HIP_FULL_FILE_PARADIGM:
            return hip_code
        kernel_code, _ = extract_kernel_from_hip_code(hip_code)
        return kernel_code or hip_code

    def persist_code(self, original_hip_code: str, parsed_hip_src: str, kernel_name: Optional[str]) -> str:
        if self.optimization_paradigm == HIP2HIP_FULL_FILE_PARADIGM:
            return parsed_hip_src
        return replace_kernel_in_hip_code(
            original_hip_code,
            parsed_hip_src,
            kernel_name=kernel_name,
        )


def get_generation_paradigm_policy(
    optimization_paradigm: str = DEFAULT_OPTIMIZATION_PARADIGM,
) -> GenerationParadigmPolicy:
    normalized = normalize_optimization_paradigm(optimization_paradigm)
    if normalized == HIP2HIP_FULL_FILE_PARADIGM:
        return GenerationParadigmPolicy(
            optimization_paradigm=normalized,
            expected_code_unit=HIP_TRANSLATION_UNIT_CODE_UNIT,
            persistence_mode=FULL_FILE_PERSISTENCE_MODE,
        )
    if normalized == KERNEL2KERNEL_SPLICE_PARADIGM:
        return GenerationParadigmPolicy(
            optimization_paradigm=normalized,
            expected_code_unit=KERNEL_FUNCTION_CODE_UNIT,
            persistence_mode=SPLICE_PERSISTENCE_MODE,
        )
    raise ValueError(f"Unsupported optimization_paradigm: {optimization_paradigm!r}")
