"""Shared optimization contract helpers for dataset, launch, and reward code.

This module is intentionally small and side-effect free. It centralizes the
semantic differences between kernel-splice and full-file HIP optimization so
callers do not each re-implement their own data_source string checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .prompts import (
    DEFAULT_OPTIMIZATION_PARADIGM,
    HIP2HIP_FULL_FILE_PARADIGM,
    KERNEL2KERNEL_SPLICE_PARADIGM,
    normalize_optimization_paradigm,
)

AUTO_OUTPUT_CONTRACT = "auto"
SAMPLE_JSON_OUTPUT_CONTRACT = "sample_json_v1"
LEGACY_HIP_FENCE_OUTPUT_CONTRACT = "legacy_hip_fence_v1"

KERNEL_FUNCTION_CODE_UNIT = "kernel_function"
HIP_TRANSLATION_UNIT_CODE_UNIT = "hip_translation_unit"
SPLICE_PERSISTENCE_MODE = "splice_kernel"
FULL_FILE_PERSISTENCE_MODE = "direct_full_file"

HIP2HIP_TRAIN_DATA_SOURCE = "hip2hip-train"
KERNEL2KERNEL_TRAIN_DATA_SOURCE = "kernel2kernel-train"
KERNEL_AGENT_SINGLE_SFT_DATA_SOURCE = "kernel-agent-single-sft-train"
KERNEL_AGENT_REACT_DATA_SOURCE = "kernel-agent-react-train"
TORCH2HIP_TRAIN_DATA_SOURCE = "torch2hip-train"

SUPPORTED_REWARD_TRAIN_DATA_SOURCES = {
    TORCH2HIP_TRAIN_DATA_SOURCE,
    HIP2HIP_TRAIN_DATA_SOURCE,
    KERNEL2KERNEL_TRAIN_DATA_SOURCE,
    KERNEL_AGENT_SINGLE_SFT_DATA_SOURCE,
    KERNEL_AGENT_REACT_DATA_SOURCE,
}


@dataclass(frozen=True)
class OptimizationContract:
    data_source: str
    output_contract: str
    optimization_paradigm: str
    expected_code_unit: str
    persistence_mode: str
    requires_kernel_splice: bool
    requires_strict_parse_gate: bool


def normalize_output_contract(output_contract: Optional[str]) -> str:
    normalized = str(output_contract or "").strip().lower()
    if not normalized or normalized in {"auto", "none"}:
        return AUTO_OUTPUT_CONTRACT
    if normalized in {
        SAMPLE_JSON_OUTPUT_CONTRACT,
        "sample_json",
        "sample-json",
        "json",
    }:
        return SAMPLE_JSON_OUTPUT_CONTRACT
    if normalized in {
        LEGACY_HIP_FENCE_OUTPUT_CONTRACT,
        "legacy_hip_fence",
        "legacy-hip-fence",
        "hip_fence",
        "fenced_hip",
    }:
        return LEGACY_HIP_FENCE_OUTPUT_CONTRACT
    return normalized


def normalize_expected_code_unit(expected_code_unit: Optional[str]) -> str:
    normalized = str(expected_code_unit or KERNEL_FUNCTION_CODE_UNIT).strip().lower()
    aliases = {
        "kernel": KERNEL_FUNCTION_CODE_UNIT,
        "kernel_snippet": KERNEL_FUNCTION_CODE_UNIT,
        "kernel-function": KERNEL_FUNCTION_CODE_UNIT,
        "function": KERNEL_FUNCTION_CODE_UNIT,
        "hip": HIP_TRANSLATION_UNIT_CODE_UNIT,
        "hip_file": HIP_TRANSLATION_UNIT_CODE_UNIT,
        "hip-file": HIP_TRANSLATION_UNIT_CODE_UNIT,
        "full_file": HIP_TRANSLATION_UNIT_CODE_UNIT,
        "full-file": HIP_TRANSLATION_UNIT_CODE_UNIT,
        "translation_unit": HIP_TRANSLATION_UNIT_CODE_UNIT,
        "translation-unit": HIP_TRANSLATION_UNIT_CODE_UNIT,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {KERNEL_FUNCTION_CODE_UNIT, HIP_TRANSLATION_UNIT_CODE_UNIT}:
        supported = ", ".join(sorted({KERNEL_FUNCTION_CODE_UNIT, HIP_TRANSLATION_UNIT_CODE_UNIT}))
        raise ValueError(f"Unsupported expected_code_unit={expected_code_unit!r}; supported values: {supported}")
    return normalized


def infer_optimization_paradigm(data_source: str, requested: str = "") -> str:
    if requested:
        return normalize_optimization_paradigm(requested)
    normalized_source = (data_source or "").strip().lower()
    if normalized_source.startswith("hip2hip") or normalized_source.startswith("torch2hip"):
        return HIP2HIP_FULL_FILE_PARADIGM
    if (
        normalized_source.startswith("kernel2kernel")
        or normalized_source.startswith("kernel-agent")
    ):
        return KERNEL2KERNEL_SPLICE_PARADIGM
    return DEFAULT_OPTIMIZATION_PARADIGM


def expected_code_unit_for_paradigm(optimization_paradigm: str) -> str:
    normalized = normalize_optimization_paradigm(optimization_paradigm)
    return (
        HIP_TRANSLATION_UNIT_CODE_UNIT
        if normalized == HIP2HIP_FULL_FILE_PARADIGM
        else KERNEL_FUNCTION_CODE_UNIT
    )


def _infer_paradigm_from_explicit_code_unit(expected_code_unit: str) -> str:
    return (
        HIP2HIP_FULL_FILE_PARADIGM
        if expected_code_unit == HIP_TRANSLATION_UNIT_CODE_UNIT
        else KERNEL2KERNEL_SPLICE_PARADIGM
    )


def resolve_optimization_contract(
    data_source: str,
    extra_info: Optional[Mapping[str, Any]] = None,
    requested_paradigm: str = "",
    output_contract: Optional[str] = None,
) -> OptimizationContract:
    extra = extra_info or {}
    normalized_source = str(data_source or extra.get("data_source") or "").strip().lower()
    if not normalized_source:
        raise ValueError("data_source is required to resolve an optimization contract")

    explicit_code_unit = extra.get("expected_code_unit")
    expected_code_unit = (
        normalize_expected_code_unit(str(explicit_code_unit))
        if explicit_code_unit
        else ""
    )

    paradigm_value = requested_paradigm or str(extra.get("optimization_paradigm") or "")
    if paradigm_value:
        optimization_paradigm = normalize_optimization_paradigm(paradigm_value)
    elif expected_code_unit:
        optimization_paradigm = _infer_paradigm_from_explicit_code_unit(expected_code_unit)
    else:
        optimization_paradigm = infer_optimization_paradigm(normalized_source)

    inferred_code_unit = expected_code_unit_for_paradigm(optimization_paradigm)
    if expected_code_unit and expected_code_unit != inferred_code_unit:
        raise ValueError(
            "optimization contract mismatch: "
            f"optimization_paradigm={optimization_paradigm!r} implies "
            f"expected_code_unit={inferred_code_unit!r}, got {expected_code_unit!r}"
        )
    expected_code_unit = expected_code_unit or inferred_code_unit

    if (
        normalized_source.startswith("hip2hip")
        or normalized_source.startswith("torch2hip")
    ) and optimization_paradigm != HIP2HIP_FULL_FILE_PARADIGM:
        raise ValueError(
            f"data_source={normalized_source!r} requires optimization_paradigm={HIP2HIP_FULL_FILE_PARADIGM!r}"
        )
    if (
        normalized_source.startswith("kernel2kernel")
        or normalized_source.startswith("kernel-agent")
    ) and optimization_paradigm == HIP2HIP_FULL_FILE_PARADIGM:
        raise ValueError(
            f"data_source={normalized_source!r} is a kernel-splice source and cannot use "
            f"optimization_paradigm={HIP2HIP_FULL_FILE_PARADIGM!r}"
        )

    normalized_output_contract = normalize_output_contract(
        output_contract if output_contract is not None else extra.get("output_contract")
    )
    if optimization_paradigm == HIP2HIP_FULL_FILE_PARADIGM:
        return OptimizationContract(
            data_source=normalized_source,
            output_contract=normalized_output_contract,
            optimization_paradigm=optimization_paradigm,
            expected_code_unit=HIP_TRANSLATION_UNIT_CODE_UNIT,
            persistence_mode=FULL_FILE_PERSISTENCE_MODE,
            requires_kernel_splice=False,
            requires_strict_parse_gate=True,
        )

    return OptimizationContract(
        data_source=normalized_source,
        output_contract=normalized_output_contract,
        optimization_paradigm=optimization_paradigm,
        expected_code_unit=KERNEL_FUNCTION_CODE_UNIT,
        persistence_mode=SPLICE_PERSISTENCE_MODE,
        requires_kernel_splice=normalized_source in {
            KERNEL2KERNEL_TRAIN_DATA_SOURCE,
            KERNEL_AGENT_SINGLE_SFT_DATA_SOURCE,
            KERNEL_AGENT_REACT_DATA_SOURCE,
        },
        requires_strict_parse_gate=normalized_source in {
            KERNEL_AGENT_SINGLE_SFT_DATA_SOURCE,
            KERNEL_AGENT_REACT_DATA_SOURCE,
        },
    )


def contract_to_extra_info(contract: OptimizationContract) -> dict[str, str]:
    return {
        "output_contract": contract.output_contract,
        "optimization_paradigm": contract.optimization_paradigm,
        "expected_code_unit": contract.expected_code_unit,
        "persistence_mode": contract.persistence_mode,
    }


def validate_training_row_contract(
    row: Mapping[str, Any],
    expected_contract: Optional[OptimizationContract] = None,
) -> list[str]:
    errors: list[str] = []
    data_source = str(row.get("data_source") or "").strip()
    extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), Mapping) else {}
    try:
        contract = resolve_optimization_contract(data_source=data_source, extra_info=extra_info)
    except Exception as exc:
        return [f"invalid optimization contract: {exc}"]

    if expected_contract is not None:
        for field_name in (
            "data_source",
            "output_contract",
            "optimization_paradigm",
            "expected_code_unit",
            "persistence_mode",
        ):
            actual_value = getattr(contract, field_name)
            expected_value = getattr(expected_contract, field_name)
            if actual_value != expected_value:
                errors.append(
                    f"{field_name} mismatch: expected {expected_value!r}, got {actual_value!r}"
                )

    reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), Mapping) else {}
    ground_truth = (
        reward_model.get("ground_truth")
        if isinstance(reward_model.get("ground_truth"), Mapping)
        else {}
    )
    if not ground_truth.get("hip_code"):
        errors.append("reward_model.ground_truth.hip_code is required")
    if not ground_truth.get("pytorch_module_code"):
        errors.append("reward_model.ground_truth.pytorch_module_code is required")
    if not ground_truth.get("pytorch_functional_code"):
        errors.append("reward_model.ground_truth.pytorch_functional_code is required")

    if contract.expected_code_unit == HIP_TRANSLATION_UNIT_CODE_UNIT:
        prompt = row.get("prompt")
        prompt_text = ""
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], Mapping):
            prompt_text = str(prompt[0].get("content") or "")
        if "Starter HIP File (reference)" not in prompt_text:
            errors.append("hip2hip prompt must include a full-file starter section")
        hip_code = str(ground_truth.get("hip_code") or "")
        if "__global__" not in hip_code and "hipLaunchKernelGGL" not in hip_code:
            errors.append("hip2hip ground truth must look like a complete HIP translation unit")

    return errors
