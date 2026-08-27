# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared kernel-agent response parsing helpers."""

from __future__ import annotations

from typing import Any, Optional

from reward.utils import (
    normalize_output_contract,
    parse_generation_response,
    parse_kernel_generation_response,
    strip_code_fences,
)

KERNEL_AGENT_DEFAULT_DATA_SOURCE = "kernel-agent-single-sft-train"
KERNEL_AGENT_REACT_DATA_SOURCE = "kernel-agent-react-train"
KERNEL_AGENT_PARSE_SOURCES = {
    KERNEL_AGENT_DEFAULT_DATA_SOURCE,
    KERNEL_AGENT_REACT_DATA_SOURCE,
}


def parse_kernel_agent_generation_response(
    raw_response: Any,
    *,
    output_contract: Optional[str],
    kernel_name: Optional[str],
    hip_ref: str,
    data_source: str = KERNEL_AGENT_DEFAULT_DATA_SOURCE,
    expected_code_unit: Optional[str] = None,
) -> dict:
    """Mirror the kernel-agent parse gate used in reward_batch."""
    normalized_contract = normalize_output_contract(output_contract)
    if expected_code_unit:
        return parse_generation_response(
            raw_response,
            data_source=data_source,
            kernel_name=kernel_name,
            hip_ref=hip_ref,
            output_contract=output_contract,
            expected_code_unit=expected_code_unit,
        )
    if data_source in KERNEL_AGENT_PARSE_SOURCES:
        return parse_kernel_generation_response(
            raw_response,
            data_source=data_source,
            kernel_name=kernel_name,
            hip_ref=hip_ref,
            output_contract=output_contract,
        )
    return {
        "hip_src": strip_code_fences(raw_response),
        "parse_mode": "raw_strip_fence",
        "parse_ok": True,
        "parse_error": "",
        "output_contract": normalized_contract,
        "attempted_parse_modes": ["raw_strip_fence"],
        "parse_attempt_chain": "raw_strip_fence",
    }


def build_parse_attempt_chain(parse_result: Optional[dict]) -> str:
    """Mirror reward_batch._parse_attempt_chain for consistent diagnostics."""
    result = parse_result or {}
    chain = str(result.get("parse_attempt_chain") or "").strip()
    if chain:
        return chain
    attempted_modes = result.get("attempted_parse_modes") or []
    if attempted_modes:
        return "->".join(str(mode) for mode in attempted_modes if mode)
    return str(result.get("parse_mode") or "").strip()
