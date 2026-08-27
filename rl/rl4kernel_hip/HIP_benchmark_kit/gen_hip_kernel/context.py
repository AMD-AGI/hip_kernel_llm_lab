# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prompt-context loading and resolution."""

from __future__ import annotations

import json
import os
from typing import Any


def _extract_prompt_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("prompt_text", "context", "card_text", "profile_card", "text"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    raise ValueError(f"Unsupported prompt-map entry: {type(value).__name__}")


def load_prompt_map(prompt_map_json: str, prompt_map_arm: str = "") -> dict[str, str]:
    if not prompt_map_json:
        return {}

    with open(prompt_map_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    selected = payload
    if prompt_map_arm and isinstance(payload, dict) and "arms" in payload:
        arms = payload.get("arms") or {}
        if prompt_map_arm not in arms:
            available = ", ".join(sorted(str(key) for key in arms))
            raise KeyError(f"Prompt arm {prompt_map_arm!r} not found. Available arms: {available}")
        selected = arms[prompt_map_arm]

    if isinstance(selected, dict):
        for key in ("prompt_map", "prompts", "cards"):
            if isinstance(selected.get(key), dict):
                selected = selected[key]
                break

    if not isinstance(selected, dict):
        raise ValueError(f"Prompt map must resolve to a JSON object, got {type(selected).__name__}")

    return {str(key): _extract_prompt_text(value) for key, value in selected.items()}


def load_feedback_context(feedback_context_json: str) -> dict[str, dict[str, Any]]:
    if not feedback_context_json:
        return {}
    with open(feedback_context_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    selected = payload.get("feedback_map", payload) if isinstance(payload, dict) else payload
    if not isinstance(selected, dict):
        raise ValueError(f"Feedback context must resolve to a JSON object, got {type(selected).__name__}")
    contexts: dict[str, dict[str, Any]] = {}
    for key, value in selected.items():
        if not isinstance(value, dict):
            raise ValueError(f"Unsupported feedback context entry for {key!r}: {type(value).__name__}")
        contexts[str(key)] = value
    return contexts


def resolve_feedback_context(feedback_context: dict[str, dict[str, Any]], input_file: str) -> tuple[dict[str, Any], str]:
    stem = os.path.splitext(os.path.basename(input_file))[0]
    candidates = (input_file, os.path.basename(input_file), stem)
    for key in candidates:
        if key in feedback_context:
            return feedback_context[key], key
    return {}, ""


def resolve_prompt_text(base_prompt_text: str, prompt_map: dict[str, str], input_file: str) -> tuple[str, str]:
    stem = os.path.splitext(os.path.basename(input_file))[0]
    candidates = (input_file, os.path.basename(input_file), stem)
    prompt_parts = []
    base_prompt_text = base_prompt_text or ""
    if base_prompt_text.strip():
        prompt_parts.append(base_prompt_text.strip())

    matched_key = ""
    for key in candidates:
        if key in prompt_map:
            matched_key = key
            mapped_prompt = prompt_map[key]
            if mapped_prompt.strip():
                prompt_parts.append(mapped_prompt.strip())
            break

    return "\n\n".join(prompt_parts), matched_key
