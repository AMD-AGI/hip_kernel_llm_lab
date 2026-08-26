# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import os

from models import (
    ClaudeModel,
    GeminiModel,
    OpenAIModel,
    StandardClaudeModel,
    StandardOpenAIModel,
)


def create_model_client(provider: str, model_id: str, api_key: str | None):
    resolved_api_key = api_key or os.getenv("TORCH_MODU2FUNC_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "An API key is required. Pass --api-key or set TORCH_MODU2FUNC_API_KEY."
        )

    normalized = provider.strip().lower()
    if normalized == "openai":
        return OpenAIModel(api_key=resolved_api_key, model_id=model_id)
    if normalized == "standard-openai":
        return StandardOpenAIModel(api_key=resolved_api_key, model_id=model_id)
    if normalized == "claude":
        return ClaudeModel(api_key=resolved_api_key, model_id=model_id)
    if normalized == "standard-claude":
        return StandardClaudeModel(api_key=resolved_api_key, model_id=model_id)
    if normalized == "gemini":
        return GeminiModel(api_key=resolved_api_key, model_id=model_id)

    supported = ["openai", "standard-openai", "claude", "standard-claude", "gemini"]
    raise ValueError(f"Unsupported provider '{provider}'. Expected one of {supported}.")
