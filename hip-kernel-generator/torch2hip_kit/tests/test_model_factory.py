# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch2hip_kit.model_factory as model_factory


def test_model_factory_accepts_legacy_env_var(monkeypatch) -> None:
    monkeypatch.delenv("TORCH2HIP_API_KEY", raising=False)
    monkeypatch.setenv("TORCH_MODU2FUNC_API_KEY", "legacy-key")

    created = {}

    class FakeStandardOpenAIModel:
        def __init__(self, model_id, api_key):
            created["model_id"] = model_id
            created["api_key"] = api_key

    monkeypatch.setattr("torch2hip_kit.model_clients.StandardOpenAIModel", FakeStandardOpenAIModel)

    model_factory.create_model_client("standard-openai", "gpt-test", None)

    assert created == {"model_id": "gpt-test", "api_key": "legacy-key"}
