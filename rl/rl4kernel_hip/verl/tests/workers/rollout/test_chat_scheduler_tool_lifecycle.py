# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.workers.rollout.chat_scheduler import ToolCompletionCallback


class DummyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)


class TokenizingDummyTokenizer:
    eos_token_id = 999
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<{message['role']}>{message.get('content', '')}§" for message in messages)

    def _encode(self, text):
        return [self.eos_token_id if ch == "§" else (ord(ch) % 251) + 1 for ch in text]

    def __call__(self, texts, return_tensors="pt", padding="longest", padding_side="right"):
        encoded = [self._encode(text) for text in texts]
        max_len = max(len(item) for item in encoded)
        input_ids, attention_mask = [], []
        for item in encoded:
            pad_len = max_len - len(item)
            if padding_side == "left":
                input_ids.append([self.pad_token_id] * pad_len + item)
                attention_mask.append([0] * pad_len + [1] * len(item))
            else:
                input_ids.append(item + [self.pad_token_id] * pad_len)
                attention_mask.append([1] * len(item) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class DummyScheduler:
    def __init__(self):
        self.submit_calls = []

    def submit_chat_completions(self, *, messages, request_id, info):
        self.submit_calls.append(
            {
                "messages": messages,
                "request_id": request_id,
                "info": info,
            }
        )


class FakeKernelTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.create_count = 0
        self.execute_count = 0
        self.release_count = 0
        self.last_instance_id = None

    async def create(self, instance_id=None, **kwargs):
        self.create_count += 1
        self.last_instance_id = instance_id
        return instance_id

    async def execute(self, instance_id, parameters, **kwargs):
        self.execute_count += 1
        return json.dumps({"instance_id": instance_id, "parameters": parameters}), 0.0, {}

    async def release(self, instance_id, **kwargs):
        self.release_count += 1


def _tool_schema():
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "kernel_eval",
                "description": "fake kernel evaluation tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                    },
                    "required": ["action"],
                },
            },
        }
    )


def _fake_completion():
    tool_call = SimpleNamespace(
        id="tool-call-1",
        function=SimpleNamespace(name="kernel_eval", arguments=json.dumps({"action": "compile_check"})),
    )
    message = SimpleNamespace(
        tool_calls=[tool_call],
        model_dump=lambda exclude_unset=True, exclude_none=True: {"role": "assistant", "content": ""},
    )
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(id="chatcmpl-test", choices=[choice])


@pytest.mark.asyncio
async def test_tool_completion_callback_reuses_request_scoped_tool(monkeypatch):
    monkeypatch.setattr("verl.workers.rollout.chat_scheduler.copy_to_local", lambda path: path)
    monkeypatch.setattr("verl.workers.rollout.chat_scheduler.hf_tokenizer", lambda path, trust_remote_code=True: DummyTokenizer())

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": "/tmp/model"},
                "rollout": {
                    "multi_turn": {
                        "max_assistant_turns": 8,
                        "tool_config_path": None,
                        "enable_thinking": True,
                        "max_tool_calls": 4,
                        "max_tool_wallclock_s": 60,
                    }
                },
            }
        }
    )
    scheduler = DummyScheduler()
    callback = ToolCompletionCallback(config, scheduler)
    fake_tool = FakeKernelTool({}, _tool_schema())
    callback.tools = {"kernel_eval": fake_tool}

    info = {
        "__request_key__": "req-1",
        "__request_started_at__": time.time(),
        "__tool_calls__": 0,
        "__tools_kwargs__": {
            "kernel_eval": {
                "create_kwargs": {"reference": {"kernel_name": "k", "hip_ref_code": "__global__ void k() {}"}},
                "release_kwargs": {},
            }
        },
    }
    messages = [{"role": "user", "content": "test"}]
    completion = _fake_completion()

    await callback(messages, completion, info)
    assert fake_tool.create_count == 1
    assert fake_tool.execute_count == 1
    assert fake_tool.last_instance_id == "req-1"
    assert len(scheduler.submit_calls) == 1

    await callback(messages, completion, info)
    assert fake_tool.create_count == 1
    assert fake_tool.execute_count == 2

    await callback.on_request_complete(info)
    assert fake_tool.release_count == 1


def test_tool_completion_callback_postprocess_emits_full_loss_mask(monkeypatch):
    monkeypatch.setattr("verl.workers.rollout.chat_scheduler.copy_to_local", lambda path: path)
    monkeypatch.setattr("verl.workers.rollout.chat_scheduler.hf_tokenizer", lambda path, trust_remote_code=True: TokenizingDummyTokenizer())

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": "/tmp/model"},
                "rollout": {
                    "multi_turn": {
                        "max_assistant_turns": 8,
                        "tool_config_path": None,
                        "enable_thinking": True,
                        "max_tool_calls": 4,
                        "max_tool_wallclock_s": 60,
                    }
                },
            }
        }
    )
    callback = ToolCompletionCallback(config, DummyScheduler())

    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(
                [
                    np.array(
                        [
                            {"role": "user", "content": "hi"},
                        ],
                        dtype=object,
                    )
                ],
                dtype=object,
            )
        }
    )
    batch_conversations = [
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "draft"},
            {"role": "tool", "content": "compiled"},
            {"role": "assistant", "content": "final"},
        ]
    ]

    result = callback.postprocess(batch, batch_conversations, n=1)

    assert "loss_mask" in result.batch.keys()
    assert result.batch["loss_mask"].shape == result.batch["input_ids"].shape
    assert result.batch["response_mask"].shape == result.batch["responses"].shape

    prompt_width = result.batch["prompts"].shape[1]
    response_width = result.batch["responses"].shape[1]
    assert torch.count_nonzero(result.batch["loss_mask"][:, :prompt_width]) == 0
    assert torch.equal(result.batch["loss_mask"][:, -response_width:], result.batch["response_mask"])
    assert torch.any(result.batch["response_mask"][0] == 0)
    assert torch.any(result.batch["response_mask"][0] == 1)
