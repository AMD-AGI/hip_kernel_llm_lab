# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import torch

from verl.protocol import DataProto
from verl.workers.reward_manager.batch_parallel import BatchParallelRewardManager


class DummyTokenizer:
    eos_token = None

    def decode(self, token_ids, skip_special_tokens=True):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return " ".join(str(token) for token in token_ids)


def test_batch_parallel_reward_manager_enriches_extra_info_with_train_metadata():
    captured = {}

    def fake_compute_score_batch(**kwargs):
        captured.update(kwargs)
        return [0.0, 0.0]

    manager = BatchParallelRewardManager(
        tokenizer=DummyTokenizer(),
        num_examine=0,
        compute_score_batch=fake_compute_score_batch,
    )

    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "responses": torch.tensor([[5, 6], [7, 8]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long),
        },
        non_tensors={
            "reward_model": np.array(
                [
                    {"ground_truth": {"kernel_name": "kernel_a"}},
                    {"ground_truth": {"kernel_name": "kernel_b"}},
                ],
                dtype=object,
            ),
            "data_source": np.array(
                ["kernel-agent-single-sft-train", "kernel-agent-single-sft-train"],
                dtype=object,
            ),
            "extra_info": np.array(
                [
                    {"sandbox_url": "http://mock:8000/run_code", "custom_tag": "keep-me"},
                    {},
                ],
                dtype=object,
            ),
            "uid": np.array(["uid-a", "uid-b"], dtype=object),
            "index": np.array([11, 22], dtype=object),
        },
        meta_info={"train_step": 42},
    )

    scores, decoded_data = manager.verify_batch(batch)

    assert scores == [0.0, 0.0]
    assert len(decoded_data) == 2
    assert captured["solution_strs"] == ["5 6", "7 8"]
    assert captured["ground_truths"][0]["kernel_name"] == "kernel_a"

    extra_infos = captured["extra_infos"]
    assert extra_infos[0]["train_step"] == 42
    assert extra_infos[0]["prompt_uid"] == "uid-a"
    assert extra_infos[0]["sample_index"] == 11
    assert extra_infos[0]["index"] == 11
    assert extra_infos[0]["custom_tag"] == "keep-me"

    assert extra_infos[1]["train_step"] == 42
    assert extra_infos[1]["prompt_uid"] == "uid-b"
    assert extra_infos[1]["sample_index"] == 22
    assert extra_infos[1]["index"] == 22
