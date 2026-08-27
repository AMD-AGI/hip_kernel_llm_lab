# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import json
import logging
import os
import sys
import typing as T

import numpy as np


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _color(text: T.Any, *codes: str) -> str:
    rendered = str(text)
    if not codes or not _supports_color():
        return rendered
    return f"\033[{';'.join(codes)}m{rendered}\033[0m"


def _reward_tone(value: float) -> tuple[str, ...]:
    if value <= 0.0:
        return ("1", "31")  # red
    if value < 1.0:
        return ("1", "33")  # yellow
    return ("1", "32")  # green


def _format_hist_line(value: float, count: int, total: int) -> str:
    percentage = (count / total * 100.0) if total else 0.0
    return (
        f"    {_color(f'{value:7.4f}', *_reward_tone(value))}: "
        f"{_color(f'{count:3d}', '1', '97')} samples "
        f"({_color(f'{percentage:5.1f}%', '1', '96')})"
    )


def emit_reward_distribution_debug(
    batch_rewards: T.Sequence[T.Any],
    uids: T.Sequence[T.Any],
    kernel_names: T.Sequence[str],
    *,
    logger: logging.Logger | None = None,
) -> None:
    rewards = np.asarray(batch_rewards, dtype=np.float64)
    if rewards.size == 0:
        return

    uid_array = np.asarray([str(uid) for uid in uids], dtype=object)
    if uid_array.shape[0] != rewards.shape[0]:
        uid_array = np.array([str(i) for i in range(rewards.shape[0])], dtype=object)

    kernels = list(kernel_names)
    if len(kernels) < rewards.shape[0]:
        kernels.extend(["unknown"] * (rewards.shape[0] - len(kernels)))

    width = 78
    line = _color("=" * width, "2", "36")
    print(f"\n{line}")
    print(_color(f"[Reward Distribution] Batch size: {rewards.size}", "1", "96"))
    print(line)

    print(_color("  Reward Statistics:", "1", "94"))
    print(f"    Mean: {_color(f'{rewards.mean():.4f}', '1', '97')}")
    print(f"    Std:  {_color(f'{rewards.std():.4f}', '1', '97')}")
    print(f"    Min:  {_color(f'{rewards.min():.4f}', *_reward_tone(float(rewards.min())))}")
    print(f"    Max:  {_color(f'{rewards.max():.4f}', *_reward_tone(float(rewards.max())))}")

    unique_rewards, reward_counts = np.unique(rewards, return_counts=True)
    print(_color("\n  Reward Value Distribution:", "1", "94"))
    for value, count in zip(unique_rewards, reward_counts):
        print(_format_hist_line(float(value), int(count), int(rewards.size)))

    unique_uids = np.unique(uid_array)
    print(_color("\n  Group Statistics:", "1", "94"))
    print(f"    Total samples: {_color(len(rewards), '1', '97')}")
    print(f"    Unique prompts: {_color(len(unique_uids), '1', '97')}")
    sample_per_prompt = (len(rewards) / len(unique_uids)) if len(unique_uids) else 0.0
    print(f"    Samples per prompt: {_color(f'{sample_per_prompt:.1f}', '1', '97')}")

    print(_color("\n  Per-Group Reward Distribution (all groups):", "1", "95"))
    group_records: list[dict[str, T.Any]] = []
    for uid in unique_uids:
        group_mask = uid_array == uid
        group_rewards = rewards[group_mask]
        group_indices = np.where(group_mask)[0]
        kernel_name = kernels[int(group_indices[0])] if group_indices.size else "unknown"
        mean_val = float(group_rewards.mean()) if group_rewards.size else 0.0
        std_val = float(group_rewards.std()) if group_rewards.size else 0.0

        print(f"    {_color('Group', '1', '94')} {_color(uid, '1', '97')} ({_color(kernel_name, '1', '36')}):")
        print(f"      Rewards: {_color(group_rewards.tolist(), '0', '37')}")
        print(
            "      "
            f"Mean: {_color(f'{mean_val:.4f}', *_reward_tone(mean_val))}, "
            f"Std: {_color(f'{std_val:.4f}', '1', '97')}"
        )

        group_records.append(
            {
                "uid": str(uid),
                "kernel_name": str(kernel_name),
                "rewards": [float(v) for v in group_rewards.tolist()],
                "mean": mean_val,
                "std": std_val,
                "count": int(group_rewards.size),
            }
        )

    compile_fail = int(np.sum(rewards == -0.9))
    run_fail = int(np.sum(rewards == -0.5))
    match_fail = int(np.sum(rewards == 0.0))
    success = int(np.sum(rewards > 0.0))

    total = float(len(rewards)) if len(rewards) else 1.0
    print(_color("\n  Failure Analysis:", "1", "94"))
    print(
        f"    Compile failed: {_color(compile_fail, '1', '31')} "
        f"({_color(f'{compile_fail / total * 100:.1f}%', '1', '31')})"
    )
    print(
        f"    Run failed:     {_color(run_fail, '1', '31')} "
        f"({_color(f'{run_fail / total * 100:.1f}%', '1', '31')})"
    )
    print(
        f"    Match failed:   {_color(match_fail, '1', '33')} "
        f"({_color(f'{match_fail / total * 100:.1f}%', '1', '33')})"
    )
    print(
        f"    Success:        {_color(success, '1', '32')} "
        f"({_color(f'{success / total * 100:.1f}%', '1', '32')})"
    )
    print(f"{line}\n")

    payload = {
        "batch_size": int(rewards.size),
        "group_count": int(len(unique_uids)),
        "groups": group_records,
    }
    # Keep a complete, machine-readable per-group log for post-mortem analysis.
    print(f"[PER_GROUP_REWARD_LOG_JSON] {json.dumps(payload, ensure_ascii=True, sort_keys=True)}")

    active_logger = logger or logging.getLogger(__name__)
    active_logger.info("[PER_GROUP_REWARD_LOG_JSON] %s", json.dumps(payload, ensure_ascii=True, sort_keys=True))
