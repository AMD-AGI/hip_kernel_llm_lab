# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generation manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from dataset.prompts import DEFAULT_OPTIMIZATION_PARADIGM, DEFAULT_TARGET_GPU
from dataset.response_parser import KERNEL_AGENT_DEFAULT_DATA_SOURCE

from .cli import DEFAULT_OUTPUT_CONTRACT

ROLLOUT_STRATEGY = "serial_n1_per_rollout_idx"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_rollout_indices(value: str, rollout_n: int) -> list[int]:
    if not value:
        return list(range(int(rollout_n)))

    indices: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid rollout index range: {token}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(token))

    deduped = sorted(set(indices))
    invalid = [idx for idx in deduped if idx < 0 or idx >= int(rollout_n)]
    if invalid:
        raise ValueError(f"Rollout indices out of range for rollout_n={rollout_n}: {invalid}")
    return deduped


def load_existing_manifest_records(output_dir: str) -> list[dict[str, Any]]:
    manifest_path = os.path.join(output_dir, "generation_manifest.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Existing generation manifest has non-list records: {manifest_path}")
    return records


def merge_generation_records(existing_records: list[dict], fresh_records: list[dict]) -> list[dict]:
    merged: dict[tuple[str, int], dict] = {}
    passthrough: list[dict] = []
    for record in existing_records + fresh_records:
        input_file = record.get("input_file")
        sample_idx = record.get("sample_idx")
        if input_file is None or sample_idx is None:
            passthrough.append(record)
            continue
        merged[(str(input_file), int(sample_idx))] = record
    return passthrough + [merged[key] for key in sorted(merged, key=lambda item: (item[0], item[1]))]


def write_generation_manifest(output_dir: str, config, records: list[dict]) -> str:
    manifest_path = os.path.join(output_dir, "generation_manifest.json")
    reused_records = [record for record in records if record.get("reused")]
    fresh_records = [record for record in records if not record.get("reused")]
    prompt_map_json = config.generation.get("prompt_map_json", "")
    prompt_map_sha256 = sha256_file(prompt_map_json) if prompt_map_json and os.path.exists(prompt_map_json) else ""
    feedback_context_json = config.generation.get("feedback_context_json", "")
    feedback_context_sha256 = sha256_file(feedback_context_json) if feedback_context_json and os.path.exists(feedback_context_json) else ""
    identity_payload = {
        "model_path": config.model.path,
        "input_dir": config.data.input_dir,
        "output_contract": config.generation.get("output_contract", DEFAULT_OUTPUT_CONTRACT),
        "optimization_paradigm": config.generation.get("optimization_paradigm", DEFAULT_OPTIMIZATION_PARADIGM),
        "target_gpu": config.generation.get("target_gpu", DEFAULT_TARGET_GPU),
        "data_source": config.generation.get("data_source", KERNEL_AGENT_DEFAULT_DATA_SOURCE),
        "experiment_arm": config.generation.get("experiment_arm", ""),
        "prompt_map_arm": config.generation.get("prompt_map_arm", ""),
        "prompt_map_sha256": prompt_map_sha256,
        "feedback_context_sha256": feedback_context_sha256,
        "rollout_strategy": ROLLOUT_STRATEGY,
        "seed_base": config.generation.get("seed_base", None),
        "temperature": config.generation.get("temperature", None),
        "top_p": config.generation.get("top_p", None),
        "top_k": config.generation.get("top_k", None),
    }
    identity_hash = sha256_text(json.dumps(identity_payload, sort_keys=True, separators=(",", ":")))
    manifest = {
        "model_path": config.model.path,
        "input_dir": config.data.input_dir,
        "output_dir": config.data.output_dir,
        "output_contract": config.generation.get("output_contract", DEFAULT_OUTPUT_CONTRACT),
        "optimization_paradigm": config.generation.get("optimization_paradigm", DEFAULT_OPTIMIZATION_PARADIGM),
        "target_gpu": config.generation.get("target_gpu", DEFAULT_TARGET_GPU),
        "data_source": config.generation.get("data_source", KERNEL_AGENT_DEFAULT_DATA_SOURCE),
        "experiment_arm": config.generation.get("experiment_arm", ""),
        "prompt_map_json": config.generation.get("prompt_map_json", ""),
        "prompt_map_arm": config.generation.get("prompt_map_arm", ""),
        "prompt_map_sha256": prompt_map_sha256,
        "feedback_context_json": feedback_context_json,
        "feedback_context_sha256": feedback_context_sha256,
        "raw_response_dir": config.generation.get("raw_response_dir", ""),
        "rollout_n": config.rollout.get("n", 1),
        "rollout_strategy": ROLLOUT_STRATEGY,
        "seed_base": config.generation.get("seed_base", None),
        "temperature": config.generation.get("temperature", None),
        "top_p": config.generation.get("top_p", None),
        "top_k": config.generation.get("top_k", None),
        "generation_identity": identity_payload,
        "generation_identity_hash": identity_hash,
        "reuse_summary": {
            "identity_hash": identity_hash,
            "reused_record_count": len(reused_records),
            "fresh_record_count": len(fresh_records),
            "reuse_sources": sorted(
                {
                    record.get("reuse_source_run_root", "")
                    for record in reused_records
                    if record.get("reuse_source_run_root")
                }
            ),
        },
        "records": records,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path
