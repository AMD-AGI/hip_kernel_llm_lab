"""Batch generation pipeline and HIP output persistence."""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from dataset.response_parser import build_parse_attempt_chain, parse_kernel_agent_generation_response
from dataset.utils import build_hip_kernel_agent_chat_messages, build_hip_kernel_agent_multiturn_chat_messages

from .context import resolve_feedback_context, resolve_prompt_text
from .manifest import sha256_text
from .paradigms import get_generation_paradigm_policy
from .vllm_generator import read_hip_file


def write_raw_response_sidecar(
    raw_response_dir: str,
    context: dict[str, Any],
    sample_idx: int,
    seed_base,
    raw_response,
    *,
    output_contract: str,
    data_source: str,
    experiment_arm: str,
) -> str:
    if not raw_response_dir:
        return ""

    os.makedirs(raw_response_dir, exist_ok=True)
    input_stem = os.path.splitext(context["input_file"])[0]
    raw_response_text = str(raw_response)
    sidecar_path = os.path.join(raw_response_dir, f"{input_stem}_gen{sample_idx}_raw_response.json")
    payload = {
        "experiment_arm": experiment_arm,
        "input_file": context["input_file"],
        "input_path": context["input_path"],
        "sample_idx": sample_idx,
        "rollout_idx": sample_idx,
        "sampling_seed": (int(seed_base) + int(sample_idx)) if seed_base is not None else None,
        "kernel_name": context["kernel_name"] or "",
        "starter_code_kind": context["starter_code_kind"],
        "output_contract": output_contract,
        "data_source": data_source,
        "prompt_map_key": context.get("prompt_map_key", ""),
        "feedback_context_key": context.get("feedback_context_key", ""),
        "previous_generated_path": context.get("previous_generated_path", ""),
        "blocked_reason": context.get("blocked_reason", ""),
        "prompt_text_sha256": context.get("prompt_text_sha256", ""),
        "prompt_text_char_count": context.get("prompt_text_char_count", 0),
        "prompt_text": context.get("prompt_text", ""),
        "raw_response_sha256": sha256_text(raw_response_text),
        "raw_response_char_count": len(raw_response_text),
        "raw_response": raw_response_text,
        "optimization_paradigm": context.get("optimization_paradigm", ""),
        "expected_code_unit": context.get("expected_code_unit", ""),
        "persistence_mode": context.get("persistence_mode", ""),
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return sidecar_path


def process_batch(
    batch_files: Sequence[str],
    generator,
    *,
    prompt_text: str,
    prompt_map: dict[str, str],
    feedback_context: dict[str, dict[str, Any]] | None,
    target_gpu: str,
    output_contract: str,
    rollout_indices: list[int],
    optimization_paradigm: str,
):
    prompts = []
    batch_contexts = []
    paradigm_policy = get_generation_paradigm_policy(optimization_paradigm)

    for fpath in batch_files:
        hip_code = read_hip_file(fpath)
        prompt_source = paradigm_policy.build_prompt_source(hip_code)
        starter_code = prompt_source.starter_code
        per_file_prompt, prompt_map_key = resolve_prompt_text(prompt_text, prompt_map, os.path.basename(fpath))
        feedback_entry, feedback_context_key = resolve_feedback_context(feedback_context or {}, os.path.basename(fpath))
        previous_generated_code = ""
        if feedback_entry:
            previous_generated_path = str(feedback_entry.get("previous_generated_path") or "")
            if previous_generated_path and os.path.isfile(previous_generated_path):
                previous_hip_code = read_hip_file(previous_generated_path)
                previous_generated_code = paradigm_policy.extract_previous_generated_code(previous_hip_code)
            prompt_messages = build_hip_kernel_agent_multiturn_chat_messages(
                previous_thought=str(feedback_entry.get("thought") or ""),
                original_starter_code=starter_code,
                previous_generated_code=previous_generated_code,
                previous_feedback=str(feedback_entry.get("feedback_text") or ""),
                prompt=per_file_prompt,
                target_gpu=target_gpu,
                output_contract=output_contract,
                optimization_paradigm=paradigm_policy.optimization_paradigm,
            )
            per_file_prompt = str(feedback_entry.get("feedback_text") or "")
        else:
            prompt_messages = build_hip_kernel_agent_chat_messages(
                prompt=per_file_prompt,
                starter_code=starter_code,
                target_gpu=target_gpu,
                output_contract=output_contract,
                optimization_paradigm=paradigm_policy.optimization_paradigm,
            )
        if prompt_source.starter_code_kind == "full_file_fallback":
            print(f"[WARN] Cannot extract kernel from {fpath}; using the full HIP file as starter code")

        batch_contexts.append(
            {
                "input_path": fpath,
                "input_file": os.path.basename(fpath),
                "original_hip_code": hip_code,
                "kernel_name": prompt_source.kernel_name,
                "starter_code_kind": prompt_source.starter_code_kind,
                "optimization_paradigm": paradigm_policy.optimization_paradigm,
                "expected_code_unit": prompt_source.expected_code_unit,
                "persistence_mode": prompt_source.persistence_mode,
                "prompt_map_key": prompt_map_key,
                "prompt_text": per_file_prompt,
                "prompt_text_sha256": sha256_text(per_file_prompt),
                "prompt_text_char_count": len(per_file_prompt),
                "feedback_context_key": feedback_context_key,
                "previous_generated_path": str(feedback_entry.get("previous_generated_path") or "") if feedback_entry else "",
                "blocked_reason": str(feedback_entry.get("blocked_reason") or "") if feedback_entry else "",
            }
        )
        prompts.append(prompt_messages)

    raw_results = generator.generate(prompts, rollout_indices=rollout_indices)
    return batch_contexts, raw_results


def _build_output_filename(input_file: str, sample_idx: int, n_samples: int) -> str:
    if n_samples == 1:
        return input_file
    stem, _ = os.path.splitext(input_file)
    return f"{stem}_gen{sample_idx}.hip"


def save_results(
    batch_contexts,
    raw_results,
    *,
    output_dir: str,
    n_samples: int,
    seed_base,
    output_contract: str,
    data_source: str,
    raw_response_dir: str = "",
    experiment_arm: str = "",
    optimization_paradigm: str = "",
):
    records = []
    saved_count = 0

    for context, sample_list in zip(batch_contexts, raw_results):
        paradigm_policy = get_generation_paradigm_policy(
            context.get("optimization_paradigm") or optimization_paradigm
        )
        for item in sample_list:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                sample_idx, raw_response = int(item[0]), item[1]
            else:
                sample_idx = len([record for record in records if record.get("input_file") == context["input_file"]])
                raw_response = item
            raw_response_text = str(raw_response)
            raw_response_path = write_raw_response_sidecar(
                raw_response_dir,
                context,
                sample_idx,
                seed_base,
                raw_response_text,
                output_contract=output_contract,
                data_source=data_source,
                experiment_arm=experiment_arm,
            )
            parse_result = parse_kernel_agent_generation_response(
                raw_response_text,
                output_contract=output_contract,
                kernel_name=context["kernel_name"],
                hip_ref=context["original_hip_code"],
                data_source=data_source,
                expected_code_unit=context.get("expected_code_unit"),
            )
            parse_attempt_chain = build_parse_attempt_chain(parse_result)
            record = {
                "input_file": context["input_file"],
                "input_path": context["input_path"],
                "experiment_arm": experiment_arm,
                "sample_idx": sample_idx,
                "rollout_idx": sample_idx,
                "sampling_seed": (int(seed_base) + int(sample_idx)) if seed_base is not None else None,
                "starter_code_kind": context["starter_code_kind"],
                "optimization_paradigm": paradigm_policy.optimization_paradigm,
                "expected_code_unit": context.get("expected_code_unit", ""),
                "persistence_mode": context.get("persistence_mode", ""),
                "kernel_name": context["kernel_name"] or "",
                "prompt_map_key": context.get("prompt_map_key", ""),
                "feedback_context_key": context.get("feedback_context_key", ""),
                "previous_generated_path": context.get("previous_generated_path", ""),
                "blocked_reason": context.get("blocked_reason", ""),
                "prompt_text_sha256": context.get("prompt_text_sha256", ""),
                "prompt_text_char_count": context.get("prompt_text_char_count", 0),
                "output_contract": parse_result["output_contract"],
                "parse_mode": parse_result.get("parse_mode") or "",
                "attempted_parse_modes": parse_result.get("attempted_parse_modes") or [],
                "parse_attempt_chain": parse_attempt_chain,
                "parse_ok": bool(parse_result.get("parse_ok")),
                "parse_error": parse_result.get("parse_error") or "",
                "saved": False,
                "output_file": "",
                "output_path": "",
                "raw_response_path": raw_response_path,
                "raw_response_sha256": sha256_text(raw_response_text),
                "raw_response_char_count": len(raw_response_text),
                "raw_response_preview": raw_response_text[:240],
            }

            if not record["parse_ok"]:
                records.append(record)
                print(
                    "[GEN PARSE FAIL] "
                    f"file={context['input_file']} sample={sample_idx} "
                    f"contract={record['output_contract']} "
                    f"parse_mode={record['parse_mode'] or 'none'} "
                    f"attempts={record['parse_attempt_chain'] or 'none'} "
                    f"error={record['parse_error']}"
                )
                continue

            final_code = paradigm_policy.persist_code(
                context["original_hip_code"],
                parse_result["hip_src"],
                kernel_name=context["kernel_name"],
            )
            output_file = _build_output_filename(context["input_file"], sample_idx, n_samples)
            output_path = os.path.join(output_dir, output_file)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_code)

            record["saved"] = True
            record["output_file"] = output_file
            record["output_path"] = output_path
            records.append(record)
            saved_count += 1

    return saved_count, records
