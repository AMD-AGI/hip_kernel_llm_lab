#!/usr/bin/env python3
"""HIP kernel-agent generation entrypoint."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.prompts import DEFAULT_TARGET_GPU
from dataset.response_parser import KERNEL_AGENT_DEFAULT_DATA_SOURCE

from .backend_vllm import build_generator
from .cli import DEFAULT_OUTPUT_CONTRACT, apply_config_overrides, parse_args, print_config_summary
from .context import load_feedback_context, load_prompt_map
from .manifest import (
    load_existing_manifest_records,
    merge_generation_records,
    parse_rollout_indices,
    write_generation_manifest,
)
from .pipeline import process_batch, save_results
from .vllm_generator import load_config


def main(argv: list[str] | None = None):
    """Main entry point for kernel-agent HIP code generation."""
    args = parse_args(argv)
    config = load_config(args.config)
    config = apply_config_overrides(config, args)

    output_contract = config.generation.get("output_contract", DEFAULT_OUTPUT_CONTRACT)
    target_gpu = config.generation.get("target_gpu", DEFAULT_TARGET_GPU)
    optimization_paradigm = config.generation.get("optimization_paradigm", "")
    prompt_text = config.generation.get("prompt_text", "")
    prompt_map_json = config.generation.get("prompt_map_json", "")
    prompt_map_arm = config.generation.get("prompt_map_arm", "")
    raw_response_dir = config.generation.get("raw_response_dir", "")
    feedback_context_json = config.generation.get("feedback_context_json", "")
    experiment_arm = config.generation.get("experiment_arm", "")
    data_source = config.generation.get("data_source", KERNEL_AGENT_DEFAULT_DATA_SOURCE)
    seed_base = config.generation.get("seed_base", None)
    prompt_map = load_prompt_map(prompt_map_json, prompt_map_arm)
    feedback_context = load_feedback_context(feedback_context_json)

    pattern = os.path.join(config.data.input_dir, config.data.file_pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found: {pattern}")
        return

    os.makedirs(config.data.output_dir, exist_ok=True)

    print(f"Found {len(files)} HIP files")
    print_config_summary(config)

    generator = build_generator(config)
    batch_size = config.generation.batch_size
    n_batches = (len(files) + batch_size - 1) // batch_size
    n_samples = config.rollout.get("n", 1)
    rollout_indices = parse_rollout_indices(
        config.generation.get("rollout_indices", ""),
        int(n_samples),
    )

    print(f"Generating {len(rollout_indices)} of {n_samples} sample(s) per input file")
    print(f"Rollout indices: {','.join(map(str, rollout_indices))}")
    print(f"Optimization paradigm: {optimization_paradigm or 'kernel2kernel_splice'}")
    print(f"Kernel-agent data source: {data_source}\n")

    total_saved = 0
    existing_records = []
    if config.generation.get("merge_existing_manifest", False):
        existing_records = load_existing_manifest_records(config.data.output_dir)
        print(f"Loaded {len(existing_records)} existing manifest record(s) for merge")
    all_records = []
    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(files))
        batch_files = files[start:end]

        print(
            f"\nBatch {batch_idx + 1}/{n_batches}: "
            f"processing {len(batch_files)} files with {n_samples} sample(s) each..."
        )
        batch_contexts, raw_results = process_batch(
            batch_files=batch_files,
            generator=generator,
            prompt_text=prompt_text,
            prompt_map=prompt_map,
            feedback_context=feedback_context,
            target_gpu=target_gpu,
            output_contract=output_contract,
            rollout_indices=rollout_indices,
            optimization_paradigm=optimization_paradigm,
        )
        saved_count, batch_records = save_results(
            batch_contexts,
            raw_results,
            output_dir=config.data.output_dir,
            n_samples=n_samples,
            seed_base=seed_base,
            output_contract=output_contract,
            data_source=data_source,
            raw_response_dir=raw_response_dir,
            experiment_arm=experiment_arm,
            optimization_paradigm=optimization_paradigm,
        )
        total_saved += saved_count
        all_records.extend(batch_records)

    merged_records = merge_generation_records(existing_records, all_records)
    manifest_path = write_generation_manifest(config.data.output_dir, config, merged_records)
    parse_failures = sum(1 for record in merged_records if not record["parse_ok"])

    print(f"\n{'=' * 60}")
    print(f"Completed! Processed {len(files)} input files")
    print(f"Requested generations this invocation: {len(files) * len(rollout_indices)}")
    print(f"Manifest records after merge: {len(merged_records)}")
    print(f"Saved HIP outputs: {total_saved}")
    print(f"Parse failures: {parse_failures}")
    print(f"Results saved to: {config.data.output_dir}")
    print(f"Generation manifest: {manifest_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

