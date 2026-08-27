# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CLI and config handling for kernel-agent generation."""

from __future__ import annotations

import argparse

from dataset.prompts import DEFAULT_OPTIMIZATION_PARADIGM, DEFAULT_TARGET_GPU, normalize_optimization_paradigm

DEFAULT_OUTPUT_CONTRACT = "sample_json_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate optimized HIP kernels with vLLM.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--model_path", type=str, help="Override model path from config.")
    parser.add_argument("--input_dir", type=str, help="Override input directory from config.")
    parser.add_argument("--output_dir", type=str, help="Override output directory from config.")
    parser.add_argument("--batch_size", type=int, help="Override batch size from config.")
    parser.add_argument("--temperature", type=float, help="Override temperature from config.")
    parser.add_argument("--n_gpus", type=int, help="Override tensor parallel size from config.")
    parser.add_argument("--rollout_n", type=int, help="Override rollout sample count from config.")
    parser.add_argument(
        "--rollout_indices",
        type=str,
        help="Comma-separated global rollout indices to generate, e.g. 4,5,6 or 4-15.",
    )
    parser.add_argument(
        "--merge_existing_manifest",
        action="store_true",
        help="Merge records from output_dir/generation_manifest.json before writing the new manifest.",
    )
    parser.add_argument("--seed_base", type=int, help="Override base seed for serial rollouts.")
    parser.add_argument("--prompt_text", type=str, help="Additional global prompt text.")
    parser.add_argument(
        "--prompt_map_json",
        type=str,
        help="JSON file mapping input HIP files/stems to per-task prompt text.",
    )
    parser.add_argument(
        "--prompt_map_arm",
        type=str,
        help="Optional arm name to select from a prompt-map JSON with an arms object.",
    )
    parser.add_argument(
        "--raw_response_dir",
        type=str,
        help="Directory for full raw response JSON sidecars.",
    )
    parser.add_argument(
        "--feedback_context_json",
        type=str,
        help="JSON feedback context for multi-turn generation rounds after the first.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        help="Parser data source, e.g. kernel-agent-react-train.",
    )
    parser.add_argument(
        "--experiment_arm",
        type=str,
        help="Optional experiment arm label recorded in manifests and raw sidecars.",
    )
    parser.add_argument(
        "--output_contract",
        type=str,
        help="Output contract for prompt construction and parsing (default: sample_json_v1).",
    )
    parser.add_argument(
        "--target_gpu",
        type=str,
        help=f"Target GPU profile used by dataset prompts (default: {DEFAULT_TARGET_GPU}).",
    )
    parser.add_argument(
        "--optimization_paradigm",
        "--optimization-paradigm",
        dest="optimization_paradigm",
        type=str,
        help=f"Optimization paradigm for prompts/parsing/persistence (default: {DEFAULT_OPTIMIZATION_PARADIGM}).",
    )
    return parser.parse_args(argv)


def apply_config_overrides(config, args: argparse.Namespace):
    if args.model_path:
        config.model.path = args.model_path
    if args.input_dir:
        config.data.input_dir = args.input_dir
    if args.output_dir:
        config.data.output_dir = args.output_dir
    if args.batch_size:
        config.generation.batch_size = args.batch_size
    if args.temperature is not None:
        config.generation.temperature = args.temperature
    if args.n_gpus:
        config.rollout.tensor_model_parallel_size = args.n_gpus
    if args.rollout_n:
        config.rollout.n = args.rollout_n
    if args.rollout_indices:
        config.generation.rollout_indices = args.rollout_indices
    if args.merge_existing_manifest:
        config.generation.merge_existing_manifest = True
    if args.seed_base is not None:
        config.generation.seed_base = args.seed_base
    if args.prompt_text is not None:
        config.generation.prompt_text = args.prompt_text
    if args.prompt_map_json:
        config.generation.prompt_map_json = args.prompt_map_json
    if args.prompt_map_arm:
        config.generation.prompt_map_arm = args.prompt_map_arm
    if args.raw_response_dir:
        config.generation.raw_response_dir = args.raw_response_dir
    if args.feedback_context_json:
        config.generation.feedback_context_json = args.feedback_context_json
    if args.data_source:
        config.generation.data_source = args.data_source
    if args.experiment_arm:
        config.generation.experiment_arm = args.experiment_arm
    if args.output_contract:
        config.generation.output_contract = args.output_contract
    if args.target_gpu:
        config.generation.target_gpu = args.target_gpu
    if args.optimization_paradigm:
        config.generation.optimization_paradigm = normalize_optimization_paradigm(args.optimization_paradigm)
    else:
        config.generation.optimization_paradigm = normalize_optimization_paradigm(
            config.generation.get("optimization_paradigm", DEFAULT_OPTIMIZATION_PARADIGM)
        )

    substitutions = {
        "{rollout_n}": str(config.rollout.get("n", 1)),
        "{target_gpu}": str(config.generation.get("target_gpu", DEFAULT_TARGET_GPU)),
        "{output_contract}": str(config.generation.get("output_contract", DEFAULT_OUTPUT_CONTRACT)),
        "{optimization_paradigm}": str(config.generation.get("optimization_paradigm", DEFAULT_OPTIMIZATION_PARADIGM)),
    }
    for placeholder, value in substitutions.items():
        if placeholder in config.data.output_dir:
            config.data.output_dir = config.data.output_dir.replace(placeholder, value)
    return config


def print_config_summary(config) -> None:
    output_contract = config.generation.get("output_contract", DEFAULT_OUTPUT_CONTRACT)
    target_gpu = config.generation.get("target_gpu", DEFAULT_TARGET_GPU)
    optimization_paradigm = config.generation.get("optimization_paradigm", DEFAULT_OPTIMIZATION_PARADIGM)
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"Model: {config.model.path}")
    print(f"Input: {config.data.input_dir}")
    print(f"Output: {config.data.output_dir}")
    print(f"Target GPU: {target_gpu}")
    print(f"Output contract: {output_contract}")
    print(f"Optimization paradigm: {optimization_paradigm}")
    print(f"Batch size: {config.generation.batch_size}")
    print(f"Serial rollouts per input (n): {config.rollout.get('n', 1)}")
    print(f"Seed base: {config.generation.get('seed_base', None)}")
    print(f"Temperature: {config.generation.temperature}")
    print(f"Top-p: {config.generation.top_p}")
    print(f"Top-k: {config.generation.top_k}")
    print(f"Response length: {config.generation.response_length}")
    print(f"Tensor parallel size: {config.rollout.tensor_model_parallel_size}")
    print(f"GPU memory utilization: {config.rollout.gpu_memory_utilization}")
    print(f"Prompt map: {config.generation.get('prompt_map_json', '') or 'none'}")
    print(f"Prompt arm: {config.generation.get('prompt_map_arm', '') or 'none'}")
    print(f"Feedback context: {config.generation.get('feedback_context_json', '') or 'none'}")
    print(f"Experiment arm: {config.generation.get('experiment_arm', '') or 'none'}")
    print(f"Raw response dir: {config.generation.get('raw_response_dir', '') or 'none'}")
    print("=" * 60 + "\n")
