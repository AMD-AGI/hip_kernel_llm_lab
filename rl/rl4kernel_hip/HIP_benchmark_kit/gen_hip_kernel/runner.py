# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Python runner for model cache setup and vLLM HIP generation."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from HIP_benchmark_kit.contracts.layout import repo_root
from dataset.prompts import DEFAULT_OPTIMIZATION_PARADIGM

KIT_ROOT = repo_root() / "HIP_benchmark_kit"
SCRIPT_DIR = KIT_ROOT / "gen_hip_kernel"


def default_gpu_ids() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES") or "0,1,2,3,4,5,6,7"


def cache_differs_from_source(source_dir: Path, cache_dir: Path) -> bool:
    if not source_dir.is_dir() or not cache_dir.is_dir():
        return True
    result = subprocess.run(
        [
            "rsync",
            "-aniO",
            "--delete",
            "--exclude",
            ".model_source",
            f"{source_dir}/",
            f"{cache_dir}/",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return bool(result.stdout.strip())


def prepare_model_cache(model_source: str, cache_dir: Path) -> Path:
    source_file = cache_dir / ".model_source"
    if model_source:
        source_dir = Path(model_source)
        if not source_dir.is_dir():
            raise SystemExit(f"Model source directory not found: {source_dir}")
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        need_recache = False
        if not cache_dir.is_dir():
            need_recache = True
        elif source_file.is_file():
            cached_source = source_file.read_text(encoding="utf-8").strip()
            if cached_source != str(source_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
                need_recache = True
            elif cache_differs_from_source(source_dir, cache_dir):
                print("Model cache differs from source, refreshing cache...")
                shutil.rmtree(cache_dir, ignore_errors=True)
                need_recache = True
        else:
            shutil.rmtree(cache_dir, ignore_errors=True)
            need_recache = True

        if need_recache:
            print(f"Caching model into: {cache_dir}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-ah", "--delete", "--info=progress2", f"{source_dir}/", f"{cache_dir}/"], check=True)
            source_file.write_text(str(source_dir), encoding="utf-8")
        else:
            print(f"Using existing model cache: {cache_dir}")
        return cache_dir

    if cache_dir.is_dir():
        print(f"Using cached model: {cache_dir}")
        return cache_dir
    raise SystemExit(f"No cached model found at {cache_dir} and no --model source was provided.")


def build_default_output_dir(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    if args.model:
        source = Path(args.model)
        model_tag = Path(source.parent.name) / source.name
    else:
        model_tag = Path("cached_model")
    return (
        output_root
        / model_tag
        / "hip_code_optimized_kernel-agent-single-sft-val"
        / f"target_{args.target_gpu}"
        / f"contract_{args.output_contract}"
        / f"rollout_n_{args.rollout_n}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Core kernel-agent generation runner for HIP_benchmark_kit.")
    parser.add_argument("--model", default="", help="Source model directory to cache into --cache_dir.")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"), help="Generation config file.")
    parser.add_argument(
        "--input_dir",
        "--input-dir",
        dest="input_dir",
        default=str(KIT_ROOT / "data" / os.environ.get("DATASET_NAME", "hip_eval_dataset_kernelbench_25_tasks") / "hip_code"),
    )
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="")
    parser.add_argument("--output_root", "--output-root", dest="output_root", default=str(repo_root() / "outputs" / "HIP_benchmark_kit" / "gen_hip_kernel"))
    parser.add_argument("--cache_dir", "--cache-dir", dest="cache_dir", default=os.environ.get("MODEL_CACHE", "/dev/shm/hip_vllm_model_cache"))
    parser.add_argument("--output_contract", "--output-contract", dest="output_contract", default="sample_json_v1")
    parser.add_argument("--optimization_paradigm", "--optimization-paradigm", dest="optimization_paradigm", default=DEFAULT_OPTIMIZATION_PARADIGM)
    parser.add_argument("--target_gpu", "--target-gpu", dest="target_gpu", default="mi300x")
    parser.add_argument("--rollout_n", "--rollout-n", dest="rollout_n", default="4")
    parser.add_argument("--rollout_indices", "--rollout-indices", dest="rollout_indices", default="")
    parser.add_argument("--n_gpus", "--n-gpus", dest="n_gpus", default="8")
    parser.add_argument("--temperature", default="0.8")
    parser.add_argument("--seed_base", "--seed-base", dest="seed_base", default="")
    parser.add_argument("--gpu_ids", "--gpu-ids", dest="gpu_ids", default="")
    parser.add_argument("--prompt_text", "--prompt-text", dest="prompt_text", default="")
    parser.add_argument("--prompt_map_json", "--prompt-map-json", dest="prompt_map_json", default="")
    parser.add_argument("--prompt_map_arm", "--prompt-map-arm", dest="prompt_map_arm", default="")
    parser.add_argument("--raw_response_dir", "--raw-response-dir", dest="raw_response_dir", default="")
    parser.add_argument("--feedback_context_json", "--feedback-context-json", dest="feedback_context_json", default="")
    parser.add_argument("--data_source", "--data-source", dest="data_source", default="")
    parser.add_argument("--experiment_arm", "--experiment-arm", dest="experiment_arm", default="")
    parser.add_argument("--clear-cache", action="store_true", help="Remove the cached model and exit.")
    return parser


def run_generation(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    input_dir = Path(args.input_dir)
    cache_dir = Path(args.cache_dir)
    if args.clear_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"Cleared model cache: {cache_dir}")
        return
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")
    if not input_dir.is_dir():
        raise SystemExit(f"Input HIP directory not found: {input_dir}")

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    visible_devices = args.gpu_ids or default_gpu_ids()
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    os.environ.setdefault("VLLM_ENGINE_ITERATION_TIMEOUT_S", "1000000000")

    effective_model_path = prepare_model_cache(args.model, cache_dir)
    output_dir = Path(args.output_dir) if args.output_dir else build_default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==============================================")
    print("HIP Code Generation - Pure vLLM Mode")
    print("==============================================")
    print(f"Config file: {config_path}")
    print("Pipeline: kernel-agent-single-sft-val")
    print(f"Model path: {effective_model_path}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output contract: {args.output_contract}")
    print(f"Optimization paradigm: {args.optimization_paradigm}")
    print(f"Target GPU: {args.target_gpu}")
    print(f"Serial rollouts per input: {args.rollout_n}")
    if args.rollout_indices:
        print(f"Rollout indices: {args.rollout_indices}")
    print(f"Tensor parallel size: {args.n_gpus}")
    print(f"Temperature: {args.temperature}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print("==============================================")

    generation_argv = [
        "--config",
        str(config_path),
        "--model_path",
        str(effective_model_path),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--rollout_n",
        str(args.rollout_n),
        "--n_gpus",
        str(args.n_gpus),
        "--temperature",
        str(args.temperature),
        "--output_contract",
        args.output_contract,
        "--optimization_paradigm",
        args.optimization_paradigm,
        "--target_gpu",
        args.target_gpu,
    ]
    optional_pairs = (
        ("--rollout_indices", args.rollout_indices),
        ("--seed_base", args.seed_base),
        ("--prompt_text", args.prompt_text),
        ("--prompt_map_json", args.prompt_map_json),
        ("--prompt_map_arm", args.prompt_map_arm),
        ("--raw_response_dir", args.raw_response_dir),
        ("--feedback_context_json", args.feedback_context_json),
        ("--data_source", args.data_source),
        ("--experiment_arm", args.experiment_arm),
    )
    for flag, value in optional_pairs:
        if value:
            generation_argv.extend([flag, str(value)])
    if args.rollout_indices:
        generation_argv.append("--merge_existing_manifest")

    from HIP_benchmark_kit.gen_hip_kernel.main import main as generation_main

    generation_main(generation_argv)


def main(argv: list[str] | None = None) -> None:
    run_generation(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
