# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Python orchestration for HIP_benchmark_kit runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path
from shlex import quote
from typing import Any, Sequence

from HIP_benchmark_kit.contracts.eval_schema import BASELINE_RESULTS_JSON, COMPARISON_RESULTS_JSON
from HIP_benchmark_kit.contracts.layout import DEFAULT_LEVELS, KernelBenchRunLayout, repo_root
from HIP_benchmark_kit.contracts.manifests import read_json, write_json
from HIP_benchmark_kit.eval.eval_reuse_identity import load_reusable_results
from HIP_benchmark_kit.eval.runner import compare_hip_dirs, run_comprehensive_eval
from HIP_benchmark_kit.gen_hip_kernel.runner import run_generation as run_generation_api
from HIP_benchmark_kit.eval.merge_origin_optimized_eval import main as merge_results_api
from HIP_benchmark_kit.orchestration.rollout_reuse import materialize_generation_reuse
from HIP_benchmark_kit.profiling_context.cards import build_profile_cards
from HIP_benchmark_kit.profiling_context.ensure import ensure_metrix_profiles
from HIP_benchmark_kit.profiling_context.feedback import (
    build_feedback_context,
    index_comparison_candidates,
    index_comparison_rows,
    load_raw_response_thought,
    stage_generated_kernels_for_profile,
)
from dataset.prompts import DEFAULT_OPTIMIZATION_PARADIGM, normalize_optimization_paradigm

REPO_ROOT = repo_root()
KIT_ROOT = REPO_ROOT / "HIP_benchmark_kit"
DEFAULT_PROFILE_SCRIPT = (
    REPO_ROOT / "profiler" / "intellikit" / "metrix" / "examples" / "01_basic_profiling" / "kernelbench_sample" / "profile_kernelbench.py"
)
ORIGIN_PROFILE_CONTEXT_CHOICES = ("off", "use_existing", "ensure_and_use")


def default_gpu_ids() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES") or "0,1,2,3,4,5,6,7"


def command_text(cmd: Sequence[object]) -> str:
    return " ".join(quote(str(part)) for part in cmd)


def run_command(
    cmd: Sequence[object],
    *,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
    capture_stdout: bool = False,
) -> str:
    if dry_run:
        print(f"[DRY RUN] {command_text(cmd)}")
        return ""
    completed = subprocess.run(
        [str(part) for part in cmd],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=None,
        env=env,
    )
    if capture_stdout and completed.stdout:
        print(completed.stdout, end="")
    return completed.stdout.strip() if capture_stdout and completed.stdout else ""


def prepare_gpu_env(gpu_ids: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("HIP_VISIBLE_DEVICES", None)
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    return env


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="", help="Source model directory to cache for generation.")
    parser.add_argument(
        "--kernelbench_hip_root",
        "--kernelbench-hip-root",
        dest="kernelbench_hip_root",
        default=str(KIT_ROOT / "data" / "hip_eval_neurlps" / "kernelbench_hip"),
        help="Source kernelbench_hip root.",
    )
    parser.add_argument("--cache_dir", "--cache-dir", dest="cache_dir", default=os.environ.get("MODEL_CACHE", "/dev/shm/hip_vllm_model_cache"))
    parser.add_argument("--output_root", "--output-root", dest="output_root", default="")
    parser.add_argument("--gpu_ids", "--gpu-ids", dest="gpu_ids", default=default_gpu_ids())
    parser.add_argument("--n_gpus", "--n-gpus", dest="n_gpus", default="8")
    parser.add_argument("--rollout_n", "--rollout-n", dest="rollout_n", default="1")
    parser.add_argument("--level1_count", "--level1-count", dest="level1_count", default="35")
    parser.add_argument("--level2_count", "--level2-count", dest="level2_count", default="35")
    parser.add_argument("--level3_count", "--level3-count", dest="level3_count", default="30")
    parser.add_argument("--eval_workers", "--eval-workers", dest="eval_workers", default="8")
    parser.add_argument("--perf_iterations", "--perf-iterations", dest="perf_iterations", default="10")
    parser.add_argument("--temperature", default="1")
    parser.add_argument("--seed_base", "--seed-base", dest="seed_base", default="")
    parser.add_argument("--output_contract", "--output-contract", dest="output_contract", default="sample_json_v1")
    parser.add_argument(
        "--optimization_paradigm",
        "--optimization-paradigm",
        dest="optimization_paradigm",
        default=DEFAULT_OPTIMIZATION_PARADIGM,
    )
    parser.add_argument("--target_gpu", "--target-gpu", dest="target_gpu", default="mi300x")
    parser.add_argument("--rtol", default="1e-4")
    parser.add_argument("--atol", default="1e-5")
    parser.add_argument("--local_work_root", "--local-work-root", dest="local_work_root", default="")
    parser.add_argument("--eval_backend", "--eval-backend", dest="eval_backend", default="server-inprocess")
    parser.add_argument("--reference_hip_dir", "--reference-hip-dir", dest="reference_hip_dir", default="")
    parser.add_argument("--context_mode", "--context-mode", dest="context_mode", default="A_control")
    parser.add_argument("--profile_artifact_root", "--profile-artifact-root", dest="profile_artifact_root", default="")
    parser.add_argument("--profile_prompt_root", "--profile-prompt-root", dest="profile_prompt_root", default="")
    parser.add_argument("--profile_missing_policy", "--profile-missing-policy", dest="profile_missing_policy", default="fail")
    parser.add_argument("--ensure_profile_artifacts", "--ensure-profile-artifacts", dest="ensure_profile_artifacts", action="store_true")
    parser.add_argument("--skip_profile_ensure", "--skip-profile-ensure", dest="skip_profile_ensure", action="store_true")
    parser.add_argument("--profile_script", "--profile-script", dest="profile_script", default=str(DEFAULT_PROFILE_SCRIPT))
    parser.add_argument("--profile_gpu_ids", "--profile-gpu-ids", dest="profile_gpu_ids", default="")
    parser.add_argument("--profile_parallel_workers", "--profile-parallel-workers", dest="profile_parallel_workers", default="8")
    parser.add_argument("--profile_compile_workers", "--profile-compile-workers", dest="profile_compile_workers", default="0")
    parser.add_argument("--profile_prewarm_iters", "--profile-prewarm-iters", dest="profile_prewarm_iters", default="2")
    parser.add_argument("--profile_iters", "--profile-iters", dest="profile_iters", default="5")
    parser.add_argument("--profile_timeout_seconds", "--profile-timeout-seconds", dest="profile_timeout_seconds", default="600")
    parser.add_argument(
        "--profile_metadata_mode",
        "--profile-metadata-mode",
        dest="profile_metadata_mode",
        default="deferred",
        choices=("deferred", "full"),
    )
    parser.add_argument("--reuse_outputs", "--reuse-outputs", dest="reuse_outputs", action="store_true", default=True)
    parser.add_argument("--overwrite_outputs", "--overwrite-outputs", dest="reuse_outputs", action="store_false")
    parser.add_argument("--reuse_rollout_from_root", "--reuse-rollout-from-root", dest="reuse_rollout_from_root", default="")
    parser.add_argument("--reuse_generation_from_root", "--reuse-generation-from-root", dest="reuse_generation_from_root", default="")
    parser.add_argument("--reuse_eval_from_root", "--reuse-eval-from-root", dest="reuse_eval_from_root", default="")
    parser.add_argument("--origin_baseline_eval_root", "--origin-baseline-eval-root", dest="origin_baseline_eval_root", default="")
    parser.add_argument("--shared_compile_cache_root", "--shared-compile-cache-root", dest="shared_compile_cache_root", default="")
    parser.add_argument("--disable_rollout_reuse", "--disable-rollout-reuse", action="store_true", help="Accepted for launcher parity.")
    parser.add_argument("--skip_generation", "--skip-generation", dest="skip_generation", action="store_true")
    parser.add_argument("--skip_eval", "--skip-eval", dest="skip_eval", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true")


def resolved_kernelbench_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.output_root:
        args.output_root = str(
            REPO_ROOT
            / "outputs"
            / "HIP_benchmark_kit"
            / "kernelbench_hip_eval_100"
            / "global_step_300"
            / f"rollout_n_{args.rollout_n}"
        )
    if not args.profile_prompt_root:
        args.profile_prompt_root = str(Path(args.output_root) / "profiling_context")
    if not args.profile_gpu_ids:
        args.profile_gpu_ids = args.gpu_ids
    if args.context_mode == "B_profile_raw" and not args.profile_artifact_root:
        args.profile_artifact_root = str(REPO_ROOT / "outputs" / "HIP_benchmark_kit" / "kernelbench_hip_eval_100" / "metrix_profiles")
    if not args.ensure_profile_artifacts and not args.skip_profile_ensure:
        args.skip_profile_ensure = args.context_mode != "B_profile_raw"
    if args.reuse_rollout_from_root:
        args.reuse_generation_from_root = args.reuse_rollout_from_root
        args.reuse_eval_from_root = args.reuse_rollout_from_root
    if args.context_mode not in {"A_control", "B_profile_raw"}:
        raise SystemExit(f"Unsupported context mode: {args.context_mode}")
    args.optimization_paradigm = normalize_optimization_paradigm(args.optimization_paradigm)
    if args.profile_missing_policy not in {"fail", "skip", "empty"}:
        raise SystemExit(f"Unsupported profile missing policy: {args.profile_missing_policy}")
    if args.eval_backend not in {"server-inprocess", "sandbox-inprocess"}:
        raise SystemExit(f"Unsupported eval backend: {args.eval_backend}")
    return args


def report_cmd(*parts: object) -> list[object]:
    return [sys.executable, "-m", "HIP_benchmark_kit.reports.kernelbench", *parts]


def stage_subset(args: argparse.Namespace, layout: KernelBenchRunLayout, env: dict[str, str]) -> None:
    cmd = report_cmd(
        "stage-subset",
        "--source-root",
        args.kernelbench_hip_root,
        "--subset-root",
        layout.subset_root,
        "--manifest",
        layout.subset_manifest,
        "--level",
        f"level-1:{args.level1_count}",
        "--level",
        f"level-2:{args.level2_count}",
        "--level",
        f"level-3:{args.level3_count}",
    )
    run_command(cmd, dry_run=args.dry_run, env=env)


def resolve_profile_dir_for_level(args: argparse.Namespace, level: str) -> Path:
    nested = Path(args.profile_artifact_root) / level
    return nested if nested.is_dir() else Path(args.profile_artifact_root)


def ensure_profiles_for_level(args: argparse.Namespace, layout: KernelBenchRunLayout, level: str, dataset_root: Path, env: dict[str, str]) -> None:
    if args.context_mode != "B_profile_raw":
        return
    if args.skip_profile_ensure:
        print(f"Skipping profiling artifact ensure for {level}")
        return
    profile_script = Path(args.profile_script)
    if not profile_script.is_file():
        raise SystemExit(f"Profiling script not found: {profile_script}")
    cmd = [
        sys.executable,
        "-m",
        "HIP_benchmark_kit.profiling_context.ensure",
        "--subset-manifest",
        layout.subset_manifest,
        "--level",
        level,
        "--dataset-root",
        dataset_root,
        "--profile-artifact-root",
        args.profile_artifact_root,
        "--profile-script",
        profile_script,
        "--gpu-ids",
        args.profile_gpu_ids,
        "--parallel-workers",
        args.profile_parallel_workers,
        "--compile-workers",
        args.profile_compile_workers,
        "--prewarm-iters",
        args.profile_prewarm_iters,
        "--profile-iters",
        args.profile_iters,
        "--timeout-seconds",
        args.profile_timeout_seconds,
        "--metadata-mode",
        args.profile_metadata_mode,
    ]
    print(f"Ensuring Metrix profiling artifacts for {level}: {command_text(cmd)}")
    if args.dry_run:
        run_command(cmd, dry_run=True, env=env)
        return
    ensure_metrix_profiles(
        subset_manifest=layout.subset_manifest,
        level=level,
        dataset_root=dataset_root,
        profile_artifact_root=Path(args.profile_artifact_root),
        profile_script=profile_script,
        gpu_ids=args.profile_gpu_ids,
        parallel_workers=int(args.profile_parallel_workers),
        compile_workers=int(args.profile_compile_workers),
        prewarm_iters=int(args.profile_prewarm_iters),
        profile_iters=int(args.profile_iters),
        timeout_seconds=int(args.profile_timeout_seconds),
        metadata_mode=args.profile_metadata_mode,
    )


def write_context_manifest_entry(args: argparse.Namespace, level: str, profile_dir: Path, prompt_map_json: Path, input_dir: Path) -> None:
    manifest_path = Path(args.profile_prompt_root) / "profiling_context_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        manifest = {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "context_mode": args.context_mode,
            "profile_artifact_root": args.profile_artifact_root,
            "profile_missing_policy": args.profile_missing_policy,
            "levels": {},
        }
    prompt_payload = read_json(prompt_map_json) if prompt_map_json.exists() else {}
    metadata = prompt_payload.get("metadata", {}) if isinstance(prompt_payload, dict) else {}
    manifest["levels"][level] = {
        "input_dir": str(input_dir),
        "profile_dir": str(profile_dir),
        "prompt_map_json": str(prompt_map_json),
        "sample_count": metadata.get("sample_count"),
        "missing_count": metadata.get("missing_count"),
        "missing_inputs": metadata.get("missing_inputs", []),
        "arms": metadata.get("arms", []),
    }
    write_json(manifest_path, manifest)


def prepare_context_for_level(args: argparse.Namespace, level: str, input_dir: Path, env: dict[str, str]) -> dict[str, str]:
    context = {
        "prompt_map_json": "",
        "prompt_map_arm": "",
        "raw_response_dir": "",
        "experiment_arm": "",
        "data_source": "",
    }
    if args.context_mode == "A_control":
        return context
    profile_dir = resolve_profile_dir_for_level(args, level)
    prompt_map_json = Path(args.profile_prompt_root) / f"{level}_prompt_map.json"
    context = {
        "prompt_map_json": str(prompt_map_json),
        "prompt_map_arm": args.context_mode,
        "raw_response_dir": str(Path(args.output_root) / level / "raw_responses"),
        "experiment_arm": args.context_mode,
        "data_source": "kernel-agent-react-train",
    }
    cmd = [
        sys.executable,
        "-m",
        "HIP_benchmark_kit.profiling_context.cards",
        "--profile-dir",
        profile_dir,
        "--input-dir",
        input_dir,
        "--output-json",
        prompt_map_json,
        "--arms",
        args.context_mode,
        "--missing-policy",
        args.profile_missing_policy,
    ]
    print(f"Preparing profiling context for {level}: {command_text(cmd)}")
    if args.dry_run:
        run_command(cmd, dry_run=True, env=env)
    else:
        build_profile_cards(
            profile_dir=profile_dir,
            input_dir=input_dir,
            output_json=prompt_map_json,
            arms=[args.context_mode],
            missing_policy=args.profile_missing_policy,
        )
    if not args.dry_run:
        write_context_manifest_entry(args, level, profile_dir, prompt_map_json, input_dir)
    return context


def generation_complete_for_level(manifest_path: Path, args: argparse.Namespace) -> bool:
    if not manifest_path.is_file():
        return False
    manifest = read_json(manifest_path)
    errors: list[str] = []
    if int(manifest.get("rollout_n") or 0) != int(args.rollout_n):
        errors.append(f"rollout_n={manifest.get('rollout_n')} expected={args.rollout_n}")
    if str(manifest.get("output_contract") or "") != args.output_contract:
        errors.append(f"output_contract={manifest.get('output_contract')} expected={args.output_contract}")
    if str(manifest.get("optimization_paradigm") or "") != args.optimization_paradigm:
        errors.append(
            f"optimization_paradigm={manifest.get('optimization_paradigm')} expected={args.optimization_paradigm}"
        )
    experiment_arm = str(manifest.get("experiment_arm") or "")
    prompt_map_arm = str(manifest.get("prompt_map_arm") or "")
    if args.context_mode == "B_profile_raw":
        if experiment_arm != "B_profile_raw":
            errors.append(f"experiment_arm={experiment_arm!r} expected='B_profile_raw'")
        if prompt_map_arm != "B_profile_raw":
            errors.append(f"prompt_map_arm={prompt_map_arm!r} expected='B_profile_raw'")
    elif experiment_arm or prompt_map_arm:
        errors.append(
            f"context_mode={args.context_mode} but manifest has experiment_arm={experiment_arm!r}, prompt_map_arm={prompt_map_arm!r}"
        )
    if errors:
        raise SystemExit(
            "Existing generation manifest is incompatible with this run; refusing to overwrite by default:\n  - "
            + "\n  - ".join(errors)
            + "\nUse --overwrite_outputs or a different --output_root if you intend to replace it."
        )
    return True


def eval_complete_for_level(eval_dir: Path) -> bool:
    return (eval_dir / "comparison" / COMPARISON_RESULTS_JSON).is_file()


def infer_rollout_n_from_root(run_root: str) -> int | None:
    for part in reversed(Path(run_root).parts):
        if not part.startswith("rollout_n_"):
            continue
        try:
            return int(part.removeprefix("rollout_n_"))
        except ValueError:
            return None
    return None


def materialize_reuse_for_level(
    args: argparse.Namespace,
    level: str,
    generation_dir: Path,
    context: dict[str, str],
    env: dict[str, str],
) -> str:
    if args.disable_rollout_reuse:
        return ""
    if not args.reuse_generation_from_root:
        return ""
    if args.dry_run:
        print(f"[DRY RUN] Would materialize generation reuse for {level} from {args.reuse_generation_from_root}")
        source_rollout_n = infer_rollout_n_from_root(args.reuse_generation_from_root)
        if source_rollout_n is not None and source_rollout_n < int(args.rollout_n):
            return ",".join(str(idx) for idx in range(source_rollout_n, int(args.rollout_n)))
        return ""
    if not Path(args.reuse_generation_from_root).is_dir():
        raise SystemExit(f"Generation reuse root not found: {args.reuse_generation_from_root}")
    reuse_plan = generation_dir / "rollout_reuse_plan.json"
    print(f"Materializing generation reuse for {level} from {args.reuse_generation_from_root}")
    plan = materialize_generation_reuse(
        source_run_root=Path(args.reuse_generation_from_root),
        target_run_root=Path(args.output_root),
        level=level,
        target_rollout_n=int(args.rollout_n),
        context_mode=args.context_mode,
        expected_model_path=args.cache_dir,
        expected_output_contract=args.output_contract,
        expected_optimization_paradigm=args.optimization_paradigm,
        expected_target_gpu=args.target_gpu,
        expected_data_source=context.get("data_source", ""),
        expected_seed_base=args.seed_base,
        expected_temperature=args.temperature,
        expected_prompt_map_arm=context.get("prompt_map_arm", ""),
        target_prompt_map_json=Path(context["prompt_map_json"]) if context.get("prompt_map_json") else None,
        target_raw_response_dir=Path(context["raw_response_dir"]) if context.get("raw_response_dir") else None,
        target_generation_dir=generation_dir,
        output_plan=reuse_plan,
    )
    rollout_indices = ",".join(str(idx) for idx in plan["missing_rollout_indices"])
    if rollout_indices:
        print(f"Generation reuse for {level}: generating missing rollout indices {rollout_indices}")
    else:
        print(f"Generation reuse for {level}: no missing rollout indices")
    return rollout_indices


def run_generation(
    args: argparse.Namespace,
    level: str,
    input_dir: Path,
    generation_dir: Path,
    rollout_indices: str,
    context: dict[str, str],
    env: dict[str, str],
) -> None:
    cmd: list[object] = [
        sys.executable,
        "-m",
        "HIP_benchmark_kit.gen_hip_kernel.runner",
        "--input_dir",
        input_dir,
        "--output_dir",
        generation_dir,
        "--cache_dir",
        args.cache_dir,
        "--rollout_n",
        args.rollout_n,
        "--n_gpus",
        args.n_gpus,
        "--temperature",
        args.temperature,
        "--output_contract",
        args.output_contract,
        "--optimization_paradigm",
        args.optimization_paradigm,
        "--target_gpu",
        args.target_gpu,
    ]
    if args.seed_base:
        cmd += ["--seed_base", args.seed_base]
    if rollout_indices:
        cmd += ["--rollout_indices", rollout_indices]
    if args.model:
        cmd += ["--model", args.model]
    for flag, key in (
        ("--prompt_map_json", "prompt_map_json"),
        ("--prompt_map_arm", "prompt_map_arm"),
        ("--raw_response_dir", "raw_response_dir"),
        ("--feedback_context_json", "feedback_context_json"),
        ("--data_source", "data_source"),
        ("--experiment_arm", "experiment_arm"),
    ):
        if context.get(key):
            cmd += [flag, context[key]]
    print(f"Running generation for {level}: {command_text(cmd)}")
    if args.dry_run:
        run_command(cmd, dry_run=True, env=env)
        return
    if getattr(args, "generation_subprocess", False):
        run_command(cmd, env=env)
        return
    run_generation_api(
        argparse.Namespace(
            model=args.model,
            config=str(KIT_ROOT / "gen_hip_kernel" / "config.yaml"),
            input_dir=str(input_dir),
            output_dir=str(generation_dir),
            output_root="",
            cache_dir=args.cache_dir,
            output_contract=args.output_contract,
            optimization_paradigm=args.optimization_paradigm,
            target_gpu=args.target_gpu,
            rollout_n=args.rollout_n,
            rollout_indices=rollout_indices,
            n_gpus=args.n_gpus,
            temperature=args.temperature,
            seed_base=args.seed_base,
            gpu_ids=args.gpu_ids,
            prompt_text="",
            prompt_map_json=context.get("prompt_map_json", ""),
            prompt_map_arm=context.get("prompt_map_arm", ""),
            raw_response_dir=context.get("raw_response_dir", ""),
            feedback_context_json=context.get("feedback_context_json", ""),
            data_source=context.get("data_source", ""),
            experiment_arm=context.get("experiment_arm", ""),
            clear_cache=False,
        )
    )


def assert_origin_baseline_reusable(
    args: argparse.Namespace,
    dataset_root: Path,
    reuse_json: Path,
    reuse_hip_dir: Path,
) -> None:
    if not reuse_json.is_file():
        raise SystemExit(f"Shared origin baseline JSON missing: {reuse_json}")
    if not reuse_hip_dir.is_dir():
        raise SystemExit(f"Shared origin baseline HIP dir missing: {reuse_hip_dir}")

    expected_count = len([path for path in reuse_hip_dir.glob("*.hip") if path.is_file()])
    reusable = load_reusable_results(
        reuse_json=str(reuse_json),
        reuse_hip_code_dir=str(reuse_hip_dir),
        current_hip_code_dir=str(dataset_root / "hip_code"),
        pytorch_func_dir=str(dataset_root / "pytorch_code_functional"),
        pytorch_modu_dir=str(dataset_root / "pytorch_code_module"),
        rtol=float(args.rtol),
        atol=float(args.atol),
        perf_iterations=int(args.perf_iterations),
        artifact_side="origin",
        eval_backend=args.eval_backend,
        reference_hip_code_dir=str(dataset_root / "hip_code"),
        reference_cache_mode="golden+compile",
    )
    if len(reusable) < expected_count:
        raise SystemExit(
            "Shared origin baseline identity mismatch; refusing to silently re-evaluate baseline: "
            f"reusable={len(reusable)} expected={expected_count} json={reuse_json}"
        )


def run_evaluation(
    args: argparse.Namespace,
    level: str,
    dataset_root: Path,
    generation_dir: Path,
    eval_dir: Path,
    env: dict[str, str],
    origin_baseline_eval_dir: Path | None = None,
) -> None:
    reuse_root = Path(args.reuse_eval_from_root) / level / "eval" if args.reuse_eval_from_root else None
    reuse_origin_json = str(reuse_root / "origin_eval" / BASELINE_RESULTS_JSON) if reuse_root else ""
    reuse_origin_hip_dir = str(reuse_root / "staging" / "origin") if reuse_root else ""
    reuse_optimized_json = str(reuse_root / "optimized_eval" / BASELINE_RESULTS_JSON) if reuse_root else ""
    reuse_optimized_hip_dir = str(reuse_root / "staging" / "optimized") if reuse_root else ""
    if origin_baseline_eval_dir:
        reuse_origin_json = str(origin_baseline_eval_dir / BASELINE_RESULTS_JSON)
        reuse_origin_hip_dir = str(dataset_root / "hip_code")
        if not args.dry_run:
            assert_origin_baseline_reusable(args, dataset_root, Path(reuse_origin_json), Path(reuse_origin_hip_dir))

    cmd: list[object] = [
        sys.executable,
        "-m",
        "HIP_benchmark_kit.eval.runner",
        "compare",
        "--skip-fix",
        "--origin_hip_dir",
        dataset_root / "hip_code",
        "--optimized_hip_dir",
        generation_dir,
        "--pytorch_func_dir",
        dataset_root / "pytorch_code_functional",
        "--pytorch_modu_dir",
        dataset_root / "pytorch_code_module",
        "--output_dir",
        eval_dir,
        "--max-workers",
        args.eval_workers,
        "--perf-iterations",
        args.perf_iterations,
        "--rtol",
        args.rtol,
        "--atol",
        args.atol,
        "--gpu-ids",
        args.gpu_ids,
        "--eval-backend",
        args.eval_backend,
        "--reference-hip-dir",
        args.reference_hip_dir or str(dataset_root / "hip_code"),
    ]
    if args.local_work_root:
        cmd += ["--local-work-root", args.local_work_root]
    if args.shared_compile_cache_root:
        cmd += ["--compile-cache-root", args.shared_compile_cache_root]
    if reuse_origin_json and reuse_origin_hip_dir:
        cmd += [
            "--reuse-origin-json",
            reuse_origin_json,
            "--reuse-origin-hip-dir",
            reuse_origin_hip_dir,
        ]
    if reuse_optimized_json and reuse_optimized_hip_dir:
        cmd += [
            "--reuse-optimized-json",
            reuse_optimized_json,
            "--reuse-optimized-hip-dir",
            reuse_optimized_hip_dir,
        ]
    print(f"Running evaluation for {level}: {command_text(cmd)}")
    if args.dry_run:
        run_command(cmd, dry_run=True, env=env)
        return
    compare_hip_dirs(
        argparse.Namespace(
            origin_hip_dir=str(dataset_root / "hip_code"),
            optimized_hip_dir=str(generation_dir),
            pytorch_func_dir=str(dataset_root / "pytorch_code_functional"),
            pytorch_modu_dir=str(dataset_root / "pytorch_code_module"),
            output_dir=str(eval_dir),
            max_workers=args.eval_workers,
            perf_iterations=args.perf_iterations,
            rtol=args.rtol,
            atol=args.atol,
            gpu_ids=args.gpu_ids,
            fix_script="",
            skip_fix=True,
            skip_clear_cache=True,
            local_work_root=args.local_work_root,
            runtime_root="",
            compile_cache_root=args.shared_compile_cache_root,
            clear_compile_cache=False,
            reuse_origin_json=reuse_origin_json,
            reuse_origin_hip_dir=reuse_origin_hip_dir,
            reuse_optimized_json=reuse_optimized_json,
            reuse_optimized_hip_dir=reuse_optimized_hip_dir,
            eval_backend=args.eval_backend,
            reference_hip_dir=args.reference_hip_dir or str(dataset_root / "hip_code"),
            reference_cache_mode="golden+compile",
        )
    )


def run_kernelbench_level(args: argparse.Namespace, layout: KernelBenchRunLayout, level: str, env: dict[str, str]) -> None:
    level_layout = layout.level(level)
    dataset_root = layout.subset_root / level
    input_dir = dataset_root / "hip_code"
    generation_dir = level_layout.generated_dir
    eval_dir = level_layout.eval_dir

    if not input_dir.is_dir() and not args.dry_run:
        raise SystemExit(f"Input subset directory not found: {input_dir}")
    if not args.reuse_outputs and not args.skip_generation and not args.dry_run:
        shutil.rmtree(generation_dir, ignore_errors=True)
    if not args.reuse_outputs and not args.skip_eval and not args.dry_run:
        shutil.rmtree(eval_dir, ignore_errors=True)
    if not args.dry_run:
        generation_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Kernelbench HIP case: {level}")
    print(f"Rollout n: {args.rollout_n}")
    print(f"Context mode: {args.context_mode}")
    print(f"Optimization paradigm: {args.optimization_paradigm}")
    print(f"Dataset root: {dataset_root}")
    print(f"Generation output: {generation_dir}")
    print(f"Evaluation output: {eval_dir}")
    print("=" * 60)

    if not args.skip_generation:
        if args.reuse_outputs and not args.dry_run and generation_complete_for_level(level_layout.generation_manifest, args):
            print(f"Skipping generation for {level}; existing manifest found: {level_layout.generation_manifest}")
        else:
            ensure_profiles_for_level(args, layout, level, dataset_root, env)
            context = prepare_context_for_level(args, level, input_dir, env)
            rollout_indices = materialize_reuse_for_level(args, level, generation_dir, context, env)
            if rollout_indices or not args.reuse_generation_from_root:
                run_generation(args, level, input_dir, generation_dir, rollout_indices, context, env)
            else:
                print(f"Skipping generation for {level}; generation reuse materialized all requested rollouts.")
    else:
        print(f"Skipping generation for {level}")

    if not args.dry_run:
        run_command(report_cmd("summarize-generation", "--manifest", level_layout.generation_manifest, "--label", level), env=env)

    if not args.skip_eval:
        if args.reuse_outputs and not args.dry_run and eval_complete_for_level(eval_dir):
            print(f"Skipping evaluation for {level}; existing comparison found: {level_layout.comparison_json}")
        else:
            run_evaluation(args, level, dataset_root, generation_dir, eval_dir, env)
    else:
        print(f"Skipping evaluation for {level}")

    if not args.dry_run and not args.skip_eval:
        print(f"[KERNELBENCH_HIP:{level}] comparison results:")
        print(f"  {level_layout.comparison_json}")
        print(f"  {level_layout.comparison_csv}")
        print(f"  {level_layout.comparison_perf_trace_csv}")
    print()


def kernelbench_run(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="Run KernelBench HIP subset generation and server-backed comparison eval.")
    add_common_run_args(parser)
    args = resolved_kernelbench_args(parser.parse_args(argv))
    env = prepare_gpu_env(args.gpu_ids)
    layout = KernelBenchRunLayout(Path(args.output_root))

    print("=" * 60)
    print("Kernelbench HIP evaluation")
    print(f"Output root: {args.output_root}")
    print(f"Subset root: {layout.subset_root}")
    print(f"Model source: {args.model or f'cached model at {args.cache_dir}'}")
    print(f"GPU IDs: {args.gpu_ids}")
    print(f"Context mode: {args.context_mode}")
    print(f"Optimization paradigm: {args.optimization_paradigm}")
    print(f"Eval backend: {args.eval_backend}")
    print(f"Reuse outputs: {args.reuse_outputs}")
    print("=" * 60)

    if not args.reuse_outputs and not args.skip_generation and not args.dry_run:
        shutil.rmtree(args.output_root, ignore_errors=True)
    if not args.dry_run:
        Path(args.output_root).mkdir(parents=True, exist_ok=True)

    stage_subset(args, layout, env)
    for level in DEFAULT_LEVELS:
        run_kernelbench_level(args, layout, level, env)

    if not args.dry_run and not args.skip_eval:
        cmd = report_cmd("summarize-run", "--run-root", args.output_root, "--subset-manifest", layout.subset_manifest)
        for level in DEFAULT_LEVELS:
            cmd += ["--level", level]
        run_command(cmd, env=env)
    elif args.dry_run:
        print("[DRY RUN] Skipping summary until evaluation results exist.")
    else:
        print("Skipping summary because evaluation was skipped.")
    print("Dry run completed." if args.dry_run else "Kernelbench HIP evaluation run completed.")


def launch_rollouts(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="Launch KernelBench HIP rollout sweep.")
    parser.add_argument("--model", default=os.environ.get("HIP_KIT_MODEL_SOURCE", ""))
    parser.add_argument("--gpu_ids", "--gpu-ids", dest="gpu_ids", default=default_gpu_ids())
    parser.add_argument("--n_gpus", "--n-gpus", dest="n_gpus", default="8")
    parser.add_argument("--rollout_ns", "--rollout-ns", dest="rollout_ns", default="4,16")
    parser.add_argument(
        "--output_root_base",
        "--output-root-base",
        dest="output_root_base",
        default=os.environ.get(
            "OUTPUT_ROOT_BASE",
            str(REPO_ROOT / "outputs" / "HIP_benchmark_kit" / "kernelbench_hip_eval_100" / "rollout_sweep"),
        ),
    )
    parser.add_argument("--eval_workers", "--eval-workers", dest="eval_workers", default="8")
    parser.add_argument("--perf_iterations", "--perf-iterations", dest="perf_iterations", default="10")
    parser.add_argument("--temperature", default="1")
    parser.add_argument("--disable_rollout_reuse", "--disable-rollout-reuse", action="store_true")
    known, forwarded = parser.parse_known_args(argv)
    rollout_ns = [item.strip() for item in known.rollout_ns.replace(" ", "").split(",") if item.strip()]
    has_reuse = any(
        item.startswith(("--reuse_rollout_from_root", "--reuse-rollout-from-root", "--reuse_generation_from_root", "--reuse_eval_from_root"))
        or item.startswith(("--reuse-generation-from-root", "--reuse-eval-from-root"))
        for item in forwarded
    )
    for rollout_n in rollout_ns:
        print(f"=== Running kernelbench_hip rollout_n={rollout_n} ===")
        rollout_args = list(forwarded)
        if not known.disable_rollout_reuse and not has_reuse and rollout_n == "16":
            rollout_args += ["--reuse_rollout_from_root", str(Path(known.output_root_base) / "rollout_n_4")]
        kernelbench_run(
            [
                "--model",
                known.model,
                "--gpu_ids",
                known.gpu_ids,
                "--n_gpus",
                known.n_gpus,
                "--eval_workers",
                known.eval_workers,
                "--perf_iterations",
                known.perf_iterations,
                "--temperature",
                known.temperature,
                *rollout_args,
                "--rollout_n",
                rollout_n,
                "--output_root",
                str(Path(known.output_root_base) / f"rollout_n_{rollout_n}"),
            ]
        )


def add_multiturn_args(parser: argparse.ArgumentParser) -> None:
    add_common_run_args(parser)
    parser.add_argument("--turn_mode", "--turn-mode", dest="turn_mode", choices=("single", "multi"), default="single")
    parser.add_argument("--turns", default="1", help="Total generation/eval rounds. Must be 1 in single mode and >=2 in multi mode.")
    parser.add_argument("--level", default="level-3", choices=(*DEFAULT_LEVELS, "all"))
    parser.add_argument("--sample_count", "--sample-count", dest="sample_count", default="8")
    parser.add_argument("--profile_generated", "--profile-generated", dest="profile_generated", choices=("always", "valid-only", "never"), default="valid-only")
    parser.add_argument("--origin_profile_context", "--origin-profile-context", dest="origin_profile_context", choices=ORIGIN_PROFILE_CONTEXT_CHOICES, default="off")
    parser.add_argument("--feedback_max_chars", "--feedback-max-chars", dest="feedback_max_chars", default="4000")


def validate_multiturn_args(args: argparse.Namespace) -> argparse.Namespace:
    args.turns = int(args.turns)
    args.sample_count = int(args.sample_count)
    args.feedback_max_chars = int(args.feedback_max_chars)
    if args.sample_count <= 0:
        raise SystemExit("--sample_count must be positive")
    if args.turn_mode == "single" and args.turns != 1:
        raise SystemExit("--turn_mode single requires --turns 1")
    if args.turn_mode == "multi" and args.turns < 2:
        raise SystemExit("--turn_mode multi requires --turns >= 2")
    if args.turn_mode == "multi" and int(args.rollout_n) != 1:
        raise SystemExit("multi-turn profiling currently requires --rollout_n 1")
    if args.origin_profile_context not in ORIGIN_PROFILE_CONTEXT_CHOICES:
        raise SystemExit(f"Unsupported origin_profile_context: {args.origin_profile_context}")
    if args.profile_missing_policy not in {"fail", "skip", "empty"}:
        raise SystemExit(f"Unsupported profile missing policy: {args.profile_missing_policy}")
    if args.eval_backend not in {"server-inprocess", "sandbox-inprocess"}:
        raise SystemExit(f"Unsupported eval backend: {args.eval_backend}")
    if not args.output_root:
        args.output_root = str(
            REPO_ROOT
            / "outputs"
            / "HIP_benchmark_kit"
            / "multiturn_profile"
            / f"global_step_300_{args.level}_{args.sample_count}_{args.turn_mode}_turns{args.turns}"
        )
    if not args.profile_artifact_root:
        args.profile_artifact_root = str(Path(args.output_root) / "starter_profiling" / "artifacts")
    if not args.profile_prompt_root:
        args.profile_prompt_root = str(Path(args.output_root) / "starter_profiling" / "prompt_maps")
    if not args.profile_gpu_ids:
        args.profile_gpu_ids = args.gpu_ids
    return args


def stage_single_level_subset(args: argparse.Namespace, layout: KernelBenchRunLayout, env: dict[str, str]) -> None:
    cmd = report_cmd(
        "stage-subset",
        "--source-root",
        args.kernelbench_hip_root,
        "--subset-root",
        layout.subset_root,
        "--manifest",
        layout.subset_manifest,
        "--level",
        f"{args.level}:{args.sample_count}",
    )
    run_command(cmd, dry_run=args.dry_run, env=env)


def origin_baseline_complete(baseline_eval_dir: Path, expected_count: int) -> bool:
    rows = _safe_read_json(baseline_eval_dir / BASELINE_RESULTS_JSON)
    if not isinstance(rows, list) or len(rows) < int(expected_count):
        return False
    return all(isinstance(row, dict) and row.get("base_name") and row.get("hip_file") for row in rows[: int(expected_count)])


def prepare_origin_baseline_eval(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    baseline_eval_dir: Path,
    env: dict[str, str],
) -> None:
    cmd: list[object] = [
        sys.executable,
        "-m",
        "HIP_benchmark_kit.eval.runner",
        "comprehensive",
        "--skip-fix",
        "--skip-clear-cache",
        "--hip_code_dir",
        dataset_root / "hip_code",
        "--pytorch_func_dir",
        dataset_root / "pytorch_code_functional",
        "--pytorch_modu_dir",
        dataset_root / "pytorch_code_module",
        "--output_dir",
        baseline_eval_dir,
        "--max-workers",
        args.eval_workers,
        "--perf-iterations",
        args.perf_iterations,
        "--rtol",
        args.rtol,
        "--atol",
        args.atol,
        "--gpu-ids",
        args.gpu_ids,
        "--eval-backend",
        args.eval_backend,
        "--reference-hip-code-dir",
        dataset_root / "hip_code",
        "--artifact-side",
        "origin",
    ]
    if args.local_work_root:
        cmd += ["--local-work-root", args.local_work_root]
    if args.shared_compile_cache_root:
        cmd += ["--compile-cache-root", args.shared_compile_cache_root]
    print(f"Preparing fixed origin baseline: {command_text(cmd)}")
    if args.dry_run:
        run_command(cmd, dry_run=True, env=env)
        return
    baseline_eval_dir.mkdir(parents=True, exist_ok=True)
    run_comprehensive_eval(
        argparse.Namespace(
            dataset_name="",
            hip_code_dir=str(dataset_root / "hip_code"),
            pytorch_func_dir=str(dataset_root / "pytorch_code_functional"),
            pytorch_modu_dir=str(dataset_root / "pytorch_code_module"),
            output_dir=str(baseline_eval_dir),
            max_workers=args.eval_workers,
            perf_iterations=args.perf_iterations,
            rtol=args.rtol,
            atol=args.atol,
            gpu_ids=args.gpu_ids,
            fix_script="",
            skip_fix=True,
            skip_clear_cache=True,
            cleanup_input_dir=False,
            local_work_root=args.local_work_root,
            runtime_dir=str(baseline_eval_dir / "runtime"),
            compile_cache_root=args.shared_compile_cache_root or str(baseline_eval_dir / "reference_cache"),
            clear_compile_cache=False,
            disable_compile_cache=False,
            artifact_side="origin",
            reuse_json="",
            reuse_hip_code_dir="",
            eval_backend=args.eval_backend,
            reference_hip_code_dir=str(dataset_root / "hip_code"),
            reference_cache_mode="golden+compile",
        )
    )


def prepare_starter_profile_context(
    args: argparse.Namespace,
    layout: KernelBenchRunLayout,
    *,
    dataset_root: Path,
    input_dir: Path,
    env: dict[str, str],
) -> dict[str, str]:
    profile_dir = Path(args.profile_artifact_root) / args.level
    if not args.skip_profile_ensure:
        profile_script = Path(args.profile_script)
        if not profile_script.is_file() and not args.dry_run:
            raise SystemExit(f"Profiling script not found: {profile_script}")
        cmd = [
            sys.executable,
            "-m",
            "HIP_benchmark_kit.profiling_context.ensure",
            "--subset-manifest",
            layout.subset_manifest,
            "--level",
            args.level,
            "--dataset-root",
            dataset_root,
            "--profile-artifact-root",
            args.profile_artifact_root,
            "--profile-script",
            profile_script,
            "--gpu-ids",
            args.profile_gpu_ids,
            "--parallel-workers",
            args.profile_parallel_workers,
            "--compile-workers",
            args.profile_compile_workers,
            "--prewarm-iters",
            args.profile_prewarm_iters,
            "--profile-iters",
            args.profile_iters,
            "--timeout-seconds",
            args.profile_timeout_seconds,
            "--metadata-mode",
            args.profile_metadata_mode,
        ]
        print(f"Profiling starter kernels before turn_01: {command_text(cmd)}")
        if args.dry_run:
            run_command(cmd, dry_run=True, env=env)
        else:
            profile_dir = ensure_metrix_profiles(
                subset_manifest=layout.subset_manifest,
                level=args.level,
                dataset_root=dataset_root,
                profile_artifact_root=Path(args.profile_artifact_root),
                profile_script=profile_script,
                gpu_ids=args.profile_gpu_ids,
                parallel_workers=int(args.profile_parallel_workers),
                compile_workers=int(args.profile_compile_workers),
                prewarm_iters=int(args.profile_prewarm_iters),
                profile_iters=int(args.profile_iters),
                timeout_seconds=int(args.profile_timeout_seconds),
                metadata_mode=args.profile_metadata_mode,
            )
    else:
        print("Skipping starter profiling ensure; building prompt map from existing artifacts.")

    arm = "starter_profile_raw"
    prompt_map_json = Path(args.profile_prompt_root) / f"{args.level}_starter_prompt_map.json"
    cmd = [
        sys.executable,
        "-m",
        "HIP_benchmark_kit.profiling_context.cards",
        "--profile-dir",
        profile_dir,
        "--input-dir",
        input_dir,
        "--output-json",
        prompt_map_json,
        "--arms",
        arm,
        "--missing-policy",
        args.profile_missing_policy,
    ]
    print(f"Preparing starter profiling prompt map: {command_text(cmd)}")
    if args.dry_run:
        run_command(cmd, dry_run=True, env=env)
    else:
        build_profile_cards(
            profile_dir=profile_dir,
            input_dir=input_dir,
            output_json=prompt_map_json,
            arms=[arm],
            missing_policy=args.profile_missing_policy,
        )
    return {
        "prompt_map_json": str(prompt_map_json),
        "prompt_map_arm": arm,
        "profile_dir": str(profile_dir),
    }


def prepare_initial_profile_context(
    args: argparse.Namespace,
    layout: KernelBenchRunLayout,
    *,
    dataset_root: Path,
    input_dir: Path,
    env: dict[str, str],
) -> dict[str, str] | None:
    if args.turn_mode == "multi":
        return prepare_starter_profile_context(
            args,
            layout,
            dataset_root=dataset_root,
            input_dir=input_dir,
            env=env,
        )
    if args.origin_profile_context == "off":
        print("Skipping origin profiling context for single-turn run: origin_profile_context=off.")
        return None

    profile_args = argparse.Namespace(**vars(args))
    profile_args.skip_profile_ensure = args.origin_profile_context == "use_existing"
    if args.origin_profile_context == "ensure_and_use":
        profile_args.skip_profile_ensure = False
    print(f"Preparing origin profiling context for single-turn run: origin_profile_context={args.origin_profile_context}.")
    return prepare_starter_profile_context(
        profile_args,
        layout,
        dataset_root=dataset_root,
        input_dir=input_dir,
        env=env,
    )


def generation_context_for_turn(
    args: argparse.Namespace,
    turn: int,
    turn_layout,
    feedback_context_json: Path | None,
    starter_profile_context: dict[str, str] | None = None,
) -> dict[str, str]:
    context = {
        "prompt_map_json": "",
        "prompt_map_arm": "",
        "raw_response_dir": str(turn_layout.raw_response_dir),
        "feedback_context_json": str(feedback_context_json) if feedback_context_json else "",
        "experiment_arm": f"{args.turn_mode}_turn_{turn:02d}",
        "data_source": "kernel-agent-react-train",
        "optimization_paradigm": getattr(args, "optimization_paradigm", DEFAULT_OPTIMIZATION_PARADIGM),
    }
    if turn == 1 and starter_profile_context:
        context["prompt_map_json"] = starter_profile_context.get("prompt_map_json", "")
        context["prompt_map_arm"] = starter_profile_context.get("prompt_map_arm", "")
    return context


def _safe_read_json(path: Path) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError, TypeError):
        return None


def turn_generation_complete(
    turn_layout,
    expected_count: int,
    optimization_paradigm: str = "",
    rollout_n: int = 1,
) -> bool:
    manifest_path = turn_layout.generated_dir / "generation_manifest.json"
    manifest = _safe_read_json(manifest_path)
    if not isinstance(manifest, dict):
        return False
    if optimization_paradigm and manifest.get("optimization_paradigm") != optimization_paradigm:
        return False
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) < int(expected_count) * int(rollout_n):
        return False

    by_input_sample: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        input_file = str(record.get("input_file") or "")
        sample_idx = record.get("sample_idx")
        if sample_idx is None and int(rollout_n) == 1:
            sample_idx = 0
        if input_file and sample_idx is not None:
            by_input_sample[(input_file, int(sample_idx))] = record
    input_files = sorted({input_file for input_file, _ in by_input_sample})
    if len(input_files) < int(expected_count):
        return False

    for input_file in input_files[: int(expected_count)]:
        for sample_idx in range(int(rollout_n)):
            record = by_input_sample.get((input_file, sample_idx))
            if not record:
                return False
            if record.get("parse_ok") is True:
                output_path = Path(str(record.get("output_path") or ""))
                if record.get("saved") is not True:
                    return False
                if not output_path.is_file():
                    return False
                continue
            raw_response_path = Path(str(record.get("raw_response_path") or ""))
            if not raw_response_path.is_file():
                return False
    return True


def turn_eval_complete(turn_layout) -> bool:
    rows = _safe_read_json(turn_layout.comparison_json)
    if not isinstance(rows, list) or not rows:
        return False
    required = ("base_name", "origin_hip_file", "optimized_hip_file")
    return all(isinstance(row, dict) and all(row.get(field) for field in required) for row in rows)


def expected_profile_basenames(args: argparse.Namespace, input_dir: Path, comparison_json: Path) -> set[str]:
    if args.profile_generated == "never":
        return set()
    if args.profile_generated == "always":
        return {path.stem for path in input_dir.glob("*.hip") if path.is_file()}
    rows = index_comparison_rows(comparison_json)
    return {
        base_name
        for base_name, row in rows.items()
        if row.get("optimized_compile_ok") is True
        and row.get("optimized_run_ok") is True
        and row.get("optimized_match_ok") is True
    }


def turn_generated_profile_complete(args: argparse.Namespace, input_dir: Path, turn_layout) -> bool:
    expected = expected_profile_basenames(args, input_dir, turn_layout.comparison_json)
    if not expected:
        return True
    staging_manifest = turn_layout.profiling_dir / "generated_profile_staging_manifest.json"
    if not staging_manifest.is_file():
        return False
    return all((turn_layout.profiling_generated_dir / f"{base_name}_filtered.json").is_file() for base_name in expected)


def feedback_context_complete(path: Path, expected_count: int) -> bool:
    payload = _safe_read_json(path)
    if not isinstance(payload, dict):
        return False
    feedback_map = payload.get("feedback_map")
    if not isinstance(feedback_map, dict) or len(feedback_map) < int(expected_count):
        return False
    entries = list(feedback_map.values())[: int(expected_count)]
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        if not entry.get("base_name") or not entry.get("feedback_text") or entry.get("previous_turn") is None:
            return False
        if not entry.get("blocked_reason"):
            previous_generated_path = Path(str(entry.get("previous_generated_path") or ""))
            if not previous_generated_path.is_file():
                return False
    return True


def profile_generated_for_turn(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    input_dir: Path,
    turn_layout,
    env: dict[str, str],
) -> None:
    if args.profile_generated == "never":
        print(f"Skipping generated profiling for {turn_layout.name}: profile_generated=never")
        return
    comparison_rows = index_comparison_rows(turn_layout.comparison_json)
    if args.dry_run:
        profile_script = Path(args.profile_script)
        cmd = [
            sys.executable,
            str(profile_script),
            "--hip-dir",
            turn_layout.profiling_staging_dir,
            "--functional-dir",
            dataset_root / "pytorch_code_functional",
            "--output-dir",
            turn_layout.profiling_generated_dir,
            "--gpu-ids",
            args.profile_gpu_ids,
            "--parallel-workers",
            args.profile_parallel_workers,
            "--compile-workers",
            args.profile_compile_workers,
            "--prewarm-iters",
            args.profile_prewarm_iters,
            "--profile-iters",
            args.profile_iters,
            "--timeout-seconds",
            args.profile_timeout_seconds,
            "--metadata-mode",
            args.profile_metadata_mode,
        ]
        run_command(cmd, dry_run=True, env=env)
        return
    staged_manifest = stage_generated_kernels_for_profile(
        generated_dir=turn_layout.generated_dir,
        original_input_dir=input_dir,
        staging_dir=turn_layout.profiling_staging_dir,
        comparison_rows=comparison_rows,
        profile_generated=args.profile_generated,
    )
    write_json(turn_layout.profiling_dir / "generated_profile_staging_manifest.json", staged_manifest)
    staged = staged_manifest.get("staged", [])
    if not staged:
        print(f"Skipping generated profiling for {turn_layout.name}; no generated kernels staged.")
        return
    profile_script = Path(args.profile_script)
    if not profile_script.is_file():
        raise SystemExit(f"Profiling script not found: {profile_script}")
    samples = ",".join(str(item["base_name"]) for item in staged)
    cmd = [
        sys.executable,
        str(profile_script),
        "--hip-dir",
        turn_layout.profiling_staging_dir,
        "--functional-dir",
        dataset_root / "pytorch_code_functional",
        "--samples",
        samples,
        "--output-dir",
        turn_layout.profiling_generated_dir,
        "--gpu-ids",
        args.profile_gpu_ids,
        "--parallel-workers",
        args.profile_parallel_workers,
        "--compile-workers",
        args.profile_compile_workers,
        "--prewarm-iters",
        args.profile_prewarm_iters,
        "--profile-iters",
        args.profile_iters,
        "--timeout-seconds",
        args.profile_timeout_seconds,
        "--metadata-mode",
        args.profile_metadata_mode,
    ]
    print(f"Profiling generated kernels for {turn_layout.name}: {command_text(cmd)}")
    run_command(cmd, env=env)


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _input_base_name(input_file: str) -> str:
    return input_file[:-4] if input_file.endswith(".hip") else input_file


def _record_summary_key(record: dict[str, object]) -> tuple[str, int] | None:
    input_file = str(record.get("input_file") or "")
    if not input_file:
        return None
    sample_idx = record.get("sample_idx")
    if sample_idx is None:
        sample_idx = record.get("rollout_idx")
    return _input_base_name(input_file), _coerce_int(sample_idx, 0)


def _record_path(record: dict[str, object], key: str, base_dir: Path) -> Path | None:
    raw_value = str(record.get(key) or "")
    if raw_value:
        return Path(raw_value)
    file_key = key.removesuffix("_path") + "_file"
    file_value = str(record.get(file_key) or "")
    return base_dir / file_value if file_value else None


def _summary_profile_path(turn_layout, base_name: str, gen_idx: int) -> Path:
    candidate = turn_layout.profiling_generated_dir / f"{base_name}_gen{gen_idx}_filtered.json"
    if candidate.is_file():
        return candidate
    return turn_layout.profiling_generated_dir / f"{base_name}_filtered.json"


def build_turn_summary_rows(args: argparse.Namespace, layout: KernelBenchRunLayout) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for turn in range(1, args.turns + 1):
        turn_layout = layout.turn(args.level, turn)
        manifest_path = turn_layout.generated_dir / "generation_manifest.json"
        records = []
        if manifest_path.is_file():
            loaded_records = read_json(manifest_path).get("records", [])
            records = loaded_records if isinstance(loaded_records, list) else []
        comparison_rows = index_comparison_candidates(turn_layout.comparison_json)
        by_candidate = {
            key: record
            for record in records
            if isinstance(record, dict)
            for key in [_record_summary_key(record)]
            if key is not None
        }
        candidate_keys = sorted({*by_candidate.keys(), *comparison_rows.keys()})
        for base_name, gen_idx in candidate_keys:
            record = by_candidate.get((base_name, gen_idx), {})
            eval_row = comparison_rows.get((base_name, gen_idx), {})
            input_file = str(record.get("input_file") or eval_row.get("origin_hip_file") or f"{base_name}.hip")
            generated_path = _record_path(record, "output_path", turn_layout.generated_dir)
            if generated_path is None:
                optimized_file = str(eval_row.get("optimized_hip_file") or "")
                generated_path = turn_layout.generated_dir / optimized_file if optimized_file else turn_layout.generated_dir / f"{base_name}_gen{gen_idx}.hip"
            raw_path = _record_path(record, "raw_response_path", turn_layout.raw_response_dir)
            if raw_path is None:
                raw_path = turn_layout.raw_response_dir / f"{base_name}_gen{gen_idx}_raw_response.json"
            profile_path = _summary_profile_path(turn_layout, base_name, gen_idx)
            rows.append(
                {
                    "base_name": base_name,
                    "turn": turn,
                    "sample_idx": record.get("sample_idx", gen_idx),
                    "gen_idx": eval_row.get("gen_idx", gen_idx),
                    "input_file": input_file,
                    "optimized_hip_file": eval_row.get("optimized_hip_file") or generated_path.name,
                    "generated_path": str(generated_path),
                    "parse_ok": record.get("parse_ok"),
                    "saved": record.get("saved"),
                    "compile_ok": eval_row.get("optimized_compile_ok"),
                    "run_ok": eval_row.get("optimized_run_ok"),
                    "match_ok": eval_row.get("optimized_match_ok"),
                    "origin_hip_time_ms": eval_row.get("origin_hip_time_ms"),
                    "candidate_hip_time_ms": eval_row.get("optimized_hip_time_ms"),
                    "speedup": eval_row.get("speedup"),
                    "profile_artifact": str(profile_path) if profile_path.is_file() else "",
                    "raw_response_path": str(raw_path) if raw_path.is_file() else "",
                    "thought": load_raw_response_thought(raw_path),
                    "blocked_reason": record.get("blocked_reason") or eval_row.get("compare_error") or "",
                }
            )
    return rows


def _valid_speedup(row: dict[str, object]) -> float | None:
    if not (
        row.get("compile_ok") is True
        and row.get("run_ok") is True
        and row.get("match_ok") is True
    ):
        return None
    try:
        speedup = float(row.get("speedup"))
    except (TypeError, ValueError):
        return None
    return speedup if speedup > 0 else None


def _best_source_generated_path(args: argparse.Namespace, layout: KernelBenchRunLayout, best: dict[str, object]) -> Path:
    generated_path = str(best.get("generated_path") or "")
    if generated_path:
        return Path(generated_path)
    selected_turn = _coerce_int(best.get("turn"), 1)
    turn_generated_dir = layout.turn(args.level, selected_turn).generated_dir
    optimized_file = str(best.get("optimized_hip_file") or "")
    if optimized_file:
        return turn_generated_dir / optimized_file
    base_name = str(best.get("base_name") or "")
    gen_idx = _coerce_int(best.get("gen_idx"), 0)
    return turn_generated_dir / f"{base_name}_gen{gen_idx}.hip"


def write_best_valid_final(args: argparse.Namespace, layout: KernelBenchRunLayout, rows: list[dict[str, object]]) -> None:
    level_root = Path(args.output_root) / args.level
    final_root = level_root / "final"
    best_dir = final_root / "best_valid_generated"
    shutil.rmtree(best_dir, ignore_errors=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    by_base: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        base_name = str(row.get("base_name") or "")
        if base_name:
            by_base.setdefault(base_name, []).append(row)

    manifest_rows: list[dict[str, object]] = []
    for base_name in sorted(by_base):
        valid_rows = [row for row in by_base[base_name] if _valid_speedup(row) is not None]
        if not valid_rows:
            manifest_rows.append({"base_name": base_name, "status": "no_valid_candidate"})
            continue
        best = max(valid_rows, key=lambda row: _valid_speedup(row) or 0.0)
        selected_turn = _coerce_int(best.get("turn"), 1)
        source_generated_path = _best_source_generated_path(args, layout, best)
        entry: dict[str, object] = {
            "base_name": base_name,
            "status": "selected",
            "selected_turn": selected_turn,
            "selected_sample_idx": best.get("sample_idx"),
            "selected_gen_idx": best.get("gen_idx"),
            "speedup": best.get("speedup"),
            "origin_hip_time_ms": best.get("origin_hip_time_ms"),
            "candidate_hip_time_ms": best.get("candidate_hip_time_ms"),
            "optimized_hip_file": best.get("optimized_hip_file") or source_generated_path.name,
            "source_generated_path": str(source_generated_path),
            "raw_response_path": best.get("raw_response_path") or "",
            "profile_artifact": best.get("profile_artifact") or "",
        }
        if source_generated_path.is_file():
            shutil.copy2(source_generated_path, best_dir / source_generated_path.name)
        else:
            entry["status"] = "source_missing"
        manifest_rows.append(entry)

    write_json(final_root / "best_valid_manifest.json", manifest_rows)
    csv_path = final_root / "best_valid_summary.csv"
    fieldnames: list[str] = []
    for row in manifest_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)


def write_multiturn_summary(args: argparse.Namespace, layout: KernelBenchRunLayout) -> None:
    rows = build_turn_summary_rows(args, layout)
    level_root = Path(args.output_root) / args.level
    write_json(level_root / "multi_turn_summary.json", rows)
    csv_path = level_root / "multi_turn_summary.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    final_dir = level_root / "final" / "generated"
    shutil.rmtree(final_dir, ignore_errors=True)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    last_generated = layout.turn(args.level, args.turns).generated_dir
    if last_generated.is_dir():
        shutil.copytree(last_generated, final_dir, dirs_exist_ok=True)
    write_best_valid_final(args, layout, rows)
    print(f"[MULTITURN:{args.level}] summary: {level_root / 'multi_turn_summary.json'}")
    print(f"[MULTITURN:{args.level}] final generated: {final_dir}")
    print(f"[MULTITURN:{args.level}] best valid generated: {level_root / 'final' / 'best_valid_generated'}")


def multiturn_profile_run(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="Run explicit single-turn or multi-turn KernelBench profiling feedback experiments.")
    add_multiturn_args(parser)
    args = validate_multiturn_args(parser.parse_args(argv))
    # vLLM owns process groups and worker subprocesses. Multi-turn runs create
    # multiple engines sequentially, so isolate every generation turn in its own
    # process instead of reinitializing vLLM inside this orchestrator process.
    args.generation_subprocess = True
    env = prepare_gpu_env(args.gpu_ids)
    layout = KernelBenchRunLayout(Path(args.output_root))
    dataset_root = layout.subset_root / args.level
    input_dir = dataset_root / "hip_code"

    print("=" * 60)
    print("KernelBench HIP multi-turn profiling evaluation")
    print(f"Output root: {args.output_root}")
    print(f"Turn mode: {args.turn_mode}")
    print(f"Turns: {args.turns}")
    print(f"Level/sample count: {args.level}/{args.sample_count}")
    print(f"Model source: {args.model or f'cached model at {args.cache_dir}'}")
    print(f"Optimization paradigm: {args.optimization_paradigm}")
    print("=" * 60)

    if not args.reuse_outputs and not args.dry_run:
        shutil.rmtree(args.output_root, ignore_errors=True)
    if not args.dry_run:
        Path(args.output_root).mkdir(parents=True, exist_ok=True)
    stage_single_level_subset(args, layout, env)
    starter_profile_context = prepare_initial_profile_context(
        args,
        layout,
        dataset_root=dataset_root,
        input_dir=input_dir,
        env=env,
    )
    origin_baseline_eval_dir = (
        Path(args.origin_baseline_eval_root)
        if args.origin_baseline_eval_root
        else Path(args.output_root) / args.level / "origin_baseline" / "eval"
    )
    if not args.skip_eval:
        if args.reuse_outputs and not args.dry_run and origin_baseline_complete(origin_baseline_eval_dir, args.sample_count):
            print(f"Reusing fixed origin baseline: {origin_baseline_eval_dir / BASELINE_RESULTS_JSON}")
        else:
            prepare_origin_baseline_eval(
                args,
                dataset_root=dataset_root,
                baseline_eval_dir=origin_baseline_eval_dir,
                env=env,
            )

    feedback_context_json: Path | None = None
    for turn in range(1, args.turns + 1):
        turn_layout = layout.turn(args.level, turn)
        print("=" * 60)
        print(f"Turn {turn}/{args.turns}: {turn_layout.root}")
        print("=" * 60)
        if not args.dry_run:
            for path in (
                turn_layout.generated_dir,
                turn_layout.raw_response_dir,
                turn_layout.eval_dir,
                turn_layout.profiling_generated_dir,
                turn_layout.feedback_dir,
            ):
                path.mkdir(parents=True, exist_ok=True)
        context = generation_context_for_turn(args, turn, turn_layout, feedback_context_json, starter_profile_context)
        reuse_enabled = args.reuse_outputs and not args.dry_run
        generation_done = reuse_enabled and turn_generation_complete(
            turn_layout,
            args.sample_count,
            args.optimization_paradigm,
            int(args.rollout_n),
        )
        if not args.skip_generation:
            if generation_done:
                print(
                    f"Skipping generation for {turn_layout.name}; complete manifest found: "
                    f"{turn_layout.generated_dir / 'generation_manifest.json'}"
                )
            else:
                rollout_indices = ""
                if args.turn_mode == "single":
                    rollout_indices = materialize_reuse_for_level(
                        args,
                        args.level,
                        turn_layout.generated_dir,
                        context,
                        env,
                    )
                if rollout_indices or not args.reuse_generation_from_root or args.disable_rollout_reuse:
                    run_generation(args, args.level, input_dir, turn_layout.generated_dir, rollout_indices, context, env)
                else:
                    print(
                        f"Skipping generation for {turn_layout.name}; "
                        "generation reuse materialized all requested rollouts."
                    )
        else:
            print(f"Skipping generation for turn {turn}")
        if not args.dry_run:
            run_command(report_cmd("summarize-generation", "--manifest", turn_layout.generated_dir / "generation_manifest.json", "--label", f"{args.level}:{turn_layout.name}"), env=env)
        eval_done = reuse_enabled and turn_eval_complete(turn_layout)
        if not args.skip_eval:
            if eval_done:
                print(f"Skipping evaluation for {turn_layout.name}; comparison JSON found: {turn_layout.comparison_json}")
            else:
                run_evaluation(
                    args,
                    args.level,
                    dataset_root,
                    turn_layout.generated_dir,
                    turn_layout.eval_dir,
                    env,
                    origin_baseline_eval_dir=origin_baseline_eval_dir,
                )
        else:
            print(f"Skipping evaluation for turn {turn}")
        if not args.skip_eval:
            profile_done = reuse_enabled and turn_generated_profile_complete(args, input_dir, turn_layout)
            if profile_done:
                print(f"Skipping generated profiling for {turn_layout.name}; profile artifacts complete")
            else:
                profile_generated_for_turn(args, dataset_root=dataset_root, input_dir=input_dir, turn_layout=turn_layout, env=env)
        if turn < args.turns:
            feedback_context_json = turn_layout.feedback_context_json_for_next_turn()
            if args.dry_run:
                print(f"[DRY RUN] Would build feedback context: {feedback_context_json}")
            elif args.reuse_outputs and feedback_context_complete(feedback_context_json, args.sample_count):
                print(f"Reusing feedback context for turn {turn + 1}: {feedback_context_json}")
            else:
                print(f"Building feedback context for turn {turn + 1}: {feedback_context_json}")
                build_feedback_context(
                    original_input_dir=input_dir,
                    previous_generated_dir=turn_layout.generated_dir,
                    previous_raw_response_dir=turn_layout.raw_response_dir,
                    previous_comparison_json=turn_layout.comparison_json,
                    previous_profile_dir=turn_layout.profiling_generated_dir,
                    output_json=feedback_context_json,
                    previous_turn=turn,
                    feedback_max_chars=args.feedback_max_chars,
                )
    if args.dry_run:
        print("[DRY RUN] Skipping multi-turn summary until artifacts exist.")
    else:
        write_multiturn_summary(args, layout)
    print("Dry run completed." if args.dry_run else "Multi-turn profiling run completed.")


def launch_neurlps(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="Launch NeurIPS rollout sweep.")
    parser.add_argument("--model", default=os.environ.get("HIP_KIT_MODEL_SOURCE", ""))
    parser.add_argument("--gpu_ids", "--gpu-ids", dest="gpu_ids", default=default_gpu_ids())
    parser.add_argument("--n_gpus", "--n-gpus", dest="n_gpus", default="8")
    parser.add_argument("--rollout_ns", "--rollout-ns", dest="rollout_ns", default="1,4,16")
    parser.add_argument("--datasets", default="kernelbench25")
    parser.add_argument("--eval_workers", "--eval-workers", dest="eval_workers", default="8")
    parser.add_argument("--perf_iterations", "--perf-iterations", dest="perf_iterations", default="10")
    parser.add_argument("--temperature", default="1")
    parser.add_argument(
        "--output_root_base",
        "--output-root-base",
        dest="output_root_base",
        default=str(REPO_ROOT / "outputs" / "HIP_benchmark_kit" / "neurlps_full_rollout_sweep"),
    )
    known, forwarded = parser.parse_known_args(argv)
    for rollout_n in split_csv(known.rollout_ns):
        print(f"=== Running rollout_n={rollout_n} ===")
        neurlps_run(
            [
                "--mode",
                "full",
                "--model",
                known.model,
                "--gpu_ids",
                known.gpu_ids,
                "--n_gpus",
                known.n_gpus,
                "--datasets",
                known.datasets,
                "--eval_workers",
                known.eval_workers,
                "--perf_iterations",
                known.perf_iterations,
                "--temperature",
                known.temperature,
                *forwarded,
                "--rollout_n",
                rollout_n,
                "--output_root",
                str(Path(known.output_root_base) / f"rollout_n_{rollout_n}"),
            ]
        )


def neurlps_dataset_root(label: str) -> Path:
    mapping = {
        "kernelbench25": KIT_ROOT / "data" / "hip_eval_neurlps" / "hip_eval_dataset_kernelbench_25_tasks",
        "aicuda75": KIT_ROOT / "data" / "hip_eval_neurlps" / "hip_eval_aicuda_gpumode_dataset_75_tasks",
    }
    if label not in mapping:
        raise SystemExit(f"Unsupported dataset label: {label}")
    return mapping[label]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_neurlps_case(args: argparse.Namespace, label: str, dataset_root: Path, samples: list[str] | None, env: dict[str, str]) -> None:
    case_root = Path(args.output_root) / label
    staging_dir = case_root / "staging" / "hip_code"
    generation_dir = case_root / "generated"
    eval_dir = case_root / "eval"
    input_dir = staging_dir if samples is not None else dataset_root / "hip_code"
    if samples is not None:
        if not args.dry_run:
            shutil.rmtree(case_root, ignore_errors=True)
            staging_dir.mkdir(parents=True, exist_ok=True)
            generation_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)
            for sample_file in samples:
                source = dataset_root / "hip_code" / sample_file
                if not source.is_file():
                    raise SystemExit(f"Sample file not found: {source}")
                shutil.copy2(source, staging_dir / source.name)
    else:
        if not input_dir.is_dir():
            raise SystemExit(f"Input dataset directory not found: {input_dir}")
        if not args.reuse_outputs and not args.skip_generation and not args.dry_run:
            shutil.rmtree(generation_dir, ignore_errors=True)
        if not args.reuse_outputs and not args.skip_eval and not args.dry_run:
            shutil.rmtree(eval_dir, ignore_errors=True)
        if not args.dry_run:
            generation_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"NeurIPS case: {label}")
    print(f"Dataset root: {dataset_root}")
    print(f"Input HIP dir: {input_dir}")
    print(f"Optimization paradigm: {args.optimization_paradigm}")
    print("=" * 60)

    gen_cmd: list[object] = [
        sys.executable,
        "-m",
        "HIP_benchmark_kit.gen_hip_kernel.runner",
        "--input_dir",
        input_dir,
        "--output_dir",
        generation_dir,
        "--cache_dir",
        args.cache_dir,
        "--rollout_n",
        args.rollout_n,
        "--n_gpus",
        args.n_gpus,
        "--temperature",
        args.temperature,
        "--output_contract",
        args.output_contract,
        "--optimization_paradigm",
        args.optimization_paradigm,
        "--target_gpu",
        args.target_gpu,
    ]
    if args.seed_base:
        gen_cmd += ["--seed_base", args.seed_base]
    if args.model:
        gen_cmd += ["--model", args.model]
    manifest_path = generation_dir / "generation_manifest.json"
    if not args.skip_generation:
        if args.reuse_outputs and manifest_path.is_file() and not args.dry_run:
            print(f"Skipping generation for {label}; existing manifest found: {manifest_path}")
        else:
            print(f"Running generation: {command_text(gen_cmd)}")
            if args.dry_run:
                run_command(gen_cmd, dry_run=True, env=env)
            else:
                run_generation_api(
                    argparse.Namespace(
                        model=args.model,
                        config=str(KIT_ROOT / "gen_hip_kernel" / "config.yaml"),
                        input_dir=str(input_dir),
                        output_dir=str(generation_dir),
                        output_root="",
                        cache_dir=args.cache_dir,
                        output_contract=args.output_contract,
                        optimization_paradigm=args.optimization_paradigm,
                        target_gpu=args.target_gpu,
                        rollout_n=args.rollout_n,
                        rollout_indices="",
                        n_gpus=args.n_gpus,
                        temperature=args.temperature,
                        seed_base=args.seed_base,
                        gpu_ids=args.gpu_ids,
                        prompt_text="",
                        prompt_map_json="",
                        prompt_map_arm="",
                        raw_response_dir="",
                        feedback_context_json="",
                        data_source="",
                        experiment_arm="",
                        clear_cache=False,
                    )
                )
    if not args.dry_run:
        run_command(report_cmd("summarize-generation", "--manifest", manifest_path, "--label", label), env=env)

    eval_cmd = [
        sys.executable,
        "-m",
        "HIP_benchmark_kit.eval.runner",
        "compare",
        "--skip-fix",
        "--origin_hip_dir",
        input_dir,
        "--optimized_hip_dir",
        generation_dir,
        "--pytorch_func_dir",
        dataset_root / "pytorch_code_functional",
        "--pytorch_modu_dir",
        dataset_root / "pytorch_code_module",
        "--output_dir",
        eval_dir,
        "--max-workers",
        args.eval_workers,
        "--perf-iterations",
        args.perf_iterations,
        "--rtol",
        args.rtol,
        "--atol",
        args.atol,
        "--gpu-ids",
        args.gpu_ids,
        "--eval-backend",
        "server-inprocess",
        "--reference-hip-dir",
        input_dir,
    ]
    if not args.skip_eval:
        comparison_json = eval_dir / "comparison" / COMPARISON_RESULTS_JSON
        if args.reuse_outputs and comparison_json.is_file() and not args.dry_run:
            print(f"Skipping evaluation for {label}; existing comparison found: {comparison_json}")
        else:
            print(f"Running evaluation: {command_text(eval_cmd)}")
            if args.dry_run:
                run_command(eval_cmd, dry_run=True, env=env)
            else:
                compare_hip_dirs(
                    argparse.Namespace(
                        origin_hip_dir=str(input_dir),
                        optimized_hip_dir=str(generation_dir),
                        pytorch_func_dir=str(dataset_root / "pytorch_code_functional"),
                        pytorch_modu_dir=str(dataset_root / "pytorch_code_module"),
                        output_dir=str(eval_dir),
                        max_workers=args.eval_workers,
                        perf_iterations=args.perf_iterations,
                        rtol=args.rtol,
                        atol=args.atol,
                        gpu_ids=args.gpu_ids,
                        fix_script="",
                        skip_fix=True,
                        skip_clear_cache=True,
                        local_work_root="",
                        runtime_root="",
                        compile_cache_root="",
                        clear_compile_cache=False,
                        reuse_origin_json="",
                        reuse_origin_hip_dir="",
                        reuse_optimized_json="",
                        reuse_optimized_hip_dir="",
                        eval_backend="server-inprocess",
                        reference_hip_dir=str(input_dir),
                        reference_cache_mode="golden+compile",
                    )
                )


def neurlps_run(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="Run NeurIPS HIP generation and server-backed eval.")
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--model", default="")
    parser.add_argument("--cache_dir", "--cache-dir", dest="cache_dir", default=os.environ.get("MODEL_CACHE", "/dev/shm/hip_vllm_model_cache"))
    parser.add_argument("--output_root", "--output-root", dest="output_root", default="")
    parser.add_argument("--datasets", default="kernelbench25,aicuda75")
    parser.add_argument("--gpu_ids", "--gpu-ids", dest="gpu_ids", default=default_gpu_ids())
    parser.add_argument("--n_gpus", "--n-gpus", dest="n_gpus", default="8")
    parser.add_argument("--rollout_n", "--rollout-n", dest="rollout_n", default="8")
    parser.add_argument("--eval_workers", "--eval-workers", dest="eval_workers", default="8")
    parser.add_argument("--perf_iterations", "--perf-iterations", dest="perf_iterations", default="10")
    parser.add_argument("--temperature", default="1.0")
    parser.add_argument("--seed_base", "--seed-base", dest="seed_base", default="")
    parser.add_argument("--output_contract", "--output-contract", dest="output_contract", default="sample_json_v1")
    parser.add_argument(
        "--optimization_paradigm",
        "--optimization-paradigm",
        dest="optimization_paradigm",
        default=DEFAULT_OPTIMIZATION_PARADIGM,
    )
    parser.add_argument("--target_gpu", "--target-gpu", dest="target_gpu", default="mi300x")
    parser.add_argument("--rtol", default="1e-4")
    parser.add_argument("--atol", default="1e-5")
    parser.add_argument("--kernelbench_samples", "--kernelbench-samples", dest="kernelbench_samples", default="768_matmul_warp_optimized_edit_1.hip")
    parser.add_argument("--aicuda_samples", "--aicuda-samples", dest="aicuda_samples", default="hip_4618_Mul.hip")
    parser.add_argument("--reuse_outputs", "--reuse-outputs", dest="reuse_outputs", action="store_true", default=False)
    parser.add_argument("--skip_generation", "--skip-generation", dest="skip_generation", action="store_true")
    parser.add_argument("--skip_eval", "--skip-eval", dest="skip_eval", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true")
    args = parser.parse_args(argv)
    args.optimization_paradigm = normalize_optimization_paradigm(args.optimization_paradigm)
    if not args.output_root:
        args.output_root = str(REPO_ROOT / "outputs" / "HIP_benchmark_kit" / ("tests/neurlps_smoke" if args.mode == "smoke" else "neurlps_full"))
    env = prepare_gpu_env(args.gpu_ids)
    labels = ["kernelbench25", "aicuda75"] if args.datasets == "all" else split_csv(args.datasets)
    for label in labels:
        samples = None
        if args.mode == "smoke":
            samples = split_csv(args.kernelbench_samples if label == "kernelbench25" else args.aicuda_samples)
        run_neurlps_case(args, label, neurlps_dataset_root(label), samples, env)
    print("Dry run completed." if args.dry_run else "NeurIPS run completed.")


def reeval_existing(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate existing KernelBench HIP rollout outputs.")
    parser.add_argument("--source_root", "--source-root", dest="source_root", default=str(REPO_ROOT / "outputs/HIP_benchmark_kit/kernelbench_hip_eval_100/global_step_300"))
    parser.add_argument("--reeval_root", "--reeval-root", dest="reeval_root", default="")
    parser.add_argument("--subset_source_rollout", "--subset-source-rollout", dest="subset_source_rollout", default="16")
    parser.add_argument("--gpu_ids", "--gpu-ids", dest="gpu_ids", default=default_gpu_ids())
    parser.add_argument("--eval_workers", "--eval-workers", dest="eval_workers", default="4")
    parser.add_argument("--perf_iterations", "--perf-iterations", dest="perf_iterations", default="10")
    parser.add_argument("--rtol", default="1e-4")
    parser.add_argument("--atol", default="1e-5")
    parser.add_argument("--rollouts", default="1,4,16")
    parser.add_argument("--reuse_outputs", "--reuse-outputs", dest="reuse_outputs", action="store_true", default=False)
    parser.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true")
    args = parser.parse_args(argv)
    source_root = Path(args.source_root)
    reeval_root = Path(args.reeval_root) if args.reeval_root else source_root / "reeval_fixed_origin"
    subset_root = source_root / f"rollout_n_{args.subset_source_rollout}" / "subset" / "kernelbench_hip_100"
    fixed_origin_root = reeval_root / "fixed_origin"
    local_work_root = reeval_root / "local_work"
    runtime_root = reeval_root / "runtime"
    baseline_cache_root = reeval_root / "reference_cache" / "baseline"
    optimized_cache_root = reeval_root / "reference_cache" / "optimized"
    origin_clean_manifest = fixed_origin_root / "origin_clean_manifest.json"
    rollouts = split_csv(args.rollouts)
    env = prepare_gpu_env(args.gpu_ids)

    def require_dir(path: Path) -> None:
        if not path.is_dir() and not args.dry_run:
            raise SystemExit(f"Required directory not found: {path}")

    def run_eval(hip_dir: Path, dataset_root: Path, output_dir: Path, runtime_dir: Path, cache_root: Path, artifact_side: str) -> None:
        cmd = [
            sys.executable,
            "-m",
            "HIP_benchmark_kit.eval.runner",
            "comprehensive",
            "--skip-fix",
            "--skip-clear-cache",
            "--hip_code_dir",
            hip_dir,
            "--pytorch_func_dir",
            dataset_root / "pytorch_code_functional",
            "--pytorch_modu_dir",
            dataset_root / "pytorch_code_module",
            "--output_dir",
            output_dir,
            "--max-workers",
            args.eval_workers,
            "--perf-iterations",
            args.perf_iterations,
            "--rtol",
            args.rtol,
            "--atol",
            args.atol,
            "--gpu-ids",
            args.gpu_ids,
            "--local-work-root",
            local_work_root,
            "--runtime-dir",
            runtime_dir,
            "--compile-cache-root",
            cache_root,
            "--artifact-side",
            artifact_side,
        ]
        if args.dry_run:
            run_command(cmd, dry_run=True, env=env)
            return
        run_comprehensive_eval(
            argparse.Namespace(
                dataset_name="",
                hip_code_dir=str(hip_dir),
                pytorch_func_dir=str(dataset_root / "pytorch_code_functional"),
                pytorch_modu_dir=str(dataset_root / "pytorch_code_module"),
                output_dir=str(output_dir),
                max_workers=args.eval_workers,
                perf_iterations=args.perf_iterations,
                rtol=args.rtol,
                atol=args.atol,
                gpu_ids=args.gpu_ids,
                fix_script="",
                skip_fix=True,
                skip_clear_cache=True,
                cleanup_input_dir=False,
                local_work_root=str(local_work_root),
                runtime_dir=str(runtime_dir),
                compile_cache_root=str(cache_root),
                clear_compile_cache=False,
                disable_compile_cache=False,
                artifact_side=artifact_side,
                reuse_json="",
                reuse_hip_code_dir="",
                eval_backend="server-inprocess",
                reference_hip_code_dir="",
                reference_cache_mode="golden+compile",
            )
        )

    def merge_results(origin_dir: Path, optimized_dir: Path, dataset_root: Path, output_dir: Path) -> None:
        cmd = [
            sys.executable,
            "-m",
            "HIP_benchmark_kit.eval.merge_origin_optimized_eval",
            "--origin-json",
            origin_dir / "baseline_hip_results.json",
            "--origin-csv",
            origin_dir / "baseline_hip_results.csv",
            "--optimized-json",
            optimized_dir / "baseline_hip_results.json",
            "--optimized-csv",
            optimized_dir / "baseline_hip_results.csv",
            "--pytorch-func-dir",
            dataset_root / "pytorch_code_functional",
            "--pytorch-modu-dir",
            dataset_root / "pytorch_code_module",
            "--output-dir",
            output_dir,
        ]
        if args.dry_run:
            run_command(cmd, dry_run=True, env=env)
            return
        merge_results_api(
            [
                "--origin-json",
                str(origin_dir / "baseline_hip_results.json"),
                "--origin-csv",
                str(origin_dir / "baseline_hip_results.csv"),
                "--optimized-json",
                str(optimized_dir / "baseline_hip_results.json"),
                "--optimized-csv",
                str(optimized_dir / "baseline_hip_results.csv"),
                "--pytorch-func-dir",
                str(dataset_root / "pytorch_code_functional"),
                "--pytorch-modu-dir",
                str(dataset_root / "pytorch_code_module"),
                "--output-dir",
                str(output_dir),
            ]
        )

    for level in DEFAULT_LEVELS:
        dataset_root = subset_root / level
        require_dir(dataset_root / "hip_code")
        print(f"=== Re-evaluating fixed origin {level} ===")
        run_eval(dataset_root / "hip_code", dataset_root, fixed_origin_root / level, runtime_root / "fixed_origin" / level, baseline_cache_root, "origin")

    cmd = report_cmd("origin-clean-manifest", "--origin-root", fixed_origin_root, "--output", origin_clean_manifest)
    for level in DEFAULT_LEVELS:
        cmd += ["--level", level]
    run_command(cmd, dry_run=args.dry_run, env=env)

    for rollout_n in rollouts:
        for level in DEFAULT_LEVELS:
            dataset_root = subset_root / level
            generated_dir = source_root / f"rollout_n_{rollout_n}" / level / "generated"
            run_root = reeval_root / f"rollout_n_{rollout_n}" / level
            origin_dir = run_root / "origin_eval"
            optimized_dir = run_root / "optimized_eval"
            comparison_dir = run_root / "comparison"
            require_dir(generated_dir)
            require_dir(dataset_root / "hip_code")
            print(f"=== Re-evaluating rollout_n={rollout_n} {level}: cached origin perf ===")
            run_eval(dataset_root / "hip_code", dataset_root, origin_dir, runtime_root / f"rollout_n_{rollout_n}" / level / "origin", baseline_cache_root, "origin")
            print(f"=== Re-evaluating rollout_n={rollout_n} {level}: optimized ===")
            run_eval(generated_dir, dataset_root, optimized_dir, runtime_root / f"rollout_n_{rollout_n}" / level / "optimized", optimized_cache_root, "optimized")
            print(f"=== Merging rollout_n={rollout_n} {level} ===")
            merge_results(origin_dir, optimized_dir, dataset_root, comparison_dir)

    summary_cmd = report_cmd("summarize-reeval", "--reeval-root", reeval_root, "--origin-clean-manifest", origin_clean_manifest)
    for rollout_n in rollouts:
        summary_cmd += ["--rollout", rollout_n]
    for level in DEFAULT_LEVELS:
        summary_cmd += ["--level", level]
    run_command(summary_cmd, dry_run=args.dry_run, env=env)
    diagnose_cmd = report_cmd("diagnose-reeval", "--reeval-root", reeval_root)
    for rollout_n in rollouts:
        diagnose_cmd += ["--rollout", rollout_n]
    for level in DEFAULT_LEVELS:
        diagnose_cmd += ["--level", level]
    run_command(diagnose_cmd, dry_run=args.dry_run, env=env)


COMMANDS = {
    "kernelbench-run": kernelbench_run,
    "launch-rollouts": launch_rollouts,
    "launch-neurlps": launch_neurlps,
    "neurlps-run": neurlps_run,
    "multiturn-profile-run": multiturn_profile_run,
    "reeval-existing": reeval_existing,
}


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        names = "\n  ".join(sorted(COMMANDS))
        print(f"Usage: python -m HIP_benchmark_kit.orchestration <command> [options]\n\nCommands:\n  {names}")
        return
    command = argv.pop(0)
    if command not in COMMANDS:
        raise SystemExit(f"Unknown orchestration command: {command}")
    COMMANDS[command](argv)


if __name__ == "__main__":
    main()
