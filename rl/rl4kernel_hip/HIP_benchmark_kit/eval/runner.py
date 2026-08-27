# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Python runners for server-backed HIP evaluation workflows."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from HIP_benchmark_kit.contracts.layout import repo_root

KIT_ROOT = repo_root() / "HIP_benchmark_kit"


def default_gpu_ids() -> str:
    return os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1,2,3,4,5,6,7"


def run_compiler_fix(fix_script: str, *, skip_fix: bool) -> None:
    if skip_fix:
        return
    if not fix_script:
        print("Skipping HIP compiler fix preflight; no fix script configured.")
        return
    path = Path(fix_script)
    if not path.is_file():
        raise SystemExit(f"Configured HIP compiler fix script not found: {path}")
    print("=" * 74)
    print("Running HIP compiler fix preflight")
    print("=" * 74)
    subprocess.run(["sudo", "bash", str(path)], check=True)


def clear_torch_extension_cache(enabled: bool) -> None:
    if enabled:
        torch_cache = Path.home() / ".cache" / "torch_extensions"
        print(f"Clearing torch extension cache: {torch_cache}")
        shutil.rmtree(torch_cache, ignore_errors=True)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        if not path.is_dir():
            raise SystemExit(f"Required directory not found: {path}")


def comprehensive_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-name", default=os.environ.get("DATASET_NAME", "hip_eval_dataset_kernelbench_25_tasks"))
    parser.add_argument("--hip_code_dir", "--hip-code-dir", dest="hip_code_dir", default="")
    parser.add_argument("--pytorch_func_dir", "--pytorch-func-dir", dest="pytorch_func_dir", default="")
    parser.add_argument("--pytorch_modu_dir", "--pytorch-modu-dir", dest="pytorch_modu_dir", default="")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="")
    parser.add_argument("--max-workers", default="8")
    parser.add_argument("--perf-iterations", default="10")
    parser.add_argument("--rtol", default="1e-4")
    parser.add_argument("--atol", default="1e-5")
    parser.add_argument("--gpu-ids", default=default_gpu_ids())
    parser.add_argument("--fix-script", default=os.environ.get("HIP_COMPILER_FIX_SCRIPT", ""))
    parser.add_argument("--skip-fix", action="store_true")
    parser.add_argument("--skip-clear-cache", action="store_true")
    parser.add_argument("--cleanup-input-dir", action="store_true")
    parser.add_argument("--local-work-root", default=os.environ.get("HIP_EVAL_LOCAL_ROOT", ""))
    parser.add_argument("--runtime-dir", default="")
    parser.add_argument("--compile-cache-root", default="")
    parser.add_argument("--clear-compile-cache", action="store_true")
    parser.add_argument("--disable-compile-cache", action="store_true")
    parser.add_argument("--artifact-side", default="single")
    parser.add_argument("--reuse-json", default="")
    parser.add_argument("--reuse-hip-code-dir", default="")
    parser.add_argument("--eval-backend", default="server-inprocess")
    parser.add_argument("--reference-hip-code-dir", "--reference_hip_code_dir", dest="reference_hip_code_dir", default="")
    parser.add_argument("--reference-cache-mode", default=os.environ.get("HIP_REFERENCE_CACHE_MODE", "golden+compile"))


def default_dataset_paths(dataset_name: str) -> tuple[Path, Path, Path, Path]:
    benchmark_data = KIT_ROOT / "data" / dataset_name
    return (
        benchmark_data / "hip_code",
        benchmark_data / "pytorch_code_functional",
        benchmark_data / "pytorch_code_module",
        repo_root() / "outputs" / "HIP_benchmark_kit" / "eval" / dataset_name / "baseline",
    )


def run_comprehensive_eval(args: argparse.Namespace) -> None:
    default_hip, default_func, default_modu, default_output = default_dataset_paths(args.dataset_name)
    hip_code_dir = Path(args.hip_code_dir or default_hip)
    pytorch_func_dir = Path(args.pytorch_func_dir or default_func)
    pytorch_modu_dir = Path(args.pytorch_modu_dir or default_modu)
    output_dir = Path(args.output_dir or default_output)
    ensure_dirs(hip_code_dir, pytorch_func_dir, pytorch_modu_dir)
    run_compiler_fix(args.fix_script, skip_fix=args.skip_fix)
    clear_torch_extension_cache(not args.skip_clear_cache)

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_argv = [
        "--hip_code_dir",
        str(hip_code_dir),
        "--pytorch_func_dir",
        str(pytorch_func_dir),
        "--pytorch_modu_dir",
        str(pytorch_modu_dir),
        "--output_json",
        str(output_dir / "baseline_hip_results.json"),
        "--output_csv",
        str(output_dir / "baseline_hip_results.csv"),
        "--max_workers",
        str(args.max_workers),
        "--perf_iterations",
        str(args.perf_iterations),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
        "--gpu_ids",
        str(args.gpu_ids),
        "--artifact-side",
        args.artifact_side,
        "--eval-backend",
        args.eval_backend,
        "--reference-cache-mode",
        args.reference_cache_mode,
    ]
    optional_pairs = (
        ("--local-work-root", args.local_work_root),
        ("--runtime-dir", args.runtime_dir),
        ("--compile-cache-root", args.compile_cache_root),
        ("--reuse-json", args.reuse_json),
        ("--reuse-hip-code-dir", args.reuse_hip_code_dir),
        ("--reference-hip-code-dir", args.reference_hip_code_dir),
    )
    for flag, value in optional_pairs:
        if value:
            eval_argv.extend([flag, str(value)])
    if args.cleanup_input_dir:
        eval_argv.append("--cleanup-input-dir")
    if args.clear_compile_cache:
        eval_argv.append("--clear-compile-cache")
    if args.disable_compile_cache:
        eval_argv.append("--disable-compile-cache")

    from HIP_benchmark_kit.eval.eval_hip_kernel_comprehensive import main as eval_main

    eval_main(eval_argv)


def copy_canonical_hip_tree(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.is_dir():
        raise SystemExit(f"HIP directory not found: {src_dir}")
    shutil.rmtree(dst_dir, ignore_errors=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    include_src = src_dir / "include"
    if include_src.is_dir():
        shutil.copytree(include_src, dst_dir / "include", dirs_exist_ok=True)
    copied_count = 0
    for hip_path in sorted(src_dir.glob("*.hip")):
        hip_name = hip_path.name
        if hip_name.endswith("_hip.hip") or "_hip_" in hip_name:
            continue
        shutil.copy2(hip_path, dst_dir / hip_name)
        copied_count += 1
    if copied_count == 0:
        raise SystemExit(f"No canonical HIP files found in: {src_dir}")
    print(f"Staged {copied_count} canonical HIP files: {src_dir} -> {dst_dir}")
    return copied_count


def resolve_local_work_root(preferred_root: str) -> Path:
    candidates = [Path(preferred_root)] if preferred_root else [Path("/dev/shm"), Path("/tmp")]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(candidate, os.W_OK | os.X_OK):
            return candidate.resolve()
    raise SystemExit("Unable to find writable local work root.")


def compare_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--origin_hip_dir", "--origin-hip-dir", dest="origin_hip_dir", required=True)
    parser.add_argument("--optimized_hip_dir", "--optimized-hip-dir", dest="optimized_hip_dir", required=True)
    parser.add_argument("--pytorch_func_dir", "--pytorch-func-dir", dest="pytorch_func_dir", required=True)
    parser.add_argument("--pytorch_modu_dir", "--pytorch-modu-dir", dest="pytorch_modu_dir", required=True)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default=str(repo_root() / "outputs" / "HIP_benchmark_kit" / "eval" / "compare_hip_dirs"))
    parser.add_argument("--max-workers", default="8")
    parser.add_argument("--perf-iterations", default="10")
    parser.add_argument("--rtol", default="1e-4")
    parser.add_argument("--atol", default="1e-5")
    parser.add_argument("--gpu-ids", default=default_gpu_ids())
    parser.add_argument("--fix-script", default=os.environ.get("HIP_COMPILER_FIX_SCRIPT", ""))
    parser.add_argument("--skip-fix", action="store_true")
    parser.add_argument("--skip-clear-cache", action="store_true")
    parser.add_argument("--local-work-root", default=os.environ.get("HIP_EVAL_LOCAL_ROOT", ""))
    parser.add_argument("--runtime-root", default="")
    parser.add_argument("--compile-cache-root", default="")
    parser.add_argument("--clear-compile-cache", action="store_true")
    parser.add_argument("--reuse-origin-json", default="")
    parser.add_argument("--reuse-origin-hip-dir", default="")
    parser.add_argument("--reuse-optimized-json", default="")
    parser.add_argument("--reuse-optimized-hip-dir", default="")
    parser.add_argument("--eval-backend", default="server-inprocess")
    parser.add_argument("--reference-hip-dir", "--reference_hip_dir", dest="reference_hip_dir", default="")
    parser.add_argument("--reference-cache-mode", default=os.environ.get("HIP_REFERENCE_CACHE_MODE", "golden+compile"))


def eval_side(
    *,
    side: str,
    hip_dir: Path,
    reference_hip_dir: Path,
    args: argparse.Namespace,
    output_dir: Path,
    runtime_dir: Path,
    cache_root: Path,
    reuse_json: str,
    reuse_hip_dir: str,
) -> None:
    eval_args = argparse.Namespace(
        dataset_name="",
        hip_code_dir=str(hip_dir),
        pytorch_func_dir=str(Path(args.pytorch_func_dir)),
        pytorch_modu_dir=str(Path(args.pytorch_modu_dir)),
        output_dir=str(output_dir),
        max_workers=args.max_workers,
        perf_iterations=args.perf_iterations,
        rtol=args.rtol,
        atol=args.atol,
        gpu_ids=args.gpu_ids,
        fix_script="",
        skip_fix=True,
        skip_clear_cache=True,
        cleanup_input_dir=False,
        local_work_root=args.local_work_root,
        runtime_dir=str(runtime_dir),
        compile_cache_root=str(cache_root),
        clear_compile_cache=False,
        disable_compile_cache=False,
        artifact_side=side,
        reuse_json=reuse_json,
        reuse_hip_code_dir=reuse_hip_dir,
        eval_backend=args.eval_backend,
        reference_hip_code_dir=str(reference_hip_dir),
        reference_cache_mode=args.reference_cache_mode,
    )
    run_comprehensive_eval(eval_args)


def compare_hip_dirs(args: argparse.Namespace) -> None:
    origin_hip_dir = Path(args.origin_hip_dir)
    optimized_hip_dir = Path(args.optimized_hip_dir)
    pytorch_func_dir = Path(args.pytorch_func_dir)
    pytorch_modu_dir = Path(args.pytorch_modu_dir)
    ensure_dirs(origin_hip_dir, optimized_hip_dir, pytorch_func_dir, pytorch_modu_dir)
    run_compiler_fix(args.fix_script, skip_fix=args.skip_fix)
    clear_torch_extension_cache(not args.skip_clear_cache)

    output_dir = Path(args.output_dir)
    local_work_root = resolve_local_work_root(args.local_work_root)
    args.local_work_root = str(local_work_root)
    runtime_root = Path(args.runtime_root) if args.runtime_root else output_dir / "runtime"
    cache_root = Path(args.compile_cache_root) if args.compile_cache_root else output_dir / "reference_cache"
    if args.clear_compile_cache:
        shutil.rmtree(cache_root, ignore_errors=True)
    staging_root = output_dir / "staging"
    origin_staging = staging_root / "origin"
    optimized_staging = staging_root / "optimized"
    origin_eval_dir = output_dir / "origin_eval"
    optimized_eval_dir = output_dir / "optimized_eval"
    comparison_dir = output_dir / "comparison"
    for path in (runtime_root, cache_root, origin_eval_dir, optimized_eval_dir, comparison_dir):
        path.mkdir(parents=True, exist_ok=True)

    copy_canonical_hip_tree(origin_hip_dir, origin_staging)
    copy_canonical_hip_tree(optimized_hip_dir, optimized_staging)
    reference_hip_dir = Path(args.reference_hip_dir) if args.reference_hip_dir else origin_staging

    eval_side(
        side="origin",
        hip_dir=origin_staging,
        reference_hip_dir=reference_hip_dir,
        args=args,
        output_dir=origin_eval_dir,
        runtime_dir=runtime_root / "origin",
        cache_root=cache_root,
        reuse_json=args.reuse_origin_json,
        reuse_hip_dir=args.reuse_origin_hip_dir,
    )
    eval_side(
        side="optimized",
        hip_dir=optimized_staging,
        reference_hip_dir=reference_hip_dir,
        args=args,
        output_dir=optimized_eval_dir,
        runtime_dir=runtime_root / "optimized",
        cache_root=cache_root,
        reuse_json=args.reuse_optimized_json,
        reuse_hip_dir=args.reuse_optimized_hip_dir,
    )

    from HIP_benchmark_kit.eval.merge_origin_optimized_eval import main as merge_main

    merge_main(
        [
            "--origin-json",
            str(origin_eval_dir / "baseline_hip_results.json"),
            "--origin-csv",
            str(origin_eval_dir / "baseline_hip_results.csv"),
            "--optimized-json",
            str(optimized_eval_dir / "baseline_hip_results.json"),
            "--optimized-csv",
            str(optimized_eval_dir / "baseline_hip_results.csv"),
            "--pytorch-func-dir",
            str(pytorch_func_dir),
            "--pytorch-modu-dir",
            str(pytorch_modu_dir),
            "--output-dir",
            str(comparison_dir),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Server-backed HIP eval runners.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    comprehensive = subparsers.add_parser("comprehensive")
    comprehensive_parser(comprehensive)
    comprehensive.set_defaults(func=run_comprehensive_eval)
    compare = subparsers.add_parser("compare")
    compare_parser(compare)
    compare.set_defaults(func=compare_hip_dirs)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
