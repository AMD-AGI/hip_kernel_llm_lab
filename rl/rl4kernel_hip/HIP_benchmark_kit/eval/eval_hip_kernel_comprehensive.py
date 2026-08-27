# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    from .eval_reuse_identity import annotate_eval_result, build_eval_identity, load_reusable_results
    from .server_eval_adapter import (
        SERVER_INPROCESS_BACKEND,
        SUPPORTED_REFERENCE_CACHE_MODES,
        evaluate_hip_files_with_server,
        normalize_eval_backend,
        parse_hip_filename,
        validate_legacy_eval_records,
    )
except ImportError:
    from eval_reuse_identity import annotate_eval_result, build_eval_identity, load_reusable_results
    from server_eval_adapter import (
        SERVER_INPROCESS_BACKEND,
        SUPPORTED_REFERENCE_CACHE_MODES,
        evaluate_hip_files_with_server,
        normalize_eval_backend,
        parse_hip_filename,
        validate_legacy_eval_records,
    )


LOCAL_WORK_ROOT_ENV = "HIP_EVAL_LOCAL_ROOT"
LOCAL_WORK_ROOT_CANDIDATES = ("/dev/shm", "/tmp")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIP kernel evaluation via server-inprocess sandbox_core.")
    parser.add_argument("--hip_code_dir", required=True, help="Directory containing candidate HIP code.")
    parser.add_argument("--pytorch_func_dir", required=True, help="Directory containing PyTorch functional code.")
    parser.add_argument("--pytorch_modu_dir", required=True, help="Directory containing PyTorch module code.")
    parser.add_argument("--output_json", default="eval_results.json", help="Path to save evaluation results JSON.")
    parser.add_argument("--output_csv", default="eval_results.csv", help="Path to save evaluation results CSV.")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum runtime GPU workers.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for result matching.")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for result matching.")
    parser.add_argument("--perf_iterations", type=int, default=100, help="Performance iterations.")
    parser.add_argument("--gpu_ids", default=None, help="Comma-separated physical GPU IDs to use.")
    parser.add_argument("--error_log_dir", default=None, help="Directory to save sandbox error logs.")
    parser.add_argument("--cleanup-input-dir", action="store_true", help="Remove generated _hip artifacts.")
    parser.add_argument("--local-work-root", default=None, help="Preferred local root for runtime/cache data.")
    parser.add_argument("--runtime-dir", default=None, help="Runtime directory for sandbox temporary data.")
    parser.add_argument(
        "--compile-cache-root",
        default=None,
        help="Reference cache root passed to hip_kernel_evaluation_server.",
    )
    parser.add_argument("--clear-compile-cache", action="store_true", help="Clear the reference cache root first.")
    parser.add_argument("--disable-compile-cache", action="store_true", help="Disable reference compile cache.")
    parser.add_argument("--artifact-side", default="single", help="Logical side label: single/origin/optimized.")
    parser.add_argument("--reuse-json", default=None, help="Prior baseline_hip_results.json to reuse.")
    parser.add_argument("--reuse-hip-code-dir", default=None, help="HIP directory corresponding to --reuse-json.")
    parser.add_argument(
        "--eval-backend",
        choices=("server-inprocess", "sandbox-inprocess"),
        default=SERVER_INPROCESS_BACKEND,
        help="Evaluation backend. sandbox-inprocess is a compatibility alias.",
    )
    parser.add_argument(
        "--reference_hip_code_dir",
        "--reference-hip-code-dir",
        dest="reference_hip_code_dir",
        default=None,
        help="Reference HIP directory. Defaults to --hip_code_dir.",
    )
    parser.add_argument(
        "--reference-cache-mode",
        choices=tuple(sorted(SUPPORTED_REFERENCE_CACHE_MODES)),
        default="golden+compile",
        help="Reference cache mode for sandbox_core.",
    )
    return parser.parse_args(argv)


def _ensure_writable_directory(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return False
    return os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK)


def resolve_local_work_root(local_work_root: Optional[str] = None) -> str:
    candidate_roots: List[str] = []
    if local_work_root:
        candidate_roots.append(local_work_root)
    env_root = os.environ.get(LOCAL_WORK_ROOT_ENV)
    if env_root:
        candidate_roots.append(env_root)
    candidate_roots.extend(LOCAL_WORK_ROOT_CANDIDATES)

    seen = set()
    checked = []
    for candidate in candidate_roots:
        expanded = os.path.abspath(os.path.expanduser(candidate))
        if expanded in seen:
            continue
        seen.add(expanded)
        checked.append(expanded)
        if _ensure_writable_directory(expanded):
            return expanded

    checked_msg = ", ".join(checked) if checked else "<none>"
    raise RuntimeError(
        f"Unable to resolve writable local work root. Checked: {checked_msg}. "
        f"Set --local-work-root or {LOCAL_WORK_ROOT_ENV}."
    )


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def build_default_hot_paths(*, output_dir: str, local_work_root: str) -> Dict[str, str]:
    output_fingerprint = _sha256_text(os.path.abspath(output_dir))[:12]
    run_token = f"{int(time.time())}_{os.getpid()}"
    root_base = os.path.join(local_work_root, "hip_eval_hot")
    return {
        "runtime_dir": os.path.join(root_base, "runtime", f"{output_fingerprint}_{run_token}"),
        "reference_cache_root": os.path.join(root_base, "reference_cache", output_fingerprint),
    }


def get_free_amd_gpus() -> List[int]:
    """Best-effort ROCm free-GPU discovery."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmemuse"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except Exception:
        return []

    gpu_ids: List[int] = []
    for line in result.stdout.splitlines():
        match = re.search(r"GPU\[(\d+)\].*VRAM%\):\s*(\d+)", line)
        if match and int(match.group(2)) < 1:
            gpu_ids.append(int(match.group(1)))
    return gpu_ids


def cleanup_hip_intermediate_files(hip_code_dir: str) -> None:
    if not os.path.isdir(hip_code_dir):
        return
    cleaned_count = 0
    for filename in os.listdir(hip_code_dir):
        if filename.endswith(".hip") and (filename.endswith("_hip.hip") or "_hip_" in filename):
            try:
                os.remove(os.path.join(hip_code_dir, filename))
                cleaned_count += 1
            except OSError as exc:
                print(f"Warning: failed to remove {filename}: {exc}")
    if cleaned_count:
        print(f"Cleaned up {cleaned_count} intermediate _hip files from {hip_code_dir}")


def discover_hip_files(hip_code_dir: str) -> List[str]:
    hip_files = sorted(
        filename
        for filename in os.listdir(hip_code_dir)
        if filename.endswith(".hip") and "_hip_" not in filename and not filename.endswith("_hip.hip")
    )
    if not hip_files:
        raise SystemExit(f"No HIP files found in: {hip_code_dir}")
    return hip_files


def parse_gpu_ids(raw: Optional[str]) -> List[Optional[int]]:
    if raw:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    detected = get_free_amd_gpus()
    if detected:
        print(f"Detected free AMD GPU(s): {detected}")
        return detected
    print("Warning: no free AMD GPU detected, using default visible GPU")
    return [None]


def annotate_rows(
    *,
    rows: Iterable[Dict[str, object]],
    hip_code_dir: str,
    pytorch_func_dir: str,
    pytorch_modu_dir: str,
    rtol: float,
    atol: float,
    perf_iterations: int,
    artifact_side: str,
    eval_backend: str,
    reference_hip_code_dir: str,
    reference_cache_mode: str,
) -> List[Dict[str, object]]:
    annotated: List[Dict[str, object]] = []
    for row in rows:
        identity = build_eval_identity(
            hip_code_dir=hip_code_dir,
            hip_file=str(row["hip_file"]),
            pytorch_func_dir=pytorch_func_dir,
            pytorch_modu_dir=pytorch_modu_dir,
            rtol=rtol,
            atol=atol,
            perf_iterations=perf_iterations,
            artifact_side=artifact_side,
            eval_backend=eval_backend,
            reference_hip_code_dir=reference_hip_code_dir,
            reference_cache_mode=reference_cache_mode,
        )
        annotated.append(annotate_eval_result(row, identity))
    return annotated


def print_summary_table(results: List[Dict[str, object]]) -> None:
    df = pd.DataFrame(results)
    total = len(df)
    compile_ok = int(df["compile_ok"].sum())
    run_ok = int(df["run_ok"].sum())
    match_ok = int(df["match_ok"].sum())

    print("\n" + "=" * 100)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 100)
    print(f"\nTotal HIP kernels evaluated: {total}")
    print(f"Unique base kernels:         {df['base_name'].nunique()}")
    print("\n" + "-" * 100)
    print("Statistics")
    print("-" * 100)
    print(f"Compile success:             {compile_ok} ({compile_ok / total * 100:.1f}%)")
    print(f"Execution success:           {run_ok} ({run_ok / total * 100:.1f}%)")
    print(f"Result match success:        {match_ok} ({match_ok / total * 100:.1f}%)")

    matched_df = df[df["match_ok"] == True]
    speedup_df = matched_df[matched_df["speedup"].notna()]
    if not speedup_df.empty:
        print("\nSpeedup statistics (matched kernels):")
        print(f"  Average speedup:  {speedup_df['speedup'].mean():.3f}x")
        print(f"  Median speedup:   {speedup_df['speedup'].median():.3f}x")
        print(f"  Max speedup:      {speedup_df['speedup'].max():.3f}x")
        print(f"  Min speedup:      {speedup_df['speedup'].min():.3f}x")

    print("\n" + "=" * 100)
    print("DETAILED RESULTS")
    print("=" * 100)
    display_cols = [
        "base_name",
        "gen_idx",
        "compile_ok",
        "run_ok",
        "match_ok",
        "pytorch_time_ms",
        "hip_time_ms",
        "speedup",
    ]
    display_df = df[[col for col in display_cols if col in df.columns]]
    display_df = display_df.sort_values(["base_name", "gen_idx"], na_position="first")
    print(display_df.to_string(index=False))
    print("\n")


def main(argv: List[str] | None = None) -> None:
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args(argv)
    args.eval_backend = normalize_eval_backend(args.eval_backend)
    if args.eval_backend != SERVER_INPROCESS_BACKEND:
        raise SystemExit(f"Unsupported eval backend: {args.eval_backend}")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if args.gpu_ids:
        print(f"Using specified GPU IDs: {gpu_ids}")

    output_dir = os.path.abspath(os.path.dirname(args.output_json) or ".")
    error_log_dir = os.path.abspath(args.error_log_dir or os.path.join(output_dir, "error_logs"))
    local_work_root = resolve_local_work_root(args.local_work_root)
    default_hot_paths = build_default_hot_paths(output_dir=output_dir, local_work_root=local_work_root)
    runtime_dir = os.path.abspath(args.runtime_dir or default_hot_paths["runtime_dir"])
    reference_cache_root = os.path.abspath(args.compile_cache_root or default_hot_paths["reference_cache_root"])
    reference_hip_code_dir = os.path.abspath(args.reference_hip_code_dir or args.hip_code_dir)

    if args.clear_compile_cache:
        shutil.rmtree(reference_cache_root, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(error_log_dir, exist_ok=True)
    os.makedirs(runtime_dir, exist_ok=True)
    os.makedirs(reference_cache_root, exist_ok=True)

    hip_file_list = discover_hip_files(args.hip_code_dir)
    num_tasks = len(hip_file_list)
    num_gpus = max(1, len([gpu for gpu in gpu_ids if gpu is not None]))
    cpu_count = multiprocessing.cpu_count()
    max_workers = args.max_workers or max(1, min(num_gpus, num_tasks, max(1, cpu_count // 2), 8))

    print("\n" + "=" * 100)
    print("HIP KERNEL EVALUATION (server-inprocess)")
    print("=" * 100)
    print(f"Total HIP files:             {num_tasks}")
    print(f"Parallel runtime workers:    {max_workers}")
    print(f"Available GPUs:              {gpu_ids}")
    print(f"HIP code directory:          {args.hip_code_dir}")
    print(f"Reference HIP directory:     {reference_hip_code_dir}")
    print(f"PyTorch func directory:      {args.pytorch_func_dir}")
    print(f"PyTorch module directory:    {args.pytorch_modu_dir}")
    print(f"Output JSON:                 {args.output_json}")
    print(f"Output CSV:                  {args.output_csv}")
    print(f"Error log directory:         {error_log_dir}")
    print(f"Runtime directory:           {runtime_dir}")
    print(f"Reference cache root:        {reference_cache_root}")
    print(f"Reference cache mode:        {args.reference_cache_mode}")
    print(f"Reference compile cache:     {not args.disable_compile_cache}")
    print(f"Artifact side:               {args.artifact_side}")
    print(f"Tolerance:                   rtol={args.rtol}, atol={args.atol}")
    print(f"Performance iterations:      {args.perf_iterations}")
    print("=" * 100)

    reusable_results = load_reusable_results(
        reuse_json=args.reuse_json,
        reuse_hip_code_dir=args.reuse_hip_code_dir,
        current_hip_code_dir=args.hip_code_dir,
        pytorch_func_dir=args.pytorch_func_dir,
        pytorch_modu_dir=args.pytorch_modu_dir,
        rtol=args.rtol,
        atol=args.atol,
        perf_iterations=args.perf_iterations,
        artifact_side=args.artifact_side,
        eval_backend=args.eval_backend,
        reference_hip_code_dir=reference_hip_code_dir,
        reference_cache_mode=args.reference_cache_mode,
    )

    fresh_hip_files: List[str] = []
    results: List[Dict[str, object]] = []
    for hip_file in hip_file_list:
        base_name, gen_idx = parse_hip_filename(hip_file)
        reused = reusable_results.get((base_name, gen_idx))
        if reused is not None:
            results.append(reused)
        else:
            fresh_hip_files.append(hip_file)

    start_time = time.time()
    print(f"\nStarting evaluation: fresh={len(fresh_hip_files)} reused={len(results)}\n")
    if fresh_hip_files:
        fresh_rows = evaluate_hip_files_with_server(
            hip_file_list=fresh_hip_files,
            hip_code_dir=args.hip_code_dir,
            reference_hip_code_dir=reference_hip_code_dir,
            pytorch_func_dir=args.pytorch_func_dir,
            pytorch_modu_dir=args.pytorch_modu_dir,
            rtol=args.rtol,
            atol=args.atol,
            perf_iterations=args.perf_iterations,
            gpu_ids=gpu_ids,
            error_log_dir=error_log_dir,
            runtime_dir=runtime_dir,
            max_workers=max_workers,
            artifact_side=args.artifact_side,
            cache_root=reference_cache_root,
            reference_cache_mode=args.reference_cache_mode,
            disable_compile_cache=args.disable_compile_cache,
            eval_backend=args.eval_backend,
        )
        results.extend(
            annotate_rows(
                rows=fresh_rows,
                hip_code_dir=args.hip_code_dir,
                pytorch_func_dir=args.pytorch_func_dir,
                pytorch_modu_dir=args.pytorch_modu_dir,
                rtol=args.rtol,
                atol=args.atol,
                perf_iterations=args.perf_iterations,
                artifact_side=args.artifact_side,
                eval_backend=args.eval_backend,
                reference_hip_code_dir=reference_hip_code_dir,
                reference_cache_mode=args.reference_cache_mode,
            )
        )

    total_elapsed = time.time() - start_time
    success_count = sum(1 for row in results if row["compile_ok"] and row["run_ok"] and row["match_ok"])
    failed_count = len(results) - success_count
    results.sort(key=lambda row: (row["base_name"], row["gen_idx"] if row["gen_idx"] is not None else -1))
    validate_legacy_eval_records(results)

    print("\n" + "=" * 100)
    print("EXECUTION SUMMARY")
    print("=" * 100)
    print(f"Successful: {success_count}/{num_tasks} ({success_count / num_tasks * 100:.1f}%)")
    print(f"Failed:     {failed_count}/{num_tasks} ({failed_count / num_tasks * 100:.1f}%)")
    print(f"Total time: {total_elapsed:.2f}s")
    print("=" * 100)

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON results to: {args.output_json}")

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"Saved CSV results to: {args.output_csv}")

    if args.cleanup_input_dir:
        cleanup_hip_intermediate_files(args.hip_code_dir)

    if args.runtime_dir is None:
        shutil.rmtree(runtime_dir, ignore_errors=True)

    print_summary_table(results)


if __name__ == "__main__":
    main()
