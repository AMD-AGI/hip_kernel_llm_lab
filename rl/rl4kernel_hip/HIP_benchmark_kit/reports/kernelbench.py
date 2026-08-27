#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reusable staging and reporting commands for KernelBench HIP runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
import statistics
import hashlib
from pathlib import Path
from typing import Any


def parse_level_spec(spec: str) -> tuple[str, int]:
    level, count_text = spec.split(":", 1)
    count = int(count_text)
    if count <= 0:
        raise SystemExit(f"Level quota must be positive: {spec}")
    return level, count


def stable_task_key(name: str) -> tuple[int, int | str, str]:
    match = re.match(r"^(\d+)(?:_|$)", name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, name, name)


def file_map(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise SystemExit(f"Required source directory not found: {directory}")
    return {path.stem: path for path in directory.glob(f"*{suffix}") if path.is_file()}


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def copy_unique_file(source: Path, dest: Path) -> None:
    if dest.exists():
        if sha256_file(source) == sha256_file(dest):
            return
        raise SystemExit(f"Refusing to overwrite different file in flat subset: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def copy_include_tree(source_dir: Path, dest_dir: Path) -> None:
    if not source_dir.is_dir():
        return
    for source in source_dir.rglob("*"):
        if not source.is_file():
            continue
        copy_unique_file(source, dest_dir / source.relative_to(source_dir))


def select_level_records(source_root: Path, level: str, quota: int) -> tuple[list[str], dict[str, Path], dict[str, Path], dict[str, Path]]:
    level_root = source_root / level
    hip_files = file_map(level_root / "hip_code", ".hip")
    func_files = file_map(level_root / "pytorch_code_functional", ".py")
    modu_files = file_map(level_root / "pytorch_code_module", ".py")
    common_names = sorted(set(hip_files) & set(func_files) & set(modu_files), key=stable_task_key)

    if len(common_names) < quota:
        raise SystemExit(
            f"{level} has only {len(common_names)} files present in all three trees; "
            f"quota={quota}"
        )
    return common_names[:quota], hip_files, func_files, modu_files


def stage_subset(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    subset_root = args.subset_root.resolve()
    manifest_path = args.manifest.resolve()

    if not source_root.is_dir():
        raise SystemExit(f"kernelbench_hip root not found: {source_root}")

    if subset_root.exists():
        shutil.rmtree(subset_root)
    subset_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source_root": str(source_root),
        "subset_root": str(subset_root),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_strategy": "stable_filename_sort_on_hip_functional_module_intersection",
        "levels": {},
    }

    total_selected = 0
    for level, quota in map(parse_level_spec, args.level):
        level_root = source_root / level
        hip_dir = level_root / "hip_code"
        selected_names, hip_files, func_files, modu_files = select_level_records(source_root, level, quota)
        dst_level_root = subset_root / level
        dst_hip_dir = dst_level_root / "hip_code"
        dst_func_dir = dst_level_root / "pytorch_code_functional"
        dst_modu_dir = dst_level_root / "pytorch_code_module"
        dst_hip_dir.mkdir(parents=True, exist_ok=True)
        dst_func_dir.mkdir(parents=True, exist_ok=True)
        dst_modu_dir.mkdir(parents=True, exist_ok=True)

        include_src = hip_dir / "include"
        if include_src.is_dir():
            shutil.copytree(include_src, dst_hip_dir / "include", dirs_exist_ok=True)

        selected_records = []
        for name in selected_names:
            shutil.copy2(hip_files[name], dst_hip_dir / hip_files[name].name)
            shutil.copy2(func_files[name], dst_func_dir / func_files[name].name)
            shutil.copy2(modu_files[name], dst_modu_dir / modu_files[name].name)
            selected_records.append(
                {
                    "base_name": name,
                    "hip_file": hip_files[name].name,
                    "pytorch_functional_file": func_files[name].name,
                    "pytorch_module_file": modu_files[name].name,
                }
            )

        missing_from_func = sorted(set(hip_files) - set(func_files), key=stable_task_key)
        missing_from_modu = sorted(set(hip_files) - set(modu_files), key=stable_task_key)
        available_intersection = len(set(hip_files) & set(func_files) & set(modu_files))
        manifest["levels"][level] = {
            "quota": quota,
            "available_intersection": available_intersection,
            "hip_count": len(hip_files),
            "pytorch_functional_count": len(func_files),
            "pytorch_module_count": len(modu_files),
            "selected_count": len(selected_records),
            "selected_files": selected_records,
            "hip_without_functional_reference_preview": missing_from_func[:20],
            "hip_without_module_reference_preview": missing_from_modu[:20],
        }
        total_selected += len(selected_records)
        print(
            f"[SUBSET] {level}: selected={len(selected_records)} "
            f"available_intersection={available_intersection} quota={quota}"
        )

    manifest["total_selected"] = total_selected
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[SUBSET] wrote manifest: {manifest_path}")


def stage_flat_subset(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    subset_root = args.subset_root.resolve()
    manifest_path = (args.manifest or subset_root / "manifest.json").resolve()

    if not source_root.is_dir():
        raise SystemExit(f"kernelbench_hip root not found: {source_root}")
    if subset_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Flat subset already exists: {subset_root}. Pass --overwrite to replace it.")
        shutil.rmtree(subset_root)

    hip_out = subset_root / "hip_code"
    func_out = subset_root / "pytorch_code_functional"
    modu_out = subset_root / "pytorch_code_module"
    for directory in (hip_out, func_out, modu_out):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source_root": str(source_root),
        "subset_root": str(subset_root),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "layout": "flat_kernelbench_hip",
        "selection_strategy": "stable_filename_sort_on_hip_functional_module_intersection",
        "levels": {},
        "records": [],
    }

    selected_by_base: dict[str, str] = {}
    total_selected = 0
    for level, quota in map(parse_level_spec, args.level):
        selected_names, hip_files, func_files, modu_files = select_level_records(source_root, level, quota)
        copy_include_tree(source_root / level / "hip_code" / "include", hip_out / "include")

        selected_records = []
        for name in selected_names:
            if name in selected_by_base:
                raise SystemExit(f"Duplicate base name across levels: {name} ({selected_by_base[name]} and {level})")
            selected_by_base[name] = level

            hip_source = hip_files[name]
            func_source = func_files[name]
            modu_source = modu_files[name]
            copy_unique_file(hip_source, hip_out / hip_source.name)
            copy_unique_file(func_source, func_out / func_source.name)
            copy_unique_file(modu_source, modu_out / modu_source.name)

            record = {
                "level": level,
                "base_name": name,
                "hip_file": hip_source.name,
                "pytorch_functional_file": func_source.name,
                "pytorch_module_file": modu_source.name,
                "hip_sha256": sha256_file(hip_source),
                "pytorch_functional_sha256": sha256_file(func_source),
                "pytorch_module_sha256": sha256_file(modu_source),
            }
            selected_records.append(record)
            manifest["records"].append(record)

        manifest["levels"][level] = {
            "quota": quota,
            "available_intersection": len(set(hip_files) & set(func_files) & set(modu_files)),
            "selected_count": len(selected_records),
            "selected_files": selected_records,
        }
        total_selected += len(selected_records)
        print(f"[FLAT_SUBSET] {level}: selected={len(selected_records)} quota={quota}")

    manifest["total_selected"] = total_selected
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[FLAT_SUBSET] wrote subset: {subset_root}")
    print(f"[FLAT_SUBSET] wrote manifest: {manifest_path}")


def summarize_generation(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    saved_records = [record for record in records if record.get("saved")]
    parse_failures = [record for record in records if not record.get("parse_ok")]
    input_files = sorted({record.get("input_file") for record in records if record.get("input_file")})
    saved_inputs = sorted({record.get("input_file") for record in saved_records if record.get("input_file")})
    missing_inputs = [name for name in input_files if name not in set(saved_inputs)]

    print(f"[KERNELBENCH_HIP:{args.label}] manifest={args.manifest}")
    print(
        f"[KERNELBENCH_HIP:{args.label}] inputs={len(input_files)} "
        f"saved_outputs={len(saved_records)} "
        f"inputs_with_saved_output={len(saved_inputs)} "
        f"parse_failures={len(parse_failures)}"
    )
    if parse_failures:
        first = parse_failures[0]
        print(
            f"[KERNELBENCH_HIP:{args.label}] first_parse_failure="
            f"{first.get('input_file')} sample={first.get('sample_idx')} "
            f"parse_mode={first.get('parse_mode')} error={first.get('parse_error')}"
        )
    if missing_inputs:
        print(f"[KERNELBENCH_HIP:{args.label}] inputs_without_saved_output={', '.join(missing_inputs[:10])}")
    if not saved_records:
        raise SystemExit(f"No saved generation outputs found for {args.label}")


def is_true(record: dict[str, Any], key: str) -> bool:
    return record.get(key) is True


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def task_id(record: dict[str, Any]) -> str | None:
    base_name = record.get("base_name")
    if not base_name:
        return None
    return f"{record.get('_level', '')}:{base_name}"


def metrics_for(level: str, records: list[dict[str, Any]], expected_tasks: int, comparison_json: Path) -> dict[str, Any]:
    tasks = sorted({item for record in records for item in [task_id(record)] if item})
    pair_ok_records = [record for record in records if is_true(record, "pair_ok")]
    valid_speedups = [
        speedup
        for record in pair_ok_records
        for speedup in [finite_float(record.get("speedup"))]
        if speedup is not None
    ]

    task_best_speedup: dict[str, float] = {}
    for record in pair_ok_records:
        item = task_id(record)
        speedup = finite_float(record.get("speedup"))
        if not item or speedup is None:
            continue
        task_best_speedup[item] = max(speedup, task_best_speedup.get(item, float("-inf")))

    task_best_values = list(task_best_speedup.values())
    total_records = len(records)
    optimized_compile_ok = sum(is_true(record, "optimized_compile_ok") for record in records)
    optimized_run_ok = sum(is_true(record, "optimized_run_ok") for record in records)
    optimized_match_ok = sum(is_true(record, "optimized_match_ok") for record in records)
    origin_compile_ok = sum(is_true(record, "origin_compile_ok") for record in records)
    origin_match_ok = sum(is_true(record, "origin_match_ok") for record in records)

    return {
        "level": level,
        "expected_tasks": expected_tasks,
        "tasks_in_comparison": len(tasks),
        "total_records": total_records,
        "origin_compile_success_count": origin_compile_ok,
        "origin_compile_success_rate": rate(origin_compile_ok, total_records),
        "origin_correctness_success_count": origin_match_ok,
        "origin_correctness_success_rate": rate(origin_match_ok, total_records),
        "optimized_compile_success_count": optimized_compile_ok,
        "optimized_compile_success_rate": rate(optimized_compile_ok, total_records),
        "optimized_run_success_count": optimized_run_ok,
        "optimized_run_success_rate": rate(optimized_run_ok, total_records),
        "optimized_correctness_success_count": optimized_match_ok,
        "optimized_correctness_success_rate": rate(optimized_match_ok, total_records),
        "pair_ok_count": len(pair_ok_records),
        "record_pass_rate": rate(len(pair_ok_records), total_records),
        "task_pass_count": len(task_best_speedup),
        "task_pass_rate": rate(len(task_best_speedup), expected_tasks),
        "valid_speedup_count": len(valid_speedups),
        "mean_speedup": mean(valid_speedups),
        "median_speedup": median(valid_speedups),
        "best_speedup": max(valid_speedups) if valid_speedups else None,
        "mean_task_best_speedup": mean(task_best_values),
        "median_task_best_speedup": median(task_best_values),
        "best_task_speedup": max(task_best_values) if task_best_values else None,
        "comparison_json": str(comparison_json),
    }


def summarize_run(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    subset_manifest_path = args.subset_manifest.resolve()
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    subset_manifest = json.loads(subset_manifest_path.read_text(encoding="utf-8"))
    level_metrics = []
    all_records = []

    for level in args.level:
        comparison_json = run_root / level / "eval" / "comparison" / "origin_vs_optimized_results.json"
        if not comparison_json.is_file():
            raise SystemExit(f"Comparison JSON missing for {level}: {comparison_json}")

        records = json.loads(comparison_json.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise SystemExit(f"Expected list comparison JSON for {level}: {comparison_json}")

        expected_tasks = int(subset_manifest["levels"][level]["selected_count"])
        level_records = [dict(record, _level=level) for record in records]
        level_metrics.append(metrics_for(level, level_records, expected_tasks, comparison_json))
        all_records.extend(level_records)

    expected_tasks = sum(int(item["expected_tasks"]) for item in level_metrics)
    overall = metrics_for("overall", all_records, expected_tasks, Path(""))
    overall["comparison_json"] = ""
    overall["levels_included"] = [item["level"] for item in level_metrics]

    summary = {
        "run_root": str(run_root),
        "subset_manifest": str(subset_manifest_path),
        "total_selected": subset_manifest.get("total_selected"),
        "overall": overall,
        "levels": {item["level"]: item for item in level_metrics},
    }

    summary_json = summary_dir / "kernelbench_hip_100_summary.json"
    summary_csv = summary_dir / "kernelbench_hip_100_summary.csv"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "level",
        "expected_tasks",
        "tasks_in_comparison",
        "total_records",
        "origin_compile_success_rate",
        "origin_correctness_success_rate",
        "optimized_compile_success_rate",
        "optimized_run_success_rate",
        "optimized_correctness_success_rate",
        "pair_ok_count",
        "record_pass_rate",
        "task_pass_count",
        "task_pass_rate",
        "valid_speedup_count",
        "mean_speedup",
        "median_speedup",
        "best_speedup",
        "mean_task_best_speedup",
        "median_task_best_speedup",
        "best_task_speedup",
        "comparison_json",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in [*level_metrics, overall]:
            writer.writerow({field: row.get(field) for field in fieldnames})

    print(f"[SUMMARY] wrote JSON: {summary_json}")
    print(f"[SUMMARY] wrote CSV:  {summary_csv}")
    print(
        "[SUMMARY] overall "
        f"tasks={overall['expected_tasks']} "
        f"record_pass_rate={overall['record_pass_rate']} "
        f"task_pass_rate={overall['task_pass_rate']} "
        f"mean_speedup={overall['mean_speedup']} "
        f"best_speedup={overall['best_speedup']}"
    )


def record_base_name(record: dict[str, Any]) -> str:
    base_name = record.get("base_name")
    if base_name:
        return str(base_name)
    hip_file = record.get("hip_file")
    if hip_file:
        return Path(str(hip_file)).stem
    raise ValueError(f"Record has no base_name/hip_file: {record}")


def is_eval_clean(record: dict[str, Any]) -> bool:
    return all(
        is_true(record, key)
        for key in ("compile_ok", "run_ok", "match_ok")
    )


def build_origin_clean_manifest(args: argparse.Namespace) -> None:
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "origin_root": str(args.origin_root.resolve()),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "levels": {},
        "clean_task_ids": [],
        "dirty_task_ids": [],
    }

    for level in args.level:
        result_path = args.origin_root / level / "baseline_hip_results.json"
        if not result_path.is_file():
            raise SystemExit(f"Origin result missing for {level}: {result_path}")

        records = json.loads(result_path.read_text(encoding="utf-8"))
        clean_records = [record for record in records if is_eval_clean(record)]
        dirty_records = [record for record in records if not is_eval_clean(record)]
        clean_tasks = sorted(record_base_name(record) for record in clean_records)
        dirty_tasks = sorted(record_base_name(record) for record in dirty_records)

        manifest["levels"][level] = {
            "origin_result": str(result_path),
            "total_tasks": len(records),
            "clean_count": len(clean_tasks),
            "dirty_count": len(dirty_tasks),
            "clean_tasks": clean_tasks,
            "dirty_tasks": dirty_tasks,
        }
        manifest["clean_task_ids"].extend(f"{level}:{name}" for name in clean_tasks)
        manifest["dirty_task_ids"].extend(f"{level}:{name}" for name in dirty_tasks)

    manifest["clean_task_ids"] = sorted(manifest["clean_task_ids"])
    manifest["dirty_task_ids"] = sorted(manifest["dirty_task_ids"])
    manifest["total_tasks"] = len(manifest["clean_task_ids"]) + len(manifest["dirty_task_ids"])
    manifest["clean_count"] = len(manifest["clean_task_ids"])
    manifest["dirty_count"] = len(manifest["dirty_task_ids"])

    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[ORIGIN_CLEAN] wrote {output_path} "
        f"clean={manifest['clean_count']} dirty={manifest['dirty_count']}"
    )


def task_key_from_pair(record: dict[str, Any]) -> str | None:
    base_name = record.get("base_name")
    level = record.get("_level")
    if not base_name or not level:
        return None
    return f"{level}:{base_name}"


def comparison_metrics(records: list[dict[str, Any]], expected_tasks: int, comparison_json: Path) -> dict[str, Any]:
    result = metrics_for("placeholder", records, expected_tasks, comparison_json)
    compare_errors: dict[str, int] = {}
    for record in records:
        key = str(record.get("compare_error") or "none")
        compare_errors[key] = compare_errors.get(key, 0) + 1
    result["compare_error_counts"] = dict(sorted(compare_errors.items()))
    return result


def summarize_reeval(args: argparse.Namespace) -> None:
    reeval_root = args.reeval_root.resolve()
    manifest = json.loads(args.origin_clean_manifest.read_text(encoding="utf-8"))
    clean_ids = set(manifest["clean_task_ids"])
    summary_root = reeval_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    rollout_rows = []
    for rollout_n in args.rollout:
        run_root = reeval_root / f"rollout_n_{rollout_n}"
        run_summary_dir = run_root / "summary"
        run_summary_dir.mkdir(parents=True, exist_ok=True)

        level_summaries = {}
        all_records: list[dict[str, Any]] = []
        clean_records: list[dict[str, Any]] = []

        for level in args.level:
            comparison_json = run_root / level / "comparison" / "origin_vs_optimized_results.json"
            if not comparison_json.is_file():
                raise SystemExit(f"Comparison JSON missing: {comparison_json}")
            records = json.loads(comparison_json.read_text(encoding="utf-8"))
            level_records = [dict(record, _level=level) for record in records]
            level_clean_ids = {f"{level}:{name}" for name in manifest["levels"][level]["clean_tasks"]}
            level_clean_records = [
                record
                for record in level_records
                if task_key_from_pair(record) in level_clean_ids
            ]

            all_records.extend(level_records)
            clean_records.extend(level_clean_records)

            all_metrics = comparison_metrics(
                level_records,
                int(manifest["levels"][level]["total_tasks"]),
                comparison_json,
            )
            clean_metrics = comparison_metrics(
                level_clean_records,
                int(manifest["levels"][level]["clean_count"]),
                comparison_json,
            )
            all_metrics["level"] = level
            clean_metrics["level"] = level
            level_summaries[level] = {
                "all": all_metrics,
                "origin_clean": clean_metrics,
            }

        overall_all = comparison_metrics(all_records, int(manifest["total_tasks"]), Path(""))
        overall_clean = comparison_metrics(clean_records, int(manifest["clean_count"]), Path(""))
        overall_all["level"] = "overall"
        overall_clean["level"] = "overall"

        summary = {
            "run_root": str(run_root),
            "origin_clean_manifest": str(args.origin_clean_manifest.resolve()),
            "rollout_n": rollout_n,
            "overall": {
                "all": overall_all,
                "origin_clean": overall_clean,
            },
            "levels": level_summaries,
        }
        summary_json = run_summary_dir / "kernelbench_hip_100_reeval_summary.json"
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        summary_csv = run_summary_dir / "kernelbench_hip_100_reeval_summary.csv"
        fieldnames = [
            "scope",
            "level",
            "expected_tasks",
            "tasks_in_comparison",
            "total_records",
            "optimized_compile_success_rate",
            "optimized_run_success_rate",
            "optimized_correctness_success_rate",
            "record_pass_rate",
            "task_pass_count",
            "task_pass_rate",
            "mean_speedup",
            "median_speedup",
            "best_speedup",
            "mean_task_best_speedup",
            "median_task_best_speedup",
            "best_task_speedup",
        ]
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            rows = []
            for level in args.level:
                rows.append(("all", level, level_summaries[level]["all"]))
                rows.append(("origin_clean", level, level_summaries[level]["origin_clean"]))
            rows.append(("all", "overall", overall_all))
            rows.append(("origin_clean", "overall", overall_clean))
            for scope, level, metrics in rows:
                row = {"scope": scope, "level": level}
                row.update({field: metrics.get(field) for field in fieldnames if field not in row})
                writer.writerow(row)

        rollout_rows.append(
            {
                "rollout_n": rollout_n,
                "origin_clean_tasks": overall_clean["expected_tasks"],
                "origin_clean_task_pass_count": overall_clean["task_pass_count"],
                "origin_clean_task_pass_rate": overall_clean["task_pass_rate"],
                "origin_clean_record_pass_rate": overall_clean["record_pass_rate"],
                "origin_clean_mean_task_best_speedup": overall_clean["mean_task_best_speedup"],
                "origin_clean_median_task_best_speedup": overall_clean["median_task_best_speedup"],
                "origin_clean_best_task_speedup": overall_clean["best_task_speedup"],
                "all_task_pass_count": overall_all["task_pass_count"],
                "all_task_pass_rate": overall_all["task_pass_rate"],
                "summary_json": str(summary_json),
            }
        )
        print(
            f"[REEVAL_SUMMARY] rollout_n={rollout_n} "
            f"clean_task_pass={overall_clean['task_pass_count']}/{overall_clean['expected_tasks']} "
            f"clean_task_best_mean={overall_clean['mean_task_best_speedup']}"
        )

    comparison_csv = summary_root / "origin_clean_rollout_comparison.csv"
    with comparison_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rollout_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rollout_rows)
    print(f"[REEVAL_SUMMARY] wrote rollout comparison: {comparison_csv}")


def diagnose_reeval(args: argparse.Namespace) -> None:
    reeval_root = args.reeval_root.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def load_rows(rollout_n: str, level: str) -> list[dict[str, Any]]:
        path = reeval_root / f"rollout_n_{rollout_n}" / level / "comparison" / "origin_vs_optimized_results.json"
        return [dict(record, _level=level) for record in json.loads(path.read_text(encoding="utf-8"))]

    diagnostics: dict[str, Any] = {
        "reeval_root": str(reeval_root),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rollouts": args.rollout,
        "levels": args.level,
        "n4_vs_n16_first4": {},
        "error_counts": {},
    }

    if "4" in args.rollout and "16" in args.rollout:
        for level in args.level:
            n4 = {
                (record.get("base_name"), record.get("gen_idx")): record
                for record in load_rows("4", level)
                if record.get("gen_idx") is not None
            }
            n16 = {
                (record.get("base_name"), record.get("gen_idx")): record
                for record in load_rows("16", level)
                if record.get("gen_idx") is not None and int(record.get("gen_idx")) < 4
            }
            common = sorted(set(n4) & set(n16))
            diff_rows = []
            fields = [
                "origin_compile_ok",
                "origin_run_ok",
                "origin_match_ok",
                "optimized_compile_ok",
                "optimized_run_ok",
                "optimized_match_ok",
                "pair_ok",
            ]
            for key in common:
                left = n4[key]
                right = n16[key]
                if any(left.get(field) != right.get(field) for field in fields):
                    diff_rows.append(
                        {
                            "base_name": key[0],
                            "gen_idx": key[1],
                            "n4": {field: left.get(field) for field in fields},
                            "n16": {field: right.get(field) for field in fields},
                            "n4_error": left.get("compare_error") or left.get("optimized_preflight_error_message"),
                            "n16_error": right.get("compare_error") or right.get("optimized_preflight_error_message"),
                        }
                    )
            diagnostics["n4_vs_n16_first4"][level] = {
                "common_candidates": len(common),
                "status_differences": len(diff_rows),
                "differences_preview": diff_rows[:20],
            }

    for rollout_n in args.rollout:
        diagnostics["error_counts"][rollout_n] = {}
        for level in args.level:
            rows = load_rows(rollout_n, level)
            counts: dict[str, int] = {}
            for record in rows:
                err = str(
                    record.get("compare_error")
                    or record.get("optimized_preflight_error_message")
                    or "none"
                )
                if "No space left on device" in err:
                    key = "no_space_left_on_device"
                elif "out of memory" in err.lower():
                    key = "out_of_memory"
                else:
                    key = err[:120]
                counts[key] = counts.get(key, 0) + 1
            diagnostics["error_counts"][rollout_n][level] = dict(sorted(counts.items()))

    output_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[REEVAL_DIAGNOSE] wrote diagnostics: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage-subset", help="Create a staged kernelbench_hip subset")
    stage.add_argument("--source-root", type=Path, required=True)
    stage.add_argument("--subset-root", type=Path, required=True)
    stage.add_argument("--manifest", type=Path, required=True)
    stage.add_argument("--level", action="append", required=True, help="LEVEL:COUNT, repeatable")
    stage.set_defaults(func=stage_subset)

    flat = subparsers.add_parser("stage-flat-subset", help="Create a persistent flat kernelbench_hip subset")
    flat.add_argument("--source-root", type=Path, required=True)
    flat.add_argument("--subset-root", type=Path, required=True)
    flat.add_argument("--manifest", type=Path)
    flat.add_argument("--level", action="append", required=True, help="LEVEL:COUNT, repeatable")
    flat.add_argument("--overwrite", action="store_true")
    flat.set_defaults(func=stage_flat_subset)

    generation = subparsers.add_parser("summarize-generation", help="Validate and print generation manifest stats")
    generation.add_argument("--manifest", type=Path, required=True)
    generation.add_argument("--label", required=True)
    generation.set_defaults(func=summarize_generation)

    summary = subparsers.add_parser("summarize-run", help="Aggregate per-level comparison outputs")
    summary.add_argument("--run-root", type=Path, required=True)
    summary.add_argument("--subset-manifest", type=Path, required=True)
    summary.add_argument("--level", action="append", required=True)
    summary.set_defaults(func=summarize_run)

    clean = subparsers.add_parser("origin-clean-manifest", help="Build fixed origin-clean task manifest")
    clean.add_argument("--origin-root", type=Path, required=True)
    clean.add_argument("--output", type=Path, required=True)
    clean.add_argument("--level", action="append", required=True)
    clean.set_defaults(func=build_origin_clean_manifest)

    reeval = subparsers.add_parser("summarize-reeval", help="Summarize fixed-origin re-evaluation outputs")
    reeval.add_argument("--reeval-root", type=Path, required=True)
    reeval.add_argument("--origin-clean-manifest", type=Path, required=True)
    reeval.add_argument("--rollout", action="append", required=True)
    reeval.add_argument("--level", action="append", required=True)
    reeval.set_defaults(func=summarize_reeval)

    diagnose = subparsers.add_parser("diagnose-reeval", help="Diagnose re-evaluation consistency and errors")
    diagnose.add_argument("--reeval-root", type=Path, required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--rollout", action="append", required=True)
    diagnose.add_argument("--level", action="append", required=True)
    diagnose.set_defaults(func=diagnose_reeval)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
