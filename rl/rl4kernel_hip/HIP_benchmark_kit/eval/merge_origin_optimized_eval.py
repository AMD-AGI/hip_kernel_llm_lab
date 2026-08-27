# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge origin/optimized server-inprocess eval records."
    )
    parser.add_argument("--origin-json", required=True, help="Origin preflight JSON path.")
    parser.add_argument("--origin-csv", default="", help="Origin preflight CSV path (fallback).")
    parser.add_argument("--optimized-json", required=True, help="Optimized preflight JSON path.")
    parser.add_argument("--optimized-csv", default="", help="Optimized preflight CSV path (fallback).")
    parser.add_argument("--pytorch-func-dir", required=True, help="PyTorch functional reference directory.")
    parser.add_argument("--pytorch-modu-dir", required=True, help="PyTorch module reference directory.")
    parser.add_argument("--output-dir", required=True, help="Comparison output directory.")
    return parser.parse_args(argv)


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso_timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no", ""}:
            return False
    return bool(value)


def as_int_or_none(value):
    if value in (None, "", "None"):
        return None
    return int(float(value))


def as_float_or_none(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def load_eval_records(json_path, csv_path=""):
    if json_path and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list of records in {json_path}")
        return payload
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise FileNotFoundError(f"Missing eval results: json={json_path}, csv={csv_path}")


def normalize_records(records):
    normalized = []
    for idx, record in enumerate(records):
        hip_file = record.get("hip_file")
        base_name = record.get("base_name")
        if not hip_file or not base_name:
            raise ValueError(f"Eval record missing hip_file/base_name: {record}")
        normalized.append({
            "_record_id": idx,
            "hip_file": hip_file,
            "base_name": base_name,
            "gen_idx": as_int_or_none(record.get("gen_idx")),
            "compile_ok": as_bool(record.get("compile_ok")),
            "run_ok": as_bool(record.get("run_ok")),
            "match_ok": as_bool(record.get("match_ok")),
            "hip_time_ms": as_float_or_none(record.get("hip_time_ms")),
            "pytorch_time_ms": as_float_or_none(record.get("pytorch_time_ms")),
            "compile_cache_key": record.get("compile_cache_key"),
            "compile_cache_hit": record.get("compile_cache_hit"),
            "compile_artifact_path": record.get("compile_artifact_path"),
            "compiled_library_path": record.get("compiled_library_path"),
            "compiled_module_name": record.get("compiled_module_name"),
            "artifact_side": record.get("artifact_side"),
            "perf_gpu_id": record.get("perf_gpu_id"),
            "perf_started_at": record.get("perf_started_at"),
            "perf_finished_at": record.get("perf_finished_at"),
            "error_message": record.get("error_message"),
        })
    return normalized


def ensure_unique_exact_keys(records, side_label):
    by_exact = defaultdict(list)
    for record in records:
        by_exact[(record["base_name"], record["gen_idx"])].append(record)
    duplicates = {
        key: rows
        for key, rows in by_exact.items()
        if len(rows) > 1
    }
    if duplicates:
        details = ", ".join(
            f"{base_name}[gen={gen_idx}] x{len(rows)}"
            for (base_name, gen_idx), rows in sorted(duplicates.items())
        )
        raise ValueError(f"Ambiguous duplicate keys in {side_label} results: {details}")
    return {key: rows[0] for key, rows in by_exact.items()}


def index_by_base(records):
    by_base = defaultdict(list)
    for record in records:
        by_base[record["base_name"]].append(record)
    return by_base


def make_pair_row(origin_record, optimized_record, compare_error=None):
    sample = optimized_record or origin_record
    return {
        "base_name": sample["base_name"] if sample else None,
        "gen_idx": (
            optimized_record["gen_idx"]
            if optimized_record and optimized_record["gen_idx"] is not None
            else origin_record["gen_idx"] if origin_record else None
        ),
        "origin_hip_file": origin_record["hip_file"] if origin_record else None,
        "optimized_hip_file": optimized_record["hip_file"] if optimized_record else None,
        "origin_compile_ok": origin_record["compile_ok"] if origin_record else None,
        "origin_run_ok": origin_record["run_ok"] if origin_record else None,
        "origin_match_ok": origin_record["match_ok"] if origin_record else None,
        "optimized_compile_ok": optimized_record["compile_ok"] if optimized_record else None,
        "optimized_run_ok": optimized_record["run_ok"] if optimized_record else None,
        "optimized_match_ok": optimized_record["match_ok"] if optimized_record else None,
        "origin_hip_time_ms": None,
        "optimized_hip_time_ms": None,
        "speedup": None,
        "perf_gpu_id": None,
        "perf_started_at": None,
        "perf_finished_at": None,
        "pair_ok": False,
        "compare_error": compare_error,
        "origin_preflight_error_message": origin_record["error_message"] if origin_record else None,
        "optimized_preflight_error_message": optimized_record["error_message"] if optimized_record else None,
        "origin_perf_error_message": origin_record["error_message"] if origin_record else None,
        "optimized_perf_error_message": optimized_record["error_message"] if optimized_record else None,
        "origin_compile_cache_hit_perf": origin_record.get("compile_cache_hit") if origin_record else None,
        "optimized_compile_cache_hit_perf": optimized_record.get("compile_cache_hit") if optimized_record else None,
        "origin_perf_gpu_id": origin_record.get("perf_gpu_id") if origin_record else None,
        "optimized_perf_gpu_id": optimized_record.get("perf_gpu_id") if optimized_record else None,
        "origin_perf_started_at": origin_record.get("perf_started_at") if origin_record else None,
        "origin_perf_finished_at": origin_record.get("perf_finished_at") if origin_record else None,
        "optimized_perf_started_at": optimized_record.get("perf_started_at") if optimized_record else None,
        "optimized_perf_finished_at": optimized_record.get("perf_finished_at") if optimized_record else None,
        "_origin_record": origin_record,
        "_optimized_record": optimized_record,
    }


def sort_pair_rows(rows):
    return sorted(rows, key=lambda row: (row["base_name"], -1 if row["gen_idx"] is None else row["gen_idx"]))


def build_pair_plan(origin_records, optimized_records):
    origin_exact = ensure_unique_exact_keys(origin_records, "origin")
    optimized_exact = ensure_unique_exact_keys(optimized_records, "optimized")
    origin_by_base = index_by_base(origin_records)
    optimized_by_base = index_by_base(optimized_records)

    pair_rows = []
    used_origin_ids = set()
    used_optimized_ids = set()

    for key, optimized_record in optimized_exact.items():
        origin_record = origin_exact.get(key)
        if origin_record is not None:
            pair_rows.append(make_pair_row(origin_record, optimized_record))
            used_origin_ids.add(origin_record["_record_id"])
            used_optimized_ids.add(optimized_record["_record_id"])

    for optimized_record in optimized_records:
        if optimized_record["_record_id"] in used_optimized_ids:
            continue

        candidates = origin_by_base.get(optimized_record["base_name"], [])
        if len(candidates) == 1 and candidates[0]["gen_idx"] is None:
            pair_rows.append(make_pair_row(candidates[0], optimized_record))
            used_origin_ids.add(candidates[0]["_record_id"])
            used_optimized_ids.add(optimized_record["_record_id"])
            continue

        if (
            optimized_record["gen_idx"] is None
            and len(candidates) > 1
        ):
            raise ValueError(
                "Ambiguous many-to-many join for "
                f"{optimized_record['base_name']}: optimized row has no gen_idx but origin has "
                f"{len(candidates)} candidates"
            )

        pair_rows.append(make_pair_row(None, optimized_record, compare_error="missing_origin_pair"))
        used_optimized_ids.add(optimized_record["_record_id"])

    for origin_record in origin_records:
        if origin_record["_record_id"] in used_origin_ids:
            continue
        candidates = optimized_by_base.get(origin_record["base_name"], [])
        if origin_record["gen_idx"] is None and len(candidates) > 1:
            # Fanout should already have consumed this origin row above.
            raise ValueError(
                "Ambiguous many-to-many join for "
                f"{origin_record['base_name']}: origin row has no gen_idx but optimized has "
                f"{len(candidates)} candidates"
            )
        pair_rows.append(make_pair_row(origin_record, None, compare_error="missing_optimized_pair"))

    return sort_pair_rows(pair_rows)


def has_valid_perf(value):
    return value is not None and value > 0


def infer_preflight_failure_reason(pair_row):
    origin_ok = all([
        pair_row["origin_compile_ok"],
        pair_row["origin_run_ok"],
        pair_row["origin_match_ok"],
    ])
    optimized_ok = all([
        pair_row["optimized_compile_ok"],
        pair_row["optimized_run_ok"],
        pair_row["optimized_match_ok"],
    ])
    if not origin_ok and not optimized_ok:
        return "both_preflight_failed"
    if not origin_ok:
        return "origin_preflight_failed"
    if not optimized_ok:
        return "optimized_preflight_failed"
    return "preflight_failed"


def finalize_pair_row(pair_row):
    if pair_row["compare_error"] is not None:
        return pair_row

    required = [
        pair_row["origin_hip_file"],
        pair_row["optimized_hip_file"],
    ]
    if any(value is None for value in required):
        pair_row["compare_error"] = "missing_pair"
        return pair_row

    if not all([
        pair_row["origin_compile_ok"],
        pair_row["origin_run_ok"],
        pair_row["origin_match_ok"],
        pair_row["optimized_compile_ok"],
        pair_row["optimized_run_ok"],
        pair_row["optimized_match_ok"],
    ]):
        pair_row["compare_error"] = infer_preflight_failure_reason(pair_row)
        return pair_row

    origin_record = pair_row["_origin_record"]
    optimized_record = pair_row["_optimized_record"]
    pair_row["origin_hip_time_ms"] = origin_record.get("hip_time_ms")
    pair_row["optimized_hip_time_ms"] = optimized_record.get("hip_time_ms")

    pair_row["perf_gpu_id"] = origin_record.get("perf_gpu_id")
    pair_row["perf_started_at"] = origin_record.get("perf_started_at")
    pair_row["perf_finished_at"] = optimized_record.get("perf_finished_at")

    if not has_valid_perf(pair_row["origin_hip_time_ms"]):
        pair_row["compare_error"] = "invalid_origin_perf"
        return pair_row
    if not has_valid_perf(pair_row["optimized_hip_time_ms"]):
        pair_row["compare_error"] = "invalid_optimized_perf"
        return pair_row

    pair_row["speedup"] = pair_row["origin_hip_time_ms"] / pair_row["optimized_hip_time_ms"]
    pair_row["pair_ok"] = True
    pair_row["compare_error"] = None
    return pair_row


def build_perf_trace(pair_rows):
    perf_trace = []
    for row in pair_rows:
        for side in ("origin", "optimized"):
            started_at = row.get(f"{side}_perf_started_at")
            finished_at = row.get(f"{side}_perf_finished_at")
            gpu_id = row.get(f"{side}_perf_gpu_id")
            if not started_at or not finished_at:
                continue
            perf_trace.append({
                "base_name": row["base_name"],
                "gen_idx": row["gen_idx"],
                "side": side,
                "perf_gpu_id": gpu_id,
                "perf_started_at": started_at,
                "perf_finished_at": finished_at,
            })
    return perf_trace


def validate_non_overlapping_trace(perf_trace):
    by_gpu = defaultdict(list)
    for row in perf_trace:
        gpu_key = row.get("perf_gpu_id", "default")
        by_gpu[gpu_key].append(row)
    for gpu_rows in by_gpu.values():
        ordered = sorted(
            gpu_rows,
            key=lambda row: parse_iso_timestamp(row.get("perf_started_at", row.get("pair_started_at"))),
        )
        for prev, curr in zip(ordered, ordered[1:]):
            prev_end = parse_iso_timestamp(prev.get("perf_finished_at", prev.get("pair_finished_at")))
            curr_start = parse_iso_timestamp(curr.get("perf_started_at", curr.get("pair_started_at")))
            if curr_start < prev_end:
                return False
    return True


def count_trace_overlaps(perf_trace):
    overlaps = 0
    by_gpu = defaultdict(list)
    for row in perf_trace:
        gpu_key = row.get("perf_gpu_id", "default")
        by_gpu[gpu_key].append(row)
    for gpu_rows in by_gpu.values():
        ordered = sorted(
            gpu_rows,
            key=lambda row: parse_iso_timestamp(row.get("perf_started_at", row.get("pair_started_at"))),
        )
        for prev, curr in zip(ordered, ordered[1:]):
            prev_end = parse_iso_timestamp(prev.get("perf_finished_at", prev.get("pair_finished_at")))
            curr_start = parse_iso_timestamp(curr.get("perf_started_at", curr.get("pair_started_at")))
            if curr_start < prev_end:
                overlaps += 1
    return overlaps


def sanitize_pair_rows(pair_rows):
    cleaned_rows = []
    for row in pair_rows:
        cleaned = {key: value for key, value in row.items() if not key.startswith("_")}
        cleaned_rows.append(cleaned)
    return cleaned_rows


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(pair_rows):
    total = len(pair_rows)
    valid_rows = [row for row in pair_rows if row["pair_ok"]]
    print("\n" + "=" * 100)
    print("ORIGIN VS OPTIMIZED COMPARISON")
    print("=" * 100)
    print(f"Total comparison rows:       {total}")
    print(f"Valid perf pairs:            {len(valid_rows)} ({(len(valid_rows) / total * 100) if total else 0:.1f}%)")
    if valid_rows:
        speedups = [row["speedup"] for row in valid_rows]
        speedups.sort()
        midpoint = len(speedups) // 2
        median = (
            speedups[midpoint]
            if len(speedups) % 2 == 1
            else (speedups[midpoint - 1] + speedups[midpoint]) / 2
        )
        print(f"Average speedup:             {sum(speedups) / len(speedups):.3f}x")
        print(f"Median speedup:              {median:.3f}x")
        print(f"Max speedup:                 {max(speedups):.3f}x")
        print(f"Min speedup:                 {min(speedups):.3f}x")
    print("=" * 100)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    origin_records = normalize_records(load_eval_records(args.origin_json, args.origin_csv))
    optimized_records = normalize_records(load_eval_records(args.optimized_json, args.optimized_csv))
    pair_rows = [finalize_pair_row(row) for row in build_pair_plan(origin_records, optimized_records)]
    perf_trace = build_perf_trace(pair_rows)

    overlap_count = count_trace_overlaps(perf_trace) if perf_trace else 0
    if overlap_count:
        print(
            "Warning: detected overlapping perf windows in the comparison trace; "
            f"continuing because batched/two-stage eval and reused baselines are not isolated traces "
            f"(overlap_count={overlap_count}).",
            file=sys.stderr,
        )

    clean_rows = sanitize_pair_rows(pair_rows)
    output_json = os.path.join(args.output_dir, "origin_vs_optimized_results.json")
    output_csv = os.path.join(args.output_dir, "origin_vs_optimized_results.csv")
    perf_trace_csv = os.path.join(args.output_dir, "origin_vs_optimized_perf_trace.csv")

    write_json(output_json, clean_rows)
    write_csv(output_csv, clean_rows)
    write_csv(perf_trace_csv, perf_trace)

    print_summary(clean_rows)
    print("Saved comparison results:")
    print(f"  JSON: {output_json}")
    print(f"  CSV:  {output_csv}")
    print(f"  Trace: {perf_trace_csv}")


if __name__ == "__main__":
    main()
