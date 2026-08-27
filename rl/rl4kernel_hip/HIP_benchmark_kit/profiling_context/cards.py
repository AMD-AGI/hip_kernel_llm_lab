# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Build generation prompt maps from Metrix profiling artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from HIP_benchmark_kit.contracts.manifests import write_json


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, dict):
        return _format_metric(value.get("avg"))
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _select_primary_kernel(payload: dict[str, Any]) -> dict[str, Any]:
    primary = payload.get("primary_kernel")
    if isinstance(primary, dict):
        return primary
    kernels = payload.get("kernels") or []
    kernel_dicts = [kernel for kernel in kernels if isinstance(kernel, dict)]
    if not kernel_dicts:
        return {}
    return max(kernel_dicts, key=lambda kernel: _metric_float(kernel.get("duration_us")))


def _metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("avg")
    return value


def _metric_float(value: Any) -> float:
    try:
        return float(_metric_value(value) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _kernel_metric(kernel: dict[str, Any], name: str) -> Any:
    if name in kernel:
        return _metric_value(kernel.get(name))
    metrics = kernel.get("metrics")
    if isinstance(metrics, dict):
        return _metric_value(metrics.get(name))
    return None


def _profile_timings(payload: dict[str, Any]) -> dict[str, Any]:
    timings = payload.get("stage_timings_s")
    if isinstance(timings, dict):
        return timings
    run_config = payload.get("run_config")
    if isinstance(run_config, dict) and isinstance(run_config.get("stage_timings_s"), dict):
        return run_config["stage_timings_s"]
    return {}


def _profile_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    sample = payload.get("sample")
    if isinstance(sample, dict):
        return sample
    return {}


def _inventory_kernel_names(payload: dict[str, Any]) -> list[str]:
    names = payload.get("inventory_kernel_names")
    if isinstance(names, list):
        return [str(name) for name in names if name]
    kernels = payload.get("kernels") or []
    return [str(kernel.get("name")) for kernel in kernels if isinstance(kernel, dict) and kernel.get("name")]


def render_profile_card(sample_name: str, payload: dict[str, Any]) -> str:
    primary = _select_primary_kernel(payload)
    metadata = _profile_metadata(payload)
    timings = _profile_timings(payload)
    lines = [
        "Use this profiling context when optimizing the HIP kernel.",
        f"Sample: {sample_name}",
        f"Category: {payload.get('category') or metadata.get('category') or 'unknown'}",
        f"Note: {payload.get('note') or metadata.get('note') or 'n/a'}",
        f"Custom kernels: {', '.join(metadata.get('custom_kernel_names') or []) or 'n/a'}",
        f"Input summary: {json.dumps(metadata.get('inputs') or [], ensure_ascii=False)}",
        f"Init summary: {json.dumps(metadata.get('init_inputs') or [], ensure_ascii=False)}",
        f"Inventory kernels: {', '.join(_inventory_kernel_names(payload)) or 'n/a'}",
        f"Applied kernel filter: {payload.get('kernel_filter') or 'none'}",
        f"Interpretation: {payload.get('interpretation') or 'n/a'}",
        f"Compile seconds: {_format_metric(timings.get('compile_seconds'))}",
        f"Prewarm seconds: {_format_metric(timings.get('prewarm_seconds'))}",
        f"Inventory profile seconds: {_format_metric(timings.get('inventory_profile_seconds'))}",
        f"Filtered profile seconds: {_format_metric(timings.get('filtered_profile_seconds'))}",
    ]
    if primary:
        lines.extend(
            [
                f"Primary kernel: {primary.get('name') or 'n/a'}",
                f"Primary duration us: {_format_metric(primary.get('duration_us'))}",
                f"HBM bandwidth utilization: {_format_metric(_kernel_metric(primary, 'memory.hbm_bandwidth_utilization'))}",
                f"L2 hit rate: {_format_metric(_kernel_metric(primary, 'memory.l2_hit_rate'))}",
                f"Coalescing efficiency: {_format_metric(_kernel_metric(primary, 'memory.coalescing_efficiency'))}",
                f"Total FLOPs: {_format_metric(_kernel_metric(primary, 'compute.total_flops'))}",
            ]
        )
    return "\n".join(lines)


def build_profile_cards(
    *,
    profile_dir: Path,
    input_dir: Path,
    output_json: Path,
    arms: list[str],
    missing_policy: str,
) -> dict[str, Any]:
    if missing_policy not in {"fail", "skip", "empty"}:
        raise SystemExit(f"Unsupported profile missing policy: {missing_policy}")
    if not input_dir.is_dir():
        raise SystemExit(f"Input HIP directory not found: {input_dir}")
    if not profile_dir.is_dir() and missing_policy == "fail":
        raise SystemExit(f"Profile artifact directory not found: {profile_dir}")
    arms = arms or ["B_profile_raw"]
    prompt_maps = {arm: {"prompt_map": {}} for arm in arms}
    missing_inputs: list[str] = []
    sample_count = 0

    for hip_file in sorted(input_dir.glob("*.hip")):
        sample_count += 1
        artifact = profile_dir / f"{hip_file.stem}_filtered.json"
        if not artifact.is_file():
            missing_inputs.append(hip_file.name)
            if missing_policy == "fail":
                continue
            card_text = "" if missing_policy == "empty" else None
        else:
            card_text = render_profile_card(hip_file.stem, _load_json(artifact))
        if card_text is None:
            continue
        for arm in arms:
            prompt_maps[arm]["prompt_map"][hip_file.name] = {
                "prompt_text": card_text,
                "profile_artifact": str(artifact) if artifact.is_file() else "",
            }

    if missing_inputs and missing_policy == "fail":
        preview = ", ".join(missing_inputs[:10])
        raise SystemExit(f"Missing profiling artifacts for {len(missing_inputs)} input(s): {preview}")

    payload = {
        "metadata": {
            "profile_dir": str(profile_dir),
            "input_dir": str(input_dir),
            "sample_count": sample_count,
            "missing_count": len(missing_inputs),
            "missing_inputs": missing_inputs,
            "arms": arms,
        },
        "arms": prompt_maps,
    }
    write_json(output_json, payload)
    print(f"Wrote profile prompt map: {output_json}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--arms", default="B_profile_raw")
    parser.add_argument("--missing-policy", choices=("fail", "skip", "empty"), default="fail")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_profile_cards(
        profile_dir=args.profile_dir,
        input_dir=args.input_dir,
        output_json=args.output_json,
        arms=[item.strip() for item in args.arms.split(",") if item.strip()],
        missing_policy=args.missing_policy,
    )


if __name__ == "__main__":
    main()
