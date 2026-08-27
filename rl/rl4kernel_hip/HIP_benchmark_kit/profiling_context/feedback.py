# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Build compact multi-turn feedback contexts from prior generation rounds."""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path
from typing import Any

from HIP_benchmark_kit.contracts.manifests import write_json
from reward.utils import strip_think_blocks


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_value(value: Any, *, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}g}"
    return str(value)


def _balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_single = False
        in_double = False
        escaped = False
        for idx in range(start, len(text)):
            current = text[idx]
            if escaped:
                escaped = False
                continue
            if current == "\\" and (in_single or in_double):
                escaped = True
                continue
            if current == "'" and not in_double:
                in_single = not in_single
                continue
            if current == '"' and not in_single:
                in_double = not in_double
                continue
            if in_single or in_double:
                continue
            if current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
    return candidates


def extract_thought_from_raw_response(raw_response: Any) -> str:
    """Extract the JSON `thought` field from a kernel-agent raw response."""
    if raw_response is None:
        return ""
    if isinstance(raw_response, dict):
        raw_payload = raw_response.get("raw_response", raw_response)
        if isinstance(raw_payload, dict):
            thought = raw_payload.get("thought")
            return thought.strip() if isinstance(thought, str) else ""
        raw_response = raw_payload

    text = strip_think_blocks(str(raw_response))
    for candidate in _balanced_json_candidates(text):
        parsed = None
        try:
            parsed = json.loads(candidate)
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
            except Exception:
                parsed = None
        if isinstance(parsed, dict):
            thought = parsed.get("thought")
            if isinstance(thought, str) and thought.strip():
                return thought.strip()
    return ""


def load_raw_response_thought(path: Path) -> str:
    if not path.is_file():
        return ""
    return extract_thought_from_raw_response(_load_json(path))


def _metric_from_kernel(kernel: dict[str, Any], name: str) -> Any:
    if name in kernel:
        return kernel.get(name)
    metrics = kernel.get("metrics") or {}
    value = metrics.get(name)
    if isinstance(value, dict):
        return value.get("avg")
    return value


def _duration_from_kernel(kernel: dict[str, Any]) -> Any:
    duration = kernel.get("duration_us")
    if isinstance(duration, dict):
        return duration.get("avg")
    return duration


def _select_primary_kernel(profile_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not profile_payload:
        return {}
    primary = profile_payload.get("primary_kernel")
    if isinstance(primary, dict):
        return primary
    kernels = profile_payload.get("kernels") or []
    kernel_dicts = [kernel for kernel in kernels if isinstance(kernel, dict)]
    if not kernel_dicts:
        return {}

    def duration_key(kernel: dict[str, Any]) -> float:
        try:
            return float(_duration_from_kernel(kernel) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return max(kernel_dicts, key=duration_key)


def summarize_profile_payload(profile_payload: dict[str, Any] | None) -> list[str]:
    if not profile_payload:
        return ["profiler: unavailable"]
    primary = _select_primary_kernel(profile_payload)
    if not primary:
        return ["profiler: no primary kernel captured"]
    interpretation = profile_payload.get("interpretation") or profile_payload.get("note") or "n/a"
    return [
        f"profiler.primary_kernel: {_format_value(primary.get('name'))}",
        f"profiler.duration_us: {_format_value(_duration_from_kernel(primary))}",
        f"profiler.hbm_bandwidth_utilization: {_format_value(_metric_from_kernel(primary, 'memory.hbm_bandwidth_utilization'))}",
        f"profiler.l2_hit_rate: {_format_value(_metric_from_kernel(primary, 'memory.l2_hit_rate'))}",
        f"profiler.coalescing_efficiency: {_format_value(_metric_from_kernel(primary, 'memory.coalescing_efficiency'))}",
        f"profiler.total_flops: {_format_value(_metric_from_kernel(primary, 'compute.total_flops'))}",
        f"profiler.interpretation: {interpretation}",
    ]


def render_feedback_card(
    *,
    sample_name: str,
    turn: int,
    thought: str,
    eval_row: dict[str, Any] | None,
    profile_payload: dict[str, Any] | None,
    blocked_reason: str = "",
    max_chars: int = 4000,
) -> str:
    """Render a small human-readable feedback card for the next generation turn."""
    eval_row = eval_row or {}
    lines = [
        f"Previous turn: {turn}",
        f"Sample: {sample_name}",
        "Previous optimization summary:",
        thought.strip() or "n/a",
        "Previous eval results:",
        f"- correctness: compile={_format_value(eval_row.get('optimized_compile_ok'))}, run={_format_value(eval_row.get('optimized_run_ok'))}, match={_format_value(eval_row.get('optimized_match_ok'))}",
        f"- origin_hip_time_ms: {_format_value(eval_row.get('origin_hip_time_ms'))}",
        f"- previous_candidate_hip_time_ms: {_format_value(eval_row.get('optimized_hip_time_ms'))}",
        f"- speedup_vs_origin: {_format_value(eval_row.get('speedup'))}",
    ]
    error = eval_row.get("compare_error") or eval_row.get("optimized_preflight_error_message") or blocked_reason
    if error:
        lines.append(f"- failure_detail: {error}")
    lines.append("Previous profiling results:")
    lines.extend(f"- {line}" for line in summarize_profile_payload(profile_payload))
    card = "\n".join(lines).strip()
    if max_chars > 0 and len(card) > max_chars:
        return card[: max(0, max_chars - 24)].rstrip() + "\n[feedback truncated]"
    return card


_GEN_SUFFIX_RE = re.compile(r"_gen(?P<gen_idx>\d+)(?:_hip)?$")


def _valid_comparison_speedup(row: dict[str, Any]) -> float | None:
    if not (
        row.get("optimized_compile_ok") is True
        and row.get("optimized_run_ok") is True
        and row.get("optimized_match_ok") is True
    ):
        return None
    try:
        speedup = float(row.get("speedup"))
    except (TypeError, ValueError):
        return None
    return speedup if speedup > 0 else None


def load_comparison_rows(comparison_json: Path) -> list[dict[str, Any]]:
    if not comparison_json.is_file():
        return []
    rows = _load_json(comparison_json)
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def comparison_candidate_key(row: dict[str, Any]) -> tuple[str, int] | None:
    base_name = str(row.get("base_name") or "")
    if not base_name:
        return None

    gen_idx_value = row.get("gen_idx")
    if gen_idx_value not in (None, ""):
        try:
            return base_name, int(gen_idx_value)
        except (TypeError, ValueError):
            pass

    optimized_file = Path(str(row.get("optimized_hip_file") or "")).stem
    match = _GEN_SUFFIX_RE.search(optimized_file)
    if match:
        return base_name, int(match.group("gen_idx"))
    return base_name, 0


def _comparison_selection_key(row: dict[str, Any]) -> tuple[int, float, int, int, int]:
    speedup = _valid_comparison_speedup(row)
    if speedup is not None:
        return (1, speedup, 1, 1, 1)
    return (
        0,
        0.0,
        int(row.get("optimized_compile_ok") is True),
        int(row.get("optimized_run_ok") is True),
        int(row.get("optimized_match_ok") is True),
    )


def index_comparison_candidates(comparison_json: Path) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in load_comparison_rows(comparison_json):
        key = comparison_candidate_key(row)
        if key is None:
            continue
        current = indexed.get(key)
        if current is None or _comparison_selection_key(row) > _comparison_selection_key(current):
            indexed[key] = row
    return indexed


def index_comparison_rows(comparison_json: Path) -> dict[str, dict[str, Any]]:
    by_base: dict[str, dict[str, Any]] = {}
    for row in load_comparison_rows(comparison_json):
        base_name = str(row.get("base_name") or "")
        if not base_name:
            continue
        current = by_base.get(base_name)
        if current is None or _comparison_selection_key(row) > _comparison_selection_key(current):
            by_base[base_name] = row
    return by_base


def stage_generated_kernels_for_profile(
    *,
    generated_dir: Path,
    original_input_dir: Path,
    staging_dir: Path,
    comparison_rows: dict[str, dict[str, Any]] | None = None,
    profile_generated: str = "valid-only",
) -> dict[str, Any]:
    """Stage generated kernels under original filenames so profiler pairing works."""
    if profile_generated not in {"always", "valid-only", "never"}:
        raise ValueError(f"Unsupported profile_generated value: {profile_generated}")
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    include_src = original_input_dir / "include"
    if include_src.is_dir():
        shutil.copytree(include_src, staging_dir / "include", dirs_exist_ok=True)
    staged: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    rows = comparison_rows or {}
    if profile_generated == "never":
        return {"staged": staged, "skipped": [{"reason": "profile_generated=never"}]}

    for original_path in sorted(original_input_dir.glob("*.hip")):
        base_name = original_path.stem
        row = rows.get(base_name, {})
        if profile_generated == "valid-only" and row:
            if not all(row.get(key) is True for key in ("optimized_compile_ok", "optimized_run_ok", "optimized_match_ok")):
                skipped.append({"base_name": base_name, "reason": row.get("compare_error") or "optimized_not_valid"})
                continue
        candidates = [
            generated_dir / original_path.name,
            generated_dir / f"{base_name}_gen0.hip",
            generated_dir / f"{base_name}_gen0_hip.hip",
        ]
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            skipped.append({"base_name": base_name, "reason": "missing_generated_kernel"})
            continue
        shutil.copy2(source, staging_dir / original_path.name)
        staged.append({"base_name": base_name, "source": str(source), "staged": str(staging_dir / original_path.name)})
    return {"staged": staged, "skipped": skipped}


def build_feedback_context(
    *,
    original_input_dir: Path,
    previous_generated_dir: Path,
    previous_raw_response_dir: Path,
    previous_comparison_json: Path,
    previous_profile_dir: Path,
    output_json: Path,
    previous_turn: int,
    feedback_max_chars: int = 4000,
) -> dict[str, Any]:
    rows = index_comparison_rows(previous_comparison_json)
    feedback_map: dict[str, dict[str, Any]] = {}
    for original_path in sorted(original_input_dir.glob("*.hip")):
        base_name = original_path.stem
        generated_candidates = [
            previous_generated_dir / original_path.name,
            previous_generated_dir / f"{base_name}_gen0.hip",
            previous_generated_dir / f"{base_name}_gen0_hip.hip",
        ]
        generated_path = next((path for path in generated_candidates if path.is_file()), None)
        raw_response_path = previous_raw_response_dir / f"{base_name}_gen0_raw_response.json"
        profile_path = previous_profile_dir / f"{base_name}_filtered.json"
        profile_payload = _load_json(profile_path) if profile_path.is_file() else None
        thought = load_raw_response_thought(raw_response_path)
        blocked_reason = "" if generated_path else "blocked_no_previous_code"
        feedback_text = render_feedback_card(
            sample_name=base_name,
            turn=previous_turn,
            thought=thought,
            eval_row=rows.get(base_name),
            profile_payload=profile_payload,
            blocked_reason=blocked_reason,
            max_chars=feedback_max_chars,
        )
        feedback_map[original_path.name] = {
            "base_name": base_name,
            "previous_turn": previous_turn,
            "previous_generated_path": str(generated_path) if generated_path else "",
            "previous_raw_response_path": str(raw_response_path) if raw_response_path.is_file() else "",
            "previous_profile_path": str(profile_path) if profile_path.is_file() else "",
            "thought": thought,
            "feedback_text": feedback_text,
            "blocked_reason": blocked_reason,
        }
    payload = {
        "metadata": {
            "previous_turn": previous_turn,
            "original_input_dir": str(original_input_dir),
            "previous_generated_dir": str(previous_generated_dir),
            "previous_comparison_json": str(previous_comparison_json),
            "previous_profile_dir": str(previous_profile_dir),
            "feedback_max_chars": feedback_max_chars,
        },
        "feedback_map": feedback_map,
    }
    write_json(output_json, payload)
    return payload
