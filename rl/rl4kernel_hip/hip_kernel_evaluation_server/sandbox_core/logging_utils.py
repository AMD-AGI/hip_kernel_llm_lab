# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
WHITESPACE_RE = re.compile(r"\s+")

RESET = "\033[0m"
PASS_COLOR = "\033[38;5;78m"
SPEEDUP_FAST_COLOR = "\033[38;5;28m"
SPEEDUP_SLOW_COLOR = "\033[38;5;120m"
FAIL_COLOR = "\033[38;5;223m"
SUMMARY_COLOR = "\033[38;5;81m"

_STAGE_LABELS = {
    "COMPILATION": "compilation",
    "TEST_RUN": "test run",
    "REF_RUN": "reference run",
    "REF_COMPILE_RUN": "reference compile run",
    "REF_GOLDEN_RUN": "reference golden run",
    "REF_PERF_RUN": "reference perf run",
    "RESULT_MISMATCH": "result mismatch",
    "EXCEPTION": "evaluation",
    "PARALLEL_WORKER": "parallel worker",
}


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    term = os.environ.get("TERM", "")
    is_tty = getattr(sys.stderr, "isatty", lambda: False)()
    return bool(term and term.lower() != "dumb" and is_tty)


def colorize(text: str, color: str) -> str:
    if not text or not _supports_color():
        return text
    return f"{color}{text}{RESET}"


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _normalize_text(text: Any) -> str:
    return WHITESPACE_RE.sub(" ", strip_ansi(str(text or "")).replace("\r", "\n")).strip()


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


class StripAnsiFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return strip_ansi(super().format(record))


def configure_logging(log_file: str | Path) -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(StripAnsiFormatter(LOG_FORMAT))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def format_speedup(speedup: float) -> str:
    color = SPEEDUP_FAST_COLOR if speedup > 1.0 else SPEEDUP_SLOW_COLOR
    return colorize(f"{speedup:.3f}x", color)


def _reference_cache_hit_labels(timing: Optional[Mapping[str, Any]]) -> list[str]:
    timing = timing or {}
    labels = []
    if timing.get("reference_compile_cache_hit") is True:
        labels.append("compile")
    if timing.get("reference_golden_cache_hit") is True:
        labels.append("golden")
    if timing.get("reference_perf_cache_hit") is True:
        labels.append("perf")
    return labels


def _format_reference_cache_hits(
    timing: Optional[Mapping[str, Any]],
    *,
    color: bool,
) -> Optional[str]:
    labels = _reference_cache_hit_labels(timing)
    if not labels:
        return None
    text = f"ref_cache={','.join(labels)}"
    return colorize(text, SUMMARY_COLOR) if color else text


def _stage_label(stage: str) -> str:
    return _STAGE_LABELS.get(stage, stage.lower().replace("_", " "))


def _score_error_line(line: str) -> int:
    lower = line.lower()
    score = 0
    if "error:" in lower:
        score += 6
    if any(
        token in lower
        for token in (
            "assertionerror",
            "runtimeerror",
            "valueerror",
            "typeerror",
            "nameerror",
            "indexerror",
            "keyerror",
            "importerror",
            "modulenotfounderror",
            "filenotfounderror",
            "syntaxerror",
            "exception",
        )
    ):
        score += 6
    if "timeout" in lower or "timed out" in lower:
        score += 5
    if "mismatch" in lower:
        score += 4
    if lower.startswith("traceback"):
        score -= 10
    if lower.startswith("stderr output"):
        score -= 2
    if "failed for" in lower:
        score -= 1
    return score


def extract_concise_error(raw_text: Any, *, max_len: int = 140) -> Optional[str]:
    text = strip_ansi(str(raw_text or "")).replace("\r", "\n")
    lines = []
    for raw_line in text.splitlines():
        line = WHITESPACE_RE.sub(" ", raw_line).strip()
        if not line or line.lower().startswith("traceback "):
            continue
        line = re.sub(r"^\[(?:ERROR|TIMEOUT|EXCEPTION|MISMATCH)\]\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^[A-Z_ ]+ failed for [^:]+:\s*", "", line, flags=re.IGNORECASE)
        lines.append(line)

    if not lines:
        return None

    best_line = lines[0]
    best_score = _score_error_line(best_line)
    for line in lines[1:]:
        score = _score_error_line(line)
        if score > best_score:
            best_line = line
            best_score = score

    return _truncate(best_line, max_len)


def summarize_failure_exception(stage: str, exc: Exception) -> tuple[str, Optional[str]]:
    stage_label = _stage_label(stage)

    if isinstance(exc, subprocess.TimeoutExpired):
        timeout_s = getattr(exc, "timeout", None)
        detail = f"timed out after {timeout_s}s" if timeout_s is not None else "timed out"
        return f"{stage_label} timeout", detail

    raw_message = str(exc)
    detail = extract_concise_error(raw_message)
    lowered = raw_message.lower()

    if "candidate perf is zero" in lowered:
        return "candidate perf is zero", detail
    if "perf payload missing" in lowered:
        return "perf payload missing", detail
    if "pytorch_functional_code is required" in lowered:
        return "missing pytorch functional code", detail
    if "timeout" in lowered or "timed out" in lowered:
        return f"{stage_label} timeout", detail
    if "mismatch" in lowered or "results differ" in lowered:
        return "result mismatch", detail
    return f"{stage_label} failed", detail


def failure_category(
    compile_ok: bool,
    run_ok: bool,
    match_ok: bool,
    timing: Optional[Mapping[str, Any]] = None,
    reason: Optional[str] = None,
) -> str:
    timing = timing or {}
    reason_label = _normalize_text(timing.get("failure_reason"))
    if not reason_label and reason:
        reason_label = _normalize_text(reason).split(":", 1)[0]
    if reason_label:
        return _truncate(reason_label, 80)
    if not compile_ok:
        return "compilation/reference failed"
    if not run_ok:
        return "test run failed"
    if not match_ok:
        return "result mismatch"
    return "evaluation failed"


def derive_failure_reason(
    compile_ok: bool,
    run_ok: bool,
    match_ok: bool,
    timing: Optional[Mapping[str, Any]] = None,
    reason: Optional[str] = None,
) -> Optional[str]:
    if compile_ok and run_ok and match_ok:
        return None

    timing = timing or {}
    combined_reason = _normalize_text(reason) if reason else ""
    if combined_reason:
        return _truncate(combined_reason, 180)

    reason_label = _normalize_text(timing.get("failure_reason"))
    detail = _normalize_text(timing.get("failure_detail"))
    if reason_label and detail and detail.lower() not in reason_label.lower():
        return _truncate(f"{reason_label}: {detail}", 180)
    if reason_label:
        return _truncate(reason_label, 180)
    if detail:
        return _truncate(detail, 180)
    return failure_category(compile_ok, run_ok, match_ok, timing)


def format_kernel_success(
    kernel_name: str,
    *,
    speedup: float,
    timing: Optional[Mapping[str, Any]] = None,
) -> str:
    timing = timing or {}
    parts = [
        colorize("[PASS]", PASS_COLOR),
        kernel_name,
        f"speedup={format_speedup(speedup)}",
    ]
    cache_hits = _format_reference_cache_hits(timing, color=True)
    if cache_hits:
        parts.append(cache_hits)
    total = timing.get("total")
    if isinstance(total, (int, float)):
        parts.append(f"total={total:.2f}s")
    return " ".join(parts)


def format_kernel_failure(
    kernel_name: str,
    *,
    compile_ok: bool,
    run_ok: bool,
    match_ok: bool,
    timing: Optional[Mapping[str, Any]] = None,
    reason: Optional[str] = None,
) -> str:
    timing = timing or {}
    failure_reason = derive_failure_reason(compile_ok, run_ok, match_ok, timing, reason) or "evaluation failed"
    status = f"compile={'Y' if compile_ok else 'N'} run={'Y' if run_ok else 'N'} match={'Y' if match_ok else 'N'}"
    parts = [
        "[FAIL]",
        kernel_name,
        f"reason={failure_reason}",
        status,
    ]
    cache_hits = _format_reference_cache_hits(timing, color=False)
    if cache_hits:
        parts.append(cache_hits)
    total = timing.get("total")
    if isinstance(total, (int, float)):
        parts.append(f"total={total:.2f}s")
    return colorize(" ".join(parts), FAIL_COLOR)


def format_batch_request_footer(label: str, *, batch_size: int, total_time: float) -> str:
    avg = total_time / max(1, batch_size)
    return (
        f"{colorize('[BATCH]', SUMMARY_COLOR)} {label} "
        f"kernels={batch_size} wall={total_time:.2f}s avg={avg:.2f}s"
    )


def format_evaluation_summary(
    title: str,
    records: Sequence[Mapping[str, Any]],
    *,
    total_elapsed: float,
) -> str:
    total_tasks = len(records)
    success_records = []
    fail_counter: Counter[str] = Counter()
    compile_cache_hits = 0
    golden_cache_hits = 0
    perf_cache_hits = 0
    compile_build_samples = []
    golden_build_samples = []
    perf_build_samples = []

    for record in records:
        compile_ok = bool(record.get("compile_ok"))
        run_ok = bool(record.get("run_ok"))
        match_ok = bool(record.get("match_ok"))
        timing = record.get("timing") or {}
        if timing.get("reference_compile_cache_hit") is True:
            compile_cache_hits += 1
        if timing.get("reference_golden_cache_hit") is True:
            golden_cache_hits += 1
        if timing.get("reference_perf_cache_hit") is True:
            perf_cache_hits += 1
        for key, bucket in (
            ("reference_compile_build_s", compile_build_samples),
            ("reference_golden_build_s", golden_build_samples),
            ("reference_perf_build_s", perf_build_samples),
        ):
            value = timing.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bucket.append(float(value))
        if compile_ok and run_ok and match_ok:
            success_records.append(
                (
                    str(record.get("kernel_name") or "<unknown>"),
                    float(record.get("speedup") or 0.0),
                )
            )
        else:
            fail_counter[failure_category(
                compile_ok,
                run_ok,
                match_ok,
                record.get("timing"),
                record.get("reason"),
            )] += 1

    success_count = len(success_records)
    failed_count = total_tasks - success_count
    success_rate = (success_count / total_tasks) if total_tasks else 0.0
    avg_per_task = total_elapsed / total_tasks if total_tasks else 0.0

    lines = [
        colorize(f"[SUMMARY] {title}", SUMMARY_COLOR),
        f"  tasks        : {total_tasks}",
        f"  success      : {success_count} ({success_rate:.1%})",
        f"  failed       : {failed_count}",
        f"  total_time   : {total_elapsed:.2f}s",
        f"  avg_per_task : {avg_per_task:.2f}s",
    ]
    if total_tasks:
        lines.append(
            "  cache_hits   : "
            f"compile={compile_cache_hits}/{total_tasks}, "
            f"golden={golden_cache_hits}/{total_tasks}, "
            f"perf={perf_cache_hits}/{total_tasks}"
        )
    if compile_build_samples or golden_build_samples or perf_build_samples:
        lines.append(
            "  avg_ref_build: "
            f"compile={sum(compile_build_samples) / len(compile_build_samples):.2f}s "
            if compile_build_samples else "  avg_ref_build: compile=n/a "
        )
        lines[-1] += (
            f"golden={sum(golden_build_samples) / len(golden_build_samples):.2f}s "
            if golden_build_samples else "golden=n/a "
        )
        lines[-1] += (
            f"perf={sum(perf_build_samples) / len(perf_build_samples):.2f}s"
            if perf_build_samples else "perf=n/a"
        )

    if success_records:
        avg_speedup = sum(speedup for _, speedup in success_records) / success_count
        faster_count = sum(1 for _, speedup in success_records if speedup > 1.0)
        slower_count = success_count - faster_count
        best_kernel, best_speedup = max(success_records, key=lambda item: item[1])
        worst_kernel, worst_speedup = min(success_records, key=lambda item: item[1])
        lines.extend(
            [
                f"  avg_speedup  : {format_speedup(avg_speedup)}",
                f"  speedup_mix  : >1.0x={faster_count}, <=1.0x={slower_count}",
                f"  best_kernel  : {best_kernel} ({format_speedup(best_speedup)})",
                f"  worst_kernel : {worst_kernel} ({format_speedup(worst_speedup)})",
            ]
        )
    else:
        lines.append("  avg_speedup  : n/a")

    if failed_count:
        fail_summary = ", ".join(f"{label} x{count}" for label, count in fail_counter.most_common(3))
        lines.append(f"  fail_reasons : {fail_summary or 'unknown'}")

    return "\n".join(lines)
