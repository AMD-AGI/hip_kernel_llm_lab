# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Batch-optimized reward.py for veRL
支持批量调用HIP server API实现端到端加速
"""
from __future__ import annotations
import os
import json
import hashlib
import math
import socket
import time
import typing as T
import sys
from pathlib import Path
from typing import Optional, List
import requests

from dataset.contracts import (
    HIP_TRANSLATION_UNIT_CODE_UNIT,
    SUPPORTED_REWARD_TRAIN_DATA_SOURCES,
    OptimizationContract,
    resolve_optimization_contract,
)
from sandbox.client_adapter import (
    EvalRequest,
    EvalResponse,
    call_run_code,
)
from reward.utils import (
    parse_generation_response,
    replace_kernel_in_hip_code,
    maybe_read_text,
    compute_dtw_to_ref,
    get_adaptive_thresholds,
    extract_kernel_body,
)
from reward.kernel_novelty_tracker import get_global_tracker
from reward.reasoning_visualizer import (
    summarize_think_blocks,
    format_think_inspection_log,
    format_think_block_log,
)

SF_URL_ENV = os.environ.get("SF_URL", "").strip()
REWARD_EVAL_AUDIT_LOG_ENV = os.environ.get("REWARD_EVAL_AUDIT_LOG", "").strip()
ARCHIVE_VERSION = 1
LEGACY_REWARD_MODES = {"", "legacy", "legacy_default"}


def _supports_reward_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _reward_color(text: T.Any, *codes: str) -> str:
    rendered = str(text)
    if not codes or not _supports_reward_color():
        return rendered
    return f"\033[{';'.join(codes)}m{rendered}\033[0m"


def _reward_bool(label: str, value: bool) -> str:
    tone = ("1", "32") if value else ("1", "31")
    return f"{_reward_color(label, '36')}={_reward_color(value, *tone)}"


def _reward_metric(label: str, value: T.Any, tone: tuple[str, ...] = ("1", "97")) -> str:
    return f"{_reward_color(label, '36')}={_reward_color(value, *tone)}"


def _print_transient_reward_block(title: str, lines: List[str], title_tone: tuple[str, ...] = ("1", "95")) -> None:
    border = _reward_color("-" * 86, "2", "36")
    print(border)
    print(f"{_reward_color(title, *title_tone)}")
    for line in lines:
        print(f"  {line}")
    print(border)


def _get_env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except Exception:
        print(f"[REWARD WARN] Invalid float env {name}={v!r}, using default {default}")
        return default


def _coerce_optional_float(value: T.Any, source_name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        print(f"[REWARD WARN] Invalid float from {source_name}={value!r}, ignore it")
        return None


def _resolve_reward_float(
    explicit_value: T.Any,
    extra_info: Optional[dict],
    extra_keys: List[str],
    env_name: str,
    default: float,
) -> float:
    explicit = _coerce_optional_float(explicit_value, f"kwarg:{env_name}")
    if explicit is not None:
        return explicit

    extra = extra_info or {}
    for key in extra_keys:
        if key in extra:
            resolved = _coerce_optional_float(extra.get(key), f"extra_info:{key}")
            if resolved is not None:
                return resolved

    return _get_env_float(env_name, default)


def _resolve_reward_mode(explicit_mode: Optional[str], extra_info: Optional[dict]) -> str:
    if explicit_mode is not None and str(explicit_mode).strip():
        return str(explicit_mode).strip().lower()

    extra = extra_info or {}
    extra_mode = extra.get("reward_mode")
    if extra_mode is not None and str(extra_mode).strip():
        return str(extra_mode).strip().lower()

    return str(os.environ.get("REWARD_MODE", "")).strip().lower()


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _resolve_eval_audit_log_path(extra_infos: Optional[List[Optional[dict]]]) -> Optional[str]:
    for extra in extra_infos or []:
        if not extra:
            continue
        candidate = str(extra.get("reward_eval_audit_log") or extra.get("eval_audit_log") or "").strip()
        if candidate:
            return candidate
    return REWARD_EVAL_AUDIT_LOG_ENV or None


def _json_safe(value: T.Any) -> T.Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _append_jsonl_records(path: str, rows: List[dict]) -> None:
    if not path or not rows:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_json_safe(row), ensure_ascii=True, sort_keys=True) + "\n")
    except Exception as exc:
        print(f"[REWARD WARN] Failed to append evaluator audit log {path}: {exc}")


def _env_text(name: str) -> str:
    return str(os.environ.get(name, "")).strip()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _sha256_text(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_eval_archive_dir(extra_infos: Optional[List[Optional[dict]]]) -> Optional[str]:
    for extra in extra_infos or []:
        if not extra:
            continue
        candidate = str(extra.get("reward_eval_archive_dir") or extra.get("eval_archive_dir") or "").strip()
        if candidate:
            return candidate
    return _env_text("REWARD_EVAL_ARCHIVE_DIR") or None


def _resolve_archive_experiment_name(
    extra_infos: Optional[List[Optional[dict]]],
    archive_dir: Optional[str],
) -> str:
    for extra in extra_infos or []:
        if not extra:
            continue
        candidate = str(extra.get("reward_eval_experiment_name") or extra.get("experiment_name") or "").strip()
        if candidate:
            return candidate
    for env_name in ("REWARD_EVAL_EXPERIMENT_NAME", "EXPERIMENT_NAME", "WANDB_NAME"):
        candidate = _env_text(env_name)
        if candidate:
            return candidate
    if archive_dir:
        return Path(archive_dir).expanduser().name or "unknown_experiment"
    return "unknown_experiment"


def _resolve_archive_run_id(extra_infos: Optional[List[Optional[dict]]], experiment_name: str) -> str:
    for extra in extra_infos or []:
        if not extra:
            continue
        candidate = str(extra.get("reward_eval_run_id") or extra.get("run_id") or "").strip()
        if candidate:
            return candidate
    for env_name in ("REWARD_EVAL_RUN_ID", "WANDB_RUN_ID"):
        candidate = _env_text(env_name)
        if candidate:
            return candidate
    host = socket.gethostname()
    pid = os.getpid()
    experiment = experiment_name or "unknown_experiment"
    return f"{experiment}.{host}.{pid}"


def _resolve_archive_root(archive_dir: str, experiment_name: str) -> str:
    root = Path(archive_dir).expanduser()
    experiment = str(experiment_name or "").strip()
    if experiment and root.name != experiment:
        root = root / experiment
    return str(root)


def _build_archive_context(extra_infos: Optional[List[Optional[dict]]]) -> Optional[dict]:
    archive_dir = _resolve_eval_archive_dir(extra_infos)
    if not archive_dir:
        return None
    experiment_name = _resolve_archive_experiment_name(extra_infos, archive_dir)
    hostname = socket.gethostname()
    pid = os.getpid()
    return {
        "archive_version": ARCHIVE_VERSION,
        "archive_root": _resolve_archive_root(archive_dir, experiment_name),
        "experiment_name": experiment_name,
        "run_id": _resolve_archive_run_id(extra_infos, experiment_name),
        "hostname": hostname,
        "pid": pid,
        "include_raw_response": _env_flag("REWARD_EVAL_ARCHIVE_INCLUDE_RAW_RESPONSE", False),
    }


def _append_archive_records(archive_context: Optional[dict], rows: List[dict]) -> Optional[str]:
    if not archive_context or not rows:
        return None
    archive_root = str(archive_context["archive_root"])
    shard_path = os.path.join(
        archive_root,
        f"records.{archive_context['hostname']}.{archive_context['pid']}.jsonl",
    )
    _append_jsonl_records(shard_path, rows)
    return shard_path


def _build_archive_row(
    archive_context: dict,
    request_context: dict,
    *,
    compile_ok: bool,
    run_ok: bool,
    match_ok: bool,
    speedup: float,
    score: float,
    reason: str,
    server_total_time_s: float,
    timing: Optional[dict] = None,
) -> dict:
    timing = timing or {}
    raw_response = str(request_context.get("raw_response") or "")
    hip_code = str(request_context.get("hip_code") or "")
    row = {
        "archive_version": archive_context["archive_version"],
        "timestamp_epoch": time.time(),
        "run_id": archive_context["run_id"],
        "experiment_name": archive_context["experiment_name"],
        "hostname": archive_context["hostname"],
        "pid": archive_context["pid"],
        "train_step": request_context.get("train_step"),
        "prompt_uid": str(request_context.get("prompt_uid") or ""),
        "sample_index": request_context.get("sample_index"),
        "data_source": str(request_context.get("data_source") or ""),
        "kernel_name_base": str(request_context.get("kernel_name_base") or ""),
        "kernel_name": str(request_context.get("candidate_kernel_name") or request_context.get("kernel_name_base") or ""),
        "parse_ok": bool(request_context.get("parse_ok", True)),
        "parse_mode": str(request_context.get("parse_mode") or ""),
        "parse_attempt_chain": str(request_context.get("parse_attempt_chain") or ""),
        "attempted_parse_modes": list(request_context.get("attempted_parse_modes") or []),
        "parse_error": str(request_context.get("parse_error") or ""),
        "output_contract": str(request_context.get("output_contract") or ""),
        "expected_code_unit": str(request_context.get("expected_code_unit") or ""),
        "persistence_mode": str(request_context.get("persistence_mode") or ""),
        "hip_code": hip_code,
        "hip_code_sha256": _sha256_text(hip_code),
        "hip_code_num_chars": len(hip_code),
        "raw_response_sha256": _sha256_text(raw_response),
        "raw_response_num_chars": len(raw_response),
        "compile_ok": bool(compile_ok),
        "run_ok": bool(run_ok),
        "match_ok": bool(match_ok),
        "speedup": float(speedup or 0.0),
        "score": float(score),
        "reason": str(reason or ""),
        "server_total_time_s": float(server_total_time_s or 0.0),
        "reference_compile_cache_hit": bool(timing.get("reference_compile_cache_hit")),
        "reference_golden_cache_hit": bool(timing.get("reference_golden_cache_hit")),
        "reference_perf_cache_hit": bool(timing.get("reference_perf_cache_hit")),
        "reference_compile_build_s": float(timing.get("reference_compile_build_s") or 0.0),
        "reference_golden_build_s": float(timing.get("reference_golden_build_s") or 0.0),
        "reference_perf_build_s": float(timing.get("reference_perf_build_s") or 0.0),
        "candidate_perf_ms": float(timing.get("candidate_perf_ms") or 0.0),
        "reference_perf_ms": float(timing.get("reference_perf_ms") or 0.0),
    }
    if archive_context.get("include_raw_response"):
        row["raw_response"] = raw_response
    return row


def _summarize_eval_batch(responses: List[dict], server_total_time_s: float, train_steps: List[T.Any]) -> dict:
    success_count = 0
    compile_hits = 0
    golden_hits = 0
    perf_hits = 0
    for item in responses:
        timing = item.get("timing") or {}
        if item.get("compile_ok") and item.get("run_ok") and item.get("match_ok"):
            success_count += 1
        if timing.get("reference_compile_cache_hit") is True:
            compile_hits += 1
        if timing.get("reference_golden_cache_hit") is True:
            golden_hits += 1
        if timing.get("reference_perf_cache_hit") is True:
            perf_hits += 1
    return {
        "timestamp_epoch": time.time(),
        "train_steps": train_steps,
        "batch_size": len(responses),
        "success_count": success_count,
        "compile_cache_hits": compile_hits,
        "golden_cache_hits": golden_hits,
        "perf_cache_hits": perf_hits,
        "server_total_time_s": float(server_total_time_s or 0.0),
    }


def _require_run_code_url(url: Optional[str]) -> str:
    """
    校验端点URL（必须以/run_code结尾）
    
    设计说明：
    - 输入URL应以 /run_code 结尾（例如：http://localhost:8080/run_code）
    - 批量模式下会自动转换为 /run_code_batch
    - 这样可以统一配置，减少配置错误
    """
    u = (url or "").strip()
    if not u or not u.endswith("/run_code"):
        raise ValueError(f"Sandbox url must end with /run_code, got: {u!r}")
    return u


def call_batch_run_code(url: str, requests_data: List[dict], timeout_s: int = 600) -> requests.Response:
    """
    调用批量评估API
    
    Args:
        url: base URL（以/run_code结尾，会自动替换为/run_code_batch）
        requests_data: 批量请求数据列表
        timeout_s: 超时时间（秒）
        
    Returns:
        requests.Response: server响应
    """
    batch_url = url.replace("/run_code", "/run_code_batch")
    
    batch_payload = {
        "requests": requests_data
    }
    
    resp = requests.post(
        batch_url,
        json=batch_payload,
        timeout=timeout_s
    )
    return resp


def _parse_solution_response(
    *,
    data_source: str,
    raw_response: str,
    extra_info: Optional[dict],
    kernel_name: str,
    hip_ref: str,
    contract: Optional[OptimizationContract] = None,
) -> dict:
    resolved_contract = contract or resolve_optimization_contract(
        data_source=data_source,
        extra_info=extra_info or {},
    )
    return parse_generation_response(
        raw_response,
        data_source=data_source,
        kernel_name=kernel_name,
        hip_ref=hip_ref,
        output_contract=resolved_contract.output_contract,
        expected_code_unit=resolved_contract.expected_code_unit,
    )


def _materialize_eval_hip_code(
    *,
    contract: OptimizationContract,
    hip_ref: str,
    parsed_hip_src: str,
    kernel_name: str,
) -> str:
    if contract.requires_kernel_splice:
        if hip_ref:
            return replace_kernel_in_hip_code(hip_ref, parsed_hip_src, kernel_name=kernel_name)
        print("[WARN] kernel-splice mode but hip_ref is empty, using parsed HIP code")
    return parsed_hip_src


def _parse_attempt_chain(parse_result: Optional[dict]) -> str:
    result = parse_result or {}
    chain = str(result.get("parse_attempt_chain") or "").strip()
    if chain:
        return chain
    attempted_modes = result.get("attempted_parse_modes") or []
    if attempted_modes:
        return "->".join(str(mode) for mode in attempted_modes if mode)
    return str(result.get("parse_mode") or "").strip()


# ============================================================================
# 批量优化版本：用于veRL训练
# ============================================================================
def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[T.Any],
    extra_infos: List[Optional[dict]] = None,
    reward_mode: Optional[str] = None,
    reward_correct_speedup_r_ok: Optional[float] = None,
    reward_correct_speedup_cap: Optional[float] = None,
    reward_correct_speedup_copy_reward: Optional[float] = None,
) -> List[float]:
    """
    批量计算score，充分利用server侧并行编译
    
    Args:
        data_sources: 数据源列表
        solution_strs: HIP代码列表
        ground_truths: ground truth列表
        extra_infos: 额外信息列表
        
    Returns:
        scores: 分数列表
    """
    batch_size = len(data_sources)
    extra_infos = extra_infos or [{}] * batch_size
    # 初始化结果（默认0分）
    scores = [0.0] * batch_size
    audit_log_path = _resolve_eval_audit_log_path(extra_infos)
    archive_context = _build_archive_context(extra_infos)
    
    # 过滤出需要评估的样本
    valid_indices = []
    batch_requests = []
    request_contexts = []
    audit_rows: List[dict] = []
    archive_rows: List[dict] = []
    
    for i in range(batch_size):
        data_source = str(data_sources[i] or "").strip()
        
        # 只处理HIP相关数据源
        if data_source not in SUPPORTED_REWARD_TRAIN_DATA_SOURCES:
            continue
        
        extra = extra_infos[i] or {}
        
        # 解析端点
        try:
            sf_url = _require_run_code_url(extra.get("sandbox_url") or SF_URL_ENV)
        except Exception:
            continue
        
        # 解析参数
        code_root = extra.get("code_root")
        gt = ground_truths[i] if isinstance(ground_truths[i], dict) else {}
        current_train_step = extra.get("train_step")
        prompt_uid = extra.get("prompt_uid") or extra.get("uid")
        sample_index = extra.get("sample_index", extra.get("index", i))
        
        kernel_name_base = gt.get("kernel_name") or extra.get("kernel_name") or f"kernel_{i}"
        
        module_src = (
            gt.get("pytorch_module_code")
            or maybe_read_text(gt.get("pytorch_code_module"), code_root)
            or maybe_read_text(extra.get("pytorch_module_code"), code_root)
        )
        functional_src = (
            gt.get("pytorch_functional_code")
            or maybe_read_text(gt.get("pytorch_code_functional"), code_root)
            or maybe_read_text(extra.get("pytorch_functional_code"), code_root)
        )
        
        hip_ref = gt.get("hip_code") or ""
        raw_response = solution_strs[i]
        contract: Optional[OptimizationContract] = None
        try:
            contract = resolve_optimization_contract(data_source=data_source, extra_info=extra)
            parse_result = _parse_solution_response(
                data_source=data_source,
                raw_response=raw_response,
                extra_info=extra,
                kernel_name=kernel_name_base,
                hip_ref=hip_ref,
                contract=contract,
            )
        except Exception as exc:
            parse_result = {
                "hip_src": "",
                "parse_mode": "contract_resolution",
                "parse_ok": False,
                "parse_error": str(exc),
                "output_contract": str(extra.get("output_contract") or ""),
                "attempted_parse_modes": ["contract_resolution"],
                "parse_attempt_chain": "contract_resolution",
            }
        parse_attempt_chain = _parse_attempt_chain(parse_result)
        hip_src = parse_result["hip_src"]
        base_request_context = {
            "idx": i,
            "data_source": data_source,
            "train_step": current_train_step,
            "prompt_uid": prompt_uid,
            "sample_index": sample_index,
            "kernel_name_base": kernel_name_base,
            "candidate_kernel_name": kernel_name_base,
            "parse_ok": bool(parse_result["parse_ok"]),
            "parse_mode": parse_result["parse_mode"] or "",
            "parse_attempt_chain": parse_attempt_chain,
            "attempted_parse_modes": parse_result.get("attempted_parse_modes") or [],
            "parse_error": parse_result["parse_error"],
            "output_contract": parse_result["output_contract"],
            "expected_code_unit": contract.expected_code_unit if contract else str(extra.get("expected_code_unit") or ""),
            "persistence_mode": contract.persistence_mode if contract else str(extra.get("persistence_mode") or ""),
            "raw_response": raw_response,
            "hip_code": hip_src,
        }

        # 🔧 kernel-agent-react-train 模式：兼容 think+JSON 与 legacy fenced HIP
        if data_source == "kernel-agent-react-train":
            think_summary = summarize_think_blocks(raw_response)
            print(format_think_inspection_log(i, think_summary, hip_src))
            for block in think_summary["blocks"][:3]:
                print(format_think_block_log(i, block))
        if contract is not None and contract.requires_strict_parse_gate:
            print(
                "[REWARD PARSE] "
                f"index={i} data_source={data_source} "
                f"output_contract={parse_result['output_contract']} "
                f"expected_code_unit={contract.expected_code_unit} "
                f"parse_mode={parse_result['parse_mode'] or 'none'} "
                f"parse_attempt_chain={parse_attempt_chain or 'none'} "
                f"parse_ok={parse_result['parse_ok']}"
            )
            # 格式门控统一基于解析结果：解析失败直接给 0.0 并跳过评估
            if not parse_result["parse_ok"]:
                print(
                    "[REWARD WARN] "
                    f"candidate parse failed for index={i}, data_source={data_source}, "
                    f"output_contract={parse_result['output_contract']}, "
                    f"parse_mode={parse_result['parse_mode'] or 'none'}, "
                    f"parse_attempt_chain={parse_attempt_chain or 'none'}, "
                    f"error={parse_result['parse_error']}"
                )
                scores[i] = 0.0
                audit_rows.append(
                    {
                        "timestamp_epoch": time.time(),
                        "train_step": current_train_step,
                        "data_source": data_source,
                        "kernel_name": kernel_name_base,
                        "output_contract": parse_result["output_contract"],
                        "parse_mode": parse_result["parse_mode"] or "",
                        "parse_attempt_chain": parse_attempt_chain,
                        "attempted_parse_modes": parse_result.get("attempted_parse_modes") or [],
                        "parse_ok": False,
                        "parse_error": parse_result["parse_error"],
                        "compile_ok": False,
                        "run_ok": False,
                        "match_ok": False,
                        "speedup": 0.0,
                        "score": 0.0,
                        "server_total_time_s": 0.0,
                        "reason": "parse_failed",
                    }
                )
                if archive_context is not None:
                    archive_rows.append(
                        _build_archive_row(
                            archive_context,
                            base_request_context,
                            compile_ok=False,
                            run_ok=False,
                            match_ok=False,
                            speedup=0.0,
                            score=0.0,
                            reason="parse_failed",
                            server_total_time_s=0.0,
                        )
                    )
                continue
        elif contract is None or not parse_result["parse_ok"]:
            scores[i] = 0.0
            audit_rows.append(
                {
                    "timestamp_epoch": time.time(),
                    "train_step": current_train_step,
                    "data_source": data_source,
                    "kernel_name": kernel_name_base,
                    "output_contract": parse_result["output_contract"],
                    "parse_mode": parse_result["parse_mode"] or "",
                    "parse_attempt_chain": parse_attempt_chain,
                    "attempted_parse_modes": parse_result.get("attempted_parse_modes") or [],
                    "parse_ok": False,
                    "parse_error": parse_result["parse_error"],
                    "compile_ok": False,
                    "run_ok": False,
                    "match_ok": False,
                    "speedup": 0.0,
                    "score": 0.0,
                    "server_total_time_s": 0.0,
                    "reason": "contract_invalid" if contract is None else "parse_failed",
                }
            )
            if archive_context is not None:
                archive_rows.append(
                    _build_archive_row(
                        archive_context,
                        base_request_context,
                        compile_ok=False,
                        run_ok=False,
                        match_ok=False,
                        speedup=0.0,
                        score=0.0,
                        reason="contract_invalid" if contract is None else "parse_failed",
                        server_total_time_s=0.0,
                    )
                )
            continue

        hip_src = _materialize_eval_hip_code(
            contract=contract,
            hip_ref=hip_ref,
            parsed_hip_src=hip_src,
            kernel_name=kernel_name_base,
        )
        
        # 🔧 关键修复：添加代码hash避免PyTorch JIT缓存冲突
        code_hash = hashlib.md5(hip_src.encode()).hexdigest()[:8]
        kernel_name = f"{kernel_name_base}_{code_hash}"
        
        # 超时与容差（可由 extra_info 覆盖）
        atol = float(extra.get("atol", 1e-4))
        rtol = float(extra.get("rtol", 1e-3))
        compile_timeout_s = int(extra.get("compile_timeout_s", 600))
        run_timeout_s = int(extra.get("run_timeout_s", 120))
        
        # 构造请求
        batch_requests.append({
            "kernel_name": kernel_name,
            "hip_code": hip_src,
            "hip_ref_code": hip_ref,
            "pytorch_module_code": module_src or "",
            "pytorch_functional_code": functional_src or "",
            "atol": atol,
            "rtol": rtol,
            "compile_timeout_s": compile_timeout_s,
            "run_timeout_s": run_timeout_s,
        })
        valid_indices.append(i)
        request_contexts.append(
            {
                **base_request_context,
                "candidate_kernel_name": kernel_name,
                "hip_code": hip_src,
            }
        )
    
    if not batch_requests:
        _append_jsonl_records(audit_log_path, audit_rows)
        _append_archive_records(archive_context, archive_rows)
        return scores
    
    # 批量调用server
    try:
        # 从 extra_infos 获取超时设置，默认600秒
        # 注意：extra_infos 来自数据集，而不是 hydra 配置
        timeout_candidates = []
        for extra_info in extra_infos or []:
            if extra_info:
                timeout_candidates.append(int(extra_info.get("sf_timeout_s", 2400)))
        sf_timeout_s = max(timeout_candidates) if timeout_candidates else 2400
        
        print(f"[REWARD INFO] Using HTTP timeout: {sf_timeout_s}s")
        resp = call_batch_run_code(sf_url, batch_requests, timeout_s=sf_timeout_s)
        
        if resp.status_code != 200:
            # 批量失败，所有样本给负分
            print(f"[REWARD ERROR] HTTP status code: {resp.status_code}")
            print(f"[REWARD ERROR] Response: {resp.text[:500]}")  # 打印前500字符
            for ctx in request_contexts:
                idx = ctx["idx"]
                scores[idx] = 0.0      ##[NOTE-2026.03.23] 给所有样本0分，这样这一batch的response都会产生梯度
                current_extra_info = (extra_infos[idx] or {}) if extra_infos and idx < len(extra_infos) else {}
                audit_rows.append(
                    {
                        "timestamp_epoch": time.time(),
                        "train_step": current_extra_info.get("train_step"),
                        "data_source": data_sources[idx],
                        "kernel_name": ctx["kernel_name_base"],
                        "output_contract": ctx["output_contract"],
                        "parse_mode": ctx["parse_mode"],
                        "parse_attempt_chain": ctx["parse_attempt_chain"],
                        "attempted_parse_modes": ctx["attempted_parse_modes"],
                        "parse_ok": True,
                        "parse_error": ctx["parse_error"],
                        "compile_ok": False,
                        "run_ok": False,
                        "match_ok": False,
                        "speedup": 0.0,
                        "score": 0.0,
                        "server_total_time_s": 0.0,
                        "reason": f"http_status_{resp.status_code}",
                    }
                )
                if archive_context is not None:
                    archive_rows.append(
                        _build_archive_row(
                            archive_context,
                            ctx,
                            compile_ok=False,
                            run_ok=False,
                            match_ok=False,
                            speedup=0.0,
                            score=0.0,
                            reason=f"http_status_{resp.status_code}",
                            server_total_time_s=0.0,
                        )
                    )
            _append_jsonl_records(audit_log_path, audit_rows)
            _append_archive_records(archive_context, archive_rows)
            return scores
        
        # 解析批量响应
        resp_data = resp.json()
        
        # 分配结果
        # responses = resp_data.get("responses", [])
        responses = resp_data["responses"]
        server_total_time_s = float(resp_data.get("total_time") or 0.0)
        train_steps = sorted(
            {
                extra_infos[idx].get("train_step")
                for idx in valid_indices
                if extra_infos and idx < len(extra_infos) and extra_infos[idx] and extra_infos[idx].get("train_step") is not None
            }
        )
        batch_eval_summary = _summarize_eval_batch(responses, server_total_time_s, train_steps)
        print(f"[REWARD EVAL BATCH] {json.dumps(batch_eval_summary, ensure_ascii=True, sort_keys=True)}")
        for i, resp_item in enumerate(responses):
            if i >= len(request_contexts):
                break
            
            ctx = request_contexts[i]
            idx = ctx["idx"]
            
            # 获取对应的 request 数据用于计算 DTW
            req_data = batch_requests[i]
            hip_ref_code = req_data.get("hip_ref_code", "")
            hip_gen_code = req_data.get("hip_code", "")
            kernel_name = req_data.get("kernel_name", "").rsplit("_", 1)[0]  # 去掉 hash 后缀
            
            # 计算 DTW 距离（用于 copy detection / diagnostics）
            dtw_to_ref = 0.0
            token_len = 0
            if hip_ref_code and hip_gen_code:
                try:
                    dtw_to_ref, token_len = compute_dtw_to_ref(hip_ref_code, hip_gen_code, kernel_name)
                    _print_transient_reward_block(
                        title=f"[DTW] kernel={kernel_name}",
                        lines=[
                            " ".join(
                                [
                                    _reward_metric("dtw_to_ref", f"{dtw_to_ref:.4f}"),
                                    _reward_metric("token_len", token_len),
                                ]
                            )
                        ],
                        title_tone=("1", "96"),
                    )
                except Exception as e:
                    _print_transient_reward_block(
                        title="[DTW WARN] Failed to compute DTW",
                        lines=[_reward_metric("error", str(e), ("1", "31"))],
                        title_tone=("1", "31"),
                    )
                    dtw_to_ref = 0.0
                    token_len = 0
            
            # Select reward computation mode:
            # - legacy_default: original DTW novelty shaping + copy penalty logic
            # - soft_clip_novelty: gate + clip(speedup-1) + soft novelty regularizer
            # - correct_speedup_copy_penalty: correctness + speedup with binary copy penalty
            current_extra_info = extra_infos[idx] if extra_infos and idx < len(extra_infos) else None
            mode = _resolve_reward_mode(reward_mode, current_extra_info)
            if mode == "soft_clip_novelty":
                score, is_copy, novelty, thresholds = _compute_single_score_soft_clip_novelty(
                    resp_item, dtw_to_ref, token_len, return_details=True
                )
            elif mode == "correct_speedup_copy_penalty":
                r_ok = _resolve_reward_float(
                    reward_correct_speedup_r_ok,
                    current_extra_info,
                    ["reward_correct_speedup_r_ok", "REWARD_CORRECT_SPEEDUP_R_OK"],
                    "REWARD_CORRECT_SPEEDUP_R_OK",
                    0.3,
                )
                speedup_cap = max(
                    0.0,
                    _resolve_reward_float(
                        reward_correct_speedup_cap,
                        current_extra_info,
                        ["reward_correct_speedup_cap", "REWARD_CORRECT_SPEEDUP_CAP"],
                        "REWARD_CORRECT_SPEEDUP_CAP",
                        10.0,
                    ),
                )
                copy_reward = _resolve_reward_float(
                    reward_correct_speedup_copy_reward,
                    current_extra_info,
                    ["reward_correct_speedup_copy_reward", "REWARD_CORRECT_SPEEDUP_COPY_REWARD"],
                    "REWARD_CORRECT_SPEEDUP_COPY_REWARD",
                    0.0,
                )
                score, is_copy, novelty, thresholds = _compute_single_score_correct_speedup_copy_penalty(
                    resp_item,
                    dtw_to_ref,
                    token_len,
                    kernel_name=kernel_name,
                    r_ok=r_ok,
                    speedup_cap=speedup_cap,
                    copy_reward=copy_reward,
                    return_details=True,
                )
            elif mode in LEGACY_REWARD_MODES:
                score, is_copy, novelty, thresholds = _compute_single_score_with_novelty(
                    resp_item, dtw_to_ref, token_len, return_details=True
                )
            else:
                print(f"[REWARD WARN] Unknown REWARD_MODE={mode!r}; assigning score 0.0")
                score, is_copy, novelty, thresholds = (0.0, False, -1.0, (0.0, 0.0, 0.0))
            scores[idx] = score
            timing = resp_item.get("timing") or {}
            current_train_step = current_extra_info.get("train_step") if current_extra_info else None
            audit_rows.append(
                {
                    "timestamp_epoch": time.time(),
                    "train_step": current_train_step,
                    "data_source": data_sources[idx],
                    "kernel_name": kernel_name,
                    "output_contract": ctx["output_contract"],
                    "parse_mode": ctx["parse_mode"],
                    "parse_attempt_chain": ctx["parse_attempt_chain"],
                    "attempted_parse_modes": ctx["attempted_parse_modes"],
                    "parse_ok": True,
                    "parse_error": ctx["parse_error"],
                    "compile_ok": bool(resp_item.get("compile_ok")),
                    "run_ok": bool(resp_item.get("run_ok")),
                    "match_ok": bool(resp_item.get("match_ok")),
                    "speedup": float(resp_item.get("speedup") or 0.0),
                    "score": float(score),
                    "server_total_time_s": server_total_time_s,
                    "reference_compile_cache_hit": bool(timing.get("reference_compile_cache_hit")),
                    "reference_golden_cache_hit": bool(timing.get("reference_golden_cache_hit")),
                    "reference_perf_cache_hit": bool(timing.get("reference_perf_cache_hit")),
                    "reference_compile_build_s": float(timing.get("reference_compile_build_s") or 0.0),
                    "reference_golden_build_s": float(timing.get("reference_golden_build_s") or 0.0),
                    "reference_perf_build_s": float(timing.get("reference_perf_build_s") or 0.0),
                    "candidate_perf_ms": float(timing.get("candidate_perf_ms") or 0.0),
                    "reference_perf_ms": float(timing.get("reference_perf_ms") or 0.0),
                }
            )
            if archive_context is not None:
                archive_rows.append(
                    _build_archive_row(
                        archive_context,
                        ctx,
                        compile_ok=bool(resp_item.get("compile_ok")),
                        run_ok=bool(resp_item.get("run_ok")),
                        match_ok=bool(resp_item.get("match_ok")),
                        speedup=float(resp_item.get("speedup") or 0.0),
                        score=float(score),
                        reason=str(resp_item.get("reason") or ""),
                        server_total_time_s=server_total_time_s,
                        timing=timing,
                    )
                )
            
            # 记录到 Kernel Novelty Tracker（用于观测重复性）
            try:
                tracker = get_global_tracker()
                # 获取当前训练 step（如果在 extra_info 中提供）
                train_step = (extra_infos[idx] or {}).get("train_step", 0) if extra_infos and idx < len(extra_infos) else 0
                # 只记录 kernel function body，而不是整个 HIP 代码
                ref_kernel_body = extract_kernel_body(hip_ref_code, kernel_name)
                gen_kernel_body = extract_kernel_body(hip_gen_code, kernel_name)
                tracker.record(
                    kernel_name=kernel_name,
                    dtw_distance=dtw_to_ref,
                    token_len=token_len,
                    is_copy=is_copy,
                    novelty=novelty,
                    thresholds=thresholds,
                    ref_code=ref_kernel_body,
                    gen_code=gen_kernel_body,
                    reward=score,
                    step=train_step,
                    extra_info={"data_source": data_sources[idx], "idx": idx}
                )
            except Exception as e:
                print(f"[WARN] Failed to record to novelty tracker: {e}")
        _append_jsonl_records(audit_log_path, audit_rows)
        _append_archive_records(archive_context, archive_rows)
            
    except Exception as e:
        # 批量请求失败，给所有有效样本负分
        import traceback
        print(f"[REWARD EXCEPTION] Batch evaluation failed!")
        print(f"[REWARD EXCEPTION] Error type: {type(e).__name__}")
        print(f"[REWARD EXCEPTION] Error message: {e}")
        print(f"[REWARD EXCEPTION] Traceback:\n{traceback.format_exc()}")
        for ctx in request_contexts:
            idx = ctx["idx"]
            scores[idx] = 0.0
            current_extra_info = (extra_infos[idx] or {}) if extra_infos and idx < len(extra_infos) else {}
            audit_rows.append(
                {
                    "timestamp_epoch": time.time(),
                    "train_step": current_extra_info.get("train_step"),
                    "data_source": data_sources[idx],
                    "kernel_name": ctx["kernel_name_base"],
                    "output_contract": ctx["output_contract"],
                    "parse_mode": ctx["parse_mode"],
                    "parse_attempt_chain": ctx["parse_attempt_chain"],
                    "attempted_parse_modes": ctx["attempted_parse_modes"],
                    "parse_ok": True,
                    "parse_error": ctx["parse_error"],
                    "compile_ok": False,
                    "run_ok": False,
                    "match_ok": False,
                    "speedup": 0.0,
                    "score": 0.0,
                    "server_total_time_s": 0.0,
                    "reason": f"batch_exception:{type(e).__name__}",
                }
            )
            if archive_context is not None:
                archive_rows.append(
                    _build_archive_row(
                        archive_context,
                        ctx,
                        compile_ok=False,
                        run_ok=False,
                        match_ok=False,
                        speedup=0.0,
                        score=0.0,
                        reason=f"batch_exception:{type(e).__name__}",
                        server_total_time_s=0.0,
                    )
                )
        _append_jsonl_records(audit_log_path, audit_rows)
        _append_archive_records(archive_context, archive_rows)
    
    return scores


def _compute_single_score(resp_data: dict) -> float:
    """从响应数据计算单个分数（原版本，不含 novelty）"""
    S_REF = 100.0  # 可按数据分布调整

    # 门槛：编译/运行失败 → 负分；数值不匹配 → 0 分（可按需改为轻微负分）
    if not resp_data.get("compile_ok", False):
        return -0.9
    if not resp_data.get("run_ok", False):
        return -0.5
    if not resp_data.get("match_ok", False):
        return 0.0

    speedup = max(0.0, float(resp_data.get("speedup") or 0.0))
    if S_REF > 0:
        gain = math.log1p(speedup) / math.log1p(S_REF)
        gain = min(max(gain, 0.0), 1.0)
    else:
        gain = 0.0

    return 0.5 + gain  # ∈ [0.5, 1.5]


# ============================================================================
# DTW-based Novelty Reward Shaping
# ============================================================================
#
# 整体 Pipeline:
#
#            ┌─────────────────────────┐
#            │ rollout: 生成代码 + 评测 │
#            └────────────┬────────────┘
#                         │
#                         ▼
#         ┌──────────────────────────────────┐
#         │ 1. 硬门槛：compile/run/match gate │
#         └────────────┬─────────────────────┘
#                      │  (不满足 → 直接给负/零分)
#                      ▼
#         ┌──────────────────────────────────┐
#         │ 2. 计算 base 性能分：速度 speedup  │
#         └────────────┬─────────────────────┘
#                      │
#                      ▼
#         ┌──────────────────────────────────┐
#         │ 3. 相似度分析：DTW(ref, gen)      │
#         │    → d_ref, novelty ∈ [0,1]      │
#         └────────────┬─────────────────────┘
#                      │
#                      ▼
#         ┌──────────────────────────────────┐
#         │ 4. 复制惩罚 & 多样性 shaping       │
#         │    - 完全复制：硬惩罚             │
#         │    - 非复制：base + diversity    │
#         └────────────┬─────────────────────┘
#                      │
#                      ▼
#         ┌──────────────────────────────────┐
#         │         5. 输出最终 reward        │
#         └──────────────────────────────────┘
#
# ============================================================================

def compute_novelty_from_dtw(d_ref: float,
                             d_low: float = 0.05,
                             d_high: float = 0.20) -> float:
    """
    从 DTW 距离计算 novelty 值
    
    映射规则：
    - d_ref <= d_low (0.05): 非常像原文 → novelty = 0.0
    - d_ref >= d_high (0.20): 结构明显不同 → novelty = 1.0
    - d_low < d_ref < d_high: 线性插值
    
    注意：完全复制（d_ref < d_copy）的情况在调用方单独处理，不经过此函数
    
    Args:
        d_ref: 归一化 DTW 距离 ∈ [0, 1]
        d_low: 低差异阈值（默认 0.05）
        d_high: 高差异阈值（默认 0.20）
        
    Returns:
        novelty ∈ [0, 1]
    """
    if d_ref <= d_low:
        return 0.0
    if d_ref >= d_high:
        return 1.0
    # 线性插值
    return (d_ref - d_low) / (d_high - d_low)


def compute_diversity_shaping(base: float, 
                              d_ref: float, 
                              token_len: int,
                              lambda_coef: float = 0.25,
                              copy_penalty: float = -0.2) -> tuple:
    """
    计算多样性 shaping 后的最终 reward
    
    逻辑：
    1. 根据 token_len 获取自适应阈值 (d_copy, d_low, d_high)
    2. 如果 d_ref < d_copy：完全复制 → 返回 copy_penalty（忽略 base）
    3. 否则：计算 novelty，返回 base + diversity_bonus
    
    Args:
        base: 基于 speedup 的 base reward ∈ [0.5, 1.5]
        d_ref: 归一化 DTW 距离 ∈ [0, 1]
        token_len: kernel 的 token 数量
        lambda_coef: diversity bonus 系数（默认 0.25）
        copy_penalty: 完全复制的惩罚分数（默认 -0.2）
        
    Returns:
        (final_reward, is_copy, novelty, diversity_bonus, thresholds):
        - final_reward: 最终 reward
        - is_copy: 是否检测为完全复制
        - novelty: novelty 值 ∈ [0, 1]（如果 is_copy=True 则为 -1）
        - diversity_bonus: 多样性加成（如果 is_copy=True 则为 0）
        - thresholds: (d_copy, d_low, d_high) 使用的阈值
    """
    # 获取自适应阈值
    if token_len > 0:
        d_copy, d_low, d_high = get_adaptive_thresholds(token_len)
    else:
        d_copy, d_low, d_high = 0.08, 0.10, 0.20
    
    thresholds = (d_copy, d_low, d_high)
    
    # 完全复制检测
    if d_ref < d_copy:
        return (copy_penalty, True, -1.0, 0.0, thresholds)
    
    # 非完全复制：计算 novelty 和 diversity bonus
    novelty = compute_novelty_from_dtw(d_ref, d_low, d_high)  # ∈ [0, 1]
    diversity_bonus = lambda_coef * (novelty - 0.5)          # ∈ [-λ/2, +λ/2]
    final_reward = base + diversity_bonus
    
    return (final_reward, False, novelty, diversity_bonus, thresholds)


def _compute_single_score_with_novelty(resp_data: dict,
                                       dtw_to_ref: float = 0.0,
                                       token_len: int = 0,
                                       return_details: bool = False):
    """
    带 novelty reward shaping 的评分函数（支持自适应阈值）
    
    Reward Pipeline:
    ================
    1. 硬门槛 Gate（优先级最高）
       - compile_ok == False → -0.9
       - run_ok == False → -0.5
       - match_ok == False → 0.0
       
    2. Base 性能分（只看 speedup）
       - base = 0.5 + normalized_log_speedup ∈ [0.5, 1.5]
       
    3. 相似度分析 & 多样性 Shaping（自适应阈值）
       - d_ref < d_copy: 完全复制 → 直接返回 -0.2（硬惩罚）
       - d_ref >= d_copy: base + diversity_bonus
         - novelty = compute_novelty_from_dtw(d_ref) ∈ [0, 1]
         - diversity_bonus = λ * (novelty - 0.5) ∈ [-0.125, +0.125]
    
    Args:
        resp_data: 服务器返回的评估结果
        dtw_to_ref: 生成代码与参考代码的 DTW 距离 ∈ [0, 1]
        token_len: kernel 的 token 数量（用于自适应阈值）
        return_details: 是否返回详细信息（用于 tracker）
        
    Returns:
        如果 return_details=False: reward ∈ [-0.9, ~1.625]
        如果 return_details=True: (reward, is_copy, novelty, thresholds)
    """
    # ========== 超参数配置 ==========
    S_REF = 100.0       # speedup 归一化参考值
    LAMBDA = 0.25       # diversity bonus 系数
    COPY_PENALTY = -0.2 # 完全复制的惩罚分数
    
    # 默认的 thresholds（用于 gate 失败时返回）
    default_thresholds = (0.08, 0.10, 0.20)
    
    # ========== 1. 硬门槛 Gate ==========
    if not resp_data.get("compile_ok", False):
        if return_details:
            return (-0.9, False, -1.0, default_thresholds)
        return -0.9
    if not resp_data.get("run_ok", False):
        if return_details:
            return (-0.5, False, -1.0, default_thresholds)
        return -0.5
    if not resp_data.get("match_ok", False):
        if return_details:
            return (0.0, False, -1.0, default_thresholds)
        return 0.0
    
    # ========== 2. Base 性能分（speedup） ==========
    speedup = max(0.0, float(resp_data.get("speedup") or 0.0))
    if S_REF > 0:
        gain = math.log1p(speedup) / math.log1p(S_REF)
        gain = min(max(gain, 0.0), 1.0)
    else:
        gain = 0.0
    base = 0.5 + gain  # ∈ [0.5, 1.5]
    
    # ========== 3. 相似度分析 & 多样性 Shaping ==========
    d_ref = float(dtw_to_ref)
    final_reward, is_copy, novelty, diversity_bonus, thresholds = compute_diversity_shaping(
        base=base,
        d_ref=d_ref,
        token_len=token_len,
        lambda_coef=LAMBDA,
        copy_penalty=COPY_PENALTY
    )
    
    # 日志输出
    d_copy, d_low, d_high = thresholds
    if is_copy:
        print(f"[NOVELTY] Copy detected (d_ref={d_ref:.4f} < d_copy={d_copy:.4f}, "
              f"token_len={token_len}), forcing reward={COPY_PENALTY}")
    else:
        print(f"[NOVELTY] d_ref={d_ref:.4f}, token_len={token_len}, "
              f"thresholds=({d_copy:.3f},{d_low:.3f},{d_high:.3f}), "
              f"novelty={novelty:.4f}, base={base:.4f}, bonus={diversity_bonus:.4f}, "
              f"final={final_reward:.4f}")
    
    if return_details:
        return (final_reward, is_copy, novelty, thresholds)
    return final_reward


def _compute_single_score_soft_clip_novelty(
    resp_data: dict,
    dtw_to_ref: float = 0.0,
    token_len: int = 0,
    return_details: bool = False,
):
    """
    用户指定的 reward（可通过 REWARD_MODE=soft_clip_novelty 启用）：
    
    Gate:
      - !compile_ok -> -0.9
      - !run_ok     -> -0.7
      - !match_ok   -> -0.3
    
    Match OK:
      - s = max(eps, speedup)
      - r_perf = clip(s-1, -a, b)
      - R_base = r_ok + beta * r_perf
      - novelty ∈ [0,1] from DTW (adaptive thresholds)
      - r_nov = novelty - 0.5
      - R_final = R_base + alpha * r_nov
    
    Env hyperparams (defaults):
      - REWARD_SOFT_EPS=1e-6
      - REWARD_SOFT_A=1
      - REWARD_SOFT_B=9
      - REWARD_SOFT_R_OK=0.3
      - REWARD_SOFT_BETA=0.5
      - REWARD_SOFT_ALPHA=0.5
    """
    default_thresholds = (0.08, 0.10, 0.20)
    if not resp_data.get("compile_ok", False):
        if return_details:
            return (-0.9, False, -1.0, default_thresholds)
        return -0.9
    if not resp_data.get("run_ok", False):
        if return_details:
            return (-0.7, False, -1.0, default_thresholds)
        return -0.7
    if not resp_data.get("match_ok", False):
        if return_details:
            return (-0.3, False, -1.0, default_thresholds)
        return -0.3

    eps = _get_env_float("REWARD_SOFT_EPS", 1e-6)
    a = _get_env_float("REWARD_SOFT_A", 1.0)
    b = _get_env_float("REWARD_SOFT_B", 9.0)
    r_ok = _get_env_float("REWARD_SOFT_R_OK", 0.3)
    beta = _get_env_float("REWARD_SOFT_BETA", 0.5)
    alpha = _get_env_float("REWARD_SOFT_ALPHA", 0.5)

    speedup = float(resp_data.get("speedup") or 0.0)
    s = max(float(eps), speedup)
    r_perf = _clip(s - 1.0, -float(a), float(b))
    r_base = float(r_ok) + float(beta) * float(r_perf)

    # novelty ∈ [0,1] from DTW with adaptive thresholds; no hard copy penalty in this mode
    if token_len > 0:
        d_copy, d_low, d_high = get_adaptive_thresholds(token_len)
    else:
        d_copy, d_low, d_high = default_thresholds
    thresholds = (d_copy, d_low, d_high)
    d_ref = float(dtw_to_ref)
    novelty = compute_novelty_from_dtw(d_ref, d_low=d_low, d_high=d_high)
    is_copy = d_ref < d_copy

    r_nov = novelty - 0.5
    r_final = r_base + float(alpha) * float(r_nov)

    # Copy cap: don't allow copy cases to score higher than -0.2.
    # Important: use min() (cap), not overwrite, to avoid "rescuing" already-bad rewards.
    copy_cap = -0.2
    if is_copy:
        r_final = min(r_final, copy_cap)

    print(
        f"[REWARD soft_clip_novelty] speedup={speedup:.4f}, s={s:.4f}, "
        f"r_perf={r_perf:.4f}, r_base={r_base:.4f}, "
        f"d_ref={d_ref:.4f}, novelty={novelty:.4f}, is_copy={is_copy}, "
        f"alpha={alpha}, beta={beta}, r_ok={r_ok}, a={a}, b={b}, eps={eps}, "
        f"final={r_final:.4f}, copy_cap={copy_cap if is_copy else 'n/a'}"
    )

    if return_details:
        return (r_final, is_copy, novelty, thresholds)
    return r_final


def _compute_single_score_correct_speedup_copy_penalty(
    resp_data: dict,
    dtw_to_ref: float = 0.0,
    token_len: int = 0,
    kernel_name: str = "",
    r_ok: Optional[float] = None,
    speedup_cap: Optional[float] = None,
    copy_reward: Optional[float] = None,
    return_details: bool = False,
):
    """
    论文风格 reward（correctness + speedup）并加入 binary copy penalty：

    Gate:
      - !compile_ok -> 0.0
      - !run_ok     -> 0.0
      - !match_ok   -> 0.0

    Match OK:
      - speedup_eff = clip(speedup, 0, speedup_cap)
      - speedup_bonus = bonus if speedup > bonus_threshold else 0
      - if is_copy: reward = copy_reward
      - else:       reward = r_ok + speedup_eff + speedup_bonus

    Env hyperparams (defaults):
      - REWARD_CORRECT_SPEEDUP_R_OK=0.3
      - REWARD_CORRECT_SPEEDUP_CAP=10
      - REWARD_CORRECT_SPEEDUP_COPY_REWARD=0.0
      - REWARD_CORRECT_SPEEDUP_BONUS=0.3
      - REWARD_CORRECT_SPEEDUP_BONUS_THRESHOLD=1.1
    """
    default_thresholds = (0.08, 0.10, 0.20)
    compile_ok = bool(resp_data.get("compile_ok", False))
    run_ok = bool(resp_data.get("run_ok", False))
    match_ok = bool(resp_data.get("match_ok", False))

    r_ok = _get_env_float("REWARD_CORRECT_SPEEDUP_R_OK", 0.3) if r_ok is None else float(r_ok)
    speedup_cap = max(
        0.0,
        _get_env_float("REWARD_CORRECT_SPEEDUP_CAP", 10.0) if speedup_cap is None else float(speedup_cap),
    )
    copy_reward = (
        _get_env_float("REWARD_CORRECT_SPEEDUP_COPY_REWARD", 0.0)
        if copy_reward is None
        else float(copy_reward)
    )
    speedup_bonus = _get_env_float("REWARD_CORRECT_SPEEDUP_BONUS", 0.3)
    bonus_threshold = _get_env_float("REWARD_CORRECT_SPEEDUP_BONUS_THRESHOLD", 1.1)

    speedup = max(0.0, float(resp_data.get("speedup") or 0.0))
    speedup_eff = _clip(speedup, 0.0, speedup_cap)

    if token_len > 0:
        d_copy, d_low, d_high = get_adaptive_thresholds(token_len)
    else:
        d_copy, d_low, d_high = default_thresholds
    thresholds = (d_copy, d_low, d_high)

    d_ref = float(dtw_to_ref)
    copy_candidate = d_ref < d_copy

    if not compile_ok:
        r_final = 0.0
        is_copy = False
        novelty = -1.0
        decision = "compile_fail"
    elif not run_ok:
        r_final = 0.0
        is_copy = False
        novelty = -1.0
        decision = "run_fail"
    elif not match_ok:
        r_final = 0.0
        is_copy = False
        novelty = -1.0
        decision = "match_fail"
    else:
        is_copy = copy_candidate
        novelty = -1.0 if is_copy else compute_novelty_from_dtw(d_ref, d_low=d_low, d_high=d_high)
        if is_copy:
            r_final = copy_reward
            decision = "copy_penalty"
        else:
            bonus_applied = float(speedup_bonus) != 0.0 and float(speedup) > float(bonus_threshold)
            r_final = float(r_ok) + float(speedup_eff)
            if bonus_applied:
                r_final += float(speedup_bonus)
                decision = "correct_speedup_bonus"
            else:
                decision = "correct_speedup"

    kernel_label = kernel_name or "unknown_kernel"
    bonus_applied_flag = bool(not is_copy and match_ok and speedup_bonus != 0.0 and speedup > bonus_threshold)
    _print_transient_reward_block(
        title=f"[REWARD correct_speedup_copy_penalty] kernel={kernel_label}",
        lines=[
            "gate : "
            + " ".join(
                [
                    _reward_bool("compile_ok", compile_ok),
                    _reward_bool("run_ok", run_ok),
                    _reward_bool("match_ok", match_ok),
                    _reward_metric("decision", decision, ("1", "93")),
                ]
            ),
            "perf : "
            + " ".join(
                [
                    _reward_metric("speedup", f"{speedup:.4f}"),
                    _reward_metric("speedup_eff", f"{speedup_eff:.4f}"),
                    _reward_metric("speedup_cap", f"{speedup_cap:.4f}"),
                    _reward_metric("r_ok", f"{r_ok:.4f}"),
                    _reward_metric("bonus", f"{speedup_bonus:.4f}"),
                    _reward_metric("bonus_threshold", f"{bonus_threshold:.4f}"),
                    _reward_bool("bonus_applied", bonus_applied_flag),
                ]
            ),
            "copy : "
            + " ".join(
                [
                    _reward_metric("d_ref", f"{d_ref:.4f}"),
                    _reward_metric("d_copy", f"{d_copy:.4f}"),
                    _reward_metric("token_len", token_len),
                    _reward_bool("is_copy", is_copy),
                    _reward_metric("copy_reward", f"{copy_reward:.4f}"),
                ]
            ),
            "final: "
            + _reward_metric(
                "reward",
                f"{r_final:.4f}",
                ("1", "31") if r_final <= 0.0 else ("1", "32"),
            ),
        ],
        title_tone=("1", "95"),
    )

    if return_details:
        return (r_final, is_copy, novelty, thresholds)
    return r_final


# ============================================================================
# 单样本版本：兼容性
# ============================================================================
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: T.Any,
    extra_info: Optional[dict] = None,
    reward_mode: Optional[str] = None,
    reward_correct_speedup_r_ok: Optional[float] = None,
    reward_correct_speedup_cap: Optional[float] = None,
    reward_correct_speedup_copy_reward: Optional[float] = None,
) -> float:
    """单样本计算（兼容旧接口）"""
    scores = compute_score_batch(
        data_sources=[data_source],
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        extra_infos=[extra_info],
        reward_mode=reward_mode,
        reward_correct_speedup_r_ok=reward_correct_speedup_r_ok,
        reward_correct_speedup_cap=reward_correct_speedup_cap,
        reward_correct_speedup_copy_reward=reward_correct_speedup_copy_reward,
    )
    return scores[0]


# ============================================================================
# 测试
# ============================================================================
def main():
    """测试批量评估"""
    from unittest.mock import patch, Mock
    
    print("=" * 60)
    print("Testing Batch HIP Kernel Evaluation")
    print("=" * 60)
    
    # 准备测试数据
    test_data_sources = ["hip2hip-train"] * 3
    test_solutions = [
        "#include <hip/hip_runtime.h>\n__global__ void kernel1() {}",
        "#include <hip/hip_runtime.h>\n__global__ void kernel2() {}",
        "#include <hip/hip_runtime.h>\n__global__ void kernel3() {}",
    ]
    test_ground_truths = [
        {
            "kernel_name": f"kernel{i}",
            "hip_code": test_solutions[i],
            "pytorch_module_code": "import torch",
            "pytorch_functional_code": "import torch.nn.functional as F",
        }
        for i in range(3)
    ]
    test_extra_infos = [
        {"sandbox_url": "http://mock:8000/run_code"}
        for _ in range(3)
    ]
    
    # Mock批量响应（匹配 server 端 BatchEvalResponse 格式）
    mock_batch_response = {
        "responses": [
            {"compile_ok": True, "run_ok": True, "match_ok": True, "speedup": 1.5},
            {"compile_ok": True, "run_ok": False, "match_ok": False, "speedup": 0.0},
            {"compile_ok": False, "run_ok": False, "match_ok": False, "speedup": 0.0},
        ],
        "total_time": 10.0,
        "batch_size": 3
    }
    
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_batch_response
    
    with patch('reward_batch.call_batch_run_code', return_value=mock_resp):
        scores = compute_score_batch(
            data_sources=test_data_sources,
            solution_strs=test_solutions,
            ground_truths=test_ground_truths,
            extra_infos=test_extra_infos
        )
        
        print(f"\nBatch scores: {scores}")
        print(f"Expected: [~1.1, -0.5, -0.9]")
        print(f"Match: {abs(scores[0] - 1.1) < 0.1 and scores[1] == -0.5 and scores[2] == -0.9}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
