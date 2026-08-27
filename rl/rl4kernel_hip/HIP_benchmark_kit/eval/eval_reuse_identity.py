# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Optional, Tuple

try:
    from .server_eval_adapter import (
        SERVER_INPROCESS_BACKEND,
        normalize_eval_backend,
        parse_hip_filename,
        read_text_file,
        reference_python_file,
    )
except ImportError:
    from server_eval_adapter import (
        SERVER_INPROCESS_BACKEND,
        normalize_eval_backend,
        parse_hip_filename,
        read_text_file,
        reference_python_file,
    )


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SERVER_ROOT = os.path.join(REPO_ROOT, "hip_kernel_evaluation_server")


def _ensure_server_path() -> None:
    if SERVER_ROOT not in sys.path:
        sys.path.insert(0, SERVER_ROOT)


def _server_helpers():
    _ensure_server_path()
    from sandbox_core.cache import sha256_text
    from sandbox_core.codegen import get_template_bundle_hash
    from sandbox_core.config import resolve_effective_arch

    return sha256_text, get_template_bundle_hash, resolve_effective_arch


def build_eval_identity(
    *,
    hip_code_dir: str,
    hip_file: str,
    pytorch_func_dir: str,
    pytorch_modu_dir: str,
    rtol: float,
    atol: float,
    perf_iterations: int,
    artifact_side: str,
    eval_backend: str,
    reference_hip_code_dir: Optional[str] = None,
    reference_cache_mode: str = "golden+compile",
) -> Dict[str, object]:
    sha256_text, get_template_bundle_hash, resolve_effective_arch = _server_helpers()
    normalized_backend = normalize_eval_backend(eval_backend)
    if normalized_backend != SERVER_INPROCESS_BACKEND:
        raise ValueError(f"Unsupported eval backend for identity: {eval_backend}")

    base_name, gen_idx = parse_hip_filename(hip_file)
    reference_dir = reference_hip_code_dir or hip_code_dir
    reference_hip_file = f"{base_name}.hip"
    payload: Dict[str, object] = {
        "hip_file": hip_file,
        "base_name": base_name,
        "gen_idx": gen_idx,
        "hip_source_sha256": sha256_text(read_text_file(os.path.join(hip_code_dir, hip_file))),
        "reference_hip_file": reference_hip_file,
        "reference_hip_source_sha256": sha256_text(
            read_text_file(os.path.join(reference_dir, reference_hip_file))
        ),
        "pytorch_functional_sha256": sha256_text(
            read_text_file(reference_python_file(pytorch_func_dir, base_name))
        ),
        "pytorch_module_sha256": sha256_text(read_text_file(reference_python_file(pytorch_modu_dir, base_name))),
        "rtol": float(rtol),
        "atol": float(atol),
        "perf_iterations": int(perf_iterations),
        "artifact_side": artifact_side,
        "eval_backend": SERVER_INPROCESS_BACKEND,
        "reference_cache_mode": reference_cache_mode,
        "template_bundle_sha256": get_template_bundle_hash(),
        "effective_arch": resolve_effective_arch(),
    }
    payload["eval_identity_hash"] = sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return payload


def annotate_eval_result(
    result: Dict[str, object],
    identity: Dict[str, object],
    *,
    reused_from: Optional[str] = None,
    reuse_validation: str = "",
) -> Dict[str, object]:
    annotated = dict(result)
    annotated.update(
        {
            "hip_source_sha256": identity["hip_source_sha256"],
            "reference_hip_file": identity["reference_hip_file"],
            "reference_hip_source_sha256": identity["reference_hip_source_sha256"],
            "pytorch_functional_sha256": identity["pytorch_functional_sha256"],
            "pytorch_module_sha256": identity["pytorch_module_sha256"],
            "eval_identity_hash": identity["eval_identity_hash"],
            "eval_backend": SERVER_INPROCESS_BACKEND,
            "reused_from": reused_from or "",
            "reuse_validation": reuse_validation,
        }
    )
    return annotated


def load_reusable_results(
    *,
    reuse_json: Optional[str],
    reuse_hip_code_dir: Optional[str],
    current_hip_code_dir: str,
    pytorch_func_dir: str,
    pytorch_modu_dir: str,
    rtol: float,
    atol: float,
    perf_iterations: int,
    artifact_side: str,
    eval_backend: str,
    reference_hip_code_dir: Optional[str] = None,
    reference_cache_mode: str = "golden+compile",
) -> Dict[Tuple[object, object], Dict[str, object]]:
    sha256_text, _, _ = _server_helpers()
    normalized_backend = normalize_eval_backend(eval_backend)
    if normalized_backend != SERVER_INPROCESS_BACKEND:
        raise ValueError(f"Unsupported eval backend for reuse: {eval_backend}")
    if not reuse_json or not reuse_hip_code_dir:
        return {}
    if not os.path.exists(reuse_json):
        print(f"[REUSE] prior results missing, skipping: {reuse_json}")
        return {}
    if not os.path.isdir(reuse_hip_code_dir):
        print(f"[REUSE] prior HIP dir missing, skipping: {reuse_hip_code_dir}")
        return {}

    with open(reuse_json, "r", encoding="utf-8") as handle:
        prior_rows = json.load(handle)
    reusable: Dict[Tuple[object, object], Dict[str, object]] = {}
    for row in prior_rows:
        hip_file = row.get("hip_file")
        if not hip_file:
            continue
        current_path = os.path.join(current_hip_code_dir, hip_file)
        prior_path = os.path.join(reuse_hip_code_dir, hip_file)
        if not os.path.isfile(current_path) or not os.path.isfile(prior_path):
            continue
        if sha256_text(read_text_file(current_path)) != sha256_text(read_text_file(prior_path)):
            continue

        prior_backend = row.get("eval_backend")
        if prior_backend and normalize_eval_backend(str(prior_backend)) != normalized_backend:
            continue

        identity = build_eval_identity(
            hip_code_dir=current_hip_code_dir,
            hip_file=hip_file,
            pytorch_func_dir=pytorch_func_dir,
            pytorch_modu_dir=pytorch_modu_dir,
            rtol=rtol,
            atol=atol,
            perf_iterations=perf_iterations,
            artifact_side=artifact_side,
            eval_backend=normalized_backend,
            reference_hip_code_dir=reference_hip_code_dir,
            reference_cache_mode=reference_cache_mode,
        )
        prior_identity = row.get("eval_identity_hash")
        if not prior_identity or prior_identity != identity["eval_identity_hash"]:
            continue
        key = (row.get("base_name"), row.get("gen_idx"))
        reusable[key] = annotate_eval_result(
            row,
            identity,
            reused_from=reuse_json,
            reuse_validation="identity_hash",
        )
    print(f"[REUSE] reusable prior eval rows: {len(reusable)} from {reuse_json}")
    return reusable
