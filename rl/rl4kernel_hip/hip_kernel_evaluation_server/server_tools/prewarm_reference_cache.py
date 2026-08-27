#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from sandbox_core.config import load_eval_settings
from sandbox_core.eval import prewarm_reference_artifacts
from sandbox_core.reference import build_reference_keys
from sandbox_core.protocol import EvalRequest
from sandbox_core.cache import sha256_text


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            try:
                loaded = ast.literal_eval(text)
                return loaded if isinstance(loaded, dict) else {}
            except Exception:
                return {}
    return {}


def _iter_records_from_file(path: Path) -> Iterator[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield _coerce_dict(json.loads(line))
        return
    if suffix == '.json':
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for item in payload:
                yield _coerce_dict(item)
        elif isinstance(payload, dict):
            yield payload
        return
    if suffix == '.parquet':
        import pandas as pd

        frame = pd.read_parquet(path)
        for record in frame.to_dict(orient='records'):
            yield record
        return
    raise ValueError(f'Unsupported input format: {path}')


def _extract_request(record: Dict[str, Any]) -> Optional[EvalRequest]:
    reward_model = _coerce_dict(record.get('reward_model'))
    ground_truth = _coerce_dict(reward_model.get('ground_truth'))
    extra_info = _coerce_dict(record.get('extra_info'))

    kernel_name = ground_truth.get('kernel_name') or extra_info.get('kernel_name') or 'unknown'
    hip_code = ground_truth.get('hip_code') or ''
    pytorch_functional_code = ground_truth.get('pytorch_functional_code') or ''
    pytorch_module_code = ground_truth.get('pytorch_module_code') or ''
    if not hip_code or not pytorch_functional_code:
        return None

    compile_timeout_s = ground_truth.get('compile_timeout_s')
    run_timeout_s = ground_truth.get('run_timeout_s')
    return EvalRequest(
        kernel_name=kernel_name,
        hip_code=hip_code,
        hip_ref_code=hip_code,
        pytorch_module_code=pytorch_module_code,
        pytorch_functional_code=pytorch_functional_code,
        atol=float(ground_truth.get('atol', 1e-4)),
        rtol=float(ground_truth.get('rtol', 1e-3)),
        compile_timeout_s=int(compile_timeout_s) if compile_timeout_s is not None else None,
        run_timeout_s=int(run_timeout_s) if run_timeout_s is not None else None,
    )


def _iter_requests(paths: Iterable[Path]) -> Iterator[EvalRequest]:
    for path in paths:
        for record in _iter_records_from_file(path):
            request = _extract_request(record)
            if request is not None:
                yield request


def main() -> int:
    parser = argparse.ArgumentParser(description='Prewarm reference golden/perf cache for sandbox evaluation.')
    parser.add_argument('--input', action='append', required=True, help='Input parquet/json/jsonl file. Can be passed multiple times.')
    parser.add_argument('--with-perf', action='store_true', help='Also prewarm conservative worker-local reference perf cache.')
    parser.add_argument('--gpu-id', type=int, default=None, help='Optional GPU id for local prewarm execution.')
    parser.add_argument('--max-samples', type=int, default=None, help='Stop after this many unique reference tasks.')
    parser.add_argument('--manifest-out', type=str, default=None, help='Optional output JSONL manifest path.')
    args = parser.parse_args()
    if args.with_perf and args.gpu_id is None:
        raise ValueError('--with-perf requires --gpu-id so the prewarmed perf key matches online per-GPU runtime fingerprints')

    input_paths = [Path(item).expanduser().resolve() for item in args.input]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f'Input does not exist: {path}')

    settings = load_eval_settings()
    manifest_path = Path(args.manifest_out).expanduser().resolve() if args.manifest_out else Path(settings.cache_root) / 'prewarm_manifest.jsonl'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    rows_written = 0
    unique_requests = 0
    failures = 0

    with open(manifest_path, 'w', encoding='utf-8') as manifest:
        for request in _iter_requests(input_paths):
            compile_key, golden_key, perf_key = build_reference_keys(request, settings=settings, gpu_id=args.gpu_id)
            dedupe_key = (golden_key.cache_id, perf_key.cache_id if args.with_perf else '')
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            unique_requests += 1
            if args.max_samples is not None and unique_requests > args.max_samples:
                break

            status = 'ok'
            error_message = None
            try:
                result = prewarm_reference_artifacts(
                    request,
                    settings=settings,
                    gpu_id=args.gpu_id,
                    with_perf=args.with_perf,
                )
            except Exception as exc:
                failures += 1
                status = 'error'
                error_message = str(exc)
                result = {
                    'kernel_name': request.kernel_name,
                    'compile_key': compile_key.cache_id,
                    'golden_key': golden_key.cache_id,
                    'perf_key': perf_key.cache_id,
                    'compile_cache_hit': False,
                    'golden_cache_hit': False,
                    'perf_cache_hit': False,
                    'perf_runtime_fingerprint': perf_key.runtime_fingerprint if args.with_perf else None,
                    'with_perf': args.with_perf,
                }

            manifest_record = {
                'status': status,
                'error': error_message,
                'kernel_name': request.kernel_name,
                'logical_kernel_name': golden_key.logical_kernel_name,
                'hip_ref_sha256': sha256_text(request.hip_ref_code),
                'pytorch_functional_sha256': sha256_text(request.pytorch_functional_code),
                'compile_key': result['compile_key'],
                'golden_key': result['golden_key'],
                'perf_key': result['perf_key'],
                'compile_cache_hit': result['compile_cache_hit'],
                'golden_cache_hit': result['golden_cache_hit'],
                'perf_cache_hit': result['perf_cache_hit'],
                'perf_runtime_fingerprint': result.get('perf_runtime_fingerprint'),
                'with_perf': result['with_perf'],
            }
            manifest.write(json.dumps(manifest_record, ensure_ascii=True) + '\n')
            rows_written += 1

    print('=== Reference Cache Prewarm Summary ===')
    print(f'Inputs: {len(input_paths)} file(s)')
    print(f'Unique reference tasks: {unique_requests}')
    print(f'Manifest rows written: {rows_written}')
    print(f'Failures: {failures}')
    print(f'Manifest path: {manifest_path}')
    print(f'Compile cache enabled: {settings.enable_ref_compile_cache}')
    print(f'Golden cache dir: {settings.cache_root}')
    print(f'Perf prewarm enabled: {args.with_perf}')
    print(f'Perf prewarm gpu_id: {args.gpu_id}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
