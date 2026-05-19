#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
sys.path.insert(0, str(REPO_ROOT))

from reward.reward_batch import call_batch_run_code
from reward.utils import (
    LEGACY_HIP_FENCE_OUTPUT_CONTRACT,
    SAMPLE_JSON_OUTPUT_CONTRACT,
    extract_kernel_name,
    extract_kernel_snippet_from_code,
    parse_kernel_generation_response,
    replace_kernel_in_hip_code,
)

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - fallback path
    pq = None


DEFAULT_HIP_FIXTURE_PATH = (
    REPO_ROOT
    / "HIP_benchmark_kit"
    / "data"
    / "hip_eval_dataset_kernelbench_gpumode_50_tasks"
    / "hip_code"
    / "hip_90_L1.hip"
)
DEFAULT_FUNCTIONAL_FIXTURE_PATH = (
    REPO_ROOT
    / "HIP_benchmark_kit"
    / "data"
    / "hip_eval_dataset_kernelbench_gpumode_50_tasks"
    / "pytorch_code_functional"
    / "py_90_L1.py"
)


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _require_run_code_url(url: str) -> str:
    normalized = (url or "").strip()
    if not normalized.endswith("/run_code"):
        raise ValueError(f"sf_url must end with /run_code, got: {url!r}")
    return normalized


def _load_sample_row(parquet_path: Path, sample_index: int) -> Dict[str, Any]:
    if sample_index < 0:
        raise ValueError("sample_index must be >= 0")

    if pq is not None:
        parquet_file = pq.ParquetFile(parquet_path)
        for row_idx, batch in enumerate(
            parquet_file.iter_batches(
                batch_size=1,
                columns=["reward_model", "extra_info"],
            )
        ):
            if row_idx == sample_index:
                rows = batch.to_pylist()
                if rows:
                    return rows[0]
                break

    import pandas as pd  # Fallback when pyarrow iteration is unavailable.

    dataframe = pd.read_parquet(parquet_path)
    if sample_index >= len(dataframe):
        raise IndexError(
            f"sample_index={sample_index} is out of range for parquet with {len(dataframe)} rows"
        )
    return dataframe.iloc[sample_index].to_dict()


def _load_reference_bundle_from_parquet(
    parquet_path: Path,
    sample_index: int,
) -> dict[str, Any]:
    row = _load_sample_row(parquet_path, sample_index)
    reward_model = _coerce_dict(row.get("reward_model"))
    ground_truth = _coerce_dict(reward_model.get("ground_truth"))
    extra_info = _coerce_dict(row.get("extra_info"))

    hip_ref = ground_truth.get("hip_code") or ""
    module_code = ground_truth.get("pytorch_module_code") or ""
    functional_code = ground_truth.get("pytorch_functional_code") or ""
    kernel_name = (
        ground_truth.get("kernel_name")
        or extra_info.get("kernel_name")
        or extract_kernel_name(hip_ref)
        or "smoke_kernel"
    )
    if not hip_ref:
        raise ValueError("smoke test requires reward_model.ground_truth.hip_code")
    if not functional_code:
        raise ValueError("smoke test requires reward_model.ground_truth.pytorch_functional_code")

    return {
        "source_mode": "parquet",
        "sample_index": sample_index,
        "hip_ref": hip_ref,
        "module_code": module_code,
        "functional_code": functional_code,
        "kernel_name": kernel_name,
        "atol": float(ground_truth.get("atol", 1e-4)),
        "rtol": float(ground_truth.get("rtol", 1e-3)),
        "compile_timeout_s": int(ground_truth.get("compile_timeout_s", 600)),
        "run_timeout_s": int(ground_truth.get("run_timeout_s", 600)),
    }


def _load_reference_bundle_from_fixture(
    hip_path: Path,
    functional_path: Path,
    module_path: Path | None,
    kernel_name: str | None,
) -> dict[str, Any]:
    hip_path = hip_path.expanduser().resolve()
    functional_path = functional_path.expanduser().resolve()
    if not hip_path.is_file():
        raise FileNotFoundError(f"hip fixture not found: {hip_path}")
    if not functional_path.is_file():
        raise FileNotFoundError(f"functional fixture not found: {functional_path}")

    hip_ref = hip_path.read_text(encoding="utf-8")
    functional_code = functional_path.read_text(encoding="utf-8")
    module_code = ""
    if module_path is not None:
        module_path = module_path.expanduser().resolve()
        if not module_path.is_file():
            raise FileNotFoundError(f"module fixture not found: {module_path}")
        module_code = module_path.read_text(encoding="utf-8")

    resolved_kernel_name = kernel_name or extract_kernel_name(hip_ref) or "smoke_kernel"
    return {
        "source_mode": "fixture",
        "sample_index": None,
        "hip_ref": hip_ref,
        "module_code": module_code,
        "functional_code": functional_code,
        "kernel_name": resolved_kernel_name,
        "atol": 1e-4,
        "rtol": 1e-3,
        "compile_timeout_s": 600,
        "run_timeout_s": 600,
    }


def _build_sample_responses(kernel_snippet: str) -> list[dict[str, str]]:
    json_response = (
        "<think>\nPreserve the original signature and emit JSON only.\n</think>\n"
        + json.dumps(
            {
                "thought": "Preserve the original signature and emit JSON only.",
                "code": kernel_snippet,
            }
        )
    )
    legacy_response = (
        "Reason about occupancy and memory traffic first.\n"
        f"```hip\n{kernel_snippet}\n```"
    )
    return [
        {
            "label": "sample_json",
            "output_contract": SAMPLE_JSON_OUTPUT_CONTRACT,
            "response": json_response,
        },
        {
            "label": "legacy_hip_fence",
            "output_contract": LEGACY_HIP_FENCE_OUTPUT_CONTRACT,
            "response": legacy_response,
        },
    ]


def _build_request_payload(
    sample: dict[str, str],
    *,
    kernel_name: str,
    hip_ref: str,
    module_code: str,
    functional_code: str,
    atol: float,
    rtol: float,
    compile_timeout_s: int,
    run_timeout_s: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parse_result = parse_kernel_generation_response(
        sample["response"],
        data_source="kernel-agent-react-train",
        kernel_name=kernel_name,
        hip_ref=hip_ref,
        output_contract=sample["output_contract"],
    )
    if not parse_result["parse_ok"]:
        raise RuntimeError(
            f"parse failed for {sample['label']}: {parse_result['parse_error']}"
        )

    hip_src = parse_result["hip_src"]
    hip_code = (
        replace_kernel_in_hip_code(hip_ref, hip_src, kernel_name=kernel_name)
        if hip_ref
        else hip_src
    )
    payload = {
        "kernel_name": f"{kernel_name}_{sample['label']}",
        "hip_code": hip_code,
        "hip_ref_code": hip_ref,
        "pytorch_module_code": module_code,
        "pytorch_functional_code": functional_code,
        "atol": atol,
        "rtol": rtol,
        "compile_timeout_s": compile_timeout_s,
        "run_timeout_s": run_timeout_s,
    }
    return payload, parse_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live smoke test for single-turn react parsing + sandbox batch evaluation."
    )
    parser.add_argument(
        "--reference-source",
        choices=["fixture", "parquet"],
        default="fixture",
        help="Reference source for the smoke payloads.",
    )
    parser.add_argument("--parquet", type=Path, help="Training parquet path for parquet-mode smoke.")
    parser.add_argument("--sf-url", required=True, help="Sandbox URL ending with /run_code.")
    parser.add_argument("--sample-index", type=int, default=0, help="Parquet row used for smoke payloads.")
    parser.add_argument("--timeout-s", type=int, default=1800, help="HTTP timeout for /run_code_batch.")
    parser.add_argument("--hip-path", type=Path, default=DEFAULT_HIP_FIXTURE_PATH, help="Known-good HIP reference fixture.")
    parser.add_argument("--functional-path", type=Path, default=DEFAULT_FUNCTIONAL_FIXTURE_PATH, help="Known-good PyTorch functional fixture.")
    parser.add_argument("--module-path", type=Path, default=None, help="Optional PyTorch module fixture.")
    parser.add_argument("--kernel-name", default=None, help="Optional kernel name override for fixture mode.")
    args = parser.parse_args()

    sf_url = _require_run_code_url(args.sf_url)
    if args.reference_source == "parquet":
        if args.parquet is None:
            raise ValueError("--parquet is required when --reference-source=parquet")
        parquet_path = args.parquet.expanduser().resolve()
        if not parquet_path.is_file():
            raise FileNotFoundError(f"parquet file not found: {parquet_path}")
        reference_bundle = _load_reference_bundle_from_parquet(
            parquet_path,
            args.sample_index,
        )
    else:
        reference_bundle = _load_reference_bundle_from_fixture(
            args.hip_path,
            args.functional_path,
            args.module_path,
            args.kernel_name,
        )

    hip_ref = reference_bundle["hip_ref"]
    module_code = reference_bundle["module_code"]
    functional_code = reference_bundle["functional_code"]
    kernel_name = reference_bundle["kernel_name"]

    kernel_snippet = extract_kernel_snippet_from_code(
        hip_ref,
        kernel_name=kernel_name,
        hip_ref=hip_ref,
    )
    if not kernel_snippet:
        raise ValueError(f"failed to extract kernel snippet for {kernel_name!r}")

    samples = _build_sample_responses(kernel_snippet)
    batch_requests = []
    parse_summaries = []
    for sample in samples:
        payload, parse_result = _build_request_payload(
            sample,
            kernel_name=kernel_name,
            hip_ref=hip_ref,
            module_code=module_code,
            functional_code=functional_code,
            atol=float(reference_bundle["atol"]),
            rtol=float(reference_bundle["rtol"]),
            compile_timeout_s=int(reference_bundle["compile_timeout_s"]),
            run_timeout_s=int(reference_bundle["run_timeout_s"]),
        )
        batch_requests.append(payload)
        parse_summaries.append(
            {
                "label": sample["label"],
                "output_contract": sample["output_contract"],
                "parse_mode": parse_result["parse_mode"],
                "hip_src_len": len(parse_result["hip_src"]),
            }
        )

    print(
        json.dumps(
            {
                "event": "react_contract_smoke_parse_summary",
                "reference_source": reference_bundle["source_mode"],
                "sample_index": reference_bundle["sample_index"],
                "kernel_name": kernel_name,
                "samples": parse_summaries,
            },
            indent=2,
            ensure_ascii=True,
        )
    )

    response = call_batch_run_code(sf_url, batch_requests, timeout_s=args.timeout_s)
    if response.status_code != 200:
        print(
            json.dumps(
                {
                    "event": "react_contract_smoke_http_error",
                    "status_code": response.status_code,
                    "body_preview": response.text[:500],
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 1

    payload = response.json()
    response_items = payload.get("responses") or []
    if len(response_items) != len(samples):
        print(
            json.dumps(
                {
                    "event": "react_contract_smoke_response_count_mismatch",
                    "expected": len(samples),
                    "actual": len(response_items),
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return 1

    all_ok = True
    results = []
    for sample, parse_summary, resp_item in zip(samples, parse_summaries, response_items):
        compile_ok = bool(resp_item.get("compile_ok"))
        run_ok = bool(resp_item.get("run_ok"))
        match_ok = bool(resp_item.get("match_ok"))
        ok = compile_ok and run_ok and match_ok
        all_ok = all_ok and ok
        results.append(
            {
                "label": sample["label"],
                "output_contract": sample["output_contract"],
                "parse_mode": parse_summary["parse_mode"],
                "compile_ok": compile_ok,
                "run_ok": run_ok,
                "match_ok": match_ok,
                "speedup": float(resp_item.get("speedup") or 0.0),
                "reason": resp_item.get("reason", ""),
            }
        )

    print(
        json.dumps(
            {
                "event": "react_contract_smoke_result",
                "reference_source": reference_bundle["source_mode"],
                "kernel_name": kernel_name,
                "server_total_time_s": float(payload.get("total_time") or 0.0),
                "results": results,
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
