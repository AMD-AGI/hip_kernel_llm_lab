#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import requests


SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent.parent
DEFAULT_HIP_PATH = (
    REPO_ROOT
    / "HIP_benchmark_kit"
    / "data"
    / "hip_eval_dataset_kernelbench_gpumode_50_tasks"
    / "hip_code"
    / "hip_90_L1.hip"
)
DEFAULT_FUNCTIONAL_PATH = (
    REPO_ROOT
    / "HIP_benchmark_kit"
    / "data"
    / "hip_eval_dataset_kernelbench_gpumode_50_tasks"
    / "pytorch_code_functional"
    / "py_90_L1.py"
)


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def build_request(kernel_name: str, hip_code: str, functional_code: str) -> Dict:
    return {
        "kernel_name": kernel_name,
        "hip_code": hip_code,
        "hip_ref_code": hip_code,
        "pytorch_module_code": "",
        "pytorch_functional_code": functional_code,
        "atol": 1e-4,
        "rtol": 1e-3,
        "compile_timeout_s": 600,
        "run_timeout_s": 600,
    }


def post_single(url: str, request_payload: Dict, timeout_s: int = 1200) -> Dict:
    response = requests.post(f"{url}/run_code", json=request_payload, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    return payload["msg"] if "msg" in payload else payload


def post_single_gpu(url: str, request_payload: Dict, gpu_id: int, timeout_s: int = 1200) -> Dict:
    response = requests.post(
        f"{url}/run_code_single_gpu",
        json={**request_payload, "gpu_id": gpu_id},
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json()


def post_batch(url: str, requests_payload: List[Dict], timeout_s: int = 2400) -> Dict:
    response = requests.post(f"{url}/run_code_batch", json={"requests": requests_payload}, timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def print_result(label: str, payload: Dict) -> None:
    timing = payload.get("timing") or {}
    print(f"[{label}] compile_ok={payload.get('compile_ok')} run_ok={payload.get('run_ok')} match_ok={payload.get('match_ok')} speedup={payload.get('speedup')}")
    print(
        f"[{label}] cache flags: compile={timing.get('reference_compile_cache_hit')} "
        f"golden={timing.get('reference_golden_cache_hit')} perf={timing.get('reference_perf_cache_hit')} "
        f"prepare={timing.get('prepare_code')} ref_compile={timing.get('reference_compile_build_s')} "
        f"ref_golden={timing.get('reference_golden_build_s')} ref_perf={timing.get('reference_perf_build_s')} "
        f"test_run={timing.get('test_run')} ref_run={timing.get('ref_run')} total={timing.get('total')}"
    )


def run_cache_smoke(url: str, kernel_name: str, hip_code: str, functional_code: str) -> None:
    request_payload = build_request(kernel_name, hip_code, functional_code)
    first = post_single(url, request_payload)
    second = post_single(url, request_payload)
    print_result("first", first)
    print_result("second", second)


def run_cache_smoke_single_gpu(url: str, kernel_name: str, hip_code: str, functional_code: str, gpu_id: int) -> None:
    request_payload = build_request(kernel_name, hip_code, functional_code)
    first = post_single_gpu(url, request_payload, gpu_id=gpu_id)
    second = post_single_gpu(url, request_payload, gpu_id=gpu_id)
    print_result("first", first)
    print_result("second", second)


def run_cache_smoke_batch(url: str, kernel_name: str, hip_code: str, functional_code: str) -> None:
    request_payload = build_request(kernel_name, hip_code, functional_code)
    first = post_batch(url, [request_payload])
    second = post_batch(url, [request_payload])
    print_result("first", first["responses"][0])
    print_result("second", second["responses"][0])


def run_throughput(
    url: str,
    prefix: str,
    hip_code: str,
    functional_code: str,
    count: int,
    *,
    stable_names: bool = False,
) -> Dict[str, float]:
    suffix = "" if stable_names else f"_{int(time.time())}"
    requests_payload = [
        build_request(f"{prefix}_{idx:02d}{suffix}", hip_code, functional_code)
        for idx in range(count)
    ]
    start = time.time()
    payload = post_batch(url, requests_payload)
    wall_time = time.time() - start
    responses = payload.get("responses", [])
    success_count = sum(1 for item in responses if item.get("compile_ok") and item.get("run_ok") and item.get("match_ok"))
    return {
        "wall_time_s": wall_time,
        "server_total_time_s": float(payload.get("total_time", 0.0)),
        "batch_size": int(payload.get("batch_size", len(requests_payload))),
        "success_count": success_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real smoke / throughput validation against the sandbox server.")
    parser.add_argument(
        "--mode",
        choices=["cache-smoke", "cache-smoke-single-gpu", "cache-smoke-batch", "throughput"],
        required=True,
    )
    parser.add_argument("--url", required=True, help="Base URL without trailing slash, e.g. http://127.0.0.1:18082")
    parser.add_argument("--kernel-name", default="real_smoke_l1")
    parser.add_argument("--count", type=int, default=4, help="Batch size for throughput mode.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id for single-gpu smoke mode.")
    parser.add_argument("--runs", type=int, default=1, help="Number of throughput runs to execute.")
    parser.add_argument("--stable-names", action="store_true", help="Reuse identical kernel names across repeated runs.")
    parser.add_argument("--hip-path", type=Path, default=DEFAULT_HIP_PATH)
    parser.add_argument("--functional-path", type=Path, default=DEFAULT_FUNCTIONAL_PATH)
    parser.add_argument("--compare-url", default=None, help="Optional baseline URL for throughput comparison.")
    args = parser.parse_args()

    hip_code = read_text(args.hip_path)
    functional_code = read_text(args.functional_path)

    if args.mode == "cache-smoke":
        run_cache_smoke(args.url.rstrip("/"), args.kernel_name, hip_code, functional_code)
        return 0
    if args.mode == "cache-smoke-single-gpu":
        run_cache_smoke_single_gpu(args.url.rstrip("/"), args.kernel_name, hip_code, functional_code, args.gpu_id)
        return 0
    if args.mode == "cache-smoke-batch":
        run_cache_smoke_batch(args.url.rstrip("/"), args.kernel_name, hip_code, functional_code)
        return 0

    for run_idx in range(args.runs):
        result = run_throughput(
            args.url.rstrip("/"),
            args.kernel_name,
            hip_code,
            functional_code,
            args.count,
            stable_names=args.stable_names,
        )
        print(json.dumps({"target": args.url, "run_idx": run_idx, **result}, indent=2))
    if args.compare_url:
        for run_idx in range(args.runs):
            baseline = run_throughput(
                args.compare_url.rstrip("/"),
                f"{args.kernel_name}_baseline",
                hip_code,
                functional_code,
                args.count,
                stable_names=args.stable_names,
            )
            print(json.dumps({"baseline": args.compare_url, "run_idx": run_idx, **baseline}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
