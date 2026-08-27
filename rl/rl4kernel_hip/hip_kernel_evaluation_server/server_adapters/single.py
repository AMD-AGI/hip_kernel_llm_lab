# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import concurrent.futures
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, HTTPException

from sandbox_core.config import load_eval_settings
from sandbox_core.logging_utils import (
    configure_logging,
    derive_failure_reason,
    format_kernel_failure,
    format_kernel_success,
)
from sandbox_core.protocol import EvalRequest, EvalResponse
from sandbox_core.result import EvalRunResult

RUNTIME_LOG_DIR = Path(__file__).resolve().parent.parent / "runtime" / "logs"
RUNTIME_LOG_DIR.mkdir(parents=True, exist_ok=True)
from sandbox_core.eval import run_eval_request

configure_logging(RUNTIME_LOG_DIR / "app.log")
logger = logging.getLogger("hip_kernel_evaluator_app")


def _result_to_response(kernel_name: str, result: EvalRunResult) -> EvalResponse:
    return EvalResponse(
        kernel_name=kernel_name,
        compile_ok=result.compile_ok,
        run_ok=result.run_ok,
        match_ok=result.match_ok,
        speedup=result.speedup,
        reason=derive_failure_reason(result.compile_ok, result.run_ok, result.match_ok, result.timing),
        timing=result.timing,
    )


class PerformanceStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        self.timing_stats = defaultdict(list)
        self.gpu_usage = defaultdict(int)

    def record_request(self, success: bool, timing: Dict, gpu_id: int):
        with self.lock:
            self.completed_requests += 1
            if not success:
                self.failed_requests += 1
            self.gpu_usage[gpu_id] += 1
            for stage, duration in (timing or {}).items():
                if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                    self.timing_stats[stage].append(duration)

    def get_stats(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            throughput = self.completed_requests / elapsed if elapsed > 0 else 0
            avg_timing = {}
            for stage, durations in self.timing_stats.items():
                if durations:
                    avg_timing[f"avg_{stage}"] = sum(durations) / len(durations)
            return {
                "total_requests": self.total_requests,
                "completed_requests": self.completed_requests,
                "failed_requests": self.failed_requests,
                "success_rate": (self.completed_requests - self.failed_requests) / self.completed_requests if self.completed_requests > 0 else 0,
                "elapsed_time_s": elapsed,
                "throughput_per_s": throughput,
                "gpu_usage": dict(self.gpu_usage),
                **avg_timing,
            }


settings = load_eval_settings()
perf_stats = PerformanceStats()
GPU_IDS = settings.gpu_ids
TIME_OUT = settings.handler_timeout_s
os.environ["HIP_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))
os.makedirs(settings.error_log_dir, exist_ok=True)

gpu_queue = queue.Queue()
for gpu_id in GPU_IDS:
    gpu_queue.put(gpu_id)

log_lock = threading.Lock()
app = FastAPI(title="HIP Kernel Eval Server", version="0.2.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu_ids": GPU_IDS,
        "perf_iterations": settings.perf_iterations,
        "effective_arch": settings.effective_arch,
        "error_log_dir": settings.error_log_dir,
        "compile_cache_enabled": settings.enable_ref_compile_cache,
        "golden_cache_enabled": settings.enable_ref_golden_cache,
        "perf_cache_enabled": settings.enable_ref_perf_cache,
        "perf_cache_ttl_s": settings.ref_perf_cache_ttl_s,
        "reference_cache_dir": settings.cache_root,
    }


@app.post("/run_code_naive")
def evaluate_naive(req: EvalRequest):
    logger.info(f"Start an evaluation for {req.kernel_name} hip kernel.")
    start_time = time.time()
    result = run_eval_request(
        req,
        settings=settings,
    )
    compile_ok, run_ok, match_ok, speedup, timing = result
    if compile_ok and run_ok and match_ok:
        logger.info(format_kernel_success(req.kernel_name, speedup=speedup, timing=timing))
    else:
        logger.warning(
            format_kernel_failure(
                req.kernel_name,
                compile_ok=compile_ok,
                run_ok=run_ok,
                match_ok=match_ok,
                timing=timing,
            )
        )
    logger.info(f"Cost time for evaluation: {time.time() - start_time}.")
    logger.info(f"Timing breakdown: {timing}")
    response = _result_to_response(req.kernel_name, EvalRunResult(compile_ok, run_ok, match_ok, speedup, timing))
    return {"msg": response}


def evaluate_with_gpu(req: EvalRequest):
    wait_start = time.time()
    gpu_id = gpu_queue.get()
    wait_time = time.time() - wait_start
    task_id = f"{int(time.time() * 1000)}_{threading.current_thread().ident}"
    tmp_dir = f"/tmp/hip_eval_single_gpu{gpu_id}_{task_id}"
    error_log_file = os.path.join(settings.error_log_dir, f"{req.kernel_name}_error.log")

    try:
        with log_lock:
            logger.info(f"[GPU {gpu_id}] Start evaluation for {req.kernel_name} (waited {wait_time:.2f}s for GPU)")
        start_time = time.time()
        result = run_eval_request(
            req,
            tmp_dir=tmp_dir,
            gpu_id=gpu_id,
            error_log_file=error_log_file,
            settings=settings,
        )
        compile_ok, run_ok, match_ok, speedup, timing = result
        end_time = time.time()
        timing = timing or {}
        timing["gpu_wait_time"] = wait_time
        timing["total_with_wait"] = end_time - wait_start
        success = compile_ok and run_ok and match_ok
        perf_stats.record_request(success, timing, gpu_id)
        with log_lock:
            log_kernel_name = f"[GPU {gpu_id}] {req.kernel_name}"
            if success:
                logger.info(format_kernel_success(log_kernel_name, speedup=speedup, timing=timing))
            else:
                logger.warning(
                    format_kernel_failure(
                        log_kernel_name,
                        compile_ok=compile_ok,
                        run_ok=run_ok,
                        match_ok=match_ok,
                        timing=timing,
                    )
                )
            logger.info(
                f"[GPU {gpu_id}] Timing breakdown: compilation={timing.get('compilation', 0):.2f}s, "
                f"ref_compile={timing.get('reference_compile_build_s', 0):.2f}s, "
                f"ref_golden={timing.get('reference_golden_build_s', 0):.2f}s, "
                f"ref_perf={timing.get('reference_perf_build_s', 0):.2f}s, "
                f"ref_run={timing.get('ref_run', 0):.2f}s, test_run={timing.get('test_run', 0):.2f}s"
            )
        return compile_ok, run_ok, match_ok, speedup, timing
    finally:
        gpu_queue.put(gpu_id)


@app.post("/run_code")
def evaluate(req: EvalRequest):
    logger.info(f"Received evaluation request for {req.kernel_name}")
    perf_stats.total_requests += 1
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(GPU_IDS))) as executor:
        future = executor.submit(evaluate_with_gpu, req)
        try:
            compile_ok, run_ok, match_ok, speedup, timing = future.result(timeout=TIME_OUT)
        except concurrent.futures.TimeoutError:
            logger.error(f"Evaluation for {req.kernel_name} timed out (>{TIME_OUT}s).")
            perf_stats.failed_requests += 1
            raise HTTPException(status_code=504, detail="Evaluation timed out")
        except Exception as exc:
            logger.error(f"Evaluation for {req.kernel_name} failed: {exc}")
            perf_stats.failed_requests += 1
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")

    response = _result_to_response(req.kernel_name, EvalRunResult(compile_ok, run_ok, match_ok, speedup, timing))

    try:
        timing_record = {
            "timestamp": datetime.now().isoformat(),
            "kernel_name": req.kernel_name,
            "compile_ok": compile_ok,
            "run_ok": run_ok,
            "match_ok": match_ok,
            "speedup": speedup,
            "timing": timing,
        }
        with open(RUNTIME_LOG_DIR / "timing_stats.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(timing_record) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to save timing stats: {exc}")

    logger.info(f"Total time for {req.kernel_name}: {time.time() - start_time:.2f}s")
    return {"msg": response}
