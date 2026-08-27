# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
HIP Kernel Evaluation Server - Batch API version.
Supports multi-GPU batch evaluation and worker-node execution.
"""

import asyncio
import logging
import os
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException

from sandbox_core.config import load_eval_settings
from sandbox_core.eval import evaluate_requests_parallel, run_eval_request
from sandbox_core.logging_utils import (
    configure_logging,
    derive_failure_reason,
    format_evaluation_summary,
    format_batch_request_footer,
)
from sandbox_core.protocol import BatchEvalRequest, BatchEvalResponse, EvalRequest, EvalResponse, SingleGPUEvalRequest
from sandbox_core.result import EvalRunResult
from server_adapters.tool_router import router as tool_router, tool_runtime

RUNTIME_LOG_DIR = Path(__file__).resolve().parent.parent / "runtime" / "logs"
RUNTIME_LOG_DIR.mkdir(parents=True, exist_ok=True)

configure_logging(RUNTIME_LOG_DIR / "app_batch.log")
logger = logging.getLogger("hip_kernel_batch_server")


def _result_to_response(kernel_name: str, result: EvalRunResult, *, reason: str | None = None) -> EvalResponse:
    return EvalResponse(
        kernel_name=kernel_name,
        compile_ok=result.compile_ok,
        run_ok=result.run_ok,
        match_ok=result.match_ok,
        speedup=result.speedup,
        reason=reason if reason is not None else derive_failure_reason(result.compile_ok, result.run_ok, result.match_ok, result.timing),
        timing=result.timing,
    )

settings = load_eval_settings()
GPU_IDS = settings.gpu_ids
NODE_ID = os.environ.get("NODE_ID", socket.gethostname())
ERROR_LOG_DIR = settings.error_log_dir
PERF_ITERATIONS = settings.perf_iterations
MAX_BATCH_WORKERS = int(os.environ.get("HIP_MAX_BATCH_WORKERS", str(max(1, len(GPU_IDS)))))
GPU_SEMAPHORES: Dict[int, asyncio.Semaphore] = {}

os.makedirs(ERROR_LOG_DIR, exist_ok=True)
os.environ["HIP_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))
logger.info(f"Error log directory: {ERROR_LOG_DIR}")
logger.info(f"Performance test iterations: {PERF_ITERATIONS}")
logger.info(f"Effective arch: {settings.effective_arch}")
logger.info(f"Reference compile cache enabled: {settings.enable_ref_compile_cache}")
logger.info(f"Golden cache enabled: {settings.enable_ref_golden_cache}")
logger.info(f"Perf cache enabled: {settings.enable_ref_perf_cache}")
logger.info(f"Perf cache TTL (s): {settings.ref_perf_cache_ttl_s}")
logger.info(f"Reference cache dir: {settings.cache_root}")

app = FastAPI(title="HIP Kernel Batch Eval Server", version="0.4.0")
app.include_router(tool_router)


@app.on_event("startup")
async def startup_event():
    global GPU_SEMAPHORES
    for gpu_id in GPU_IDS:
        GPU_SEMAPHORES[gpu_id] = asyncio.Semaphore(1)
    logger.info(f"Initialized GPU semaphores for GPUs: {GPU_IDS}")


@app.get("/health")
def health():
    scheduler_status = tool_runtime.scheduler_status()
    return {
        "status": "ok",
        "role": "worker",
        "node_id": NODE_ID,
        "gpu_ids": GPU_IDS,
        "gpu_count": len(GPU_IDS),
        "max_batch_workers": MAX_BATCH_WORKERS,
        "compile_cpu_slots": settings.compile_cpu_slots,
        "compile_inner_jobs": settings.compile_inner_jobs,
        "two_stage_batch_enabled": settings.enable_two_stage_batch,
        "cpu_affinity_enabled": os.environ.get("HIP_ENABLE_CPU_AFFINITY", "").strip().lower() in {"1", "true", "yes", "on"},
        "perf_iterations": PERF_ITERATIONS,
        "effective_arch": settings.effective_arch,
        "error_log_dir": ERROR_LOG_DIR,
        "compile_cache_enabled": settings.enable_ref_compile_cache,
        "golden_cache_enabled": settings.enable_ref_golden_cache,
        "perf_cache_enabled": settings.enable_ref_perf_cache,
        "perf_cache_ttl_s": settings.ref_perf_cache_ttl_s,
        "reference_cache_dir": settings.cache_root,
        "tool_scheduler": scheduler_status.model_dump(),
    }


@app.get("/worker/info")
def worker_info():
    scheduler_status = tool_runtime.scheduler_status()
    return {
        "node_id": NODE_ID,
        "hostname": socket.gethostname(),
        "gpu_ids": GPU_IDS,
        "gpu_count": len(GPU_IDS),
        "max_batch_workers": MAX_BATCH_WORKERS,
        "compile_cpu_slots": settings.compile_cpu_slots,
        "compile_inner_jobs": settings.compile_inner_jobs,
        "two_stage_batch_enabled": settings.enable_two_stage_batch,
        "cpu_affinity_enabled": os.environ.get("HIP_ENABLE_CPU_AFFINITY", "").strip().lower() in {"1", "true", "yes", "on"},
        "perf_iterations": PERF_ITERATIONS,
        "effective_arch": settings.effective_arch,
        "error_log_dir": ERROR_LOG_DIR,
        "compile_cache_enabled": settings.enable_ref_compile_cache,
        "golden_cache_enabled": settings.enable_ref_golden_cache,
        "perf_cache_enabled": settings.enable_ref_perf_cache,
        "perf_cache_ttl_s": settings.ref_perf_cache_ttl_s,
        "reference_cache_dir": settings.cache_root,
        "tool_scheduler": scheduler_status.model_dump(),
        "status": "ready",
    }


@app.post("/run_code_single_gpu", response_model=EvalResponse)
async def evaluate_single_gpu(req: SingleGPUEvalRequest):
    gpu_id = req.gpu_id
    if gpu_id not in GPU_IDS:
        raise HTTPException(status_code=400, detail=f"Invalid gpu_id {gpu_id}. Available GPUs: {GPU_IDS}")

    logger.info(f"Single GPU evaluation request for {req.kernel_name} on GPU {gpu_id}")
    semaphore = GPU_SEMAPHORES.get(gpu_id)
    if semaphore is None:
        raise HTTPException(status_code=500, detail=f"Semaphore not initialized for GPU {gpu_id}")

    async with semaphore:
        loop = asyncio.get_event_loop()
        task_id = f"{int(time.time() * 1000)}_{threading.current_thread().ident}"
        tmp_dir = f"/tmp/hip_eval_gpu{gpu_id}_{task_id}"
        error_log_file = os.path.join(ERROR_LOG_DIR, f"{req.kernel_name}_error.log")
        try:
            eval_request = EvalRequest(**req.model_dump(exclude={"gpu_id"}))
            result = await loop.run_in_executor(
                None,
                lambda: run_eval_request(
                    eval_request,
                    tmp_dir=tmp_dir,
                    gpu_id=gpu_id,
                    error_log_file=error_log_file,
                    settings=settings,
                ),
            )
            return _result_to_response(req.kernel_name, result)
        except Exception as exc:
            logger.error(f"Single GPU evaluation failed for {req.kernel_name}: {exc}")
            return _result_to_response(req.kernel_name, EvalRunResult.failure(compile_ok=False, run_ok=False, match_ok=False), reason=str(exc))


@app.post("/run_code_batch", response_model=BatchEvalResponse)
def evaluate_batch(batch_req: BatchEvalRequest):
    batch_size = len(batch_req.requests)
    logger.info(f"Received batch evaluation request with {batch_size} kernels")
    if batch_size == 0:
        return BatchEvalResponse(responses=[], total_time=0.0, batch_size=0)

    start_time = time.time()
    batch_id = f"{int(time.time() * 1000)}_{threading.current_thread().ident}"
    batch_tmp_dir = f"/tmp/hip_eval_batch_{batch_id}"
    max_workers = min(batch_size, len(GPU_IDS), MAX_BATCH_WORKERS if MAX_BATCH_WORKERS > 0 else max(1, len(GPU_IDS)))
    logger.info(f"Using {max_workers} parallel workers for {batch_size} kernels")
    logger.info(f"Batch temp directory: {batch_tmp_dir}")

    try:
        results = evaluate_requests_parallel(
            batch_req.requests,
            max_workers=max_workers,
            base_tmp_dir=batch_tmp_dir,
            gpu_ids=GPU_IDS,
            error_log_dir=ERROR_LOG_DIR,
            settings=settings,
        )
    except Exception as exc:
        logger.error(f"Batch evaluation failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {exc}")

    responses = []
    for req, result in zip(batch_req.requests, results):
        responses.append(_result_to_response(req.kernel_name, result))

    total_time = time.time() - start_time
    logger.info(
        format_evaluation_summary(
            "Worker batch evaluation",
            [
                {
                    "kernel_name": req.kernel_name,
                    "compile_ok": result.compile_ok,
                    "run_ok": result.run_ok,
                    "match_ok": result.match_ok,
                    "speedup": result.speedup,
                    "timing": result.timing,
                }
                for req, result in zip(batch_req.requests, results)
            ],
            total_elapsed=total_time,
        )
    )
    logger.info(format_batch_request_footer("Batch request completed", batch_size=batch_size, total_time=total_time))
    return BatchEvalResponse(responses=responses, total_time=total_time, batch_size=batch_size)


gpu_queue = queue.Queue()
for gpu_id in GPU_IDS:
    gpu_queue.put(gpu_id)


@app.post("/run_code", response_model=dict)
def evaluate_single(req: EvalRequest):
    logger.info(f"Single evaluation request for {req.kernel_name}")
    gpu_id = gpu_queue.get()
    task_id = f"{int(time.time() * 1000)}_{threading.current_thread().ident}"
    tmp_dir = f"/tmp/hip_eval_single_gpu{gpu_id}_{task_id}"
    error_log_file = os.path.join(ERROR_LOG_DIR, f"{req.kernel_name}_error.log")
    try:
        result = run_eval_request(
            req,
            tmp_dir=tmp_dir,
            gpu_id=gpu_id,
            error_log_file=error_log_file,
            settings=settings,
        )
        return {"msg": _result_to_response(req.kernel_name, result)}
    finally:
        gpu_queue.put(gpu_id)
