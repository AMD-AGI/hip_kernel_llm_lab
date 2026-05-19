"""
HIP Kernel Evaluation Master Server - Multi-node distributed version.
"""

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import httpx
import yaml
from fastapi import FastAPI, HTTPException

from sandbox_core.config import EvalSettings, eval_settings_from_payload, load_eval_settings
from sandbox_core.eval import run_eval_request
from sandbox_core.logging_utils import (
    configure_logging,
    derive_failure_reason,
    format_evaluation_summary,
)
from sandbox_core.protocol import BatchEvalRequest, BatchEvalResponse, EvalRequest, EvalResponse
from sandbox_core.result import EvalRunResult

RUNTIME_LOG_DIR = Path(__file__).resolve().parent.parent / "runtime" / "logs"
RUNTIME_LOG_DIR.mkdir(parents=True, exist_ok=True)

configure_logging(RUNTIME_LOG_DIR / "master_server.log")
logger = logging.getLogger("hip_master_server")


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


@dataclass
class WorkerNode:
    host: str
    port: int
    gpus: int
    healthy: bool = True
    last_check: float = 0.0


@dataclass
class GPUSlot:
    node_type: str
    node_host: str
    node_port: int
    gpu_id: int
    semaphore: asyncio.Semaphore = None

    def __post_init__(self):
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(1)


def load_config(config_path: str = None) -> Dict:
    if config_path is None:
        config_path = os.environ.get("WORKER_CONFIG", "workers.yaml")
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {
            "master": {"gpus": [0, 1, 2, 3, 4, 5, 6, 7]},
            "workers": [],
            "settings": {
                "health_check_interval": 30,
                "request_timeout": 600,
                "perf_iterations": 1000,
                "enable_ref_compile_cache": False,
                "enable_ref_golden_cache": False,
                "enable_ref_perf_cache": False,
                "ref_perf_cache_ttl_s": 3600,
                "speedup_confirm_enabled": False,
                "speedup_confirm_threshold": 1.05,
                "speedup_confirm_band": 0.02,
                "speedup_confirm_iterations": 3000,
            },
        }
    with open(config_path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


class GPUSlotManager:
    def __init__(self, config: Dict):
        self.config = config
        self.slots: List[GPUSlot] = []
        self.workers: List[WorkerNode] = []
        self.settings_dict = config.get("settings", {})
        self.request_timeout = self.settings_dict.get("request_timeout", 600)
        self.perf_iterations = self.settings_dict.get("perf_iterations", 1000)
        self.enable_ref_compile_cache = self.settings_dict.get("enable_ref_compile_cache")
        self.enable_ref_golden_cache = self.settings_dict.get("enable_ref_golden_cache")
        self.enable_ref_perf_cache = self.settings_dict.get("enable_ref_perf_cache")
        self.ref_perf_cache_ttl_s = self.settings_dict.get("ref_perf_cache_ttl_s")
        self.speedup_confirm_enabled = self.settings_dict.get("speedup_confirm_enabled", False)
        self.speedup_confirm_threshold = self.settings_dict.get("speedup_confirm_threshold", 1.05)
        self.speedup_confirm_band = self.settings_dict.get("speedup_confirm_band", 0.02)
        self.speedup_confirm_iterations = self.settings_dict.get("speedup_confirm_iterations", 3000)
        self.error_log_dir = self.settings_dict.get("error_log_dir", "./error_log")
        self.local_executor = None
        os.makedirs(self.error_log_dir, exist_ok=True)
        self.eval_settings = load_eval_settings(
            perf_iterations=self.perf_iterations,
            speedup_confirm_enabled=self.speedup_confirm_enabled,
            speedup_confirm_threshold=self.speedup_confirm_threshold,
            speedup_confirm_band=self.speedup_confirm_band,
            speedup_confirm_iterations=self.speedup_confirm_iterations,
            error_log_dir=self.error_log_dir,
            gpu_ids=(self.config.get("master", {}) or {}).get("gpus", []),
            enable_ref_compile_cache=self.enable_ref_compile_cache,
            enable_ref_golden_cache=self.enable_ref_golden_cache,
            enable_ref_perf_cache=self.enable_ref_perf_cache,
            ref_perf_cache_ttl_s=self.ref_perf_cache_ttl_s,
        )
        self._init_slots()

    def _init_slots(self):
        master_gpus = (self.config.get("master", {}) or {}).get("gpus", [])
        for gpu_id in master_gpus:
            self.slots.append(GPUSlot(node_type="local", node_host="local", node_port=0, gpu_id=gpu_id))
            logger.info(f"Added local GPU slot: GPU {gpu_id}")
        for worker_cfg in (self.config.get("workers") or []):
            host = worker_cfg["host"]
            port = worker_cfg.get("port", 8080)
            num_gpus = worker_cfg.get("gpus", 8)
            worker = WorkerNode(host=host, port=port, gpus=num_gpus)
            self.workers.append(worker)
            for gpu_id in range(num_gpus):
                self.slots.append(GPUSlot(node_type="remote", node_host=host, node_port=port, gpu_id=gpu_id))
            logger.info(f"Added remote worker: {host}:{port} with {num_gpus} GPUs")
        logger.info(f"Total GPU slots: {len(self.slots)}")

    def get_total_slots(self) -> int:
        return len(self.slots)

    def get_local_slots(self) -> int:
        return len([slot for slot in self.slots if slot.node_type == "local"])

    def get_remote_slots(self) -> int:
        return len([slot for slot in self.slots if slot.node_type == "remote"])

    async def execute_on_slot(self, slot: GPUSlot, task: EvalRequest, batch_tmp_dir: str) -> EvalResponse:
        async with slot.semaphore:
            if slot.node_type == "local":
                return await self._execute_local(slot, task, batch_tmp_dir)
            return await self._execute_remote(slot, task)

    async def _execute_local(self, slot: GPUSlot, task: EvalRequest, batch_tmp_dir: str) -> EvalResponse:
        loop = asyncio.get_event_loop()
        task_tmp_dir = os.path.join(batch_tmp_dir, f"{task.kernel_name}_local_gpu{slot.gpu_id}")
        error_log_file = os.path.join(self.error_log_dir, f"{task.kernel_name}_error.log")
        try:
            result = await loop.run_in_executor(
                self.local_executor,
                _run_local_evaluation,
                task.model_dump(),
                task_tmp_dir,
                slot.gpu_id,
                error_log_file,
                asdict(self.eval_settings),
            )
            return _result_to_response(task.kernel_name, result)
        except Exception as exc:
            logger.error(f"Local execution failed for {task.kernel_name}: {exc}")
            return _result_to_response(task.kernel_name, EvalRunResult.failure(compile_ok=False, run_ok=False, match_ok=False), reason=str(exc))

    async def _execute_remote(self, slot: GPUSlot, task: EvalRequest) -> EvalResponse:
        url = f"http://{slot.node_host}:{slot.node_port}/run_code_single_gpu"
        payload = {**task.model_dump(), "gpu_id": slot.gpu_id}
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
            return _result_to_response(
                task.kernel_name,
                EvalRunResult(
                    result.get("compile_ok", False),
                    result.get("run_ok", False),
                    result.get("match_ok", False),
                    result.get("speedup", 0.0),
                    result.get("timing") or {},
                ),
                reason=result.get("reason"),
            )
        except httpx.TimeoutException:
            logger.error(f"Remote execution timeout for {task.kernel_name} on {slot.node_host}:GPU{slot.gpu_id}")
            return _result_to_response(task.kernel_name, EvalRunResult.failure(compile_ok=False, run_ok=False, match_ok=False), reason=f"Timeout on {slot.node_host}:GPU{slot.gpu_id}")
        except Exception as exc:
            logger.error(f"Remote execution failed for {task.kernel_name}: {exc}")
            return _result_to_response(task.kernel_name, EvalRunResult.failure(compile_ok=False, run_ok=False, match_ok=False), reason=str(exc))

    async def dispatch_batch(self, tasks: List[EvalRequest], batch_tmp_dir: str) -> List[EvalResponse]:
        if not tasks:
            return []
        num_slots = len(self.slots)
        logger.info(f"Dispatching {len(tasks)} tasks across {num_slots} GPU slots")
        coroutines = []
        for idx, task in enumerate(tasks):
            slot = self.slots[idx % num_slots]
            coroutines.append(self.execute_on_slot(slot, task, batch_tmp_dir))
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        responses = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {tasks[idx].kernel_name} failed with exception: {result}")
                responses.append(
                    _result_to_response(tasks[idx].kernel_name, EvalRunResult.failure(compile_ok=False, run_ok=False, match_ok=False), reason=str(result))
                )
            else:
                responses.append(result)
        return responses

    async def health_check_all(self) -> Dict[str, bool]:
        results = {}
        async with httpx.AsyncClient(timeout=10) as client:
            for worker in self.workers:
                url = f"http://{worker.host}:{worker.port}/health"
                try:
                    response = await client.get(url)
                    worker.healthy = response.status_code == 200
                    worker.last_check = time.time()
                    results[f"{worker.host}:{worker.port}"] = worker.healthy
                except Exception as exc:
                    worker.healthy = False
                    results[f"{worker.host}:{worker.port}"] = False
                    logger.warning(f"Health check failed for {worker.host}:{worker.port}: {exc}")
        return results


def _run_local_evaluation(
    task_payload: Dict,
    tmp_dir: str,
    gpu_id: int,
    error_log_file: str,
    settings_payload: Dict,
) -> Tuple[bool, bool, bool, float, Dict]:
    settings = eval_settings_from_payload(settings_payload)
    task = EvalRequest(**task_payload)
    result = run_eval_request(
        task,
        tmp_dir=tmp_dir,
        gpu_id=gpu_id,
        error_log_file=error_log_file,
        settings=settings,
    )
    return result.compile_ok, result.run_ok, result.match_ok, result.speedup, result.timing


CONFIG_PATH = os.environ.get("WORKER_CONFIG", "workers.yaml")
config = load_config(CONFIG_PATH)
slot_manager = GPUSlotManager(config)

app = FastAPI(
    title="HIP Kernel Master Server",
    version="1.1.0",
    description="Multi-node distributed HIP kernel evaluation server",
)


@app.on_event("startup")
async def startup_event():
    num_local_gpus = slot_manager.get_local_slots()
    if num_local_gpus > 0:
        slot_manager.local_executor = ProcessPoolExecutor(max_workers=num_local_gpus)
        logger.info(f"Created local executor with {num_local_gpus} workers")
    logger.info("Performing initial health check of remote workers...")
    health_results = await slot_manager.health_check_all()
    for worker, healthy in health_results.items():
        status = "✓ healthy" if healthy else "✗ unhealthy"
        logger.info(f"  {worker}: {status}")


@app.on_event("shutdown")
async def shutdown_event():
    if slot_manager.local_executor:
        slot_manager.local_executor.shutdown(wait=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "role": "master",
        "total_slots": slot_manager.get_total_slots(),
        "local_slots": slot_manager.get_local_slots(),
        "remote_slots": slot_manager.get_remote_slots(),
        "perf_iterations": slot_manager.perf_iterations,
        "effective_arch": slot_manager.eval_settings.effective_arch,
        "compile_cache_enabled": slot_manager.eval_settings.enable_ref_compile_cache,
        "golden_cache_enabled": slot_manager.eval_settings.enable_ref_golden_cache,
        "perf_cache_enabled": slot_manager.eval_settings.enable_ref_perf_cache,
        "perf_cache_ttl_s": slot_manager.eval_settings.ref_perf_cache_ttl_s,
        "reference_cache_dir": slot_manager.eval_settings.cache_root,
    }


@app.get("/cluster/status")
async def cluster_status():
    health_results = await slot_manager.health_check_all()
    healthy_workers = sum(1 for healthy in health_results.values() if healthy)
    healthy_remote_gpus = sum(worker.gpus for worker in slot_manager.workers if worker.healthy)
    return {
        "total_nodes": len(health_results) + 1,
        "healthy_nodes": healthy_workers + 1,
        "total_gpus": slot_manager.get_total_slots(),
        "local_gpus": slot_manager.get_local_slots(),
        "healthy_remote_gpus": healthy_remote_gpus,
        "workers": health_results,
        "ready": healthy_workers > 0 or slot_manager.get_local_slots() > 0,
    }


@app.post("/run_code_batch", response_model=BatchEvalResponse)
async def evaluate_batch(batch_req: BatchEvalRequest):
    batch_size = len(batch_req.requests)
    logger.info(f"Received batch evaluation request with {batch_size} kernels")
    if batch_size == 0:
        return BatchEvalResponse(responses=[], total_time=0.0, batch_size=0)

    start_time = time.time()
    batch_id = f"{int(time.time() * 1000)}_{threading.current_thread().ident}"
    batch_tmp_dir = f"/tmp/hip_eval_master_{batch_id}"
    os.makedirs(batch_tmp_dir, exist_ok=True)
    try:
        responses = await slot_manager.dispatch_batch(batch_req.requests, batch_tmp_dir)
    except Exception as exc:
        logger.error(f"Batch evaluation failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {exc}")

    total_time = time.time() - start_time
    logger.info(
        format_evaluation_summary(
            "Distributed batch evaluation",
            [
                {
                    "kernel_name": response.kernel_name,
                    "compile_ok": response.compile_ok,
                    "run_ok": response.run_ok,
                    "match_ok": response.match_ok,
                    "speedup": response.speedup,
                    "reason": response.reason,
                    "timing": response.timing,
                }
                for response in responses
            ],
            total_elapsed=total_time,
        )
    )
    return BatchEvalResponse(responses=responses, total_time=total_time, batch_size=batch_size)


@app.post("/run_code", response_model=dict)
async def evaluate_single(req: EvalRequest):
    logger.info(f"Single evaluation request for {req.kernel_name}")
    result = await evaluate_batch(BatchEvalRequest(requests=[req]))
    if result.responses:
        return {"msg": result.responses[0]}
    raise HTTPException(status_code=500, detail="Evaluation failed")
