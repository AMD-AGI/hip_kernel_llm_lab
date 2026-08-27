# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from .config import EvalSettings, load_eval_settings, parse_gpu_ids
from .eval import run_compile_request, run_eval_request
from .protocol import EvalRequest
from .tool_protocol import (
    KernelReferenceBundle,
    KernelToolActionRequest,
    KernelToolBudget,
    KernelToolCreateSessionRequest,
    KernelToolCreateSessionResponse,
    KernelToolDiagnosticsResponse,
    KernelToolFinalizeResponse,
    KernelToolObservation,
    KernelToolSchedulerStatus,
    KernelToolUpdateCandidateRequest,
    KernelToolUpdateCandidateResponse,
)


def _md5_short(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:8]


def _stable_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class EvalScheduler:
    def __init__(self, *, gpu_ids: list[int], max_cpu_jobs: int):
        self.gpu_ids = list(gpu_ids)
        self.cpu_slots = max(1, int(max_cpu_jobs))
        self.cpu_semaphore = asyncio.Semaphore(self.cpu_slots)
        self.cpu_executor = ThreadPoolExecutor(max_workers=self.cpu_slots, thread_name_prefix="hip-tool-cpu")
        self.gpu_semaphores = {gpu_id: asyncio.Semaphore(1) for gpu_id in self.gpu_ids}
        self._cpu_in_use = 0
        self._cpu_pending = 0
        self._gpu_in_use = {gpu_id: 0 for gpu_id in self.gpu_ids}
        self._gpu_pending = {gpu_id: 0 for gpu_id in self.gpu_ids}
        self._lock = asyncio.Lock()

    async def run_cpu(self, fn: Callable[[], Any]) -> tuple[Any, Dict[str, Any]]:
        queued_at = time.monotonic()
        self._cpu_pending += 1
        async with self.cpu_semaphore:
            wait_s = time.monotonic() - queued_at
            self._cpu_pending = max(0, self._cpu_pending - 1)
            self._cpu_in_use += 1
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(self.cpu_executor, fn)
                return result, {
                    "resource_kind": "cpu",
                    "queue_wait_s": wait_s,
                }
            finally:
                self._cpu_in_use -= 1

    async def _acquire_gpu(self) -> tuple[int, float]:
        queued_at = time.monotonic()
        async with self._lock:
            selected_gpu = min(self.gpu_ids, key=lambda gpu_id: (self._gpu_pending[gpu_id] + self._gpu_in_use[gpu_id], gpu_id))
            self._gpu_pending[selected_gpu] += 1

        await self.gpu_semaphores[selected_gpu].acquire()

        async with self._lock:
            self._gpu_pending[selected_gpu] -= 1
            self._gpu_in_use[selected_gpu] += 1

        return selected_gpu, time.monotonic() - queued_at

    async def run_gpu(self, fn: Callable[[int], Any]) -> tuple[Any, Dict[str, Any]]:
        selected_gpu, wait_s = await self._acquire_gpu()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: fn(selected_gpu))
            return result, {
                "resource_kind": "gpu",
                "assigned_gpu_id": selected_gpu,
                "queue_wait_s": wait_s,
            }
        finally:
            async with self._lock:
                self._gpu_in_use[selected_gpu] = max(0, self._gpu_in_use[selected_gpu] - 1)
            self.gpu_semaphores[selected_gpu].release()

    def status(self, session_count: int) -> KernelToolSchedulerStatus:
        return KernelToolSchedulerStatus(
            cpu_slots=self.cpu_slots,
            cpu_slots_in_use=self._cpu_in_use,
            cpu_slots_pending=self._cpu_pending,
            cpu_executor_max_workers=self.cpu_slots,
            gpu_slots={
                str(gpu_id): {
                    "in_use": self._gpu_in_use[gpu_id],
                    "pending": self._gpu_pending[gpu_id],
                }
                for gpu_id in self.gpu_ids
            },
            session_count=session_count,
        )


@dataclass
class KernelToolSession:
    session_id: str
    reference: KernelReferenceBundle
    budget: KernelToolBudget
    root_dir: str
    created_at: float
    updated_at: float
    tool_calls_used: int = 0
    current_candidate_code: str = ""
    current_candidate_hash: str = ""
    current_artifact_id: str = ""
    current_kernel_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_observation: Optional[KernelToolObservation] = None
    observation_cache: Dict[str, KernelToolObservation] = field(default_factory=dict)


class KernelToolRuntime:
    def __init__(self, settings: Optional[EvalSettings] = None):
        self.settings = settings or load_eval_settings()
        self.quick_perf_iterations = max(1, int(os.environ.get("HIP_TOOL_QUICK_PERF_ITERATIONS", "5")))
        self.profile_perf_iterations = max(
            self.quick_perf_iterations,
            int(os.environ.get("HIP_TOOL_PROFILE_PERF_ITERATIONS", str(max(10, self.quick_perf_iterations)))),
        )
        self.session_ttl_s = max(60, int(os.environ.get("HIP_TOOL_SESSION_TTL_S", "7200")))
        tool_gpu_ids_raw = os.environ.get("HIP_TOOL_GPU_IDS", "")
        tool_gpu_ids = parse_gpu_ids(tool_gpu_ids_raw) if tool_gpu_ids_raw.strip() else self.settings.gpu_ids
        if not tool_gpu_ids:
            tool_gpu_ids = list(self.settings.gpu_ids)
        default_cpu_jobs = getattr(self.settings, "compile_cpu_slots", max(1, min(8, len(tool_gpu_ids) * 2 or 2)))
        max_cpu_jobs = int(os.environ.get("HIP_TOOL_MAX_CPU_JOBS", str(default_cpu_jobs)))
        self.scheduler = EvalScheduler(gpu_ids=tool_gpu_ids, max_cpu_jobs=max_cpu_jobs)
        self.sessions: Dict[str, KernelToolSession] = {}
        self.runtime_root = Path(__file__).resolve().parent.parent / "runtime" / "tool_sessions"
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self._session_lock = asyncio.Lock()

    async def _cleanup_expired_sessions(self) -> None:
        now = time.time()
        expired_ids = []
        async with self._session_lock:
            for session_id, session in self.sessions.items():
                if (now - session.updated_at) > self.session_ttl_s:
                    expired_ids.append(session_id)
            for session_id in expired_ids:
                self.sessions.pop(session_id, None)

    def _session_budget_snapshot(self, session: KernelToolSession) -> Dict[str, Any]:
        now = time.time()
        wallclock_s = max(0.0, now - session.created_at)
        max_calls = int(session.budget.max_tool_calls)
        remaining_calls = max(0, max_calls - session.tool_calls_used)
        max_wallclock_s = float(session.budget.max_wallclock_s)
        wallclock_remaining_s = max(0.0, max_wallclock_s - wallclock_s)
        return {
            "max_tool_calls": max_calls,
            "tool_calls_used": session.tool_calls_used,
            "tool_calls_remaining": remaining_calls,
            "max_wallclock_s": max_wallclock_s,
            "wallclock_s": wallclock_s,
            "wallclock_remaining_s": wallclock_remaining_s,
        }

    def _candidate_identity(self, session: KernelToolSession, hip_code: str, kernel_name: Optional[str]) -> tuple[str, str, str]:
        candidate_hash = _md5_short(hip_code)
        logical_kernel_name = (kernel_name or session.reference.kernel_name or "kernel").strip() or "kernel"
        if logical_kernel_name.endswith(f"_{candidate_hash}"):
            candidate_kernel_name = logical_kernel_name
        else:
            candidate_kernel_name = f"{logical_kernel_name}_{candidate_hash}"
        return candidate_hash, candidate_hash, candidate_kernel_name

    def _artifact_root(self, session: KernelToolSession, artifact_id: str) -> str:
        path = os.path.join(session.root_dir, "artifacts", artifact_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _cache_key(
        self,
        *,
        operation: str,
        artifact_id: str,
        compile_timeout_s: Optional[int],
        run_timeout_s: Optional[int],
        perf_iterations: Optional[int],
    ) -> str:
        return _stable_json(
            {
                "operation": operation,
                "artifact_id": artifact_id,
                "compile_timeout_s": compile_timeout_s,
                "run_timeout_s": run_timeout_s,
                "perf_iterations": perf_iterations,
            }
        )

    async def create_session(self, request: KernelToolCreateSessionRequest) -> KernelToolCreateSessionResponse:
        await self._cleanup_expired_sessions()
        session_id = request.session_id or uuid4().hex
        async with self._session_lock:
            existing = self.sessions.get(session_id)
            if existing is None:
                root_dir = str(self.runtime_root / session_id)
                os.makedirs(root_dir, exist_ok=True)
                existing = KernelToolSession(
                    session_id=session_id,
                    reference=request.reference,
                    budget=request.budget,
                    root_dir=root_dir,
                    created_at=time.time(),
                    updated_at=time.time(),
                )
                self.sessions[session_id] = existing
        return KernelToolCreateSessionResponse(
            session_id=session_id,
            kernel_name=existing.reference.kernel_name,
            budget=self._session_budget_snapshot(existing),
        )

    async def release_session(self, session_id: str) -> None:
        async with self._session_lock:
            self.sessions.pop(session_id, None)

    async def _get_session(self, session_id: str) -> KernelToolSession:
        await self._cleanup_expired_sessions()
        async with self._session_lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown tool session: {session_id}")
        return session

    async def update_candidate(self, request: KernelToolUpdateCandidateRequest) -> KernelToolUpdateCandidateResponse:
        session = await self._get_session(request.session_id)
        candidate_hash, artifact_id, kernel_name = self._candidate_identity(session, request.hip_code, request.kernel_name)
        updated = artifact_id != session.current_artifact_id or request.hip_code != session.current_candidate_code
        artifact_root = self._artifact_root(session, artifact_id)
        if updated:
            with open(os.path.join(artifact_root, "candidate.hip"), "w", encoding="utf-8") as handle:
                handle.write(request.hip_code)
            with open(os.path.join(artifact_root, "metadata.json"), "w", encoding="utf-8") as handle:
                json.dump(request.metadata, handle, ensure_ascii=True, sort_keys=True, indent=2)
        session.current_candidate_code = request.hip_code
        session.current_candidate_hash = candidate_hash
        session.current_artifact_id = artifact_id
        session.current_kernel_name = kernel_name
        session.updated_at = time.time()
        if request.metadata:
            session.metadata.update(request.metadata)
        return KernelToolUpdateCandidateResponse(
            session_id=session.session_id,
            artifact_id=artifact_id,
            kernel_name=kernel_name,
            updated=updated,
            candidate_hash=candidate_hash,
            message="candidate updated" if updated else "candidate unchanged",
        )

    def _build_eval_request(
        self,
        session: KernelToolSession,
        *,
        compile_timeout_s: Optional[int],
        run_timeout_s: Optional[int],
    ) -> EvalRequest:
        return EvalRequest(
            kernel_name=session.current_kernel_name,
            hip_code=session.current_candidate_code,
            hip_ref_code=session.reference.hip_ref_code,
            pytorch_module_code=session.reference.pytorch_module_code,
            pytorch_functional_code=session.reference.pytorch_functional_code,
            atol=session.reference.atol,
            rtol=session.reference.rtol,
            compile_timeout_s=compile_timeout_s if compile_timeout_s is not None else session.reference.compile_timeout_s,
            run_timeout_s=run_timeout_s if run_timeout_s is not None else session.reference.run_timeout_s,
        )

    def _limit_observation(self, session: KernelToolSession, operation: str, reason: str) -> KernelToolObservation:
        budget = self._session_budget_snapshot(session)
        return KernelToolObservation(
            session_id=session.session_id,
            operation=operation,
            status="rejected",
            artifact_id=session.current_artifact_id or None,
            kernel_name=session.current_kernel_name or session.reference.kernel_name,
            candidate_hash=session.current_candidate_hash or None,
            reason=reason,
            observation=reason,
            budget=budget,
            metadata={"limit_exceeded": True},
        )

    def _claim_tool_budget(self, session: KernelToolSession, operation: str) -> Optional[KernelToolObservation]:
        budget = self._session_budget_snapshot(session)
        if not session.current_candidate_code:
            return self._limit_observation(session, operation, "No candidate code is attached to the session.")
        if budget["tool_calls_remaining"] <= 0:
            return self._limit_observation(session, operation, "Tool call budget exhausted for this rollout session.")
        if budget["wallclock_remaining_s"] <= 0:
            return self._limit_observation(session, operation, "Wallclock budget exhausted for this rollout session.")
        session.tool_calls_used += 1
        session.updated_at = time.time()
        return None

    def _format_observation_text(self, operation: str, result: KernelToolObservation) -> str:
        summary = [f"operation={operation}", f"status={result.status}"]
        if result.compile_ok is not None:
            summary.append(f"compile_ok={result.compile_ok}")
        if result.run_ok is not None:
            summary.append(f"run_ok={result.run_ok}")
        if result.match_ok is not None:
            summary.append(f"match_ok={result.match_ok}")
        if result.speedup is not None:
            summary.append(f"speedup={result.speedup:.4f}x")
        if result.reason:
            summary.append(f"reason={result.reason}")
        return "; ".join(summary)

    def _build_observation(
        self,
        session: KernelToolSession,
        *,
        operation: str,
        result,
        cached: bool,
        scheduler_meta: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KernelToolObservation:
        timing = dict(result.timing)
        timing.update(scheduler_meta)
        observation = KernelToolObservation(
            session_id=session.session_id,
            operation=operation,
            status="ok" if result.compile_ok else "error",
            artifact_id=session.current_artifact_id,
            kernel_name=session.current_kernel_name,
            candidate_hash=session.current_candidate_hash,
            cached=cached,
            compile_ok=result.compile_ok,
            run_ok=result.run_ok,
            match_ok=result.match_ok,
            speedup=result.speedup,
            reason=str(result.timing.get("failure_reason") or result.timing.get("failure_detail") or ""),
            timing=timing,
            budget=self._session_budget_snapshot(session),
            metadata=metadata or {},
        )
        observation.observation = self._format_observation_text(operation, observation)
        return observation

    async def _update_candidate_if_needed(self, request: KernelToolActionRequest) -> Optional[KernelToolUpdateCandidateResponse]:
        if request.hip_code is None:
            return None
        return await self.update_candidate(
            KernelToolUpdateCandidateRequest(
                session_id=request.session_id,
                hip_code=request.hip_code,
                kernel_name=request.kernel_name,
                metadata=request.metadata,
            )
        )

    async def _run_operation(
        self,
        *,
        request: KernelToolActionRequest,
        operation: str,
        perf_iterations: Optional[int],
    ) -> KernelToolObservation:
        await self._update_candidate_if_needed(request)
        session = await self._get_session(request.session_id)
        rejected = self._claim_tool_budget(session, operation)
        if rejected is not None:
            session.last_observation = rejected
            return rejected

        cache_key = self._cache_key(
            operation=operation,
            artifact_id=session.current_artifact_id,
            compile_timeout_s=request.compile_timeout_s,
            run_timeout_s=request.run_timeout_s,
            perf_iterations=perf_iterations,
        )
        cached = session.observation_cache.get(cache_key)
        if cached is not None:
            cached_copy = cached.model_copy(deep=True)
            cached_copy.cached = True
            cached_copy.budget = self._session_budget_snapshot(session)
            cached_copy.observation = self._format_observation_text(operation, cached_copy)
            session.last_observation = cached_copy
            return cached_copy

        eval_request = self._build_eval_request(
            session,
            compile_timeout_s=request.compile_timeout_s,
            run_timeout_s=request.run_timeout_s,
        )

        if operation == "compile_check":
            def job_fn():
                tmp_dir = os.path.join(self._artifact_root(session, session.current_artifact_id), operation)
                os.makedirs(tmp_dir, exist_ok=True)
                return run_compile_request(
                    eval_request,
                    tmp_dir=tmp_dir,
                    settings=self.settings,
                )

            result, scheduler_meta = await self.scheduler.run_cpu(job_fn)
        else:
            op_settings = replace(
                self.settings,
                perf_iterations=max(1, perf_iterations or self.quick_perf_iterations),
                speedup_confirm_enabled=False,
            )

            def job_fn(gpu_id: int):
                tmp_dir = os.path.join(self._artifact_root(session, session.current_artifact_id), operation)
                os.makedirs(tmp_dir, exist_ok=True)
                return run_eval_request(
                    eval_request,
                    tmp_dir=tmp_dir,
                    gpu_id=gpu_id,
                    settings=op_settings,
                )

            result, scheduler_meta = await self.scheduler.run_gpu(job_fn)

        observation = self._build_observation(
            session,
            operation=operation,
            result=result,
            cached=False,
            scheduler_meta=scheduler_meta,
            metadata={"perf_iterations": perf_iterations} if perf_iterations is not None else {},
        )
        session.observation_cache[cache_key] = observation.model_copy(deep=True)
        session.last_observation = observation
        return observation

    async def compile_check(self, request: KernelToolActionRequest) -> KernelToolObservation:
        return await self._run_operation(
            request=request,
            operation="compile_check",
            perf_iterations=None,
        )

    async def correctness_quick(self, request: KernelToolActionRequest) -> KernelToolObservation:
        perf_iterations = request.perf_iterations if request.perf_iterations is not None else self.quick_perf_iterations
        return await self._run_operation(
            request=request,
            operation="correctness_quick",
            perf_iterations=perf_iterations,
        )

    async def profile_quick(self, request: KernelToolActionRequest) -> KernelToolObservation:
        perf_iterations = request.perf_iterations if request.perf_iterations is not None else self.profile_perf_iterations
        return await self._run_operation(
            request=request,
            operation="profile_quick",
            perf_iterations=perf_iterations,
        )

    async def extract_kernel_diagnostics(self, request: KernelToolActionRequest) -> KernelToolDiagnosticsResponse:
        await self._update_candidate_if_needed(request)
        session = await self._get_session(request.session_id)
        budget = self._session_budget_snapshot(session)
        return KernelToolDiagnosticsResponse(
            session_id=session.session_id,
            artifact_id=session.current_artifact_id or None,
            kernel_name=session.current_kernel_name or session.reference.kernel_name,
            candidate_hash=session.current_candidate_hash or None,
            tool_calls_used=session.tool_calls_used,
            tool_calls_remaining=budget["tool_calls_remaining"],
            wallclock_s=budget["wallclock_s"],
            wallclock_remaining_s=budget["wallclock_remaining_s"],
            last_observation=session.last_observation,
            metadata=dict(session.metadata),
        )

    async def finalize_candidate(self, request: KernelToolActionRequest) -> KernelToolFinalizeResponse:
        await self._update_candidate_if_needed(request)
        session = await self._get_session(request.session_id)
        if not session.current_candidate_code:
            raise ValueError("No candidate code is attached to the session.")
        eval_request = self._build_eval_request(
            session,
            compile_timeout_s=request.compile_timeout_s,
            run_timeout_s=request.run_timeout_s,
        )
        return KernelToolFinalizeResponse(
            session_id=session.session_id,
            artifact_id=session.current_artifact_id,
            kernel_name=session.current_kernel_name,
            candidate_hash=session.current_candidate_hash,
            eval_request=eval_request.model_dump(),
            last_observation=session.last_observation,
        )

    def scheduler_status(self) -> KernelToolSchedulerStatus:
        return self.scheduler.status(session_count=len(self.sessions))
