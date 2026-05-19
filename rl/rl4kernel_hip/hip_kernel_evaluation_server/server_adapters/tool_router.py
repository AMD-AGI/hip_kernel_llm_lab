from __future__ import annotations

from fastapi import APIRouter, HTTPException

from sandbox_core.config import load_eval_settings
from sandbox_core.tool_protocol import (
    KernelToolActionRequest,
    KernelToolCreateSessionRequest,
    KernelToolCreateSessionResponse,
    KernelToolDiagnosticsResponse,
    KernelToolFinalizeResponse,
    KernelToolObservation,
    KernelToolSchedulerStatus,
    KernelToolUpdateCandidateRequest,
    KernelToolUpdateCandidateResponse,
)
from sandbox_core.tool_runtime import KernelToolRuntime


router = APIRouter(prefix="/tool", tags=["kernel-tool"])
tool_runtime = KernelToolRuntime(load_eval_settings())


def _as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.get("/status", response_model=KernelToolSchedulerStatus)
async def tool_status():
    return tool_runtime.scheduler_status()


@router.post("/create_session", response_model=KernelToolCreateSessionResponse)
async def create_session(request: KernelToolCreateSessionRequest):
    try:
        return await tool_runtime.create_session(request)
    except Exception as exc:  # pragma: no cover - defensive mapping
        raise _as_http_error(exc) from exc


@router.delete("/session/{session_id}")
async def release_session(session_id: str):
    await tool_runtime.release_session(session_id)
    return {"session_id": session_id, "released": True}


@router.post("/update_candidate", response_model=KernelToolUpdateCandidateResponse)
async def update_candidate(request: KernelToolUpdateCandidateRequest):
    try:
        return await tool_runtime.update_candidate(request)
    except Exception as exc:  # pragma: no cover - defensive mapping
        raise _as_http_error(exc) from exc


@router.post("/compile_check", response_model=KernelToolObservation)
async def compile_check(request: KernelToolActionRequest):
    try:
        return await tool_runtime.compile_check(request)
    except Exception as exc:  # pragma: no cover - defensive mapping
        raise _as_http_error(exc) from exc


@router.post("/correctness_quick", response_model=KernelToolObservation)
async def correctness_quick(request: KernelToolActionRequest):
    try:
        return await tool_runtime.correctness_quick(request)
    except Exception as exc:  # pragma: no cover - defensive mapping
        raise _as_http_error(exc) from exc


@router.post("/profile_quick", response_model=KernelToolObservation)
async def profile_quick(request: KernelToolActionRequest):
    try:
        return await tool_runtime.profile_quick(request)
    except Exception as exc:  # pragma: no cover - defensive mapping
        raise _as_http_error(exc) from exc


@router.post("/extract_kernel_diagnostics", response_model=KernelToolDiagnosticsResponse)
async def extract_kernel_diagnostics(request: KernelToolActionRequest):
    try:
        return await tool_runtime.extract_kernel_diagnostics(request)
    except Exception as exc:  # pragma: no cover - defensive mapping
        raise _as_http_error(exc) from exc


@router.post("/finalize_candidate", response_model=KernelToolFinalizeResponse)
async def finalize_candidate(request: KernelToolActionRequest):
    try:
        return await tool_runtime.finalize_candidate(request)
    except Exception as exc:  # pragma: no cover - defensive mapping
        raise _as_http_error(exc) from exc
