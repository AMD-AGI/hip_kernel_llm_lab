# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class KernelReferenceBundle(BaseModel):
    problem_id: Optional[str] = None
    kernel_name: str
    hip_ref_code: str
    pytorch_module_code: str = ""
    pytorch_functional_code: str = ""
    atol: float = 1e-4
    rtol: float = 1e-3
    compile_timeout_s: Optional[int] = None
    run_timeout_s: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KernelToolBudget(BaseModel):
    max_tool_calls: int = 8
    max_wallclock_s: int = 900


class KernelToolCreateSessionRequest(BaseModel):
    session_id: Optional[str] = None
    reference: KernelReferenceBundle
    budget: KernelToolBudget = Field(default_factory=KernelToolBudget)


class KernelToolCreateSessionResponse(BaseModel):
    session_id: str
    kernel_name: str
    budget: Dict[str, Any]
    status: str = "created"


class KernelToolUpdateCandidateRequest(BaseModel):
    session_id: str
    hip_code: str
    kernel_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KernelToolUpdateCandidateResponse(BaseModel):
    session_id: str
    artifact_id: str
    kernel_name: str
    updated: bool
    candidate_hash: str
    message: str


class KernelToolActionRequest(BaseModel):
    session_id: str
    hip_code: Optional[str] = None
    kernel_name: Optional[str] = None
    compile_timeout_s: Optional[int] = None
    run_timeout_s: Optional[int] = None
    perf_iterations: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KernelToolObservation(BaseModel):
    session_id: str
    operation: str
    status: str
    artifact_id: Optional[str] = None
    kernel_name: Optional[str] = None
    candidate_hash: Optional[str] = None
    cached: bool = False
    compile_ok: Optional[bool] = None
    run_ok: Optional[bool] = None
    match_ok: Optional[bool] = None
    speedup: Optional[float] = None
    reason: Optional[str] = None
    observation: str = ""
    timing: Dict[str, Any] = Field(default_factory=dict)
    budget: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KernelToolFinalizeResponse(BaseModel):
    session_id: str
    artifact_id: str
    kernel_name: str
    candidate_hash: str
    eval_request: Dict[str, Any]
    last_observation: Optional[KernelToolObservation] = None


class KernelToolDiagnosticsResponse(BaseModel):
    session_id: str
    artifact_id: Optional[str] = None
    kernel_name: Optional[str] = None
    candidate_hash: Optional[str] = None
    tool_calls_used: int
    tool_calls_remaining: int
    wallclock_s: float
    wallclock_remaining_s: float
    last_observation: Optional[KernelToolObservation] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KernelToolSchedulerStatus(BaseModel):
    cpu_slots: int
    cpu_slots_in_use: int
    cpu_slots_pending: int = 0
    cpu_executor_max_workers: int = 0
    gpu_slots: Dict[str, Dict[str, int]]
    session_count: int
