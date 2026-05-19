from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import requests

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema


class HIPKernelEvalTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        if tool_schema is None:
            tool_schema = self.get_openai_tool_schema()
        super().__init__(config, tool_schema)
        self.server_url = str(config.get("server_url", "")).rstrip("/")
        if not self.server_url:
            raise ValueError("HIPKernelEvalTool requires `server_url` in config.")
        self.request_timeout_s = int(config.get("request_timeout_s", 300))
        self._instance_dict: Dict[str, Dict[str, Any]] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "kernel_eval",
                    "description": "Evaluate a HIP kernel candidate with session-aware compile, correctness, and quick profiling checks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [
                                    "compile_check",
                                    "correctness_quick",
                                    "profile_quick",
                                    "extract_kernel_diagnostics",
                                    "finalize_candidate",
                                ],
                                "description": "Which kernel evaluation action to run.",
                            },
                            "hip_code": {
                                "type": "string",
                                "description": "Optional full HIP kernel candidate. If provided, it becomes the current candidate before the action runs.",
                            },
                            "kernel_name": {
                                "type": "string",
                                "description": "Optional logical kernel name override for the current candidate.",
                            },
                            "perf_iterations": {
                                "type": "integer",
                                "description": "Optional override for quick performance iterations.",
                            },
                            "compile_timeout_s": {
                                "type": "integer",
                                "description": "Optional compile timeout override in seconds.",
                            },
                            "run_timeout_s": {
                                "type": "integer",
                                "description": "Optional execution timeout override in seconds.",
                            },
                        },
                        "required": ["action"],
                    },
                },
            }
        )

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.server_url}{path}",
            json=payload,
            timeout=self.request_timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def _delete(self, path: str) -> None:
        response = requests.delete(
            f"{self.server_url}{path}",
            timeout=self.request_timeout_s,
        )
        response.raise_for_status()

    def _normalize_create_payload(self, instance_id: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        reference = kwargs.get("reference") or kwargs.get("reference_bundle")
        if reference is None:
            raise ValueError("HIPKernelEvalTool.create requires `reference` or `reference_bundle` in create_kwargs.")
        budget = kwargs.get("budget", {})
        return {
            "session_id": instance_id,
            "reference": reference,
            "budget": budget,
        }

    def _render_tool_response(self, action: str, payload: Dict[str, Any]) -> str:
        if action == "finalize_candidate":
            compact = {
                "action": action,
                "status": "ready",
                "session_id": payload.get("session_id"),
                "artifact_id": payload.get("artifact_id"),
                "kernel_name": payload.get("kernel_name"),
                "candidate_hash": payload.get("candidate_hash"),
            }
            return json.dumps(compact, ensure_ascii=True, sort_keys=True)
        if action == "extract_kernel_diagnostics":
            compact = {
                "action": action,
                "session_id": payload.get("session_id"),
                "artifact_id": payload.get("artifact_id"),
                "kernel_name": payload.get("kernel_name"),
                "candidate_hash": payload.get("candidate_hash"),
                "tool_calls_used": payload.get("tool_calls_used"),
                "tool_calls_remaining": payload.get("tool_calls_remaining"),
                "wallclock_remaining_s": payload.get("wallclock_remaining_s"),
                "last_observation": payload.get("last_observation"),
            }
            return json.dumps(compact, ensure_ascii=True, sort_keys=True)
        compact = {
            "action": action,
            "status": payload.get("status"),
            "session_id": payload.get("session_id"),
            "artifact_id": payload.get("artifact_id"),
            "kernel_name": payload.get("kernel_name"),
            "candidate_hash": payload.get("candidate_hash"),
            "compile_ok": payload.get("compile_ok"),
            "run_ok": payload.get("run_ok"),
            "match_ok": payload.get("match_ok"),
            "speedup": payload.get("speedup"),
            "reason": payload.get("reason"),
            "budget": payload.get("budget"),
            "timing": payload.get("timing"),
            "observation": payload.get("observation"),
        }
        return json.dumps(compact, ensure_ascii=True, sort_keys=True)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        payload = self._normalize_create_payload(instance_id, kwargs)
        result = await asyncio.to_thread(self._post_json, "/tool/create_session", payload)
        self._instance_dict[instance_id] = {
            "session_id": result["session_id"],
            "create_kwargs": kwargs,
        }
        return result["session_id"]

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        action = str(parameters.get("action", "")).strip()
        if not action:
            return json.dumps({"error": "Missing required parameter `action`."}), 0.0, {}
        route = {
            "compile_check": "/tool/compile_check",
            "correctness_quick": "/tool/correctness_quick",
            "profile_quick": "/tool/profile_quick",
            "extract_kernel_diagnostics": "/tool/extract_kernel_diagnostics",
            "finalize_candidate": "/tool/finalize_candidate",
        }.get(action)
        if route is None:
            return json.dumps({"error": f"Unsupported action: {action}"}), 0.0, {}

        payload = {
            "session_id": instance_id,
            "hip_code": parameters.get("hip_code"),
            "kernel_name": parameters.get("kernel_name"),
            "perf_iterations": parameters.get("perf_iterations"),
            "compile_timeout_s": parameters.get("compile_timeout_s"),
            "run_timeout_s": parameters.get("run_timeout_s"),
            "metadata": {"tool_name": self.name},
        }

        result = await asyncio.to_thread(self._post_json, route, payload)
        return (
            self._render_tool_response(action, result),
            0.0,
            {
                "tool_name": self.name,
                "tool_action": action,
                "tool_status": result.get("status", "unknown"),
            },
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
        try:
            await asyncio.to_thread(self._delete, f"/tool/session/{instance_id}")
        except Exception:
            # Session release is best-effort so late cleanup failures do not poison rollout teardown.
            return
