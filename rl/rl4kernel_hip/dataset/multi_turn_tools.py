from __future__ import annotations

from typing import Any, Dict


KERNEL_TOOL_NAME = "kernel_eval"
SUPPORTED_KERNEL_DATA_SOURCES = {
    "torch2hip-train",
    "hip2hip-train",
    "kernel2kernel-train",
    "kernel-agent-single-sft-train",
    "kernel-agent-react-train",
}


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _build_reference_bundle(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    reward_model = _coerce_dict(row_dict.get("reward_model"))
    ground_truth = _coerce_dict(reward_model.get("ground_truth"))
    extra_info = _coerce_dict(row_dict.get("extra_info"))
    problem_id = (
        extra_info.get("kernel_id_name")
        or extra_info.get("sample_id")
        or extra_info.get("kernel_name")
        or ground_truth.get("kernel_name")
        or "unknown"
    )
    metadata = {
        "target_gpu": extra_info.get("target_gpu"),
        "source_type": extra_info.get("source_type"),
        "paths": extra_info.get("paths", {}),
        "reference_lookup": extra_info.get("reference_lookup", {}),
    }
    return {
        "problem_id": str(problem_id),
        "kernel_name": str(ground_truth.get("kernel_name") or extra_info.get("kernel_name") or "kernel"),
        "hip_ref_code": ground_truth.get("hip_code") or "",
        "pytorch_module_code": ground_truth.get("pytorch_module_code") or "",
        "pytorch_functional_code": ground_truth.get("pytorch_functional_code") or "",
        "atol": float(ground_truth.get("atol", 1e-4)),
        "rtol": float(ground_truth.get("rtol", 1e-3)),
        "compile_timeout_s": ground_truth.get("compile_timeout_s"),
        "run_timeout_s": ground_truth.get("run_timeout_s"),
        "metadata": metadata,
    }


def build_kernel_eval_tools_kwargs(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    data_source = str(row_dict.get("data_source") or "").strip()
    if data_source not in SUPPORTED_KERNEL_DATA_SOURCES:
        return {}

    extra_info = _coerce_dict(row_dict.get("extra_info"))
    reference = _build_reference_bundle(row_dict)
    max_tool_calls = int(extra_info.get("max_tool_calls", 4))
    max_wallclock_s = int(extra_info.get("max_tool_wallclock_s", 600))
    return {
        KERNEL_TOOL_NAME: {
            "create_kwargs": {
                "reference": reference,
                "budget": {
                    "max_tool_calls": max_tool_calls,
                    "max_wallclock_s": max_wallclock_s,
                },
            },
            "execute_kwargs": {},
            "calc_reward_kwargs": {},
            "release_kwargs": {},
        }
    }
