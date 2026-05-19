"""Shared dataset-side prompt and parsing utilities."""

from .contracts import (
    FULL_FILE_PERSISTENCE_MODE,
    HIP2HIP_TRAIN_DATA_SOURCE,
    HIP_TRANSLATION_UNIT_CODE_UNIT,
    KERNEL_FUNCTION_CODE_UNIT,
    SAMPLE_JSON_OUTPUT_CONTRACT,
    SPLICE_PERSISTENCE_MODE,
    OptimizationContract,
    contract_to_extra_info,
    expected_code_unit_for_paradigm,
    infer_optimization_paradigm,
    resolve_optimization_contract,
    validate_training_row_contract,
)
from .prompts import (
    DEFAULT_OPTIMIZATION_PARADIGM,
    DEFAULT_TARGET_GPU,
    GPU_HARDWARE_CONFIGS,
    HIP2HIP_FULL_FILE_PARADIGM,
    KERNEL2KERNEL_SPLICE_PARADIGM,
    get_kernel_agent_prompt_template,
    get_prompt_template,
    normalize_optimization_paradigm,
)
from .response_parser import (
    KERNEL_AGENT_DEFAULT_DATA_SOURCE,
    build_parse_attempt_chain,
    parse_kernel_agent_generation_response,
)
from .utils import (
    build_hip_kernel_agent_chat_messages,
    build_hip_kernel_agent_multiturn_chat_messages,
    fetch_hip_kernel_agent_system_prompt,
)

__all__ = [
    "FULL_FILE_PERSISTENCE_MODE",
    "HIP2HIP_TRAIN_DATA_SOURCE",
    "HIP_TRANSLATION_UNIT_CODE_UNIT",
    "KERNEL_FUNCTION_CODE_UNIT",
    "SAMPLE_JSON_OUTPUT_CONTRACT",
    "SPLICE_PERSISTENCE_MODE",
    "OptimizationContract",
    "contract_to_extra_info",
    "expected_code_unit_for_paradigm",
    "infer_optimization_paradigm",
    "resolve_optimization_contract",
    "validate_training_row_contract",
    "DEFAULT_TARGET_GPU",
    "DEFAULT_OPTIMIZATION_PARADIGM",
    "KERNEL2KERNEL_SPLICE_PARADIGM",
    "HIP2HIP_FULL_FILE_PARADIGM",
    "GPU_HARDWARE_CONFIGS",
    "get_kernel_agent_prompt_template",
    "get_prompt_template",
    "normalize_optimization_paradigm",
    "KERNEL_AGENT_DEFAULT_DATA_SOURCE",
    "build_parse_attempt_chain",
    "parse_kernel_agent_generation_response",
    "build_hip_kernel_agent_chat_messages",
    "build_hip_kernel_agent_multiturn_chat_messages",
    "fetch_hip_kernel_agent_system_prompt",
]
