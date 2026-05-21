from typing import Dict, List, Optional

from .prompts import (
    DEFAULT_TARGET_GPU,
    DEFAULT_OPTIMIZATION_PARADIGM,
    HIP2HIP_FULL_FILE_PARADIGM,
    HIP_FULL_FILE_LEGACY_CODE_FORMAT_WITH_STARTER_CODE,
    HIP_FULL_FILE_LEGACY_CODE_FORMAT_WITHOUT_STARTER_CODE,
    HIP_FULL_FILE_REASONING_AND_CODE_FORMAT_WITH_STARTER_CODE,
    HIP_FULL_FILE_REASONING_AND_CODE_FORMAT_WITHOUT_STARTER_CODE,
    HIP_FULL_FILE_REASONING_AND_JSON_RESPONSE_FORMAT_WITH_STARTER_CODE,
    HIP_FULL_FILE_REASONING_AND_JSON_RESPONSE_FORMAT_WITHOUT_STARTER_CODE,
    HIP_LEGACY_CODE_FORMAT_WITH_STARTER_CODE,
    HIP_LEGACY_CODE_FORMAT_WITHOUT_STARTER_CODE,
    HIP_REASONING_AND_CODE_FORMAT_WITH_STARTER_CODE,
    HIP_REASONING_AND_CODE_FORMAT_WITHOUT_STARTER_CODE,
    HIP_REASONING_AND_JSON_RESPONSE_FORMAT_WITH_STARTER_CODE,
    HIP_REASONING_AND_JSON_RESPONSE_FORMAT_WITHOUT_STARTER_CODE,
    get_prompt_template,
    normalize_optimization_paradigm,
)


DEFAULT_OUTPUT_CONTRACT = "sample_json_v1"


def fetch_hip_kernel_agent_system_prompt(
    prompt: str,
    starter_code: str = None,
    target_gpu: str = DEFAULT_TARGET_GPU,
    output_contract: str = DEFAULT_OUTPUT_CONTRACT,
    optimization_paradigm: str = DEFAULT_OPTIMIZATION_PARADIGM,
):
    normalized_output_contract = (output_contract or DEFAULT_OUTPUT_CONTRACT).strip().lower()
    if normalized_output_contract in {"", "auto", "none"}:
        normalized_output_contract = DEFAULT_OUTPUT_CONTRACT
    normalized_paradigm = normalize_optimization_paradigm(optimization_paradigm)
    use_full_file = normalized_paradigm == HIP2HIP_FULL_FILE_PARADIGM
    use_json_output = normalized_output_contract in {
        "sample_json_v1",
        "sample_json",
        "sample-json",
        "json",
    }
    use_hip_fence_output = normalized_output_contract in {
        "legacy_hip_fence_v1",
        "legacy_hip_fence",
        "legacy-hip-fence",
        "hip_fence",
        "fenced_hip",
    }
    if not use_json_output and not use_hip_fence_output:
        raise ValueError(
            f"Unsupported output_contract {output_contract!r}. "
            "Expected a legacy hip fence or sample_json variant."
        )

    base_prompt = get_prompt_template(target_gpu, normalized_paradigm)
    if use_hip_fence_output:
        base_prompt = base_prompt.replace("\nYou are working in think mode.\n", "\n")
    prompt_sections = [base_prompt]
    if prompt and prompt.strip():
        prompt_sections.append(prompt)

    json_with_starter_format = (
        HIP_FULL_FILE_REASONING_AND_JSON_RESPONSE_FORMAT_WITH_STARTER_CODE
        if use_full_file
        else HIP_REASONING_AND_JSON_RESPONSE_FORMAT_WITH_STARTER_CODE
    )
    json_without_starter_format = (
        HIP_FULL_FILE_REASONING_AND_JSON_RESPONSE_FORMAT_WITHOUT_STARTER_CODE
        if use_full_file
        else HIP_REASONING_AND_JSON_RESPONSE_FORMAT_WITHOUT_STARTER_CODE
    )
    fence_with_starter_format = (
        (
            HIP_FULL_FILE_LEGACY_CODE_FORMAT_WITH_STARTER_CODE
            if use_hip_fence_output
            else HIP_FULL_FILE_REASONING_AND_CODE_FORMAT_WITH_STARTER_CODE
        )
        if use_full_file
        else (
            HIP_LEGACY_CODE_FORMAT_WITH_STARTER_CODE
            if use_hip_fence_output
            else HIP_REASONING_AND_CODE_FORMAT_WITH_STARTER_CODE
        )
    )
    fence_without_starter_format = (
        (
            HIP_FULL_FILE_LEGACY_CODE_FORMAT_WITHOUT_STARTER_CODE
            if use_hip_fence_output
            else HIP_FULL_FILE_REASONING_AND_CODE_FORMAT_WITHOUT_STARTER_CODE
        )
        if use_full_file
        else (
            HIP_LEGACY_CODE_FORMAT_WITHOUT_STARTER_CODE
            if use_hip_fence_output
            else HIP_REASONING_AND_CODE_FORMAT_WITHOUT_STARTER_CODE
        )
    )
    starter_heading = "### Starter HIP File (reference)" if use_full_file else "### Starter Code (reference)"
    code_description = "complete optimized .hip source file" if use_full_file else "full optimized function"

    if use_json_output:
        if starter_code:
            prompt_sections.append(
                f"### Format: {json_with_starter_format}"
            )
            prompt_sections.append(
                f"{starter_heading}\n"
                f"```hip\n{starter_code}\n```"
            )
        else:
            prompt_sections.append(
                f"### Format: {json_without_starter_format}"
            )
            if use_full_file:
                prompt_sections.append(
                    '{"thought": "concise optimization summary", "code": "#include <hip/hip_runtime.h>\\n// optimized implementation\\n"}'
                )
            else:
                prompt_sections.append(
                    '{"thought": "concise optimization summary", "code": "__global__ void your_kernel(/* keep original signature when provided */) {\\n    // optimized implementation\\n}"}'
                )

        prompt_sections.append(
            "### Answer Order (strict)\n"
            "1. First do optimization reasoning.\n"
            "2. Then output exactly one JSON object with `thought` and `code` fields.\n"
            f"3. The `code` field must contain the {code_description} with no markdown fence.\n"
            "4. Do not add any text after the closing `}`."
        )
    else:
        if starter_code:
            prompt_sections.append(
                f"### Format: {fence_with_starter_format}"
            )
            prompt_sections.append(
                f"{starter_heading}\n"
                f"```hip\n{starter_code}\n```"
            )
        else:
            prompt_sections.append(
                f"### Format: {fence_without_starter_format}"
            )
            if use_full_file:
                prompt_sections.append(
                    "```hip\n"
                    "#include <hip/hip_runtime.h>\n"
                    "// optimized implementation\n"
                    "```"
                )
            else:
                prompt_sections.append(
                    "```hip\n"
                    "__global__ void your_kernel(/* keep original signature when provided */) {\n"
                    "    // optimized implementation\n"
                    "}\n"
                    "```"
                )

        prompt_sections.append(
            "### Answer Order (strict)\n"
            "1. Output exactly one fenced HIP code block using ```hip ... ```.\n"
            "2. Do not include reasoning, JSON, or any prose outside the code block.\n"
            "3. Do not add any text after the closing code fence."
        )
    return "\n\n".join(prompt_sections)


def build_hip_kernel_agent_chat_messages(
    prompt: str = "",
    starter_code: Optional[str] = None,
    target_gpu: str = DEFAULT_TARGET_GPU,
    output_contract: str = DEFAULT_OUTPUT_CONTRACT,
    optimization_paradigm: str = DEFAULT_OPTIMIZATION_PARADIGM,
) -> List[Dict[str, str]]:
    """Build the canonical user-only chat payload for kernel-agent generation."""
    return [
        {
            "role": "user",
            "content": fetch_hip_kernel_agent_system_prompt(
                prompt=prompt,
                starter_code=starter_code,
                target_gpu=target_gpu,
                output_contract=output_contract,
                optimization_paradigm=optimization_paradigm,
            ),
        }
    ]


def build_hip_kernel_agent_multiturn_chat_messages(
    *,
    previous_thought: str,
    original_starter_code: str,
    previous_generated_code: str,
    previous_feedback: str,
    prompt: str = "",
    target_gpu: str = DEFAULT_TARGET_GPU,
    output_contract: str = DEFAULT_OUTPUT_CONTRACT,
    optimization_paradigm: str = DEFAULT_OPTIMIZATION_PARADIGM,
) -> List[Dict[str, str]]:
    """Build the explicit feedback prompt for rounds after the first."""
    normalized_output_contract = (output_contract or DEFAULT_OUTPUT_CONTRACT).strip().lower()
    normalized_paradigm = normalize_optimization_paradigm(optimization_paradigm)
    use_full_file = normalized_paradigm == HIP2HIP_FULL_FILE_PARADIGM
    if normalized_output_contract not in {
        "sample_json_v1",
        "sample_json",
        "sample-json",
        "json",
    }:
        raise ValueError("Multi-turn feedback prompts require the JSON output contract.")

    original_heading = "### Original HIP File" if use_full_file else "### Original Starter Code"
    previous_heading = "### Previous Generated HIP File" if use_full_file else "### Previous Generated HIP Code"
    feedback_heading = (
        "### Previous HIP Candidate Profiling And Eval Results"
        if use_full_file
        else "### Previous Kernel Profiling And Eval Results"
    )
    action_text = (
        "Use the previous summary, generated HIP file, and profiling/eval evidence above. "
        "If correctness failed, fix correctness first. Otherwise, preserve correctness "
        "and target the observed bottleneck with a conservative HIP-level optimization."
        if use_full_file
        else (
            "Use the previous summary, generated code, and profiling/eval evidence above. "
            "If correctness failed, fix correctness first. Otherwise, preserve correctness "
            "and target the observed bottleneck with a conservative body-only optimization."
        )
    )
    format_text = (
        HIP_FULL_FILE_REASONING_AND_JSON_RESPONSE_FORMAT_WITH_STARTER_CODE
        if use_full_file
        else HIP_REASONING_AND_JSON_RESPONSE_FORMAT_WITH_STARTER_CODE
    )
    code_description = "complete optimized .hip source file" if use_full_file else "full optimized function"
    prompt_sections = [
        get_prompt_template(target_gpu, normalized_paradigm),
    ]
    if prompt and prompt.strip():
        prompt_sections.append("### Task-Specific Context\n" + prompt.strip())
    prompt_sections.extend(
        [
            "### Previous Round Optimization Summary\n" + (previous_thought or "n/a").strip(),
            f"{original_heading}\n" f"```hip\n{original_starter_code or ''}\n```",
            f"{previous_heading}\n" f"```hip\n{previous_generated_code or ''}\n```",
            (
                f"{feedback_heading}\n"
                f"{(previous_feedback or 'n/a').strip()}\n\n"
                f"{action_text}"
            ),
            f"### Format: {format_text}",
            (
                "### Answer Order (strict)\n"
                "1. First do optimization reasoning.\n"
                "2. Then output exactly one JSON object with `thought` and `code` fields.\n"
                f"3. The `code` field must contain the {code_description} with no markdown fence.\n"
                "4. Do not add any text after the closing `}`."
            ),
        ]
    )
    return [{"role": "user", "content": "\n\n".join(prompt_sections)}]