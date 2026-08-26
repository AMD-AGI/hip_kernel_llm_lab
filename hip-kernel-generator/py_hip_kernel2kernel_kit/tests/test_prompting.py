# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

from py_hip_kernel2kernel_kit.config import AttemptRecord
from py_hip_kernel2kernel_kit.hip_parser import extract_gpu_functions, select_optimization_target
from py_hip_kernel2kernel_kit.prompt_assets import resolve_prompt_assets
from py_hip_kernel2kernel_kit.prompting import build_prompt, format_response


HIP_SOURCE = """
__global__ void sample_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx];
    }
}
"""


def test_format_response_extracts_first_fenced_block() -> None:
    response = """
    Here is the optimized kernel.

    ```cpp
    __global__ void sample_kernel(const float* x, float* out, int n) {}
    ```
    """
    assert "__global__ void sample_kernel" in format_response(response)


def test_build_prompt_includes_baseline_context_and_attempt_history(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.cpp"
    candidate_path.write_text(
        "__global__ void sample_kernel(const float* x, float* out, int n) { out[0] = x[0]; }",
        encoding="utf-8",
    )

    attempts = [
        AttemptRecord(
            attempt=1,
            prompt_path="prompt_1.txt",
            function_candidate_path=candidate_path.as_posix(),
            candidate_path="candidate_1.hip",
            status="success",
            feedback="Correctness passed. Speedup vs baseline=1.2500x.",
            correctness_success=True,
            speedup_vs_baseline=1.25,
            baseline_latency_ms=2.0,
            candidate_latency_ms=1.6,
        )
    ]

    target = select_optimization_target(extract_gpu_functions(HIP_SOURCE))
    assert target is not None

    prompt = build_prompt(
        "class Model: pass",
        "def module_fn(x, fn=None): return x",
        HIP_SOURCE,
        Path("level_1/sample.hip"),
        target,
        attempts,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
        code_char_limit=1000,
        feedback_char_limit=1000,
        baseline_message="Baseline HIP correctness passed.",
        module_latency_ms=3.0,
        baseline_latency_ms=2.0,
    )

    assert "Full baseline HIP source:" in prompt
    assert "Target GPU function to optimize:" in prompt
    assert "Current best validated speedup vs baseline HIP: 1.2500x." in prompt
    assert "Attempt 1:" in prompt
    assert "Candidate GPU function:" in prompt


def test_resolve_prompt_assets_accepts_override_files(tmp_path: Path) -> None:
    instruction_file = tmp_path / "instruction.py"
    few_shot_file = tmp_path / "few_shot.py"
    instruction_file.write_text('hip_kernel_opt_req = "CUSTOM INSTRUCTION"', encoding="utf-8")
    few_shot_file.write_text('few_shot_code_instructions = "CUSTOM FEW SHOT"', encoding="utf-8")

    instruction, few_shot = resolve_prompt_assets(
        instruction_file=instruction_file,
        few_shot_file=few_shot_file,
    )

    assert instruction == "CUSTOM INSTRUCTION"
    assert few_shot == "CUSTOM FEW SHOT"
