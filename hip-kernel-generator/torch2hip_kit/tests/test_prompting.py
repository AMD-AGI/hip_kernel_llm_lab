from pathlib import Path

from torch2hip_kit.config import AttemptRecord
from torch2hip_kit.prompt_assets import resolve_prompt_assets
from torch2hip_kit.prompting import build_prompt, format_response


def test_format_response_extracts_first_fenced_block() -> None:
    response = """
    Here is the implementation.

    ```cpp
    extern "C" void kernel() {}
    ```
    """
    assert format_response(response) == 'extern "C" void kernel() {}'


def test_build_prompt_includes_functional_code_and_attempt_history(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.hip"
    candidate_path.write_text('extern "C" void kernel() {}', encoding="utf-8")

    attempts = [
        AttemptRecord(
            attempt=1,
            prompt_path="prompt_1.txt",
            candidate_path=candidate_path.as_posix(),
            status="success",
            feedback="Correctness passed. Speedup=1.2500x.",
            correctness_success=True,
            speedup=1.25,
        )
    ]

    prompt = build_prompt(
        "class Model: pass",
        "def module_fn(x): return x",
        Path("level_1/sample.py"),
        attempts,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
        code_char_limit=1000,
        feedback_char_limit=1000,
    )

    assert "Paired PyTorch functional code:" in prompt
    assert "def module_fn(x): return x" in prompt
    assert "Attempt 1:" in prompt
    assert "Current best validated speedup: 1.2500x." in prompt
    assert "Speedup=1.2500x." in prompt
    assert "Verifier feedback, error trace, or performance note:" in prompt


def test_resolve_prompt_assets_uses_embedded_few_shot_cases() -> None:
    instruction, few_shot = resolve_prompt_assets()

    assert "Given the following PyTorch code" in instruction
    assert "### Example 1:" in few_shot
    assert "### Example 2:" in few_shot
    assert "torch2hip/few_shot_examples_torch2hip.py" not in few_shot


def test_resolve_prompt_assets_accepts_override_files(tmp_path: Path) -> None:
    instruction_file = tmp_path / "instruction.py"
    few_shot_file = tmp_path / "few_shot.py"
    instruction_file.write_text('hip_generation_req = "CUSTOM INSTRUCTION"', encoding="utf-8")
    few_shot_file.write_text('few_shot_code_instructions = "CUSTOM FEW SHOT"', encoding="utf-8")

    instruction, few_shot = resolve_prompt_assets(
        instruction_file=instruction_file,
        few_shot_file=few_shot_file,
    )

    assert instruction == "CUSTOM INSTRUCTION"
    assert few_shot == "CUSTOM FEW SHOT"
