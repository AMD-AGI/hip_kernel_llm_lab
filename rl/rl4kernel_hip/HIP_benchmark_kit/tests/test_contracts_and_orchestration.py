import argparse
import json
from pathlib import Path

import pytest

from HIP_benchmark_kit.contracts.eval_schema import validate_eval_records
from HIP_benchmark_kit.contracts.layout import KernelBenchRunLayout
from HIP_benchmark_kit.contracts.manifests import validate_generation_manifest, validate_subset_manifest
from HIP_benchmark_kit.gen_hip_kernel.manifest import parse_rollout_indices
from HIP_benchmark_kit.gen_hip_kernel.paradigms import get_generation_paradigm_policy
from HIP_benchmark_kit.orchestration.__main__ import command_text
from HIP_benchmark_kit.orchestration.rollout_reuse import materialize_generation_reuse
from HIP_benchmark_kit.reports.kernelbench import stage_flat_subset


def test_kernelbench_layout_paths_are_canonical() -> None:
    layout = KernelBenchRunLayout(Path("/runs/sample"))
    level = layout.level("level-1")

    assert layout.subset_manifest == Path("/runs/sample/subset/subset_manifest.json")
    assert layout.subset_root == Path("/runs/sample/subset/kernelbench_hip_100")
    assert level.generation_manifest == Path("/runs/sample/level-1/generated/generation_manifest.json")
    assert level.comparison_json == Path("/runs/sample/level-1/eval/comparison/origin_vs_optimized_results.json")


def test_manifest_validators_reject_missing_fields() -> None:
    with pytest.raises(ValueError, match="generation manifest missing fields"):
        validate_generation_manifest({"records": []})
    with pytest.raises(ValueError, match="subset manifest missing fields"):
        validate_subset_manifest({"levels": {}})


def test_eval_schema_validator_rejects_incomplete_rows() -> None:
    with pytest.raises(ValueError, match="Eval record schema violation"):
        validate_eval_records([{"hip_file": "1.hip"}])


def test_rollout_index_parser_dedupes_and_checks_bounds() -> None:
    assert parse_rollout_indices("3,1-2,2", 4) == [1, 2, 3]
    with pytest.raises(ValueError, match="out of range"):
        parse_rollout_indices("4", 4)


def test_command_text_quotes_shell_sensitive_arguments() -> None:
    assert command_text(["python", "script.py", "a b"]) == "python script.py 'a b'"


def test_hip2hip_policy_uses_full_file_starter_and_direct_persistence() -> None:
    hip_src = "#include <hip/hip_runtime.h>\n__global__ void k(float* x) { x[0] = 1; }\n"
    optimized = "#include <hip/hip_runtime.h>\n__global__ void k(float* x) { x[0] = 2; }\n"

    policy = get_generation_paradigm_policy("hip2hip_full_file")
    prompt_source = policy.build_prompt_source(hip_src)

    assert prompt_source.starter_code == hip_src
    assert prompt_source.starter_code_kind == "full_file"
    assert prompt_source.expected_code_unit == "hip_translation_unit"
    assert policy.persist_code(hip_src, optimized, kernel_name="k") == optimized


def _write_level_sample(root: Path, level: str, name: str) -> None:
    for subdir, suffix, body in (
        ("hip_code", ".hip", f"// {name}\n"),
        ("pytorch_code_functional", ".py", f"# functional {name}\n"),
        ("pytorch_code_module", ".py", f"# module {name}\n"),
    ):
        path = root / level / subdir / f"{name}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


def test_stage_flat_subset_preserves_all_layout_and_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "kernelbench_hip"
    _write_level_sample(source_root, "level-1", "1_A")
    _write_level_sample(source_root, "level-2", "2_B")
    _write_level_sample(source_root, "level-3", "3_C")
    subset_root = source_root / "kernelbench_hip_100_l1_35_l2_35_l3_30"

    stage_flat_subset(
        argparse.Namespace(
            source_root=source_root,
            subset_root=subset_root,
            manifest=None,
            level=["level-1:1", "level-2:1", "level-3:1"],
            overwrite=False,
        )
    )

    assert (subset_root / "hip_code" / "1_A.hip").is_file()
    assert (subset_root / "pytorch_code_functional" / "2_B.py").is_file()
    assert (subset_root / "pytorch_code_module" / "3_C.py").is_file()
    manifest = json.loads((subset_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["layout"] == "flat_kernelbench_hip"
    assert manifest["total_selected"] == 3
    assert manifest["levels"]["level-1"]["selected_count"] == 1
    assert manifest["records"][0]["hip_sha256"]


def test_rollout_reuse_materializes_n1_as_gen0_for_multi_rollout(tmp_path: Path) -> None:
    source_root = tmp_path / "rollout_n_1"
    target_root = tmp_path / "rollout_n_4"
    level = "all"
    for run_root in (source_root, target_root):
        subset_manifest = {
            "levels": {level: {"selected_files": [{"hip_file": "foo.hip"}]}},
        }
        path = run_root / "subset" / "subset_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(subset_manifest), encoding="utf-8")

    source_generated = source_root / level / "turn_01" / "generated"
    source_generated.mkdir(parents=True)
    (source_generated / "foo.hip").write_text("// generated\n", encoding="utf-8")
    (source_generated / "generation_manifest.json").write_text(
        json.dumps(
            {
                "model_path": "/cache/model",
                "output_contract": "sample_json_v1",
                "optimization_paradigm": "hip2hip_full_file",
                "target_gpu": "mi300x",
                "data_source": "kernel-agent-react-train",
                "seed_base": "",
                "temperature": "1",
                "prompt_map_arm": "",
                "rollout_n": 1,
                "records": [
                    {
                        "input_file": "foo.hip",
                        "sample_idx": 0,
                        "parse_ok": True,
                        "saved": True,
                        "output_file": "foo.hip",
                        "output_path": str(source_generated / "foo.hip"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = materialize_generation_reuse(
        source_run_root=source_root,
        target_run_root=target_root,
        level=level,
        target_rollout_n=4,
        context_mode="A_control",
        expected_model_path="/cache/model",
        expected_output_contract="sample_json_v1",
        expected_optimization_paradigm="hip2hip_full_file",
        expected_target_gpu="mi300x",
        expected_data_source="kernel-agent-react-train",
        expected_seed_base="",
        expected_temperature="1",
        target_generation_dir=target_root / level / "turn_01" / "generated",
    )

    target_file = target_root / level / "turn_01" / "generated" / "foo_gen0.hip"
    assert target_file.read_text(encoding="utf-8") == "// generated\n"
    assert plan["missing_rollout_indices"] == [1, 2, 3]
    manifest = json.loads((target_file.parent / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["records"][0]["output_file"] == "foo_gen0.hip"
