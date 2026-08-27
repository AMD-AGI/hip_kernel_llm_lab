# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.utils import build_hip_kernel_agent_multiturn_chat_messages
from HIP_benchmark_kit.contracts.layout import KernelBenchRunLayout
from HIP_benchmark_kit.eval.merge_origin_optimized_eval import (
    build_pair_plan,
    build_perf_trace,
    finalize_pair_row,
    normalize_records,
)
from HIP_benchmark_kit.orchestration.__main__ import (
    expected_profile_basenames,
    feedback_context_complete,
    generation_context_for_turn,
    prepare_initial_profile_context,
    run_evaluation,
    turn_eval_complete,
    turn_generated_profile_complete,
    turn_generation_complete,
    validate_multiturn_args,
    write_multiturn_summary,
)
from HIP_benchmark_kit.profiling_context.cards import render_profile_card
from HIP_benchmark_kit.profiling_context.feedback import (
    extract_thought_from_raw_response,
    index_comparison_candidates,
    index_comparison_rows,
    render_feedback_card,
    stage_generated_kernels_for_profile,
)


def test_extract_thought_from_react_raw_response() -> None:
    response = '<think>private chain</think>{"thought": "keep coalescing", "code": "__global__ void k() {}"}'
    assert extract_thought_from_raw_response({"raw_response": response}) == "keep coalescing"


def test_multiturn_prompt_preserves_required_section_order() -> None:
    message = build_hip_kernel_agent_multiturn_chat_messages(
        previous_thought="try tiling",
        original_starter_code="__global__ void k(float* x) { x[0] = 1; }",
        previous_generated_code="__global__ void k(float* x) { x[0] = 2; }",
        previous_feedback="speedup_vs_origin: 1.2",
        target_gpu="mi300x",
        output_contract="sample_json_v1",
    )[0]["content"]

    sections = [
        "### Previous Round Optimization Summary",
        "### Original Starter Code",
        "### Previous Generated HIP Code",
        "### Previous Kernel Profiling And Eval Results",
        "### Format:",
        "### Answer Order (strict)",
    ]
    offsets = [message.index(section) for section in sections]
    assert offsets == sorted(offsets)
    assert "try tiling" in message
    assert "speedup_vs_origin: 1.2" in message


def test_hip2hip_multiturn_prompt_uses_full_file_sections() -> None:
    message = build_hip_kernel_agent_multiturn_chat_messages(
        previous_thought="specialized helper",
        original_starter_code="#include <hip/hip_runtime.h>\n__global__ void k(float* x) { x[0] = 1; }",
        previous_generated_code="#include <hip/hip_runtime.h>\n__global__ void k(float* x) { x[0] = 2; }",
        previous_feedback="speedup_vs_origin: 1.2",
        prompt="Keep the K=32 specialization guarded and preserve the generic fallback.",
        target_gpu="mi300x",
        output_contract="sample_json_v1",
        optimization_paradigm="hip2hip_full_file",
    )[0]["content"]

    sections = [
        "### Task-Specific Context",
        "### Previous Round Optimization Summary",
        "### Original HIP File",
        "### Previous Generated HIP File",
        "### Previous HIP Candidate Profiling And Eval Results",
        "### Format:",
        "### Answer Order (strict)",
    ]
    offsets = [message.index(section) for section in sections]
    assert offsets == sorted(offsets)
    assert "complete optimized .hip source file" in message
    assert "body-only optimization" not in message
    assert "PYBIND11_MODULE" in message
    assert "torch::Tensor forward" in message
    assert "generic fallback" in message


def test_feedback_card_renders_eval_and_profile_signals() -> None:
    card = render_feedback_card(
        sample_name="foo",
        turn=1,
        thought="removed redundant loads",
        eval_row={
            "optimized_compile_ok": True,
            "optimized_run_ok": True,
            "optimized_match_ok": True,
            "origin_hip_time_ms": 4.0,
            "optimized_hip_time_ms": 2.0,
            "speedup": 2.0,
        },
        profile_payload={
            "kernels": [
                {
                    "name": "foo_kernel",
                    "duration_us": {"avg": 12.5},
                    "metrics": {
                        "memory.hbm_bandwidth_utilization": {"avg": 60.0},
                        "memory.l2_hit_rate": {"avg": 80.0},
                        "memory.coalescing_efficiency": {"avg": 99.0},
                        "compute.total_flops": {"avg": 1234.0},
                    },
                }
            ]
        },
    )
    assert "removed redundant loads" in card
    assert "speedup_vs_origin: 2" in card
    assert "profiler.primary_kernel: foo_kernel" in card
    assert "profiler.hbm_bandwidth_utilization: 60" in card


def test_profile_card_reads_metrix_kernel_list_and_run_config_timings() -> None:
    card = render_profile_card(
        "foo",
        {
            "kernel_filter": "foo_kernel",
            "run_config": {"stage_timings_s": {"compile_seconds": 1.25, "filtered_profile_seconds": 2.5}},
            "sample": {
                "category": "Dataset sample",
                "note": "auto",
                "custom_kernel_names": ["foo_kernel"],
                "inputs": [{"shape": [1, 2]}],
                "init_inputs": [0.1],
            },
            "kernels": [
                {
                    "name": "foo_kernel",
                    "duration_us": {"avg": 12.5},
                    "metrics": {
                        "memory.hbm_bandwidth_utilization": {"avg": 60.0},
                        "memory.l2_hit_rate": {"avg": 80.0},
                        "memory.coalescing_efficiency": {"avg": 99.0},
                        "compute.total_flops": {"avg": 1234.0},
                    },
                }
            ],
        },
    )

    assert "Category: Dataset sample" in card
    assert "Compile seconds: 1.25" in card
    assert "Filtered profile seconds: 2.5" in card
    assert "Primary kernel: foo_kernel" in card
    assert "Primary duration us: 12.5" in card
    assert "HBM bandwidth utilization: 60" in card
    assert "Total FLOPs: 1234" in card


def test_profile_staging_preserves_original_basenames(tmp_path: Path) -> None:
    original = tmp_path / "original"
    generated = tmp_path / "generated"
    staging = tmp_path / "staging"
    (original / "include").mkdir(parents=True)
    generated.mkdir()
    (original / "foo.hip").write_text("original", encoding="utf-8")
    (original / "include" / "helper.h").write_text("include", encoding="utf-8")
    (generated / "foo_gen0.hip").write_text("generated", encoding="utf-8")

    manifest = stage_generated_kernels_for_profile(
        generated_dir=generated,
        original_input_dir=original,
        staging_dir=staging,
        comparison_rows={"foo": {"optimized_compile_ok": True, "optimized_run_ok": True, "optimized_match_ok": True}},
        profile_generated="valid-only",
    )

    assert manifest["staged"][0]["base_name"] == "foo"
    assert (staging / "foo.hip").read_text(encoding="utf-8") == "generated"
    assert (staging / "include" / "helper.h").is_file()


def test_comparison_indexes_preserve_rollout_candidates(tmp_path: Path) -> None:
    comparison_json = tmp_path / "comparison.json"
    comparison_json.write_text(
        json.dumps(
            [
                {
                    "base_name": "foo",
                    "gen_idx": 0,
                    "optimized_hip_file": "foo_gen0.hip",
                    "optimized_compile_ok": True,
                    "optimized_run_ok": True,
                    "optimized_match_ok": True,
                    "speedup": 0.9,
                },
                {
                    "base_name": "foo",
                    "gen_idx": 1,
                    "optimized_hip_file": "foo_gen1.hip",
                    "optimized_compile_ok": True,
                    "optimized_run_ok": True,
                    "optimized_match_ok": True,
                    "speedup": 1.5,
                },
                {
                    "base_name": "foo",
                    "gen_idx": 2,
                    "optimized_hip_file": "foo_gen2.hip",
                    "optimized_compile_ok": True,
                    "optimized_run_ok": True,
                    "optimized_match_ok": False,
                    "speedup": None,
                },
            ]
        ),
        encoding="utf-8",
    )

    candidates = index_comparison_candidates(comparison_json)
    assert sorted(candidates) == [("foo", 0), ("foo", 1), ("foo", 2)]
    by_base = index_comparison_rows(comparison_json)
    assert by_base["foo"]["gen_idx"] == 1
    assert by_base["foo"]["speedup"] == 1.5


def _args(**overrides):
    defaults = {
        "turn_mode": "single",
        "turns": "1",
        "sample_count": "8",
        "feedback_max_chars": "4000",
        "rollout_n": "1",
        "profile_missing_policy": "fail",
        "eval_backend": "server-inprocess",
        "optimization_paradigm": "kernel2kernel_splice",
        "output_root": "",
        "level": "level-3",
        "profile_artifact_root": "",
        "profile_prompt_root": "",
        "profile_gpu_ids": "",
        "skip_profile_ensure": False,
        "origin_profile_context": "off",
        "gpu_ids": "0",
        "eval_workers": "1",
        "perf_iterations": "10",
        "rtol": "1e-4",
        "atol": "1e-5",
        "reference_hip_dir": "",
        "local_work_root": "",
        "reuse_eval_from_root": "",
        "shared_compile_cache_root": "",
        "origin_baseline_eval_root": "",
        "disable_rollout_reuse": False,
        "dry_run": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_multiturn_cli_validation_enforces_turn_mode_semantics() -> None:
    assert validate_multiturn_args(_args(turn_mode="single", turns="1")).turns == 1
    assert validate_multiturn_args(_args(turn_mode="single", turns="1", rollout_n="4")).turns == 1
    assert validate_multiturn_args(_args(turn_mode="single", origin_profile_context="use_existing")).origin_profile_context == "use_existing"
    assert validate_multiturn_args(_args(turn_mode="single", origin_profile_context="ensure_and_use")).origin_profile_context == "ensure_and_use"
    assert validate_multiturn_args(_args(turn_mode="multi", turns="2")).turns == 2
    with pytest.raises(SystemExit, match="single requires"):
        validate_multiturn_args(_args(turn_mode="single", turns="2"))
    with pytest.raises(SystemExit, match="multi requires"):
        validate_multiturn_args(_args(turn_mode="multi", turns="1"))
    with pytest.raises(SystemExit, match="rollout_n 1"):
        validate_multiturn_args(_args(turn_mode="multi", turns="2", rollout_n="4"))
    with pytest.raises(SystemExit, match="origin_profile_context"):
        validate_multiturn_args(_args(turn_mode="single", origin_profile_context="existing"))


def test_single_turn_origin_profile_context_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_prepare(args, layout, *, dataset_root, input_dir, env):
        calls.append(bool(args.skip_profile_ensure))
        return {
            "prompt_map_json": "/shared/origin_profiling/prompt_maps/all_starter_prompt_map.json",
            "prompt_map_arm": "starter_profile_raw",
            "profile_dir": "/shared/origin_profiling/artifacts/all",
        }

    monkeypatch.setattr("HIP_benchmark_kit.orchestration.__main__.prepare_starter_profile_context", fake_prepare)
    layout = KernelBenchRunLayout(Path("/run"))
    dataset_root = Path("/run/subset/kernelbench_hip_100/all")
    input_dir = dataset_root / "hip_code"

    assert prepare_initial_profile_context(
        _args(turn_mode="single", origin_profile_context="off"),
        layout,
        dataset_root=dataset_root,
        input_dir=input_dir,
        env={},
    ) is None
    assert calls == []

    ensure_context = prepare_initial_profile_context(
        _args(turn_mode="single", origin_profile_context="ensure_and_use", skip_profile_ensure=True),
        layout,
        dataset_root=dataset_root,
        input_dir=input_dir,
        env={},
    )
    assert ensure_context and ensure_context["prompt_map_arm"] == "starter_profile_raw"
    assert calls[-1] is False

    existing_context = prepare_initial_profile_context(
        _args(turn_mode="single", origin_profile_context="use_existing", skip_profile_ensure=False),
        layout,
        dataset_root=dataset_root,
        input_dir=input_dir,
        env={},
    )
    assert existing_context and existing_context["prompt_map_json"].endswith("all_starter_prompt_map.json")
    assert calls[-1] is True


def test_turn_one_generation_context_includes_starter_profile_prompt_map() -> None:
    turn_layout = argparse.Namespace(raw_response_dir=Path("/run/level-3/turn_01/raw_responses"))
    context = generation_context_for_turn(
        _args(turn_mode="multi"),
        1,
        turn_layout,
        feedback_context_json=None,
        starter_profile_context={
            "prompt_map_json": "/run/starter_profiling/prompt_maps/level-3_starter_prompt_map.json",
            "prompt_map_arm": "starter_profile_raw",
        },
    )

    assert context["prompt_map_json"].endswith("level-3_starter_prompt_map.json")
    assert context["prompt_map_arm"] == "starter_profile_raw"
    assert context["feedback_context_json"] == ""


def test_later_turn_generation_context_uses_feedback_not_starter_prompt_map() -> None:
    turn_layout = argparse.Namespace(raw_response_dir=Path("/run/level-3/turn_02/raw_responses"))
    context = generation_context_for_turn(
        _args(turn_mode="multi"),
        2,
        turn_layout,
        feedback_context_json=Path("/run/level-3/turn_01/feedback/turn_02_context.json"),
        starter_profile_context={
            "prompt_map_json": "/run/starter_profiling/prompt_maps/level-3_starter_prompt_map.json",
            "prompt_map_arm": "starter_profile_raw",
        },
    )

    assert context["prompt_map_json"] == ""
    assert context["prompt_map_arm"] == ""
    assert context["feedback_context_json"].endswith("turn_02_context.json")


def test_launcher_is_human_readable_and_forwards_args() -> None:
    canonical = Path("scripts/eval/multi_turn/run_kernelbench_multiturn_profile_kernel2kernel.sh").read_text(encoding="utf-8")
    for token in (
        "MODEL=${MODEL:-",
        "TURN_MODE=${TURN_MODE:-multi}",
        "TURNS=${TURNS:-4}",
        "OPTIMIZATION_PARADIGM=${OPTIMIZATION_PARADIGM:-kernel2kernel_splice}",
        '"$@"',
    ):
        assert token in canonical
    hip2hip = Path("scripts/eval/multi_turn/run_kernelbench_multiturn_profile_hip2hip.sh").read_text(encoding="utf-8")
    assert "--profile_generated" in hip2hip
    assert "--shared_compile_cache_root" in hip2hip
    single_turn = Path("scripts/eval/single_turn/run_kernelbench_singleturn_hip2hip.sh").read_text(encoding="utf-8")
    assert "ORIGIN_PROFILE_CONTEXT=${ORIGIN_PROFILE_CONTEXT:-off}" in single_turn
    assert "--origin_profile_context" in single_turn
    assert "ORIGIN_PROFILE_CONTEXT\" != \"off\"" in single_turn
    assert "--profile_artifact_root" in single_turn
    assert "--profile_missing_policy" in single_turn


def test_kernelbench_profiler_uses_server_sandbox_helpers() -> None:
    sample_dir = REPO_ROOT / "profiler" / "intellikit" / "metrix" / "examples" / "01_basic_profiling" / "kernelbench_sample"
    if str(sample_dir) not in sys.path:
        sys.path.insert(0, str(sample_dir))

    import runtime_common

    assert runtime_common.SERVER_SANDBOX_DIR.name == "sandbox_core"
    assert (runtime_common.SERVER_SANDBOX_DIR / "fix_seed.py").is_file()
    assert (runtime_common.SERVER_SANDBOX_DIR / "safe_call_helper.py").is_file()
    assert callable(runtime_common.set_seed)
    assert "def _safe_call" in runtime_common.SAFE_CALL_HELPER


def _turn_layout(root: Path):
    return argparse.Namespace(
        generated_dir=root / "generated",
        comparison_json=root / "eval" / "comparison" / "origin_vs_optimized_results.json",
        profiling_dir=root / "profiling",
        profiling_generated_dir=root / "profiling" / "generated",
    )


def test_turn_generation_complete_requires_saved_outputs(tmp_path: Path) -> None:
    layout = _turn_layout(tmp_path / "turn_01")
    layout.generated_dir.mkdir(parents=True)
    output_path = layout.generated_dir / "foo.hip"
    output_path.write_text("hip", encoding="utf-8")
    (layout.generated_dir / "generation_manifest.json").write_text(
        (
            '{ "records": ['
            '{"input_file":"foo.hip","parse_ok":true,"saved":true,'
            f'"output_path":"{output_path}"'
            "}] }"
        ),
        encoding="utf-8",
    )

    assert turn_generation_complete(layout, 1)
    output_path.unlink()
    assert not turn_generation_complete(layout, 1)


def test_turn_generation_complete_requires_all_rollout_indices(tmp_path: Path) -> None:
    layout = _turn_layout(tmp_path / "turn_01")
    layout.generated_dir.mkdir(parents=True)
    records = []
    for idx in range(3):
        output_path = layout.generated_dir / f"foo_gen{idx}.hip"
        output_path.write_text("hip", encoding="utf-8")
        records.append(
            {
                "input_file": "foo.hip",
                "sample_idx": idx,
                "parse_ok": True,
                "saved": True,
                "output_path": str(output_path),
            }
        )
    (layout.generated_dir / "generation_manifest.json").write_text(
        json.dumps({"records": records}),
        encoding="utf-8",
    )

    assert not turn_generation_complete(layout, 1, rollout_n=4)
    output_path = layout.generated_dir / "foo_gen3.hip"
    output_path.write_text("hip", encoding="utf-8")
    records.append(
        {
            "input_file": "foo.hip",
            "sample_idx": 3,
            "parse_ok": True,
            "saved": True,
            "output_path": str(output_path),
        }
    )
    (layout.generated_dir / "generation_manifest.json").write_text(
        json.dumps({"records": records}),
        encoding="utf-8",
    )
    assert turn_generation_complete(layout, 1, rollout_n=4)


def test_turn_generation_complete_accepts_recorded_parse_failure(tmp_path: Path) -> None:
    layout = _turn_layout(tmp_path / "turn_01")
    layout.generated_dir.mkdir(parents=True)
    raw_response_path = layout.generated_dir / "foo_gen0_raw_response.json"
    raw_response_path.write_text("{}", encoding="utf-8")
    (layout.generated_dir / "generation_manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "input_file": "foo.hip",
                        "sample_idx": 0,
                        "parse_ok": False,
                        "saved": False,
                        "raw_response_path": str(raw_response_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert turn_generation_complete(layout, 1)
    raw_response_path.unlink()
    assert not turn_generation_complete(layout, 1)


def test_turn_eval_complete_accepts_failed_pairs(tmp_path: Path) -> None:
    layout = _turn_layout(tmp_path / "turn_01")
    layout.comparison_json.parent.mkdir(parents=True)
    layout.comparison_json.write_text(
        '[{"base_name":"foo","origin_hip_file":"foo.hip","optimized_hip_file":"foo.hip","pair_ok":false}]',
        encoding="utf-8",
    )

    assert turn_eval_complete(layout)


def test_profile_completeness_respects_profile_modes(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("foo.hip", "bar.hip"):
        (input_dir / name).write_text("hip", encoding="utf-8")
    layout = _turn_layout(tmp_path / "turn_01")
    layout.comparison_json.parent.mkdir(parents=True)
    layout.comparison_json.write_text(
        (
            "["
            '{"base_name":"foo","origin_hip_file":"foo.hip","optimized_hip_file":"foo.hip",'
            '"optimized_compile_ok":true,"optimized_run_ok":true,"optimized_match_ok":true},'
            '{"base_name":"bar","origin_hip_file":"bar.hip","optimized_hip_file":"bar.hip",'
            '"optimized_compile_ok":false,"optimized_run_ok":false,"optimized_match_ok":false}'
            "]"
        ),
        encoding="utf-8",
    )
    layout.profiling_generated_dir.mkdir(parents=True)
    layout.profiling_dir.mkdir(exist_ok=True)
    (layout.profiling_dir / "generated_profile_staging_manifest.json").write_text("{}", encoding="utf-8")
    (layout.profiling_generated_dir / "foo_filtered.json").write_text("{}", encoding="utf-8")

    assert expected_profile_basenames(_args(profile_generated="valid-only"), input_dir, layout.comparison_json) == {"foo"}
    assert turn_generated_profile_complete(_args(profile_generated="valid-only"), input_dir, layout)
    assert turn_generated_profile_complete(_args(profile_generated="never"), input_dir, layout)
    assert not turn_generated_profile_complete(_args(profile_generated="always"), input_dir, layout)
    (layout.profiling_generated_dir / "bar_filtered.json").write_text("{}", encoding="utf-8")
    assert turn_generated_profile_complete(_args(profile_generated="always"), input_dir, layout)


def test_feedback_context_complete_requires_feedback_and_previous_code(tmp_path: Path) -> None:
    previous = tmp_path / "foo.hip"
    previous.write_text("hip", encoding="utf-8")
    feedback = tmp_path / "turn_02_context.json"
    feedback.write_text(
        (
            '{"feedback_map":{"foo.hip":{'
            '"base_name":"foo","feedback_text":"ok","previous_turn":1,'
            f'"previous_generated_path":"{previous}","blocked_reason":""'
            "}}}"
        ),
        encoding="utf-8",
    )
    assert feedback_context_complete(feedback, 1)

    previous.unlink()
    assert not feedback_context_complete(feedback, 1)


def test_multiturn_summary_writes_best_valid_final(tmp_path: Path) -> None:
    layout = KernelBenchRunLayout(tmp_path)
    args = _args(output_root=str(tmp_path), level="level-3", turns=2)
    turn1 = layout.turn("level-3", 1)
    turn2 = layout.turn("level-3", 2)
    for turn_layout in (turn1, turn2):
        turn_layout.generated_dir.mkdir(parents=True)
        turn_layout.comparison_json.parent.mkdir(parents=True)
        (turn_layout.generated_dir / "generation_manifest.json").write_text('{"records":[]}', encoding="utf-8")

    (turn1.generated_dir / "foo.hip").write_text("turn1 foo", encoding="utf-8")
    (turn2.generated_dir / "foo.hip").write_text("turn2 foo", encoding="utf-8")
    (turn2.generated_dir / "bar.hip").write_text("bad bar", encoding="utf-8")
    turn1.comparison_json.write_text(
        json.dumps(
            [
                {
                    "base_name": "foo",
                    "origin_hip_file": "foo.hip",
                    "optimized_hip_file": "foo.hip",
                    "optimized_compile_ok": True,
                    "optimized_run_ok": True,
                    "optimized_match_ok": True,
                    "origin_hip_time_ms": 4.0,
                    "optimized_hip_time_ms": 2.0,
                    "speedup": 2.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    turn2.comparison_json.write_text(
        json.dumps(
            [
                {
                    "base_name": "foo",
                    "origin_hip_file": "foo.hip",
                    "optimized_hip_file": "foo.hip",
                    "optimized_compile_ok": True,
                    "optimized_run_ok": True,
                    "optimized_match_ok": True,
                    "origin_hip_time_ms": 3.0,
                    "optimized_hip_time_ms": 2.0,
                    "speedup": 1.5,
                },
                {
                    "base_name": "bar",
                    "origin_hip_file": "bar.hip",
                    "optimized_hip_file": "bar.hip",
                    "optimized_compile_ok": True,
                    "optimized_run_ok": True,
                    "optimized_match_ok": False,
                    "speedup": None,
                },
            ]
        ),
        encoding="utf-8",
    )

    write_multiturn_summary(args, layout)

    best_dir = tmp_path / "level-3" / "final" / "best_valid_generated"
    assert (best_dir / "foo.hip").read_text(encoding="utf-8") == "turn1 foo"
    assert not (best_dir / "bar.hip").exists()
    manifest = json.loads((tmp_path / "level-3" / "final" / "best_valid_manifest.json").read_text(encoding="utf-8"))
    by_name = {row["base_name"]: row for row in manifest}
    assert by_name["foo"]["selected_turn"] == 1
    assert by_name["foo"]["speedup"] == 2.0
    assert by_name["bar"]["status"] == "no_valid_candidate"


def test_multiturn_summary_selects_best_rollout_candidate(tmp_path: Path) -> None:
    layout = KernelBenchRunLayout(tmp_path)
    args = _args(output_root=str(tmp_path), level="level-3", turns=1, rollout_n="4")
    turn = layout.turn("level-3", 1)
    turn.generated_dir.mkdir(parents=True)
    turn.raw_response_dir.mkdir(parents=True)
    turn.comparison_json.parent.mkdir(parents=True)

    records = []
    for sample_idx in range(4):
        output_path = turn.generated_dir / f"foo_gen{sample_idx}.hip"
        output_path.write_text(f"gen{sample_idx}", encoding="utf-8")
        raw_path = turn.raw_response_dir / f"foo_gen{sample_idx}_raw_response.json"
        raw_path.write_text(json.dumps({"raw_response": {"thought": f"thought {sample_idx}"}}), encoding="utf-8")
        records.append(
            {
                "input_file": "foo.hip",
                "sample_idx": sample_idx,
                "parse_ok": True,
                "saved": True,
                "output_file": output_path.name,
                "output_path": str(output_path),
                "raw_response_path": str(raw_path),
            }
        )
    (turn.generated_dir / "generation_manifest.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    turn.comparison_json.write_text(
        json.dumps(
            [
                {
                    "base_name": "foo",
                    "gen_idx": 0,
                    "origin_hip_file": "foo.hip",
                    "optimized_hip_file": "foo_gen0.hip",
                    "optimized_compile_ok": True,
                    "optimized_run_ok": True,
                    "optimized_match_ok": True,
                    "origin_hip_time_ms": 4.0,
                    "optimized_hip_time_ms": 5.0,
                    "speedup": 0.8,
                },
                {
                    "base_name": "foo",
                    "gen_idx": 1,
                    "origin_hip_file": "foo.hip",
                    "optimized_hip_file": "foo_gen1.hip",
                    "optimized_compile_ok": True,
                    "optimized_run_ok": True,
                    "optimized_match_ok": False,
                    "speedup": None,
                },
                {
                    "base_name": "foo",
                    "gen_idx": 2,
                    "origin_hip_file": "foo.hip",
                    "optimized_hip_file": "foo_gen2.hip",
                    "optimized_compile_ok": True,
                    "optimized_run_ok": True,
                    "optimized_match_ok": True,
                    "origin_hip_time_ms": 4.0,
                    "optimized_hip_time_ms": 2.0,
                    "speedup": 2.0,
                },
                {
                    "base_name": "foo",
                    "gen_idx": 3,
                    "origin_hip_file": "foo.hip",
                    "optimized_hip_file": "foo_gen3.hip",
                    "optimized_compile_ok": True,
                    "optimized_run_ok": True,
                    "optimized_match_ok": True,
                    "origin_hip_time_ms": 4.0,
                    "optimized_hip_time_ms": 4.2,
                    "speedup": 0.95,
                },
            ]
        ),
        encoding="utf-8",
    )

    write_multiturn_summary(args, layout)

    summary = json.loads((tmp_path / "level-3" / "multi_turn_summary.json").read_text(encoding="utf-8"))
    assert [row["gen_idx"] for row in summary] == [0, 1, 2, 3]
    best_dir = tmp_path / "level-3" / "final" / "best_valid_generated"
    assert (best_dir / "foo_gen2.hip").read_text(encoding="utf-8") == "gen2"
    manifest = json.loads((tmp_path / "level-3" / "final" / "best_valid_manifest.json").read_text(encoding="utf-8"))
    assert manifest == [
        {
            "base_name": "foo",
            "status": "selected",
            "selected_turn": 1,
            "selected_sample_idx": 2,
            "selected_gen_idx": 2,
            "speedup": 2.0,
            "origin_hip_time_ms": 4.0,
            "candidate_hip_time_ms": 2.0,
            "optimized_hip_file": "foo_gen2.hip",
            "source_generated_path": str(turn.generated_dir / "foo_gen2.hip"),
            "raw_response_path": str(turn.raw_response_dir / "foo_gen2_raw_response.json"),
            "profile_artifact": "",
        }
    ]


def test_run_evaluation_forwards_fixed_origin_baseline_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_root = tmp_path / "dataset"
    generation_dir = tmp_path / "generated"
    eval_dir = tmp_path / "eval"
    baseline_eval_dir = tmp_path / "origin_baseline" / "eval"
    captured: dict[str, list[object]] = {}

    def fake_run_command(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return ""

    monkeypatch.setattr("HIP_benchmark_kit.orchestration.__main__.run_command", fake_run_command)

    run_evaluation(
        _args(dry_run=True, shared_compile_cache_root=str(tmp_path / "shared_cache")),
        "level-3",
        dataset_root,
        generation_dir,
        eval_dir,
        {},
        origin_baseline_eval_dir=baseline_eval_dir,
    )

    cmd = [str(part) for part in captured["cmd"]]
    assert "--reuse-origin-json" in cmd
    assert str(baseline_eval_dir / "baseline_hip_results.json") in cmd
    assert "--reuse-origin-hip-dir" in cmd
    assert str(dataset_root / "hip_code") in cmd
    assert "--compile-cache-root" in cmd
    assert str(tmp_path / "shared_cache") in cmd


def test_merge_preserves_perf_trace_fields() -> None:
    origin = normalize_records(
        [
            {
                "hip_file": "foo.hip",
                "base_name": "foo",
                "compile_ok": True,
                "run_ok": True,
                "match_ok": True,
                "hip_time_ms": 4.0,
                "perf_gpu_id": 0,
                "perf_started_at": "2026-01-01T00:00:00Z",
                "perf_finished_at": "2026-01-01T00:00:01Z",
            }
        ]
    )
    optimized = normalize_records(
        [
            {
                "hip_file": "foo.hip",
                "base_name": "foo",
                "compile_ok": True,
                "run_ok": True,
                "match_ok": True,
                "hip_time_ms": 2.0,
                "perf_gpu_id": 0,
                "perf_started_at": "2026-01-01T00:00:01Z",
                "perf_finished_at": "2026-01-01T00:00:02Z",
            }
        ]
    )

    rows = [finalize_pair_row(row) for row in build_pair_plan(origin, optimized)]
    trace = build_perf_trace(rows)

    assert rows[0]["origin_perf_started_at"] == "2026-01-01T00:00:00Z"
    assert rows[0]["optimized_perf_finished_at"] == "2026-01-01T00:00:02Z"
    assert [row["side"] for row in trace] == ["origin", "optimized"]
