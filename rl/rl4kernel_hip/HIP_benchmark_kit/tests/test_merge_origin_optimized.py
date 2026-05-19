import json

from HIP_benchmark_kit.eval.merge_origin_optimized_eval import build_pair_plan, main, validate_non_overlapping_trace


def make_record(hip_file, *, base_name, gen_idx=None, compile_ok=True, run_ok=True, match_ok=True):
    return {
        "_record_id": hash((hip_file, gen_idx)) & 0xFFFFFFFF,
        "hip_file": hip_file,
        "base_name": base_name,
        "gen_idx": gen_idx,
        "compile_ok": compile_ok,
        "run_ok": run_ok,
        "match_ok": match_ok,
        "hip_time_ms": None,
        "pytorch_time_ms": None,
        "error_message": None,
    }


def test_build_pair_plan_exact_match():
    origin = [make_record("foo.hip", base_name="foo")]
    optimized = [make_record("foo.hip", base_name="foo")]

    rows = build_pair_plan(origin, optimized)

    assert len(rows) == 1
    assert rows[0]["origin_hip_file"] == "foo.hip"
    assert rows[0]["optimized_hip_file"] == "foo.hip"
    assert rows[0]["compare_error"] is None


def test_build_pair_plan_fanout_single_origin_to_many_generations():
    origin = [make_record("foo.hip", base_name="foo")]
    optimized = [
        make_record("foo_gen0.hip", base_name="foo", gen_idx=0),
        make_record("foo_gen1.hip", base_name="foo", gen_idx=1),
    ]

    rows = build_pair_plan(origin, optimized)

    assert len(rows) == 2
    assert {row["optimized_hip_file"] for row in rows} == {"foo_gen0.hip", "foo_gen1.hip"}
    assert {row["origin_hip_file"] for row in rows} == {"foo.hip"}
    assert all(row["compare_error"] is None for row in rows)


def test_build_pair_plan_marks_missing_pairs():
    origin = [make_record("foo.hip", base_name="foo")]
    optimized = [make_record("bar.hip", base_name="bar")]

    rows = build_pair_plan(origin, optimized)

    assert len(rows) == 2
    errors = {row["compare_error"] for row in rows}
    assert errors == {"missing_origin_pair", "missing_optimized_pair"}


def test_build_pair_plan_rejects_duplicate_exact_keys():
    origin = [
        make_record("foo.hip", base_name="foo"),
        make_record("foo_dup.hip", base_name="foo"),
    ]
    optimized = [make_record("foo.hip", base_name="foo")]

    try:
        build_pair_plan(origin, optimized)
    except ValueError as exc:
        assert "duplicate keys" in str(exc)
    else:
        raise AssertionError("Expected duplicate exact-key pairing to fail")


def test_build_pair_plan_rejects_ambiguous_many_to_many_without_gen_idx():
    origin = [
        make_record("foo_gen0.hip", base_name="foo", gen_idx=0),
        make_record("foo_gen1.hip", base_name="foo", gen_idx=1),
    ]
    optimized = [make_record("foo.hip", base_name="foo", gen_idx=None)]

    try:
        build_pair_plan(origin, optimized)
    except ValueError as exc:
        assert "Ambiguous many-to-many join" in str(exc)
    else:
        raise AssertionError("Expected ambiguous pairing to fail")


def test_validate_non_overlapping_trace_accepts_sequential_windows():
    trace = [
        {
            "pair_started_at": "2026-01-01T00:00:00Z",
            "pair_finished_at": "2026-01-01T00:00:05Z",
        },
        {
            "pair_started_at": "2026-01-01T00:00:05Z",
            "pair_finished_at": "2026-01-01T00:00:10Z",
        },
    ]

    assert validate_non_overlapping_trace(trace) is True


def test_validate_non_overlapping_trace_rejects_overlap():
    trace = [
        {
            "pair_started_at": "2026-01-01T00:00:00Z",
            "pair_finished_at": "2026-01-01T00:00:05Z",
        },
        {
            "pair_started_at": "2026-01-01T00:00:04Z",
            "pair_finished_at": "2026-01-01T00:00:10Z",
        },
    ]

    assert validate_non_overlapping_trace(trace) is False


def test_merge_main_warns_but_writes_when_perf_windows_overlap(tmp_path, capsys):
    origin_json = tmp_path / "origin.json"
    optimized_json = tmp_path / "optimized.json"
    output_dir = tmp_path / "comparison"
    origin_json.write_text(
        json.dumps(
            [
                {
                    "hip_file": "foo.hip",
                    "base_name": "foo",
                    "compile_ok": True,
                    "run_ok": True,
                    "match_ok": True,
                    "hip_time_ms": 2.0,
                    "pytorch_time_ms": 4.0,
                    "perf_gpu_id": 0,
                    "perf_started_at": "2026-01-01T00:00:00Z",
                    "perf_finished_at": "2026-01-01T00:00:10Z",
                }
            ]
        ),
        encoding="utf-8",
    )
    optimized_json.write_text(
        json.dumps(
            [
                {
                    "hip_file": "foo.hip",
                    "base_name": "foo",
                    "compile_ok": True,
                    "run_ok": True,
                    "match_ok": True,
                    "hip_time_ms": 1.0,
                    "pytorch_time_ms": 4.0,
                    "perf_gpu_id": 0,
                    "perf_started_at": "2026-01-01T00:00:05Z",
                    "perf_finished_at": "2026-01-01T00:00:15Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    main(
        [
            "--origin-json",
            str(origin_json),
            "--optimized-json",
            str(optimized_json),
            "--pytorch-func-dir",
            str(tmp_path),
            "--pytorch-modu-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert "overlapping perf windows" in captured.err
    assert (output_dir / "origin_vs_optimized_results.json").is_file()
