# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SERVER_ROOT = os.path.join(REPO_ROOT, "hip_kernel_evaluation_server")
if SERVER_ROOT not in sys.path:
    sys.path.insert(0, SERVER_ROOT)

from sandbox_core.result import EvalRunResult

from HIP_benchmark_kit.eval.eval_reuse_identity import build_eval_identity, load_reusable_results
from HIP_benchmark_kit.eval.server_eval_adapter import (
    build_input_manifest,
    build_sandbox_settings,
    map_result_to_legacy_record,
    parse_hip_filename,
    validate_legacy_eval_record,
)


class ServerEvalAdapterTests(unittest.TestCase):
    def test_parse_hip_filename_keeps_generation_identity(self):
        self.assertEqual(parse_hip_filename("foo.hip"), ("foo", None))
        self.assertEqual(parse_hip_filename("foo_gen3.hip"), ("foo", 3))
        self.assertEqual(parse_hip_filename("foo_hip_gen2.hip"), ("foo", 2))
        self.assertEqual(parse_hip_filename("foo_gen1_hip.hip"), ("foo", 1))

    def test_build_sandbox_settings_matches_batch_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {}, clear=True):
            settings = build_sandbox_settings(
                gpu_ids=[0, 1],
                error_log_dir=os.path.join(tmpdir, "errors"),
                cache_root=os.path.join(tmpdir, "cache"),
                perf_iterations=10,
                reference_cache_mode="golden+compile",
                disable_compile_cache=False,
            )

        self.assertEqual(settings.gpu_ids, [0, 1])
        self.assertEqual(settings.effective_arch, "gfx942")
        self.assertEqual(settings.perf_iterations, 10)
        self.assertEqual(settings.compile_timeout_s, 600)
        self.assertEqual(settings.run_timeout_s, 600)
        self.assertTrue(settings.enable_two_stage_batch)
        self.assertEqual(settings.compile_cpu_slots, 16)
        self.assertEqual(settings.compile_inner_jobs, 4)
        self.assertTrue(settings.enable_ref_compile_cache)
        self.assertTrue(settings.enable_ref_golden_cache)
        self.assertFalse(settings.enable_ref_perf_cache)
        self.assertTrue(settings.cache_golden_on_cpu)

    def test_build_sandbox_settings_can_disable_compile_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {}, clear=True):
            settings = build_sandbox_settings(
                gpu_ids=[0],
                error_log_dir=os.path.join(tmpdir, "errors"),
                cache_root=os.path.join(tmpdir, "cache"),
                perf_iterations=10,
                reference_cache_mode="golden+compile+perf",
                disable_compile_cache=True,
            )

        self.assertFalse(settings.enable_ref_compile_cache)
        self.assertTrue(settings.enable_ref_golden_cache)
        self.assertTrue(settings.enable_ref_perf_cache)

    def test_input_manifest_resolves_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hip_dir = os.path.join(tmpdir, "hip")
            ref_dir = os.path.join(tmpdir, "ref")
            func_dir = os.path.join(tmpdir, "func")
            modu_dir = os.path.join(tmpdir, "modu")
            for path in (hip_dir, ref_dir, func_dir, modu_dir):
                os.makedirs(path)
            for path, text in (
                (os.path.join(hip_dir, "foo_gen0.hip"), "candidate"),
                (os.path.join(ref_dir, "foo.hip"), "reference"),
                (os.path.join(func_dir, "foo.py"), "functional"),
                (os.path.join(modu_dir, "foo.py"), "module"),
            ):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(text)

            manifest = build_input_manifest(
                hip_file="foo_gen0.hip",
                hip_code_dir=hip_dir,
                reference_hip_code_dir=ref_dir,
                pytorch_func_dir=func_dir,
                pytorch_modu_dir=modu_dir,
            )

        self.assertEqual(manifest.base_name, "foo")
        self.assertEqual(manifest.gen_idx, 0)
        self.assertTrue(manifest.reference_path.endswith("foo.hip"))

    def test_result_mapper_preserves_legacy_schema(self):
        manifest = type(
            "Manifest",
            (),
            {
                "hip_file": "foo_gen0.hip",
                "base_name": "foo",
                "gen_idx": 0,
            },
        )()
        result = EvalRunResult(
            True,
            True,
            True,
            2.0,
            {
                "reference_perf_ms": 4.0,
                "candidate_perf_ms": 2.0,
                "assigned_gpu_id": 1,
                "reference_compile_cache_hit": True,
                "candidate_module_name": "hip_foo",
            },
        )
        row = map_result_to_legacy_record(
            manifest=manifest,
            result=result,
            artifact_side="optimized",
            eval_backend="server-inprocess",
        )

        validate_legacy_eval_record(row)
        self.assertEqual(row["hip_file"], "foo_gen0.hip")
        self.assertEqual(row["base_name"], "foo")
        self.assertEqual(row["gen_idx"], 0)
        self.assertTrue(row["compile_ok"])
        self.assertEqual(row["pytorch_time_ms"], 4.0)
        self.assertEqual(row["hip_time_ms"], 2.0)
        self.assertEqual(row["speedup"], 2.0)
        self.assertEqual(row["perf_gpu_id"], 1)

    def test_malformed_legacy_row_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_legacy_eval_record({"hip_file": "foo.hip"})

    def test_server_backend_rejects_legacy_reuse_without_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            current_hip = os.path.join(tmpdir, "current")
            prior_hip = os.path.join(tmpdir, "prior")
            func_dir = os.path.join(tmpdir, "func")
            modu_dir = os.path.join(tmpdir, "modu")
            for path in (current_hip, prior_hip, func_dir, modu_dir):
                os.makedirs(path)
            for directory in (current_hip, prior_hip):
                with open(os.path.join(directory, "foo.hip"), "w", encoding="utf-8") as handle:
                    handle.write("hip")
            for directory in (func_dir, modu_dir):
                with open(os.path.join(directory, "foo.py"), "w", encoding="utf-8") as handle:
                    handle.write("class Model: pass\n")
            reuse_json = os.path.join(tmpdir, "baseline_hip_results.json")
            with open(reuse_json, "w", encoding="utf-8") as handle:
                handle.write(
                    '[{"hip_file":"foo.hip","base_name":"foo","gen_idx":null,'
                    '"compile_ok":true,"run_ok":true,"match_ok":true,'
                    '"eval_backend":"local"}]'
                )

            reusable = load_reusable_results(
                reuse_json=reuse_json,
                reuse_hip_code_dir=prior_hip,
                current_hip_code_dir=current_hip,
                pytorch_func_dir=func_dir,
                pytorch_modu_dir=modu_dir,
                rtol=1e-4,
                atol=1e-5,
                perf_iterations=10,
                artifact_side="origin",
                eval_backend="server-inprocess",
                reference_hip_code_dir=current_hip,
                reference_cache_mode="golden+compile",
            )

        self.assertEqual(reusable, {})

    def test_eval_identity_includes_reference_hip_and_server_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            current_hip = os.path.join(tmpdir, "current")
            reference_hip = os.path.join(tmpdir, "reference")
            func_dir = os.path.join(tmpdir, "func")
            modu_dir = os.path.join(tmpdir, "modu")
            for path in (current_hip, reference_hip, func_dir, modu_dir):
                os.makedirs(path)
            with open(os.path.join(current_hip, "foo_gen0.hip"), "w", encoding="utf-8") as handle:
                handle.write("candidate")
            with open(os.path.join(reference_hip, "foo.hip"), "w", encoding="utf-8") as handle:
                handle.write("reference")
            for directory in (func_dir, modu_dir):
                with open(os.path.join(directory, "foo.py"), "w", encoding="utf-8") as handle:
                    handle.write("class Model: pass\n")

            identity = build_eval_identity(
                hip_code_dir=current_hip,
                hip_file="foo_gen0.hip",
                pytorch_func_dir=func_dir,
                pytorch_modu_dir=modu_dir,
                rtol=1e-4,
                atol=1e-5,
                perf_iterations=10,
                artifact_side="optimized",
                eval_backend="sandbox-inprocess",
                reference_hip_code_dir=reference_hip,
                reference_cache_mode="golden+compile",
            )

        self.assertEqual(identity["eval_backend"], "server-inprocess")
        self.assertEqual(identity["reference_hip_file"], "foo.hip")
        self.assertNotEqual(identity["hip_source_sha256"], identity["reference_hip_source_sha256"])


if __name__ == "__main__":
    unittest.main()
