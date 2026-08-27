# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from reward.reward_batch import _compute_single_score_correct_speedup_copy_penalty, compute_score_batch
from reward.utils import SAMPLE_JSON_OUTPUT_CONTRACT


HIP_REF = """#include <hip/hip_runtime.h>
__global__ void test_kernel(float* x) {
    int idx = threadIdx.x;
    x[idx] = x[idx] + 1.0f;
}
"""

KERNEL_SNIPPET = """__global__ void test_kernel(float* x) {
    int idx = threadIdx.x;
    x[idx] = x[idx] + 1.0f;
}
"""

FUNCTIONAL_CODE = """class Model:
    pass


def get_inputs():
    return []


def get_init_inputs():
    return []
"""


def _make_json_response(code: str) -> str:
    return json.dumps(
        {
            "thought": "Keep the original signature and optimize conservatively.",
            "code": code,
        }
    )


def _make_ground_truth() -> dict:
    return {
        "kernel_name": "test_kernel",
        "hip_code": HIP_REF,
        "pytorch_module_code": "class Model: pass",
        "pytorch_functional_code": FUNCTIONAL_CODE,
    }


class _DummyTracker:
    def record(self, **kwargs):
        return None


class RewardBatchTests(unittest.TestCase):
    def _base_extra_info(self) -> dict:
        return {
            "sandbox_url": "http://mock:8000/run_code",
            "output_contract": SAMPLE_JSON_OUTPUT_CONTRACT,
            "train_step": 17,
            "prompt_uid": "uid-17",
            "sample_index": 3,
        }

    def _load_archive_rows(self, archive_root: str) -> list[dict]:
        rows = []
        for path in sorted(Path(archive_root).rglob("records.*.jsonl")):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        return rows

    def test_correct_speedup_bonus_applies_above_threshold(self):
        resp_data = {
            "compile_ok": True,
            "run_ok": True,
            "match_ok": True,
            "speedup": 1.06,
        }
        with patch.dict(
            os.environ,
            {
                "REWARD_CORRECT_SPEEDUP_R_OK": "0.3",
                "REWARD_CORRECT_SPEEDUP_BONUS": "0.3",
                "REWARD_CORRECT_SPEEDUP_BONUS_THRESHOLD": "1.05",
            },
            clear=False,
        ):
            reward = _compute_single_score_correct_speedup_copy_penalty(
                resp_data,
                dtw_to_ref=0.2,
                token_len=0,
            )
        self.assertAlmostEqual(reward, 1.66, places=6)

    def test_correct_speedup_bonus_is_strictly_above_threshold(self):
        resp_data = {
            "compile_ok": True,
            "run_ok": True,
            "match_ok": True,
            "speedup": 1.05,
        }
        with patch.dict(
            os.environ,
            {
                "REWARD_CORRECT_SPEEDUP_R_OK": "0.3",
                "REWARD_CORRECT_SPEEDUP_BONUS": "0.3",
                "REWARD_CORRECT_SPEEDUP_BONUS_THRESHOLD": "1.05",
            },
            clear=False,
        ):
            reward = _compute_single_score_correct_speedup_copy_penalty(
                resp_data,
                dtw_to_ref=0.2,
                token_len=0,
            )
        self.assertAlmostEqual(reward, 1.35, places=6)

    def test_copy_penalty_blocks_speedup_bonus(self):
        resp_data = {
            "compile_ok": True,
            "run_ok": True,
            "match_ok": True,
            "speedup": 1.20,
        }
        with patch.dict(
            os.environ,
            {
                "REWARD_CORRECT_SPEEDUP_COPY_REWARD": "0.0",
                "REWARD_CORRECT_SPEEDUP_BONUS": "0.3",
                "REWARD_CORRECT_SPEEDUP_BONUS_THRESHOLD": "1.05",
            },
            clear=False,
        ):
            reward = _compute_single_score_correct_speedup_copy_penalty(
                resp_data,
                dtw_to_ref=0.0,
                token_len=0,
            )
        self.assertAlmostEqual(reward, 0.0, places=6)

    def test_archive_row_written_for_success(self):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "responses": [
                {
                    "compile_ok": True,
                    "run_ok": True,
                    "match_ok": True,
                    "speedup": 1.25,
                    "reason": "",
                    "timing": {
                        "reference_compile_cache_hit": True,
                        "candidate_perf_ms": 1.2,
                        "reference_perf_ms": 1.5,
                    },
                }
            ],
            "total_time": 2.5,
            "batch_size": 1,
        }

        with TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "REWARD_EVAL_ARCHIVE_DIR": tmpdir,
                "REWARD_EVAL_EXPERIMENT_NAME": "unit-exp",
                "REWARD_EVAL_RUN_ID": "run-123",
            },
            clear=False,
        ), patch("reward.reward_batch.call_batch_run_code", return_value=mock_resp), patch(
            "reward.reward_batch.compute_dtw_to_ref",
            return_value=(0.2, 12),
        ), patch(
            "reward.reward_batch._compute_single_score_correct_speedup_copy_penalty",
            return_value=(1.23, False, 0.4, (0.1, 0.2, 0.3)),
        ), patch(
            "reward.reward_batch.get_global_tracker",
            return_value=_DummyTracker(),
        ):
            scores = compute_score_batch(
                data_sources=["kernel-agent-single-sft-train"],
                solution_strs=[_make_json_response(KERNEL_SNIPPET)],
                ground_truths=[_make_ground_truth()],
                extra_infos=[self._base_extra_info()],
                reward_mode="correct_speedup_copy_penalty",
                reward_correct_speedup_r_ok=0.3,
                reward_correct_speedup_cap=10.0,
                reward_correct_speedup_copy_reward=0.0,
            )

            rows = self._load_archive_rows(tmpdir)

        self.assertEqual(scores, [1.23])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["archive_version"], 1)
        self.assertEqual(row["run_id"], "run-123")
        self.assertEqual(row["experiment_name"], "unit-exp")
        self.assertEqual(row["train_step"], 17)
        self.assertEqual(row["prompt_uid"], "uid-17")
        self.assertEqual(row["sample_index"], 3)
        self.assertEqual(row["kernel_name_base"], "test_kernel")
        self.assertTrue(row["kernel_name"].startswith("test_kernel_"))
        self.assertTrue(row["compile_ok"])
        self.assertTrue(row["run_ok"])
        self.assertTrue(row["match_ok"])
        self.assertAlmostEqual(row["speedup"], 1.25)
        self.assertAlmostEqual(row["score"], 1.23)
        self.assertIn("#include <hip/hip_runtime.h>", row["hip_code"])
        self.assertTrue(row["hip_code_sha256"])
        self.assertNotIn("raw_response", row)

    def test_hip2hip_full_file_is_sent_directly_without_splice(self):
        optimized_full_file = (
            HIP_REF
            + "\n// full-file sentinel\n"
            + 'extern "C" void launch_test_kernel() {}\n'
        )

        def _mock_call(url, requests_data, timeout_s=600):
            self.assertEqual(url, "http://mock:8000/run_code")
            self.assertEqual(len(requests_data), 1)
            self.assertEqual(requests_data[0]["hip_code"], optimized_full_file.strip())
            self.assertIn("full-file sentinel", requests_data[0]["hip_code"])
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "responses": [
                    {"compile_ok": True, "run_ok": True, "match_ok": True, "speedup": 1.1}
                ],
                "total_time": 1.0,
                "batch_size": 1,
            }
            return mock_resp

        extra = {
            **self._base_extra_info(),
            "optimization_paradigm": "hip2hip_full_file",
            "expected_code_unit": "hip_translation_unit",
            "persistence_mode": "direct_full_file",
        }
        with patch("reward.reward_batch.call_batch_run_code", side_effect=_mock_call), patch(
            "reward.reward_batch.compute_dtw_to_ref",
            return_value=(0.2, 12),
        ), patch(
            "reward.reward_batch._compute_single_score_correct_speedup_copy_penalty",
            return_value=(1.0, False, 0.5, (0.1, 0.2, 0.3)),
        ), patch(
            "reward.reward_batch.get_global_tracker",
            return_value=_DummyTracker(),
        ):
            scores = compute_score_batch(
                data_sources=["hip2hip-train"],
                solution_strs=[_make_json_response(optimized_full_file)],
                ground_truths=[_make_ground_truth()],
                extra_infos=[extra],
                reward_mode="correct_speedup_copy_penalty",
                reward_correct_speedup_r_ok=0.3,
                reward_correct_speedup_cap=10.0,
                reward_correct_speedup_copy_reward=0.0,
            )

        self.assertEqual(scores, [1.0])

    def test_archive_row_written_for_parse_failure(self):
        bad_response = '{"thought":"missing code"}'

        with TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "REWARD_EVAL_ARCHIVE_DIR": tmpdir,
                "REWARD_EVAL_EXPERIMENT_NAME": "unit-exp",
                "REWARD_EVAL_RUN_ID": "run-parse-fail",
            },
            clear=False,
        ), patch("reward.reward_batch.call_batch_run_code") as patched_call:
            scores = compute_score_batch(
                data_sources=["kernel-agent-single-sft-train"],
                solution_strs=[bad_response],
                ground_truths=[_make_ground_truth()],
                extra_infos=[self._base_extra_info()],
                reward_mode="correct_speedup_copy_penalty",
                reward_correct_speedup_r_ok=0.3,
                reward_correct_speedup_cap=10.0,
                reward_correct_speedup_copy_reward=0.0,
            )

            rows = self._load_archive_rows(tmpdir)

        patched_call.assert_not_called()
        self.assertEqual(scores, [0.0])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertFalse(row["parse_ok"])
        self.assertEqual(row["reason"], "parse_failed")
        self.assertEqual(row["hip_code"], "")
        self.assertEqual(row["compile_ok"], False)
        self.assertTrue(row["parse_error"])

    def test_hip2hip_parse_failure_skips_sandbox(self):
        extra = {
            **self._base_extra_info(),
            "optimization_paradigm": "hip2hip_full_file",
            "expected_code_unit": "hip_translation_unit",
            "persistence_mode": "direct_full_file",
        }

        with patch("reward.reward_batch.call_batch_run_code") as patched_call:
            scores = compute_score_batch(
                data_sources=["hip2hip-train"],
                solution_strs=['{"thought":"missing code"}'],
                ground_truths=[_make_ground_truth()],
                extra_infos=[extra],
                reward_mode="correct_speedup_copy_penalty",
                reward_correct_speedup_r_ok=0.3,
                reward_correct_speedup_cap=10.0,
                reward_correct_speedup_copy_reward=0.0,
            )

        patched_call.assert_not_called()
        self.assertEqual(scores, [0.0])

    def test_archive_row_written_for_http_failure(self):
        mock_resp = Mock()
        mock_resp.status_code = 503
        mock_resp.text = "server unavailable"

        with TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "REWARD_EVAL_ARCHIVE_DIR": tmpdir,
                "REWARD_EVAL_EXPERIMENT_NAME": "unit-exp",
                "REWARD_EVAL_RUN_ID": "run-http-fail",
            },
            clear=False,
        ), patch("reward.reward_batch.call_batch_run_code", return_value=mock_resp):
            scores = compute_score_batch(
                data_sources=["kernel-agent-single-sft-train"],
                solution_strs=[_make_json_response(KERNEL_SNIPPET)],
                ground_truths=[_make_ground_truth()],
                extra_infos=[self._base_extra_info()],
                reward_mode="correct_speedup_copy_penalty",
                reward_correct_speedup_r_ok=0.3,
                reward_correct_speedup_cap=10.0,
                reward_correct_speedup_copy_reward=0.0,
            )

            rows = self._load_archive_rows(tmpdir)

        self.assertEqual(scores, [0.0])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["reason"], "http_status_503")
        self.assertFalse(row["compile_ok"])
        self.assertIn("#include <hip/hip_runtime.h>", row["hip_code"])

    def test_archive_row_written_for_batch_exception(self):
        with TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "REWARD_EVAL_ARCHIVE_DIR": tmpdir,
                "REWARD_EVAL_EXPERIMENT_NAME": "unit-exp",
                "REWARD_EVAL_RUN_ID": "run-exception",
            },
            clear=False,
        ), patch(
            "reward.reward_batch.call_batch_run_code",
            side_effect=RuntimeError("boom"),
        ):
            scores = compute_score_batch(
                data_sources=["kernel-agent-single-sft-train"],
                solution_strs=[_make_json_response(KERNEL_SNIPPET)],
                ground_truths=[_make_ground_truth()],
                extra_infos=[self._base_extra_info()],
                reward_mode="correct_speedup_copy_penalty",
                reward_correct_speedup_r_ok=0.3,
                reward_correct_speedup_cap=10.0,
                reward_correct_speedup_copy_reward=0.0,
            )

            rows = self._load_archive_rows(tmpdir)

        self.assertEqual(scores, [0.0])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["reason"], "batch_exception:RuntimeError")
        self.assertFalse(row["compile_ok"])
        self.assertIn("#include <hip/hip_runtime.h>", row["hip_code"])


if __name__ == "__main__":
    unittest.main()
