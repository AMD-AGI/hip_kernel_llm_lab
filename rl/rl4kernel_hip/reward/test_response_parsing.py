import json
import os
import sys
import unittest
from unittest.mock import Mock, patch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from reward.reward_batch import compute_score_batch
from dataset.contracts import (
    SAMPLE_JSON_OUTPUT_CONTRACT,
    resolve_optimization_contract,
    validate_training_row_contract,
)
from reward.utils import (
    LEGACY_HIP_FENCE_OUTPUT_CONTRACT,
    parse_generation_response,
    parse_kernel_generation_response,
)


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
    return (
        "<think>\nKeep the original signature and optimize conservatively.\n</think>\n"
        + json.dumps(
            {
                "thought": "Keep the original signature and optimize conservatively.",
                "code": code,
            }
        )
    )


def _make_legacy_fenced_response(code: str) -> str:
    return (
        "Reason about coalescing first.\n"
        f"```hip\n{code}\n```"
    )


def _make_ground_truth() -> dict:
    return {
        "kernel_name": "test_kernel",
        "hip_code": HIP_REF,
        "pytorch_module_code": "class Model: pass",
        "pytorch_functional_code": FUNCTIONAL_CODE,
    }


class ResponseParsingTests(unittest.TestCase):
    def test_sample_json_response_is_parsed_for_react(self):
        result = parse_kernel_generation_response(
            _make_json_response(KERNEL_SNIPPET),
            data_source="kernel-agent-react-train",
            kernel_name="test_kernel",
            hip_ref=HIP_REF,
        )

        self.assertTrue(result["parse_ok"])
        self.assertEqual(result["parse_mode"], SAMPLE_JSON_OUTPUT_CONTRACT)
        self.assertIn("__global__ void test_kernel", result["hip_src"])

    def test_legacy_fenced_response_is_still_supported(self):
        result = parse_kernel_generation_response(
            _make_legacy_fenced_response(KERNEL_SNIPPET),
            data_source="kernel-agent-react-train",
            kernel_name="test_kernel",
            hip_ref=HIP_REF,
            output_contract=LEGACY_HIP_FENCE_OUTPUT_CONTRACT,
        )

        self.assertTrue(result["parse_ok"])
        self.assertEqual(result["parse_mode"], LEGACY_HIP_FENCE_OUTPUT_CONTRACT)
        self.assertIn("__global__ void test_kernel", result["hip_src"])

    def test_sample_json_contract_rejects_legacy_fence(self):
        result = parse_kernel_generation_response(
            _make_legacy_fenced_response(KERNEL_SNIPPET),
            data_source="kernel-agent-react-train",
            kernel_name="test_kernel",
            hip_ref=HIP_REF,
            output_contract=SAMPLE_JSON_OUTPUT_CONTRACT,
        )

        self.assertFalse(result["parse_ok"])
        self.assertEqual(result["parse_mode"], SAMPLE_JSON_OUTPUT_CONTRACT)
        self.assertIn("sample_json_v1", result["parse_error"])
        self.assertNotIn(LEGACY_HIP_FENCE_OUTPUT_CONTRACT, result["parse_error"])
        self.assertEqual(
            result["attempted_parse_modes"],
            [SAMPLE_JSON_OUTPUT_CONTRACT],
        )
        self.assertEqual(result["parse_attempt_chain"], SAMPLE_JSON_OUTPUT_CONTRACT)

    def test_malformed_json_response_is_rejected(self):
        malformed = '<think>inspect memory</think>{"thought": "ok", "code": "__global__ void test_kernel(float* x) { "'
        result = parse_kernel_generation_response(
            malformed,
            data_source="kernel-agent-react-train",
            kernel_name="test_kernel",
            hip_ref=HIP_REF,
            output_contract=SAMPLE_JSON_OUTPUT_CONTRACT,
        )

        self.assertFalse(result["parse_ok"])
        self.assertIn("sample_json_v1", result["parse_error"])
        self.assertEqual(result["parse_mode"], SAMPLE_JSON_OUTPUT_CONTRACT)

    def test_json_missing_code_field_is_rejected(self):
        response = '<think>inspect occupancy</think>' + json.dumps({"thought": "missing code"})
        result = parse_kernel_generation_response(
            response,
            data_source="kernel-agent-react-train",
            kernel_name="test_kernel",
            hip_ref=HIP_REF,
            output_contract=SAMPLE_JSON_OUTPUT_CONTRACT,
        )

        self.assertFalse(result["parse_ok"])
        self.assertIn("missing non-empty string `code` field", result["parse_error"])
        self.assertEqual(result["parse_mode"], SAMPLE_JSON_OUTPUT_CONTRACT)

    def test_json_code_without_global_kernel_is_rejected(self):
        response = _make_json_response("void helper(float* x) { x[0] = 1.0f; }")
        result = parse_kernel_generation_response(
            response,
            data_source="kernel-agent-react-train",
            kernel_name="test_kernel",
            hip_ref=HIP_REF,
            output_contract=SAMPLE_JSON_OUTPUT_CONTRACT,
        )

        self.assertFalse(result["parse_ok"])
        self.assertIn("does not contain a valid __global__ kernel", result["parse_error"])
        self.assertEqual(result["parse_mode"], SAMPLE_JSON_OUTPUT_CONTRACT)

    def test_hip_translation_unit_json_response_preserves_full_file(self):
        full_file = HIP_REF + "\nextern \"C\" void launch() {}\n"
        result = parse_generation_response(
            _make_json_response(full_file),
            data_source="hip2hip-train",
            output_contract=SAMPLE_JSON_OUTPUT_CONTRACT,
            expected_code_unit="hip_translation_unit",
        )

        self.assertTrue(result["parse_ok"])
        self.assertEqual(result["parse_mode"], SAMPLE_JSON_OUTPUT_CONTRACT)
        self.assertIn("#include <hip/hip_runtime.h>", result["hip_src"])
        self.assertIn("extern \"C\" void launch", result["hip_src"])

    def test_hip_translation_unit_rejects_markdown_residue_in_json_code(self):
        result = parse_generation_response(
            _make_json_response(f"```hip\n{HIP_REF}\n```"),
            data_source="hip2hip-train",
            output_contract=SAMPLE_JSON_OUTPUT_CONTRACT,
            expected_code_unit="hip_translation_unit",
        )

        self.assertFalse(result["parse_ok"])
        self.assertIn("markdown fence", result["parse_error"])

    def test_hip2hip_training_row_contract_is_validated(self):
        expected = resolve_optimization_contract(
            data_source="hip2hip-train",
            requested_paradigm="hip2hip",
            output_contract=SAMPLE_JSON_OUTPUT_CONTRACT,
        )
        row = {
            "data_source": "hip2hip-train",
            "prompt": [
                {
                    "role": "user",
                    "content": "### Starter HIP File (reference)\n```hip\n" + HIP_REF + "\n```",
                }
            ],
            "reward_model": {"ground_truth": _make_ground_truth()},
            "extra_info": {
                "output_contract": SAMPLE_JSON_OUTPUT_CONTRACT,
                "optimization_paradigm": "hip2hip_full_file",
                "expected_code_unit": "hip_translation_unit",
                "persistence_mode": "direct_full_file",
            },
        }

        self.assertEqual(validate_training_row_contract(row, expected), [])

    def test_hip2hip_contract_rejects_kernel_agent_data_source(self):
        row = {
            "data_source": "kernel-agent-react-train",
            "prompt": [
                {
                    "role": "user",
                    "content": "### Starter HIP File (reference)\n```hip\n" + HIP_REF + "\n```",
                }
            ],
            "reward_model": {"ground_truth": _make_ground_truth()},
            "extra_info": {
                "output_contract": SAMPLE_JSON_OUTPUT_CONTRACT,
                "optimization_paradigm": "hip2hip_full_file",
                "expected_code_unit": "hip_translation_unit",
            },
        }

        errors = validate_training_row_contract(row)
        self.assertTrue(errors)
        self.assertIn("invalid optimization contract", errors[0])

    def test_compute_score_batch_accepts_json_and_legacy_responses(self):
        def _mock_call(url, requests_data, timeout_s=600):
            self.assertEqual(url, "http://mock:8000/run_code")
            self.assertEqual(len(requests_data), 2)
            for request in requests_data:
                self.assertIn("#include <hip/hip_runtime.h>", request["hip_code"])
                self.assertIn("__global__ void test_kernel", request["hip_code"])
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "responses": [
                    {"compile_ok": True, "run_ok": True, "match_ok": True, "speedup": 1.0},
                    {"compile_ok": True, "run_ok": True, "match_ok": True, "speedup": 1.0},
                ],
                "total_time": 1.0,
                "batch_size": 2,
            }
            return mock_resp

        with patch("reward.reward_batch.call_batch_run_code", side_effect=_mock_call) as patched_call:
            scores = compute_score_batch(
                data_sources=["kernel-agent-react-train", "kernel-agent-react-train"],
                solution_strs=[
                    _make_json_response(KERNEL_SNIPPET),
                    _make_legacy_fenced_response(KERNEL_SNIPPET),
                ],
                ground_truths=[_make_ground_truth(), _make_ground_truth()],
                extra_infos=[
                    {
                        "sandbox_url": "http://mock:8000/run_code",
                        "output_contract": SAMPLE_JSON_OUTPUT_CONTRACT,
                    },
                    {
                        "sandbox_url": "http://mock:8000/run_code",
                        "output_contract": LEGACY_HIP_FENCE_OUTPUT_CONTRACT,
                    },
                ],
                reward_mode="correct_speedup_copy_penalty",
                reward_correct_speedup_r_ok=0.3,
                reward_correct_speedup_cap=10.0,
                reward_correct_speedup_copy_reward=0.0,
            )

        patched_call.assert_called_once()
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(isinstance(score, float) for score in scores))


if __name__ == "__main__":
    unittest.main()
