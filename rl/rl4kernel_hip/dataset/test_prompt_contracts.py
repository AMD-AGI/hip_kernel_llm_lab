import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.contracts import LEGACY_HIP_FENCE_OUTPUT_CONTRACT
from dataset.utils import fetch_hip_kernel_agent_system_prompt


KERNEL_SNIPPET = """__global__ void test_kernel(float* x) {
    int idx = threadIdx.x;
    x[idx] = x[idx] + 1.0f;
}
"""


class PromptContractTests(unittest.TestCase):
    def test_default_prompt_uses_sample_json_contract(self):
        prompt = fetch_hip_kernel_agent_system_prompt(
            prompt="",
            starter_code=KERNEL_SNIPPET,
        )

        self.assertIn("exactly one JSON object", prompt)
        self.assertIn('"thought"', prompt)
        self.assertIn("First do optimization reasoning", prompt)

    def test_legacy_prompt_uses_fenced_hip_without_cot(self):
        prompt = fetch_hip_kernel_agent_system_prompt(
            prompt="",
            starter_code=KERNEL_SNIPPET,
            output_contract=LEGACY_HIP_FENCE_OUTPUT_CONTRACT,
        )

        self.assertIn("Output exactly one HIP code block", prompt)
        self.assertIn("```hip", prompt)
        self.assertNotIn("exactly one JSON object", prompt)
        self.assertNotIn('"thought"', prompt)
        self.assertNotIn("First do optimization reasoning", prompt)
        self.assertNotIn("You are working in think mode.", prompt)


if __name__ == "__main__":
    unittest.main()
