# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import inspect
import os
import sys
import unittest

import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(CURRENT_DIR)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from sandbox_core import eval as eval_core
from sandbox_core import codegen
from sandbox_core.runtime import compare_results


class CpuMaterializationTests(unittest.TestCase):
    def test_generated_result_dump_moves_tensors_to_cpu(self):
        script = codegen.LEGACY_DUMP_RESULT_CODE.format(result_path="/tmp/result.pt")
        self.assertIn("detach().cpu()", script)
        self.assertIn("torch.save(_to_cpu_obj(result_gold)", script)

    def test_generated_golden_dump_moves_tensors_to_cpu(self):
        script = codegen.GOLDEN_ONLY_DUMP_CODE.format(result_path="/tmp/result.pt")
        self.assertIn("detach().cpu()", script)
        self.assertIn("'golden': _to_cpu_obj(result_gold)", script)

    def test_eval_loads_candidate_payloads_on_cpu(self):
        runtime_source = inspect.getsource(eval_core.run_runtime_stage_request)
        full_source = inspect.getsource(eval_core.run_eval_request)
        self.assertIn("candidate_result_file']), map_location='cpu'", runtime_source)
        self.assertIn("candidate_result_file'], map_location='cpu'", full_source)

    def test_compare_results_normalizes_tensors_to_cpu(self):
        left = torch.tensor([1.0, 2.0])
        right = torch.tensor([1.0, 2.0])
        self.assertTrue(compare_results(left, right))


if __name__ == "__main__":
    unittest.main()
