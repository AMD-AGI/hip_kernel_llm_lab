import os
import sys
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(CURRENT_DIR)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from sandbox_core.logging_utils import (
    derive_failure_reason,
    extract_concise_error,
    format_evaluation_summary,
    format_kernel_failure,
    format_kernel_success,
    strip_ansi,
)


class LoggingUtilsTests(unittest.TestCase):
    def test_extract_concise_error_prefers_real_error_line(self):
        raw_text = """
        Traceback (most recent call last):
          File "runner.py", line 1, in <module>
            raise AssertionError("shape mismatch")
        AssertionError: shape mismatch on dim=1
        """
        self.assertEqual(
            extract_concise_error(raw_text),
            "AssertionError: shape mismatch on dim=1",
        )

    def test_extract_concise_error_strips_stage_prefix(self):
        raw_text = "[ERROR] COMPILATION failed for fused_kernel_abc123: hipcc: error: unsupported argument '--bad-flag'"
        self.assertEqual(
            extract_concise_error(raw_text),
            "hipcc: error: unsupported argument '--bad-flag'",
        )

    def test_derive_failure_reason_uses_reason_and_detail(self):
        timing = {
            "failure_reason": "test run failed",
            "failure_detail": "AssertionError: shape mismatch on dim=1",
        }
        self.assertEqual(
            derive_failure_reason(False, False, False, timing),
            "test run failed: AssertionError: shape mismatch on dim=1",
        )

    def test_format_evaluation_summary_is_structured(self):
        summary = format_evaluation_summary(
            "Parallel evaluation",
            [
                {
                    "kernel_name": "kernel_fast",
                    "compile_ok": True,
                    "run_ok": True,
                    "match_ok": True,
                    "speedup": 1.125,
                    "timing": {"total": 10.0},
                },
                {
                    "kernel_name": "kernel_slow",
                    "compile_ok": True,
                    "run_ok": True,
                    "match_ok": True,
                    "speedup": 0.925,
                    "timing": {"total": 11.0},
                },
                {
                    "kernel_name": "kernel_fail",
                    "compile_ok": True,
                    "run_ok": True,
                    "match_ok": False,
                    "speedup": 0.0,
                    "timing": {"failure_reason": "result mismatch"},
                },
            ],
            total_elapsed=30.0,
        )
        plain_summary = strip_ansi(summary)
        self.assertIn("[SUMMARY] Parallel evaluation", plain_summary)
        self.assertIn("tasks        : 3", plain_summary)
        self.assertIn("success      : 2 (66.7%)", plain_summary)
        self.assertIn("failed       : 1", plain_summary)
        self.assertIn("avg_per_task : 10.00s", plain_summary)
        self.assertIn("speedup_mix  : >1.0x=1, <=1.0x=1", plain_summary)
        self.assertIn("fail_reasons : result mismatch x1", plain_summary)

    def test_format_kernel_success_includes_reference_cache_hits(self):
        rendered = strip_ansi(
            format_kernel_success(
                "kernel_hit",
                speedup=1.25,
                timing={
                    "total": 12.0,
                    "reference_compile_cache_hit": True,
                    "reference_golden_cache_hit": True,
                },
            )
        )
        self.assertIn("[PASS]", rendered)
        self.assertIn("kernel_hit", rendered)
        self.assertIn("ref_cache=compile,golden", rendered)
        self.assertIn("total=12.00s", rendered)

    def test_format_kernel_failure_includes_reference_cache_hits(self):
        rendered = strip_ansi(
            format_kernel_failure(
                "kernel_hit_fail",
                compile_ok=False,
                run_ok=False,
                match_ok=False,
                timing={
                    "failure_reason": "parallel worker failed",
                    "reference_golden_cache_hit": True,
                    "reference_perf_cache_hit": True,
                },
            )
        )
        self.assertIn("[FAIL]", rendered)
        self.assertIn("kernel_hit_fail", rendered)
        self.assertIn("ref_cache=golden,perf", rendered)


if __name__ == "__main__":
    unittest.main()
