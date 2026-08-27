# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(CURRENT_DIR)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from sandbox_core import eval as eval_core
from sandbox_core import parallel as parallel_core
from sandbox_core.config import EvalSettings, eval_settings_from_payload, load_eval_settings
from sandbox_core.protocol import EvalResponse
from sandbox_core.result import EvalRunResult
from sandbox_core.tool_protocol import (
    KernelToolCreateSessionResponse,
    KernelToolObservation,
    KernelToolSchedulerStatus,
    KernelToolUpdateCandidateResponse,
)

from server_adapters import master as master_server
from server_adapters import single as server_req_deploy_hip2hip
from server_adapters import worker as server_req_deploy_hip2hip_batch
from server_adapters import tool_router


SAMPLE_REQUEST = {
    'kernel_name': 'test_kernel',
    'hip_code': '__global__ void test_kernel() {}',
    'hip_ref_code': '__global__ void test_kernel() {}',
    'pytorch_module_code': 'class Model: pass',
    'pytorch_functional_code': 'class Model: pass\n\ndef get_inputs():\n    return []\n\ndef get_init_inputs():\n    return []',
    'atol': 1e-4,
    'rtol': 1e-3,
}


class ServerContractTests(unittest.TestCase):
    def _make_eval_settings(self, **overrides):
        settings = {
            'gpu_ids': [0],
            'node_id': 'test-node',
            'error_log_dir': '/tmp/error_log',
            'perf_iterations': 1000,
            'speedup_confirm_enabled': True,
            'speedup_confirm_threshold': 1.05,
            'speedup_confirm_band': 0.02,
            'speedup_confirm_iterations': 3000,
            'compile_timeout_s': 600,
            'run_timeout_s': 600,
            'handler_timeout_s': 1200,
            'effective_arch': 'gfx942',
            'cache_root': '/tmp/reference_cache',
            'enable_ref_compile_cache': False,
            'enable_ref_golden_cache': False,
            'enable_ref_perf_cache': False,
            'ref_perf_cache_ttl_s': 0,
            'cache_golden_on_cpu': True,
            'cleanup_tmp_on_success': False,
            'retain_tmp_on_failure': True,
        }
        settings.update(overrides)
        return EvalSettings(**settings)

    def test_single_server_run_code_contract(self):
        with TestClient(server_req_deploy_hip2hip.app) as client, patch.object(
            server_req_deploy_hip2hip,
            'run_eval_request',
            return_value=EvalRunResult(True, True, True, 1.5, {'total': 1.0}),
        ):
            response = client.post('/run_code', json=SAMPLE_REQUEST)
        self.assertEqual(response.status_code, 200)
        payload = response.json()['msg']
        self.assertTrue(payload['compile_ok'])
        self.assertTrue(payload['run_ok'])
        self.assertTrue(payload['match_ok'])
        self.assertAlmostEqual(payload['speedup'], 1.5)

    def test_load_eval_settings_disables_confirmation_by_default(self):
        with patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "0"}, clear=True):
            settings = load_eval_settings()
        self.assertFalse(settings.speedup_confirm_enabled)

    def test_eval_settings_from_payload_backfills_new_fields(self):
        payload = {
            'gpu_ids': [0],
            'node_id': 'test-node',
            'error_log_dir': '/tmp/error_log',
            'perf_iterations': 1000,
            'speedup_confirm_enabled': False,
            'speedup_confirm_threshold': 1.05,
            'speedup_confirm_band': 0.02,
            'speedup_confirm_iterations': 3000,
            'compile_timeout_s': 600,
            'run_timeout_s': 600,
            'handler_timeout_s': 1200,
            'effective_arch': 'gfx942',
            'cache_root': '/tmp/reference_cache',
            'enable_ref_golden_cache': False,
            'enable_ref_perf_cache': False,
            'ref_perf_cache_ttl_s': 3600,
            'cache_golden_on_cpu': True,
            'cleanup_tmp_on_success': False,
            'retain_tmp_on_failure': True,
        }
        settings = eval_settings_from_payload(payload)
        self.assertFalse(settings.enable_ref_compile_cache)

    def test_partition_cpu_ids_evenly_for_gpu_slots(self):
        groups = parallel_core._partition_cpu_ids(list(range(16)), 8)
        self.assertEqual(len(groups), 8)
        self.assertEqual(groups[0], [0, 1])
        self.assertEqual(groups[-1], [14, 15])

    def test_build_affinity_metadata_assigns_cpu_group_per_gpu(self):
        with patch.dict(os.environ, {"HIP_ENABLE_CPU_AFFINITY": "1"}, clear=False), patch.object(
            parallel_core,
            "_available_cpu_ids",
            return_value=list(range(16)),
        ), patch("sandbox_core.parallel.os.sched_setaffinity") as affinity_mock:
            metadata = parallel_core._build_affinity_metadata(
                3,
                {"gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7]},
            )
        affinity_mock.assert_called_once()
        self.assertTrue(metadata["cpu_affinity_enabled"])
        self.assertTrue(metadata["cpu_affinity_applied"])
        self.assertEqual(metadata["assigned_gpu_id"], 3)
        self.assertEqual(metadata["assigned_cpu_cores"], [6, 7])

    def test_single_server_preserves_confirmation_timing(self):
        timing = {
            'total': 1.0,
            'speedup_confirm_used': True,
            'speedup_initial': 1.051,
            'speedup_final': 1.04,
            'speedup_confirm_status': 'confirmed',
        }
        with TestClient(server_req_deploy_hip2hip.app) as client, patch.object(
            server_req_deploy_hip2hip,
            'run_eval_request',
            return_value=EvalRunResult(True, True, True, 1.04, timing),
        ):
            response = client.post('/run_code', json=SAMPLE_REQUEST)
        self.assertEqual(response.status_code, 200)
        payload = response.json()['msg']
        self.assertTrue(payload['timing']['speedup_confirm_used'])
        self.assertEqual(payload['timing']['speedup_confirm_status'], 'confirmed')
        self.assertAlmostEqual(payload['timing']['speedup_final'], 1.04)

    def test_worker_batch_contract(self):
        with TestClient(server_req_deploy_hip2hip_batch.app) as client, patch.object(
            server_req_deploy_hip2hip_batch,
            'evaluate_requests_parallel',
            return_value=[EvalRunResult(True, True, True, 2.0, {'total': 2.0})],
        ):
            response = client.post('/run_code_batch', json={'requests': [SAMPLE_REQUEST]})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload['batch_size'], 1)
        self.assertEqual(len(payload['responses']), 1)
        self.assertAlmostEqual(payload['responses'][0]['speedup'], 2.0)

    def test_worker_batch_failure_reason_is_exposed(self):
        with TestClient(server_req_deploy_hip2hip_batch.app) as client, patch.object(
            server_req_deploy_hip2hip_batch,
            'evaluate_requests_parallel',
            return_value=[
                EvalRunResult(
                    False,
                    False,
                    False,
                    0.0,
                    {
                        'failure_reason': 'compilation failed',
                        'failure_detail': 'error: hipcc exited with code 1',
                    },
                )
            ],
        ):
            response = client.post('/run_code_batch', json={'requests': [SAMPLE_REQUEST]})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload['responses'][0]['reason'],
            'compilation failed: error: hipcc exited with code 1',
        )

    def test_worker_single_gpu_contract(self):
        gpu_request = {**SAMPLE_REQUEST, 'gpu_id': server_req_deploy_hip2hip_batch.GPU_IDS[0]}
        with TestClient(server_req_deploy_hip2hip_batch.app) as client, patch.object(
            server_req_deploy_hip2hip_batch,
            'run_eval_request',
            return_value=EvalRunResult(True, True, True, 3.0, {'total': 3.0}),
        ):
            response = client.post('/run_code_single_gpu', json=gpu_request)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload['match_ok'])
        self.assertAlmostEqual(payload['speedup'], 3.0)

    def test_worker_health_exposes_tool_scheduler(self):
        with TestClient(server_req_deploy_hip2hip_batch.app) as client, patch.object(
            tool_router.tool_runtime,
            "scheduler_status",
            return_value=KernelToolSchedulerStatus(
                cpu_slots=4,
                cpu_slots_in_use=1,
                gpu_slots={"0": {"in_use": 0, "pending": 0}},
                session_count=2,
            ),
        ):
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("tool_scheduler", payload)
        self.assertEqual(payload["tool_scheduler"]["session_count"], 2)

    def test_worker_tool_routes_contract(self):
        observation = KernelToolObservation(
            session_id="req-1",
            operation="compile_check",
            status="ok",
            artifact_id="cand1234",
            kernel_name="test_kernel_cand1234",
            candidate_hash="cand1234",
            compile_ok=True,
            run_ok=False,
            match_ok=False,
            speedup=0.0,
            observation="operation=compile_check; status=ok; compile_ok=True",
            budget={"tool_calls_used": 1, "tool_calls_remaining": 3},
        )
        with TestClient(server_req_deploy_hip2hip_batch.app) as client, patch.object(
            tool_router.tool_runtime,
            "create_session",
            return_value=KernelToolCreateSessionResponse(
                session_id="req-1",
                kernel_name="test_kernel",
                budget={"tool_calls_used": 0, "tool_calls_remaining": 4},
            ),
        ), patch.object(
            tool_router.tool_runtime,
            "update_candidate",
            return_value=KernelToolUpdateCandidateResponse(
                session_id="req-1",
                artifact_id="cand1234",
                kernel_name="test_kernel_cand1234",
                updated=True,
                candidate_hash="cand1234",
                message="candidate updated",
            ),
        ), patch.object(
            tool_router.tool_runtime,
            "compile_check",
            return_value=observation,
        ):
            create_resp = client.post(
                "/tool/create_session",
                json={
                    "session_id": "req-1",
                    "reference": {
                        "problem_id": "sample-1",
                        "kernel_name": "test_kernel",
                        "hip_ref_code": "__global__ void test_kernel() {}",
                        "pytorch_functional_code": "def get_inputs():\n    return []",
                    },
                    "budget": {"max_tool_calls": 4, "max_wallclock_s": 300},
                },
            )
            update_resp = client.post(
                "/tool/update_candidate",
                json={
                    "session_id": "req-1",
                    "hip_code": "__global__ void test_kernel() {}",
                },
            )
            compile_resp = client.post(
                "/tool/compile_check",
                json={"session_id": "req-1"},
            )
        self.assertEqual(create_resp.status_code, 200)
        self.assertEqual(update_resp.status_code, 200)
        self.assertEqual(compile_resp.status_code, 200)
        self.assertEqual(compile_resp.json()["artifact_id"], "cand1234")
        self.assertTrue(compile_resp.json()["compile_ok"])

    def test_master_batch_contract(self):
        async def fake_dispatch_batch(tasks, batch_tmp_dir):
            return [
                EvalResponse(
                    kernel_name=tasks[0].kernel_name,
                    compile_ok=True,
                    run_ok=True,
                    match_ok=True,
                    speedup=4.0,
                    timing={'total': 4.0},
                )
            ]

        with TestClient(master_server.app) as client, patch.object(master_server.slot_manager, 'dispatch_batch', side_effect=fake_dispatch_batch):
            response = client.post('/run_code_batch', json={'requests': [SAMPLE_REQUEST]})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload['batch_size'], 1)
        self.assertAlmostEqual(payload['responses'][0]['speedup'], 4.0)

    def test_confirm_speedup_uses_conservative_min_value(self):
        request = eval_core.EvalRequest(**SAMPLE_REQUEST)
        settings = self._make_eval_settings()
        timing = {}
        confirm_paths = {
            'candidate_confirm_script': '/tmp/candidate_confirm.py',
            'candidate_confirm_result_file': '/tmp/candidate_confirm.pt',
            'reference_confirm_script': '/tmp/reference_confirm.py',
            'reference_confirm_result_file': '/tmp/reference_confirm.pt',
        }
        with patch.object(eval_core, '_prepare_speedup_confirmation_files', return_value=confirm_paths), patch.object(
            eval_core,
            '_run_perf_script_for_speedup',
            side_effect=[(100.0, 0.1), (104.0, 0.1)],
        ):
            speedup = eval_core._confirm_speedup_if_needed(
                request,
                tmp_dir='/tmp/eval',
                env={},
                run_timeout_s=600,
                error_log_file='/tmp/eval/error.log',
                settings=settings,
                timing=timing,
                first_pass_speedup=1.051,
            )
        self.assertAlmostEqual(speedup, 1.04, places=6)
        self.assertTrue(timing['speedup_confirm_used'])
        self.assertEqual(timing['speedup_confirm_status'], 'confirmed')
        self.assertAlmostEqual(timing['speedup_confirm_speedup'], 1.04, places=6)
        self.assertAlmostEqual(timing['speedup_final'], 1.04, places=6)

    def test_confirm_speedup_skips_outside_window(self):
        request = eval_core.EvalRequest(**SAMPLE_REQUEST)
        settings = self._make_eval_settings()
        timing = {}
        with patch.object(eval_core, '_prepare_speedup_confirmation_files') as prepare_mock, patch.object(
            eval_core,
            '_run_perf_script_for_speedup',
        ) as run_mock:
            speedup = eval_core._confirm_speedup_if_needed(
                request,
                tmp_dir='/tmp/eval',
                env={},
                run_timeout_s=600,
                error_log_file='/tmp/eval/error.log',
                settings=settings,
                timing=timing,
                first_pass_speedup=1.20,
            )
        self.assertAlmostEqual(speedup, 1.20, places=6)
        self.assertFalse(timing['speedup_confirm_used'])
        self.assertEqual(timing['speedup_confirm_status'], 'skipped')
        prepare_mock.assert_not_called()
        run_mock.assert_not_called()

    def test_confirm_speedup_falls_back_to_threshold_on_error(self):
        request = eval_core.EvalRequest(**SAMPLE_REQUEST)
        settings = self._make_eval_settings()
        timing = {}
        confirm_paths = {
            'candidate_confirm_script': '/tmp/candidate_confirm.py',
            'candidate_confirm_result_file': '/tmp/candidate_confirm.pt',
            'reference_confirm_script': '/tmp/reference_confirm.py',
            'reference_confirm_result_file': '/tmp/reference_confirm.pt',
        }
        with patch.object(eval_core, '_prepare_speedup_confirmation_files', return_value=confirm_paths), patch.object(
            eval_core,
            '_run_perf_script_for_speedup',
            side_effect=RuntimeError('confirm failed'),
        ):
            speedup = eval_core._confirm_speedup_if_needed(
                request,
                tmp_dir='/tmp/eval',
                env={},
                run_timeout_s=600,
                error_log_file='/tmp/eval/error.log',
                settings=settings,
                timing=timing,
                first_pass_speedup=1.051,
            )
        self.assertAlmostEqual(speedup, 1.05, places=6)
        self.assertTrue(timing['speedup_confirm_used'])
        self.assertEqual(timing['speedup_confirm_status'], 'fallback')
        self.assertIn('confirm failed', timing['speedup_confirm_error'])
        self.assertAlmostEqual(timing['speedup_final'], 1.05, places=6)


if __name__ == '__main__':
    unittest.main()
