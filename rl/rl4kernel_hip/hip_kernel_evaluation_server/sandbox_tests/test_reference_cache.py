import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(CURRENT_DIR)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import torch

from sandbox_core.cache import (
    ReferenceCache,
    ReferenceCompileArtifactKey,
    ReferenceGoldenKey,
    ReferencePerfKey,
    strip_candidate_hash_suffix,
)
from sandbox_core.config import load_eval_settings
from sandbox_core.codegen import construct_reference_perf_script
from sandbox_core.eval import prewarm_reference_artifacts
from sandbox_core.logging_utils import format_evaluation_summary
from sandbox_core.protocol import EvalRequest
from sandbox_core.reference import build_reference_keys


class ReferenceCacheTests(unittest.TestCase):
    def test_strip_candidate_hash_suffix(self):
        self.assertEqual(strip_candidate_hash_suffix('kernel_abc123ef'), 'kernel')
        self.assertEqual(strip_candidate_hash_suffix('kernel_name'), 'kernel_name')

    def test_store_and_load_golden(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = ReferenceCache(tmp_dir)
            key = ReferenceGoldenKey(
                logical_kernel_name='kernel',
                driver_kind='functional',
                hip_ref_sha256='ref',
                pytorch_functional_sha256='func',
                pytorch_module_sha256='',
                template_bundle_sha256='bundle',
                arch='gfx942',
                software_stack_fingerprint={'torch_version': 'test'},
            )
            payload = {'golden': {'x': torch.tensor([1.0, 2.0])}}
            cache.store_golden(key, payload, {'meta': 'ok'})
            loaded = cache.load_golden(key)
            self.assertIsNotNone(loaded)
            golden, meta = loaded
            self.assertTrue(torch.equal(golden['golden']['x'], payload['golden']['x']))
            self.assertEqual(meta['meta'], 'ok')

    def test_store_and_load_perf_with_ttl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = ReferenceCache(tmp_dir)
            key = ReferencePerfKey(
                logical_kernel_name='kernel',
                driver_kind='functional',
                hip_ref_sha256='ref',
                pytorch_functional_sha256='func',
                pytorch_module_sha256='',
                template_bundle_sha256='bundle',
                arch='gfx942',
                perf_iterations=100,
                runtime_fingerprint={'host': 'nodeA'},
            )
            cache.store_perf(key, 1.23, {'meta': 'ok'})
            loaded = cache.load_perf(key, ttl_s=0)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded['meta'], 'ok')
            self.assertAlmostEqual(loaded['reference_perf_ms'], 1.23)
            loaded['created_at_epoch'] = time.time() - 100
            perf_path = Path(tmp_dir) / 'perf' / key.cache_id / 'perf.json'
            perf_path.write_text(__import__('json').dumps(loaded), encoding='utf-8')
            self.assertIsNone(cache.load_perf(key, ttl_s=1))

    def test_store_and_load_compile_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = ReferenceCache(tmp_dir)
            key = ReferenceCompileArtifactKey(
                logical_kernel_name='kernel',
                driver_kind='functional',
                hip_ref_sha256='ref',
                pytorch_functional_sha256='func',
                pytorch_module_sha256='',
                template_bundle_sha256='bundle',
                arch='gfx942',
                compiler_identity={'torch_version': 'test', 'node_id': 'nodeA'},
            )
            layout = cache.ensure_compile_source(key, module_name='hip_ref_test', source_text='extern "C" __global__ void k() {}')
            build_dir = Path(layout['build_directory'])
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / 'hip_ref_test.so').write_text('binary', encoding='utf-8')
            cache.store_compile_artifact(
                key,
                {
                    'compile_key': asdict(key),
                    'module_name': 'hip_ref_test',
                    'source_dir': layout['source_dir'],
                    'source_path': layout['source_path'],
                    'build_directory': layout['build_directory'],
                },
            )
            loaded = cache.load_compile_artifact(key)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded['module_name'], 'hip_ref_test')
            self.assertEqual(loaded['source_path'], layout['source_path'])

    def test_build_reference_keys_golden_key_changes_with_software_stack(self):
        request = EvalRequest(
            kernel_name='kernel_deadbeef',
            hip_code='extern "C" __global__ void k() {}',
            hip_ref_code='extern "C" __global__ void k() {}',
            pytorch_functional_code='import torch\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x\n\ndef get_inputs():\n    return [torch.ones(1, device="cuda")]\n\ndef get_init_inputs():\n    return []\n',
            pytorch_module_code='',
        )
        settings = load_eval_settings(gpu_ids=[0], explicit_arch='gfx942')
        with mock.patch('sandbox_core.reference.build_compile_identity', return_value={'compiler': 'fixed'}), \
             mock.patch('sandbox_core.reference.build_runtime_fingerprint', return_value={'runtime': 'fixed'}), \
             mock.patch('sandbox_core.reference.build_software_stack_fingerprint', return_value={'torch_version': 'stack_a'}):
            _, golden_a, _ = build_reference_keys(request, settings=settings, gpu_id=0)
        with mock.patch('sandbox_core.reference.build_compile_identity', return_value={'compiler': 'fixed'}), \
             mock.patch('sandbox_core.reference.build_runtime_fingerprint', return_value={'runtime': 'fixed'}), \
             mock.patch('sandbox_core.reference.build_software_stack_fingerprint', return_value={'torch_version': 'stack_b'}):
            _, golden_b, _ = build_reference_keys(request, settings=settings, gpu_id=0)
        self.assertNotEqual(golden_a.cache_id, golden_b.cache_id)

    def test_load_eval_settings_uses_nonzero_perf_ttl_default(self):
        with mock.patch.dict(os.environ, {'HIP_REF_PERF_CACHE_TTL_S': ''}, clear=False):
            settings = load_eval_settings()
        self.assertGreaterEqual(settings.ref_perf_cache_ttl_s, 1)

    def test_prewarm_perf_requires_gpu_id(self):
        request = EvalRequest(
            kernel_name='kernel_deadbeef',
            hip_code='extern "C" __global__ void k() {}',
            hip_ref_code='extern "C" __global__ void k() {}',
            pytorch_functional_code='def get_inputs():\n    return []\n\ndef get_init_inputs():\n    return []\n',
            pytorch_module_code='',
        )
        with self.assertRaises(ValueError):
            prewarm_reference_artifacts(request, settings=load_eval_settings(), gpu_id=None, with_perf=True)

    def test_reference_perf_script_uses_single_pre_measure_call(self):
        script = construct_reference_perf_script(
            pytorch_functional_code=(
                'import torch\n'
                'class Model(torch.nn.Module):\n'
                '    def forward(self, x):\n'
                '        return x\n\n'
                'def get_inputs():\n'
                '    return [torch.ones(1, device="cuda")]\n\n'
                'def get_init_inputs():\n'
                '    return []\n'
            ),
            kernel_name='kernel_deadbeef',
            hip_code_dir='/tmp/hip',
            hip_file='ref.hip',
            output_path='/tmp/out.pt',
            perf_iterations=10,
        )
        self.assertEqual(script.count('_ = _safe_call(model, inputs, hip_fn)'), 1)
        self.assertEqual(script.count('result_gold = _safe_call(model, inputs, hip_fn)'), 0)

    def test_evaluation_summary_reports_cache_telemetry(self):
        summary = format_evaluation_summary(
            'test summary',
            [
                {
                    'kernel_name': 'kernel_a',
                    'compile_ok': True,
                    'run_ok': True,
                    'match_ok': True,
                    'speedup': 1.2,
                    'timing': {
                        'reference_compile_cache_hit': True,
                        'reference_golden_cache_hit': False,
                        'reference_perf_cache_hit': True,
                        'reference_compile_build_s': 0.3,
                        'reference_golden_build_s': 0.7,
                        'reference_perf_build_s': 0.2,
                    },
                }
            ],
            total_elapsed=1.0,
        )
        self.assertIn('cache_hits', summary)
        self.assertIn('compile=1/1', summary)
        self.assertIn('golden=0/1', summary)
        self.assertIn('perf=1/1', summary)
        self.assertIn('avg_ref_build', summary)


if __name__ == '__main__':
    unittest.main()
