from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch

from .codegen import (
    build_candidate_compile_script,
    construct_reference_golden_script,
    construct_reference_perf_script,
    get_template_bundle_hash,
)
from .config import (
    EvalSettings,
    build_compile_identity,
    build_runtime_fingerprint,
    build_software_stack_fingerprint,
)
from .protocol import EvalRequest
from .runtime import (
    clear_pts,
    extract_golden_and_perf,
    extract_timeout,
    prepare_environment,
    run_python_script,
    to_cpu_obj,
    write_text_file,
)
from .cache import (
    ReferenceCache,
    ReferenceCompileArtifactKey,
    ReferenceGoldenKey,
    ReferencePerfKey,
    sha256_text,
    strip_candidate_hash_suffix,
)


def build_reference_keys(
    request: EvalRequest,
    *,
    settings: EvalSettings,
    gpu_id: Optional[int],
    driver_kind: str = 'functional',
) -> Tuple[ReferenceCompileArtifactKey, ReferenceGoldenKey, ReferencePerfKey]:
    logical_kernel_name = strip_candidate_hash_suffix(request.kernel_name)
    module_hash = sha256_text(request.pytorch_module_code) if driver_kind == 'module' else ''
    template_bundle_hash = get_template_bundle_hash()
    software_stack_fingerprint = build_software_stack_fingerprint(settings.effective_arch)
    reference_hip_code = request.hip_ref_code or request.hip_code
    compile_key = ReferenceCompileArtifactKey(
        logical_kernel_name=logical_kernel_name,
        driver_kind=driver_kind,
        hip_ref_sha256=sha256_text(reference_hip_code),
        pytorch_functional_sha256=sha256_text(request.pytorch_functional_code),
        pytorch_module_sha256=module_hash,
        template_bundle_sha256=template_bundle_hash,
        arch=settings.effective_arch,
        compiler_identity=build_compile_identity(settings.effective_arch),
    )
    golden_key = ReferenceGoldenKey(
        logical_kernel_name=logical_kernel_name,
        driver_kind=driver_kind,
        hip_ref_sha256=compile_key.hip_ref_sha256,
        pytorch_functional_sha256=compile_key.pytorch_functional_sha256,
        pytorch_module_sha256=compile_key.pytorch_module_sha256,
        template_bundle_sha256=template_bundle_hash,
        arch=settings.effective_arch,
        software_stack_fingerprint=software_stack_fingerprint,
    )
    perf_key = ReferencePerfKey(
        logical_kernel_name=logical_kernel_name,
        driver_kind=driver_kind,
        hip_ref_sha256=golden_key.hip_ref_sha256,
        pytorch_functional_sha256=golden_key.pytorch_functional_sha256,
        pytorch_module_sha256=golden_key.pytorch_module_sha256,
        template_bundle_sha256=template_bundle_hash,
        arch=settings.effective_arch,
        perf_iterations=settings.perf_iterations,
        runtime_fingerprint=build_runtime_fingerprint(gpu_id, settings.effective_arch),
    )
    return compile_key, golden_key, perf_key


def cache_meta_base(
    request: EvalRequest,
    settings: EvalSettings,
    *,
    gpu_id: Optional[int],
    driver_kind: str,
) -> Dict[str, Any]:
    return {
        'kernel_name': request.kernel_name,
        'logical_kernel_name': strip_candidate_hash_suffix(request.kernel_name),
        'driver_kind': driver_kind,
        'arch': settings.effective_arch,
        'template_bundle_sha256': get_template_bundle_hash(),
        'hip_ref_sha256': sha256_text(request.hip_ref_code or request.hip_code),
        'pytorch_functional_sha256': sha256_text(request.pytorch_functional_code),
        'pytorch_module_sha256': sha256_text(request.pytorch_module_code) if driver_kind == 'module' else '',
        'software_stack_fingerprint': build_software_stack_fingerprint(settings.effective_arch),
        'created_at_epoch': time.time(),
        'node_id': settings.node_id,
        'gpu_id': gpu_id,
    }


def _reference_module_name(compile_key: ReferenceCompileArtifactKey) -> str:
    return f'hip_ref_{compile_key.cache_id[:24]}'


def prepare_reference_compile_artifact(
    request: EvalRequest,
    *,
    tmp_dir: str,
    settings: EvalSettings,
    cache: ReferenceCache,
    compile_key: ReferenceCompileArtifactKey,
) -> Dict[str, Any]:
    module_name = _reference_module_name(compile_key)
    source_text = request.hip_ref_code or request.hip_code
    if settings.enable_ref_compile_cache:
        cached = cache.load_compile_artifact(compile_key)
        if cached is not None:
            return {
                **cached,
                'source_dir': cached.get('source_dir') or os.path.dirname(str(cached.get('source_path') or '')),
                'cache_hit': True,
                'persistent': True,
            }
        layout = cache.ensure_compile_source(
            compile_key,
            module_name=module_name,
            source_text=source_text,
        )
        return {
            **layout,
            'cache_hit': False,
            'persistent': True,
        }

    artifact_root = os.path.join(tmp_dir, 'reference_compile_artifact')
    source_dir = os.path.join(artifact_root, 'src')
    build_directory = os.path.join(artifact_root, 'build')
    source_path = os.path.join(source_dir, 'reference_kernel.hip')
    os.makedirs(source_dir, exist_ok=True)
    write_text_file(source_path, source_text)
    os.makedirs(build_directory, exist_ok=True)
    return {
        'artifact_root': artifact_root,
        'source_dir': source_dir,
        'source_path': source_path,
        'build_directory': build_directory,
        'meta_path': os.path.join(artifact_root, 'meta.json'),
        'module_name': module_name,
        'cache_hit': False,
        'persistent': False,
    }


def prepare_reference_compile_script(
    request: EvalRequest,
    *,
    tmp_dir: str,
    compile_artifact: Dict[str, Any],
) -> str:
    compile_script = os.path.join(tmp_dir, f'py_func_ref_compile_{request.kernel_name}.py')
    write_text_file(
        compile_script,
        build_candidate_compile_script(
            request.kernel_name,
            compile_artifact['source_dir'],
            os.path.basename(compile_artifact['source_path']),
            module_name=compile_artifact['module_name'],
            build_directory=compile_artifact['build_directory'],
        ),
    )
    return compile_script


def prepare_reference_only_files(
    request: EvalRequest,
    *,
    tmp_dir: str,
    settings: EvalSettings,
    compile_artifact: Dict[str, Any],
) -> Tuple[str, str, str, str]:
    result_dir = os.path.join(tmp_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)

    ref_golden_script = os.path.join(tmp_dir, f'py_func_ref_golden_{request.kernel_name}.py')
    ref_perf_script = os.path.join(tmp_dir, f'py_func_ref_perf_{request.kernel_name}.py')
    reference_golden_file = os.path.join(result_dir, f'reference_golden_{request.kernel_name}.pt')
    reference_perf_file = os.path.join(result_dir, f'reference_perf_{request.kernel_name}.pt')

    write_text_file(
        ref_golden_script,
        construct_reference_golden_script(
            request.pytorch_functional_code,
            request.kernel_name,
            compile_artifact['source_dir'],
            os.path.basename(compile_artifact['source_path']),
            reference_golden_file,
            module_name=compile_artifact['module_name'],
            build_directory=compile_artifact['build_directory'],
        ),
    )
    write_text_file(
        ref_perf_script,
        construct_reference_perf_script(
            request.pytorch_functional_code,
            request.kernel_name,
            compile_artifact['source_dir'],
            os.path.basename(compile_artifact['source_path']),
            reference_perf_file,
            settings.perf_iterations,
            module_name=compile_artifact['module_name'],
            build_directory=compile_artifact['build_directory'],
        ),
    )
    return ref_golden_script, ref_perf_script, reference_golden_file, reference_perf_file


def materialize_reference_cache_artifacts(
    request: EvalRequest,
    *,
    settings: EvalSettings,
    gpu_id: Optional[int],
    error_log_file: str,
    build_golden: bool,
    build_perf: bool,
    tmp_dir: str,
    tmp_dir_created: bool,
) -> Dict[str, Any]:
    env = prepare_environment(settings, gpu_id)
    _, run_timeout_s = extract_timeout(request, settings)
    cache = ReferenceCache(settings.cache_root)
    compile_key, golden_key, perf_key = build_reference_keys(request, settings=settings, gpu_id=gpu_id)
    artifacts: Dict[str, Any] = {}
    try:
        compile_artifact = prepare_reference_compile_artifact(
            request,
            tmp_dir=tmp_dir,
            settings=settings,
            cache=cache,
            compile_key=compile_key,
        )
        artifacts['compile_cache_hit'] = bool(compile_artifact.get('cache_hit'))
        if build_golden or build_perf:
            if not compile_artifact.get('cache_hit'):
                compile_script = prepare_reference_compile_script(
                    request,
                    tmp_dir=tmp_dir,
                    compile_artifact=compile_artifact,
                )
                run_python_script(
                    compile_script,
                    env=env,
                    timeout_s=run_timeout_s,
                    kernel_name=request.kernel_name,
                    stage='REF_COMPILE_PREWARM',
                    error_log_file=error_log_file,
                )
                if settings.enable_ref_compile_cache:
                    cache.store_compile_artifact(
                        compile_key,
                        {
                            **cache_meta_base(request, settings, gpu_id=gpu_id, driver_kind='functional'),
                            'compile_key': asdict(compile_key),
                            'compiler_identity': compile_key.compiler_identity,
                            'module_name': compile_artifact['module_name'],
                            'source_dir': compile_artifact['source_dir'],
                            'source_path': compile_artifact['source_path'],
                            'build_directory': compile_artifact['build_directory'],
                        },
                    )
            artifacts['compile_key'] = compile_key.cache_id
        ref_golden_script, ref_perf_script, reference_golden_file, reference_perf_file = prepare_reference_only_files(
            request,
            tmp_dir=tmp_dir,
            settings=settings,
            compile_artifact=compile_artifact,
        )
        if build_golden:
            run_python_script(
                ref_golden_script,
                env=env,
                timeout_s=run_timeout_s,
                kernel_name=request.kernel_name,
                stage='REF_GOLDEN_PREWARM',
                error_log_file=error_log_file,
            )
            ref_golden_payload = torch.load(reference_golden_file, map_location='cpu')
            reference_golden, _ = extract_golden_and_perf(ref_golden_payload)
            cache.store_golden(
                golden_key,
                to_cpu_obj(reference_golden) if settings.cache_golden_on_cpu else reference_golden,
                {
                    **cache_meta_base(request, settings, gpu_id=gpu_id, driver_kind='functional'),
                    'compile_key': asdict(compile_key),
                    'golden_key': asdict(golden_key),
                },
            )
            artifacts['golden'] = reference_golden
        if build_perf:
            run_python_script(
                ref_perf_script,
                env=env,
                timeout_s=run_timeout_s,
                kernel_name=request.kernel_name,
                stage='REF_PERF_PREWARM',
                error_log_file=error_log_file,
            )
            perf_payload = torch.load(reference_perf_file, map_location='cpu')
            _, reference_perf_ms = extract_golden_and_perf(perf_payload)
            if reference_perf_ms is None:
                raise RuntimeError('Reference perf payload missing \'perf\'')
            cache.store_perf(
                perf_key,
                reference_perf_ms,
                {
                    **cache_meta_base(request, settings, gpu_id=gpu_id, driver_kind='functional'),
                    'compile_key': asdict(compile_key),
                    'perf_key': asdict(perf_key),
                    'runtime_fingerprint': perf_key.runtime_fingerprint,
                    'perf_iterations': settings.perf_iterations,
                    'perf_cache_ttl_s': settings.ref_perf_cache_ttl_s,
                },
            )
            artifacts['perf_ms'] = reference_perf_ms
        return artifacts
    finally:
        clear_pts(os.path.join(tmp_dir, 'result'))
        if tmp_dir_created and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
