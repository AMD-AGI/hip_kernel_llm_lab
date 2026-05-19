from __future__ import annotations

import os
import time
import traceback
from dataclasses import asdict, replace
from typing import Any, Dict, List, NamedTuple, Optional

import torch

from .codegen import (
    build_candidate_compile_script,
    construct_pytorch_functional_unittest,
    construct_pytorch_module_unittest,
    construct_reference_perf_script,
    extract_module_name,
    get_template_bundle_hash,
    result_path,
)
from .config import EvalSettings, load_eval_settings
from .parallel import evaluate_requests_parallel as _evaluate_requests_parallel
from .protocol import EvalRequest
from .reference import (
    build_reference_keys,
    cache_meta_base,
    materialize_reference_cache_artifacts,
    prepare_reference_compile_artifact,
    prepare_reference_compile_script,
    prepare_reference_only_files,
)
from .result import EvalRunResult
from .logging_utils import summarize_failure_exception
from .runtime import (
    clear_pts,
    compare_results,
    extract_golden_and_perf,
    extract_timeout,
    ensure_tmp_dir,
    log_error,
    prepare_environment,
    run_python_script,
    to_cpu_obj,
    write_text_file,
)
from .cache import CACHE_SCHEMA_VERSION, ReferenceCache

TEMPLATE_BUNDLE_HASH = get_template_bundle_hash()
_compare_results = compare_results
_build_reference_keys = build_reference_keys


class CompileStageResult(NamedTuple):
    result: EvalRunResult
    artifact: Dict[str, Any]
    tmp_dir_created: bool


def _candidate_module_name(request: EvalRequest) -> str:
    return f'hip_{request.kernel_name}'


def _candidate_build_directory(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, 'candidate_build')


def _base_timing(request: EvalRequest, settings: EvalSettings) -> Dict[str, Any]:
    return {
        'reference_compile_cache_hit': False,
        'reference_golden_cache_hit': False,
        'reference_perf_cache_hit': False,
        'candidate_identity': request.kernel_name,
        'reference_identity': request.kernel_name,
        'reference_compile_cache_enabled': bool(settings.enable_ref_compile_cache),
        'reference_perf_cache_ttl_s': int(settings.ref_perf_cache_ttl_s),
        'speedup_confirm_used': False,
        'eval_pipeline': 'single_chain',
    }


def _build_failure_result(
    *,
    total_start: float,
    timing: Dict[str, Any],
    stage: str,
    reason: str,
    detail: Optional[str],
    compile_ok: bool,
    run_ok: bool,
    match_ok: bool,
) -> EvalRunResult:
    timing['failure_stage'] = stage
    timing['failure_reason'] = reason
    if detail:
        timing['failure_detail'] = detail
    timing['total'] = time.time() - total_start
    return EvalRunResult.failure(
        compile_ok=compile_ok,
        run_ok=run_ok,
        match_ok=match_ok,
        timing=timing,
    )


def _should_confirm_speedup(speedup: float, settings: EvalSettings) -> bool:
    if not settings.speedup_confirm_enabled:
        return False
    lower = max(0.0, settings.speedup_confirm_threshold - settings.speedup_confirm_band)
    upper = settings.speedup_confirm_threshold + settings.speedup_confirm_band
    return lower <= float(speedup) <= upper


def _prepare_speedup_confirmation_files(
    request: EvalRequest,
    *,
    tmp_dir: str,
    perf_iterations: int,
    module_name: Optional[str] = None,
    build_directory: Optional[str] = None,
) -> Dict[str, str]:
    hip_dir = os.path.join(tmp_dir, 'hip')
    result_dir = os.path.join(tmp_dir, 'result')
    os.makedirs(hip_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    candidate_confirm_script = os.path.join(tmp_dir, f'py_func_confirm_{request.kernel_name}.py')
    candidate_confirm_result_file = result_path(result_dir, 'py_func_confirm_', request.kernel_name)
    reference_confirm_script = os.path.join(tmp_dir, f'py_func_ref_confirm_{request.kernel_name}.py')
    reference_confirm_result_file = os.path.join(result_dir, f'reference_perf_confirm_{request.kernel_name}.pt')

    write_text_file(
        candidate_confirm_script,
        construct_pytorch_functional_unittest(
            request.pytorch_functional_code,
            request.kernel_name,
            hip_dir,
            f'hip_{request.kernel_name}.hip',
            result_dir,
            prefix='py_func_confirm_',
            perf_iterations=perf_iterations,
            module_name=module_name,
            build_directory=build_directory,
        ),
    )
    write_text_file(
        reference_confirm_script,
        construct_reference_perf_script(
            request.pytorch_functional_code,
            request.kernel_name,
            hip_dir,
            f'hip_ref_{request.kernel_name}.hip',
            reference_confirm_result_file,
            perf_iterations=perf_iterations,
        ),
    )
    return {
        'candidate_confirm_script': candidate_confirm_script,
        'candidate_confirm_result_file': candidate_confirm_result_file,
        'reference_confirm_script': reference_confirm_script,
        'reference_confirm_result_file': reference_confirm_result_file,
    }


def _run_perf_script_for_speedup(
    script_path: str,
    result_file: str,
    *,
    env: Dict[str, str],
    timeout_s: int,
    kernel_name: str,
    stage: str,
    error_log_file: str,
    perf_label: str,
) -> tuple[float, float]:
    stage_start = time.time()
    run_python_script(
        script_path,
        env=env,
        timeout_s=timeout_s,
        kernel_name=kernel_name,
        stage=stage,
        error_log_file=error_log_file,
    )
    elapsed = time.time() - stage_start
    perf_payload = torch.load(result_file, map_location='cpu')
    _, perf_ms = extract_golden_and_perf(perf_payload)
    if perf_ms is None:
        raise RuntimeError(f"{perf_label} perf payload missing 'perf'")
    return float(perf_ms), elapsed


def _confirm_speedup_if_needed(
    request: EvalRequest,
    *,
    tmp_dir: str,
    env: Dict[str, str],
    run_timeout_s: int,
    error_log_file: str,
    settings: EvalSettings,
    timing: Dict[str, Any],
    first_pass_speedup: float,
    module_name: Optional[str] = None,
    build_directory: Optional[str] = None,
) -> float:
    threshold = float(settings.speedup_confirm_threshold)
    band = float(settings.speedup_confirm_band)
    lower = max(0.0, threshold - band)
    upper = threshold + band
    timing['speedup_initial'] = float(first_pass_speedup)
    timing['speedup_confirm_enabled'] = bool(settings.speedup_confirm_enabled)
    timing['speedup_confirm_threshold'] = threshold
    timing['speedup_confirm_band'] = band
    timing['speedup_confirm_window_low'] = lower
    timing['speedup_confirm_window_high'] = upper

    if not _should_confirm_speedup(first_pass_speedup, settings):
        timing['speedup_confirm_used'] = False
        timing['speedup_confirm_status'] = 'skipped'
        timing['speedup_final'] = float(first_pass_speedup)
        return float(first_pass_speedup)

    confirm_settings = replace(settings, perf_iterations=max(settings.perf_iterations, settings.speedup_confirm_iterations))
    timing['speedup_confirm_used'] = True
    timing['speedup_confirm_iterations'] = int(confirm_settings.perf_iterations)

    try:
        confirm_paths = _prepare_speedup_confirmation_files(
            request,
            tmp_dir=tmp_dir,
            perf_iterations=confirm_settings.perf_iterations,
            module_name=module_name,
            build_directory=build_directory,
        )
        candidate_confirm_perf_ms, candidate_confirm_elapsed = _run_perf_script_for_speedup(
            confirm_paths['candidate_confirm_script'],
            confirm_paths['candidate_confirm_result_file'],
            env=env,
            timeout_s=run_timeout_s,
            kernel_name=request.kernel_name,
            stage='CONFIRM_TEST_RUN',
            error_log_file=error_log_file,
            perf_label='Candidate confirm',
        )
        reference_confirm_perf_ms, reference_confirm_elapsed = _run_perf_script_for_speedup(
            confirm_paths['reference_confirm_script'],
            confirm_paths['reference_confirm_result_file'],
            env=env,
            timeout_s=run_timeout_s,
            kernel_name=request.kernel_name,
            stage='CONFIRM_REF_PERF_RUN',
            error_log_file=error_log_file,
            perf_label='Reference confirm',
        )
        if candidate_confirm_perf_ms == 0:
            raise RuntimeError('Candidate confirm perf is zero; cannot compute confirmed speedup')

        confirm_speedup = float(reference_confirm_perf_ms) / float(candidate_confirm_perf_ms)
        final_speedup = min(float(first_pass_speedup), float(confirm_speedup))
        timing['speedup_confirm_status'] = 'confirmed'
        timing['speedup_confirm_candidate_perf_ms'] = float(candidate_confirm_perf_ms)
        timing['speedup_confirm_reference_perf_ms'] = float(reference_confirm_perf_ms)
        timing['speedup_confirm_speedup'] = float(confirm_speedup)
        timing['speedup_confirm_test_run'] = candidate_confirm_elapsed
        timing['speedup_confirm_ref_run'] = reference_confirm_elapsed
        timing['speedup_final'] = float(final_speedup)
        return float(final_speedup)
    except Exception as exc:
        fallback_speedup = min(float(first_pass_speedup), threshold)
        timing['speedup_confirm_status'] = 'fallback'
        timing['speedup_confirm_error'] = str(exc)
        timing['speedup_final'] = float(fallback_speedup)
        return float(fallback_speedup)


def perf_compile(
    kernel_name: str,
    hip_code: str,
    tmp_dir: str = 'temp',
    *,
    gpu_id: Optional[int] = None,
    settings: Optional[EvalSettings] = None,
) -> bool:
    settings = settings or load_eval_settings()
    compile_timeout_s = settings.compile_timeout_s
    env = prepare_environment(settings, gpu_id)
    hip_dir = os.path.join(tmp_dir, 'hip')
    build_directory = _candidate_build_directory(tmp_dir)
    os.makedirs(hip_dir, exist_ok=True)
    os.makedirs(build_directory, exist_ok=True)
    hip_file = os.path.join(hip_dir, f'hip_{kernel_name}.hip')
    compile_script = os.path.join(tmp_dir, f'hip_comp_{kernel_name}.py')
    write_text_file(hip_file, hip_code)
    write_text_file(
        compile_script,
        build_candidate_compile_script(
            kernel_name,
            hip_dir,
            f'hip_{kernel_name}.hip',
            build_directory=build_directory,
        ),
    )
    try:
        run_python_script(
            compile_script,
            env=env,
            timeout_s=compile_timeout_s,
            kernel_name=kernel_name,
            stage='COMPILATION',
            error_log_file=os.path.join(tmp_dir, 'error_log.txt'),
        )
        return True
    except Exception:
        return False


def run_compile_stage_request(
    request: EvalRequest,
    *,
    tmp_dir: Optional[str] = None,
    gpu_id: Optional[int] = None,
    error_log_file: Optional[str] = None,
    settings: Optional[EvalSettings] = None,
) -> CompileStageResult:
    settings = settings or load_eval_settings()
    total_start = time.time()
    timing = _base_timing(request, settings)
    timing['eval_pipeline'] = 'compile_stage'
    tmp_dir, tmp_dir_created = ensure_tmp_dir(tmp_dir, request.kernel_name)
    if error_log_file is None:
        error_log_file = os.path.join(tmp_dir, 'error_log.txt')

    compile_timeout_s, run_timeout_s = extract_timeout(request, settings)
    env = prepare_environment(settings, gpu_id)
    paths: Dict[str, Any] = {}

    try:
        stage_start = time.time()
        paths = _prepare_online_eval_files(request, tmp_dir=tmp_dir, settings=settings)
        timing['prepare_code'] = time.time() - stage_start
        timing['candidate_module_name'] = paths['module_name']
        timing['candidate_build_directory'] = paths['build_directory']

        stage_start = time.time()
        run_python_script(
            paths['hip_comp_file'],
            env=env,
            timeout_s=compile_timeout_s,
            kernel_name=request.kernel_name,
            stage='COMPILATION',
            error_log_file=error_log_file,
        )
        timing['compilation'] = time.time() - stage_start

        if settings.enable_ref_compile_cache and request.pytorch_functional_code:
            cache = ReferenceCache(settings.cache_root)
            compile_key, _, _ = build_reference_keys(request, settings=settings, gpu_id=None)
            compile_lookup_start = time.time()
            compile_artifact = prepare_reference_compile_artifact(
                request,
                tmp_dir=tmp_dir,
                settings=settings,
                cache=cache,
                compile_key=compile_key,
            )
            timing['reference_compile_cache_lookup'] = time.time() - compile_lookup_start
            timing['reference_compile_cache_key'] = compile_key.cache_id
            timing['reference_compile_cache_hit'] = bool(compile_artifact.get('cache_hit'))
            timing['reference_compile_artifact_scope'] = 'persistent' if compile_artifact.get('persistent') else 'ephemeral'
            timing['reference_compile_module_name'] = compile_artifact['module_name']
            if not compile_artifact.get('cache_hit'):
                stage_start = time.time()
                compile_script = prepare_reference_compile_script(
                    request,
                    tmp_dir=tmp_dir,
                    compile_artifact=compile_artifact,
                )
                try:
                    run_python_script(
                        compile_script,
                        env=env,
                        timeout_s=run_timeout_s,
                        kernel_name=request.kernel_name,
                        stage='REF_COMPILE_RUN',
                        error_log_file=error_log_file,
                    )
                except Exception as exc:
                    timing['reference_compile_build_s'] = time.time() - stage_start
                    timing['reference_compile_build'] = timing['reference_compile_build_s']
                    reason, detail = summarize_failure_exception('REF_COMPILE_RUN', exc)
                    failure = _build_failure_result(
                        total_start=total_start,
                        timing=timing,
                        stage='REF_COMPILE_RUN',
                        reason=reason,
                        detail=detail,
                        compile_ok=False,
                        run_ok=False,
                        match_ok=False,
                    )
                    return CompileStageResult(result=failure, artifact=paths, tmp_dir_created=tmp_dir_created)
                timing['reference_compile_build_s'] = time.time() - stage_start
                timing['reference_compile_build'] = timing['reference_compile_build_s']
                cache.store_compile_artifact(
                    compile_key,
                    {
                        **cache_meta_base(request, settings, gpu_id=None, driver_kind='functional'),
                        'compile_key': asdict(compile_key),
                        'compiler_identity': compile_key.compiler_identity,
                        'module_name': compile_artifact['module_name'],
                        'source_dir': compile_artifact['source_dir'],
                        'source_path': compile_artifact['source_path'],
                        'build_directory': compile_artifact['build_directory'],
                    },
                )
        else:
            timing['reference_compile_build_s'] = 0.0
            timing['reference_compile_build'] = 0.0

        timing['total'] = time.time() - total_start
        result = EvalRunResult(
            compile_ok=True,
            run_ok=False,
            match_ok=False,
            speedup=0.0,
            timing=timing,
        )
        return CompileStageResult(result=result, artifact=paths, tmp_dir_created=tmp_dir_created)
    except Exception as exc:
        timing['compilation'] = timing.get('compilation', 0.0)
        reason, detail = summarize_failure_exception('COMPILATION', exc)
        failure = _build_failure_result(
            total_start=total_start,
            timing=timing,
            stage='COMPILATION',
            reason=reason,
            detail=detail,
            compile_ok=False,
            run_ok=False,
            match_ok=False,
        )
        return CompileStageResult(result=failure, artifact=paths, tmp_dir_created=tmp_dir_created)


def run_compile_request(
    request: EvalRequest,
    *,
    tmp_dir: Optional[str] = None,
    gpu_id: Optional[int] = None,
    error_log_file: Optional[str] = None,
    settings: Optional[EvalSettings] = None,
) -> EvalRunResult:
    return run_compile_stage_request(
        request,
        tmp_dir=tmp_dir,
        gpu_id=gpu_id,
        error_log_file=error_log_file,
        settings=settings,
    ).result


def _prepare_online_eval_files(
    request: EvalRequest,
    *,
    tmp_dir: str,
    settings: EvalSettings,
    module_name: Optional[str] = None,
    build_directory: Optional[str] = None,
) -> Dict[str, str]:
    hip_dir = os.path.join(tmp_dir, 'hip')
    result_dir = os.path.join(tmp_dir, 'result')
    module_name = module_name or _candidate_module_name(request)
    build_directory = build_directory or _candidate_build_directory(tmp_dir)
    os.makedirs(hip_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(build_directory, exist_ok=True)

    hip_file = os.path.join(hip_dir, f'hip_{request.kernel_name}.hip')
    hip_comp_file = os.path.join(tmp_dir, f'hip_comp_{request.kernel_name}.py')
    candidate_script = os.path.join(tmp_dir, f'py_func_{request.kernel_name}.py')
    candidate_result_file = result_path(result_dir, 'py_func_', request.kernel_name)

    write_text_file(hip_file, request.hip_code)
    write_text_file(
        hip_comp_file,
        build_candidate_compile_script(
            request.kernel_name,
            hip_dir,
            f'hip_{request.kernel_name}.hip',
            module_name=module_name,
            build_directory=build_directory,
        ),
    )
    write_text_file(
        candidate_script,
        construct_pytorch_functional_unittest(
            request.pytorch_functional_code,
            request.kernel_name,
            hip_dir,
            f'hip_{request.kernel_name}.hip',
            result_dir,
            prefix='py_func_',
            perf_iterations=settings.perf_iterations,
            module_name=module_name,
            build_directory=build_directory,
        ),
    )
    return {
        'tmp_dir': tmp_dir,
        'hip_dir': hip_dir,
        'hip_file': hip_file,
        'result_dir': result_dir,
        'hip_comp_file': hip_comp_file,
        'candidate_script': candidate_script,
        'candidate_result_file': candidate_result_file,
        'module_name': module_name,
        'build_directory': build_directory,
    }


def _prepare_reference_eval_files(
    request: EvalRequest,
    *,
    tmp_dir: str,
    settings: EvalSettings,
    compile_artifact: Dict[str, Any],
) -> Dict[str, str]:
    result_dir = os.path.join(tmp_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)

    reference_combined_script = os.path.join(tmp_dir, f'py_func_ref_{request.kernel_name}.py')
    reference_combined_file = result_path(result_dir, 'py_func_ref_', request.kernel_name)
    write_text_file(
        reference_combined_script,
        construct_pytorch_functional_unittest(
            request.pytorch_functional_code,
            request.kernel_name,
            compile_artifact['source_dir'],
            os.path.basename(compile_artifact['source_path']),
            result_dir,
            prefix='py_func_ref_',
            perf_iterations=settings.perf_iterations,
            module_name=compile_artifact['module_name'],
            build_directory=compile_artifact['build_directory'],
        ),
    )
    ref_golden_script, ref_perf_script, reference_golden_file, reference_perf_file = prepare_reference_only_files(
        request,
        tmp_dir=tmp_dir,
        settings=settings,
        compile_artifact=compile_artifact,
    )
    return {
        'reference_combined_script': reference_combined_script,
        'reference_combined_file': reference_combined_file,
        'ref_golden_script': ref_golden_script,
        'ref_perf_script': ref_perf_script,
        'reference_golden_file': reference_golden_file,
        'reference_perf_file': reference_perf_file,
    }


def run_runtime_stage_request(
    request: EvalRequest,
    *,
    compile_artifact: Dict[str, Any],
    compile_timing: Optional[Dict[str, Any]] = None,
    tmp_dir_created: bool = False,
    gpu_id: Optional[int] = None,
    error_log_file: Optional[str] = None,
    settings: Optional[EvalSettings] = None,
) -> EvalRunResult:
    settings = settings or load_eval_settings()
    timing = _base_timing(request, settings)
    timing.update(compile_timing or {})
    timing['eval_pipeline'] = 'runtime_stage'
    compile_stage_total = float((compile_timing or {}).get('total') or 0.0)
    runtime_total_start = time.time()
    tmp_dir = str(compile_artifact.get('tmp_dir') or '')
    if not tmp_dir:
        tmp_dir, tmp_dir_created = ensure_tmp_dir(None, request.kernel_name)
        compile_artifact = _prepare_online_eval_files(request, tmp_dir=tmp_dir, settings=settings)
    if error_log_file is None:
        error_log_file = os.path.join(tmp_dir, 'error_log.txt')

    _, run_timeout_s = extract_timeout(request, settings)
    env = prepare_environment(settings, gpu_id)
    success = False
    result_dir = str(compile_artifact.get('result_dir') or os.path.join(tmp_dir, 'result'))
    cache = ReferenceCache(settings.cache_root)

    try:
        if not request.pytorch_functional_code:
            raise ValueError('pytorch_functional_code is required for HIP evaluation')

        stage_start = time.time()
        try:
            run_python_script(
                str(compile_artifact['candidate_script']),
                env=env,
                timeout_s=run_timeout_s,
                kernel_name=request.kernel_name,
                stage='TEST_RUN',
                error_log_file=error_log_file,
            )
        except Exception as exc:
            timing['test_run'] = time.time() - stage_start
            timing['runtime_stage_total'] = time.time() - runtime_total_start
            reason, detail = summarize_failure_exception('TEST_RUN', exc)
            failure = _build_failure_result(
                total_start=runtime_total_start,
                timing=timing,
                stage='TEST_RUN',
                reason=reason,
                detail=detail,
                compile_ok=True,
                run_ok=False,
                match_ok=False,
            )
            failure.timing['total'] = compile_stage_total + failure.timing['total']
            return failure
        timing['test_run'] = time.time() - stage_start

        candidate_result = torch.load(str(compile_artifact['candidate_result_file']), map_location='cpu')
        candidate_golden, candidate_perf = extract_golden_and_perf(candidate_result)
        if candidate_perf is None:
            raise RuntimeError("Candidate perf payload missing 'perf'")

        compile_key, golden_key, perf_key = build_reference_keys(request, settings=settings, gpu_id=gpu_id)
        timing['reference_identity'] = golden_key.logical_kernel_name
        timing['reference_compile_cache_key'] = compile_key.cache_id
        timing['reference_golden_cache_key'] = golden_key.cache_id
        timing['reference_perf_cache_key'] = perf_key.cache_id
        timing.setdefault('reference_compile_build_s', 0.0)
        timing.setdefault('reference_compile_build', 0.0)
        timing['reference_golden_build_s'] = 0.0
        timing['reference_golden_build'] = 0.0
        timing['reference_perf_build_s'] = 0.0
        timing['reference_perf_build'] = 0.0
        reference_golden = None
        reference_perf_ms = None
        reference_paths: Dict[str, str] = {}

        if settings.enable_ref_golden_cache:
            cache_lookup_start = time.time()
            cached_golden = cache.load_golden(golden_key)
            timing['reference_golden_cache_lookup'] = time.time() - cache_lookup_start
            if cached_golden is not None:
                reference_golden, _ = cached_golden
                timing['reference_golden_cache_hit'] = True

        if settings.enable_ref_perf_cache:
            perf_lookup_start = time.time()
            cached_perf = cache.load_perf(perf_key, ttl_s=settings.ref_perf_cache_ttl_s)
            timing['reference_perf_cache_lookup'] = time.time() - perf_lookup_start
            if cached_perf is not None:
                reference_perf_ms = float(cached_perf['reference_perf_ms'])
                timing['reference_perf_cache_hit'] = True

        need_reference_golden = reference_golden is None
        need_reference_perf = reference_perf_ms is None

        if need_reference_golden or need_reference_perf:
            compile_lookup_start = time.time()
            compile_artifact_ref = prepare_reference_compile_artifact(
                request,
                tmp_dir=tmp_dir,
                settings=settings,
                cache=cache,
                compile_key=compile_key,
            )
            timing['reference_compile_cache_lookup'] = time.time() - compile_lookup_start
            timing['reference_compile_cache_hit'] = bool(compile_artifact_ref.get('cache_hit'))
            timing['reference_compile_artifact_scope'] = 'persistent' if compile_artifact_ref.get('persistent') else 'ephemeral'
            timing['reference_compile_module_name'] = compile_artifact_ref['module_name']
            if not compile_artifact_ref.get('cache_hit'):
                stage_start = time.time()
                compile_script = prepare_reference_compile_script(
                    request,
                    tmp_dir=tmp_dir,
                    compile_artifact=compile_artifact_ref,
                )
                try:
                    run_python_script(
                        compile_script,
                        env=env,
                        timeout_s=run_timeout_s,
                        kernel_name=request.kernel_name,
                        stage='REF_COMPILE_RUN',
                        error_log_file=error_log_file,
                    )
                except Exception as exc:
                    timing['reference_compile_build_s'] = time.time() - stage_start
                    timing['reference_compile_build'] = timing['reference_compile_build_s']
                    timing['runtime_stage_total'] = time.time() - runtime_total_start
                    reason, detail = summarize_failure_exception('REF_COMPILE_RUN', exc)
                    failure = _build_failure_result(
                        total_start=runtime_total_start,
                        timing=timing,
                        stage='REF_COMPILE_RUN',
                        reason=reason,
                        detail=detail,
                        compile_ok=False,
                        run_ok=False,
                        match_ok=False,
                    )
                    failure.timing['total'] = compile_stage_total + failure.timing['total']
                    return failure
                timing['reference_compile_build_s'] = time.time() - stage_start
                timing['reference_compile_build'] = timing['reference_compile_build_s']
                if settings.enable_ref_compile_cache:
                    cache.store_compile_artifact(
                        compile_key,
                        {
                            **cache_meta_base(request, settings, gpu_id=gpu_id, driver_kind='functional'),
                            'compile_key': asdict(compile_key),
                            'compiler_identity': compile_key.compiler_identity,
                            'module_name': compile_artifact_ref['module_name'],
                            'source_dir': compile_artifact_ref['source_dir'],
                            'source_path': compile_artifact_ref['source_path'],
                            'build_directory': compile_artifact_ref['build_directory'],
                        },
                    )
            reference_paths = _prepare_reference_eval_files(
                request,
                tmp_dir=tmp_dir,
                settings=settings,
                compile_artifact=compile_artifact_ref,
            )

        if need_reference_golden and need_reference_perf:
            stage_start = time.time()
            try:
                run_python_script(
                    reference_paths['reference_combined_script'],
                    env=env,
                    timeout_s=run_timeout_s,
                    kernel_name=request.kernel_name,
                    stage='REF_RUN',
                    error_log_file=error_log_file,
                )
            except Exception as exc:
                combined_elapsed = time.time() - stage_start
                timing['reference_golden_build_s'] = combined_elapsed
                timing['reference_golden_build'] = combined_elapsed
                timing['reference_perf_build_s'] = combined_elapsed
                timing['reference_perf_build'] = combined_elapsed
                timing['ref_run'] = combined_elapsed
                timing['runtime_stage_total'] = time.time() - runtime_total_start
                reason, detail = summarize_failure_exception('REF_RUN', exc)
                failure = _build_failure_result(
                    total_start=runtime_total_start,
                    timing=timing,
                    stage='REF_RUN',
                    reason=reason,
                    detail=detail,
                    compile_ok=False,
                    run_ok=False,
                    match_ok=False,
                )
                failure.timing['total'] = compile_stage_total + failure.timing['total']
                return failure
            combined_elapsed = time.time() - stage_start
            timing['reference_golden_build_s'] = combined_elapsed
            timing['reference_golden_build'] = combined_elapsed
            timing['reference_perf_build_s'] = combined_elapsed
            timing['reference_perf_build'] = combined_elapsed
            timing['ref_run'] = timing.get('ref_run', 0.0) + combined_elapsed

            combined_payload = torch.load(reference_paths['reference_combined_file'], map_location='cpu')
            reference_golden, reference_perf_ms = extract_golden_and_perf(combined_payload)
            if settings.enable_ref_golden_cache:
                cache.store_golden(
                    golden_key,
                    to_cpu_obj(reference_golden) if settings.cache_golden_on_cpu else reference_golden,
                    {
                        **cache_meta_base(request, settings, gpu_id=gpu_id, driver_kind='functional'),
                        'compile_key': asdict(compile_key),
                        'golden_key': asdict(golden_key),
                    },
                )
            if settings.enable_ref_perf_cache and reference_perf_ms is not None:
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
        elif need_reference_golden:
            stage_start = time.time()
            try:
                run_python_script(
                    reference_paths['ref_golden_script'],
                    env=env,
                    timeout_s=run_timeout_s,
                    kernel_name=request.kernel_name,
                    stage='REF_GOLDEN_RUN',
                    error_log_file=error_log_file,
                )
            except Exception as exc:
                timing['reference_golden_build_s'] = time.time() - stage_start
                timing['reference_golden_build'] = timing['reference_golden_build_s']
                timing['ref_run'] = time.time() - stage_start
                timing['runtime_stage_total'] = time.time() - runtime_total_start
                reason, detail = summarize_failure_exception('REF_GOLDEN_RUN', exc)
                failure = _build_failure_result(
                    total_start=runtime_total_start,
                    timing=timing,
                    stage='REF_GOLDEN_RUN',
                    reason=reason,
                    detail=detail,
                    compile_ok=False,
                    run_ok=False,
                    match_ok=False,
                )
                failure.timing['total'] = compile_stage_total + failure.timing['total']
                return failure
            timing['reference_golden_build_s'] = time.time() - stage_start
            timing['reference_golden_build'] = timing['reference_golden_build_s']
            timing['ref_run'] = timing.get('ref_run', 0.0) + timing['reference_golden_build_s']

            ref_golden_payload = torch.load(reference_paths['reference_golden_file'], map_location='cpu')
            reference_golden, _ = extract_golden_and_perf(ref_golden_payload)
            if settings.enable_ref_golden_cache:
                cache.store_golden(
                    golden_key,
                    to_cpu_obj(reference_golden) if settings.cache_golden_on_cpu else reference_golden,
                    {
                        **cache_meta_base(request, settings, gpu_id=gpu_id, driver_kind='functional'),
                        'compile_key': asdict(compile_key),
                        'golden_key': asdict(golden_key),
                    },
                )

        compare_start = time.time()
        compare_reference = reference_golden
        compare_candidate = to_cpu_obj(candidate_golden) if timing['reference_golden_cache_hit'] else candidate_golden
        if not compare_results(compare_reference, compare_candidate, rtol=float(request.rtol or 1e-4), atol=float(request.atol or 1e-5)):
            timing['compare_cleanup'] = time.time() - compare_start
            timing['runtime_stage_total'] = time.time() - runtime_total_start
            log_error(
                error_log_file,
                request.kernel_name,
                'RESULT_MISMATCH',
                f'[MISMATCH] {request.kernel_name} results differ (rtol={request.rtol}, atol={request.atol})',
            )
            failure = _build_failure_result(
                total_start=runtime_total_start,
                timing=timing,
                stage='RESULT_MISMATCH',
                reason='result mismatch',
                detail=f'rtol={request.rtol}, atol={request.atol}',
                compile_ok=True,
                run_ok=True,
                match_ok=False,
            )
            failure.timing['total'] = compile_stage_total + failure.timing['total']
            return failure

        if need_reference_perf and reference_perf_ms is None:
            stage_start = time.time()
            try:
                run_python_script(
                    reference_paths['ref_perf_script'],
                    env=env,
                    timeout_s=run_timeout_s,
                    kernel_name=request.kernel_name,
                    stage='REF_PERF_RUN',
                    error_log_file=error_log_file,
                )
            except Exception as exc:
                timing['reference_perf_build_s'] = time.time() - stage_start
                timing['reference_perf_build'] = timing['reference_perf_build_s']
                timing['ref_run'] = timing.get('ref_run', 0.0) + timing['reference_perf_build_s']
                timing['runtime_stage_total'] = time.time() - runtime_total_start
                reason, detail = summarize_failure_exception('REF_PERF_RUN', exc)
                failure = _build_failure_result(
                    total_start=runtime_total_start,
                    timing=timing,
                    stage='REF_PERF_RUN',
                    reason=reason,
                    detail=detail,
                    compile_ok=False,
                    run_ok=False,
                    match_ok=False,
                )
                failure.timing['total'] = compile_stage_total + failure.timing['total']
                return failure
            timing['reference_perf_build_s'] = time.time() - stage_start
            timing['reference_perf_build'] = timing['reference_perf_build_s']
            timing['ref_run'] = timing.get('ref_run', 0.0) + timing['reference_perf_build_s']

            perf_payload = torch.load(reference_paths['reference_perf_file'], map_location='cpu')
            _, reference_perf_ms = extract_golden_and_perf(perf_payload)
            if reference_perf_ms is None:
                raise RuntimeError("Reference perf payload missing 'perf'")
            if settings.enable_ref_perf_cache:
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

        if candidate_perf == 0:
            raise RuntimeError('Candidate perf is zero; cannot compute speedup')
        perf_val = float(reference_perf_ms) / float(candidate_perf)
        timing['candidate_perf_ms'] = float(candidate_perf)
        timing['reference_perf_ms'] = float(reference_perf_ms)
        perf_val = _confirm_speedup_if_needed(
            request,
            tmp_dir=tmp_dir,
            env=env,
            run_timeout_s=run_timeout_s,
            error_log_file=error_log_file,
            settings=settings,
            timing=timing,
            first_pass_speedup=perf_val,
            module_name=str(compile_artifact.get('module_name') or _candidate_module_name(request)),
            build_directory=str(compile_artifact.get('build_directory') or _candidate_build_directory(tmp_dir)),
        )

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        timing['compare_cleanup'] = time.time() - compare_start
        timing['runtime_stage_total'] = time.time() - runtime_total_start
        timing['total'] = compile_stage_total + timing['runtime_stage_total']
        success = True
        return EvalRunResult(
            compile_ok=True,
            run_ok=True,
            match_ok=True,
            speedup=perf_val,
            timing=timing,
        )
    except Exception as exc:
        msg = f'[EXCEPTION] {request.kernel_name} error: {exc}'
        log_error(error_log_file, request.kernel_name, 'EXCEPTION', msg, traceback.format_exc())
        reason, detail = summarize_failure_exception('EXCEPTION', exc)
        failure = _build_failure_result(
            total_start=runtime_total_start,
            timing=timing,
            stage='EXCEPTION',
            reason=reason,
            detail=detail,
            compile_ok=False,
            run_ok=False,
            match_ok=False,
        )
        failure.timing['total'] = compile_stage_total + failure.timing['total']
        return failure
    finally:
        clear_pts(result_dir)
        should_cleanup = success and settings.cleanup_tmp_on_success
        if not success and not settings.retain_tmp_on_failure:
            should_cleanup = True
        if tmp_dir_created and should_cleanup and os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


def run_eval_request(
    request: EvalRequest,
    *,
    tmp_dir: Optional[str] = None,
    gpu_id: Optional[int] = None,
    error_log_file: Optional[str] = None,
    settings: Optional[EvalSettings] = None,
) -> EvalRunResult:
    settings = settings or load_eval_settings()
    timing = _base_timing(request, settings)
    total_start = time.time()
    tmp_dir, tmp_dir_created = ensure_tmp_dir(tmp_dir, request.kernel_name)
    if error_log_file is None:
        error_log_file = os.path.join(tmp_dir, 'error_log.txt')

    compile_timeout_s, run_timeout_s = extract_timeout(request, settings)
    env = prepare_environment(settings, gpu_id)
    success = False
    result_dir = os.path.join(tmp_dir, 'result')
    cache = ReferenceCache(settings.cache_root)

    try:
        if not request.pytorch_functional_code:
            raise ValueError('pytorch_functional_code is required for HIP evaluation')

        stage_start = time.time()
        paths = _prepare_online_eval_files(request, tmp_dir=tmp_dir, settings=settings)
        timing['prepare_code'] = time.time() - stage_start

        stage_start = time.time()
        try:
            run_python_script(
                paths['hip_comp_file'],
                env=env,
                timeout_s=compile_timeout_s,
                kernel_name=request.kernel_name,
                stage='COMPILATION',
                error_log_file=error_log_file,
            )
        except Exception as exc:
            timing['compilation'] = time.time() - stage_start
            reason, detail = summarize_failure_exception('COMPILATION', exc)
            return _build_failure_result(
                total_start=total_start,
                timing=timing,
                stage='COMPILATION',
                reason=reason,
                detail=detail,
                compile_ok=False,
                run_ok=False,
                match_ok=False,
            )
        timing['compilation'] = time.time() - stage_start

        stage_start = time.time()
        try:
            run_python_script(
                paths['candidate_script'],
                env=env,
                timeout_s=run_timeout_s,
                kernel_name=request.kernel_name,
                stage='TEST_RUN',
                error_log_file=error_log_file,
            )
        except Exception as exc:
            timing['test_run'] = time.time() - stage_start
            reason, detail = summarize_failure_exception('TEST_RUN', exc)
            return _build_failure_result(
                total_start=total_start,
                timing=timing,
                stage='TEST_RUN',
                reason=reason,
                detail=detail,
                compile_ok=True,
                run_ok=False,
                match_ok=False,
            )
        timing['test_run'] = time.time() - stage_start

        candidate_result = torch.load(paths['candidate_result_file'], map_location='cpu')
        candidate_golden, candidate_perf = extract_golden_and_perf(candidate_result)
        if candidate_perf is None:
            raise RuntimeError("Candidate perf payload missing 'perf'")

        compile_key, golden_key, perf_key = build_reference_keys(request, settings=settings, gpu_id=gpu_id)
        timing['reference_identity'] = golden_key.logical_kernel_name
        timing['reference_compile_cache_key'] = compile_key.cache_id
        timing['reference_golden_cache_key'] = golden_key.cache_id
        timing['reference_perf_cache_key'] = perf_key.cache_id
        timing['reference_compile_build_s'] = 0.0
        timing['reference_compile_build'] = 0.0
        timing['reference_golden_build_s'] = 0.0
        timing['reference_golden_build'] = 0.0
        timing['reference_perf_build_s'] = 0.0
        timing['reference_perf_build'] = 0.0
        reference_golden = None
        reference_perf_ms = None
        reference_paths: Dict[str, str] = {}

        if settings.enable_ref_golden_cache:
            cache_lookup_start = time.time()
            cached_golden = cache.load_golden(golden_key)
            timing['reference_golden_cache_lookup'] = time.time() - cache_lookup_start
            if cached_golden is not None:
                reference_golden, _ = cached_golden
                timing['reference_golden_cache_hit'] = True

        if settings.enable_ref_perf_cache:
            perf_lookup_start = time.time()
            cached_perf = cache.load_perf(perf_key, ttl_s=settings.ref_perf_cache_ttl_s)
            timing['reference_perf_cache_lookup'] = time.time() - perf_lookup_start
            if cached_perf is not None:
                reference_perf_ms = float(cached_perf['reference_perf_ms'])
                timing['reference_perf_cache_hit'] = True

        need_reference_golden = reference_golden is None
        need_reference_perf = reference_perf_ms is None

        if need_reference_golden or need_reference_perf:
            compile_lookup_start = time.time()
            compile_artifact = prepare_reference_compile_artifact(
                request,
                tmp_dir=tmp_dir,
                settings=settings,
                cache=cache,
                compile_key=compile_key,
            )
            timing['reference_compile_cache_lookup'] = time.time() - compile_lookup_start
            timing['reference_compile_cache_hit'] = bool(compile_artifact.get('cache_hit'))
            timing['reference_compile_artifact_scope'] = 'persistent' if compile_artifact.get('persistent') else 'ephemeral'
            timing['reference_compile_module_name'] = compile_artifact['module_name']
            if not compile_artifact.get('cache_hit'):
                stage_start = time.time()
                compile_script = prepare_reference_compile_script(
                    request,
                    tmp_dir=tmp_dir,
                    compile_artifact=compile_artifact,
                )
                try:
                    run_python_script(
                        compile_script,
                        env=env,
                        timeout_s=run_timeout_s,
                        kernel_name=request.kernel_name,
                        stage='REF_COMPILE_RUN',
                        error_log_file=error_log_file,
                    )
                except Exception as exc:
                    timing['reference_compile_build_s'] = time.time() - stage_start
                    timing['reference_compile_build'] = timing['reference_compile_build_s']
                    reason, detail = summarize_failure_exception('REF_COMPILE_RUN', exc)
                    return _build_failure_result(
                        total_start=total_start,
                        timing=timing,
                        stage='REF_COMPILE_RUN',
                        reason=reason,
                        detail=detail,
                        compile_ok=False,
                        run_ok=False,
                        match_ok=False,
                    )
                timing['reference_compile_build_s'] = time.time() - stage_start
                timing['reference_compile_build'] = timing['reference_compile_build_s']
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
            reference_paths = _prepare_reference_eval_files(
                request,
                tmp_dir=tmp_dir,
                settings=settings,
                compile_artifact=compile_artifact,
            )

        if need_reference_golden and need_reference_perf:
            stage_start = time.time()
            try:
                run_python_script(
                    reference_paths['reference_combined_script'],
                    env=env,
                    timeout_s=run_timeout_s,
                    kernel_name=request.kernel_name,
                    stage='REF_RUN',
                    error_log_file=error_log_file,
                )
            except Exception as exc:
                timing['reference_golden_build_s'] = time.time() - stage_start
                timing['reference_golden_build'] = timing['reference_golden_build_s']
                timing['reference_perf_build_s'] = timing['reference_golden_build_s']
                timing['reference_perf_build'] = timing['reference_perf_build_s']
                timing['ref_run'] = time.time() - stage_start
                reason, detail = summarize_failure_exception('REF_RUN', exc)
                return _build_failure_result(
                    total_start=total_start,
                    timing=timing,
                    stage='REF_RUN',
                    reason=reason,
                    detail=detail,
                    compile_ok=False,
                    run_ok=False,
                    match_ok=False,
                )
            combined_elapsed = time.time() - stage_start
            timing['reference_golden_build_s'] = combined_elapsed
            timing['reference_golden_build'] = combined_elapsed
            timing['reference_perf_build_s'] = combined_elapsed
            timing['reference_perf_build'] = combined_elapsed
            timing['ref_run'] = timing.get('ref_run', 0.0) + combined_elapsed

            combined_payload = torch.load(reference_paths['reference_combined_file'], map_location='cpu')
            reference_golden, reference_perf_ms = extract_golden_and_perf(combined_payload)
            if settings.enable_ref_golden_cache:
                cache.store_golden(
                    golden_key,
                    to_cpu_obj(reference_golden) if settings.cache_golden_on_cpu else reference_golden,
                    {
                        **cache_meta_base(request, settings, gpu_id=gpu_id, driver_kind='functional'),
                        'compile_key': asdict(compile_key),
                        'golden_key': asdict(golden_key),
                    },
                )
            if settings.enable_ref_perf_cache and reference_perf_ms is not None:
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
        elif need_reference_golden:
            stage_start = time.time()
            try:
                run_python_script(
                    reference_paths['ref_golden_script'],
                    env=env,
                    timeout_s=run_timeout_s,
                    kernel_name=request.kernel_name,
                    stage='REF_GOLDEN_RUN',
                    error_log_file=error_log_file,
                )
            except Exception as exc:
                timing['reference_golden_build_s'] = time.time() - stage_start
                timing['reference_golden_build'] = timing['reference_golden_build_s']
                timing['ref_run'] = time.time() - stage_start
                reason, detail = summarize_failure_exception('REF_GOLDEN_RUN', exc)
                return _build_failure_result(
                    total_start=total_start,
                    timing=timing,
                    stage='REF_GOLDEN_RUN',
                    reason=reason,
                    detail=detail,
                    compile_ok=False,
                    run_ok=False,
                    match_ok=False,
                )
            timing['reference_golden_build_s'] = time.time() - stage_start
            timing['reference_golden_build'] = timing['reference_golden_build_s']
            timing['ref_run'] = timing.get('ref_run', 0.0) + timing['reference_golden_build_s']

            ref_golden_payload = torch.load(reference_paths['reference_golden_file'], map_location='cpu')
            reference_golden, _ = extract_golden_and_perf(ref_golden_payload)
            if settings.enable_ref_golden_cache:
                cache.store_golden(
                    golden_key,
                    to_cpu_obj(reference_golden) if settings.cache_golden_on_cpu else reference_golden,
                    {
                        **cache_meta_base(request, settings, gpu_id=gpu_id, driver_kind='functional'),
                        'compile_key': asdict(compile_key),
                        'golden_key': asdict(golden_key),
                    },
                )

        compare_start = time.time()
        compare_reference = reference_golden
        compare_candidate = to_cpu_obj(candidate_golden) if timing['reference_golden_cache_hit'] else candidate_golden
        if not compare_results(compare_reference, compare_candidate, rtol=float(request.rtol or 1e-4), atol=float(request.atol or 1e-5)):
            timing['compare_cleanup'] = time.time() - compare_start
            log_error(
                error_log_file,
                request.kernel_name,
                'RESULT_MISMATCH',
                f'[MISMATCH] {request.kernel_name} results differ (rtol={request.rtol}, atol={request.atol})',
            )
            return _build_failure_result(
                total_start=total_start,
                timing=timing,
                stage='RESULT_MISMATCH',
                reason='result mismatch',
                detail=f'rtol={request.rtol}, atol={request.atol}',
                compile_ok=True,
                run_ok=True,
                match_ok=False,
            )

        if need_reference_perf and reference_perf_ms is None:
            stage_start = time.time()
            try:
                run_python_script(
                    reference_paths['ref_perf_script'],
                    env=env,
                    timeout_s=run_timeout_s,
                    kernel_name=request.kernel_name,
                    stage='REF_PERF_RUN',
                    error_log_file=error_log_file,
                )
            except Exception as exc:
                timing['reference_perf_build_s'] = time.time() - stage_start
                timing['reference_perf_build'] = timing['reference_perf_build_s']
                timing['ref_run'] = timing.get('ref_run', 0.0) + timing['reference_perf_build_s']
                reason, detail = summarize_failure_exception('REF_PERF_RUN', exc)
                return _build_failure_result(
                    total_start=total_start,
                    timing=timing,
                    stage='REF_PERF_RUN',
                    reason=reason,
                    detail=detail,
                    compile_ok=False,
                    run_ok=False,
                    match_ok=False,
                )
            timing['reference_perf_build_s'] = time.time() - stage_start
            timing['reference_perf_build'] = timing['reference_perf_build_s']
            timing['ref_run'] = timing.get('ref_run', 0.0) + timing['reference_perf_build_s']

            perf_payload = torch.load(reference_paths['reference_perf_file'], map_location='cpu')
            _, reference_perf_ms = extract_golden_and_perf(perf_payload)
            if reference_perf_ms is None:
                raise RuntimeError("Reference perf payload missing 'perf'")
            if settings.enable_ref_perf_cache:
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

        if candidate_perf == 0:
            raise RuntimeError('Candidate perf is zero; cannot compute speedup')
        perf_val = float(reference_perf_ms) / float(candidate_perf)
        timing['candidate_perf_ms'] = float(candidate_perf)
        timing['reference_perf_ms'] = float(reference_perf_ms)
        perf_val = _confirm_speedup_if_needed(
            request,
            tmp_dir=tmp_dir,
            env=env,
            run_timeout_s=run_timeout_s,
            error_log_file=error_log_file,
            settings=settings,
            timing=timing,
            first_pass_speedup=perf_val,
            module_name=paths['module_name'],
            build_directory=paths['build_directory'],
        )

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        timing['compare_cleanup'] = time.time() - compare_start
        timing['total'] = time.time() - total_start
        success = True
        return EvalRunResult(
            compile_ok=True,
            run_ok=True,
            match_ok=True,
            speedup=perf_val,
            timing=timing,
        )
    except Exception as exc:
        msg = f'[EXCEPTION] {request.kernel_name} error: {exc}'
        log_error(error_log_file, request.kernel_name, 'EXCEPTION', msg, traceback.format_exc())
        reason, detail = summarize_failure_exception('EXCEPTION', exc)
        return _build_failure_result(
            total_start=total_start,
            timing=timing,
            stage='EXCEPTION',
            reason=reason,
            detail=detail,
            compile_ok=False,
            run_ok=False,
            match_ok=False,
        )
    finally:
        clear_pts(result_dir)
        should_cleanup = success and settings.cleanup_tmp_on_success
        if not success and not settings.retain_tmp_on_failure:
            should_cleanup = True
        if tmp_dir_created and should_cleanup and os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


def evaluate_requests_parallel(
    requests: List[EvalRequest],
    *,
    max_workers: Optional[int] = None,
    base_tmp_dir: str = 'hip_eval_parallel',
    gpu_ids: Optional[List[int]] = None,
    error_log_dir: Optional[str] = None,
    settings: Optional[EvalSettings] = None,
):
    return _evaluate_requests_parallel(
        requests,
        max_workers=max_workers,
        base_tmp_dir=base_tmp_dir,
        gpu_ids=gpu_ids,
        error_log_dir=error_log_dir,
        settings=settings,
    )


def prewarm_reference_artifacts(
    request: EvalRequest,
    *,
    settings: Optional[EvalSettings] = None,
    gpu_id: Optional[int] = None,
    with_perf: bool = False,
) -> Dict[str, Any]:
    settings = settings or load_eval_settings()
    if with_perf and gpu_id is None:
        raise ValueError('Perf prewarm requires an explicit gpu_id so runtime fingerprints can match online evaluation')
    cache = ReferenceCache(settings.cache_root)
    compile_key, golden_key, perf_key = build_reference_keys(request, settings=settings, gpu_id=gpu_id)
    result: Dict[str, Any] = {
        'kernel_name': request.kernel_name,
        'compile_key': compile_key.cache_id,
        'golden_key': golden_key.cache_id,
        'perf_key': perf_key.cache_id,
        'compile_cache_hit': False,
        'golden_cache_hit': False,
        'perf_cache_hit': False,
        'perf_runtime_fingerprint': perf_key.runtime_fingerprint if with_perf else None,
        'with_perf': with_perf,
    }

    cached_compile = cache.load_compile_artifact(compile_key)
    cached_golden = cache.load_golden(golden_key)
    cached_perf = cache.load_perf(perf_key, ttl_s=settings.ref_perf_cache_ttl_s) if with_perf else None
    result['compile_cache_hit'] = cached_compile is not None
    result['golden_cache_hit'] = cached_golden is not None
    result['perf_cache_hit'] = cached_perf is not None

    needs_golden = cached_golden is None
    needs_perf = with_perf and cached_perf is None
    needs_compile = (cached_compile is None) and (needs_golden or needs_perf)
    if not needs_compile and not needs_golden and not needs_perf:
        return result

    error_log_file = os.path.join(settings.error_log_dir, f'{request.kernel_name}_prewarm_error.log')
    tmp_dir, tmp_dir_created = ensure_tmp_dir(None, f'ref_cache_{request.kernel_name}')
    materialize_reference_cache_artifacts(
        request,
        settings=EvalSettings(
            **{
                **settings.__dict__,
                'enable_ref_compile_cache': True,
                'enable_ref_golden_cache': True,
                'enable_ref_perf_cache': bool(with_perf),
            }
        ),
        gpu_id=gpu_id,
        error_log_file=error_log_file,
        build_golden=needs_golden,
        build_perf=needs_perf,
        tmp_dir=tmp_dir,
        tmp_dir_created=tmp_dir_created,
    )

    result['compile_cache_hit'] = cache.load_compile_artifact(compile_key) is not None
    result['golden_cache_hit'] = cache.load_golden(golden_key) is not None
    if with_perf:
        result['perf_cache_hit'] = cache.load_perf(perf_key, ttl_s=settings.ref_perf_cache_ttl_s) is not None
    return result
