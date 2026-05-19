from __future__ import annotations

import logging
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import EvalSettings, eval_settings_from_payload, load_eval_settings
from .logging_utils import (
    format_evaluation_summary,
    format_kernel_failure,
    format_kernel_success,
    summarize_failure_exception,
)
from .protocol import EvalRequest
from .result import EvalRunResult

logger = logging.getLogger('hip_eval_parallel')


def _is_success(result: EvalRunResult) -> bool:
    return result.compile_ok and result.run_ok and result.match_ok


def _log_kernel_result(kernel_name: str, result: EvalRunResult) -> None:
    if _is_success(result):
        logger.info(format_kernel_success(kernel_name, speedup=result.speedup, timing=result.timing))
        return
    logger.warning(
        format_kernel_failure(
            kernel_name,
            compile_ok=result.compile_ok,
            run_ok=result.run_ok,
            match_ok=result.match_ok,
            timing=result.timing,
        )
    )


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _available_cpu_ids() -> List[int]:
    if hasattr(os, "sched_getaffinity"):
        try:
            return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        except Exception:
            pass
    cpu_count = os.cpu_count() or 1
    return list(range(cpu_count))


def _partition_cpu_ids(cpu_ids: Sequence[int], bucket_count: int) -> List[List[int]]:
    if bucket_count <= 0:
        return [list(cpu_ids)]
    groups: List[List[int]] = []
    start = 0
    base, remainder = divmod(len(cpu_ids), bucket_count)
    for idx in range(bucket_count):
        group_size = base + (1 if idx < remainder else 0)
        groups.append(list(cpu_ids[start : start + group_size]))
        start += group_size
    return groups


def _build_affinity_metadata(gpu_id: Optional[int], settings_payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "assigned_gpu_id": gpu_id,
        "cpu_affinity_enabled": _env_flag("HIP_ENABLE_CPU_AFFINITY", False),
    }
    if not metadata["cpu_affinity_enabled"]:
        return metadata
    cpu_ids = _available_cpu_ids()
    gpu_ids = list(settings_payload.get("gpu_ids") or [])
    bucket_count = max(1, len(gpu_ids))
    groups = _partition_cpu_ids(cpu_ids, bucket_count)
    slot_index = gpu_ids.index(gpu_id) if gpu_id in gpu_ids else 0
    assigned_cpu_ids = groups[slot_index] if slot_index < len(groups) and groups[slot_index] else list(cpu_ids)
    metadata["cpu_affinity_slot_index"] = slot_index
    metadata["assigned_cpu_cores"] = assigned_cpu_ids
    metadata["cpu_affinity_source"] = "auto_split"
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, assigned_cpu_ids)
            metadata["cpu_affinity_applied"] = True
        except Exception as exc:
            metadata["cpu_affinity_applied"] = False
            metadata["cpu_affinity_error"] = str(exc)
    else:
        metadata["cpu_affinity_applied"] = False
        metadata["cpu_affinity_error"] = "sched_setaffinity unavailable"
    return metadata


def _evaluate_request_worker(args: Tuple[int, Dict[str, Any], str, Optional[int], str, Dict[str, Any]]) -> Tuple[int, EvalRunResult]:
    from .eval import run_eval_request

    idx, request_payload, tmp_dir, gpu_id, error_log_file, settings_payload = args
    request = EvalRequest(**request_payload)
    settings = eval_settings_from_payload(settings_payload)
    affinity_metadata = _build_affinity_metadata(gpu_id, settings_payload)
    perf_started_at = _now_iso()
    result = run_eval_request(
        request,
        tmp_dir=tmp_dir,
        gpu_id=gpu_id,
        error_log_file=error_log_file,
        settings=settings,
    )
    perf_finished_at = _now_iso()
    timing = {
        **(result.timing or {}),
        **affinity_metadata,
        "perf_started_at": perf_started_at,
        "perf_finished_at": perf_finished_at,
    }
    result = result._replace(timing=timing)
    return idx, result


def _compile_stage_worker(args: Tuple[int, Dict[str, Any], str, str, Dict[str, Any]]) -> Tuple[int, EvalRunResult, Dict[str, Any], bool]:
    from .eval import run_compile_stage_request

    idx, request_payload, tmp_dir, error_log_file, settings_payload = args
    request = EvalRequest(**request_payload)
    settings = eval_settings_from_payload(settings_payload)
    result = run_compile_stage_request(
        request,
        tmp_dir=tmp_dir,
        gpu_id=None,
        error_log_file=error_log_file,
        settings=settings,
    )
    timing = {
        **(result.result.timing or {}),
        "assigned_gpu_id": None,
        "cpu_affinity_enabled": False,
    }
    return idx, result.result._replace(timing=timing), result.artifact, result.tmp_dir_created


def _runtime_stage_worker(args: Tuple[int, Dict[str, Any], Dict[str, Any], Dict[str, Any], bool, Optional[int], str, Dict[str, Any]]) -> Tuple[int, EvalRunResult]:
    from .eval import run_runtime_stage_request

    idx, request_payload, compile_artifact, compile_timing, tmp_dir_created, gpu_id, error_log_file, settings_payload = args
    request = EvalRequest(**request_payload)
    settings = eval_settings_from_payload(settings_payload)
    affinity_metadata = _build_affinity_metadata(gpu_id, settings_payload)
    perf_started_at = _now_iso()
    result = run_runtime_stage_request(
        request,
        compile_artifact=compile_artifact,
        compile_timing=compile_timing,
        tmp_dir_created=tmp_dir_created,
        gpu_id=gpu_id,
        error_log_file=error_log_file,
        settings=settings,
    )
    perf_finished_at = _now_iso()
    timing = {
        **(result.timing or {}),
        **affinity_metadata,
        "perf_started_at": perf_started_at,
        "perf_finished_at": perf_finished_at,
    }
    return idx, result._replace(timing=timing)


def _parallel_worker_failure(exc: Exception) -> EvalRunResult:
    reason, detail = summarize_failure_exception('PARALLEL_WORKER', exc)
    return EvalRunResult.failure(
        compile_ok=False,
        run_ok=False,
        match_ok=False,
        timing={
            'exception': str(exc),
            'failure_stage': 'PARALLEL_WORKER',
            'failure_reason': reason,
            'failure_detail': detail,
        },
    )


def _evaluate_requests_two_stage(
    requests: Sequence[EvalRequest],
    *,
    compile_workers: int,
    runtime_gpu_ids: List[int],
    base_tmp_dir: str,
    error_log_dir: str,
    settings: EvalSettings,
) -> List[EvalRunResult]:
    num_tasks = len(requests)
    settings_payload = asdict(settings)
    results: List[EvalRunResult] = [EvalRunResult.failure(compile_ok=False, run_ok=False, match_ok=False) for _ in range(num_tasks)]
    request_payloads = [request.model_dump() for request in requests]
    log_paths = [os.path.join(error_log_dir, f'{request.kernel_name}_error.log') for request in requests]
    compile_args = [
        (
            idx,
            request_payloads[idx],
            os.path.join(base_tmp_dir, f'{requests[idx].kernel_name}_{idx}'),
            log_paths[idx],
            settings_payload,
        )
        for idx in range(num_tasks)
    ]
    runtime_pending: List[Tuple[int, Dict[str, Any], bool]] = []
    idle_gpus = list(runtime_gpu_ids)

    logger.info(
        'Starting two-stage HIP kernel evaluation: tasks=%s compile_workers=%s runtime_gpus=%s',
        num_tasks,
        compile_workers,
        runtime_gpu_ids,
    )

    with ProcessPoolExecutor(max_workers=compile_workers) as compile_executor, ProcessPoolExecutor(max_workers=max(1, len(runtime_gpu_ids))) as runtime_executor:
        compile_futures = {
            compile_executor.submit(_compile_stage_worker, args): args[0]
            for args in compile_args
        }
        runtime_futures: Dict[Any, Tuple[int, int]] = {}

        while compile_futures or runtime_futures or runtime_pending:
            while runtime_pending and idle_gpus:
                idx, artifact, tmp_dir_created = runtime_pending.pop(0)
                gpu_id = idle_gpus.pop(0)
                future = runtime_executor.submit(
                    _runtime_stage_worker,
                    (
                        idx,
                        request_payloads[idx],
                        artifact,
                        results[idx].timing,
                        tmp_dir_created,
                        gpu_id,
                        log_paths[idx],
                        settings_payload,
                    ),
                )
                runtime_futures[future] = (idx, gpu_id)

            active_futures = set(compile_futures) | set(runtime_futures)
            if not active_futures:
                continue

            done, _ = wait(active_futures, return_when=FIRST_COMPLETED)
            for future in done:
                if future in compile_futures:
                    idx = compile_futures.pop(future)
                    try:
                        result_idx, result, artifact, tmp_dir_created = future.result()
                        results[result_idx] = result
                        if result.compile_ok:
                            runtime_pending.append((result_idx, artifact, tmp_dir_created))
                        else:
                            _log_kernel_result(requests[result_idx].kernel_name, result)
                    except Exception as exc:
                        results[idx] = _parallel_worker_failure(exc)
                        logger.debug('Compile-stage worker exception for %s', requests[idx].kernel_name, exc_info=exc)
                        _log_kernel_result(requests[idx].kernel_name, results[idx])
                else:
                    idx, gpu_id = runtime_futures.pop(future)
                    idle_gpus.append(gpu_id)
                    try:
                        result_idx, result = future.result()
                        results[result_idx] = result
                        _log_kernel_result(requests[result_idx].kernel_name, result)
                    except Exception as exc:
                        results[idx] = _parallel_worker_failure(exc)
                        logger.debug('Runtime-stage worker exception for %s', requests[idx].kernel_name, exc_info=exc)
                        _log_kernel_result(requests[idx].kernel_name, results[idx])

    return results


def evaluate_requests_parallel(
    requests: Sequence[EvalRequest],
    *,
    max_workers: Optional[int] = None,
    base_tmp_dir: str = 'hip_eval_parallel',
    gpu_ids: Optional[List[int]] = None,
    error_log_dir: Optional[str] = None,
    settings: Optional[EvalSettings] = None,
) -> List[EvalRunResult]:
    settings = settings or load_eval_settings()
    if not requests:
        return []

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    resolved_gpu_ids = gpu_ids if gpu_ids is not None else settings.gpu_ids
    os.makedirs(base_tmp_dir, exist_ok=True)
    os.makedirs(error_log_dir or settings.error_log_dir, exist_ok=True)

    num_tasks = len(requests)
    num_gpus = len(resolved_gpu_ids) if resolved_gpu_ids else 1
    cpu_count = multiprocessing.cpu_count()
    if max_workers is None:
        max_workers = min(num_tasks, num_gpus, max(1, cpu_count // 2))

    logger.info('Starting parallel HIP kernel evaluation: tasks=%s workers=%s gpus=%s', num_tasks, max_workers, resolved_gpu_ids)

    if getattr(settings, 'enable_two_stage_batch', True) and num_tasks > 1:
        total_start = time.time()
        runtime_gpu_ids = list(resolved_gpu_ids[: max(1, min(len(resolved_gpu_ids), max_workers))]) if resolved_gpu_ids else [None]
        compile_workers = min(num_tasks, max(1, int(getattr(settings, 'compile_cpu_slots', max_workers))))
        results = _evaluate_requests_two_stage(
            requests,
            compile_workers=compile_workers,
            runtime_gpu_ids=runtime_gpu_ids,
            base_tmp_dir=base_tmp_dir,
            error_log_dir=error_log_dir or settings.error_log_dir,
            settings=settings,
        )
        total_elapsed = time.time() - total_start
        logger.info(
            format_evaluation_summary(
                'Two-stage parallel evaluation',
                [
                    {
                        'kernel_name': requests[idx].kernel_name,
                        'compile_ok': result.compile_ok,
                        'run_ok': result.run_ok,
                        'match_ok': result.match_ok,
                        'speedup': result.speedup,
                        'timing': result.timing,
                    }
                    for idx, result in enumerate(results)
                ],
                total_elapsed=total_elapsed,
            )
        )
        return results

    results: List[EvalRunResult] = [EvalRunResult.failure(compile_ok=False, run_ok=False, match_ok=False) for _ in range(num_tasks)]
    failed_count = 0
    success_count = 0
    total_start = time.time()

    task_args: List[Tuple[int, Dict[str, Any], str, Optional[int], str, Dict[str, Any]]] = []
    for idx, request in enumerate(requests):
        tmp_dir = os.path.join(base_tmp_dir, f'{request.kernel_name}_{idx}')
        gpu_id = resolved_gpu_ids[idx % len(resolved_gpu_ids)] if resolved_gpu_ids else None
        log_path = os.path.join(error_log_dir or settings.error_log_dir, f'{request.kernel_name}_error.log')
        task_args.append((idx, request.model_dump(), tmp_dir, gpu_id, log_path, asdict(settings)))

    if num_tasks == 1:
        idx, result = _evaluate_request_worker(task_args[0])
        results[idx] = result
        if _is_success(result):
            success_count = 1
        else:
            failed_count = 1
        _log_kernel_result(requests[idx].kernel_name, result)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_evaluate_request_worker, args): (args[0], requests[args[0]].kernel_name)
                for args in task_args
            }
            for future in as_completed(future_map):
                idx, kernel_name = future_map[future]
                try:
                    result_idx, result = future.result()
                    results[result_idx] = result
                    if _is_success(result):
                        success_count += 1
                    else:
                        failed_count += 1
                    _log_kernel_result(kernel_name, result)
                except Exception as exc:
                    failed_count += 1
                    reason, detail = summarize_failure_exception('PARALLEL_WORKER', exc)
                    results[idx] = EvalRunResult.failure(
                        compile_ok=False,
                        run_ok=False,
                        match_ok=False,
                        timing={
                            'exception': str(exc),
                            'failure_stage': 'PARALLEL_WORKER',
                            'failure_reason': reason,
                            'failure_detail': detail,
                        },
                    )
                    logger.debug('Parallel worker exception for %s', kernel_name, exc_info=exc)
                    _log_kernel_result(kernel_name, results[idx])

    total_elapsed = time.time() - total_start
    logger.info(
        format_evaluation_summary(
            'Parallel evaluation',
            [
                {
                    'kernel_name': requests[idx].kernel_name,
                    'compile_ok': result.compile_ok,
                    'run_ok': result.run_ok,
                    'match_ok': result.match_ok,
                    'speedup': result.speedup,
                    'timing': result.timing,
                }
                for idx, result in enumerate(results)
            ],
            total_elapsed=total_elapsed,
        )
    )
    return results
