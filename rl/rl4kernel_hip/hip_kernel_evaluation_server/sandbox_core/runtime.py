from __future__ import annotations

import datetime
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple

import torch

from .config import EvalSettings
from .protocol import EvalRequest


def compare_results(reference_result: Any, candidate_result: Any, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    if isinstance(reference_result, dict) and isinstance(candidate_result, dict):
        if set(reference_result.keys()) != set(candidate_result.keys()):
            return False
        return all(
            compare_results(reference_result[key], candidate_result[key], rtol=rtol, atol=atol)
            for key in reference_result
        )
    if isinstance(reference_result, (list, tuple)) and isinstance(candidate_result, (list, tuple)):
        if len(reference_result) != len(candidate_result):
            return False
        return all(
            compare_results(left, right, rtol=rtol, atol=atol)
            for left, right in zip(reference_result, candidate_result)
        )
    if torch.is_tensor(reference_result) and torch.is_tensor(candidate_result):
        ref_tensor = reference_result.detach().cpu()
        cand_tensor = candidate_result.detach().cpu()
        return torch.allclose(ref_tensor, cand_tensor, rtol=rtol, atol=atol)
    return reference_result == candidate_result


def to_cpu_obj(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu_obj(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_cpu_obj(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_obj(item) for item in value)
    return value


def coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_golden_and_perf(result_payload: Any) -> Tuple[Any, Optional[float]]:
    if isinstance(result_payload, dict):
        if 'golden' in result_payload:
            return result_payload['golden'], coerce_optional_float(result_payload.get('perf'))
        if 'perf' in result_payload:
            golden_dict = {key: value for key, value in result_payload.items() if key != 'perf'}
            golden = golden_dict if golden_dict else None
            return golden, coerce_optional_float(result_payload.get('perf'))
    return result_payload, None


def clear_pts(pt_dir: str) -> None:
    if not os.path.isdir(pt_dir):
        return
    for pt in os.listdir(pt_dir):
        if pt.endswith('.pt'):
            os.remove(os.path.join(pt_dir, pt))


def log_error(log_file: str, kernel_name: str, stage: str, message: str, stderr: Optional[str] = None) -> None:
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as handle:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            handle.write(f"\n{'=' * 80}\n")
            handle.write(f'[{timestamp}] Kernel: {kernel_name} | Stage: {stage}\n')
            handle.write(f"{'-' * 80}\n")
            handle.write(f'{message}\n')
            if stderr:
                handle.write(f'\nStderr Output:\n{stderr}\n')
            handle.write(f"{'=' * 80}\n\n")
    except Exception as exc:  # pragma: no cover
        print(f'[WARNING] Failed to write error log: {exc}')


def prepare_environment(settings: EvalSettings, gpu_id: Optional[int]) -> Dict[str, str]:
    env = os.environ.copy()
    if gpu_id is not None:
        env['HIP_VISIBLE_DEVICES'] = str(gpu_id)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['HIP_EVAL_ARCH'] = settings.effective_arch
    env['HCC_AMDGPU_TARGET'] = settings.effective_arch
    env['AMDGPU_TARGETS'] = settings.effective_arch
    env['GPU_ARCHS'] = settings.effective_arch
    env['PYTORCH_ROCM_ARCH'] = settings.effective_arch
    env['MAX_JOBS'] = str(settings.compile_inner_jobs)
    env['OMP_NUM_THREADS'] = str(settings.omp_num_threads)
    env['MKL_NUM_THREADS'] = str(settings.mkl_num_threads)
    env['OPENBLAS_NUM_THREADS'] = str(settings.openblas_num_threads)
    return env


def ensure_tmp_dir(tmp_dir: Optional[str], kernel_name: str) -> Tuple[str, bool]:
    if tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir, False
    created = tempfile.mkdtemp(prefix=f'hip_eval_{kernel_name}_')
    return created, True


def write_text_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(content)


def run_python_script(
    script_path: str,
    *,
    env: Dict[str, str],
    timeout_s: int,
    kernel_name: str,
    stage: str,
    error_log_file: str,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        msg = f'[TIMEOUT] {stage} of {kernel_name} timed out after {timeout_s}s'
        log_error(error_log_file, kernel_name, f'{stage}_TIMEOUT', msg)
        raise
    if result.returncode != 0:
        msg = f'[ERROR] {stage} failed for {kernel_name}'
        log_error(error_log_file, kernel_name, f'{stage}_FAILED', msg, result.stderr)
        raise RuntimeError(f'{msg}: {result.stderr}')
    return result


def extract_timeout(request: EvalRequest, settings: EvalSettings) -> Tuple[int, int]:
    compile_timeout_s = int(request.compile_timeout_s or settings.compile_timeout_s)
    run_timeout_s = int(request.run_timeout_s or settings.run_timeout_s)
    return compile_timeout_s, run_timeout_s
