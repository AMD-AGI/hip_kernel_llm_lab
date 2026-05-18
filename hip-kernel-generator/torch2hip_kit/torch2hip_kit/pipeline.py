from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from .config import AttemptRecord, ConversionRecord, PipelineConfig
from .discovery import iter_module_files
from .pairing import functional_path_for_module, hip_output_path_for_module
from .prompting import build_prompt, format_response
from .verifier import verify_candidate


def _path_text(path: Path) -> str:
    return path.as_posix()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _attempt_prompt_path(artifacts_dir: Path, relative_path: Path, attempt: int) -> Path:
    return artifacts_dir / "prompts" / relative_path.parent / f"{relative_path.stem}.attempt_{attempt}.txt"


def _attempt_candidate_path(artifacts_dir: Path, relative_path: Path, attempt: int) -> Path:
    return artifacts_dir / "candidates" / relative_path.parent / f"{relative_path.stem}.attempt_{attempt}.hip"


def _attempt_build_dir(build_root: Path, relative_path: Path, attempt: int) -> Path:
    return build_root / relative_path.parent / f"{relative_path.stem}.attempt_{attempt}"


def _persist_records(config: PipelineConfig, records: list[ConversionRecord]) -> None:
    serializable = [asdict(record) for record in records]
    successes = [record for record in serializable if record["status"] == "success"]
    failures = [record for record in serializable if record["status"] == "failed"]
    _write_json(config.records_file, serializable)
    _write_json(config.success_file, successes)
    _write_json(config.failure_file, failures)


def convert_single_file(source_path: Path, client, config: PipelineConfig) -> ConversionRecord:
    relative_path = source_path.relative_to(config.module_dir)
    output_path = hip_output_path_for_module(source_path, config.module_dir, config.output_dir)

    try:
        functional_path = functional_path_for_module(source_path, config.module_dir, config.functional_dir)
    except FileNotFoundError as exc:
        return ConversionRecord(
            module_source_path=_path_text(source_path),
            functional_source_path=_path_text(config.functional_dir / relative_path),
            relative_path=_path_text(relative_path),
            output_path=_path_text(output_path),
            status="failed",
            attempts_used=0,
            final_error=str(exc),
        )

    if output_path.exists() and not config.overwrite:
        return ConversionRecord(
            module_source_path=_path_text(source_path),
            functional_source_path=_path_text(functional_path),
            relative_path=_path_text(relative_path),
            output_path=_path_text(output_path),
            status="skipped",
            attempts_used=0,
        )

    module_code = source_path.read_text(encoding="utf-8")
    functional_code = functional_path.read_text(encoding="utf-8")
    attempt_records: list[AttemptRecord] = []
    best_attempt_record: AttemptRecord | None = None

    for attempt in range(1, config.max_attempts + 1):
        prompt = build_prompt(
            module_code,
            functional_code,
            relative_path,
            attempt_records,
            system_instruction=config.system_instruction or "",
            few_shot_examples=config.few_shot_examples or "",
            code_char_limit=config.history_code_char_limit,
            feedback_char_limit=config.history_feedback_char_limit,
        )
        prompt_path = _attempt_prompt_path(config.artifacts_dir, relative_path, attempt)
        candidate_path = _attempt_candidate_path(config.artifacts_dir, relative_path, attempt)
        build_dir = _attempt_build_dir(config.build_root, relative_path, attempt)
        _write_text(prompt_path, prompt)

        try:
            response = client.generate(
                [{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            candidate_code = format_response(response)
            if not candidate_code.strip():
                raise ValueError("Model returned an empty candidate.")

            _write_text(candidate_path, candidate_code)
            verification = verify_candidate(
                source_path,
                functional_path,
                candidate_path,
                build_dir=build_dir,
                seed=config.seed,
                rtol=config.rtol,
                atol=config.atol,
                perf_warmup=config.perf_warmup,
                perf_iterations=config.perf_iterations,
                keep_build_dir=config.keep_build_dirs,
            )
            attempt_record = AttemptRecord(
                attempt=attempt,
                prompt_path=_path_text(prompt_path),
                candidate_path=_path_text(candidate_path),
                status="success" if verification.success else "failed",
                feedback=verification.message,
                mismatch=verification.message if not verification.success else None,
                compile_success=verification.compile_success,
                correctness_success=verification.correctness_success,
                speedup=verification.speedup,
                module_latency_ms=verification.module_latency_ms,
                hip_latency_ms=verification.hip_latency_ms,
            )
            attempt_records.append(attempt_record)
            if verification.success and (
                best_attempt_record is None
                or (verification.speedup or 0.0) > (best_attempt_record.speedup or 0.0)
            ):
                best_attempt_record = attempt_record
        except Exception as exc:
            error_trace = traceback.format_exc()
            feedback = f"{type(exc).__name__}: {exc}\n\n{error_trace}"
            attempt_records.append(
                AttemptRecord(
                    attempt=attempt,
                    prompt_path=_path_text(prompt_path),
                    candidate_path=_path_text(candidate_path),
                    status="failed",
                    feedback=feedback,
                    error=feedback,
                )
            )

    if best_attempt_record is not None:
        best_candidate_path = Path(best_attempt_record.candidate_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(best_candidate_path.read_text(encoding="utf-8"), encoding="utf-8")
        return ConversionRecord(
            module_source_path=_path_text(source_path),
            functional_source_path=_path_text(functional_path),
            relative_path=_path_text(relative_path),
            output_path=_path_text(output_path),
            status="success",
            attempts_used=len(attempt_records),
            attempts=attempt_records,
            best_attempt=best_attempt_record.attempt,
            best_speedup=best_attempt_record.speedup,
        )

    return ConversionRecord(
        module_source_path=_path_text(source_path),
        functional_source_path=_path_text(functional_path),
        relative_path=_path_text(relative_path),
        output_path=_path_text(output_path),
        status="failed",
        attempts_used=len(attempt_records),
        attempts=attempt_records,
        final_error=attempt_records[-1].feedback if attempt_records else "No attempts were executed.",
    )


def run_conversion_pipeline(client, config: PipelineConfig) -> dict[str, int]:
    config = config.with_defaults()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    module_files = iter_module_files(config.module_dir)
    records: list[ConversionRecord] = []

    for source_path in tqdm(module_files, desc="Converting module files to HIP"):
        record = convert_single_file(source_path, client, config)
        records.append(record)
        _persist_records(config, records)

    return {
        "total": len(records),
        "success": sum(record.status == "success" for record in records),
        "failed": sum(record.status == "failed" for record in records),
        "skipped": sum(record.status == "skipped" for record in records),
    }
