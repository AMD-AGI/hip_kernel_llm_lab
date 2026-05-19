"""Context-aware rollout reuse helpers for KernelBench HIP runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from HIP_benchmark_kit.contracts.manifests import read_json, write_json


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selected_files(manifest: dict[str, Any], level: str) -> list[str]:
    return [
        str(record["hip_file"])
        for record in manifest.get("levels", {}).get(level, {}).get("selected_files", [])
    ]


@dataclass(frozen=True)
class ExpectedGenerationIdentity:
    context_mode: str
    model_path: str
    output_contract: str
    optimization_paradigm: str
    target_gpu: str
    data_source: str
    seed_base: str
    temperature: str
    prompt_map_arm: str
    target_prompt_map_json: Path | None

    def as_payload(self) -> dict[str, Any]:
        prompt_hash = ""
        if self.target_prompt_map_json and self.target_prompt_map_json.is_file():
            prompt_hash = sha256_file(self.target_prompt_map_json)
        return {
            "context_mode": self.context_mode,
            "model_path": self.model_path,
            "output_contract": self.output_contract,
            "optimization_paradigm": self.optimization_paradigm,
            "target_gpu": self.target_gpu,
            "data_source": self.data_source,
            "seed_base": self.seed_base,
            "temperature": self.temperature,
            "prompt_map_arm": self.prompt_map_arm,
            "prompt_map_sha256": prompt_hash,
        }


def source_prompt_hash(source_manifest: dict[str, Any]) -> str:
    manifest_hash = source_manifest.get("prompt_map_sha256") or ""
    if manifest_hash:
        return str(manifest_hash)
    prompt_path = source_manifest.get("prompt_map_json") or ""
    if prompt_path and Path(prompt_path).is_file():
        return sha256_file(Path(prompt_path))
    return ""


def identity_values_match(key: str, source_value: Any, expected_value: Any) -> bool:
    if key == "temperature":
        try:
            return float(source_value) == float(expected_value)
        except (TypeError, ValueError):
            return str(source_value) == str(expected_value)
    return str(source_value) == str(expected_value)


def validate_generation_identity(
    *,
    source_manifest: dict[str, Any],
    source_subset: dict[str, Any],
    target_subset: dict[str, Any],
    level: str,
    expected: ExpectedGenerationIdentity,
    target_rollout_n: int,
) -> dict[str, Any]:
    errors: list[str] = []
    source_rollout_n = int(source_manifest.get("rollout_n") or 0)
    if source_rollout_n <= 0:
        errors.append(f"invalid source rollout_n={source_rollout_n}")
    if source_rollout_n >= target_rollout_n:
        errors.append(f"source rollout_n={source_rollout_n} is not smaller than target rollout_n={target_rollout_n}")

    source_selected = selected_files(source_subset, level)
    target_selected = selected_files(target_subset, level)
    if source_selected != target_selected:
        errors.append("source and target subset selections differ")

    expected_payload = expected.as_payload()
    comparable = {
        "model_path": source_manifest.get("model_path", ""),
        "output_contract": source_manifest.get("output_contract", ""),
        "optimization_paradigm": source_manifest.get("optimization_paradigm", ""),
        "target_gpu": source_manifest.get("target_gpu", ""),
        "data_source": source_manifest.get("data_source", ""),
        "seed_base": "" if source_manifest.get("seed_base") is None else str(source_manifest.get("seed_base")),
        "temperature": "" if source_manifest.get("temperature") is None else str(source_manifest.get("temperature")),
        "prompt_map_arm": source_manifest.get("prompt_map_arm", ""),
        "prompt_map_sha256": source_prompt_hash(source_manifest),
    }
    if expected.optimization_paradigm and not comparable.get("optimization_paradigm"):
        errors.append("source manifest missing optimization_paradigm")
    for key, expected_value in expected_payload.items():
        if key == "context_mode":
            continue
        if expected_value and comparable.get(key, "") and not identity_values_match(key, comparable[key], expected_value):
            errors.append(f"{key} mismatch: source={comparable[key]!r} expected={expected_value!r}")

    if expected.context_mode == "B_profile_raw":
        if source_manifest.get("experiment_arm") != "B_profile_raw":
            errors.append("source manifest experiment_arm is not B_profile_raw")
        if source_manifest.get("prompt_map_arm") != "B_profile_raw":
            errors.append("source manifest prompt_map_arm is not B_profile_raw")
        if not source_prompt_hash(source_manifest):
            errors.append("source B_profile_raw prompt map hash is unavailable")

    identity_payload = {
        "expected": expected_payload,
        "source": comparable,
        "source_rollout_n": source_rollout_n,
        "target_rollout_n": target_rollout_n,
        "level": level,
        "selected_files": target_selected,
    }
    identity_payload["identity_hash"] = sha256_json(identity_payload)
    if errors:
        raise SystemExit("Cannot reuse rollout source:\n  - " + "\n  - ".join(errors))
    return identity_payload


def copy_raw_response(record: dict[str, Any], target_raw_response_dir: Path | None) -> str:
    source = record.get("raw_response_path") or ""
    if not source or not target_raw_response_dir:
        return ""
    source_path = Path(source)
    if not source_path.is_file():
        raise SystemExit(f"Raw response sidecar missing: {source_path}")
    target_raw_response_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_raw_response_dir / source_path.name
    if target_path.exists() and sha256_file(target_path) != sha256_file(source_path):
        raise SystemExit(f"Refusing to overwrite different raw response sidecar: {target_path}")
    shutil.copy2(source_path, target_path)
    return str(target_path)


def target_output_name(input_file: str, sample_idx: int, target_rollout_n: int) -> str:
    if target_rollout_n == 1:
        return input_file
    stem = Path(input_file).stem
    return f"{stem}_gen{int(sample_idx)}.hip"


def resolve_generation_dir(run_root: Path, level: str) -> Path:
    candidates = [
        run_root / level / "generated",
        run_root / level / "turn_01" / "generated",
    ]
    for candidate in candidates:
        if (candidate / "generation_manifest.json").is_file():
            return candidate
    return candidates[0]


def materialize_generation_reuse(
    *,
    source_run_root: Path,
    target_run_root: Path,
    level: str,
    target_rollout_n: int,
    context_mode: str,
    expected_model_path: str = "",
    expected_output_contract: str = "",
    expected_optimization_paradigm: str = "",
    expected_target_gpu: str = "",
    expected_data_source: str = "",
    expected_seed_base: str = "",
    expected_temperature: str = "",
    expected_prompt_map_arm: str = "",
    target_prompt_map_json: Path | None = None,
    target_raw_response_dir: Path | None = None,
    target_generation_dir: Path | None = None,
    output_plan: Path | None = None,
) -> dict[str, Any]:
    source_run_root = source_run_root.resolve()
    target_run_root = target_run_root.resolve()
    source_generation_dir = resolve_generation_dir(source_run_root, level)
    target_generation_dir = target_generation_dir.resolve() if target_generation_dir else target_run_root / level / "generated"
    source_manifest_path = source_generation_dir / "generation_manifest.json"
    source_subset_path = source_run_root / "subset" / "subset_manifest.json"
    target_subset_path = target_run_root / "subset" / "subset_manifest.json"

    for path in (source_manifest_path, source_subset_path, target_subset_path):
        if not path.is_file():
            raise SystemExit(f"Required reuse input missing: {path}")

    source_manifest = read_json(source_manifest_path)
    source_subset = read_json(source_subset_path)
    target_subset = read_json(target_subset_path)
    expected = ExpectedGenerationIdentity(
        context_mode=context_mode,
        model_path=expected_model_path,
        output_contract=expected_output_contract,
        optimization_paradigm=expected_optimization_paradigm,
        target_gpu=expected_target_gpu,
        data_source=expected_data_source,
        seed_base=expected_seed_base,
        temperature=expected_temperature,
        prompt_map_arm=expected_prompt_map_arm,
        target_prompt_map_json=target_prompt_map_json.resolve() if target_prompt_map_json else None,
    )
    identity = validate_generation_identity(
        source_manifest=source_manifest,
        source_subset=source_subset,
        target_subset=target_subset,
        level=level,
        expected=expected,
        target_rollout_n=target_rollout_n,
    )

    source_rollout_n = int(source_manifest["rollout_n"])
    target_generation_dir.mkdir(parents=True, exist_ok=True)
    target_raw_response_dir = target_raw_response_dir.resolve() if target_raw_response_dir else None

    imported_records: list[dict[str, Any]] = []
    copied_files = 0
    for record in source_manifest.get("records", []):
        sample_idx = record.get("sample_idx")
        if sample_idx is None or int(sample_idx) >= source_rollout_n:
            continue

        imported = dict(record)
        imported["reuse_source_run_root"] = str(source_run_root)
        imported["reuse_source_manifest"] = str(source_manifest_path)
        imported["reuse_identity_hash"] = identity["identity_hash"]
        imported["reused"] = True

        if imported.get("saved"):
            source_output = Path(record.get("output_path") or source_generation_dir / record["output_file"])
            if not source_output.is_file():
                raise SystemExit(f"Saved source HIP file missing: {source_output}")
            target_output = target_generation_dir / target_output_name(
                str(record.get("input_file") or source_output.name),
                int(sample_idx),
                target_rollout_n,
            )
            if target_output.exists() and sha256_file(target_output) != sha256_file(source_output):
                raise SystemExit(f"Refusing to overwrite different generated HIP file: {target_output}")
            shutil.copy2(source_output, target_output)
            copied_files += 1
            imported["output_file"] = target_output.name
            imported["output_path"] = str(target_output)

        copied_raw = copy_raw_response(imported, target_raw_response_dir)
        if copied_raw:
            imported["raw_response_path"] = copied_raw

        imported_records.append(imported)

    target_manifest = {key: value for key, value in source_manifest.items() if key != "records"}
    target_manifest["output_dir"] = str(target_generation_dir)
    target_manifest["rollout_n"] = target_rollout_n
    target_manifest["records"] = imported_records
    target_manifest["reuse_summary"] = {
        "source_run_root": str(source_run_root),
        "source_rollout_n": source_rollout_n,
        "target_rollout_n": target_rollout_n,
        "identity_hash": identity["identity_hash"],
        "imported_record_count": len(imported_records),
        "copied_hip_file_count": copied_files,
        "missing_rollout_indices": list(range(source_rollout_n, target_rollout_n)),
    }
    target_manifest["reuse_identity"] = identity
    write_json(target_generation_dir / "generation_manifest.json", target_manifest)

    plan = {
        **target_manifest["reuse_summary"],
        "level": level,
        "target_generation_manifest": str(target_generation_dir / "generation_manifest.json"),
    }
    if output_plan:
        write_json(output_plan.resolve(), plan)
    return plan


def materialize_generation(args: argparse.Namespace) -> None:
    plan = materialize_generation_reuse(
        source_run_root=args.source_run_root,
        target_run_root=args.target_run_root,
        level=args.level,
        target_rollout_n=args.target_rollout_n,
        context_mode=args.context_mode,
        expected_model_path=args.expected_model_path,
        expected_output_contract=args.expected_output_contract,
        expected_optimization_paradigm=args.expected_optimization_paradigm,
        expected_target_gpu=args.expected_target_gpu,
        expected_data_source=args.expected_data_source,
        expected_seed_base=args.expected_seed_base,
        expected_temperature=args.expected_temperature,
        expected_prompt_map_arm=args.expected_prompt_map_arm,
        target_prompt_map_json=args.target_prompt_map_json,
        target_raw_response_dir=args.target_raw_response_dir,
        target_generation_dir=args.target_generation_dir,
        output_plan=args.output_plan,
    )
    print(",".join(str(idx) for idx in plan["missing_rollout_indices"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize-generation")
    materialize.add_argument("--source-run-root", type=Path, required=True)
    materialize.add_argument("--target-run-root", type=Path, required=True)
    materialize.add_argument("--level", required=True)
    materialize.add_argument("--target-rollout-n", type=int, required=True)
    materialize.add_argument("--context-mode", required=True)
    materialize.add_argument("--expected-model-path", default="")
    materialize.add_argument("--expected-output-contract", default="")
    materialize.add_argument("--expected-optimization-paradigm", default="")
    materialize.add_argument("--expected-target-gpu", default="")
    materialize.add_argument("--expected-data-source", default="")
    materialize.add_argument("--expected-seed-base", default="")
    materialize.add_argument("--expected-temperature", default="")
    materialize.add_argument("--expected-prompt-map-arm", default="")
    materialize.add_argument("--target-prompt-map-json", type=Path)
    materialize.add_argument("--target-raw-response-dir", type=Path)
    materialize.add_argument("--target-generation-dir", type=Path)
    materialize.add_argument("--output-plan", type=Path)
    materialize.set_defaults(func=materialize_generation)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
