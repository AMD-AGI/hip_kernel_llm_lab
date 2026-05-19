"""Ensure Metrix profiling artifacts exist for staged KernelBench subsets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from HIP_benchmark_kit.contracts.manifests import read_json


def selected_sample_stems(subset_manifest: Path, level: str) -> list[str]:
    payload = read_json(subset_manifest)
    records = payload.get("levels", {}).get(level, {}).get("selected_files", [])
    stems = []
    for record in records:
        hip_file = str(record.get("hip_file") or "")
        if hip_file.endswith(".hip"):
            stems.append(Path(hip_file).stem)
    return stems


def missing_profile_stems(sample_stems: list[str], output_dir: Path) -> list[str]:
    return [stem for stem in sample_stems if not (output_dir / f"{stem}_filtered.json").is_file()]


def _resolve_metrix_root(profile_script: Path) -> Path | None:
    resolved_script = profile_script.resolve()
    for parent in resolved_script.parents:
        if (parent / "src" / "metrix" / "__init__.py").is_file():
            return parent
    return None


def _profile_script_env(profile_script: Path) -> dict[str, str]:
    env = os.environ.copy()
    metrix_root = _resolve_metrix_root(profile_script)
    if metrix_root is None:
        return env

    metrix_src = metrix_root / "src"
    metrix_src_text = str(metrix_src)
    existing_pythonpath = env.get("PYTHONPATH", "")
    existing_paths = [path for path in existing_pythonpath.split(os.pathsep) if path]
    if metrix_src_text not in existing_paths:
        env["PYTHONPATH"] = os.pathsep.join([metrix_src_text, *existing_paths])
    return env


def _can_import_metrix(env: dict[str, str]) -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import metrix"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return probe.returncode == 0


def _ensure_metrix_runtime(profile_script: Path) -> dict[str, str]:
    baseline_env = os.environ.copy()
    if _can_import_metrix(baseline_env):
        return _profile_script_env(profile_script)

    metrix_root = _resolve_metrix_root(profile_script)
    if metrix_root is None:
        raise SystemExit(
            "Failed to import `metrix`, and could not discover a local metrix source tree "
            f"from profile script path: {profile_script}"
        )

    install_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "-e",
        str(metrix_root),
    ]
    print(
        "Metrix is not importable in the current Python environment; "
        "attempting local editable install without dependencies."
    )
    print(" ".join(install_command))
    try:
        subprocess.run(install_command, check=True)
    except subprocess.CalledProcessError:
        runtime_env = _profile_script_env(profile_script)
        if _can_import_metrix(runtime_env):
            print(
                "Metrix auto-install failed, but local source import via PYTHONPATH fallback works."
            )
            return runtime_env
        raise

    runtime_env = _profile_script_env(profile_script)
    if _can_import_metrix(runtime_env):
        return runtime_env

    raise SystemExit(
        "Metrix auto-install finished, but `import metrix` still fails. "
        "Please inspect the Python environment and installation logs."
    )


def ensure_metrix_profiles(
    *,
    subset_manifest: Path,
    level: str,
    dataset_root: Path,
    profile_artifact_root: Path,
    profile_script: Path,
    gpu_ids: str,
    parallel_workers: int,
    compile_workers: int,
    prewarm_iters: int,
    profile_iters: int,
    timeout_seconds: int,
    metadata_mode: str = "deferred",
) -> Path:
    sample_stems = selected_sample_stems(subset_manifest, level)
    if not sample_stems:
        raise SystemExit(f"No selected samples found for {level} in {subset_manifest}")
    output_dir = profile_artifact_root / level
    missing = missing_profile_stems(sample_stems, output_dir)
    if not missing:
        print(f"Metrix profiling artifacts already complete for {level}: {output_dir}")
        return output_dir
    if not profile_script.is_file():
        raise SystemExit(f"Profiling script not found: {profile_script}")

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(profile_script),
        "--hip-dir",
        str(dataset_root / "hip_code"),
        "--functional-dir",
        str(dataset_root / "pytorch_code_functional"),
        "--samples",
        ",".join(missing),
        "--output-dir",
        str(output_dir),
        "--gpu-ids",
        gpu_ids,
        "--parallel-workers",
        str(parallel_workers),
        "--compile-workers",
        str(compile_workers),
        "--prewarm-iters",
        str(prewarm_iters),
        "--profile-iters",
        str(profile_iters),
        "--timeout-seconds",
        str(timeout_seconds),
        "--metadata-mode",
        metadata_mode,
    ]
    print(f"Profiling {len(missing)} missing sample(s) for {level}: {' '.join(command)}")
    subprocess.run(command, check=True, env=_ensure_metrix_runtime(profile_script))
    still_missing = missing_profile_stems(sample_stems, output_dir)
    if still_missing:
        preview = ", ".join(still_missing[:10])
        raise SystemExit(f"Profiling completed but artifacts are still missing: {preview}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--profile-artifact-root", type=Path, required=True)
    parser.add_argument("--profile-script", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--compile-workers", type=int, default=0)
    parser.add_argument("--prewarm-iters", type=int, default=2)
    parser.add_argument("--profile-iters", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--metadata-mode",
        choices=("deferred", "full"),
        default="deferred",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ensure_metrix_profiles(
        subset_manifest=args.subset_manifest,
        level=args.level,
        dataset_root=args.dataset_root,
        profile_artifact_root=args.profile_artifact_root,
        profile_script=args.profile_script,
        gpu_ids=args.gpu_ids,
        parallel_workers=args.parallel_workers,
        compile_workers=args.compile_workers,
        prewarm_iters=args.prewarm_iters,
        profile_iters=args.profile_iters,
        timeout_seconds=args.timeout_seconds,
        metadata_mode=args.metadata_mode,
    )


if __name__ == "__main__":
    main()
