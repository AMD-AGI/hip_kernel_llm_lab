from __future__ import annotations

from pathlib import Path


def _path_text(path: Path) -> str:
    return path.as_posix()


def _paired_python_path(source_path: Path, source_root: Path, target_root: Path) -> Path:
    relative_path = source_path.relative_to(source_root).with_suffix(".py")
    paired_path = target_root / relative_path
    if not paired_path.exists():
        raise FileNotFoundError(
            f"Paired Python file was not found for {_path_text(source_path)}: expected {_path_text(paired_path)}."
        )
    return paired_path


def module_path_for_hip(hip_path: Path, hip_root: Path, module_root: Path) -> Path:
    return _paired_python_path(hip_path, hip_root, module_root)


def functional_path_for_hip(hip_path: Path, hip_root: Path, functional_root: Path) -> Path:
    return _paired_python_path(hip_path, hip_root, functional_root)


def output_path_for_hip(hip_path: Path, hip_root: Path, output_root: Path) -> Path:
    relative_path = hip_path.relative_to(hip_root).with_suffix(".hip")
    return output_root / relative_path
