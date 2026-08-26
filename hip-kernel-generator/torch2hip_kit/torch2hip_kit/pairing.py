# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from pathlib import Path


def _path_text(path: Path) -> str:
    return path.as_posix()


def functional_path_for_module(module_path: Path, module_root: Path, functional_root: Path) -> Path:
    relative_path = module_path.relative_to(module_root)
    functional_path = functional_root / relative_path
    if not functional_path.exists():
        raise FileNotFoundError(
            f"Paired functional file was not found for {relative_path.as_posix()}: {_path_text(functional_path)}"
        )
    return functional_path


def hip_output_path_for_module(module_path: Path, module_root: Path, output_root: Path) -> Path:
    relative_path = module_path.relative_to(module_root)
    return (output_root / relative_path).with_suffix(".hip")
