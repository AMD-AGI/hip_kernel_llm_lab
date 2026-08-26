# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from pathlib import Path


def iter_hip_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.hip") if path.is_file())
