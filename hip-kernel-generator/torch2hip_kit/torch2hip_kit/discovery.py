from __future__ import annotations

from pathlib import Path


def iter_module_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.py") if path.is_file())
