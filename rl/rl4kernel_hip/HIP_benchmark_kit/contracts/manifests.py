"""Manifest filenames, JSON helpers, and lightweight validators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

GENERATION_MANIFEST = "generation_manifest.json"
SUBSET_MANIFEST = "subset_manifest.json"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_mapping(payload: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def validate_generation_manifest(payload: Mapping[str, Any]) -> None:
    required = (
        "model_path",
        "input_dir",
        "output_dir",
        "output_contract",
        "optimization_paradigm",
        "target_gpu",
        "rollout_n",
        "records",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"generation manifest missing fields: {', '.join(missing)}")
    if not isinstance(payload.get("records"), list):
        raise ValueError("generation manifest field records must be a list")


def validate_subset_manifest(payload: Mapping[str, Any]) -> None:
    required = ("source_root", "subset_root", "levels", "total_selected")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"subset manifest missing fields: {', '.join(missing)}")
    levels = payload.get("levels")
    if not isinstance(levels, Mapping):
        raise ValueError("subset manifest field levels must be an object")
    for level, level_payload in levels.items():
        if not isinstance(level_payload, Mapping):
            raise ValueError(f"subset manifest level {level!r} must be an object")
        if "selected_files" not in level_payload:
            raise ValueError(f"subset manifest level {level!r} missing selected_files")
