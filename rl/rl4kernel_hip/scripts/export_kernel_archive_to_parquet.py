#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _collect_record_paths(archive_dir: Path) -> list[Path]:
    return sorted(path for path in archive_dir.rglob("records.*.jsonl") if path.is_file())


def _load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"failed to parse {path}:{line_no}: {exc}") from exc
                record["_source_file"] = str(path)
                records.append(record)
    return records


def export_archive_to_parquet(archive_dir: Path, output_path: Path) -> tuple[int, int]:
    record_paths = _collect_record_paths(archive_dir)
    if not record_paths:
        raise FileNotFoundError(f"no archive shard files found under {archive_dir}")

    records = _load_records(record_paths)
    if not records:
        raise ValueError(f"archive shards under {archive_dir} do not contain any JSONL records")

    frame = pd.DataFrame.from_records(records)
    sort_columns = [column for column in ("train_step", "timestamp_epoch", "kernel_name") if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return (len(record_paths), len(frame))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge kernel archive JSONL shards and export a single Parquet file.",
    )
    parser.add_argument(
        "--archive-dir",
        required=True,
        type=Path,
        help="Archive directory containing records.*.jsonl shards.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output parquet path. Defaults to <archive-dir>/kernel_archive.parquet.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    archive_dir = args.archive_dir.expanduser().resolve()
    output_path = (args.output or (archive_dir / "kernel_archive.parquet")).expanduser().resolve()

    shard_count, record_count = export_archive_to_parquet(archive_dir, output_path)
    print(
        json.dumps(
            {
                "archive_dir": str(archive_dir),
                "output_path": str(output_path),
                "shard_count": shard_count,
                "record_count": record_count,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
