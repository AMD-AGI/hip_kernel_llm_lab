#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from multi_turn_tools import build_kernel_eval_tools_kwargs


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def augment_row(row: Dict[str, Any]) -> Dict[str, Any]:
    extra_info = dict(_coerce_dict(row.get("extra_info")))
    tools_kwargs = build_kernel_eval_tools_kwargs(row)
    if tools_kwargs:
        extra_info["tools_kwargs"] = tools_kwargs
        extra_info["need_tools_kwargs"] = True
        row["agent_name"] = "tool_agent_loop"
    row["extra_info"] = extra_info
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add multi-turn agentic fields to an existing verl parquet dataset.")
    parser.add_argument("--input", type=Path, required=True, help="Input parquet path.")
    parser.add_argument("--output", type=Path, default=None, help="Output parquet path. Defaults to <input>_agentic.parquet.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")
    output_path = args.output.expanduser().resolve() if args.output else input_path.with_name(f"{input_path.stem}_agentic{input_path.suffix}")

    frame = pd.read_parquet(input_path)
    records = [augment_row(dict(record)) for record in frame.to_dict(orient="records")]
    pd.DataFrame(records).to_parquet(output_path, index=False)

    print(f"[agentic] input : {input_path}")
    print(f"[agentic] output: {output_path}")
    print(f"[agentic] rows  : {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
