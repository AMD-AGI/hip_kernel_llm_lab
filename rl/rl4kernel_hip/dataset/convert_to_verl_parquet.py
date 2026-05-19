#!/usr/bin/env python3
"""
Convert kernel-agent JSON records into react-format veRL parquet files.

Supported inputs:
  - rl_data records with a data_info block.
  - legacy kernel2kernel processed records with kernel_name/input/hip_reference_cde.

Examples:
    python convert_to_verl_parquet.py --target-gpus mi300x mi325x

    python convert_to_verl_parquet.py \
      --input-jsons hip_kernel_rldataset/rl_data_hard.json hip_kernel_rldataset/rl_data_normal.json \
      --shuffle --seed 42 --output-name rl_data_hard_normal_mixed
"""

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Make repo root available for importing the dataset package.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.contracts import (
    HIP2HIP_FULL_FILE_PARADIGM,
    LEGACY_HIP_FENCE_OUTPUT_CONTRACT,
    SAMPLE_JSON_OUTPUT_CONTRACT,
    contract_to_extra_info,
    infer_optimization_paradigm,
    resolve_optimization_contract,
    validate_training_row_contract,
)
from dataset.prompts import DEFAULT_TARGET_GPU, GPU_HARDWARE_CONFIGS
from dataset.utils import fetch_hip_kernel_agent_system_prompt

# Paths
DEFAULT_INPUT_JSON = SCRIPT_DIR / "rl_data_v01.json"
DEFAULT_REFERENCE_ROOT = (
    DATASET_DIR
    / "AIG-Datasets"
    / "v0.1"
    / "PyTorch_HIP_kernel_dataset"
    / "pytorch_hip_kernel_gpumode"
)
REFERENCE_PATH_MARKERS = (
    "hip_opt",
    "pytorch_code_module",
    "pytorch_code_functional",
)
SUPPORTED_TARGET_GPUS = sorted(GPU_HARDWARE_CONFIGS.keys())
LEGACY_OUTPUT_CONTRACT = LEGACY_HIP_FENCE_OUTPUT_CONTRACT
DEFAULT_OUTPUT_CONTRACT = SAMPLE_JSON_OUTPUT_CONTRACT
SUPPORTED_OUTPUT_CONTRACTS = (
    LEGACY_OUTPUT_CONTRACT,
    SAMPLE_JSON_OUTPUT_CONTRACT,
)
OUTPUT_CONTRACT_VERSION = 1
DEFAULT_DATA_SOURCE = "kernel-agent-react-train"
SUPPORTED_INPUT_FORMATS = ("auto", "rl_data", "kernel2kernel_json")


# ANSI color codes for terminal output
class Colors:
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    RESET = "\033[0m"


def field(name: str) -> str:
    """Format a field name with highlight."""
    return f"{Colors.CYAN}{Colors.BOLD}{name}{Colors.RESET}"


def value_type(val: str) -> str:
    """Format a type annotation."""
    return f"{Colors.YELLOW}{val}{Colors.RESET}"


def section(name: str) -> str:
    """Format a section header."""
    return f"{Colors.GREEN}{Colors.BOLD}{name}{Colors.RESET}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert kernel-agent JSON records into react-format veRL parquet files."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help="Path to a single input JSON file. Ignored when --input-jsons is provided.",
    )
    parser.add_argument(
        "--input-jsons",
        type=Path,
        nargs="+",
        default=None,
        help="One or more input JSON files to concatenate before conversion.",
    )
    parser.add_argument(
        "--input-format",
        choices=SUPPORTED_INPUT_FORMATS,
        default="auto",
        help="Input schema. Use auto to infer from the first record in each input file.",
    )
    parser.add_argument(
        "--source-tags",
        nargs="+",
        default=None,
        help="Optional per-input source tags stored in extra_info.source_dataset.",
    )
    parser.add_argument(
        "--reference-root",
        "--reference-json",
        dest="reference_root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Root directory of the local v0.1 PyTorch HIP dataset tree.",
    )
    parser.add_argument(
        "--pytorch-root",
        type=Path,
        default=None,
        help="Root containing pytorch_code_module/pytorch_code_functional/hip_opt for kernel2kernel_json.",
    )
    parser.add_argument(
        "--hip-opt-dir",
        type=Path,
        default=None,
        help="Optional hip_opt directory used to map kernel function names to sample ids.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory for generated parquet files.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Base output name before target GPU and output contract suffixes.",
    )
    parser.add_argument(
        "--data-source",
        default=DEFAULT_DATA_SOURCE,
        help="data_source value stored in each parquet row.",
    )
    parser.add_argument(
        "--optimization-paradigm",
        "--prompt-mode",
        dest="optimization_paradigm",
        default="",
        help="Prompt/code-unit paradigm. Defaults from --data-source when omitted.",
    )
    parser.add_argument(
        "--target-gpus",
        nargs="+",
        default=[DEFAULT_TARGET_GPU],
        choices=SUPPORTED_TARGET_GPUS,
        help="One or more target GPU profiles used to render prompts.",
    )
    parser.add_argument(
        "--preview-records",
        type=int,
        default=1,
        help="Number of converted records to preview per output parquet.",
    )
    parser.add_argument(
        "--output-contract",
        type=str,
        default=DEFAULT_OUTPUT_CONTRACT,
        choices=SUPPORTED_OUTPUT_CONTRACTS,
        help="Assistant output contract encoded in the prompt and extra_info metadata.",
    )
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        default=None,
        help="Shuffle concatenated input records before conversion.",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable shuffling even when multiple inputs are provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling input records.",
    )
    parser.add_argument(
        "--max-kernel-input-len",
        type=int,
        default=40000,
        help="Maximum legacy kernel2kernel input snippet length to include.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def read_text_file(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


def infer_source_tag(path: Path) -> str:
    stem = path.stem
    if stem.startswith("rl_data_"):
        return stem[len("rl_data_") :]
    return stem


def detect_input_format(records: List[Dict[str, Any]], requested_format: str, path: Path) -> str:
    if requested_format != "auto":
        return requested_format
    if not records:
        raise ValueError(f"Input JSON is empty: {path}")
    first = records[0]
    if isinstance(first, dict) and isinstance(first.get("data_info"), dict):
        return "rl_data"
    if isinstance(first, dict) and {"kernel_name", "input"}.issubset(first):
        if any(key in first for key in ("hip_reference_cde", "hip_reference_code", "hip_reference")):
            return "kernel2kernel_json"
    raise ValueError(f"Unable to infer input format for {path}")


def load_source_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    input_paths = args.input_jsons or [args.input_json]
    source_tags = args.source_tags
    if source_tags is not None and len(source_tags) != len(input_paths):
        raise ValueError("--source-tags must have the same length as the selected input JSON paths.")

    source_records: List[Dict[str, Any]] = []
    for idx, input_path in enumerate(input_paths):
        expanded_path = input_path.expanduser()
        data = load_json(expanded_path)
        if not isinstance(data, list):
            raise ValueError(f"Top-level JSON must be a list: {expanded_path}")
        input_format = detect_input_format(data, args.input_format, expanded_path)
        source_tag = source_tags[idx] if source_tags else infer_source_tag(expanded_path)
        print(
            f"Loaded {len(data)} records from {expanded_path} "
            f"(format={input_format}, source_dataset={source_tag})"
        )
        for record in data:
            source_records.append(
                {
                    "record": record,
                    "source_dataset": source_tag,
                    "source_index": idx,
                    "source_json": str(expanded_path),
                    "input_format": input_format,
                }
            )

    should_shuffle = args.shuffle if args.shuffle is not None else len(input_paths) > 1
    if should_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(source_records)
        print(f"Shuffled {len(source_records)} records with seed={args.seed}")

    source_counts = Counter(item["source_dataset"] for item in source_records)
    print(f"Source distribution: {dict(source_counts)}")
    return source_records


def resolve_reference_path(raw_path: str, reference_root: Path) -> Path:
    if not raw_path:
        raise ValueError("Missing required legacy reference path in data_info")

    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    parts = candidate.parts
    anchor_name = reference_root.name

    if anchor_name in parts:
        anchor_idx = parts.index(anchor_name)
        remapped = reference_root.joinpath(*parts[anchor_idx + 1 :])
        if remapped.exists():
            return remapped

    for marker in REFERENCE_PATH_MARKERS:
        if marker in parts:
            marker_idx = parts.index(marker)
            remapped = reference_root.joinpath(*parts[marker_idx:])
            if remapped.exists():
                return remapped

    raise FileNotFoundError(
        f"Unable to map legacy path '{raw_path}' under local dataset root '{reference_root}'"
    )


def extract_sample_id(hip_code_path: str) -> str:
    if not hip_code_path:
        return "unknown"
    return Path(hip_code_path).parent.name or "unknown"


def build_prompt_messages(
    starter_code: str,
    target_gpu: str,
    output_contract: str,
    optimization_paradigm: str,
) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": fetch_hip_kernel_agent_system_prompt(
                "",
                starter_code=starter_code,
                target_gpu=target_gpu,
                output_contract=output_contract,
                optimization_paradigm=optimization_paradigm,
            ),
        }
    ]

def build_reward_model(
    *,
    kernel_name: str,
    hip_code: str,
    pytorch_module_code: str,
    pytorch_functional_code: str,
) -> Dict[str, Any]:
    return {
        "ground_truth": {
            "atol": 0.0001,
            "compile_timeout_s": 10000,
            "cuda_baseline_code": None,
            "cuda_baseline_path": None,
            "hip_code": hip_code,
            "kernel_name": kernel_name,
            "pytorch_functional_code": pytorch_functional_code,
            "pytorch_module_code": pytorch_module_code,
            "rtol": 0.001,
            "run_timeout_s": 10000,
        },
        "style": "sandbox_fusion",
    }


def finalize_record(
    *,
    prompt: List[Dict[str, str]],
    data_source: str,
    reward_model: Dict[str, Any],
    extra_info: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "data_source": data_source,
        "ability": "kernel_optimization",
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def build_kernel_func_to_id_mapping(hip_opt_dir: Optional[Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not hip_opt_dir or not hip_opt_dir.is_dir():
        print(f"[WARN] hip_opt directory not found: {hip_opt_dir}")
        return mapping

    kernel_func_pattern = re.compile(r"__global__\s+void\s+(\w+)\s*\(")
    duplicate_count = 0
    for dir_path in sorted((path for path in hip_opt_dir.iterdir() if path.is_dir()), key=lambda p: p.name):
        kernel_funcs_found = set()
        for hip_file in sorted(dir_path.glob("*.hip"), key=lambda p: p.name):
            try:
                content = read_text_file(str(hip_file))
            except Exception as exc:
                print(f"[WARN] Failed to read {hip_file}: {exc}")
                continue
            for match in kernel_func_pattern.finditer(content):
                kernel_funcs_found.add(match.group(1))

        info_json = dir_path / "info.json"
        if info_json.is_file():
            try:
                info_data = json.loads(read_text_file(str(info_json)))
            except Exception as exc:
                print(f"[WARN] Failed to read {info_json}: {exc}")
                info_data = {}
            for value in info_data.values() if isinstance(info_data, dict) else []:
                if isinstance(value, dict) and isinstance(value.get("code"), str):
                    for match in kernel_func_pattern.finditer(value["code"]):
                        kernel_funcs_found.add(match.group(1))

        for func_name in sorted(kernel_funcs_found):
            if func_name in mapping:
                duplicate_count += 1
                continue
            mapping[func_name] = dir_path.name

    print(
        f"[INFO] Built kernel function mapping with {len(mapping)} entries "
        f"from {hip_opt_dir} (duplicates ignored={duplicate_count})"
    )
    return mapping


def load_pytorch_code_mapping(pytorch_root: Optional[Path]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    if not pytorch_root:
        return mapping

    module_dir = pytorch_root / "pytorch_code_module"
    functional_dir = pytorch_root / "pytorch_code_functional"
    if not module_dir.is_dir() or not functional_dir.is_dir():
        print(f"[WARN] PyTorch code directories not found under {pytorch_root}")
        return mapping

    for module_path in sorted(module_dir.glob("py_*.py"), key=lambda p: p.name):
        kernel_id_name = module_path.name[3:-3]
        functional_path = functional_dir / f"py_{kernel_id_name}_func.py"
        try:
            module_code = read_text_file(str(module_path))
        except Exception as exc:
            print(f"[WARN] Failed to read {module_path}: {exc}")
            module_code = ""
        try:
            functional_code = read_text_file(str(functional_path)) if functional_path.is_file() else ""
        except Exception as exc:
            print(f"[WARN] Failed to read {functional_path}: {exc}")
            functional_code = ""

        mapping[kernel_id_name] = {
            "pytorch_module_code": module_code,
            "pytorch_functional_code": functional_code,
            "pytorch_module_path": str(module_path),
            "pytorch_functional_path": str(functional_path) if functional_path.is_file() else "",
        }

    print(f"[INFO] Loaded {len(mapping)} PyTorch code mappings from {pytorch_root}")
    return mapping


def convert_rl_data_record(
    *,
    source_item: Dict[str, Any],
    reference_root: Path,
    target_gpu: str,
    output_contract: str,
    data_source: str,
    optimization_paradigm: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    record = source_item["record"]
    data_info = record.get("data_info", {})
    hip_code_path = data_info.get("hip_code_path", "")
    pytorch_module_path = data_info.get("pytorch_code_module", "")
    pytorch_functional_path = data_info.get("pytorch_code_functional", "")

    kernel_names = data_info.get("name", [])
    if isinstance(kernel_names, str):
        kernel_name = kernel_names
    else:
        kernel_name = kernel_names[0] if kernel_names else "unknown"
    starter_code = (data_info.get("original_code_body") or "").strip()

    resolved_hip_path = resolve_reference_path(hip_code_path, reference_root)
    resolved_module_path = resolve_reference_path(pytorch_module_path, reference_root)
    resolved_functional_path = resolve_reference_path(pytorch_functional_path, reference_root)

    sample_id = extract_sample_id(str(resolved_hip_path))
    hip_code = read_text_file(str(resolved_hip_path))
    pytorch_module_code = read_text_file(str(resolved_module_path))
    pytorch_functional_code = read_text_file(str(resolved_functional_path))
    contract = resolve_optimization_contract(
        data_source=data_source,
        requested_paradigm=optimization_paradigm,
        output_contract=output_contract,
    )
    prompt_starter_code = hip_code if contract.optimization_paradigm == HIP2HIP_FULL_FILE_PARADIGM else starter_code

    extra_info = {
        "kernel_id_name": sample_id,
        "kernel_name": kernel_name,
        "target_gpu": target_gpu,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        **contract_to_extra_info(contract),
        "paths": {
            "hip": str(resolved_hip_path),
            "pytorch_module": str(resolved_module_path),
            "pytorch_functional": str(resolved_functional_path),
        },
        "reference_lookup": {
            "sample_id": sample_id,
            "reference_root": str(reference_root),
            "hip_resolution": "local_path_map",
            "module_resolution": "local_path_map",
            "functional_resolution": "local_path_map",
            "requested_hip_path": hip_code_path,
            "resolved_hip_path": str(resolved_hip_path),
            "requested_module_path": pytorch_module_path,
            "resolved_module_path": str(resolved_module_path),
            "requested_functional_path": pytorch_functional_path,
            "resolved_functional_path": str(resolved_functional_path),
        },
        "source_type": "hip",
        "source_dataset": source_item["source_dataset"],
        "source_json": source_item["source_json"],
        "input_format": source_item["input_format"],
        "original_code_body": data_info.get("original_code_body", ""),
        "signature": data_info.get("signature", []),
    }

    converted_record = finalize_record(
        prompt=build_prompt_messages(
            prompt_starter_code,
            target_gpu,
            contract.output_contract,
            contract.optimization_paradigm,
        ),
        data_source=data_source,
        reward_model=build_reward_model(
            kernel_name=kernel_name,
            hip_code=hip_code,
            pytorch_module_code=pytorch_module_code,
            pytorch_functional_code=pytorch_functional_code,
        ),
        extra_info=extra_info,
    )
    return converted_record, {
        "sample_id": sample_id,
        "hip_resolution": "local_path_map",
        "source_dataset": source_item["source_dataset"],
    }


def convert_kernel2kernel_record(
    *,
    source_item: Dict[str, Any],
    target_gpu: str,
    output_contract: str,
    data_source: str,
    optimization_paradigm: str,
    pytorch_mapping: Dict[str, Dict[str, str]],
    kernel_func_to_id: Dict[str, str],
    pytorch_root: Optional[Path],
    hip_opt_dir: Optional[Path],
    max_kernel_input_len: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    record = source_item["record"]
    kernel_name = str(record.get("kernel_name") or "").strip()
    starter_code = str(record.get("input") or "").strip()
    hip_reference = str(
        record.get("hip_reference_cde")
        or record.get("hip_reference_code")
        or record.get("hip_reference")
        or ""
    )
    if not kernel_name:
        raise ValueError("kernel2kernel_json item missing kernel_name")
    if not starter_code or not hip_reference:
        raise ValueError(f"kernel2kernel_json item {kernel_name} missing input or HIP reference")
    if len(starter_code) > max_kernel_input_len:
        raise ValueError(
            f"kernel2kernel_json item {kernel_name} input too long "
            f"({len(starter_code)} > {max_kernel_input_len})"
        )

    kernel_id_name = kernel_func_to_id.get(kernel_name, "")
    pytorch_bundle = pytorch_mapping.get(kernel_id_name, {}) if kernel_id_name else {}
    pytorch_module_code = pytorch_bundle.get("pytorch_module_code", "")
    pytorch_functional_code = pytorch_bundle.get("pytorch_functional_code", "")
    contract = resolve_optimization_contract(
        data_source=data_source,
        requested_paradigm=optimization_paradigm,
        output_contract=output_contract,
    )

    extra_info = {
        "kernel_id_name": kernel_id_name,
        "kernel_name": kernel_name,
        "target_gpu": target_gpu,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        **contract_to_extra_info(contract),
        "paths": {
            "hip": "",
            "pytorch_module": pytorch_bundle.get("pytorch_module_path", ""),
            "pytorch_functional": pytorch_bundle.get("pytorch_functional_path", ""),
        },
        "reference_lookup": {
            "sample_id": kernel_id_name,
            "reference_root": str(pytorch_root) if pytorch_root else "",
            "hip_opt_dir": str(hip_opt_dir) if hip_opt_dir else "",
            "hip_resolution": "embedded_reference_code",
            "module_resolution": "kernel_name_to_sample_id",
            "functional_resolution": "kernel_name_to_sample_id",
            "requested_kernel_name": kernel_name,
            "resolved_kernel_id_name": kernel_id_name,
        },
        "source_type": "hip",
        "source_dataset": source_item["source_dataset"],
        "source_json": source_item["source_json"],
        "input_format": source_item["input_format"],
        "original_code_body": starter_code,
        "signature": record.get("signature", []),
    }

    converted_record = finalize_record(
        prompt=build_prompt_messages(
            hip_reference if contract.optimization_paradigm == HIP2HIP_FULL_FILE_PARADIGM else starter_code,
            target_gpu,
            contract.output_contract,
            contract.optimization_paradigm,
        ),
        data_source=data_source,
        reward_model=build_reward_model(
            kernel_name=kernel_name,
            hip_code=hip_reference,
            pytorch_module_code=pytorch_module_code,
            pytorch_functional_code=pytorch_functional_code,
        ),
        extra_info=extra_info,
    )
    return converted_record, {
        "sample_id": kernel_id_name,
        "hip_resolution": "embedded_reference_code",
        "source_dataset": source_item["source_dataset"],
    }


def derive_output_name(output_name: Optional[str], source_records: List[Dict[str, Any]]) -> str:
    if output_name:
        return output_name
    source_order: Dict[str, int] = {}
    for item in source_records:
        tag = item["source_dataset"]
        source_order[tag] = min(source_order.get(tag, item["source_index"]), item["source_index"])
    source_tags = [tag for tag, _ in sorted(source_order.items(), key=lambda item: item[1])]
    if len(source_tags) == 1:
        return f"rl_data_{source_tags[0]}" if source_tags[0] != "v01" else "rl_data_v01"
    return "rl_data_" + "_".join(source_tags) + "_mixed"


def build_output_path(
    output_dir: Path,
    target_gpu: str,
    output_contract: str,
    output_name: str,
) -> Path:
    normalized_contract = (output_contract or LEGACY_OUTPUT_CONTRACT).strip().lower()
    if normalized_contract == LEGACY_OUTPUT_CONTRACT:
        return output_dir / f"{output_name}_{target_gpu}_react_verl.parquet"
    suffix = normalized_contract.replace("-", "_")
    return output_dir / f"{output_name}_{target_gpu}_react_{suffix}_verl.parquet"


def print_preview(converted_records: List[Dict[str, Any]], preview_records: int) -> None:
    if preview_records <= 0 or not converted_records:
        return

    print("\n" + "=" * 80)
    print(f"{Colors.GREEN}{Colors.BOLD}SAMPLE RECORDS FOR REVIEW{Colors.RESET}")
    print("=" * 80)

    for i in range(min(preview_records, len(converted_records))):
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}--- Record {i} ---{Colors.RESET}")
        record = converted_records[i]

        print(f"{field('data_source')}: {record['data_source']}")
        print(f"{field('ability')}: {record['ability']}")

        prompt_type = type(record["prompt"]).__name__
        print(f"\n{section('prompt')} {value_type(f'(type: {prompt_type})')}:")
        first_msg = record["prompt"][0]
        print(f"  {field('role')}: {first_msg['role']}")
        print(f"  {field('content')}:\n{first_msg['content']}")

        print(f"\n{section('extra_info')}:")
        print(f"  {field('kernel_id_name')}: {record['extra_info']['kernel_id_name']}")
        print(f"  {field('kernel_name')}: {record['extra_info']['kernel_name']}")
        print(f"  {field('source_dataset')}: {record['extra_info'].get('source_dataset')}")
        print(f"  {field('target_gpu')}: {record['extra_info']['target_gpu']}")
        print(f"  {field('output_contract')}: {record['extra_info']['output_contract']}")
        print(f"  {field('reference_lookup')}: {record['extra_info']['reference_lookup']}")

        print(f"\n{section('reward_model.ground_truth')}:")
        gt = record["reward_model"]["ground_truth"]
        print(f"  {field('kernel_name')}: {gt['kernel_name']}")
        print(
            f"  {field('hip_code')}: "
            f"{gt['hip_code'][:300] + '...' if gt['hip_code'] and len(gt['hip_code']) > 300 else gt['hip_code']}"
        )
        print("-" * 40)


def has_required_ground_truth(converted: Dict[str, Any]) -> bool:
    ground_truth = converted["reward_model"]["ground_truth"]
    return bool(
        ground_truth.get("hip_code")
        and ground_truth.get("pytorch_module_code")
        and ground_truth.get("pytorch_functional_code")
    )


def _prompt_text_length(record: Dict[str, Any]) -> int:
    prompt = record.get("prompt")
    if isinstance(prompt, list):
        return sum(len(str(message.get("content") or "")) for message in prompt if isinstance(message, dict))
    return len(str(prompt or ""))


def _percentile(values: List[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def print_length_stats(converted_records: List[Dict[str, Any]]) -> None:
    prompt_lengths = [_prompt_text_length(record) for record in converted_records]
    hip_ref_lengths = [
        len(str(record["reward_model"]["ground_truth"].get("hip_code") or ""))
        for record in converted_records
    ]
    print(f"{field('Prompt chars')}: min={min(prompt_lengths)} p50={_percentile(prompt_lengths, 0.50)} "
          f"p95={_percentile(prompt_lengths, 0.95)} max={max(prompt_lengths)}")
    print(f"{field('Reference HIP chars')}: min={min(hip_ref_lengths)} p50={_percentile(hip_ref_lengths, 0.50)} "
          f"p95={_percentile(hip_ref_lengths, 0.95)} max={max(hip_ref_lengths)}")


def convert_for_target_gpu(
    *,
    source_records: List[Dict[str, Any]],
    reference_root: Path,
    target_gpu: str,
    output_contract: str,
    output_path: Path,
    preview_records: int,
    data_source: str,
    optimization_paradigm: str,
    pytorch_mapping: Dict[str, Dict[str, str]],
    kernel_func_to_id: Dict[str, str],
    pytorch_root: Optional[Path],
    hip_opt_dir: Optional[Path],
    max_kernel_input_len: int,
) -> None:
    expected_contract = resolve_optimization_contract(
        data_source=data_source,
        requested_paradigm=optimization_paradigm,
        output_contract=output_contract,
    )
    print(f"\n{section('BUILD TARGET')}: {target_gpu}")
    print(f"Using data_source: {expected_contract.data_source}")
    print(f"Using output contract: {expected_contract.output_contract}")
    print(f"Using optimization paradigm: {expected_contract.optimization_paradigm}")
    print(f"Using expected code unit: {expected_contract.expected_code_unit}")
    print(f"Using persistence mode: {expected_contract.persistence_mode}")
    print(f"Writing output to: {output_path}")

    converted_records: List[Dict[str, Any]] = []
    skipped_count = 0
    resolved_path_count = 0
    converted_source_counts: Counter = Counter()
    skipped_source_counts: Counter = Counter()

    for i, source_item in enumerate(source_records):
        try:
            if source_item["input_format"] == "rl_data":
                converted, resolution_info = convert_rl_data_record(
                    source_item=source_item,
                    reference_root=reference_root,
                    target_gpu=target_gpu,
                    output_contract=output_contract,
                    data_source=data_source,
                    optimization_paradigm=optimization_paradigm,
                )
            elif source_item["input_format"] == "kernel2kernel_json":
                converted, resolution_info = convert_kernel2kernel_record(
                    source_item=source_item,
                    target_gpu=target_gpu,
                    output_contract=output_contract,
                    data_source=data_source,
                    optimization_paradigm=optimization_paradigm,
                    pytorch_mapping=pytorch_mapping,
                    kernel_func_to_id=kernel_func_to_id,
                    pytorch_root=pytorch_root,
                    hip_opt_dir=hip_opt_dir,
                    max_kernel_input_len=max_kernel_input_len,
                )
            else:
                raise ValueError(f"Unsupported input format: {source_item['input_format']}")

            if not has_required_ground_truth(converted):
                print(
                    f"Skipping record {i}: missing HIP/PyTorch reference code "
                    f"(kernel={converted['extra_info'].get('kernel_name')}, "
                    f"source={source_item['source_dataset']})"
                )
                skipped_count += 1
                skipped_source_counts[source_item["source_dataset"]] += 1
                continue

            contract_errors = validate_training_row_contract(converted, expected_contract)
            if contract_errors:
                raise ValueError("; ".join(contract_errors))

            if resolution_info["hip_resolution"] in {"local_path_map", "embedded_reference_code"}:
                resolved_path_count += 1

            converted_records.append(converted)
            converted_source_counts[source_item["source_dataset"]] += 1
        except Exception as exc:
            print(
                f"Error converting record {i} "
                f"(format={source_item['input_format']}, source={source_item['source_dataset']}): {exc}"
            )
            skipped_count += 1
            skipped_source_counts[source_item["source_dataset"]] += 1

    if not converted_records:
        raise RuntimeError("No rows constructed. Check JSON contents, paths, and PyTorch mappings.")

    print(f"Successfully converted {len(converted_records)} records")
    print(f"Skipped {skipped_count} records")
    print(f"Resolved/embedded HIP references: {resolved_path_count}")
    print(f"Converted source distribution: {dict(converted_source_counts)}")
    print(f"Skipped source distribution: {dict(skipped_source_counts)}")
    print_length_stats(converted_records)

    df = pd.DataFrame(converted_records)
    df.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")

    print_preview(converted_records, preview_records)

    print("\n" + "=" * 80)
    print(f"{Colors.GREEN}{Colors.BOLD}VERIFICATION: Reading back parquet file{Colors.RESET}")
    print("=" * 80)

    df_verify = pd.read_parquet(output_path)
    print(f"{field('Columns')}: {df_verify.columns.tolist()}")
    print(f"{field('Shape')}: {df_verify.shape}")
    print(f"{field('First row prompt type (pandas)')}: {type(df_verify.iloc[0]['prompt'])}")

    # Keep the builder process side-effect-free after writing. Some environments
    # spawn dataset-loading subprocesses that re-enter the script argv.


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_root = args.reference_root.expanduser()

    source_records = load_source_records(args)
    has_rl_data = any(item["input_format"] == "rl_data" for item in source_records)
    has_kernel2kernel_json = any(
        item["input_format"] == "kernel2kernel_json" for item in source_records
    )

    if has_rl_data and not reference_root.exists():
        raise FileNotFoundError(f"Reference root does not exist: {reference_root}")

    pytorch_root = args.pytorch_root.expanduser() if args.pytorch_root else reference_root
    hip_opt_dir = args.hip_opt_dir.expanduser() if args.hip_opt_dir else pytorch_root / "hip_opt"
    pytorch_mapping: Dict[str, Dict[str, str]] = {}
    kernel_func_to_id: Dict[str, str] = {}
    if has_kernel2kernel_json:
        pytorch_mapping = load_pytorch_code_mapping(pytorch_root)
        kernel_func_to_id = build_kernel_func_to_id_mapping(hip_opt_dir)

    output_name = derive_output_name(args.output_name, source_records)
    optimization_paradigm = infer_optimization_paradigm(args.data_source, args.optimization_paradigm)
    print(f"Using local reference root: {reference_root}")
    print(f"Using PyTorch root: {pytorch_root}")
    print(f"Using output contract: {args.output_contract}")
    print(f"Using optimization paradigm: {optimization_paradigm}")
    print(f"Using output base name: {output_name}")

    for target_gpu in args.target_gpus:
        output_path = build_output_path(
            args.output_dir,
            target_gpu,
            args.output_contract,
            output_name,
        )
        convert_for_target_gpu(
            source_records=source_records,
            reference_root=reference_root,
            target_gpu=target_gpu,
            output_contract=args.output_contract,
            output_path=output_path,
            preview_records=args.preview_records,
            data_source=args.data_source,
            optimization_paradigm=optimization_paradigm,
            pytorch_mapping=pytorch_mapping,
            kernel_func_to_id=kernel_func_to_id,
            pytorch_root=pytorch_root,
            hip_opt_dir=hip_opt_dir,
            max_kernel_input_len=args.max_kernel_input_len,
        )


if __name__ == "__main__":
    main()

