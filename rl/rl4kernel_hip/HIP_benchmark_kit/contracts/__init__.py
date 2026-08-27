# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared contracts for HIP_benchmark_kit."""

from .eval_schema import (
    BASELINE_RESULTS_JSON,
    COMPARISON_RESULTS_JSON,
    LEGACY_EVAL_REQUIRED_FIELDS,
    SUPPORTED_REFERENCE_CACHE_MODES,
    SERVER_INPROCESS_BACKEND,
    normalize_eval_backend,
    validate_eval_records,
)
from .layout import (
    DEFAULT_LEVELS,
    KERNELBENCH_SUBSET_NAME,
    KernelBenchRunLayout,
    LevelRunLayout,
    repo_root,
)
from .manifests import (
    GENERATION_MANIFEST,
    SUBSET_MANIFEST,
    read_json,
    validate_generation_manifest,
    validate_subset_manifest,
    write_json,
)

__all__ = [
    "BASELINE_RESULTS_JSON",
    "COMPARISON_RESULTS_JSON",
    "DEFAULT_LEVELS",
    "GENERATION_MANIFEST",
    "KERNELBENCH_SUBSET_NAME",
    "KernelBenchRunLayout",
    "LEGACY_EVAL_REQUIRED_FIELDS",
    "LevelRunLayout",
    "SERVER_INPROCESS_BACKEND",
    "SUBSET_MANIFEST",
    "SUPPORTED_REFERENCE_CACHE_MODES",
    "normalize_eval_backend",
    "read_json",
    "repo_root",
    "validate_eval_records",
    "validate_generation_manifest",
    "validate_subset_manifest",
    "write_json",
]
