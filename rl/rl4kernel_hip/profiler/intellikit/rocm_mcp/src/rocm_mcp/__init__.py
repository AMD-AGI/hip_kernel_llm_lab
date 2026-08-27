# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""rocm-mcp - ROCm-based tools for use by LLM agents."""

from ._version import __version__
from .compile import HipCompiler, HipCompilerResult
from .doc import HipDocs
from .sysinfo import AgentInfo, DeviceType, Rocminfo, RocminfoResult

__all__ = [
    "AgentInfo",
    "DeviceType",
    "HipCompiler",
    "HipCompilerResult",
    "HipDocs",
    "Rocminfo",
    "RocminfoResult",
    "__version__",
]
