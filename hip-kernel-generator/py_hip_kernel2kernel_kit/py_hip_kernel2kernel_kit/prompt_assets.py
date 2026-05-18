from __future__ import annotations

import importlib.util
from pathlib import Path

SYSTEM_INSTRUCTION = """
You are optimizing a single HIP GPU function extracted from a larger HIP source file.

Requirements:
- Return exactly one complete `__global__` or `__device__` function definition.
- Preserve the target function's name, qualifiers, return type, and parameter interface.
- Do not emit host-side wrappers, includes, launcher code, or `PYBIND11_MODULE`.
- The optimized function body will be transplanted back into the original HIP file.
- Preserve numerical correctness before pursuing speed.
- Favor ROCm-safe optimizations such as coalesced memory access, shared-memory tiling, register reuse, loop unrolling, reduced divergence, and avoiding redundant global loads.
- If prior attempts failed, fix the exact issue shown in the verifier feedback.
- If prior attempts were correct but slower, keep the validated semantics and improve performance against the baseline HIP kernel.
- Output code only. No prose, no markdown explanation outside the function itself.
""".strip()

FEW_SHOT_EXAMPLES = """
### Example 1

Target GPU function:

```cpp
__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
```

Optimized response:

```cpp
__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    #pragma unroll
    for (int offset = 0; offset < 4; ++offset) {
        int idx = base + offset;
        if (idx < n) {
            out[idx] = a[idx] + b[idx];
        }
    }
}
```

### Example 2

Target GPU function:

```cpp
__device__ float dot_row(const float* row, const float* col, int n) {
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
        acc += row[k] * col[k];
    }
    return acc;
}
```

Optimized response:

```cpp
__device__ float dot_row(const float* row, const float* col, int n) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    int k = 0;
    for (; k + 1 < n; k += 2) {
        acc0 += row[k] * col[k];
        acc1 += row[k + 1] * col[k + 1];
    }
    for (; k < n; ++k) {
        acc0 += row[k] * col[k];
    }
    return acc0 + acc1;
}
```
""".strip()

DEFAULT_SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION
DEFAULT_FEW_SHOT_EXAMPLES = FEW_SHOT_EXAMPLES


def _load_symbol_from_python_file(file_path: Path, symbol_name: str) -> str | None:
    if not file_path.exists():
        return None

    module_name = f"py_hip_kernel2kernel_prompt_assets_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, symbol_name, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def resolve_prompt_assets(
    instruction_file: Path | None = None,
    few_shot_file: Path | None = None,
) -> tuple[str, str]:
    system_instruction = (
        _load_symbol_from_python_file(instruction_file, "hip_kernel_opt_req")
        if instruction_file is not None
        else None
    )
    few_shot_examples = (
        _load_symbol_from_python_file(few_shot_file, "few_shot_code_instructions")
        if few_shot_file is not None
        else None
    )
    return (
        system_instruction or SYSTEM_INSTRUCTION,
        few_shot_examples or FEW_SHOT_EXAMPLES,
    )
