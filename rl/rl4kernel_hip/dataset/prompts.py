HIP_CODE_BEGIN_TAG = "```hip"
HIP_CODE_END_TAG = "```"

DEFAULT_TARGET_GPU = "mi300x"
KERNEL2KERNEL_SPLICE_PARADIGM = "kernel2kernel_splice"
HIP2HIP_FULL_FILE_PARADIGM = "hip2hip_full_file"
DEFAULT_OPTIMIZATION_PARADIGM = KERNEL2KERNEL_SPLICE_PARADIGM
SUPPORTED_OPTIMIZATION_PARADIGMS = {
    KERNEL2KERNEL_SPLICE_PARADIGM,
    HIP2HIP_FULL_FILE_PARADIGM,
}

GPU_HARDWARE_CONFIGS = {
    "mi300x": {
        "display_name": "MI300X",
        "architecture": "CDNA3",
        "rocm_target": "gfx942",
        "wavefront_size": 64,
        "lds_per_cu_kib": 64,
        "compute_units": 304,
        "vram_gib": 192,
        "optimization_notes": [
            "MI300X guidance: 64KB LDS per CU is relatively tight, so only use LDS when the data reuse is worth the occupancy cost.",
            "MI300X guidance: with 304 CUs, scalable parallelism and coalesced memory traffic usually matter more than oversized per-block tiles.",
        ],
    },
    "mi325x": {
        "display_name": "MI325X",
        "architecture": "CDNA3",
        "rocm_target": "gfx942",
        "wavefront_size": 64,
        "lds_per_cu_kib": 64,
        "compute_units": 304,
        "vram_gib": 256,
        "optimization_notes": [
            "MI325X guidance: 64KB LDS per CU still requires disciplined shared-memory usage, so prefer LDS only when data reuse clearly pays for the occupancy cost.",
            "MI325X guidance: favor wavefront-friendly execution, coalesced access, and scalable parallelism across the full device.",
        ],
    },
    "mi355x": {
        "display_name": "MI355X",
        "architecture": "CDNA4",
        "rocm_target": "gfx950",
        "wavefront_size": 64,
        "lds_per_cu_kib": 160,
        "compute_units": 256,
        "vram_gib": 288,
        "optimization_notes": [
            "MI355X guidance: 160KB LDS per CU enables more aggressive tiling and buffering, but do not spend LDS so aggressively that occupancy collapses.",
            "MI355X guidance: retune register and LDS trade-offs for this device instead of reusing older tuning assumptions blindly.",
        ],
    },
    "mi450x": {
        "display_name": "MI450X",
        "architecture": "future AMD Instinct target",
        "optimization_notes": [
            "MI450X guidance: this repository does not pin public CU or LDS numbers for MI450X, so avoid hard-coded occupancy assumptions that depend on unverified hardware numbers.",
            "MI450X guidance: prefer portable ROCm-friendly optimizations such as coalesced access, reduced divergence, balanced register pressure, and tunable LDS tiling.",
        ],
    },
}

KERNEL_AGENT_PROMPT_TEMPLATE_BASE = """Please optimize the following HIP kernel/function for better performance on the ROCm platform ({display_name} GPU).
{hardware_prompt}

You are working in think mode.
You will receive only a single kernel/function from the .hip file.
You may only modify the function body, but you must output the entire function including its signature.

Priority order (strict):

1. Do NOT copy the provided starter/reference HIP kernel body.

2. Produce code that compiles in the original file without any other edits.

3. Preserve exact algorithmic behavior and bitwise-equivalent outputs.

4. Improve performance only through safe function-body changes.

Allowed:

Rewrite or optimize the function body only while keeping the exact function name, signature, qualifiers, return type, and parameter types unchanged.

Add local variables, shared memory, unrolling, vectorized I/O, and other body-only optimizations when they are clearly safe.

Reorder code inside the function only when correctness, synchronization, and bitwise equivalence are preserved.

Add brief comments inside the function.

Not Allowed:

Do NOT change the function name.

Do NOT change the function signature, qualifiers, return type, parameter types, or parameter order.

Do NOT add, remove, or modify any code outside this function.

No helper functions, lambdas, templates, macros, new kernels, or host-side code.

No new includes, typedefs, structs, global variables, or launch-configuration changes.

Do NOT assume access to any code, symbols, headers, helper utilities, or constants outside this function.

Do NOT copy the provided starter/reference HIP kernel body verbatim.

Do NOT do near-copy rewrites that only rename variables, reorder obviously independent lines, reformat code, or add/remove comments.

Do NOT return the original/reference body unchanged. Make a real body-level optimization, but keep it conservative and safe.

Compilation and correctness rules:

Use only syntax, types, intrinsics, and symbols that are already available in the existing file/function context.

Do NOT introduce speculative AMD/HIP intrinsics, inline asm, or custom helper abstractions unless they are already clearly supported by the existing code context.

Every identifier you introduce must be defined locally and have the correct type.

Keep all reads and writes in bounds, and preserve guards for edge cases and variable sizes.

Preserve synchronization correctness: never add, remove, or move a barrier in a way that only some threads can reach it.

Do NOT change numeric semantics: no approximate math, no reduced precision, no fast-math assumptions, and no reordered reductions/accumulations unless bitwise equivalence is preserved.

Optimization guidelines (apply only when clearly safe and useful):

Chunked/tiled processing using registers or LDS

Shared-memory buffering (LDS)

Delayed stores to shared memory

Vectorized loads/stores (float2/float4/uint4/etc.)

Loop unrolling

Bound checks for variable sizes

Minimize warp/wavefront divergence

Increase ILP via interleaving independent ops

Reduce LDS/register usage for higher occupancy

Favor coalesced memory and AMD wavefront-friendly access patterns

Fuse operations where possible

Use compiler hints like #pragma unroll

When unsure, make the smallest non-trivial safe optimization instead of a risky rewrite or a copy.

Hard Requirements:

Return the full function, including the exact original function signature.

Only modify code inside the function body.

Output must be a single self-contained function that compiles in the original file without any other edits.

Preserve algorithmic correctness and bitwise-equivalent outputs.

Maintain existing formatting and comments unless improving them.

Before answering, internally check: exact signature preserved, all introduced names defined, braces balanced, no missing semicolons, no out-of-bounds access, and no deadlock/barrier divergence.

Code must be compilable and runnable.
"""


HIP2HIP_PROMPT_TEMPLATE_BASE = """Please optimize the following HIP .hip source file for better performance on the ROCm platform ({display_name} GPU).
{hardware_prompt}

You are working in think mode.
You will receive a complete .hip translation unit, not just one kernel function.
You must output a complete optimized .hip translation unit that can replace the input file.

Priority order (strict):

1. Preserve the external interface expected by the benchmark harness.

2. Produce code that compiles as a standalone replacement for the original .hip file.

3. Preserve algorithmic behavior and benchmark-correct outputs without intentionally changing math semantics or relying on tolerance to hide approximation.

4. Improve performance through conservative HIP-level changes.

Allowed:

Optimize kernel bodies, device helpers, local constants, macros, launch glue, and file-local code when the change is clearly safe.

Add file-local helper functions, templates, constants, or macros only when they are self-contained in the output file and do not require new external dependencies.

Refactor repeated code or specialize existing helpers when doing so preserves the benchmark-visible interface and behavior.

Add specialized fast paths only when every hard-coded shape, dtype, stride, layout, or contiguity assumption is protected by an explicit runtime guard and a correct generic fallback remains available.

Add brief comments for non-obvious optimizations.

Not Allowed:

Do NOT change the benchmark-visible function names, exported entry points, argument order, tensor shapes, dtypes, devices, or expected side effects.

Do NOT change the `torch::Tensor forward(...)` callable contract, the `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` binding, or the exposed `m.def(...)` name used by the benchmark harness.

Do NOT add dependencies on headers, libraries, files, environment variables, or runtime features that are not already available in the benchmark environment.

Do NOT modify PyTorch reference semantics, assume different input distributions, or rely on undefined behavior.

Do NOT introduce randomization, approximate math, reduced precision, fast-math assumptions, or reordered reductions/accumulations unless equivalence is preserved.

Do NOT replace a general implementation with an unguarded benchmark-shape-only implementation.

Do NOT return a near-copy that only renames variables, reformats code, or adds/removes comments.

Compilation and correctness rules:

Keep all reads and writes in bounds, preserve guards for edge cases and variable sizes, and keep synchronization correct.

Every identifier you introduce must be defined in the output file or come from already included HIP/C++ standard headers available to the original file.

If you change launch configuration or shared-memory usage, ensure it remains valid for ROCm and the target GPU.

If you add a shape-specialized kernel or launch path, keep the original generic behavior reachable for all unsupported shapes, dtypes, strides, and layouts.

Optimization guidelines (apply only when clearly safe and useful):

Chunked/tiled processing using registers or LDS

Shared-memory buffering (LDS)

Vectorized loads/stores (float2/float4/uint4/etc.)

Loop unrolling

Minimize warp/wavefront divergence

Increase ILP via interleaving independent ops

Reduce LDS/register usage for higher occupancy

Favor coalesced memory and AMD wavefront-friendly access patterns

Fuse operations where possible

Use compiler hints like #pragma unroll

When unsure, make the smallest non-trivial safe optimization instead of a risky rewrite or a copy.

Hard Requirements:

Return the full .hip source file, including includes, helpers, kernels, and any host/device glue needed by the original file.

The output must be a single complete HIP translation unit with no markdown fences inside the code field.

Preserve the original PyTorch extension interface, including `forward`, pybind registration, return tensor shape, dtype, device, and error-checking behavior.

Preserve benchmark-visible behavior and correctness.

Before answering, internally check: external interface preserved, all introduced names defined, braces balanced, no missing semicolons, no out-of-bounds access, and no deadlock/barrier divergence.

Code must be compilable and runnable.
"""


def normalize_target_gpu(target_gpu: str = DEFAULT_TARGET_GPU) -> str:
    normalized_target_gpu = (target_gpu or DEFAULT_TARGET_GPU).strip().lower()
    if normalized_target_gpu not in GPU_HARDWARE_CONFIGS:
        supported_gpus = ", ".join(sorted(GPU_HARDWARE_CONFIGS))
        raise ValueError(
            f"Unsupported target GPU '{target_gpu}'. Supported values: {supported_gpus}."
        )
    return normalized_target_gpu


def normalize_optimization_paradigm(
    optimization_paradigm: str = DEFAULT_OPTIMIZATION_PARADIGM,
) -> str:
    normalized = (optimization_paradigm or DEFAULT_OPTIMIZATION_PARADIGM).strip().lower()
    aliases = {
        "kernel2kernel": KERNEL2KERNEL_SPLICE_PARADIGM,
        "hip2hip": HIP2HIP_FULL_FILE_PARADIGM,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_OPTIMIZATION_PARADIGMS:
        supported = ", ".join(sorted(SUPPORTED_OPTIMIZATION_PARADIGMS))
        raise ValueError(
            f"Unsupported optimization_paradigm '{optimization_paradigm}'. Supported values: {supported}."
        )
    return normalized


def build_hardware_prompt(target_gpu: str = DEFAULT_TARGET_GPU) -> str:
    profile = GPU_HARDWARE_CONFIGS[normalize_target_gpu(target_gpu)]
    spec_parts = []

    if "rocm_target" in profile:
        spec_parts.append(f"LLVM/HIP target {profile['rocm_target']}")
    if "wavefront_size" in profile:
        spec_parts.append(f"wavefront size {profile['wavefront_size']}")
    if "lds_per_cu_kib" in profile:
        spec_parts.append(f"{profile['lds_per_cu_kib']}KB LDS per Compute Unit (CU)")
    if "compute_units" in profile:
        spec_parts.append(f"{profile['compute_units']} CUs total")
    if "vram_gib" in profile:
        spec_parts.append(f"{profile['vram_gib']}GB HBM")

    lines = [f"Target hardware: {profile['display_name']} ({profile['architecture']})."]
    if spec_parts:
        lines.append("Hardware specs: " + ", ".join(spec_parts) + ".")
    lines.extend(profile["optimization_notes"])
    return "\n".join(lines)


def get_kernel_agent_prompt_template(target_gpu: str = DEFAULT_TARGET_GPU) -> str:
    normalized_target_gpu = normalize_target_gpu(target_gpu)
    profile = GPU_HARDWARE_CONFIGS[normalized_target_gpu]
    return KERNEL_AGENT_PROMPT_TEMPLATE_BASE.format(
        display_name=profile["display_name"],
        hardware_prompt=build_hardware_prompt(normalized_target_gpu),
    )


def get_hip2hip_prompt_template(target_gpu: str = DEFAULT_TARGET_GPU) -> str:
    normalized_target_gpu = normalize_target_gpu(target_gpu)
    profile = GPU_HARDWARE_CONFIGS[normalized_target_gpu]
    return HIP2HIP_PROMPT_TEMPLATE_BASE.format(
        display_name=profile["display_name"],
        hardware_prompt=build_hardware_prompt(normalized_target_gpu),
    )


def get_prompt_template(
    target_gpu: str = DEFAULT_TARGET_GPU,
    optimization_paradigm: str = DEFAULT_OPTIMIZATION_PARADIGM,
) -> str:
    normalized = normalize_optimization_paradigm(optimization_paradigm)
    if normalized == HIP2HIP_FULL_FILE_PARADIGM:
        return get_hip2hip_prompt_template(target_gpu)
    return get_kernel_agent_prompt_template(target_gpu)


KERNEL_AGENT_PROMPT_TEMPLATES = {
    target_gpu: get_kernel_agent_prompt_template(target_gpu)
    for target_gpu in GPU_HARDWARE_CONFIGS
}
HIP2HIP_PROMPT_TEMPLATES = {
    target_gpu: get_hip2hip_prompt_template(target_gpu)
    for target_gpu in GPU_HARDWARE_CONFIGS
}

MI300X_KERNEL_AGENT_PROMPT_TEMPLATE = KERNEL_AGENT_PROMPT_TEMPLATES["mi300x"]
MI325X_KERNEL_AGENT_PROMPT_TEMPLATE = KERNEL_AGENT_PROMPT_TEMPLATES["mi325x"]
MI355X_KERNEL_AGENT_PROMPT_TEMPLATE = KERNEL_AGENT_PROMPT_TEMPLATES["mi355x"]
MI450X_KERNEL_AGENT_PROMPT_TEMPLATE = KERNEL_AGENT_PROMPT_TEMPLATES["mi450x"]

DEFAULT_KERNEL_AGENT_PROMPT_TEMPLATE = KERNEL_AGENT_PROMPT_TEMPLATES[DEFAULT_TARGET_GPU]

HIP_REASONING_AND_CODE_FORMAT_WITH_STARTER_CODE = (
    "First provide optimization reasoning naturally. Then output exactly one HIP code block "
    "using triple backticks in this format:\n"
    "```hip\n"
    "// optimized HIP kernel/function code\n"
    "```\n"
    "Use the starter code as the baseline, keep the function name and signature unchanged, "
    "and only optimize code inside the function body."
)

HIP_REASONING_AND_CODE_FORMAT_WITHOUT_STARTER_CODE = (
    "First provide optimization reasoning naturally. Then output exactly one HIP code block "
    "using triple backticks in this format:\n"
    "```hip\n"
    "// optimized HIP kernel/function code\n"
    "```\n"
    "Do not add extra prose after the closing code fence."
)

HIP_LEGACY_CODE_FORMAT_WITH_STARTER_CODE = (
    "Output exactly one HIP code block using triple backticks in this format:\n"
    "```hip\n"
    "// optimized HIP kernel/function code\n"
    "```\n"
    "The output must not include reasoning, JSON, or any prose outside the code block. "
    "Use the starter code as the baseline, keep the function name and signature unchanged, "
    "and only optimize code inside the function body."
)

HIP_LEGACY_CODE_FORMAT_WITHOUT_STARTER_CODE = (
    "Output exactly one HIP code block using triple backticks in this format:\n"
    "```hip\n"
    "// optimized HIP kernel/function code\n"
    "```\n"
    "The output must not include reasoning, JSON, or any prose outside the code block."
)

HIP_REASONING_AND_JSON_RESPONSE_FORMAT_WITH_STARTER_CODE = (
    "After reasoning, output exactly one JSON object in this format:\n"
    '{"thought": "concise optimization summary", "code": "__global__ void ..."}\n'
    "The `code` field must contain the full optimized HIP kernel/function, including the exact original signature, "
    "with no surrounding markdown fences. Use the starter code as the baseline, keep the function name and "
    "signature unchanged, and only optimize code inside the function body."
)

HIP_REASONING_AND_JSON_RESPONSE_FORMAT_WITHOUT_STARTER_CODE = (
    "After reasoning, output exactly one JSON object in this format:\n"
    '{"thought": "concise optimization summary", "code": "__global__ void ..."}\n'
    "The `code` field must contain the full optimized HIP kernel/function with no surrounding markdown fences. "
    "Do not add extra prose after the closing `}`."
)

HIP_FULL_FILE_REASONING_AND_CODE_FORMAT_WITH_STARTER_CODE = (
    "First provide optimization reasoning naturally. Then output exactly one HIP code block "
    "using triple backticks in this format:\n"
    "```hip\n"
    "// complete optimized .hip source file\n"
    "```\n"
    "Use the starter HIP file as the baseline and return a complete replacement .hip file."
)

HIP_FULL_FILE_REASONING_AND_CODE_FORMAT_WITHOUT_STARTER_CODE = (
    "First provide optimization reasoning naturally. Then output exactly one HIP code block "
    "using triple backticks in this format:\n"
    "```hip\n"
    "// complete optimized .hip source file\n"
    "```\n"
    "Do not add extra prose after the closing code fence."
)

HIP_FULL_FILE_LEGACY_CODE_FORMAT_WITH_STARTER_CODE = (
    "Output exactly one HIP code block using triple backticks in this format:\n"
    "```hip\n"
    "// complete optimized .hip source file\n"
    "```\n"
    "The output must not include reasoning, JSON, or any prose outside the code block. "
    "Use the starter HIP file as the baseline and return a complete replacement .hip file."
)

HIP_FULL_FILE_LEGACY_CODE_FORMAT_WITHOUT_STARTER_CODE = (
    "Output exactly one HIP code block using triple backticks in this format:\n"
    "```hip\n"
    "// complete optimized .hip source file\n"
    "```\n"
    "The output must not include reasoning, JSON, or any prose outside the code block."
)

HIP_FULL_FILE_REASONING_AND_JSON_RESPONSE_FORMAT_WITH_STARTER_CODE = (
    "After reasoning, output exactly one JSON object in this format:\n"
    '{"thought": "concise optimization summary", "code": "#include <hip/hip_runtime.h>\\n..."}\n'
    "The `code` field must contain the complete optimized .hip source file with no surrounding markdown fences. "
    "Use the starter HIP file as the baseline and return a complete replacement .hip file."
)

HIP_FULL_FILE_REASONING_AND_JSON_RESPONSE_FORMAT_WITHOUT_STARTER_CODE = (
    "After reasoning, output exactly one JSON object in this format:\n"
    '{"thought": "concise optimization summary", "code": "#include <hip/hip_runtime.h>\\n..."}\n'
    "The `code` field must contain the complete optimized .hip source file with no surrounding markdown fences. "
    "Do not add extra prose after the closing `}`."
)