#!/usr/bin/env python3
"""Pretty-print the content of each message in sample.json data."""

import json
import textwrap

# ANSI escape codes
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"

W = 110
THIN  = "─" * W
THICK = "━" * W

ROLE_STYLE = {
    "system":    (C.RED,   "SYSTEM"),
    "user":      (C.GREEN, "USER"),
    "assistant": (C.BLUE,  "ASSISTANT"),
}

DATA = r'''
{"messages": [{"role": "user", "content": "Please optimize the following HIP kernel/function for better performance on the ROCm platform (MI250 GPU).\nMI250 specs: 208KB LDS per Compute Unit (CU), 64 CUs total.\n\nYou will receive only a single kernel/function from the .hip file.\nYou may only modify the function body, but you must output the entire function including its signature.\n\nAllowed:\n\nRewrite or optimize the function body only.\n\nAdd local variables, shared memory, unrolling, vectorized I/O, etc.\n\nReorder code inside the function.\n\nAdd comments inside the function.\n\nNot Allowed:\n\nDo NOT change the function name.\n\nDo NOT change the function signature or parameter types.\n\nDo NOT add, remove, or modify any code outside this function.\n\nNo helper functions\n\nNo new includes\n\nNo new kernels\n\nNo changes to launch configuration\n\nDo NOT assume access to any code outside this function.\n\nOptimization guidelines (apply those that fit):\n\nChunked/tiled processing using registers or LDS\n\nShared-memory buffering (LDS)\n\nDelayed stores to shared memory\n\nVectorized loads/stores (float2/float4/uint4/etc.)\n\nLoop unrolling\n\nBound checks for variable sizes\n\nMinimize warp/wavefront divergence\n\nIncrease ILP via interleaving independent ops\n\nReduce LDS/register usage for higher occupancy\n\nFavor coalesced memory and AMD wavefront-friendly access patterns\n\nFuse operations where possible\n\nUse compiler hints like #pragma unroll\n\nHard Requirements:\n\nReturn the full function, including the exact original function signature.\n\nOnly modify code inside the function body.\n\nPreserve algorithmic correctness and bitwise-equivalent outputs.\n\nMaintains existing formatting and comments unless improving them.\n\nCode must be compilable and runnable.\nHere is an example snippet of baseline code: __global__ void fused_cross_attn_len1_kernel(const float* __restrict__ Q,\n                                             const float* __restrict__ K,\n                                             const float* __restrict__ V,\n                                             float* __restrict__ Out,\n                                             int B, int C, int Bk, int broadcast_kv,\n                                             float scale) {\n    int b = blockIdx.x;\n    if (b >= B) return;\n    int tid = threadIdx.x;\n    int T = blockDim.x;\n\n    extern __shared__ float smem[]; // size >= C (for logits/probs)\n    float* probs = smem; // reuse as logits first, then probs\n\n    // Pointers for this batch (or broadcast 0)\n    int kb = broadcast_kv ? 0 : b;\n    const float* q = Q + b * C;\n    const float* Kb = K + kb * C * C;\n    const float* Vb = V + kb * C * C;\n\n    // 1) compute logits[j] = dot(q, K[j,:]) * scale\n    for (int j = tid; j < C; j += T) {\n        const float* Kj = Kb + j * C;\n        float sum = 0.0f;\n        // dot over c\n        for (int c = 0; c < C; ++c) {\n            sum += q[c] * Kj[c];\n        }\n        probs[j] = sum * scale; // store logits\n    }\n    __syncthreads();\n\n    // 2) reduce max\n    float local_max = -INFINITY;\n    for (int j = tid; j < C; j += T) {\n        local_max = fmaxf(local_max, probs[j]);\n    }\n    // reduce within block via shared memory\n    extern __shared__ float sreduce[];\n    float* smax = sreduce; // alias to probs buffer memory after use\n    // place thread local_max into smax[tid]\n    smax[tid] = local_max;\n    __syncthreads();\n    for (int s = T >> 1; s > 0; s >>= 1) {\n        if (tid < s) smax[tid] = fmaxf(smax[tid], smax[tid + s]);\n        __syncthreads();\n    }\n    float row_max = smax[0];\n    __syncthreads();\n\n    // 3) compute exp and sum\n    float local_sum = 0.0f;\n    for (int j = tid; j < C; j += T) {\n        float e = expf(probs[j] - row_max);\n        probs[j] = e; // store numerator\n        local_sum += e;\n    }\n    smax[tid] = local_sum; // reuse smax for sum reduce\n    __syncthreads();\n    for (int s = T >> 1; s > 0; s >>= 1) {\n        if (tid < s) smax[tid] += smax[tid + s];\n        __syncthreads();\n    }\n    float denom = smax[0] + 1e-12f;\n    float inv_denom = 1.0f / denom;\n    __syncthreads();\n\n    // 4) normalize to probs\n    for (int j = tid; j < C; j += T) {\n        probs[j] *= inv_denom;\n    }\n    __syncthreads();\n\n    // 5) out[c_out] = sum_j probs[j] * V[j, c_out]\n    for (int c_out = tid; c_out < C; c_out += T) {\n        float acc = 0.0f;\n        for (int j = 0; j < C; ++j) {\n            acc += probs[j] * Vb[j * C + c_out];\n        }\n        Out[b * C + c_out] = acc;\n    }\n}\nOutput your answer in json format, with the format as follows: {\"thought\": \"\", \"code\": \"\"}. Please strictly output in JSON format.\nGenerate the correct and optimized code without explanation, which we can run directly in the \"code\" field."}, {"role": "assistant", "content": "<think>\nWe need to optimize kernel. Provide full function with same signature. Use shared memory tiling for dot product and for attention compute. Use vector loads float4 perhaps. Use loop unrolling. Use reduction via warp-level primitives (but HIP may have __shfl_down). Could use __builtin_amdgcn_*? Simpler: use block reduction with shared memory but reduce syncs. Also combine max and sum reductions in one pass maybe.\n\nWe must keep algorithm same. We'll add register tiling: compute dot product using loop over C with vectorized loads of float4 assuming C multiple of 4, else handle remainder.\n\nAlso use __syncthreads only where needed.\n\nLDS size: need C floats for probs, plus maybe reduction buffer of same size; but we can reuse same buffer and do reduction in-place using warp reduction without extra buffer.\n\nWe'll allocate shared memory as float *smem = extern __shared__ float[]; use it for probs. For reduction, we can use same smem as temporary per warp reduction using warp shuffle then store partials.\n\nSimplify: use block reduction using shared memory of size blockDim.x, which is <= 256? Not given. Use same smem for probs; after computing probs we have them in smem[0..C-1]. For reduction we can allocate separate small buffer at tail: e.g., float* sdata = smem + C; assuming shared memory size allocated enough (blockDim.x). We'll assume.\n\nImplement warp-level max reduction using __shfl_down_sync (HIP uses __shfl_down). Use mask = 0xffffffff.\n\nWe'll need to include appropriate intrinsics: __shfl_down.\n\nI'll write:\n\nint lane = tid % 64; // wavefront size 64 for AMD.\n\nUse __shfl_down.\n\nImplementation steps:\n\n1. Compute logits: vectorized dot product: for (int c=0; c+3<C; c+=4) { float4 qv = *((float4*)(q+c)); float4 kv = *((float4*)(Kj+c)); sum += qv.x*kv.x + qv.y*kv.y + qv.z*kv.z + qv.w*kv.w; } handle remainder.\n\n2. Store logits*scale into probs[j].\n\n3. Compute max: each thread computes local max over its subset, then warp reduce, then block reduce using shared mem.\n\n4. Compute exp and sum similarly.\n\n5. Normalize.\n\n6. Compute output: use vector loads from V and probs broadcast via registers? Could load probs into shared memory already.\n\nWe can also compute output using loop over j in chunks using shared memory for probs and V tile to improve locality. Since V accessed column-wise (j, c_out) which is strided by C, not coalesced. But we can loop over j and each thread loads V element for its output index. That's fine.\n\nWe'll keep similar.\n\nAdd #pragma unroll for loops.\n\nNow provide code.\n\nNeed to ensure shared memory size: we need C floats for probs + blockDim.x floats for reduction. Launch config not changed, but assume enough shared memory allocated (maybe they allocate C+blockDim.x). We'll comment.\n\nNow produce JSON with thought and code.\n</think>\n\n{\n  \"thought\": \"Optimized the attention kernel by vectorizing the dot\u2011product, using warp\u2011level reductions with __shfl_down, minimizing shared\u2011memory usage, unrolling loops, and reusing the shared buffer for reductions. The algorithmic behavior is unchanged.\",\n  \"code\": \"__global__ void fused_cross_attn_len1_kernel(const float* __restrict__ Q,\\n                                             const float* __restrict__ K,\\n                                             const float* __restrict__ V,\\n                                             float* __restrict__ Out,\\n                                             int B, int C, int Bk, int broadcast_kv,\\n                                             float scale) {\\n    int b = blockIdx.x;\\n    if (b >= B) return;\\n    int tid = threadIdx.x;                // thread id within the block\\n    int T   = blockDim.x;                 // block size (wavefront multiple)\\n\\n    // Shared memory layout:\\n    //   smem[0 .. C-1]          -> probs / logits (size C)\\n    //   smem[C .. C+T-1]        -> temporary reduction buffer (size T)\\n    extern __shared__ float smem[];\\n    float* probs = smem;                 // alias for logits/probabilities\\n    float* sreduce = smem + C;           // temporary buffer for block reductions\\n\\n    // Pointers for this batch (or broadcast 0)\\n    int kb = broadcast_kv ? 0 : b;\\n    const float* q  = Q + b * C;                     // [C]\\n    const float* Kb = K + kb * C * C;               // [C][C]\\n    const float* Vb = V + kb * C * C;               // [C][C]\\n\\n    // ---------------------------------------------------------------------\\n    // 1) Compute logits[j] = dot(q, K[j,:]) * scale\\n    // ---------------------------------------------------------------------\\n    for (int j = tid; j < C; j += T) {\\n        const float* Kj = Kb + j * C;               // row j of K\\n        float sum = 0.0f;\\n        // vectorized dot product (process 4 floats per iteration)\\n        int c = 0;\\n        #pragma unroll 4\\n        for (; c + 3 < C; c += 4) {\\n            float4 qv = *((const float4*)(q + c));\\n            float4 kv = *((const float4*)(Kj + c));\\n            sum += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;\\n        }\\n        // tail handling\\n        for (; c < C; ++c) {\\n            sum += q[c] * Kj[c];\\n        }\\n        probs[j] = sum * scale;   // store logits in shared memory\\n    }\\n    __syncthreads();\\n\\n    // ---------------------------------------------------------------------\\n    // 2) Reduce to find maximum value (row_max)\\n    // ---------------------------------------------------------------------\\n    float local_max = -INFINITY;\\n    for (int j = tid; j < C; j += T) {\\n        local_max = fmaxf(local_max, probs[j]);\\n    }\\n    // warp\u2011level reduction (wavefront = 64 on AMD)\\n    #pragma unroll\\n    for (int offset = 32; offset > 0; offset >>= 1) {\\n        float other = __shfl_down(local_max, offset);\\n        local_max = fmaxf(local_max, other);\\n    }\\n    // first lane of each wavefront writes its result to shared memory\\n    if ((tid & 63) == 0) sreduce[tid >> 6] = local_max;   // one entry per wavefront\\n    __syncthreads();\\n    // now reduce the partials (max) using a single warp\\n    if (tid < (T + 63) / 64) {\\n        float warp_max = sreduce[tid];\\n        #pragma unroll\\n        for (int offset = 32; offset > 0; offset >>= 1) {\\n            float other = __shfl_down(warp_max, offset);\\n            warp_max = fmaxf(warp_max, other);\\n        }\\n        if (tid == 0) sreduce[0] = warp_max; // final max stored at sreduce[0]\\n    }\\n    __syncthreads();\\n    float row_max = sreduce[0];\\n\\n    // ---------------------------------------------------------------------\\n    // 3) Compute exp(logits - max) and sum for softmax denominator\\n    // ---------------------------------------------------------------------\\n    float local_sum = 0.0f;\\n    for (int j = tid; j < C; j += T) {\\n        float e = expf(probs[j] - row_max);\\n        probs[j] = e;               // overwrite logits with numerator\\n        local_sum += e;\\n    }\\n    // warp\u2011level sum reduction\\n    #pragma unroll\\n    for (int offset = 32; offset > 0; offset >>= 1) {\\n        local_sum += __shfl_down(local_sum, offset);\\n    }\\n    if ((tid & 63) == 0) sreduce[tid >> 6] = local_sum;\\n    __syncthreads();\\n    if (tid < (T + 63) / 64) {\\n        float warp_sum = sreduce[tid];\\n        #pragma unroll\\n        for (int offset = 32; offset > 0; offset >>= 1) {\\n            warp_sum += __shfl_down(warp_sum, offset);\\n        }\\n        if (tid == 0) sreduce[0] = warp_sum;\\n    }\\n    __syncthreads();\\n    float denom = sreduce[0] + 1e-12f;\\n    float inv_denom = 1.0f / denom;\\n\\n    // ---------------------------------------------------------------------\\n    // 4) Normalize probabilities\\n    // ---------------------------------------------------------------------\\n    for (int j = tid; j < C; j += T) {\\n        probs[j] *= inv_denom;\\n    }\\n    __syncthreads();\\n\\n    // ---------------------------------------------------------------------\\n    // 5) Compute output vector: Out[b, c_out] = \u03a3_j probs[j] * V[j, c_out]\\n    // ---------------------------------------------------------------------\\n    for (int c_out = tid; c_out < C; c_out += T) {\\n        float acc = 0.0f;\\n        // accumulate over j in chunks of 4 for better memory bandwidth\\n        int j = 0;\\n        #pragma unroll 4\\n        for (; j + 3 < C; j += 4) {\\n            float4 p4 = make_float4(probs[j], probs[j+1], probs[j+2], probs[j+3]);\\n            float4 v0 = *((const float4*)(Vb + (j    ) * C + c_out));\\n            float4 v1 = *((const float4*)(Vb + (j+1) * C + c_out));\\n            float4 v2 = *((const float4*)(Vb + (j+2) * C + c_out));\\n            float4 v3 = *((const float4*)(Vb + (j+3) * C + c_out));\\n            acc += p4.x * v0.x + p4.y * v1.x + p4.z * v2.x + p4.w * v3.x;\\n        }\\n        for (; j < C; ++j) {\\n            acc += probs[j] * Vb[j * C + c_out];\\n        }\\n        Out[b * C + c_out] = acc;\\n    }\\n}\\n\"\n}"}], "metadata": {"kernel_name": "fused_cross_attn_len1_kernel", "index": "geak:19_2", "n_iter": 0, "speedup": 1.1230656614876646, "pass_exe": true, "pass_perf": true}}
'''.strip()


def fmt_code_block(code: str, lang: str = "", max_lines: int = 60) -> str:
    lines = code.strip().split("\n")
    truncated = len(lines) > max_lines
    display = lines[:max_lines] if truncated else lines

    header = f"  {C.DIM}┌{'─' * (W - 4)} {lang}{C.RESET}"
    body = "\n".join(f"  {C.DIM}│{C.RESET} {l}" for l in display)
    footer = f"  {C.DIM}└{'─' * (W - 4)}{C.RESET}"
    if truncated:
        footer = f"  {C.DIM}│ ... ({len(lines) - max_lines} more lines){C.RESET}\n" + footer
    return f"{header}\n{body}\n{footer}"


def print_content(content: str, role_color: str):
    segments = content.split("```")
    for i, seg in enumerate(segments):
        if i % 2 == 0:
            text = seg.strip()
            if not text:
                continue
            for para in text.split("\n\n"):
                for line in para.split("\n"):
                    wrapped = textwrap.fill(
                        line, width=W - 6,
                        initial_indent="    ", subsequent_indent="    ",
                    )
                    print(wrapped)
                print()
        else:
            code_lines = seg.split("\n")
            lang = code_lines[0].strip() if code_lines else ""
            code_body = "\n".join(code_lines[1:]) if len(code_lines) > 1 else ""
            if not code_body.strip():
                code_body, lang = lang, ""
            print(fmt_code_block(code_body, lang=lang))
            print()


def print_json_block(obj, indent=4):
    """Pretty-print a parsed JSON object (e.g. the assistant's JSON response)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            prefix = " " * indent
            if isinstance(v, str) and len(v) > 200:
                label = "code" if ("void " in v or "#include" in v) else "text"
                print(f"{prefix}{C.CYAN}{k}{C.RESET}: {C.DIM}[{label}, {len(v)} chars]{C.RESET}")
                lang = "hip" if ("__global__" in v or "#include" in v) else ""
                actual = v.replace("\\n", "\n").replace("\\t", "\t")
                print(fmt_code_block(actual, lang=lang))
            elif isinstance(v, str):
                print(f"{prefix}{C.CYAN}{k}{C.RESET}: {v}")
            elif isinstance(v, dict):
                print(f"{prefix}{C.CYAN}{k}{C.RESET}:")
                print_json_block(v, indent + 4)
            else:
                print(f"{prefix}{C.CYAN}{k}{C.RESET}: {v}")
    else:
        print(" " * indent + str(obj))


def main():
    record = json.loads(DATA)
    messages = record["messages"]
    metadata = record.get("metadata", {})

    print(f"\n{C.BOLD}{THICK}{C.RESET}")
    print(f"  {C.BOLD}{C.MAGENTA}Sample Record Pretty Printer{C.RESET}")
    print(f"{C.BOLD}{THICK}{C.RESET}\n")

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        color, label = ROLE_STYLE.get(role, (C.YELLOW, role.upper()))

        print(f"  {color}{C.BOLD}{'━' * 3} [{label}] {'━' * (W - len(label) - 8)}{C.RESET}")

        # If content looks like a JSON object, parse & pretty-print it
        try:
            parsed = json.loads(content.strip())
            if isinstance(parsed, dict):
                print_json_block(parsed)
                print()
                continue
        except (json.JSONDecodeError, ValueError):
            pass

        # For assistant responses, try to separate <think>...</think> and JSON body
        if role == "assistant" and "<think>" in content:
            think_start = content.index("<think>") + len("<think>")
            think_end = content.index("</think>")
            think_text = content[think_start:think_end].strip()
            rest = content[think_end + len("</think>"):].strip()

            print(f"\n    {C.DIM}{C.BOLD}[Thinking]{C.RESET}")
            for line in think_text.split("\n"):
                wrapped = textwrap.fill(
                    line, width=W - 8,
                    initial_indent=f"    {C.DIM}", subsequent_indent=f"    {C.DIM}",
                )
                print(f"{wrapped}{C.RESET}")
            print()

            if rest:
                print(f"    {C.BOLD}[Response]{C.RESET}")
                try:
                    parsed = json.loads(rest)
                    if isinstance(parsed, dict):
                        print_json_block(parsed)
                    else:
                        print(f"    {rest[:500]}")
                except (json.JSONDecodeError, ValueError):
                    print_content(rest, color)
            print()
        else:
            print_content(content, color)

    if metadata:
        print(f"  {C.YELLOW}{C.BOLD}{'━' * 3} [METADATA] {'━' * (W - 16)}{C.RESET}")
        print_json_block(metadata)
        print()

    print(f"{C.BOLD}{THICK}{C.RESET}\n")


if __name__ == "__main__":
    main()
