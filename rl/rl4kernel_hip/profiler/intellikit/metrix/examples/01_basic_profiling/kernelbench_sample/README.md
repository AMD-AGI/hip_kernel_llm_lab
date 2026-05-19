# KernelBench Profiling

This directory now supports profiling an arbitrary number of KernelBench HIP
extension samples with the Metrix Python API.

## Reuse strategy

The implementation intentionally reuses existing repo logic instead of inventing a
parallel path:

- `../profile_local.py`: arch resolution and dirty-environment cleanup helpers
- `HIP_benchmark_kit/eval/fix_seed.py`: deterministic seed setup
- `HIP_benchmark_kit/eval/safe_call_helper.py`: `fn=`-aware model invocation helper
- `HIP_benchmark_kit/eval/kernel_loader_template_new.py`: loader contract
  (`torch.utils.cpp_extension.load(...)` with a single `--offload-arch`)
- `HIP_benchmark_kit/eval/eval_hip_kernel_comprehensive.py`: staging logic for
  copying `.hip` sources and optional `include/` content out of the dataset tree

## Files

- `runtime_common.py`: shared helpers for path discovery, arch cleanup, staging,
  metadata collection, and result serialization
- `runner.py`: executes one sample through `torch.utils.cpp_extension.load(...)`
  and `model(..., fn=hip_fn)`
- `profile_kernelbench.py`: canonical entrypoint. It selects samples, precompiles
  them, prewarms them, runs Metrix inventory profiling, reruns with a
  custom-kernel filter, and writes artifacts. It supports arbitrary `N` samples
  plus safe sample-level multi-process parallelism across distinct GPUs.
- `output/`: generated JSON results and Markdown summary

## Default samples

- `8189_matmul_swish_scaling_2d_base.hip`
- `6190_coalesced_memory_access_kernel_base.hip`
- `172_coalesced_tiling_kernel_base.hip`

These cover:

- GEMM plus elementwise post-op
- memory-heavy conv3d fusion
- conv transpose plus custom post-process

## Usage

Run from anywhere:

```bash
python /wekafs/zepingl/rl4kernel_hip/profiler/intellikit/metrix/examples/01_basic_profiling/kernelbench_sample/profile_kernelbench.py
```

If multiple GPUs are visible and `--gpu-id` is not set, the script now
auto-spreads the selected samples across distinct GPUs.

Optional flags:

```bash
python /wekafs/zepingl/rl4kernel_hip/profiler/intellikit/metrix/examples/01_basic_profiling/kernelbench_sample/profile_kernelbench.py \
  --arch gfx942 \
  --gpu-id 0 \
  --prewarm-iters 2 \
  --profile-iters 5 \
  --timeout-seconds 600
```

Metadata collection defaults to safety-first mode:

- `--metadata-mode deferred` (default): planning phase does **not** call
  `get_inputs()`, which avoids materializing giant CPU tensors before compile.
- `--metadata-mode full`: legacy behavior; collects full input/init summaries via
  functional `get_inputs()` and `get_init_inputs()`.

Use `full` only when you explicitly need detailed input summaries and have
enough host memory headroom.

Use an explicit GPU pool for parallel profiling:

```bash
python /wekafs/zepingl/rl4kernel_hip/profiler/intellikit/metrix/examples/01_basic_profiling/kernelbench_sample/profile_kernelbench.py \
  --gpu-ids 0,1,2 \
  --parallel-workers 3
```

Force the old sequential behavior:

```bash
python /wekafs/zepingl/rl4kernel_hip/profiler/intellikit/metrix/examples/01_basic_profiling/kernelbench_sample/profile_kernelbench.py \
  --parallel-workers 1
```

Select arbitrary samples by name:

```bash
python /wekafs/zepingl/rl4kernel_hip/profiler/intellikit/metrix/examples/01_basic_profiling/kernelbench_sample/profile_kernelbench.py \
  --samples 8189_matmul_swish_scaling_2d_base,6190_coalesced_memory_access_kernel_base,172_coalesced_tiling_kernel_base,768_matmul_warp_optimized_edit_1 \
  --gpu-ids 0,1,2 \
  --parallel-workers 3
```

Select samples by regex and cap the count:

```bash
python /wekafs/zepingl/rl4kernel_hip/profiler/intellikit/metrix/examples/01_basic_profiling/kernelbench_sample/profile_kernelbench.py \
  --sample-regex matmul \
  --max-samples 5
```

Profile every dataset sample with a bounded GPU pool:

```bash
python /wekafs/zepingl/rl4kernel_hip/profiler/intellikit/metrix/examples/01_basic_profiling/kernelbench_sample/profile_kernelbench.py \
  --all-samples \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --parallel-workers 8
```

## Why prewarm is outside Metrix

Warmup cannot happen inside the profiled command; otherwise the warmup kernel
launches also appear in the profiling results. This workflow therefore:

1. runs `runner.py` once outside Metrix to build the extension and warm the path,
2. profiles a second invocation with `--warmup-iters 0`,
3. reruns the same command with `kernel_filter` to isolate the custom kernel.

## Three-stage pipeline

The orchestrator now overlaps three stages:

1. `compile`: CPU-side extension build via `runner.py --compile-only`
2. `prewarm`: first GPU-side execution to populate caches and remove cold-start noise
3. `profile`: inventory run plus filtered run under Metrix

This means future samples can keep compiling while current samples are already
being prewarmed and profiled on GPUs, which reduces end-to-end wall time for
larger `N`.

## Parallelism boundary

Only sample-level parallelism across distinct GPUs is safe here.

- Good: sample A on GPU 0, sample B on GPU 1, sample C on GPU 2
- Bad: sample A and sample B profiling concurrently on the same GPU

Same-GPU concurrent profiling would distort both timing and hardware-counter
results, so `profile_kernelbench.py` refuses configurations that would cause that.

Compile-stage parallelism is separate from GPU profiling parallelism. Compiles can
run concurrently on CPU workers, but each GPU still runs at most one active
profiled sample at a time.
