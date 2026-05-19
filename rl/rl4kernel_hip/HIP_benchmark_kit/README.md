# HIP Benchmark Kit

`HIP_benchmark_kit` is the generation, evaluation, orchestration, and reporting
layer for HIP kernel-agent experiments. It no longer owns local kernel
compile/eval logic. HIP compilation, correctness, and performance execution are
delegated to `hip_kernel_evaluation_server/sandbox_core` through in-process calls
only.

## Data Flow

Think of the kit as a thin experiment pipeline:

```text
benchmark HIP + PyTorch refs
        |
        v
stage subset
        |
        v
vLLM generation
        |
        v
generated HIP files + generation_manifest.json
        |
        v
server-inprocess eval
        |
        v
baseline_hip_results.json / origin_vs_optimized_results.json
        |
        v
summary / diagnosis
```

`PyTorch refs` are now mostly interface and metadata inputs. The core
correctness check is `candidate HIP` vs `reference HIP`, not the old path where
subprocesses saved GPU tensors and the parent process loaded them back onto the
wrong GPU.

## Directory Architecture

```text
HIP_benchmark_kit/
  contracts/          # shared manifests, eval schema, and run layouts
  gen_hip_kernel/     # vLLM-backed HIP generation
  eval/               # single-directory eval and origin-vs-optimized merge
  orchestration/      # high-level flows: stage -> generate -> eval -> summarize
  profiling_context/  # profile prompt-map support for B_profile_raw
  reports/            # subset staging, summaries, and reeval diagnosis
  tests/              # unit tests
  docs/               # data and output layout notes
```

Use `contracts/` for shared filenames, row schemas, manifest checks, and layout
helpers. New code should not hand-roll those path or JSON rules.

## Preferred Commands

Run from the repository root:

```bash
python -m HIP_benchmark_kit.orchestration --help
python -m HIP_benchmark_kit.orchestration kernelbench-run --dry_run
python -m HIP_benchmark_kit.orchestration launch-rollouts --dry_run
python -m HIP_benchmark_kit.orchestration neurlps-run --mode smoke --dry_run
python -m HIP_benchmark_kit.orchestration reeval-existing --dry_run
python -m HIP_benchmark_kit.gen_hip_kernel.runner --help
python -m HIP_benchmark_kit.eval.runner --help
```

Model paths are not hard-coded. Pass `--model` explicitly or set
`HIP_KIT_MODEL_SOURCE` for rollout launchers.

## Evaluation Methods

### Baseline Eval

Evaluate one HIP directory for compile/run/match status and HIP performance:

```bash
python -m HIP_benchmark_kit.eval.runner comprehensive \
  --hip_code_dir /path/to/hip_code \
  --reference-hip-code-dir /path/to/reference_hip_code \
  --pytorch_func_dir /path/to/pytorch_code_functional \
  --pytorch_modu_dir /path/to/pytorch_code_module \
  --output_dir /path/to/output_eval \
  --gpu-ids 0 \
  --max-workers 1 \
  --perf-iterations 10 \
  --skip-fix \
  --skip-clear-cache
```

主要输出：

```text
output_eval/
  baseline_hip_results.json
  baseline_hip_results.csv
  error_logs/
```

Key fields include `compile_ok`, `run_ok`, `match_ok`, `pytorch_time_ms`,
`hip_time_ms`, `speedup`, and `error_message`.

### Origin vs Optimized Compare

Compare original HIP kernels with generated kernels:

```bash
python -m HIP_benchmark_kit.eval.runner compare \
  --origin_hip_dir /path/to/origin/hip_code \
  --optimized_hip_dir /path/to/generated \
  --pytorch_func_dir /path/to/pytorch_code_functional \
  --pytorch_modu_dir /path/to/pytorch_code_module \
  --output_dir /path/to/compare_eval \
  --gpu-ids 0,1,2,3 \
  --max-workers 4 \
  --perf-iterations 10 \
  --skip-fix \
  --skip-clear-cache
```

主要输出：

```text
compare_eval/
  origin_eval/baseline_hip_results.json
  optimized_eval/baseline_hip_results.json
  comparison/origin_vs_optimized_results.json
  comparison/origin_vs_optimized_results.csv
  comparison/origin_vs_optimized_perf_trace.csv
  staging/
```

In this mode, `speedup` is the optimized-vs-origin result.

## Full Pipeline

The main KernelBench entry point is:

```bash
python -m HIP_benchmark_kit.orchestration kernelbench-run \
  --model /path/to/model \
  --output_root /path/to/run_root \
  --rollout_n 4 \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --n_gpus 8 \
  --eval_workers 8
```

This stages a subset from `data/hip_eval_neurlps/kernelbench_hip`, runs
`gen_hip_kernel.runner`, evaluates through `eval.runner compare`, and writes the
KernelBench summary through `reports.kernelbench`.

## Optimization Paradigms

Generation is explicit about the code unit the model is expected to optimize:

- `kernel2kernel_splice` is the default and preserves the historical behavior:
  prompt with one extracted HIP kernel/function, parse one optimized function,
  then splice it back into the original `.hip` file.
- `hip2hip_full_file` prompts with the complete `.hip` source file, parses a
  complete replacement translation unit, and writes it directly without kernel
  splicing.

Select the mode with:

```bash
python -m HIP_benchmark_kit.orchestration kernelbench-run \
  --optimization-paradigm hip2hip_full_file \
  --dry_run
```

Evaluation output remains `origin_vs_optimized` for compatibility. In
`hip2hip_full_file`, `origin` is the dataset reference HIP and `optimized` is
the generated full-file HIP candidate.

## Evaluation Contract

The only real backend is `server-inprocess`; `sandbox-inprocess` is accepted as a
compatibility alias. Eval rows keep the existing JSON/CSV field names, including
`compile_cache_*`, but those fields now describe the server reference cache.
Comparison is a pure JSON join and does not run another performance phase.

## Data And Profiling

Benchmark data under `HIP_benchmark_kit/data/` is usually gitignored and must be
materialized separately. See `docs/DATA_AND_OUTPUT_LAYOUT.md` for the canonical
dataset, output, and profiling artifact layout.

Profile mode is optional and isolated behind `--context_mode B_profile_raw`.
Normal generation/eval should use `A_control`.
