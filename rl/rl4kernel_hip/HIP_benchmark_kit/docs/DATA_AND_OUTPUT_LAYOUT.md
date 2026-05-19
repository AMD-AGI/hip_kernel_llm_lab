# Data And Output Layout

## Frozen Benchmark Assets

Benchmark inputs live under `HIP_benchmark_kit/data/`. Treat these files as
frozen assets: do not delete or rewrite benchmark content as part of run
orchestration. Scripts may stage subsets into output directories, but the source
trees remain read-only inputs.

Expected dataset shape for HIP-to-HIP eval:

```text
<dataset-root>/
  level-1/
    hip_code/
    pytorch_code_functional/
    pytorch_code_module/
  level-2/
  level-3/
```

For NeurIPS datasets, the same `hip_code`, `pytorch_code_functional`, and
`pytorch_code_module` directories are expected under each dataset root.
Large benchmark mirrors such as `HIP_benchmark_kit/data/hip_eval_neurlps/` are
gitignored local assets. A clean checkout needs those assets restored from the
external benchmark bundle before orchestration can run.

`kernelbench_hip/` may contain both `level-*` directories and an `all/` rollup.
The orchestration subset step reads the `level-*` trees so level quotas remain
explicit and reproducible.

## Canonical Run Output

Runtime outputs should live under `outputs/HIP_benchmark_kit/`, not inside the
package. KernelBench rollout runs use this layout:

```text
<run-root>/
  subset/
    kernelbench_hip_100/
    subset_manifest.json
  level-1/
    generated/
      generation_manifest.json
    eval/
      origin_eval/
        baseline_hip_results.json
      optimized_eval/
        baseline_hip_results.json
      comparison/
        origin_vs_optimized_results.json
        origin_vs_optimized_results.csv
        origin_vs_optimized_perf_trace.csv
      staging/
  level-2/
  level-3/
  summary/
```

Generation reuse and eval reuse must compare manifest identity fields before
copying or reusing records. The rollout-16-from-rollout-4 policy is implemented
by orchestration, not by ad hoc launcher string handling.

`generation_manifest.json` records the `optimization_paradigm` in both the
top-level manifest and the generation identity. Reuse must not cross paradigms:
`kernel2kernel_splice` outputs are kernel-function splice artifacts, while
`hip2hip_full_file` outputs are complete `.hip` replacement files.

The eval layout intentionally remains unchanged for both paradigms. In
`hip2hip_full_file`, `origin_eval` is still the dataset reference HIP side and
`optimized_eval` is the generated full-file candidate side.

## Profiling Context

Profile mode is optional and isolated behind `--context_mode B_profile_raw`.
Tracked package code under `HIP_benchmark_kit/profiling_context/` validates
Metrix artifacts, can call an explicit external `--profile_script` to produce
missing artifacts, and builds per-task prompt maps. Profile artifacts are
expected under an explicit `--profile_artifact_root`; prompt maps and context
manifests are written under `--profile_prompt_root`.

Normal generation/eval should use `--context_mode A_control` and must not depend
on profiling assets.

## Ignore Policy

Keep source code, docs, and small schemas tracked. Generated run outputs, model
caches, profiles, logs, core dumps, runtime work directories, and large
benchmark data mirrors should stay ignored.
