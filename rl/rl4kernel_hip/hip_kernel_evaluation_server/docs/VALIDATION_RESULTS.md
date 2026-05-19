# Validation Results

This note records the real smoke and throughput validation run performed after the sandbox refactor.

## Runtime Setup

- Worker (cache off): `http://127.0.0.1:18082`
- Worker (cache on): `http://127.0.0.1:18083`
- Master (remote-only -> cache-on worker): `http://127.0.0.1:18081`
- Baseline worker from `HEAD` worktree: `http://127.0.0.1:18084`

Workload:

- HIP code: `HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_gpumode_50_tasks/hip_code/hip_90_L1.hip`
- Functional driver: `HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_gpumode_50_tasks/pytorch_code_functional/py_90_L1.py`

## Real Single-Node Cache Smoke

Path used:

- `POST /run_code_single_gpu` on `18083`
- fixed `gpu_id=0`

Observed results:

| Run | compile_ok | run_ok | match_ok | golden_hit | perf_hit | total_s |
|---|---|---|---|---|---|---|
| first | true | true | true | false | false | 39.3617 |
| second | true | true | true | true | true | 19.7576 |

Interpretation:

- `reference_golden` hit path is working.
- `reference_perf` hit path is working when the request is pinned to the same runtime identity (`gpu_id=0`).

## Real Master -> Remote Worker Smoke

Path used:

- `POST /run_code_batch` on `18081`
- master configured with no local GPUs and one remote worker at `127.0.0.1:18083`

Observed results:

| Run | compile_ok | run_ok | match_ok | golden_hit | perf_hit | total_s |
|---|---|---|---|---|---|---|
| first | true | true | true | false | false | 38.3623 |
| second | true | true | true | true | true | 20.0163 |

Interpretation:

- master dispatch does not break either cache hit path
- remote worker cache reuse is visible from the master-facing API response

## Throughput Comparison

Comparison mode:

- batch size: `4`
- new refactored worker with cache disabled: `18082`
- baseline worker from `HEAD` worktree: `18084`

Observed results:

| Server | wall_time_s | server_total_time_s | batch_size | success_count |
|---|---:|---:|---:|---:|
| refactored (`18082`) | 42.6586 | 42.6532 | 4 | 4 |
| baseline `HEAD` (`18084`) | 42.6406 | 42.6372 | 4 | 4 |

Interpretation:

- The refactored stack did not fail functionally under real HIP compilation/execution.
- After optimizing the cold path to avoid running separate reference golden/perf scripts when both are needed, the cache-disabled path is effectively at parity with the baseline `HEAD` worker for this workload.
- The observed difference is within measurement noise:
  - about `+0.018s` absolute
  - effectively `~0.04%` relative

## Larger Batch Benchmark By Cache Mode

Comparison mode:

- batch size: `8`
- same stable kernel names reused across two runs per server
- second run is the meaningful warm-cache measurement

### Cache Off

| Run | wall_time_s | server_total_time_s | batch_size | success_count |
|---|---:|---:|---:|---:|
| 0 | 44.6233 | 44.6168 | 8 | 8 |
| 1 | 44.8439 | 44.8412 | 8 | 8 |

Interpretation:

- as expected, no meaningful warm-up benefit

### Golden Cache On

| Run | wall_time_s | server_total_time_s | batch_size | success_count |
|---|---:|---:|---:|---:|
| 0 | 44.2085 | 44.2037 | 8 | 8 |
| 1 | 44.3467 | 44.3432 | 8 | 8 |

Supplementary probe on a single-item batch:

| Run | golden_hit | perf_hit | total_s |
|---|---|---|---:|
| first | false | false | 39.4708 |
| second | true | false | 37.9881 |

Interpretation:

- `reference_golden` does hit correctly on the second request
- the benefit is small because `reference_perf` is still live-measured and remains a dominant cost
- on an 8-task fully parallel batch, candidate compile/run and live perf measurement largely hide the saved golden cost

### Golden + Perf Cache On

| Run | wall_time_s | server_total_time_s | batch_size | success_count |
|---|---:|---:|---:|---:|
| 0 | 44.6788 | 44.6735 | 8 | 8 |
| 1 | 25.4694 | 25.4657 | 8 | 8 |

Interpretation:

- warm second-run throughput improves significantly once both caches hit
- observed reduction is about `19.21s` absolute, roughly `43%` relative versus the cold run
- this matches the design expectation: when both reference correctness and reference perf are reusable, the remaining dominant work is candidate compile/run

## Notes

- Running `POST /run_code` twice against the cache-enabled worker without pinning GPU showed `golden_hit=true` but `perf_hit=false` on the second request.
- This is expected with the current conservative `ReferencePerfKey`, because the single compatibility endpoint rotates through GPUs via a queue, so the runtime fingerprint changes.
