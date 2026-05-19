# Error Log Configuration

`HIP_ERROR_LOG_DIR` controls where server adapters place per-kernel error logs.

## Primary Control

Set the environment variable before starting the server:

```bash
export HIP_ERROR_LOG_DIR="./error_log_experiment_1"
bash setup_server_req_deploy_hip2hip_batch.sh
```

This directory is read by `eval_config.load_eval_settings()` and then passed into server adapters such as:
- `server_req_deploy_hip2hip.py`
- `server_req_deploy_hip2hip_batch.py`
- `master_server.py`

## Typical Layout

```text
{HIP_ERROR_LOG_DIR}/
  kernel_name_a_error.log
  kernel_name_b_error.log
  kernel_name_c_error.log
```

## How Logs Are Produced

```mermaid
flowchart LR
    EnvVar["HIP_ERROR_LOG_DIR"] --> EvalConfig["load_eval_settings()"]
    EvalConfig --> ServerAdapters["server adapters"]
    ServerAdapters --> EvalCore["run_eval_request()"]
    EvalCore --> PerKernelLog["{kernel_name}_error.log"]
```

## Reference Prewarm Logging

`server_tools/prewarm_reference_cache.py` uses the same configured error-log root and writes:

```text
{HIP_ERROR_LOG_DIR}/{kernel_name}_prewarm_error.log
```

This keeps online-eval errors and prewarm errors in the same experiment-specific directory.

## Recommended Patterns

### Per experiment

```bash
export HIP_ERROR_LOG_DIR="./error_log_exp_001"
```

### Per day

```bash
export HIP_ERROR_LOG_DIR="./logs/$(date +%Y-%m-%d)"
```

### Per model or reward run

```bash
export HIP_ERROR_LOG_DIR="./error_log_32b_correct_speedup_runA"
```

## Notes

- The directory is created automatically if missing.
- The current implementation prefers per-kernel files over a shared monolithic log.
- If `error_log_file` is not passed into the evaluation core, the fallback remains `{tmp_dir}/error_log.txt`, but the server adapters normally provide explicit per-kernel paths.

## Related Docs

- [ERROR_LOG_README.md](ERROR_LOG_README.md)
- [../README.md](../README.md)
