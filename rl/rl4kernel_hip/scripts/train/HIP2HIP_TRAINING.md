# Hip2Hip Single-Turn Training

This path trains full-file HIP optimization. It is not a kernel-snippet splice
pipeline.

## Dataset Contract

Build veRL parquet rows with the canonical hip2hip contract:

```bash
python dataset/convert_to_verl_parquet.py \
  --input-jsons ... \
  --data-source hip2hip-train \
  --optimization-paradigm hip2hip \
  --output-contract sample_json_v1 \
  --target-gpus mi300x \
  --output-name rl_data_hard_normal_mixed_hip2hip
```

Each row must have:

- `data_source=hip2hip-train`
- `extra_info.output_contract=sample_json_v1`
- `extra_info.optimization_paradigm=hip2hip_full_file`
- `extra_info.expected_code_unit=hip_translation_unit`
- `extra_info.persistence_mode=direct_full_file`
- `reward_model.ground_truth.hip_code` as the complete reference `.hip` file

The assistant response contract is one JSON object with `thought` and `code`.
The `code` field must be a complete replacement `.hip` translation unit and
must not contain markdown fences.

The raw RL dataset is not stored in Git. Restore it from Hugging Face when
needed:

```bash
huggingface-cli download amd/EXP-Models \
  --repo-type model \
  --include "dataset/hip_kernel_rldataset/*" \
  --local-dir .
```

This places files under `dataset/hip_kernel_rldataset/`, which is ignored by
Git together with derived parquet outputs.

## Training Launcher

Use:

```bash
export WANDB_API_KEY=...
scripts/train/react_single_turn_v1_hip2hip.sh \
  --train path/to/train.parquet \
  --val path/to/val.parquet \
  --sf-url http://host:8080/run_code
```

This launcher does not build or validate dataset contracts. Prepare train/val
parquet files manually before launch.

Do not commit WandB or Hugging Face tokens. The launcher requires
`WANDB_API_KEY` from the environment and intentionally contains no default key.

Full-file prompts and responses are longer than kernel snippets. The launcher
defaults to `PROMPT_LENGTH=8192` and `RES_LENGTH=24576`; re-profile these after
building a new dataset.

## Evaluation Server

Start a batch HIP evaluation server that exposes both `/run_code` and
`/run_code_batch`, for example:

```bash
cd hip_kernel_evaluation_server
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HIP_DEFAULT_BATCH_SIZE=64 \
./setup_server_req_deploy_hip2hip_batch.sh
```

The reward client is configured with the `/run_code` URL and rewrites it to
`/run_code_batch` internally for batch scoring. The server request payload is:

- `kernel_name`
- `hip_code`
- `hip_ref_code`
- `pytorch_module_code`
- `pytorch_functional_code`
- `atol`, `rtol`, `compile_timeout_s`, `run_timeout_s`

Enable reference cache settings in the server script for stable throughput.
Training-time parse failures are scored as `0.0` and are not sent to the server.

## Mode Reference

- `react mode`: `sample_json_v1`, one JSON object with `thought` and `code`.
  This is the supported contract for new hip2hip training.
- `normal` / legacy mode: `legacy_hip_fence_v1`, markdown HIP code fences.
  Keep this for compatibility only unless reproducing older runs.
- `single-turn`: one assistant response, scored by the reward server.
- `multi-turn`: async rollout plus tool calls through `react_multi_turn.sh` and
  `scripts/train/tool_config/hip_kernel_eval_tool_config.yaml`.
