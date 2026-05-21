# CPT Usage

This directory contains launch scripts and training configurations for HIP continued pretraining (CPT) based on LLaMA-Factory.

The current CPT flow has two stages:

1. CPT on HIP-related pretraining data
2. FIM training based on the CPT checkpoint

## Directory Layout

- `run_train_cpt.sh`: launch script for the CPT stage
- `run_train_cpt_fim.sh`: launch script for the FIM training stage
- `train_yamls/qwen3-14b-pretrain.yaml`: default CPT configuration
- `train_yamls/qwen3-14b-pretrain-fim.yaml`: default FIM training configuration
- `train_yamls/deepspeed/`: DeepSpeed configuration files
- `data/dataset_info.json`: dataset registry used by the current repo snapshot

## Prerequisites

Before starting training, make sure the following are ready:

1. A working HIP/ROCm training environment
2. Access to the base model weights, currently `Qwen/Qwen3-14B`
3. A local LLaMA-Factory checkout installed in your Python environment
4. HIP continued pretraining data registered as `pretrain_hip`
5. FIM training data registered as `FIM_hip`

## 1. Prepare LLaMA-Factory

Clone and install LLaMA-Factory with its default setup:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt
```

Place the training data under `LLaMA-Factory/data/` and make sure each dataset is registered in the LLaMA-Factory dataset registry.

Notes:

- The CPT YAML uses `dataset: pretrain_hip`, mapped to `pretrain_data_merged.json`.
- The FIM YAML uses `dataset: FIM_hip`, mapped to `merged_FIM.json`.
- If your LLaMA-Factory checkout uses `data/dataset_info.json`, copy or merge the dataset entries there.
- If you use different dataset names, update both the dataset registry and the corresponding `dataset` field in the YAML files.

## 2. Stage 1: CPT

Check the GPU list in `run_train_cpt.sh` and adjust `HIP_VISIBLE_DEVICES` if needed:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Start CPT training:

```bash
bash run_train_cpt.sh
```

The script launches:

```bash
llamafactory-cli train train_yamls/qwen3-14b-pretrain.yaml
```

Default CPT configuration:

- Base model: `Qwen/Qwen3-14B`
- Stage: `pt`
- Fine-tuning type: `full`
- DeepSpeed config: `train_yamls/deepspeed/ds_z3_config.json`
- Dataset: `pretrain_hip`
- Dataset file: `pretrain_data_merged.json`
- Cutoff length: `8192`
- Per-device train batch size: `4`
- Gradient accumulation steps: `16`
- Learning rate: `1.0e-5`
- Epochs: `5`
- Output directory: `save/Qwen3-14B/cpt`

## 3. Stage 2: FIM Training

After the CPT stage finishes, update `model_name_or_path` in `train_yamls/qwen3-14b-pretrain-fim.yaml` to point to the CPT checkpoint you want to continue from.

Current placeholder:

```yaml
model_name_or_path: path_to_cpt_llm_weight
```

Check the GPU list in `run_train_cpt_fim.sh` and adjust `HIP_VISIBLE_DEVICES` if needed:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Start FIM training:

```bash
bash run_train_cpt_fim.sh
```

The script launches:

```bash
llamafactory-cli train train_yamls/qwen3-14b-pretrain-fim.yaml
```

Default FIM training configuration:

- Base checkpoint: `path_to_cpt_llm_weight`
- Stage: `sft`
- Fine-tuning type: `full`
- DeepSpeed config: `train_yamls/deepspeed/ds_z3_config.json`
- Dataset: `FIM_hip`
- Dataset file: `merged_FIM.json`
- Cutoff length: `8192`
- Per-device train batch size: `4`
- Gradient accumulation steps: `2`
- Learning rate: `1.0e-5`
- Epochs: `5`
- Output directory: `save/Qwen3-14B/cpt_fim`

## Customization

If needed, adjust the following fields in the YAML files:

- `model_name_or_path`
- `dataset`
- `output_dir`
- `deepspeed`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `learning_rate`
- `num_train_epochs`
- `cutoff_len`
