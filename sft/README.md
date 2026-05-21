# SFT Usage

This directory contains the launch script and configuration files for HIP supervised fine-tuning (SFT) based on LLaMA-Factory.

## Directory Layout

- `run_train_full_sft.sh`: launch script for full-parameter SFT
- `train_yamls/qwen3-14b-full-sft-hip.yaml`: default training configuration
- `train_yamls/deepspeed/`: DeepSpeed configuration files
- `data/dataset_info.json`: dataset registry used by the current repo snapshot

## Prerequisites

Before starting training, make sure the following are ready:

1. A working HIP/ROCm training environment
2. Access to the base model weights
3. A local LLaMA-Factory checkout installed in your Python environment
4. HIP SFT data, either:
   - self-prepared HIP-to-HIP pair data, or
   - AMD-provided HIP SFT data

## 1. Prepare the Data

Prepare one of the following:

- your own HIP-to-HIP pair dataset, or
- AMD HIP SFT data

Place the dataset files into `LLaMA-Factory/data/` and make sure the dataset is registered in the LLaMA-Factory data config.

Notes:

- Pair datasets can follow the Alpaca-style mapping already used in this repo, for example: `instruction`, `input`, `output`.
- The default training YAML in this directory uses `dataset: merged_110k_if_reasoning`.
- The dataset registry maps it to `merged_260323_110k_if_reasoning.jsonl`.
- If you train with another dataset, update both the dataset registry file and the `dataset` field in `train_yamls/qwen3-14b-full-sft-hip.yaml`.

## 2. Install LLaMA-Factory

Clone and install LLaMA-Factory with its default setup:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt
```

Then replace the LLaMA-Factory data config with the one provided in this directory.

- If your LLaMA-Factory version uses `data/data_config.py`, replace `LLaMA-Factory/data/data_config.py` with `sft/data/data_config.py`.
- In the current repo snapshot, the dataset registry file under `sft/data/` is `dataset_info.json`. If your LLaMA-Factory checkout uses `data/dataset_info.json`, copy or merge this file there instead.

Other than the dataset config replacement above, keep LLaMA-Factory unchanged.

## 3. Start Training

Check the GPU list in `run_train_full_sft.sh` and adjust `HIP_VISIBLE_DEVICES` if needed, then start training with:

```bash
bash run_train_full_sft.sh
```

The script launches:

```bash
llamafactory-cli train train_yamls/qwen3-14b-full-sft-hip.yaml
```

## Default Training Setup

The current default configuration is:

- Base model: `Qwen/Qwen3-14B`
- Stage: `sft`
- Fine-tuning type: `full`
- DeepSpeed config: `train_yamls/deepspeed/ds_z3_config.json`
- Dataset: `merged_110k_if_reasoning`
- Dataset file: `merged_260323_110k_if_reasoning.jsonl`
- Output directory: `saves/qwen3-14b/sft-merged-110k-claude-4.5-data-if-reasoning-tagged-14b-2e-5-cpt`

## Customization

If needed, adjust the following fields in `train_yamls/qwen3-14b-full-sft-hip.yaml`:

- `model_name_or_path`
- `dataset`
- `output_dir`
- batch size and gradient accumulation settings
- learning rate and training epochs
- DeepSpeed config selection
