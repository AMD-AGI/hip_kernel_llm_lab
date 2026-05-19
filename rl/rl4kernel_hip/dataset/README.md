# veRL Parquet Data Build Guide

This guide documents how to build training parquet data with
`dataset/convert_to_verl_parquet.py` for both:

- `hip2hip` (full-file optimization)
- `kernel2kernel` (kernel splice optimization)

The output rows share the same table schema:

- `prompt`
- `data_source`
- `ability`
- `reward_model`
- `extra_info`

---

## 1) Build Hip2Hip Training Parquet

Use this mode when the model should read a full HIP file and output a full HIP
translation unit.

```bash
python dataset/convert_to_verl_parquet.py \
  --input-jsons dataset/hip_kernel_rldataset/rl_data_hard.json dataset/hip_kernel_rldataset/rl_data_normal.json \
  --input-format rl_data \
  --reference-root dataset/AIG-Datasets/v0.1/PyTorch_HIP_kernel_dataset/pytorch_hip_kernel_gpumode \
  --data-source hip2hip-train \
  --optimization-paradigm hip2hip \
  --output-contract sample_json_v1 \
  --target-gpus mi300x \
  --output-dir dataset/hip2hip_parquet \
  --output-name rl_data_hard_normal_mixed_hip2hip \
  --shuffle --seed 42 \
  --preview-records 2
```

Expected contract in generated rows:

- `data_source=hip2hip-train`
- `extra_info.output_contract=sample_json_v1`
- `extra_info.optimization_paradigm=hip2hip_full_file`
- `extra_info.expected_code_unit=hip_translation_unit`
- `extra_info.persistence_mode=direct_full_file`

---

## 2) Build Kernel2Kernel Training Parquet

Use this mode when the model should generate a kernel snippet that will be
spliced into the reference HIP file at reward/eval time.

```bash
python dataset/convert_to_verl_parquet.py \
  --input-jsons path/to/kernel2kernel_processed.json \
  --input-format kernel2kernel_json \
  --pytorch-root dataset/AIG-Datasets/v0.1/PyTorch_HIP_kernel_dataset/pytorch_hip_kernel_gpumode \
  --hip-opt-dir dataset/AIG-Datasets/v0.1/PyTorch_HIP_kernel_dataset/pytorch_hip_kernel_gpumode/hip_opt \
  --data-source kernel2kernel-train \
  --optimization-paradigm kernel2kernel \
  --output-contract sample_json_v1 \
  --target-gpus mi300x \
  --output-dir dataset/kernel2kernel_parquet \
  --output-name kernel2kernel_mixed \
  --preview-records 2
```

Expected contract in generated rows:

- `data_source=kernel2kernel-train`
- `extra_info.output_contract=sample_json_v1`
- `extra_info.optimization_paradigm=kernel2kernel_splice`
- `extra_info.expected_code_unit=kernel_function`
- `extra_info.persistence_mode=splice_kernel`

---

## 3) Key Parameters

- `--input-format`
  - `rl_data` for legacy RL records with `data_info`
  - `kernel2kernel_json` for processed k2k records with
    `kernel_name/input/hip_reference_*`
- `--reference-root`
  - Required for `rl_data` records that store legacy file paths
- `--pytorch-root`, `--hip-opt-dir`
  - Needed for `kernel2kernel_json` mapping
- `--output-contract`
  - Recommended: `sample_json_v1`
  - Legacy fallback: `legacy_hip_fence_v1`
- `--target-gpus`
  - One or more of supported GPU profiles (for prompt rendering)

---

## 4) Output File Naming

The converter writes files as:

- legacy contract: `<output_name>_<gpu>_react_verl.parquet`
- `sample_json_v1`: `<output_name>_<gpu>_react_sample_json_v1_verl.parquet`

Example:

- `rl_data_hard_normal_mixed_hip2hip_mi300x_react_sample_json_v1_verl.parquet`

---

## 5) Quick Notes

- The top-level input JSON must be a list.
- Records missing required ground truth (`hip_code`, `pytorch_module_code`,
  `pytorch_functional_code`) are skipped.
- If you pass multiple input JSONs and do not set `--no-shuffle`, records are
  shuffled by default.
