# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

llamafactory-cli train train_yamls/qwen3-14b-pretrain.yaml
