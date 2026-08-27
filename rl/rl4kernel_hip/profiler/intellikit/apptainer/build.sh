#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

set -e

# Build the SIF image
IMAGE_NAME="intellikit.sif"
mkdir -p apptainer/images
apptainer build apptainer/images/"$IMAGE_NAME" apptainer/intellikit.def

echo "Built image: $IMAGE_NAME"
