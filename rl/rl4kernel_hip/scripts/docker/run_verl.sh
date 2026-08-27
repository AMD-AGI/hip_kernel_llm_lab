#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

CONTAINER_NAME="kernelrl"
DOCKER_IMAGE="verl-0.5:1013"
# DOCKER_IMAGE="verl-rocm-0.4.6:1106"
# DOCKER_IMAGE="rocm/pytorch:rocm6.3.3_ubuntu22.04_py3.10_pytorch_release_2.3.0"
HERE="$(pwd)"

docker run -it \
  --name "${CONTAINER_NAME}" \
  --network host \
  --device /dev/dri \
  --device /dev/kfd \
  --group-add video \
  --privileged \
  --shm-size 196G \
  -v "${HOME}/.ssh:/root/.ssh" \
  -v "${HOME}:${HOME}" \
  -w "${HERE}" \
  "${DOCKER_IMAGE}" \
  /bin/bash