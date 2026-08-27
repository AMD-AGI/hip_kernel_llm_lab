#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

set -euo pipefail

MODE="loop"
if [[ "${1:-}" == "--once" ]]; then
  MODE="once"
  shift
fi

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 [--once] <wandb-run-dir> [run-id]" >&2
  exit 1
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "FATAL: run dir not found: ${RUN_DIR}" >&2
  exit 1
fi

ENTITY="${WANDB_ENTITY:-eagle-training}"
PROJECT="${WANDB_PROJECT:-hip_agent}"
SYNC_INTERVAL_SECONDS="${WANDB_SYNC_INTERVAL_SECONDS:-1200}"

default_run_id="$(basename "${RUN_DIR}")"
default_run_id="${default_run_id##*-}"
RUN_ID="${2:-${WANDB_RUN_ID:-${default_run_id}}}"

sync_once() {
  python -m wandb sync \
    --include-online \
    --include-offline \
    --include-synced \
    --append \
    -e "${ENTITY}" \
    -p "${PROJECT}" \
    --id "${RUN_ID}" \
    "${RUN_DIR}"

  WANDB_SYNC_ENTITY="${ENTITY}" \
  WANDB_SYNC_PROJECT="${PROJECT}" \
  WANDB_SYNC_RUN_ID="${RUN_ID}" \
  python - <<'PY'
import os
import wandb

entity = os.environ["WANDB_SYNC_ENTITY"]
project = os.environ["WANDB_SYNC_PROJECT"]
run_id = os.environ["WANDB_SYNC_RUN_ID"]

run = wandb.Api(timeout=30).run(f"{entity}/{project}/{run_id}")
print(f"url={run.url}")
print(f"_step={run.summary.get('_step')}")
print(f"training/global_step={run.summary.get('training/global_step')}")
PY
}if [[ "${MODE}" == "once" ]]; then
  sync_once
  exit 0
fi

while true; do
  date -u +"[%Y-%m-%dT%H:%M:%SZ] sync start"
  sync_once
  date -u +"[%Y-%m-%dT%H:%M:%SZ] sleeping ${SYNC_INTERVAL_SECONDS}s"
  sleep "${SYNC_INTERVAL_SECONDS}"
done
