# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# sf_mock_server.py — 简化版本，只保留 EvalResponse 定义的字段
import argparse
import json
import os
import time
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn

app = FastAPI()

MODE = os.environ.get("SF_MOCK_MODE", "success")
DELAY_MS = int(os.environ.get("SF_MOCK_DELAY_MS", "0"))

def build_eval_response(
    *,
    kernel_name: str = "mock_kernel",
    compile_ok: bool,
    run_ok: bool,
    match_ok: bool,
    speedup: float,
    reason: str | None = None,
):
    resp = {
        "kernel_name": kernel_name,
        "compile_ok": compile_ok,
        "run_ok": run_ok,
        "match_ok": match_ok,
        "speedup": speedup,
    }
    if reason is not None:
        resp["reason"] = reason
    return resp

@app.post("/run_code", response_class=PlainTextResponse)
def run_code(_: dict):
    if DELAY_MS > 0:
        time.sleep(DELAY_MS / 1000)

    if MODE == "bad_json":
        return "some logs...\n<<NOT JSON>>"

    if MODE == "compile_fail":
        out = build_eval_response(
            compile_ok=False,
            run_ok=False,
            match_ok=False,
            speedup=0.0,
            reason="compile error",
        )
        return "compile stage log...\n" + json.dumps(out)

    if MODE == "run_fail":
        out = build_eval_response(
            compile_ok=True,
            run_ok=False,
            match_ok=False,
            speedup=0.0,
            reason="runtime error",
        )
        return "run stage log...\n" + json.dumps(out)

    # success / slow (slow just adds delay)
    out = build_eval_response(
        compile_ok=True,
        run_ok=True,
        match_ok=True,
        speedup=1.0,
    )
    return "log before...\n" + json.dumps(out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--mode", type=str, choices=["success","compile_fail","run_fail","bad_json","slow"], default=os.environ.get("SF_MOCK_MODE", "success"))
    p.add_argument("--delay-ms", type=int, default=int(os.environ.get("SF_MOCK_DELAY_MS", "0")))
    args = p.parse_args()
    os.environ["SF_MOCK_MODE"] = args.mode
    os.environ["SF_MOCK_DELAY_MS"] = str(args.delay_ms)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

if __name__ == "__main__":
    main()
