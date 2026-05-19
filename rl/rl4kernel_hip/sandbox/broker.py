import os
from fastapi import FastAPI, HTTPException
import requests, json
from pydantic import BaseModel
app = FastAPI()

# 复用定义的 EvalRequest/EvalResponse 和 build_runner_code
from client_adapter import EvalRequest, EvalResponse, build_runner_code

SF_URL = os.environ["SF_URL"]  # e.g. https://<host>/run_code

@app.post("/eval", response_model=EvalResponse)
def eval_kernel(req: EvalRequest):
    code = build_runner_code(req)
    r = requests.post(SF_URL, json={"code": code}, timeout=300)
    if r.status_code != 200:
        raise HTTPException(502, r.text)
    last = r.text.strip().splitlines()[-1]
    data = json.loads(last)
    return EvalResponse(**data)
