# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# test_client_adapter.py
import json
import types
import pytest
import client_adapter as ca


# ---------- 工具：构造假的 requests.Response ----------
def _fake_response(status_code: int, text: str):
    resp = types.SimpleNamespace()
    resp.status_code = status_code
    resp.text = text

    def raise_for_status():
        if 400 <= status_code:
            import requests
            raise requests.HTTPError(f"{status_code} Error")
    resp.raise_for_status = raise_for_status
    return resp


# ---------- 基础样例请求体 ----------
def _sample_req() -> ca.EvalRequest:
    return ca.EvalRequest(
        kernel_name="toy_kernel",
        hip_code="/* hip code stub */\nint main(){return 0;}",
        pytorch_module_code=(
            "import torch\n"
            "class Model(torch.nn.Module):\n"
            "  def __init__(self):\n"
            "    super().__init__()\n"
            "  def forward(self,x):\n"
            "    return x\n"
            "def get_init_inputs():\n"
            "  return []\n"
            "def get_inputs():\n"
            "  import torch\n"
            "  return [torch.randn(1,16,64,64)]\n"
        ),
        pytorch_functional_code="import torch\ndef run(x):\n  return x\n",
        atol=1e-4,
        rtol=1e-3,
        compile_timeout_s=10,
        run_timeout_s=10,
    )


# ---------- 成功路径：打印验证信息 ----------
def test_call_run_code_success(monkeypatch, capsys):
    req = _sample_req()

    # 伪造 /run_code 的返回：日志 + 最后一行 JSON（EvalResponse）
    fake_body = (
        "some logs...\n"
        + json.dumps({
            "kernel_name": req.kernel_name,
            "compile_ok": True,
            "run_ok": True,
            "match_ok": True,
            "speedup": 1.23,
            "reason": None,
            "stats": [{"max_abs_diff": 0.0, "max_rel_diff": 0.0, "mean_abs_diff": 0.0}],
        })
    )

    def fake_post(url, json=None, timeout=None):
        # 校验客户端请求结构
        assert json is not None and "code" in json
        assert url.endswith("/run_code")
        return _fake_response(200, fake_body)

    monkeypatch.setattr(ca.requests, "post", fake_post)

    sf_url = "https://sandbox.example.com/run_code"
    resp = ca.call_run_code(sf_url, req, timeout_s=5)

    # 断言
    assert isinstance(resp, ca.EvalResponse)
    assert resp.compile_ok and resp.run_ok and resp.match_ok
    assert resp.speedup == pytest.approx(1.23)

    # 打印验证结果（确保显示）：建议配合 `pytest -s` 或 capsys.disabled()
    with capsys.disabled():
        print(f"[PASS] success: kernel={resp.kernel_name} match_ok={resp.match_ok} "
              f"speedup={resp.speedup:.2f} stats0={resp.stats[0].dict() if resp.stats else None}")


# ---------- URL 不合法：打印预期异常 ----------
def test_call_run_code_invalid_url(capsys):
    req = _sample_req()
    with pytest.raises(ValueError) as e:
        ca.call_run_code("https://sandbox.example.com/run", req)
    with capsys.disabled():
        print(f"[PASS] invalid_url: raised ValueError as expected -> {e.value}")


# ---------- HTTP 非 2xx：打印预期异常 ----------
def test_call_run_code_http_error(monkeypatch, capsys):
    req = _sample_req()

    def fake_post(url, json=None, timeout=None):
        return _fake_response(500, "internal error")

    monkeypatch.setattr(ca.requests, "post", fake_post)

    import requests
    with pytest.raises(requests.HTTPError) as e:
        ca.call_run_code("https://sandbox.example.com/run_code", req)
    with capsys.disabled():
        print(f"[PASS] http_error: raised HTTPError as expected -> {e.value}")


# ---------- 生成的 runner 代码包含关键字段：打印片段 ----------
def test_build_runner_code_contains_payload(capsys):
    req = _sample_req()
    code = ca.build_runner_code(req)
    assert "toy_kernel" in code and "int main()" in code
    with capsys.disabled():
        snippet = code[:120].replace("\n", "\\n")
        print(f"[PASS] runner_code_contains_payload: snippet={snippet}...")
