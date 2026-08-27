# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# reward.py — 使用 sandbox.client_adapter 通过 /run_code 执行并解析 EvalResponse 计分
from __future__ import annotations
import os
import json
import math
import typing as T
from typing import Optional

from sandbox.client_adapter import (
    EvalRequest,
    EvalResponse,
    call_run_code,
)

# 允许从环境读取默认端点（训练脚本也会通过 Hydra 传递）
SF_URL_ENV = os.environ.get("SF_URL", "").strip()


# -----------------------
# 小工具
# -----------------------
def _strip_code_fences(s: Optional[str]) -> str:
    """去掉 Markdown 代码围栏，防止模型输出携带 ``` 导致编译失败。"""
    if not s:
        return ""
    s = s.strip()
    if s.startswith("```"):
        # 去首行 ```xxx
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
    if s.endswith("```"):
        s = s[: -3]
    return s.strip()


def _maybe_read_text(val: Optional[str], code_root: Optional[str] = None) -> str:
    """
    val 既可能是代码字符串，也可能是相对/绝对路径。
    - 若是可读文件（优先拼 code_root），读文件内容；
    - 否则按代码字符串返回原文。
    """
    if not val:
        return ""
    # 拼接 code_root 再判断
    paths_to_try = []
    if code_root:
        paths_to_try.append(os.path.join(code_root, val))
    paths_to_try.append(val)
    for p in paths_to_try:
        try:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception:
            pass
    return val  # 当作代码字符串


def _require_run_code_url(url: Optional[str]) -> str:
    """校验端点必须以 /run_code 结尾（与 Sandbox Fusion 示例一致）。"""
    u = (url or "").strip()
    if not u or not u.endswith("/run_code"):
        raise ValueError(f"Sandbox Fusion url must end with /run_code, got: {u!r}")
    return u  # :contentReference[oaicite:3]{index=3}


# -----------------------
# 主入口：veRL 自定义奖励函数
# -----------------------
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: T.Any,
    extra_info: Optional[dict] = None,
) -> float:
    """
    满足 veRL 的自定义奖励函数签名。配置：
      custom_reward_function.path=/abs/path/reward.py
      custom_reward_function.name=compute_score
    文档：可以用 path/name 指定自定义奖励函数文件与函数名。:contentReference[oaicite:4]{index=4}
    """
    # 仅对你的数据源启用
    if data_source not in ("torch2hip-train", "torch2hip-val", "hip2hip-train", "hip2hip-val", "kernel2kernel-train", "kernel2kernel-val"):
        return 0.0      ##NOTE-这里存在潜在问题，如果数据源不是我们需要的，应该返回一个负数，而不是0。但是怎么设计呢？ -->  直接启用这里的判断？

    extra = extra_info or {}
    # 1) 端点解析（优先 extra_info.sandbox_url，次之环境变量）
    try:
        sf_url = _require_run_code_url(extra.get("sandbox_url") or SF_URL_ENV)
    except Exception:
        # 没有合法端点，给负奖励推动配置修复
        return -1.0
    
    # 2) 解析 kernel_name / 参考代码（字符串或路径）
    code_root = extra.get("code_root")  # 可选：用于解析相对路径
    gt = ground_truth if isinstance(ground_truth, dict) else {}


    kernel_name_base = gt.get("kernel_name") or extra.get("kernel_name") or "unknown"

    # 支持两套字段名：
    # - 你前面验证 parquet 使用的:  pytorch_code_module / pytorch_code_functional（多为路径）
    # - 也支持直接传完整源码字符串：pytorch_module_code / pytorch_functional_code
    module_src = (
        gt.get("pytorch_module_code")
        or _maybe_read_text(gt.get("pytorch_code_module"), code_root)
        or _maybe_read_text(extra.get("pytorch_module_code"), code_root)
    )
    functional_src = (
        gt.get("pytorch_functional_code")
        or _maybe_read_text(gt.get("pytorch_code_functional"), code_root)
        or _maybe_read_text(extra.get("pytorch_functional_code"), code_root)
    )

    hip_ref = gt.get("hip_code") or ""
    # 3) HIP 源清理（去代码围栏）
    hip_src = _strip_code_fences(solution_str)
    
    # 🔧 关键修复：添加代码hash避免PyTorch JIT缓存冲突
    # veRL训练中，同一kernel_name会生成多个不同代码版本
    # 如果使用相同name，PyTorch会复用缓存，导致新代码未被评估
    import hashlib
    code_hash = hashlib.md5(hip_src.encode()).hexdigest()[:8]
    kernel_name = f"{kernel_name_base}_{code_hash}"

    # 4) 超时与容差（可由 extra_info 覆盖）
    atol = float(extra.get("atol", 1e-4))
    rtol = float(extra.get("rtol", 1e-3))
    compile_timeout_s = int(extra.get("compile_timeout_s", 600))
    run_timeout_s = int(extra.get("run_timeout_s", 120))
    sf_timeout_s = int(extra.get("sf_timeout_s", 300))

    # 5) 组装 EvalRequest 并调用 /run_code
    req = {
            "kernel_name": kernel_name,
            "hip_code": hip_src,
            "hip_ref_code": hip_ref,
            "pytorch_module_code": module_src or "",
            "pytorch_functional_code": functional_src or "",
            "atol": atol,
            "rtol": rtol,
            "compile_timeout_s": compile_timeout_s,
            "run_timeout_s": run_timeout_s,
        }

    try:
        resp = call_run_code(sf_url, req, timeout_s=sf_timeout_s)
    except Exception:
        # 网络/HTTP/解析异常等
        # print(f'Evaluation exception.')
        return -1.0
    
    if resp.status_code != 200:
        return -1.0

    # 解析响应：mock server 返回的是 PlainTextResponse (日志 + 最后一行JSON)
    # 格式: "log before...\n{json}"
    try:
        resp_text = resp.text.strip()
        # 取最后一行作为 JSON
        last_line = resp_text.split('\n')[-1]
        resp_data = json.loads(last_line)["msg"]
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        return -1.0
    
    # 调试输出（可选）
    print(f'resp_data:{resp_data}')
    
    # 6) 按 EvalResponse 计分
    score = _compute_single_score(resp_data)
    return score  # 约 [0.5, 1.5)


def _compute_single_score(resp_data: dict) -> float:
    S_REF = 100.0  # 可按数据分布调整

    # 门槛：编译/运行失败 → 负分；数值不匹配 → 0 分（可按需改为轻微负分）
    if not resp_data.get("compile_ok", False):
        return -0.9
    if not resp_data.get("run_ok", False):
        return -0.5
    if not resp_data.get("match_ok", False):
        return 0.0

    speedup = max(0.0, float(resp_data.get("speedup") or 0.0))
    if S_REF > 0:
        gain = math.log1p(speedup) / math.log1p(S_REF)
        gain = min(max(gain, 0.0), 1.0)
    else:
        gain = 0.0

    return 0.5 + gain  # ∈ [0.5, 1.5]


# -----------------------
# 测试主函数
# -----------------------
def main():
    """测试 compute_score 函数 - 使用 mock 测试各种评分条件"""
    from unittest.mock import patch, Mock
    
    print("=" * 60)
    print("Testing HIP Kernel Code Evaluation (Mock Tests)")
    print("=" * 60)
    
    # 准备通用测试数据
    test_data_source = "hip2hip-train"
    test_solution = """
    #include <hip/hip_runtime.h>
    __global__ void test_kernel(float* out, const float* in, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = in[idx] * 2.0f;
    }
    """
    
    test_ground_truth = {
        "kernel_name": "test_kernel",
        "hip_code": test_solution.strip(),
        "pytorch_module_code": "import torch\nimport torch.nn as nn",
        "pytorch_functional_code": "import torch.nn.functional as F",
    }
    
    test_extra_info = {
        "sandbox_url": "http://mock-server:8000/run_code",
        "code_root": ".",
    }
    
    # 测试用例定义
    test_cases = [
        {
            "name": "Case 1: Compile Failed (compile_ok=False)",
            "mock_response": {
                "compile_ok": False,
                "run_ok": False,
                "match_ok": False,
                "speedup": 0.0,
            },
            "expected_score": -0.9,
        },
        {
            "name": "Case 2: Run Failed (compile_ok=True, run_ok=False)",
            "mock_response": {
                "compile_ok": True,
                "run_ok": False,
                "match_ok": False,
                "speedup": 0.0,
            },
            "expected_score": -0.5,
        },
        {
            "name": "Case 3: Output Mismatch (compile_ok=True, run_ok=True, match_ok=False)",
            "mock_response": {
                "compile_ok": True,
                "run_ok": True,
                "match_ok": False,
                "speedup": 0.0,
            },
            "expected_score": 0.0,
        },
        {
            "name": "Case 4: Success with speedup=1.0",
            "mock_response": {
                "compile_ok": True,
                "run_ok": True,
                "match_ok": True,
                "speedup": 1.0,
            },
            "expected_score": 0.5 + math.tanh(math.log1p(1.0)),  # ~1.38
        },
        {
            "name": "Case 5: Success with speedup=2.0",
            "mock_response": {
                "compile_ok": True,
                "run_ok": True,
                "match_ok": True,
                "speedup": 2.0,
            },
            "expected_score": 0.5 + math.tanh(math.log1p(2.0)),  # ~1.50
        },
    ]
    
    # 执行测试
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test_case['name']}")
        print(f"  Mock response: {test_case['mock_response']}")
        
        # 创建 mock 响应对象
        mock_resp = Mock()
        mock_resp.status_code = 200
        # 模拟 server 返回的格式：最后一行是 JSON
        mock_resp.text = f"Evaluation log...\n{json.dumps(test_case['mock_response'])}"
        
        # Mock call_run_code 函数（patch 当前模块中的引用）
        with patch('__main__.call_run_code', return_value=mock_resp):
            try:
                score = compute_score(
                    data_source=test_data_source,
                    solution_str=test_solution,
                    ground_truth=test_ground_truth,
                    extra_info=test_extra_info,
                )
                
                expected = test_case['expected_score']
                match = abs(score - expected) < 1e-6
                status = "✓ PASS" if match else "✗ FAIL"
                
                print(f"  Expected score: {expected:.4f}")
                print(f"  Actual score:   {score:.4f}")
                print(f"  {status}")
                
            except Exception as e:
                print(f"  ✗ Exception: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("Score Interpretation:")
    print("  -0.9: Compile failed")
    print("  -0.5: Run failed")
    print("   0.0: Output mismatch")
    print("  0.5+: Success (higher speedup = higher score)")
    print("=" * 60)


if __name__ == "__main__":
    main()
