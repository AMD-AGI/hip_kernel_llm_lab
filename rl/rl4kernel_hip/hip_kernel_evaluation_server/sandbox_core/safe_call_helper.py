# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Safe call helper for HIP kernel evaluation.

This module provides a helper function definition string that handles
the 'fn' parameter conflict when calling model.forward() with inputs.
"""

SAFE_CALL_HELPER = """
def _safe_call(model, inputs, fn):
    import inspect
    use_fn = True  # 默认使用 fn
    truncated_inputs = inputs
    try:
        sig = inspect.signature(model.forward)
        params = [p for p in sig.parameters.keys() if p != 'self']
        if 'fn' in params:
            fn_idx = params.index('fn')
            truncated_inputs = inputs[:fn_idx] if len(inputs) > fn_idx else inputs
        else:
            use_fn = False
    except (ValueError, TypeError):
        # 签名检测失败，保持默认行为
        pass
    
    if use_fn:
        return model(*truncated_inputs, fn=fn)
    else:
        return model(*truncated_inputs)
"""
