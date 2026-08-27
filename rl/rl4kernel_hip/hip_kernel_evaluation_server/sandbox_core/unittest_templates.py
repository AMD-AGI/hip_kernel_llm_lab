# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from .safe_call_helper import SAFE_CALL_HELPER

pytorch_module_unittest_template = """

model = Model() if len(get_init_inputs()) == 0 else Model(*(get_init_inputs()))

model = model.to('cuda')
inputs = get_inputs()

inputs = [x.to('cuda') if isinstance(x, torch.Tensor) else x for x in inputs]

result_gold = model(*(inputs))

"""


pytorch_functional_unittest_template = SAFE_CALL_HELPER + """

model = Model() if len(get_init_inputs()) == 0 else Model(*(get_init_inputs()))

model = model.to('cuda')
inputs = get_inputs()
inputs = [x.to('cuda') if isinstance(x, torch.Tensor) else x for x in inputs]

result_gold = _safe_call(model, inputs, hip_fn)

"""

pytorch_module_specify_name_unittest_template = """

if len(get_init_inputs()) == 0:
    model = {model_name}() 
elif len(get_init_inputs()) == 2 and (isinstance(get_init_inputs()[0], list) and isinstance(get_init_inputs()[1], dict)):
    model = {model_name}() if len(get_init_inputs()[1]) == 0 else {model_name}(**(get_init_inputs()[1]))
else:
    model = {model_name}(*(get_init_inputs()))  

# model = {model_name}() if len(get_init_inputs()[1]) == 0 else {model_name}(**(get_init_inputs()[1]))

model = model.to('cuda')
inputs = get_inputs()

inputs = [x.to('cuda') if isinstance(x, torch.Tensor) else x for x in inputs]

result_gold = model(*(inputs))

"""

pytorch_functional_specify_name_unittest_template = SAFE_CALL_HELPER + """

if len(get_init_inputs()) == 0:
    model = {model_name}() 
elif len(get_init_inputs()) == 2 and (isinstance(get_init_inputs()[0], list) and isinstance(get_init_inputs()[1], dict)):
    model = {model_name}() if len(get_init_inputs()[1]) == 0 else {model_name}(**(get_init_inputs()[1]))
else:
    model = {model_name}(*(get_init_inputs()))  

# model = {model_name}() if len(get_init_inputs()[1]) == 0 else {model_name}(**(get_init_inputs()))

model = model.to('cuda')
inputs = get_inputs()

inputs = [x.to('cuda') if isinstance(x, torch.Tensor) else x for x in inputs]

result_gold = _safe_call(model, inputs, hip_fn)

"""


pytorch_functional_prepare_specify_name_unittest_template = SAFE_CALL_HELPER + """

if len(get_init_inputs()) == 0:
    model = {model_name}()
elif len(get_init_inputs()) == 2 and (isinstance(get_init_inputs()[0], list) and isinstance(get_init_inputs()[1], dict)):
    model = {model_name}() if len(get_init_inputs()[1]) == 0 else {model_name}(**(get_init_inputs()[1]))
else:
    model = {model_name}(*(get_init_inputs()))

model = model.to('cuda')
inputs = get_inputs()

inputs = [x.to('cuda') if isinstance(x, torch.Tensor) else x for x in inputs]

"""
