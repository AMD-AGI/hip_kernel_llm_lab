from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Optional

from .loader_template import kernel_loader_template
from .cache import sha256_text
from .unittest_templates import (
    pytorch_functional_prepare_specify_name_unittest_template,
    pytorch_functional_specify_name_unittest_template,
    pytorch_module_specify_name_unittest_template,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIX_SEED_FILE = os.path.join(THIS_DIR, "fix_seed.py")

CPU_SAVE_HELPER_CODE = """
def _to_cpu_obj(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {{key: _to_cpu_obj(item) for key, item in value.items()}}
    if isinstance(value, list):
        return [_to_cpu_obj(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_obj(item) for item in value)
    return value
"""

LEGACY_DUMP_RESULT_CODE = CPU_SAVE_HELPER_CODE + """
import torch
torch.save(_to_cpu_obj(result_gold), r"{result_path}")
"""

GOLDEN_ONLY_DUMP_CODE = CPU_SAVE_HELPER_CODE + """
import torch
torch.save({{'golden': _to_cpu_obj(result_gold)}}, r"{result_path}")
"""

FUNC_PERF_CODE = """
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start.record()

for _ in range({perf_iterations}):
    _safe_call(model, inputs, hip_fn)

end.record()
torch.cuda.synchronize()
elapsed = start.elapsed_time(end)  # in milliseconds

if isinstance(result_gold, dict):
    result_gold['perf'] = elapsed / {perf_iterations}
else:
    result_gold = {{'golden': result_gold, 'perf': elapsed / {perf_iterations}}}
"""

MODU_PERF_CODE = """
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start.record()

for _ in range({perf_iterations}):
    model(*(inputs))

end.record()
torch.cuda.synchronize()
elapsed = start.elapsed_time(end)  # in milliseconds

if isinstance(result_gold, dict):
    result_gold['perf'] = elapsed / {perf_iterations}
else:
    result_gold = {{'golden': result_gold, 'perf': elapsed / {perf_iterations}}}
"""

FUNC_PERF_ONLY_CODE = """
_ = _safe_call(model, inputs, hip_fn)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start.record()

for _ in range({perf_iterations}):
    _safe_call(model, inputs, hip_fn)

end.record()
torch.cuda.synchronize()
elapsed = start.elapsed_time(end)  # in milliseconds

import torch
torch.save({{'perf': elapsed / {perf_iterations}}}, r"{result_path}")
"""

MODU_PERF_ONLY_CODE = """
_ = model(*(inputs))
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start.record()

for _ in range({perf_iterations}):
    model(*(inputs))

end.record()
torch.cuda.synchronize()
elapsed = start.elapsed_time(end)  # in milliseconds

import torch
torch.save({{'perf': elapsed / {perf_iterations}}}, r"{result_path}")
"""


def result_path(result_dir: str, prefix: str, kernel_name: str) -> str:
    return os.path.join(result_dir, f"{prefix}{kernel_name}_result_gold.pt")


def build_loader_code(
    kernel_name: str,
    hip_code_dir: str,
    hip_file: str,
    *,
    module_name: Optional[str] = None,
    build_directory: Optional[str] = None,
) -> str:
    return kernel_loader_template.format(
        kernel_name=kernel_name,
        module_name=repr(module_name or f"hip_{kernel_name}"),
        code_dir=hip_code_dir,
        code_file=hip_file,
        build_directory_expr=repr(build_directory) if build_directory else "None",
    )


@lru_cache(maxsize=1)
def load_fix_seed_code() -> str:
    with open(FIX_SEED_FILE, 'r', encoding='utf-8') as handle:
        return handle.read()


@lru_cache(maxsize=1)
def get_template_bundle_hash() -> str:
    return sha256_text(
        ''.join(
            [
                load_fix_seed_code(),
                kernel_loader_template,
                pytorch_module_specify_name_unittest_template,
                pytorch_functional_prepare_specify_name_unittest_template,
                pytorch_functional_specify_name_unittest_template,
                FUNC_PERF_CODE,
                MODU_PERF_CODE,
                FUNC_PERF_ONLY_CODE,
                MODU_PERF_ONLY_CODE,
                CPU_SAVE_HELPER_CODE,
                GOLDEN_ONLY_DUMP_CODE,
                LEGACY_DUMP_RESULT_CODE,
            ]
        )
    )


def extract_module_name(pytorch_code: str, kernel_name: str) -> str:
    if "class Model" in pytorch_code:
        return "Model"

    match = re.search(r"class\s+(\w+)\s*\([^)]*nn\.Module[^)]*\)", pytorch_code or "")
    if match:
        return match.group(1)

    for line in (pytorch_code or '').split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        match = re.search(r"class\s+(\w+)\s*\(", line)
        if match:
            return match.group(1)

    parts = kernel_name.split('_')
    if len(parts) >= 3:
        last_part = parts[-1]
        if len(last_part) == 8 and all(c in '0123456789abcdef' for c in last_part.lower()):
            return '_'.join(parts[2:-1]) if len(parts) > 3 else parts[2]
        return '_'.join(parts[2:])
    return kernel_name.split('_', 2)[-1] if '_' in kernel_name else kernel_name


def construct_pytorch_module_unittest(
    pytorch_module_code: str,
    kernel_name: str,
    pt_save_dir: str,
    perf_iterations: int = 100,
    *,
    module_name: Optional[str] = None,
    build_directory: Optional[str] = None,
) -> str:
    complete_code = load_fix_seed_code()
    complete_code += pytorch_module_code

    module_name = 'Model'
    if 'class Model' not in pytorch_module_code:
        module_name = extract_module_name(pytorch_module_code, kernel_name)

    complete_code += pytorch_module_specify_name_unittest_template.format(model_name=module_name)
    complete_code += MODU_PERF_CODE.format(perf_iterations=perf_iterations)
    complete_code += LEGACY_DUMP_RESULT_CODE.format(
        result_path=result_path(pt_save_dir, 'py_modu_', kernel_name),
    )
    return complete_code


def construct_pytorch_functional_unittest(
    pytorch_functional_code: str,
    kernel_name: str,
    hip_code_dir: str,
    hip_file: str,
    pt_save_dir: str,
    prefix: str = 'py_func_',
    perf_iterations: int = 100,
    *,
    module_name: Optional[str] = None,
    build_directory: Optional[str] = None,
) -> str:
    complete_code = load_fix_seed_code()
    complete_code += build_loader_code(
        kernel_name,
        hip_code_dir,
        hip_file,
        module_name=module_name,
        build_directory=build_directory,
    )
    complete_code += pytorch_functional_code
    module_name = extract_module_name(pytorch_functional_code, kernel_name)
    complete_code += pytorch_functional_specify_name_unittest_template.format(model_name=module_name)
    complete_code += FUNC_PERF_CODE.format(perf_iterations=perf_iterations)
    complete_code += LEGACY_DUMP_RESULT_CODE.format(
        result_path=result_path(pt_save_dir, prefix, kernel_name),
    )
    return complete_code


def construct_reference_golden_script(
    pytorch_functional_code: str,
    kernel_name: str,
    hip_code_dir: str,
    hip_file: str,
    output_path: str,
    *,
    module_name: Optional[str] = None,
    build_directory: Optional[str] = None,
) -> str:
    complete_code = load_fix_seed_code()
    complete_code += build_loader_code(
        kernel_name,
        hip_code_dir,
        hip_file,
        module_name=module_name,
        build_directory=build_directory,
    )
    complete_code += pytorch_functional_code
    module_name = extract_module_name(pytorch_functional_code, kernel_name)
    complete_code += pytorch_functional_specify_name_unittest_template.format(model_name=module_name)
    complete_code += GOLDEN_ONLY_DUMP_CODE.format(result_path=output_path)
    return complete_code


def construct_reference_perf_script(
    pytorch_functional_code: str,
    kernel_name: str,
    hip_code_dir: str,
    hip_file: str,
    output_path: str,
    perf_iterations: int,
    *,
    module_name: Optional[str] = None,
    build_directory: Optional[str] = None,
) -> str:
    complete_code = load_fix_seed_code()
    complete_code += build_loader_code(
        kernel_name,
        hip_code_dir,
        hip_file,
        module_name=module_name,
        build_directory=build_directory,
    )
    complete_code += pytorch_functional_code
    module_name = extract_module_name(pytorch_functional_code, kernel_name)
    complete_code += pytorch_functional_prepare_specify_name_unittest_template.format(model_name=module_name)
    complete_code += FUNC_PERF_ONLY_CODE.format(
        perf_iterations=perf_iterations,
        result_path=output_path,
    )
    return complete_code


def build_candidate_compile_script(
    kernel_name: str,
    hip_code_dir: str,
    hip_file: str,
    *,
    module_name: Optional[str] = None,
    build_directory: Optional[str] = None,
) -> str:
    return build_loader_code(
        kernel_name,
        hip_code_dir,
        hip_file,
        module_name=module_name,
        build_directory=build_directory,
    )
