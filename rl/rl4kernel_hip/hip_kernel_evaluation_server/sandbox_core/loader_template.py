kernel_loader_template = """
import os

# Keep a single effective arch across generated scripts and server env.
effective_arch = (
    os.environ.get('HIP_EVAL_ARCH')
    or os.environ.get('HCC_AMDGPU_TARGET')
    or os.environ.get('AMDGPU_TARGETS')
    or os.environ.get('GPU_ARCHS')
    or os.environ.get('PYTORCH_ROCM_ARCH')
    or 'gfx942'
)
if ',' in effective_arch:
    effective_arch = next((part.strip() for part in effective_arch.split(',') if part.strip()), 'gfx942')

os.environ['HCC_AMDGPU_TARGET'] = effective_arch
os.environ['AMDGPU_TARGETS'] = effective_arch
os.environ['GPU_ARCHS'] = effective_arch
os.environ['PYTORCH_ROCM_ARCH'] = effective_arch

# Clear flags that can conflict with the generated per-task arch.
os.environ.pop('HIPCC_COMPILE_FLAGS_APPEND', None)

from torch.utils.cpp_extension import load

hip_{kernel_name}_ext = load(name={module_name},
               extra_include_paths=["{code_dir}/include"],
               sources=["{code_dir}/{code_file}"],
               build_directory={build_directory_expr},
               extra_cflags=["-O2", "--offload-arch=" + effective_arch],
               verbose=False)
hip_fn = hip_{kernel_name}_ext.forward

"""
