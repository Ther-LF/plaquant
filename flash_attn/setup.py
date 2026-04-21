"""Build script for Mixed-Precision FlashAttention — CUTLASS 3.x Hopper."""
import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Use CUTLASS from plaquant root
cutlass_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'cutlass')

if not os.path.exists(os.path.join(cutlass_dir, 'include', 'cutlass', 'cutlass.h')):
    raise RuntimeError(
        f"CUTLASS not found at {cutlass_dir}. "
        "Run: cd project-resq/fake_quant/csrc && git submodule update --init cutlass")

cutlass_include_dirs = [
    os.path.join(cutlass_dir, 'include'),
    os.path.join(cutlass_dir, 'tools', 'util', 'include'),
]

setup(
    name='mixed_flash_attn',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='mixed_flash_attn',
            sources=[
                'mixed_flash_attn.cu',
                'mixed_flash_attn_binding.cpp',
            ],
            include_dirs=cutlass_include_dirs,
            extra_compile_args={
                'cxx': ['-std=c++17', '-O3'],
                'nvcc': [
                    '-gencode', 'arch=compute_90a,code=sm_90a',
                    '-std=c++17',
                    '-O3',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-DCUTLASS_VERSIONS_GENERATED',
                    '-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1',
                    '-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
