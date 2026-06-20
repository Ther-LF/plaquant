from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cutlass_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party/cutlass')

setup(
    name='mixed_gemm_l20',
    ext_modules=[
        CUDAExtension(
            name='mixed_gemm_l20',
            sources=['mixed_gemm_l20.cu'],
            include_dirs=[
                os.path.join(cutlass_dir, 'include'),
                os.path.join(cutlass_dir, 'tools/util/include'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-DNDEBUG',
                    '-std=c++17',
                    '--use_fast_math',
                    '-gencode', 'arch=compute_80,code=sm_80',
                    '-gencode', 'arch=compute_89,code=sm_89',
                    '-gencode', 'arch=compute_90a,code=sm_90a',
                    '-gencode', 'arch=compute_100,code=sm_100',
                    '--ptxas-options=-v',
                    '-lineinfo',
                ],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
