from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mixed_gemm",
    ext_modules=[
        CUDAExtension(
            "mixed_gemm",
            sources=["binding.cpp", "mixed_gemm.cu"],
            extra_compile_args={
                "nvcc": [
                    "-gencode", "arch=compute_90a,code=sm_90a",
                    "-O3",
                    "--use_fast_math",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
