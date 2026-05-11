import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Path to CUTLASS include directory
CUTLASS_PATH = os.path.join(os.path.dirname(__file__), "../../third_party/cutlass")
CUTLASS_INCLUDE = os.path.join(CUTLASS_PATH, "include")
CUTLASS_TOOLS_INCLUDE = os.path.join(CUTLASS_PATH, "tools/util/include")

assert os.path.isdir(CUTLASS_INCLUDE), f"CUTLASS not found at {CUTLASS_INCLUDE}"

setup(
    name="mixed_gemm",
    ext_modules=[
        CUDAExtension(
            "mixed_gemm",
            sources=["mixed_gemm.cu", "fused_mixed_gemm.cu", "fused_mixed_gemm_v2.cu", "fused_mixed_gemm_v3.cu"],
            include_dirs=[CUTLASS_INCLUDE, CUTLASS_TOOLS_INCLUDE],
            extra_compile_args={
                "nvcc": [
                    "-gencode", "arch=compute_90a,code=sm_90a",
                    "-O3", "-DNDEBUG",
                    "--use_fast_math",
                    "-std=c++17",
                    "-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1",
                ],
                "cxx": ["-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
