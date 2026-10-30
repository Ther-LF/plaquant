"""JIT loader for the SM120 exts inside the vLLM venv (torch 2.13 / py3.12).

Loads only what PromixLinearMethod needs per forward:
  cd  codesign_fused_ext      fused dual-format GEMM + load-time SF prepack
  qpf quant_pack_float_ext    fused float->packed codes + interleaved SF

Build flags mirror promix/eval/full_network_realkernel.py (including the
act-MSE defaults, which live in compile flags) so the vLLM path quantizes
activations EXACTLY like the promix reference. First build takes minutes and is
cached under TORCH_EXTENSIONS_DIR (keyed by torch version).
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
CUTLASS = os.path.join(REPO, "third_party", "cutlass")

_cache = {}


def act_mse_cflags():
    """MUST stay in sync with promix/eval/full_network_realkernel.act_mse_cflags
    (duplicated here to avoid importing promix.eval.* in the vLLM venv)."""
    if os.environ.get("PROMIX_ACT_MSE", "1") == "0":
        return []
    fast = 0 if os.environ.get("PROMIX_ACT_MSE_FAST", "1") == "0" else 1
    steps = int(os.environ.get("PROMIX_ACT_MSE_STEPS", "3"))
    return ["-DPROMIX_ACT_MSE=1", f"-DPROMIX_ACT_MSE_FAST={fast}",
            f"-DPROMIX_ACT_MSE_STEPS={steps}"]


def load_kernels():
    """Returns (cd, qpf). Cached per process."""
    if "cd" in _cache:
        return _cache["cd"], _cache["qpf"]
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
    from torch.utils.cpp_extension import load
    inc = [KERN, os.path.join(CUTLASS, "include"),
           os.path.join(CUTLASS, "tools/util/include"),
           os.path.join(CUTLASS, "examples/common")]
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    cd = load(name="codesign_fused_ext",
              sources=[os.path.join(KERN, "codesign_fused_ext.cu")],
              extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
    qpf = load(name="quant_pack_float_ext",
               sources=[os.path.join(KERN, "quant_pack_float_ext.cu")],
               extra_include_paths=inc, extra_cuda_cflags=flags + act_mse_cflags(),
               verbose=False)
    _cache["cd"], _cache["qpf"] = cd, qpf
    return cd, qpf
