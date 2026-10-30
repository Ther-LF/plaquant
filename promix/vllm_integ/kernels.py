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
    # Keep the venv's ext cache separate from the container base env's: both are
    # py3.12+cu130 so the default cache dir name collides across torch versions.
    import torch
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        os.path.expanduser(f"~/.cache/torch_ext_promix_{torch.__version__.split('+')[0]}"))
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


def _load_ext(key, fname, extra_flags=(), with_inc=True):
    """Generic cached JIT load sharing load_kernels' build env."""
    if key in _cache:
        return _cache[key]
    load_kernels()  # ensures TORCH_CUDA_ARCH_LIST / TORCH_EXTENSIONS_DIR are set
    from torch.utils.cpp_extension import load
    inc = [KERN, os.path.join(CUTLASS, "include"),
           os.path.join(CUTLASS, "tools/util/include"),
           os.path.join(CUTLASS, "examples/common")] if with_inc else []
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    mod = load(name=fname.replace(".cu", ""),
               sources=[os.path.join(KERN, fname)],
               extra_include_paths=inc,
               extra_cuda_cflags=flags + list(extra_flags), verbose=False)
    _cache[key] = mod
    return mod


def load_smallm():
    """vllm-perf r2 lever 1: small-M fused-GEMM config variants (128x64 tiles).
    Returns None when disabled via PROMIX_SMALLM=0 (restores the fixed 128x128
    all-M behavior)."""
    if os.environ.get("PROMIX_SMALLM", "1") == "0":
        return None
    return _load_ext("sm", "cd_smallm_ext.cu")


def load_qpfm():
    """vllm-perf r2 lever 2c: single-launch fused NVFP4+MXFP8 activation pack
    (quant_pack_float_mixed) — halves the per-Linear pack launch count.
    Bit-exact to the two-launch qpf path (verbatim device quant code).
    Disabled via PROMIX_QPFM=0. Must carry the same act-MSE flags as qpf."""
    if os.environ.get("PROMIX_QPFM", "1") == "0":
        return None
    return _load_ext("qpfm", "quant_pack_fused_ext.cu", extra_flags=act_mse_cflags())


def load_had_pack():
    """vllm-perf r2 lever 2a: fused online-Hadamard + NVFP4/MXFP8 quant/pack for
    the down_proj input (hadamard_pack_mixed / hadamard28_pack_mixed).
    MEASURED NEGATIVE (r2_pack_probe.json, graph-replay): the fused one-CTA-per-
    row kernel LOSES to the fast_hadamard_transform fp16 kernel + qpfm chain —
    3B K=8192 0.82-0.97x, 8B K=14336 0.26-0.34x — so it is DEFAULT-OFF
    (PROMIX_HAD_PACK=1 opts in for experiments). Numerics were green (bit-exact
    vs the fp32-trait reference, GEMM cos >= 0.99998 vs the current chain); the
    loss is pure kernel efficiency: FHT's dedicated fp16 kernel is ~4x faster
    than the hpk transform stage at these shapes. Returns (hpk, hpk28)."""
    if os.environ.get("PROMIX_HAD_PACK", "0") != "1":
        return None, None
    hpk = _load_ext("hpk", "hadamard_pack_ext.cu", extra_flags=act_mse_cflags())
    hpk28 = _load_ext("hpk28", "hadamard28_pack_ext.cu", extra_flags=act_mse_cflags())
    return hpk, hpk28


def load_decode_gemv():
    """Decode GEMV ext (vllm-perf r1): weight-only W4A16 / legacy W4A4 GEMVs for
    small-M (captured decode batch) steps. Returns None when disabled via
    PROMIX_DECODE_GEMV=0 (restores the Phase-2/3 all-M fused-GEMM behavior).
    Cached per process; same build env as load_kernels."""
    if os.environ.get("PROMIX_DECODE_GEMV", "1") == "0":
        return None
    if "dg" in _cache:
        return _cache["dg"]
    load_kernels()  # ensures TORCH_CUDA_ARCH_LIST / TORCH_EXTENSIONS_DIR are set
    from torch.utils.cpp_extension import load
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    dg = load(name="decode_gemv_ext",
              sources=[os.path.join(KERN, "decode_gemv_ext.cu")],
              extra_cuda_cflags=flags, verbose=False)
    _cache["dg"] = dg
    return dg
