"""T3 Small-Tile-N probe: compare N=64 tile (tile_probe_ext) vs N=128 (fused_gemm_ext)
for the Rung-B presf path on decode shapes. Correctness-gated (cosine >= 0.999 vs oracle).
"""
import os
import torch
from torch.utils.cpp_extension import load
import mixed_gemm as mg

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"

_probe = None
def get_probe():
    global _probe
    if _probe is None:
        _probe = load(name="tile_probe_ext",
                      sources=[os.path.join(HERE, "tile_probe_ext.cu")],
                      extra_include_paths=[os.path.join(CUTLASS, "include"),
                                           os.path.join(CUTLASS, "tools/util/include"),
                                           os.path.join(CUTLASS, "examples/common")],
                      extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr",
                                         "--expt-extended-lambda", "-O3"], verbose=False)
    return _probe

def presf_n64(ext, p, M, N, Kl, Kh):
    return ext.mixed_gemm_fused_b_presf(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["physSFB_low"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["physSFB_high"],
        p["s_x"], p["s_w"], M, N, Kl, Kh)

SHAPES = {
    "q_proj_1B": (2048, 1792, 256),
    "k_proj_1B": (512, 1792, 256),
    "v_proj_1B": (512, 1792, 256),
    "o_proj_1B": (2048, 1792, 256),
}

def main():
    mg.get_fused_ext()
    probe = get_probe()
    print("=== T3 Small-Tile-N (N=64) vs N=128, Rung-B presf, decode shapes ===")
    ok = True
    for name, (N, Kl, Kh) in SHAPES.items():
        for M in [1, 16]:
            p, oracle = mg.prep_mixed(M, N, Kl, Kh, seed=M)
            mg.prepack_weight_sf(p, N, Kl, Kh)
            d128 = mg.mixed_gemm_fused_b_presf(p, M, N, Kl, Kh).float()
            d64 = presf_n64(probe, p, M, N, Kl, Kh).float()
            c128 = mg.cosine(d128, oracle)
            c64 = mg.cosine(d64, oracle)
            if c64 < 0.999:
                ok = False
            t128 = mg._time_ms(lambda: mg.mixed_gemm_fused_b_presf(p, M, N, Kl, Kh))
            t64 = mg._time_ms(lambda: presf_n64(probe, p, M, N, Kl, Kh))
            print(f"  {name:12s} M={M:3d} N={N:5d}: N128={t128:.4f}ms N64={t64:.4f}ms "
                  f"N64/N128={t64/t128:.2f}x  cos128={c128:.6f} cos64={c64:.6f} "
                  f"[{'PASS' if c64>=0.999 else 'FAIL'}]")
    print("ALL PASS" if ok else "SOME FAILED")

if __name__ == "__main__":
    main()
