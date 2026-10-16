"""T2 occupancy probe: TMA phase-2 kernel. Compare 128x128 (st3, minb1/2) vs
128x64 tile (half B-smem + half accumulator -> aims for 2 CTA/SM naturally).
Prints attrs (TMA_ATTRS=1), gates cosine, times compute-saturated shapes.
    CUDA_VISIBLE_DEVICES=<free> TMA_ATTRS=1 python t2_probe.py
"""
import os, torch
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as t

ext = t.get_tma()
b1.get_ext(); b1.get_presf_ext()
print("Built.\n")

cases = [
    ("down_r1", 2048, 2048, 8192),
    ("gate_r1", 2048, 8192, 2048),
    ("qkv_r1",  2048, 2048, 2048),
]

def tma(p, M, N, K, st, minb=1, tile=0):
    return ext.b1_fused_tma_mx(p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
                               p["s_x"], p["s_w"], M, N, K, st, minb, tile)

# variants: (label, st, minb, tile)
variants = [
    ("128x128.st3.b1", 3, 1, 0),
    ("128x128.st4.b1", 4, 1, 0),
    ("128x64.st3.b1",  3, 1, 1),
    ("128x64.st4.b1",  4, 1, 1),
    ("128x64.st5.b1",  5, 1, 1),
]

print("=== cosine gate ===")
ok = True
for name, M, N, K in cases:
    p, oracle = mg.prep_mixed(M, N, 0, K, seed=1)
    for lab, st, minb, tile in variants:
        D = tma(p, M, N, K, st, minb, tile).float()
        cos = b1.cosine(D, oracle)
        if cos < 0.999: ok = False
        print(f"  {name:10s} {lab:16s}: cos={cos:.6f}")
print("GATE PASS\n" if ok else "GATE FAIL\n")
if not ok:
    raise SystemExit("correctness gate failed")

print("=== latency vs pure-fp8 bound (excess %) ===")
for name, M, N, K in cases:
    p, oracle = mg.prep_mixed(M, N, 0, K, seed=1)
    _, p8 = b1.prep_bounds_presf(M, N, K, seed=1)
    t8 = t.median_ms(lambda: b1.pure_fp8_presf(p8, M, N, K))
    out = [f"  {name:10s} bound={t8:.4f}"]
    for lab, st, minb, tile in variants:
        lt = t.median_ms(lambda: tma(p, M, N, K, st, minb, tile))
        out.append(f"{lab}={lt:.4f}({(lt-t8)/t8*100:+.1f}%)")
    print("  ".join(out))
print("\nT2 DONE")
