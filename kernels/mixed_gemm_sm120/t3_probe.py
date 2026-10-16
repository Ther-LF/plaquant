"""T3 CTA-rasterization probe: grouped (Triton-style) group-M ordering for L2
reuse on large N. gate_r1/up are the band-58% large-N weak points.
Gates cosine, prints attrs (TMA_ATTRS=1), times swz in {0,2,4,8}.
    CUDA_VISIBLE_DEVICES=<free> TMA_ATTRS=1 python t3_probe.py
"""
import os, torch
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as t

ext = t.get_tma()
b1.get_ext(); b1.get_presf_ext()
print("Built.\n")

cases = [
    ("gate_r1", 2048, 8192, 2048),
    ("down_r1", 2048, 2048, 8192),
]
swzs = [0, 2, 4, 8]
STS = [3, 4]

def tma(p, M, N, K, st, swz):
    return ext.b1_fused_tma_mx(p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
                               p["s_x"], p["s_w"], M, N, K, st, 1, 0, swz)

print("=== cosine gate ===")
ok = True
for name, M, N, K in cases:
    p, oracle = mg.prep_mixed(M, N, 0, K, seed=1)
    for st in STS:
        for sw in swzs:
            D = tma(p, M, N, K, st, sw).float()
            cos = b1.cosine(D, oracle)
            if cos < 0.999: ok = False
            print(f"  {name:10s} st={st} swz={sw}: cos={cos:.6f}")
print("GATE PASS\n" if ok else "GATE FAIL\n")
if not ok:
    raise SystemExit("correctness gate failed")

print("=== latency vs pure-fp8 bound (excess %) ===")
for name, M, N, K in cases:
    p, oracle = mg.prep_mixed(M, N, 0, K, seed=1)
    _, p8 = b1.prep_bounds_presf(M, N, K, seed=1)
    t8 = t.median_ms(lambda: b1.pure_fp8_presf(p8, M, N, K))
    for st in STS:
        out = [f"  {name:10s} st={st} bound={t8:.4f}"]
        for sw in swzs:
            lt = t.median_ms(lambda: tma(p, M, N, K, st, sw))
            out.append(f"swz{sw}={lt:.4f}({(lt-t8)/t8*100:+.1f}%)")
        print("  ".join(out))
print("\nT3 DONE")
