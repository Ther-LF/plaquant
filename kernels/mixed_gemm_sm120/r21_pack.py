"""Round-21 quant/pack optimization harness (SELF-CONTAINED, pack-only).

round-20 found the per-call quant/pack kernel (cd_quant_pack_kernel, round-18) is ~half of
mid-r e2e (down.8192 r=1/2 pack ~396us vs GEMM 375us) and ~4-6x off the ~1.8TB/s roofline.
The scalar kernel reads/writes byte-at-a-time with one thread per block. round-21 adds a
vectorized variant (impl=1): one 128-bit aligned load/store per block, register nibble-pack.

This harness isolates the PACK step (no GEMM). The pack kernel only reads bytes, so inputs
are random uint8 of the exact per-shape sizes -- the data path and SF layout are identical
to the real pipeline. BIT-EXACT gate: vec output (data AND SF) == scalar output (torch.equal)
for every shape x M x format x {NAT,ILV}. Then warm median timing + achieved BW vs roofline.

CSV line-buffered; harvest from disk.
"""
import os, sys, csv, torch
HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
sys.path.insert(0, HERE)
from torch.utils.cpp_extension import load

PEAK_TBS = 1.8  # RTX Pro 6000 Blackwell GDDR7 ~1.79 TB/s

def get_cd():
    return load(name="codesign_fused_ext",
        sources=[os.path.join(HERE, "codesign_fused_ext.cu")],
        extra_include_paths=[HERE, os.path.join(CUTLASS, "include"),
                             os.path.join(CUTLASS, "tools/util/include"),
                             os.path.join(CUTLASS, "examples/common")],
        extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda", "-O3"],
        verbose=True)

cd = get_cd()
print("CD_BUILD_OK", flush=True)

SHAPES = [("qkv", 2048, 2048), ("down", 2048, 8192), ("gate", 8192, 2048)]
MS = [2048, 8192]
R = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]
dev = "cuda"

def med_us(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    ts.sort()
    return ts[len(ts)//2]

def rand_u8(*shape):
    return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev)

# bytes moved by one pack call (read data + read sf + write data + write sf)
def traffic_fp4(M, Kl, data_out, sf_out):
    return M*Kl + M*(Kl//16) + data_out.numel() + sf_out.numel()
def traffic_fp8(M, Kh, data_out, sf_out):
    return M*Kh + M*(Kh//32) + data_out.numel() + sf_out.numel()

# ---- GATE: vec (impl=1) bit-identical to scalar (impl=0) ----
def gate():
    print("=== GATE: vec==scalar (data AND SF, NAT+ILV, fp4+fp8) ===", flush=True)
    ok = True
    for name, N, KT in SHAPES:
        for M in MS:
            for frac in (0.125, 0.5, 0.875):
                Kh = int(round(frac*KT)); Kh -= Kh % 32; Kl = KT - Kh; Kl -= Kl % 16
                if Kl <= 0 or Kh <= 0: continue
                c4 = rand_u8(M, Kl); sf4 = rand_u8(M, Kl//16)
                b8 = rand_u8(M, Kh); sf8 = rand_u8(M, Kh//32)
                for ilv in (True, False):
                    s4 = cd.quant_pack_fp4(c4, sf4, M, N, Kl, ilv, 0)
                    v4 = cd.quant_pack_fp4(c4, sf4, M, N, Kl, ilv, 1)
                    s8 = cd.quant_pack_fp8(b8, sf8, M, N, Kh, ilv, 0)
                    v8 = cd.quant_pack_fp8(b8, sf8, M, N, Kh, ilv, 1)
                    good = (torch.equal(s4[0], v4[0]) and torch.equal(s4[1], v4[1])
                            and torch.equal(s8[0], v8[0]) and torch.equal(s8[1], v8[1]))
                    ok = ok and good
                    if not good:
                        print(f"  FAIL {name} M{M} Kl{Kl} Kh{Kh} ilv{ilv}", flush=True)
                del c4, sf4, b8, sf8; torch.cuda.empty_cache()
    print("R21_PACK_BITEXACT_PASS" if ok else "R21_PACK_BITEXACT_FAIL", flush=True)
    return ok

gate_ok = gate()

CSV = os.path.join(HERE, "r21_pack.csv")
f = open(CSV, "w", newline="", buffering=1); w = csv.writer(f)
w.writerow(["shape","M","N","Kt","r","fmt","layout","Kpart","bytes_MB",
            "scalar_us","vec_us","speedup","scalar_BWpct","vec_BWpct"])
hdr = f"{'shape':9s}{'M':>6s}{'r':>6s}{'fmt':>4s}{'lay':>4s}{'Kp':>6s}{'MB':>7s}{'scal':>8s}{'vec':>8s}{'spd':>6s}{'sBW%':>6s}{'vBW%':>6s}"
print(hdr, flush=True)

def run_one(name, N, KT, M, r, fmt, ilv):
    Kh = KT if r==1.0 else (0 if r==0.0 else int(round(r*KT)))
    Kh -= Kh % 32; Kl = KT - Kh; Kl -= Kl % 16
    if fmt == "fp4":
        if Kl <= 0: return None
        din = rand_u8(M, Kl); sfin = rand_u8(M, Kl//16); Kp = Kl
        fn0 = lambda: cd.quant_pack_fp4(din, sfin, M, N, Kl, ilv, 0)
        fn1 = lambda: cd.quant_pack_fp4(din, sfin, M, N, Kl, ilv, 1)
        o = cd.quant_pack_fp4(din, sfin, M, N, Kl, ilv, 1)
        byts = traffic_fp4(M, Kl, o[0], o[1])
    else:
        if Kh <= 0: return None
        din = rand_u8(M, Kh); sfin = rand_u8(M, Kh//32); Kp = Kh
        fn0 = lambda: cd.quant_pack_fp8(din, sfin, M, N, Kh, ilv, 0)
        fn1 = lambda: cd.quant_pack_fp8(din, sfin, M, N, Kh, ilv, 1)
        o = cd.quant_pack_fp8(din, sfin, M, N, Kh, ilv, 1)
        byts = traffic_fp8(M, Kh, o[0], o[1])
    t0 = med_us(fn0); t1 = med_us(fn1)
    bw0 = byts/(t0*1e-6)/1e12; bw1 = byts/(t1*1e-6)/1e12
    MB = byts/1e6
    lay = "ILV" if ilv else "NAT"
    w.writerow([name,M,N,KT,f"{r:.3f}",fmt,lay,Kp,f"{MB:.2f}",
                f"{t0:.3f}",f"{t1:.3f}",f"{t0/t1:.2f}",
                f"{bw0/PEAK_TBS*100:.1f}",f"{bw1/PEAK_TBS*100:.1f}"])
    print(f"{name:9s}{M:6d}{r:6.3f}{fmt:>4s}{lay:>4s}{Kp:6d}{MB:7.1f}{t0:8.2f}{t1:8.2f}{t0/t1:6.2f}{bw0/PEAK_TBS*100:6.1f}{bw1/PEAK_TBS*100:6.1f}", flush=True)
    del din, sfin, o; torch.cuda.empty_cache()
    return None

for name, N, KT in SHAPES:
    for M in MS:
        for r in R:
            for fmt in ("fp4", "fp8"):
                for ilv in (True, False):
                    try:
                        run_one(name, N, KT, M, r, fmt, ilv)
                    except Exception as ex:
                        print(f"  ERR {name} M{M} r{r} {fmt} ilv{ilv}: {type(ex).__name__} {str(ex).splitlines()[0][:80]}", flush=True)
f.close()
print("R21_PACK_BITEXACT_OK" if gate_ok else "R21_PACK_BITEXACT_BAD", flush=True)
print("R21_PACK_DONE", flush=True)
