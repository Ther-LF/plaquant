"""config_dispatch_test.py — exercises config_dispatch.hpp end to end.

Builds config_dispatch_test.cu THREE times (one per kind: nvfp4 / mxfp8 / codesign) via
-DDISPATCH_ENABLE_* so each TU instantiates only one family's winning configs. For every
(shape, M) it:
  1. calls the dispatcher's tag()  -> verifies the picked ConfigTag equals (or is within a
     3% latency tie of) the MEASURED best config from the round-24 grid-search CSVs
     (r24_presf_search.csv / r24_cd_search.csv);
  2. calls the dispatcher's run() (= launch_best_<kind>) -> verifies cosine >= 0.9999.

PASS requires every cell: cosine ok AND dispatcher-choice latency <= 1.03x measured best.
"""
import os, sys, csv, torch
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
sys.path.insert(0, HERE)
from torch.utils.cpp_extension import load
import nvfp4_gemm as nv
import mxfp8_gemm as mx
import mixed_gemm as mg
import b1_fused as b1

INC = [HERE, CUTLASS+"/include", CUTLASS+"/tools/util/include", CUTLASS+"/examples/common"]
BASE = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]


def build(kind):
    return load(name="cfgdisp_" + kind, sources=[os.path.join(HERE, "config_dispatch_test.cu")],
                extra_include_paths=INC,
                extra_cuda_cflags=BASE + [f"-DDISPATCH_ENABLE_{kind.upper()}"], verbose=False)


print("Building 3 dispatcher modules (nvfp4/mxfp8/codesign)...", flush=True)
M_NV = build("nvfp4"); print("  nvfp4 OK", flush=True)
M_MX = build("mxfp8"); print("  mxfp8 OK", flush=True)
M_CD = build("codesign"); print("  codesign OK", flush=True)

SHAPES = [("qkv", 2048, 2048), ("gate", 8192, 2048), ("down", 2048, 8192)]
MLIST = [512, 2048, 4096, 8192]

# ---- measured best (available-tag) from the grid CSVs ----
NV_TAGMAP = {"ps_128x128_auto": "nvfp4:128x128_auto", "ps_128x128_pp": "nvfp4:128x128_pp",
             "ps_256x128_s3": "nvfp4:256x128_s3", "ps_128x64_auto": "nvfp4:128x64_auto"}
MX_TAGMAP = {"ps_128x128_auto": "mxfp8:128x128_auto", "ps_128x128_pp": "mxfp8:128x128_pp",
             "ps_128x64_auto": "mxfp8:128x64_auto"}
CD_TAGMAP = {"t128x128_nv4mx2": "codesign:128x128_nv4mx2", "t128x64_nv4mx2": "codesign:128x64_nv4mx2"}


def load_presf():
    best = {}  # (fmt,shape,M) -> {tag: min_us}
    lat = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(HERE, "r24_presf_search.csv"))):
        try:
            if not r["build"].startswith("OK"): continue
            u = float(r["min_us"])
            if u <= 0: continue
        except Exception:
            continue
        tm = NV_TAGMAP if r["fmt"] == "fp4" else MX_TAGMAP
        if r["cfg"] in tm:
            lat[(r["fmt"], r["shape"], int(r["M"]))][tm[r["cfg"]]] = u
    return lat


def load_cd():
    lat = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(HERE, "r24_cd_search.csv"))):
        try:
            if not r["build"].startswith("OK") or float(r["cos"]) < 0.999: continue
            u = float(r["min_us"])
            if u <= 0: continue
        except Exception:
            continue
        if r["cfg"] in CD_TAGMAP and r["r"] == "0.1250":  # production r=1/8
            lat[(r["shape"], int(r["M"]))][CD_TAGMAP[r["cfg"]]] = u
    return lat


PRESF = load_presf(); CD = load_cd()
fails = []
print(f"\n{'kind':9}{'shape':6}{'M':>6}  {'dispatcher_tag':26} {'measured_best':26} {'penalty':>8} {'cos':>9}")

for (sname, N, K) in SHAPES:
    for M in MLIST:
        # ---- NVFP4 ----
        p, oracle = mg.prep_pure_fp4(M, N, K, seed=1)
        tag = M_NV.tag(M, N, K)
        D = M_NV.run(p["A_packed"], p["SFA"], p["B_packed"], p["SFB"], M, N, K).float()
        cos = b1.cosine(D, oracle); del D
        ld = PRESF.get(("fp4", sname, M), {})
        bestu = min(ld.values()) if ld else float('nan')
        selu = ld.get(tag, float('nan'))
        pen = selu / bestu if bestu == bestu and bestu > 0 else float('nan')
        ok = cos >= 0.9999 and (pen != pen or pen <= 1.03)
        if not ok: fails.append(("nvfp4", sname, M, tag, cos, pen))
        print(f"{'nvfp4':9}{sname:6}{M:>6}  {tag:26} "
              f"{min(ld,key=ld.get) if ld else '--':26} {pen:8.3f} {cos:9.6f}{'' if ok else '  FAIL'}")
        del p, oracle; torch.cuda.empty_cache()

        # ---- MXFP8 ----
        p, oracle = mg.prep_pure_fp8(M, N, K, seed=1)
        tag = M_MX.tag(M, N, K)
        D = M_MX.run(p["A_bytes"], p["SFA"], p["B_bytes"], p["SFB"], M, N, K).float()
        cos = b1.cosine(D, oracle); del D
        ld = PRESF.get(("fp8", sname, M), {})
        bestu = min(ld.values()) if ld else float('nan')
        selu = ld.get(tag, float('nan'))
        pen = selu / bestu if bestu == bestu and bestu > 0 else float('nan')
        ok = cos >= 0.9999 and (pen != pen or pen <= 1.03)
        if not ok: fails.append(("mxfp8", sname, M, tag, cos, pen))
        print(f"{'mxfp8':9}{sname:6}{M:>6}  {tag:26} "
              f"{min(ld,key=ld.get) if ld else '--':26} {pen:8.3f} {cos:9.6f}{'' if ok else '  FAIL'}")
        del p, oracle; torch.cuda.empty_cache()

        # ---- CODESIGN (production r=1/8) ----
        Kh = K // 8; Kl = K - Kh
        d, oracle = None, None
        torch.manual_seed(1); dev = "cuda"
        Xl = torch.randn(M, Kl, device=dev); Wl = torch.randn(N, Kl, device=dev)
        Al_c, SFAl, Al_d = nv.quantize_nvfp4(Xl); Bl_c, SFBl, Bl_d = nv.quantize_nvfp4(Wl)
        Al_p = nv.pack_fp4(Al_c); Bl_p = nv.pack_fp4(Bl_c)
        Xh = torch.randn(M, Kh, device=dev); Wh = torch.randn(N, Kh, device=dev)
        Ah_b, SFAh, Ah_d = mx.quantize_mxfp8(Xh); Bh_b, SFBh, Bh_d = mx.quantize_mxfp8(Wh)
        s_x = (0.5 + torch.rand(M, device=dev)).float(); s_w = (0.5 + torch.rand(N, device=dev)).float()
        oracle = ((Al_d @ Bl_d.t() + Ah_d @ Bh_d.t()) * s_x.view(M, 1) * s_w.view(1, N))
        tag = M_CD.tag(M, N, K, Kh)
        D = M_CD.run(Al_p, SFAl, Bl_p, SFBl, Ah_b, SFAh, Bh_b, SFBh, M, N, Kl, Kh).float()
        cos = b1.cosine(D * s_x.view(M, 1) * s_w.view(1, N), oracle); del D
        ld = CD.get((sname, M), {})
        bestu = min(ld.values()) if ld else float('nan')
        selu = ld.get(tag, float('nan'))
        pen = selu / bestu if bestu == bestu and bestu > 0 else float('nan')
        ok = cos >= 0.9999 and (pen != pen or pen <= 1.03)
        if not ok: fails.append(("codesign", sname, M, tag, cos, pen))
        print(f"{'codesign':9}{sname:6}{M:>6}  {tag:26} "
              f"{min(ld,key=ld.get) if ld else '--':26} {pen:8.3f} {cos:9.6f}{'' if ok else '  FAIL'}")
        torch.cuda.empty_cache()

print()
if fails:
    print(f"DISPATCH_TEST_FAILED  ({len(fails)} cells)", flush=True)
    for x in fails: print("  FAIL", x, flush=True)
else:
    print("DISPATCH_TEST_ALL_PASS  (every cell: cosine>=0.9999 AND dispatcher-choice within 3% of measured best)", flush=True)
