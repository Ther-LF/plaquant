"""rd-oppareto (E4): merge the suite JSONs (docs/rd_oppareto_data/) + the old
pre-optimization ratio curve (docs/rd_perfpuzzle_fused_khigh.csv) into the
plot-ready docs/rd_operator_pareto.csv.

Long format, one row per plotted element:
  kind      curve (ours, one row per r) | point (competitor) | old_curve
  method    promix_fused / torch_fp4 / torch_fp8 / bf16_cublas / fp16_cublas /
            vllm_nvfp4 / arcquant_s{S} / micromix_{cfg} / sharq / promix_pre_r5
  x = latency_ms, y = eff_bits (bits-or-accuracy-proxy axis).

Run from repo root: python3 promix/eval/rd_oppareto_agg.py
"""
import csv, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
D = os.path.join(REPO, "docs", "rd_oppareto_data")
OLD = os.path.join(REPO, "docs", "rd_perfpuzzle_fused_khigh.csv")
OUT = os.path.join(REPO, "docs", "rd_operator_pareto.csv")

COLS = ["kind", "method", "model", "shape", "M", "N", "K", "r_nominal",
        "r_exact", "K_high", "latency_ms", "latency_min_ms", "graph_ms",
        "tflops", "eff_bits", "cosine_vs_fp32", "dispatch", "note"]


def load(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        print(f"MISSING {p} — skipped")
        return []
    with open(p) as f:
        return json.load(f)["rows"]


def row(kind, method, r, **kw):
    d = {c: "" for c in COLS}
    d.update(kind=kind, method=method, model=r.get("model", ""),
             shape=r.get("shape", ""), M=r.get("M", ""), N=r.get("N", ""),
             K=r.get("K", ""), latency_ms=r.get("gemm_ms", ""),
             latency_min_ms=r.get("gemm_min_ms", ""),
             cosine_vs_fp32=r.get("cosine_vs_fp32", ""))
    d.update(kw)
    return d


def main():
    out = []

    # ---- ours: full curve + fp16 anchor ----
    ours_rows = []
    for p in sorted(glob.glob(os.path.join(D, "ours_curve*.json"))):
        with open(p) as f:
            ours_rows += json.load(f)["rows"]
    for r in ours_rows:
        if r.get("dispatch") == "fp16_cublas":
            out.append(row("point", "fp16_cublas", r, eff_bits=16.0,
                           tflops=r.get("tflops", ""), dispatch="fp16_cublas"))
            continue
        out.append(row("curve", "promix_fused", r,
                       r_nominal=r["r_nominal"], r_exact=round(r["r_exact"], 6),
                       K_high=r["K_high"], graph_ms=r.get("graph_ms", ""),
                       tflops=r.get("tflops", ""), eff_bits=round(r["eff_bits"], 4),
                       dispatch=r.get("dispatch", ""), note=r.get("note", "")))

    # ---- torch-native points ----
    for fn in ("torch_m2048.json", "torch_m8192.json"):
        for r in load(fn):
            out.append(row("point", r["method"], r, eff_bits=r["eff_bits"],
                           tflops=r.get("tflops", "")))

    # ---- vLLM NVFP4 ----
    for fn in ("vllm_m2048.json", "vllm_m8192.json"):
        for r in load(fn):
            out.append(row("point", "vllm_nvfp4", r, eff_bits=4.5,
                           tflops=r.get("gemm_tflops", "")))

    # ---- ARCQuant (S-bracketed; shipped S is calibration-adaptive, <=512) ----
    for fn in ("arcquant_m2048.json", "arcquant_m8192.json"):
        for r in load(fn):
            S = r["S"]
            bits = 4.5 * (r["K"] + S) / r["K"]
            note = ("shipped operating point caps S<=512; S bracket "
                    "{0,256,512}" if S == 512 else "")
            out.append(row("point", f"arcquant_s{S}", r,
                           eff_bits=round(bits, 4),
                           tflops=r.get("gemm_tflops_base", ""), note=note))

    # ---- MicroMix (segmented; CUDA-13 correctness unvalidated as shipped) ----
    for fn in ("micromix_m2048.json", "micromix_m8192.json"):
        for r in load(fn):
            if "error" in r:
                continue
            bits = r["avg_bits"] + 0.25   # +g32 scale overhead like our 8.25/4.25 convention
            out.append(row("point", f"micromix_{r['config']}", r,
                           eff_bits=round(bits, 4),
                           tflops=r.get("gemm_tflops", ""),
                           note="correctness unvalidated under CUDA 13 as shipped"))

    # ---- SharQ (dual-GEMM: sparse backbone + dense residual) ----
    for fn in ("sharq_m2048.json", "sharq_m8192.json"):
        for r in load(fn):
            if "error" in r or not r.get("gemm_ms"):
                continue
            out.append(row("point", "sharq", r, eff_bits=4.5,
                           tflops=r.get("gemm_tflops_base", ""),
                           note="4.5b payload; GEMM work ~1.85-2.07x dense FP4 "
                                "(sparse main + residual, 2 launches)"))

    # ---- old pre-R1-R5 curve (rd-perfpuzzle, 3B M=2048/4096) ----
    scen = {"qkv_3b": ("3B", "qkvo"), "gate_3b": ("3B", "gate_up"),
            "down_3b": ("3B", "down")}
    if os.path.exists(OLD):
        with open(OLD) as f:
            for r in csv.DictReader(f):
                if r["scenario"] not in scen or r["M"] not in ("2048",):
                    continue
                mdl, shp = scen[r["scenario"]]
                kh, K = int(r["K_high"]), int(r["K_total"])
                d = {c: "" for c in COLS}
                d.update(kind="old_curve", method="promix_pre_r5", model=mdl,
                         shape=shp, M=int(r["M"]), N=int(r["N"]), K=K,
                         r_nominal=float(r["r"]), r_exact=round(kh / K, 6),
                         K_high=kh, latency_ms=float(r["lat_auto_ms"]),
                         eff_bits=round((4.5 * (K - kh) + 8.25 * kh) / K, 4),
                         dispatch=r["auto_tag"],
                         note="pre-R1-R5 kernel (rd-perfpuzzle 2026-10-29)")
                out.append(d)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)
    print(f"WROTE {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
