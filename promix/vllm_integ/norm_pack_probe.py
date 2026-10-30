#!/usr/bin/env python3
"""vllm-perf r3 probe (lever 2): fused vLLM-RMSNorm + quant/pack producer.

Gates (per hidden x M x weight-kind):
  1. NORM bit gate   xn / residual_out torch.equal to vLLM's own
                     torch.ops._C.fused_add_rms_norm on cloned buffers —
                     proves the vendored replica (vec8 loops + cub BlockReduce
                     + the num_tokens-dependent block-size rule) is EXACT.
  2. PACK bit gate   packed code bytes torch.equal to quant_pack_float_mixed(xn)
                     AND the consuming fused GEMM outputs torch.equal (the SFA
                     raw compare is only meaningful at M%128==0 — atom padding
                     is uninitialized; the GEMM-output compare covers it, r2
                     lesson).
  3. BENCH           graph-replay: [vLLM fused_add_rms_norm + qpfm] chain vs
                     the ONE fused kernel, per M.

Run (vLLM venv, idle GPU):
  CUDA_VISIBLE_DEVICES=7 /home/luofan/baselines/venv/bin/python \
      -m promix.vllm_integ.norm_pack_probe [--out out.json]
"""
import argparse
import json

import torch

from promix.vllm_integ.kernels import load_kernels, load_qpfm, load_norm_pack
from promix.vllm_integ.decode_probe import make_packed, bench_graph
from promix.vllm_integ.ops import _sfa_numel

# (tag, hidden, K_high, N_gemm) — the 3B/8B qkv + gate_up norm-fed sites.
SITES = [
    ("3b_h3072_qkv", 3072, 384, 5120),
    ("3b_h3072_gu", 3072, 384, 16384),
    ("8b_h4096_qkv", 4096, 512, 6144),
]
MS = (1, 2, 4, 8, 16, 64, 255, 256, 512, 2048)
EPS = 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    cd, qpf = load_kernels()
    qpfm = load_qpfm()
    vrq = load_norm_pack()
    import vllm._custom_ops as vllm_ops  # registers torch.ops._C

    results = {"sites": {}}
    allpass = True
    for tag, H, K_high, N_gemm in SITES:
        K_low = H - K_high
        pk = make_packed(N_gemm, H, K_high, cd)
        res = {"H": H, "K_high": K_high, "m": {}}
        for M in MS:
            x = (torch.randn(M, H, device="cuda") * 0.5).half()
            residual = (torch.randn(M, H, device="cuda") * 0.5).half()
            row = {}
            for wkind in ("ones", "rand"):
                w = (torch.ones(H, device="cuda").half() if wkind == "ones"
                     else (torch.randn(H, device="cuda") * 0.1 + 1.0).half())
                # vLLM reference (mutates clones)
                inp = x.clone()
                rs = residual.clone()
                torch.ops._C.fused_add_rms_norm(inp, rs, w, EPS)
                # fused
                out = vrq.vllm_add_rms_norm_quant_pack(x, residual, w, N_gemm,
                                                       K_low, K_high, EPS)
                code_lo, sfa_lo, code_hi, sfa_hi, xn, rout = out
                norm_eq = bool(torch.equal(xn, inp))
                res_eq = bool(torch.equal(rout, rs))
                # reference pack of the SAME xn
                a = qpfm.quant_pack_float_mixed(xn.contiguous(), M, N_gemm,
                                                K_low, K_high)
                code_eq = bool(torch.equal(code_lo.flatten(), a[0].flatten())
                               and torch.equal(code_hi.flatten(), a[2].flatten()))
                sfa_size_ok = (sfa_lo.numel() == _sfa_numel(M, K_low, 16)
                               and sfa_hi.numel() == _sfa_numel(M, K_high, 32)
                               and a[1].numel() == sfa_lo.numel()
                               and a[3].numel() == sfa_hi.numel())
                # consuming-GEMM byte gate (covers SFA incl. padding semantics)
                d_f = cd.fused_gemm_prepacked(
                    code_lo, sfa_lo, pk.W_lo_packed, pk.physSFB_lo,
                    code_hi, sfa_hi, pk.W_hi_bytes, pk.physSFB_hi,
                    M, N_gemm, K_low, K_high, pk.g_w)
                d_r = cd.fused_gemm_prepacked(
                    a[0].view(M, K_low // 2), a[1], pk.W_lo_packed, pk.physSFB_lo,
                    a[2].view(M, K_high), a[3], pk.W_hi_bytes, pk.physSFB_hi,
                    M, N_gemm, K_low, K_high, pk.g_w)
                gemm_eq = bool(torch.equal(d_f, d_r))
                ok = norm_eq and res_eq and code_eq and sfa_size_ok and gemm_eq
                allpass = allpass and ok
                row[wkind] = {"norm_eq": norm_eq, "res_eq": res_eq,
                              "code_eq": code_eq, "sfa_size_ok": sfa_size_ok,
                              "gemm_eq": gemm_eq}
            if not args.quick:
                w = torch.ones(H, device="cuda").half()
                inp = x.clone()
                rs = residual.clone()

                def chain():
                    torch.ops._C.fused_add_rms_norm(inp, rs, w, EPS)
                    qpfm.quant_pack_float_mixed(inp, M, N_gemm, K_low, K_high)

                row["t_chain"] = bench_graph(chain)
                row["t_fused"] = bench_graph(
                    lambda: vrq.vllm_add_rms_norm_quant_pack(
                        x, residual, w, N_gemm, K_low, K_high, EPS))
            res["m"][M] = row
            line = (f"{tag:14s} M={M:5d} " +
                    " ".join(f"{k}:{'PASS' if all(row[k].values()) else row[k]}"
                             for k in ("ones", "rand")))
            if not args.quick:
                line += (f"  chain={row['t_chain']:6.1f}us fused={row['t_fused']:6.1f}us "
                         f"({row['t_chain']/row['t_fused']:.2f}x)")
            print(line, flush=True)
        results["sites"][tag] = res

    print(f"NORM_PACK_GATE_{'ALL_PASS' if allpass else 'FAIL'}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
