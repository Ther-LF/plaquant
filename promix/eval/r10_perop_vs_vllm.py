"""Round-10 — per-op profile: OUR FP16-rest ops vs vLLM's BF16-rest ops.

Isolates the NON-GEMM prefill stack op-by-op on IDENTICAL shapes, timing our
torch-native implementation (what the HF-based ProMix forward actually runs)
against vLLM's optimized CUDA kernels, in the SAME process so the GPU/clock
conditions are shared.

Ops compared (per decoder layer unless noted):
  * RMSNorm (x2/layer + 1 final): HF LlamaRMSNorm (fp32-upcast) vs
    vLLM `rms_norm`; plus the fused residual-add+RMSNorm variant
    (torch add + HF norm) vs vLLM `fused_add_rms_norm`.
  * RoPE: HF apply_rotary_pos_emb vs vLLM `rotary_embedding`.
  * Attention (QK/softmax/PV): F.scaled_dot_product_attention (HF sdpa backend,
    GQA via repeat_kv) vs vLLM `flash_attn_varlen_func` (native GQA).
  * residual add: torch a+b (free inside fused_add_rms_norm on the vLLM side).

Model-level ms = per-op ms x occurrence count (2 norms + 1 rope + 1 attn per
layer, +1 final norm). lm_head handled separately in r10_lmhead (needs weights).

Run inside the vLLM baselines venv (has torch + vllm._custom_ops + vLLM flash):
  /home/luofan/baselines/venv/bin/python r10_perop_vs_vllm.py \
      --sizes 3b,8b --ms 2048,4096,8192 --out /home/luofan/_r10/r10_perop.json
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

CFG = {
    # hidden, n_heads, n_kv, head_dim, intermediate, n_layers, vocab, rope_theta
    "1b": dict(hidden=2048, nh=32, nkv=8, hd=64, inter=8192, nl=16, vocab=128256, theta=500000.0),
    "3b": dict(hidden=3072, nh=24, nkv=8, hd=128, inter=8192, nl=28, vocab=128256, theta=500000.0),
    "8b": dict(hidden=4096, nh=32, nkv=8, hd=128, inter=14336, nl=32, vocab=128256, theta=500000.0),
}
DEV = "cuda"


def wait_idle(util_thresh=8, need=3, timeout_s=180):
    import subprocess
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    t0 = time.time(); ok = 0
    while time.time() - t0 < timeout_s:
        try:
            q = subprocess.check_output(
                ["nvidia-smi", "--id=" + dev, "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
            util = int(q)
        except Exception:
            return
        if util <= util_thresh:
            ok += 1
            if ok >= need:
                return
        else:
            ok = 0
        time.sleep(1.0)


def tmed(fn, iters=50, warmup=10):
    """CUDA-event median (and min) ms of a callable."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    wait_idle()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2], ts[0]


# ---------------- our (torch-native) ops = what the HF ProMix forward runs -----
class HFRMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim, device=DEV, dtype=torch.float16))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(dt)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def hf_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
    q_e = (q * cos) + (rotate_half(q) * sin)
    k_e = (k * cos) + (rotate_half(k) * sin)
    return q_e, k_e


def repeat_kv(x, n):
    b, h, s, d = x.shape
    if n == 1:
        return x
    return x[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def run_size(sz, Ms, dtype, iters):
    c = CFG[sz]
    hidden, nh, nkv, hd, nl = c["hidden"], c["nh"], c["nkv"], c["hd"], c["nl"]
    import vllm._custom_ops as ops
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    eps = 1e-5
    rms = HFRMSNorm(hidden, eps).to(DEV)
    w = rms.weight.data.contiguous()
    rows = []
    for M in Ms:
        torch.manual_seed(M)
        x = torch.randn(1, M, hidden, device=DEV, dtype=dtype) * 0.1
        res = torch.randn_like(x)

        # ---- RMSNorm ----
        def ours_rms():
            return rms(x)
        x2d = x.reshape(M, hidden).contiguous()
        out2d = torch.empty_like(x2d)
        def vllm_rms():
            ops.rms_norm(out2d, x2d, w, eps)
            return out2d
        rms_ours = tmed(ours_rms, iters)
        rms_vllm = tmed(vllm_rms, iters)

        # ---- fused add + RMSNorm ----
        def ours_add_rms():
            h = res + x
            return rms(h), h
        xf = x2d.clone(); rf = res.reshape(M, hidden).contiguous().clone()
        def vllm_add_rms():
            xi = xf.clone(); ri = rf.clone()
            ops.fused_add_rms_norm(xi, ri, w, eps)
            return xi, ri
        addrms_ours = tmed(ours_add_rms, iters)
        addrms_vllm = tmed(vllm_add_rms, iters)

        # ---- RoPE ----
        q = torch.randn(1, nh, M, hd, device=DEV, dtype=dtype)
        k = torch.randn(1, nkv, M, hd, device=DEV, dtype=dtype)
        inv = 1.0 / (c["theta"] ** (torch.arange(0, hd, 2, device=DEV).float() / hd))
        pos = torch.arange(M, device=DEV).float()
        freqs = torch.outer(pos, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None].to(dtype); sin = emb.sin()[None].to(dtype)
        def ours_rope():
            return hf_rope(q, k, cos, sin)
        # vLLM rotary: query [tokens, nh*hd], key [tokens, nkv*hd]; cos_sin_cache [max_pos, hd]
        cos_half = freqs.cos(); sin_half = freqs.sin()
        cos_sin_cache = torch.cat((cos_half, sin_half), dim=-1).to(dtype)  # [M, hd]
        qv = torch.randn(M, nh * hd, device=DEV, dtype=dtype)
        kv = torch.randn(M, nkv * hd, device=DEV, dtype=dtype)
        positions = torch.arange(M, device=DEV, dtype=torch.long)
        def vllm_rope():
            qi = qv.clone(); ki = kv.clone()
            ops.rotary_embedding(positions, qi, ki, hd, cos_sin_cache, True)
            return qi, ki
        rope_ours = tmed(ours_rope, iters)
        try:
            rope_vllm = tmed(vllm_rope, iters)
        except Exception as ex:
            rope_vllm = (float("nan"), float("nan")); print("  vllm rope err", repr(ex))

        # ---- attention ----
        qh = torch.randn(1, nh, M, hd, device=DEV, dtype=dtype)
        kh = torch.randn(1, nkv, M, hd, device=DEV, dtype=dtype)
        vh = torch.randn(1, nkv, M, hd, device=DEV, dtype=dtype)
        scale = 1.0 / math.sqrt(hd)
        def ours_attn():
            kr = repeat_kv(kh, nh // nkv); vr = repeat_kv(vh, nh // nkv)
            return F.scaled_dot_product_attention(qh, kr, vr, is_causal=True, scale=scale)
        # flash varlen: q [M, nh, hd], k/v [M, nkv, hd]
        qf = qh.squeeze(0).transpose(0, 1).contiguous()  # [M, nh, hd]
        kf = kh.squeeze(0).transpose(0, 1).contiguous()
        vf = vh.squeeze(0).transpose(0, 1).contiguous()
        cu = torch.tensor([0, M], device=DEV, dtype=torch.int32)
        def vllm_attn():
            return flash_attn_varlen_func(qf, kf, vf, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                          max_seqlen_q=M, max_seqlen_k=M,
                                          softmax_scale=scale, causal=True)
        attn_ours = tmed(ours_attn, iters)
        try:
            attn_vllm = tmed(vllm_attn, iters)
        except Exception as ex:
            attn_vllm = (float("nan"), float("nan")); print("  vllm attn err", repr(ex))

        # ---- residual add ----
        add_ours = tmed(lambda: res + x, iters)

        # model-level totals (per-forward): 2 norms/layer + 1 final; 1 rope+1 attn/layer;
        # 2 residual adds/layer. fused_add_rms covers (add + norm) for the 2 per layer.
        def modelms(per, n):
            return per * n
        row = dict(
            size=sz, M=M, n_layers=nl,
            rms_ours_ms=rms_ours[0], rms_vllm_ms=rms_vllm[0],
            addrms_ours_ms=addrms_ours[0], addrms_vllm_ms=addrms_vllm[0],
            rope_ours_ms=rope_ours[0], rope_vllm_ms=rope_vllm[0],
            attn_ours_ms=attn_ours[0], attn_vllm_ms=attn_vllm[0],
            add_ours_ms=add_ours[0],
            # model-level (per full forward)
            model_rms_ours=modelms(rms_ours[0], 2 * nl + 1),
            model_rms_vllm=modelms(rms_vllm[0], 2 * nl + 1),
            model_addrms_ours=modelms(addrms_ours[0], 2 * nl),   # 2 add+norm/layer
            model_addrms_vllm=modelms(addrms_vllm[0], 2 * nl),
            model_rope_ours=modelms(rope_ours[0], nl),
            model_rope_vllm=modelms(rope_vllm[0], nl),
            model_attn_ours=modelms(attn_ours[0], nl),
            model_attn_vllm=modelms(attn_vllm[0], nl),
        )
        print(f"[{sz} M={M}] RMSNorm ours {rms_ours[0]*1e3:.1f}us vllm {rms_vllm[0]*1e3:.1f}us | "
              f"add+RMS ours {addrms_ours[0]*1e3:.1f}us vllm {addrms_vllm[0]*1e3:.1f}us | "
              f"RoPE ours {rope_ours[0]*1e3:.1f}us vllm {rope_vllm[0]*1e3:.1f}us | "
              f"attn ours {attn_ours[0]:.3f}ms vllm {attn_vllm[0]:.3f}ms", flush=True)
        print(f"         MODEL(x{nl}): norms ours {row['model_rms_ours']:.2f} vllm {row['model_rms_vllm']:.2f}ms | "
              f"add+norm ours {row['model_addrms_ours']:.2f} vllm {row['model_addrms_vllm']:.2f}ms | "
              f"rope ours {row['model_rope_ours']:.2f} vllm {row['model_rope_vllm']:.2f}ms | "
              f"attn ours {row['model_attn_ours']:.2f} vllm {row['model_attn_vllm']:.2f}ms", flush=True)
        rows.append(row)
        del x, res, q, k, qh, kh, vh, qf, kf, vf
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="3b,8b")
    ap.add_argument("--ms", default="2048,4096,8192")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="/home/luofan/_r10/r10_perop.json")
    args = ap.parse_args()
    dtype = dict(float16=torch.float16, bfloat16=torch.bfloat16)[args.dtype]
    Ms = [int(x) for x in args.ms.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {"dtype": args.dtype, "ms": args.ms, "rows": []}
    for sz in args.sizes.split(","):
        print(f"==== size {sz} (dtype {args.dtype}) ====", flush=True)
        out["rows"].extend(run_size(sz, Ms, dtype, args.iters))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("R10_PEROP_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
