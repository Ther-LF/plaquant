"""Gate T1 (part 2, container) — replay captured vLLM layer inputs through the
promix reference implementation and assert bit-exact outputs.

For every captured merged layer: rebuild the layer from the EXPORTED checkpoint
(cat members along N + kernel prepack, i.e. the same bytes T0 verified vLLM
loaded), apply the reference glue (down_proj online Hadamard fp16/fold,
o_proj column reorder) + MixedQuantLinear (container torch-2.11 exts), and
compare against the vLLM-side y (venv torch-2.13 exts):

    torch.equal        -> PASS (the bar)
    cosine >= 0.9999   -> only acceptable WITH a documented root cause

Additionally cross-checks merged-GEMM vs per-projection GEMMs: each member's
MixedQuantLinear output must bit-match the corresponding column slice of the
merged output (validates the N-merge itself).

container run:
  docker exec -w /home/luofan/plaquant-vllm-integ -e CUDA_VISIBLE_DEVICES=6 \
    -e PYTHONPATH=/home/luofan/plaquant-vllm-integ luofan-plaquant \
    python -m promix.vllm_integ.gates.gate_t1_replay \
      --ckpt /home/luofan/vllm_integ/promix-llama-3.2-3b \
      --capture /home/luofan/vllm_integ/t1_capture.pt
"""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO)
sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

from promix.eval.full_network_realkernel import load_kernels  # noqa: E402
from promix.quantize.mixed_linear import MixedQuantLinear  # noqa: E402
from promix.quantize.weight_pack import PackedWeight  # noqa: E402
from promix.quantize.hadamard import matmul_hadU_cuda, get_hadK  # noqa: E402

MERGE_MEMBERS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "o_proj": ("o_proj",),
    "down_proj": ("down_proj",),
}


def cosine(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def build_mql(state, members, cd, qpf, dev):
    def cat(sfx):
        return torch.cat([state[m + "." + sfx].to(dev) for m in members], 0).contiguous()

    W_lo = cat("weight_lo_packed")
    SF_lo = cat("weight_lo_scale")
    N, K_low = W_lo.shape[0], W_lo.shape[1] * 2
    if (members[0] + ".weight_hi_bytes") in state:
        W_hi, SF_hi = cat("weight_hi_bytes"), cat("weight_hi_scale")
        K_high = W_hi.shape[1]
        phys_hi = cd.prepack_sf_fp8_b(SF_hi, N, K_high)
    else:
        K_high = 0
        W_hi = torch.empty((N, 0), dtype=torch.uint8, device=dev)
        phys_hi = torch.empty((0,), dtype=torch.uint8, device=dev)
    gs = torch.cat([state[m + ".weight_global_scale"].to(dev) for m in members])
    assert torch.unique(gs).numel() == 1
    pk = PackedWeight(N=N, K=K_low + K_high, K_low=K_low, K_high=K_high,
                      W_lo_packed=W_lo, physSFB_lo=cd.prepack_sf_fp4_b(SF_lo, N, K_low),
                      W_hi_bytes=W_hi, physSFB_hi=phys_hi, W_deq=None,
                      bias=None, g_w=float(gs[0]))
    return MixedQuantLinear(pk, nv=None, mx=None, cd=cd,
                            out_dtype=torch.float16, qpf=qpf).to(dev)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda"

    from safetensors.torch import load_file
    state = load_file(os.path.join(args.ckpt, "model.safetensors"))
    with open(os.path.join(args.ckpt, "config.json")) as f:
        qcfg = json.load(f)["quantization_config"]
    cap = torch.load(args.capture, weights_only=False)["captures"]
    nv, mx, cd, qpf = load_kernels()

    rows, nexact, ncos = [], 0, 0
    for prefix in sorted(cap):
        x, y_vllm = cap[prefix]
        x = x.to(dev)
        y_vllm = y_vllm.to(dev)
        kind = prefix.split(".")[-1]
        base = prefix.rsplit(".", 1)[0]
        members = [f"{base}.{m}" for m in MERGE_MEMBERS[kind]]
        mql = build_mql(state, members, cd, qpf, dev)

        xr = x
        if kind == "down_proj" and qcfg.get("hadamard_down_proj", True):
            had_K, Kfac = get_hadK(mql.K)
            xr = matmul_hadU_cuda(
                x, None if had_K is None else had_K.to(dev), Kfac,
                fold_scale=qcfg.get("fold_had_scale", True)).to(x.dtype)
        elif kind == "o_proj" and qcfg.get("o_proj_reorder", True):
            order = state[members[0] + ".column_order"].to(dev, torch.long)
            xr = x[..., order]
        y_ref = mql(xr)

        eq = torch.equal(y_ref, y_vllm)
        cos = 1.0 if eq else cosine(y_ref, y_vllm)
        maxabs = 0.0 if eq else (y_ref.float() - y_vllm.float()).abs().max().item()

        # merged-vs-split cross-check (bit-exactness of the N-merge itself)
        split_eq = True
        if len(members) > 1:
            off = 0
            for m in members:
                sub = build_mql(state, [m], cd, qpf, dev)
                y_m = sub(xr)
                split_eq &= torch.equal(y_m, y_ref[..., off:off + sub.N])
                off += sub.N
                del sub
        rows.append(dict(prefix=prefix, bit_exact=bool(eq), cosine=cos,
                         max_abs=maxabs, split_eq=bool(split_eq)))
        nexact += int(eq)
        ncos += int(eq or cos >= 0.9999)
        flag = "OK " if eq else ("cos" if cos >= 0.9999 else "BAD")
        print(f"T1 {flag} {prefix:48s} eq={eq} cos={cos:.7f} "
              f"maxabs={maxabs:.4g} split_eq={split_eq}", flush=True)
        del mql
        torch.cuda.empty_cache()

    n = len(rows)
    all_split = all(r["split_eq"] for r in rows)
    verdict = ("PASS" if nexact == n and all_split and n > 0 else
               ("COND" if ncos == n and all_split and n > 0 else "FAIL"))
    print(f"GATE_T1_{verdict} layers={n} bit_exact={nexact} cos_ok={ncos} "
          f"split_merge_exact={all_split}", flush=True)
    out = args.out or os.path.join(args.ckpt, "gate_t1.json")
    with open(out, "w") as f:
        json.dump({"verdict": verdict, "layers": n, "bit_exact": nexact,
                   "cos_ok": ncos, "split_merge_exact": all_split,
                   "detail": rows}, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
