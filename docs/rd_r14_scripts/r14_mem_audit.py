#!/usr/bin/env python3
"""rd-r14 secondary finding: audit the r=1/4 load-time memory bloat
(engine 'consumed' 18.45 GiB vs 4.7-6.6 GiB at every other r).

Replays plugin.process_weights_after_loading's allocation sequence on the
REAL checkpoint buffers (safetensors), per layer, at stage granularity,
recording torch allocated/reserved + driver free after each stage.

Usage: --ckpt <dir> [--layers N]
"""
import argparse
import json

import torch
from safetensors.torch import load_file


def mem(tag, rows):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    rows.append({"tag": tag,
                 "alloc_gib": torch.cuda.memory_allocated() / 2**30,
                 "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                 "driver_used_gib": (total - free) / 2**30})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from promix.vllm_integ.kernels import load_kernels, load_decode_gemv
    from promix.vllm_integ.marlin_w4a16 import (prepack_marlin_high,
                                                prepack_marlin_low)
    cd, qpf = load_kernels()
    load_decode_gemv()

    cfg = json.load(open(args.ckpt + "/config.json"))["quantization_config"]
    k_high = cfg["k_high"]
    sd = load_file(args.ckpt + "/model.safetensors")

    rows = []
    mem("start", rows)
    keep = []   # emulate: everything a layer keeps alive stays alive
    prefixes = sorted({k.rsplit(".", 1)[0] for k in sd
                       if k.endswith("weight_lo_packed")})
    print(len(prefixes), "quantized linears", flush=True)
    for i, pref in enumerate(prefixes):
        key = pref.replace("model.layers.", "layers.")
        kh = k_high[key]
        W_lo = sd[pref + ".weight_lo_packed"].cuda()
        S_lo = sd[pref + ".weight_lo_scale"].cuda()
        W_hi = sd[pref + ".weight_hi_bytes"].cuda()
        S_hi = sd[pref + ".weight_hi_scale"].cuda()
        g = float(sd[pref + ".weight_global_scale"][0])
        N, K_low = W_lo.shape[0], W_lo.shape[1] * 2
        K = K_low + kh
        assert kh == W_hi.shape[1]

        physSFB_lo = cd.prepack_sf_fp4_b(S_lo.contiguous(), N, K_low)
        physSFB_hi = cd.prepack_sf_fp8_b(S_hi.contiguous(), N, kh)
        if i < 4 or i % 28 == 0:
            mem(f"{key}.sfprepack", rows)
        S_lo_dec = (S_lo.view(torch.float8_e4m3fn).float() * g).contiguous()
        S_hi_dec = (torch.exp2(S_hi.float() - 127.0) * g).contiguous()
        if i < 4 or i % 28 == 0:
            mem(f"{key}.gemvscales", rows)
        ml = prepack_marlin_low(W_lo, S_lo.contiguous(), g, N, K_low)
        if i < 4 or i % 28 == 0:
            mem(f"{key}.marlin_low", rows)
        mh = prepack_marlin_high(W_hi.contiguous(), S_hi.contiguous(), g, N, kh)
        if i < 4 or i % 28 == 0:
            mem(f"{key}.marlin_high", rows)
        keep.append((W_lo, S_lo, W_hi, S_hi, physSFB_lo, physSFB_hi,
                     S_lo_dec, S_hi_dec, ml, mh))
    mem("end_all_layers", rows)
    torch.cuda.empty_cache()
    mem("after_empty_cache", rows)

    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v)
               for k, v in r.items()}, flush=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
