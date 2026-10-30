"""Gate T0 — byte-level weight identity (vLLM-loaded vs exported PackedWeight).

For every promix-quantized layer in the vLLM-loaded model, torch.equal the
post-`process_weights_after_loading` packed buffers {W_lo_packed, physSFB_lo,
W_hi_bytes, physSFB_hi} and the per-tensor g against EXPECTED buffers computed
directly from the exported safetensors (cat members along N + the same kernel
prepack). Any mismatch = loader/merge bug -> hard fail.

venv run:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=6 \
  /home/luofan/baselines/venv/bin/python -m promix.vllm_integ.gates.gate_t0 \
      --ckpt /home/luofan/vllm_integ/promix-llama-3.2-3b
"""

import argparse
import json
import os

import torch

from promix.vllm_integ.gates.common import (
    build_llm, promix_layers, expected_merged_buffers,
)
from promix.vllm_integ.kernels import load_kernels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from safetensors.torch import load_file
    state = load_file(os.path.join(args.ckpt, "model.safetensors"))
    cd, _ = load_kernels()

    llm = build_llm(args.ckpt)

    def check(model):
        results = {}
        for prefix, mod in promix_layers(model):
            mql = mod.promix_mql
            exp = expected_merged_buffers(state, prefix, cd,
                                          device=mql.W_lo_packed.device)
            r = {
                "W_lo_packed": torch.equal(mql.W_lo_packed, exp["W_lo_packed"]),
                "physSFB_lo": torch.equal(mql.physSFB_lo, exp["physSFB_lo"]),
                "W_hi_bytes": torch.equal(mql.W_hi_bytes, exp["W_hi_bytes"]),
                "physSFB_hi": torch.equal(mql.physSFB_hi, exp["physSFB_hi"]),
                "g": mql.g_w == exp["g"],
                "shape": (mql.N, mql.K, mql.K_low, mql.K_high) ==
                         (exp["N"], exp["K_low"] + exp["K_high"],
                          exp["K_low"], exp["K_high"]),
            }
            if "column_order" in exp:
                r["column_order"] = torch.equal(mod.promix_col_order,
                                                exp["column_order"])
            results[prefix] = r
        return results

    ret = llm.apply_model(check)
    results = ret[0] if isinstance(ret, list) else ret
    nfail = 0
    for prefix in sorted(results):
        r = results[prefix]
        bad = [k for k, v in r.items() if v is not True]
        if bad:
            nfail += 1
            print(f"T0 FAIL {prefix}: {bad}", flush=True)
    verdict = "PASS" if nfail == 0 and len(results) > 0 else "FAIL"
    print(f"GATE_T0_{verdict} layers={len(results)} mismatched={nfail}", flush=True)
    out = args.out or os.path.join(args.ckpt, "gate_t0.json")
    with open(out, "w") as f:
        json.dump({"verdict": verdict, "layers": len(results), "mismatched": nfail,
                   "detail": {k: {kk: bool(vv) for kk, vv in v.items()}
                              for k, v in results.items()}}, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
