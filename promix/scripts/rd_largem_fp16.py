"""rd-largem phase 3: is W4A4 e2e-competitive vs UNQUANTIZED FP16 at serving M?

Same-stack A/B: the real packed fused-kernel uniform-1/8 model vs the plain FP16
transformers model (no rotation/Hadamard/quant), interleaved paired-ratio e2e
prefill at batch x seq2048 (B = 1..16, M_eff 2048..32768). ratio = t_FP16 /
t_W4A4 (>1 means the quantized model is faster end-to-end). This is the honest
"does W4A4 beat FP16 at all, and at what M" number for OUR stack; the vLLM
NVFP4-vs-BF16 TTFT sweep is the same question for the peer serving stack.

    CUDA_VISIBLE_DEVICES=2 python3 promix/scripts/rd_largem_fp16.py \
        --model 3b --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml \
        --out /home/luofan/RD_LARGEM_FP16_3b.json
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import transformers

import rd_perf
from rd_largem import interleaved_prefill_b, paired_row, _iters_for, khigh_stats
from rd_relat import _free_wdeq
from rd_sweep import MODELS, resolve_paths
from promix.eval import ptq as ptqmod
from promix.eval.full_network_realkernel import _build_fp16_model
from promix.utils import cleanup_memory, DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=13)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"/home/luofan/RD_LARGEM_FP16_{args.model}.json"
    batches = [int(x) for x in args.batches.split(",")]

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = resolve_paths(config, args.model)
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))
    print(f"START rd_largem_fp16 model={args.model} batches={batches}", flush=True)

    print("Loading SM120 kernels...", flush=True)
    nv, mx, cd, qpf = rd_perf.load_kernels()
    dg = rd_perf.load_decode_gemv()

    print("Building FP16 reference + uniform-1/8 real-kernel models...", flush=True)
    fp16 = _build_fp16_model(config["model"]["name"])
    w4a4, packed = rd_perf.build_real(config, True, None, nv, mx, cd, qpf, dg)
    _free_wdeq(packed)
    cleanup_memory()
    vocab = w4a4.config.vocab_size
    print(f"  w4a4 avg_fp8={khigh_stats(w4a4)['avg_fp8']:.4f} "
          f"mem={torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

    out = {"meta": dict(model=args.model, name=config["model"]["name"],
                        batches=batches, seq=args.seq,
                        protocol="interleaved A/B paired: FP16 vs W4A4 uniform-1/8"),
           "rows": {}}

    for B in batches:
        M = B * args.seq
        it = _iters_for(B, args.iters)
        print(f"--- B={B} M_eff={M} iters={it} ---", flush=True)
        try:
            t0 = time.time()
            interleaved_prefill_b(fp16, w4a4, B, args.seq, vocab, 1, 0)
            af, aq = interleaved_prefill_b(fp16, w4a4, B, args.seq, vocab, it,
                                           args.warmup)
            e2e = paired_row(af, aq)   # ratio = t_FP16 / t_W4A4
            out["rows"][str(B)] = dict(batch=B, M_eff=M, e2e_fp16_over_w4a4=e2e,
                                       secs=round(time.time() - t0, 1))
            print(f"  t_FP16/t_W4A4 paired-med={e2e['ratio_pair_med']:.4f} "
                  f"(min-of-each {e2e['ratio_min']:.4f}) fp16_med={e2e['a_med']:.1f}ms "
                  f"w4a4_med={e2e['b_med']:.1f}ms ({time.time()-t0:.0f}s)", flush=True)
        except torch.cuda.OutOfMemoryError as ex:
            print(f"  OOM at B={B}: {ex}", flush=True)
            out["rows"][str(B)] = dict(batch=B, M_eff=M, oom=True)
            cleanup_memory()
            break
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"WROTE {args.out}", flush=True)
    print("RD_LARGEM_FP16_DONE", flush=True)


if __name__ == "__main__":
    main()
