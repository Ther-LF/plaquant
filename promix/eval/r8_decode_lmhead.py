"""Round-8 Part B lever: MXFP8 (W8A8) lm_head in the GRAPHED decode path.

At M=1 decode the FP16 lm_head is a large vocab GEMV (1B: 128256x2048 = 525 MB /
token; 3B: 128256x3072 = 788 MB / token) read every step. W8 MXFP8 halves that
weight traffic and runs on the SM120 block-scaled tensor cores. W8 lm_head is the
PPL-safe point (1B +0.048 <= gate; W4 fails). This harness stacks it on top of the
round-7 QKV+gate/up decode merge and measures decode b1/b8 vs FP16-graph, plus the
wikitext-2 PPL gate for the lm_head change (real-decoder FP16-head vs MXFP8-head).
"""
import argparse, json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
CUTLASS = os.path.join(REPO, "third_party", "cutlass")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

import transformers
from torch.utils.cpp_extension import load

from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import load_config
from promix.quantize import full_network as fn
from promix.eval.full_network_realkernel import load_kernels, _build_fp16_model, HIGH_FRACTION
from promix.eval.decode_throughput import set_decode_gemv, load_decode_gemv, tokenize_prompt
from promix.eval.evaluator import evaluate_ppl
from promix.eval.data import get_wikitext2
from promix.eval import decode_merge as dm


def load_lmh():
    inc = [KERN, os.path.join(CUTLASS, "include"), os.path.join(CUTLASS, "tools/util/include"),
           os.path.join(CUTLASS, "examples/common")]
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    return load(name="lmhead_mxfp8_ext", sources=[os.path.join(KERN, "lmhead_mxfp8_ext.cu")],
                extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--ppl", action="store_true", help="also run the lm_head PPL gate")
    ap.add_argument("--baseline_ppl", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    config = load_config(args.config)
    print(f"decode-lmhead | config={args.config} batches={batches}", flush=True)

    nv, mx, cd, qpf = load_kernels()
    dg = load_decode_gemv()
    lmh = load_lmh()
    model, packed = dm.build_real(config, nv, mx, cd, qpf, dg)
    set_decode_gemv(model, dg)
    info = dm.install_merge(model, dg, cd)
    fp16 = _build_fp16_model(config['model']['name'])
    fp16.config.use_cache = True

    tok, prompt1 = tokenize_prompt(config, args.prompt)
    P = prompt1.shape[1]; G = args.gen
    out = {"config": args.config, "merge_info": info, "decode": [], "ppl": None}

    # ---- accuracy gate (optional): lm_head FP16 vs MXFP8, merge OFF, teacher-forced PPL
    if args.ppl:
        dm.set_merge(False)
        model.seqlen = config['eval']['max_length']
        tl = get_wikitext2(seed=config['calibration'].get('seed', 0), seqlen=model.seqlen,
                           tokenizer=tok, eval_mode=True)
        bs = config['eval']['batch_size']
        model.config.use_cache = False
        p_fp16head = evaluate_ppl(model, tl, DEV, batch_size=bs)
        orig_head = fn.wire_lmhead_mxfp8(model, lmh, qpf)
        p_mxhead = evaluate_ppl(model, tl, DEV, batch_size=bs)
        model.lm_head = orig_head  # restore for a clean decode baseline
        d = p_mxhead - p_fp16head
        out["ppl"] = dict(fp16head=p_fp16head, mxfp8head=p_mxhead, delta=d,
                          gate_pass=bool(d < 0.05))
        print(f"  PPL lm_head FP16={p_fp16head:.4f} -> MXFP8={p_mxhead:.4f} delta={d:+.4f} "
              f"({'PASS' if d<0.05 else 'FAIL'} <0.05)", flush=True)
        model.config.use_cache = True
        cleanup_memory()

    # ---- decode throughput: merge-only (FP16 head) vs merge+MXFP8-head vs FP16 ----
    dm.set_merge(True)
    for B in batches:
        prompt = prompt1.repeat(B, 1).contiguous()
        # merge-only baseline (FP16 head)
        dm.graph_decode_median(model, prompt, P, min(G, 16))
        base = dm.graph_decode_median(model, prompt, P, G)
        # + MXFP8 lm_head
        orig_head = fn.wire_lmhead_mxfp8(model, lmh, qpf)
        try:
            dm.graph_decode_median(model, prompt, P, min(G, 16))
            lmhd = dm.graph_decode_median(model, prompt, P, G)
            capture_ok = True
        except Exception as e:
            print(f"  B={B} MXFP8-head decode failed: {e!r}", flush=True)
            lmhd = float("nan"); capture_ok = False
        model.lm_head = orig_head
        fp16_g = float("nan")
        try:
            fp16_g = dm.graph_decode_median(fp16, prompt, P, G)
        except Exception as e:
            print(f"  fp16 graph failed B={B}: {e!r}", flush=True)
        rec = dict(batch=B, merge_only_ms=base, merge_lmhead_ms=lmhd, fp16_graph_ms=fp16_g,
                   capture_ok=capture_ok,
                   speedup_mergeonly=(fp16_g / base if fp16_g == fp16_g else None),
                   speedup_lmhead=(fp16_g / lmhd if (fp16_g == fp16_g and lmhd == lmhd) else None),
                   lmhead_gain=(base / lmhd if lmhd == lmhd else None))
        out["decode"].append(rec)
        print(f"  B={B}: merge-only {base:.3f} -> +MXFP8head {lmhd:.3f} ms/tok "
              f"(gain {rec['lmhead_gain'] if rec['lmhead_gain'] else float('nan'):.3f}x) | "
              f"FP16 {fp16_g:.3f} | mergeonly {rec['speedup_mergeonly']:.3f}x "
              f"+head {rec['speedup_lmhead'] if rec['speedup_lmhead'] else float('nan'):.3f}x", flush=True)

    dm.remove_patches()
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}\nDECODE_LMHEAD_DONE", flush=True)


if __name__ == "__main__":
    main()
