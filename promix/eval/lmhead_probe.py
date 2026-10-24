"""lm_head quantization ACCURACY probe (fast, numerics-only gate).

Lever 2 of the 1B-prefill squeeze round. lm_head is a shared ~10 ms FP16 GEMM
(vocab 128256 x hidden 2048) that both the real-kernel and FP16 models pay
identically. If we can quantize it and run it on the fused kernel it cuts the
real model's absolute time and lifts the prefill speedup ratio.

lm_head is accuracy-sensitive and its input (post final-RMSNorm hidden state) is
NOT rotated/reordered, so 4-bit on it is risky. This probe gates accuracy with
the fake-quant TWIN of each candidate (bit-faithful to the real kernel per the
full_network twin methodology) BEFORE any kernel work:

  * baseline      : lm_head FP16 (decoder Linears already real-kernel, div-fold).
  * lmhead_w4a4   : lm_head mixed NVFP4(low 7/8)+MXFP8(high 1/8), act+weight.
  * lmhead_w8     : lm_head MXFP8-only (W8A8, block-32) act+weight.

Gate: PPL <= 14.95 (baseline ~14.85). Reports wikitext-2 PPL per candidate.
Remote-only, FREE GPU. Reuses prepare_ptq_model + GPTQ + real-kernel wiring.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO)
sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

import transformers

from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import (
    load_config, prepare_ptq_model, init_distributed_for_ptq_main_if_needed,
)
from promix.eval.evaluator import evaluate_ppl
from promix.eval.data import get_wikitext2
from promix.quantize import full_network as fn
from promix.eval.full_network_realkernel import load_kernels, HIGH_FRACTION
from promix.quantize.weight_pack import (
    quantize_nvfp4_spec, quantize_mxfp8_spec, split_k,
)
from promix.quantize.quant_utils import fake_quantize_nvfp4, fake_quantize_mxfp8


class FakeLmHead(nn.Module):
    """Fake-quant twin of a quantized lm_head. Weight dequant uses the SAME spec
    quantizers the real kernel would pack against (RTN; GPTQ skips lm_head), so
    PPL here is the faithful accuracy of the real kernel path."""

    def __init__(self, W, mode, high_fraction=0.125, out_dtype=torch.float16):
        super().__init__()
        self.mode = mode
        self.N, self.K = W.shape
        self.out_dtype = out_dtype
        Wf = W.detach().to(torch.float32)
        if mode == "w4a4":
            self.K_low, self.K_high = split_k(self.K, high_fraction)
            _, _, deq_lo = quantize_nvfp4_spec(Wf[:, :self.K_low].contiguous())
            _, _, deq_hi = quantize_mxfp8_spec(Wf[:, self.K_low:].contiguous())
            W_deq = torch.cat([deq_lo, deq_hi], dim=-1)
        elif mode == "w8":
            _, _, W_deq = quantize_mxfp8_spec(Wf.contiguous())
        else:
            raise ValueError(mode)
        self.register_buffer("W_deq", W_deq.contiguous())

    def forward(self, x):
        in_shape = x.shape
        x2d = x.reshape(-1, in_shape[-1]).float()
        if self.mode == "w4a4":
            xq = torch.cat([fake_quantize_nvfp4(x2d[:, :self.K_low]),
                            fake_quantize_mxfp8(x2d[:, self.K_low:])], dim=-1)
        else:
            xq = fake_quantize_mxfp8(x2d)
        D = (xq @ self.W_deq.t()).to(self.out_dtype)
        return D.reshape(*in_shape[:-1], self.N)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default=os.path.join(REPO, "promix/configs/llama-3.2-1b-fp-perhead.yaml"))
    ap.add_argument("--gptq", action="store_true", default=True)
    ap.add_argument("--no-gptq", dest="gptq", action="store_false")
    ap.add_argument("--out", type=str, default=os.path.join(KERN, "lmhead_probe.json"))
    args = ap.parse_args()

    config = load_config(args.config)
    print(f"lm_head probe | config={args.config} gptq={args.gptq}", flush=True)
    nv, mx, cd, qpf = load_kernels()
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config['calibration'].get('seed', 0))

    model = prepare_ptq_model(config, DEV, run_gptq=args.gptq)
    model.to(DEV)
    model.seqlen = config['eval']['max_length']
    tok = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    testloader = get_wikitext2(seed=config['calibration'].get('seed', 0),
                               seqlen=model.seqlen, tokenizer=tok, eval_mode=True)
    bs = config['eval']['batch_size']

    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf)  # decoder real kernel + div-fold
    model.to(DEV)

    lm = model.lm_head
    W = lm.weight.data.clone()
    orig_forward_module = lm
    print(f"lm_head W shape = {tuple(W.shape)}", flush=True)

    results = {}

    # baseline: lm_head FP16
    model.lm_head = orig_forward_module
    model.to(DEV)
    ppl_base = evaluate_ppl(model, testloader, DEV, batch_size=bs)
    print(f"PPL_BASELINE_lmhead_fp16 {ppl_base:.4f}", flush=True)
    results["baseline_fp16"] = ppl_base

    for mode in ("w4a4", "w8"):
        fake = FakeLmHead(W, mode, high_fraction=HIGH_FRACTION).to(DEV)
        model.lm_head = fake
        model.to(DEV)
        ppl = evaluate_ppl(model, testloader, DEV, batch_size=bs)
        delta = ppl - ppl_base
        gate = "PASS" if ppl <= 14.95 else "FAIL"
        print(f"PPL_lmhead_{mode} {ppl:.4f}  delta={delta:+.4f}  gate(<=14.95)={gate}", flush=True)
        results[mode] = dict(ppl=ppl, delta=delta, gate_pass=bool(ppl <= 14.95))
        model.lm_head = orig_forward_module
        cleanup_memory()

    with open(args.out, "w") as f:
        json.dump(dict(config=args.config, gptq=args.gptq, results=results), f, indent=2)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("LMHEAD_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
