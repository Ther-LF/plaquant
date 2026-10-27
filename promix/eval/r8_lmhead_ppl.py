"""Round-8: isolated lm_head MXFP8 (W8) accuracy gate. Loads the plain FP16 model
and swaps ONLY the lm_head for the bit-faithful W8 twin (quantize_mxfp8_spec, the
same spec the real MXFP8 lm_head kernel packs against), measuring the wikitext-2
PPL delta. Isolates the lm_head quant cost; no real-kernel device plumbing."""
import argparse, json, os, sys
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
DEV = "cuda"
from promix.quantize.weight_pack import quantize_mxfp8_spec
from promix.quantize.quant_utils import fake_quantize_mxfp8


class FakeW8Head(nn.Module):
    def __init__(self, W, bias=None, out_dtype=torch.float16):
        super().__init__()
        _, _, W_deq = quantize_mxfp8_spec(W.detach().float().contiguous())
        self.register_buffer("W_deq", W_deq.contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().contiguous())
        self.out_dtype = out_dtype

    def forward(self, x):
        s = x.shape
        xq = fake_quantize_mxfp8(x.reshape(-1, s[-1]).float())
        D = (xq @ self.W_deq.t()).to(self.out_dtype)
        if self.bias is not None:
            D = D + self.bias.to(D.dtype)
        return D.reshape(*s[:-1], self.W_deq.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--nsamples", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import transformers
    from promix.eval.data import get_wikitext2
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).to(DEV).eval()
    enc = get_wikitext2(seed=0, seqlen=args.seqlen, tokenizer=tok, eval_mode=True).input_ids
    n = min(args.nsamples, enc.shape[1] // args.seqlen)

    def ppl():
        nll = cnt = 0
        for i in range(n):
            ids = enc[:, i * args.seqlen:(i + 1) * args.seqlen].to(DEV)
            o = model(ids, labels=ids)
            nll += o.loss.float().item() * (args.seqlen - 1); cnt += args.seqlen - 1
        return float(torch.exp(torch.tensor(nll / cnt)))

    p_fp16 = ppl()
    orig = model.lm_head
    model.lm_head = FakeW8Head(orig.weight.data, getattr(orig, "bias", None)).to(DEV)
    p_w8 = ppl()
    d = p_w8 - p_fp16
    res = dict(model=args.model, fp16_head_ppl=p_fp16, w8_head_ppl=p_w8, delta=d,
               gate_pass=bool(d < 0.05))
    print(f"lm_head FP16={p_fp16:.4f} -> W8={p_w8:.4f} delta={d:+.4f} "
          f"({'PASS' if d<0.05 else 'FAIL'} <0.05)", flush=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"RESULTS_WRITTEN {args.out}\nLMHEAD_PPL_DONE", flush=True)


if __name__ == "__main__":
    main()
