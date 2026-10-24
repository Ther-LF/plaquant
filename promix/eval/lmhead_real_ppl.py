"""Confirm end-to-end wikitext-2 PPL with the REAL MXFP8 lm_head kernel (not the
fake twin), on top of the shipped real-kernel decoder (GPTQ, div-fold). Closes the
accuracy chain for lever 2: fake-W8-twin PPL (lmhead_probe) + real-vs-twin cosine.
"""
import os, sys, json
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
CUTLASS = os.path.join(REPO, "third_party", "cutlass")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
import transformers
from torch.utils.cpp_extension import load
from promix.utils import DEV
from promix.eval.ptq import load_config, prepare_ptq_model, init_distributed_for_ptq_main_if_needed
from promix.eval.evaluator import evaluate_ppl
from promix.eval.data import get_wikitext2
from promix.quantize import full_network as fn
from promix.eval.full_network_realkernel import load_kernels, HIGH_FRACTION

def main():
    cfg = load_config(os.path.join(REPO, "promix/configs/llama-3.2-1b-fp-perhead.yaml"))
    nv, mx, cd, qpf = load_kernels()
    inc=[KERN, os.path.join(CUTLASS,"include"), os.path.join(CUTLASS,"tools/util/include"), os.path.join(CUTLASS,"examples/common")]
    flags=["-std=c++17","--expt-relaxed-constexpr","--expt-extended-lambda","-O3"]
    lmh = load(name="lmhead_mxfp8_ext", sources=[os.path.join(KERN,"lmhead_mxfp8_ext.cu")],
               extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(cfg['calibration'].get('seed', 0))
    model = prepare_ptq_model(cfg, DEV, run_gptq=True)
    model.to(DEV); model.seqlen = cfg['eval']['max_length']
    tok = transformers.AutoTokenizer.from_pretrained(cfg['model']['name'])
    testloader = get_wikitext2(seed=cfg['calibration'].get('seed',0), seqlen=model.seqlen, tokenizer=tok, eval_mode=True)
    bs = cfg['eval']['batch_size']
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf)  # decoder real + div-fold, FP16 lm_head
    model.to(DEV)
    ppl_fp16head = evaluate_ppl(model, testloader, DEV, batch_size=bs)
    print(f"PPL_realdecoder_fp16head {ppl_fp16head:.4f}", flush=True)
    fn.wire_lmhead_mxfp8(model, lmh, qpf)
    model.to(DEV)
    ppl_mxhead = evaluate_ppl(model, testloader, DEV, batch_size=bs)
    print(f"PPL_realdecoder_MXFP8head {ppl_mxhead:.4f}  delta={ppl_mxhead-ppl_fp16head:+.4f}  "
          f"gate(<=14.95)={'PASS' if ppl_mxhead<=14.95 else 'FAIL'}", flush=True)
    json.dump(dict(fp16head=ppl_fp16head, mxfp8head=ppl_mxhead), open(os.path.join(KERN,"lmhead_real_ppl.json"),"w"), indent=2)
    print("LMHEAD_REAL_PPL_DONE", flush=True)

if __name__=="__main__":
    main()
