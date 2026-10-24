"""1B-prefill squeeze harness: lever 1 (single-launch fused pack) + lever 2
(MXFP8 lm_head), numerics-gated, full-model prefill before->after vs FP16 per M
with CUDA-graph. Companion to docs/prefill_1b_squeeze_report.md.

Configs (all on top of the shipped div-fold real-kernel decoder):
  fp16                 : plain FP16 Llama-3.2-1B
  base                 : real decoder, 2-launch qpf pack, FP16 lm_head (current ship)
  packfuse             : + single-launch fused pack (lever 1, bit-exact)
  packfuse_lmhead      : + MXFP8 (W8A8) lm_head (lever 2)

Numerics gates (real wikitext ids):
  cos(packfuse logits vs base logits)            >= 0.9999  (expect ~1.0, bit-exact)
  cos(lmhead real-kernel logits vs fake-W8 twin) >= 0.9999  (kernel == twin)
PPL of the W8 lm_head itself is gated separately (lmhead_probe: +0.048).
"""
import argparse, json, os, subprocess, sys, time
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
from promix.eval.ptq import load_config, prepare_ptq_model, init_distributed_for_ptq_main_if_needed
from promix.quantize import full_network as fn
from promix.eval.data import get_wikitext2
from promix.eval.full_network_realkernel import load_kernels, _build_fp16_model, cosine, HIGH_FRACTION
from promix.eval.prefill_graph import capture_forward, time_eager, time_graph
from promix.eval.lmhead_probe import FakeLmHead


def load_extra_kernels():
    inc=[KERN, os.path.join(CUTLASS,"include"), os.path.join(CUTLASS,"tools/util/include"),
         os.path.join(CUTLASS,"examples/common")]
    flags=["-std=c++17","--expt-relaxed-constexpr","--expt-extended-lambda","-O3"]
    qpfm = load(name="quant_pack_fused_ext",
                sources=[os.path.join(KERN,"quant_pack_fused_ext.cu")],
                extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
    lmh = load(name="lmhead_mxfp8_ext",
               sources=[os.path.join(KERN,"lmhead_mxfp8_ext.cu")],
               extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
    return qpfm, lmh


def move_col_order(model):
    for m in model.modules():
        for attr in ("_column_order", "new_column_order"):
            t = getattr(m, attr, None)
            if torch.is_tensor(t) and t.device.type != "cuda":
                setattr(m, attr, t.to(DEV))


def gpu_busy():
    try:
        out = subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used",
                                       "--format=csv,noheader,nounits"]).decode().strip().splitlines()
        return out
    except Exception:
        return ["?"]


@torch.no_grad()
def graph_time(model, static_ids, M):
    g, static_logits = capture_forward(model, static_ids)
    g.replay(); torch.cuda.synchronize()
    out = static_logits.clone()
    t = time_graph(g, iters=(40 if M <= 4096 else 20), warmup=8)
    del g
    cleanup_memory()
    return t, out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default=os.path.join(REPO, "promix/configs/llama-3.2-1b-fp-perhead.yaml"))
    ap.add_argument("--ms", type=str, default="2048,4096,8192")
    ap.add_argument("--out", type=str, default=os.path.join(KERN, "prefill_squeeze_perf.json"))
    args = ap.parse_args()
    Ms = [int(x) for x in args.ms.split(",")]

    config = load_config(args.config)
    print(f"squeeze harness | gpu={gpu_busy()}", flush=True)
    nv, mx, cd, qpf = load_kernels()
    qpfm, lmh = load_extra_kernels()
    print("kernels: nv,mx,cd,qpf,qpfm(fused-pack),lmh(mxfp8)", flush=True)

    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    model = prepare_ptq_model(config, DEV, run_gptq=False)
    model.to(DEV); model.config.use_cache = False
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)
    model_name = config['model']['name']
    fp16 = _build_fp16_model(model_name)
    vocab = model.config.vocab_size

    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    wiki = get_wikitext2(seed=0, seqlen=max(Ms), tokenizer=tok, eval_mode=True)
    wiki_ids_full = wiki.input_ids
    lmW = model.lm_head.weight.data.clone()

    rows = []
    for M in Ms:
        static_ids = wiki_ids_full[:, :M].contiguous().to(DEV)
        tf = graph_time(fp16, static_ids, M)[0]

        # base: 2-launch pack, FP16 lm_head
        fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf, qpfm=None)
        for _,_,w in fn.iter_target_wrappers(model):
            if getattr(w, "online_full_had", False): w.fold_had_scale = True
        move_col_order(model); model.to(DEV).eval()
        t_base, log_base = graph_time(model, static_ids, M)

        # packfuse: single-launch fused pack
        fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf, qpfm=qpfm)
        for _,_,w in fn.iter_target_wrappers(model):
            if getattr(w, "online_full_had", False): w.fold_had_scale = True
        move_col_order(model); model.to(DEV).eval()
        t_pf, log_pf = graph_time(model, static_ids, M)
        cos_pf = cosine(log_base, log_pf)

        # packfuse + MXFP8 lm_head (real kernel)
        orig_head = fn.wire_lmhead_mxfp8(model, lmh, qpf)
        model.to(DEV).eval()
        t_lh, log_lh = graph_time(model, static_ids, M)
        # twin: fake W8 head on the SAME decoder (eager; fake-quant uses CPU
        # constants that are not graph-capturable, and it is only a numerics ref)
        model.lm_head = FakeLmHead(lmW, "w8", high_fraction=HIGH_FRACTION).to(DEV)
        model.to(DEV).eval()
        log_twin = model(static_ids).logits.clone()
        cos_lh = cosine(log_twin, log_lh)
        model.lm_head = orig_head  # restore FP16 head

        print(f"  M={M:5d} fp16={tf:7.3f} | base={t_base:7.3f} ({tf/t_base:.3f}x) | "
              f"packfuse={t_pf:7.3f} ({tf/t_pf:.3f}x) cos={cos_pf:.6f} | "
              f"+lmhead={t_lh:7.3f} ({tf/t_lh:.3f}x) cos(twin)={cos_lh:.6f}", flush=True)
        rows.append(dict(M=M, fp16_ms=tf,
                         base_ms=t_base, base_x=tf/t_base,
                         packfuse_ms=t_pf, packfuse_x=tf/t_pf, cos_packfuse_vs_base=cos_pf,
                         lmhead_ms=t_lh, lmhead_x=tf/t_lh, cos_lmhead_vs_twin=cos_lh))
        del static_ids, log_base, log_pf, log_lh, log_twin
        cleanup_memory()

    json.dump(dict(config=args.config, rows=rows), open(args.out,"w"), indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("PREFILL_SQUEEZE_PERF_DONE", flush=True)


if __name__ == "__main__":
    main()
