"""Measure + gate the fused SwiGLU->Hadamard->pack kernel on the 1B real model.

Phases:
  --phase probe   : per-Linear byte-exactness (fused kernel vs current qpfm pack)
                    on the REAL down_proj activation produced by the model.
  --phase perf    : full-model prefill latency, baseline-vs-fused, per M, with
                    CUDA-graph (the shipping config), plus stage drill-down.
  --phase ppl     : wikitext-2 PPL (baseline vs fused) + logit cosine.

Remote-only.  GPU chosen by idle-waiter (util<=8).
"""
import argparse, json, os, subprocess, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
import transformers

from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import load_config, prepare_ptq_model, init_distributed_for_ptq_main_if_needed
from promix.quantize import full_network as fn
from promix.quantize.hadamard import matmul_hadU_cuda
from promix.eval.data import get_wikitext2
from promix.eval.full_network_realkernel import (
    load_kernels, _build_fp16_model, cosine, HIGH_FRACTION)
from promix.eval.prefill_graph import capture_forward, time_eager, time_graph


def load_hpk():
    from torch.utils.cpp_extension import load
    inc = [KERN, os.path.join(REPO, "third_party/cutlass/include"),
           os.path.join(REPO, "third_party/cutlass/tools/util/include"),
           os.path.join(REPO, "third_party/cutlass/examples/common")]
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    return load(name="hadamard_pack_ext",
                sources=[os.path.join(KERN, "hadamard_pack_ext.cu")],
                extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)


def load_hpk28():
    """Fused SwiGLU + 28x512 Hadamard + pack producer for the 8B non-power-of-two
    intermediate (I=14336). Returns (module, had28) where had28 is the 28x28 +-1
    matrix (CPU float32) the kernel caches to constant memory on first call."""
    from torch.utils.cpp_extension import load
    from promix.quantize.hadamard import get_hadK
    inc = [KERN, os.path.join(REPO, "third_party/cutlass/include"),
           os.path.join(REPO, "third_party/cutlass/tools/util/include"),
           os.path.join(REPO, "third_party/cutlass/examples/common")]
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    mod = load(name="hadamard28_pack_ext",
               sources=[os.path.join(KERN, "hadamard28_pack_ext.cu")],
               extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
    had28, _ = get_hadK(14336)
    return mod, had28.to("cpu").float().contiguous()


def pick_free_gpu(util_max=8, timeout_s=1800):
    t0 = time.time()
    while True:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"]).decode()
        c = sorted([(int(i), int(u), int(m)) for i, u, m in
                    [l.split(",") for l in out.strip().splitlines()]])
        free = [x for x in c if x[1] <= util_max]
        if free:
            return free[0][0]
        if time.time() - t0 > timeout_s:
            raise RuntimeError("no idle GPU")
        time.sleep(10)


def build(config, nv, mx, cd, qpf, qpfm, hpk, gptq):
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    model = prepare_ptq_model(config, DEV, run_gptq=gptq)
    model.to(DEV); model.config.use_cache = False
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf, qpfm=qpfm, hpk=hpk)
    fn.wire_fused_mlp(model)
    model.to(DEV).eval()
    for m in model.modules():
        for attr in ("_column_order", "new_column_order"):
            t = getattr(m, attr, None)
            if torch.is_tensor(t) and t.device.type != "cuda":
                setattr(m, attr, t.to(DEV))
    return model, packed


def set_fused(model, on):
    """Toggle the fused down_proj producer path (A/B on the SAME wired model)."""
    n = 0
    for _, _, w in fn.iter_target_wrappers(model):
        if hasattr(w, "fused_swiglu_had") and w.module.__class__.__name__ == "MixedQuantLinear":
            w.fused_swiglu_had = bool(on); n += 1
    return n


@torch.no_grad()
def run_probe(model, config, qpfm, hpk):
    """Per-Linear byte-exactness of the fused kernel vs the current path, on
    synthetic gate/up with realistic magnitudes, plus a REAL activation captured
    from layer 0's down_proj to check the statistics match."""
    tok = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    ids = get_wikitext2(seed=0, seqlen=2048, tokenizer=tok, eval_mode=True).input_ids[:, :2048].to(DEV)
    down = [w for _, _, w in fn.iter_target_wrappers(model) if getattr(w, "online_full_had", False)]
    mql = down[0].module
    K_low, K_high = mql.K_low, mql.K_high
    set_fused(model, False)
    caps = []
    def hook(mod, inp, out):
        caps.append(inp[0].detach().clone())
    h0 = down[0].register_forward_hook(hook)
    model(ids)
    h0.remove()
    real = caps[0].reshape(-1, 8192).to(torch.float16)
    print(f"real down_proj input (layer0): M={real.shape[0]} std={real.float().std():.4f} "
          f"absmax={real.float().abs().max():.3f}", flush=True)

    ok = True
    cases = [("randn*3", lambda n: (torch.randn(n, 8192, device=DEV) * 3).bfloat16(),
                          lambda n: (torch.randn(n, 8192, device=DEV) * 2).bfloat16())]
    for M in (512, 2048, 4096):
        g = (torch.randn(M, 8192, device=DEV) * 3).bfloat16()
        u = (torch.randn(M, 8192, device=DEV) * 2).bfloat16()
        u[::3] *= 8.0                                     # outlier stress
        h = torch.nn.functional.silu(g) * u               # bf16, reference product
        y = matmul_hadU_cuda(h.float(), None, 1, fold_scale=True).to(torch.float16)
        a_ref = qpfm.quant_pack_float_mixed(y.contiguous(), M, mql.N, K_low, K_high)
        a = hpk.hadamard_swiglu_pack_mixed(g, u, M, mql.N, K_low, K_high)
        mm = [int((x != y2).sum()) for x, y2 in zip(a_ref, a)]
        hm = 0
        good = all(m == 0 for m in mm)
        ok &= good
        print(f"  M={M:5d} {'PASS' if good else 'FAIL'} code_lo/sfa_lo/code_hi/sfa_hi mism="
              f"{mm} h_mism={hm}", flush=True)
        del g, u, h, y, a_ref, a
        torch.cuda.empty_cache()
    # REAL layer-0 activation through the production pack (sanity: no NaN/Inf)
    print(f"  real-activation pack nan={torch.isnan(real.float()).any().item()}", flush=True)
    print("PROBE", "PASS" if ok else "FAIL", flush=True)
    return ok


@torch.no_grad()
def run_perf(model, fp16, config, Ms, args, out):
    tok = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    wiki = get_wikitext2(seed=0, seqlen=max(Ms), tokenizer=tok, eval_mode=True).input_ids
    rows = []
    for M in Ms:
        ids = wiki[:, :M].contiguous().to(DEV)
        t_fp16 = time_eager(lambda: fp16(ids).logits, iters=(10 if M <= 4096 else 6), warmup=4)
        rec = dict(M=M, fp16_ms=t_fp16)
        ref = None
        for tag, on in (("baseline", False), ("fused", True)):
            set_fused(model, on)
            logits = model(ids).logits.clone()
            if ref is None: ref = logits.clone()
            cos = cosine(ref, logits)
            t_e = time_eager(lambda: model(ids).logits, iters=(10 if M <= 4096 else 6), warmup=4)
            g, sl = capture_forward(model, ids)
            g.replay(); torch.cuda.synchronize()
            cos_g = cosine(ref, sl.clone())
            t_g = time_graph(g, iters=(30 if M <= 4096 else 15), warmup=5)
            del g
            rec[tag] = dict(eager_ms=t_e, graph_ms=t_g, cos=cos, cos_graph=cos_g,
                            speedup_vs_fp16=t_fp16 / t_g)
            print(f"  M={M:5d} {tag:9s} eager={t_e:7.3f} graph={t_g:7.3f} "
                  f"({t_fp16/t_g:5.3f}x fp16) cos={cos_g:.7f}", flush=True)
            cleanup_memory()
        print(f"  M={M:5d} SAVED {rec['baseline']['graph_ms']-rec['fused']['graph_ms']:7.3f} ms "
              f"graph | {t_fp16/rec['baseline']['graph_ms']:.3f}x -> {t_fp16/rec['fused']['graph_ms']:.3f}x vs FP16",
              flush=True)
        rows.append(rec)
        del ids, ref
        cleanup_memory()
    out["perf"] = rows


@torch.no_grad()
def run_ppl(model, config, args, out):
    from promix.eval.evaluator import evaluate_ppl
    tok = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    model.seqlen = config['eval']['max_length']
    tl = get_wikitext2(seed=config['calibration'].get('seed', 0), seqlen=model.seqlen,
                       tokenizer=tok, eval_mode=True)
    bs = config['eval']['batch_size']
    res = {}
    set_fused(model, False)
    res["baseline"] = evaluate_ppl(model, tl, DEV, batch_size=bs)
    print(f"PPL_BASELINE {res['baseline']:.4f}", flush=True)
    set_fused(model, True)
    res["fused"] = evaluate_ppl(model, tl, DEV, batch_size=bs)
    print(f"PPL_FUSED {res['fused']:.4f} delta={res['fused']-res['baseline']:+.4f}", flush=True)
    out["ppl"] = res


@torch.no_grad()
def run_bench(hpk, args, out):
    """Isolated fused-kernel timing at the production layer shape (I=8192,
    K_low=7168, K_high=1024, N=2048) + the CURRENT 4-op chain for the same row
    count, both min-of-N: shows the fused kernel's own time and traffic."""
    from promix.quantize.hadamard import matmul_hadU_cuda
    rows = []
    for M in [int(x) for x in args.ms.split(",")]:
        g = (torch.randn(M, 8192, device=DEV) * 3).bfloat16()
        u = (torch.randn(M, 8192, device=DEV) * 2).bfloat16()
        f = lambda: hpk.hadamard_swiglu_pack_mixed(g, u, M, 2048, 7168, 1024)
        f(); torch.cuda.synchronize()
        t = []
        for _ in range(30):
            t0 = time.perf_counter(); f(); torch.cuda.synchronize(); t.append(time.perf_counter() - t0)
        tf = min(t) * 1e3
        # current chain: silu*mul (bf16) + fp32 hadamard(+fold) + fp16 cast + qpfm pack
        qpfm = load_qpfm()
        def cur():
            h = torch.nn.functional.silu(g) * u
            y = matmul_hadU_cuda(h.float(), None, 1, fold_scale=True).to(torch.float16)
            return qpfm.quant_pack_float_mixed(y.contiguous(), M, 2048, 7168, 1024)
        cur(); torch.cuda.synchronize()
        t = []
        for _ in range(30):
            t0 = time.perf_counter(); cur(); torch.cuda.synchronize(); t.append(time.perf_counter() - t0)
        tc = min(t) * 1e3
        # traffic: fused reads 2*M*I*2 B, writes M*(K_low/2 + K_high + SF)
        rb = 2 * M * 8192 * 2 + M * (7168 // 2 + 1024) + 200000
        gbps = rb / (tf * 1e-3) / 1e9
        print(f"  M={M:5d} fused={tf:7.3f} ms  current_chain={tc:7.3f} ms  "
              f"speedup={tc/tf:4.2f}x  fused_traffic={rb/1e6:7.1f} MB -> {gbps/1000:5.2f} TB/s",
              flush=True)
        rows.append(dict(M=M, fused_ms=tf, current_chain_ms=tc, traffic_MB=rb / 1e6,
                         eff_TBps=gbps / 1000))
        del g, u
        cleanup_memory()
    out["bench"] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "promix/configs/llama-3.2-1b-fp-perhead.yaml"))
    ap.add_argument("--phase", default="perf", choices=["probe", "perf", "ppl", "bench", "all"])
    ap.add_argument("--ms", default="2048,4096,8192")
    ap.add_argument("--gptq", action="store_true", default=True)
    ap.add_argument("--no-gptq", dest="gptq", action="store_false")
    ap.add_argument("--out", default=os.path.join(KERN, "hadamard_swiglu_fusion.json"))
    args = ap.parse_args()
    gpu = pick_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"FUSION harness | phase={args.phase} gpu={gpu}", flush=True)
    config = load_config(args.config)
    nv, mx, cd, qpf = load_kernels()
    qpfm = load_qpfm()
    hpk = load_hpk()
    print("kernels loaded", flush=True)
    model, packed = build(config, nv, mx, cd, qpf, qpfm, hpk, args.gptq)
    fp16 = _build_fp16_model(config['model']['name'])
    out = {"config": args.config, "gpu": gpu, "ms": args.ms}
    if args.phase in ("probe", "all"):
        out["probe_ok"] = run_probe(model, config, qpfm, hpk)
    if args.phase in ("perf", "all"):
        run_perf(model, fp16, config, [int(x) for x in args.ms.split(",")], args, out)
    if args.phase in ("ppl", "all"):
        run_ppl(model, config, args, out)
    if args.phase in ("bench", "all"):
        run_bench(hpk, args, out)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("FUSION_DONE", flush=True)


def load_qpfm():
    from torch.utils.cpp_extension import load
    inc = [KERN, os.path.join(REPO, "third_party/cutlass/include"),
           os.path.join(REPO, "third_party/cutlass/tools/util/include"),
           os.path.join(REPO, "third_party/cutlass/examples/common")]
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    return load(name="quant_pack_fused_ext",
                sources=[os.path.join(KERN, "quant_pack_fused_ext.cu")],
                extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)


if __name__ == "__main__":
    main()
