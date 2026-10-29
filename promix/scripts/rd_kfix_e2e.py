"""rd-kfix full-model prefill e2e: real r=0 (pure-NVFP4) vs 1/8, now that r=0 RUNS.

Builds the real packed fused-kernel 1B-Instruct model twice (block-aligned K-split
driven through configure_quantizers' `_rd_khigh_override` seam) and times TTFT
prefill (M=2048):
  * 1/8  : every target Linear K_high = round(K/8) ring-aligned
  * r=0  : every target Linear K_high = 0  (pure-NVFP4, the fix)
Prints the REALIZED per-Linear K_high so r=0 is PROVEN served as pure-NVFP4
(not the old silent 1/8 fallback). Latency depends only on the K-split, so using
a block-aligned uniform split (vs the resq per-head scheme) is valid here.

    CUDA_VISIBLE_DEVICES=5 python /home/luofan/plaquant-kfix/promix/scripts/rd_kfix_e2e.py
"""
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import promix.eval.ptq as ptqmod  # noqa: E402
import promix.scripts.rd_validity as rdv  # noqa: E402
from promix.scripts import rd_perf  # noqa: E402
import promix.quantize.full_network as fn  # noqa: E402
from promix.quantize.quant_utils import find_qlayers, ActQuantWrapper  # noqa: E402
from promix.quantize.kv_quant import setup_k_quant  # noqa: E402

CONFIG = os.path.join(REPO, "promix/configs/llama-3.2-1b-resq.yaml")
DEV = "cuda"
M = 2048


def _khigh_eighth(K, ring=32):
    return int(round(0.125 * K / ring)) * ring


def build(config, khigh_fn, nv, mx, cd, qpf):
    model = rdv.build_base(config, False)
    model.to(DEV)
    override = {}
    for name, short, w in fn.iter_target_wrappers(model):
        override[name] = int(khigh_fn(int(w.module.weight.shape[1])))
    model._rd_khigh_override = override
    ptqmod.configure_quantizers(model, config)
    setup_k_quant(model, config, basis_path=config["paths"]["basis"],
                  rotation_path=config["paths"]["rotation"])
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=0.125, verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf, dg=None)
    model.to(DEV).eval()
    model.config.use_cache = True
    return model


def realized(model):
    return sorted({int(w.module.K_high) for _, _, w in fn.iter_target_wrappers(model)})


def main():
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29601")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    config = ptqmod.load_config(CONFIG)
    config = rdv.resolve_paths(config)
    print(f"config={CONFIG}\nmodel={config['model']['name']}", flush=True)
    nv, mx, cd, qpf = rd_perf.load_kernels()
    vocab = None

    m8 = build(config, _khigh_eighth, nv, mx, cd, qpf)
    vocab = m8.config.vocab_size
    print(f"1/8  realized K_high set: {realized(m8)}", flush=True)
    rd_perf.measure_prefill_ms(m8, 64, vocab, iters=1, warmup=1)
    t8 = rd_perf.measure_prefill_ms(m8, M, vocab, iters=20, warmup=8)
    del m8
    rd_perf.cleanup_memory()

    m0 = build(config, lambda K: 0, nv, mx, cd, qpf)
    print(f"r=0  realized K_high set: {realized(m0)}", flush=True)
    rd_perf.measure_prefill_ms(m0, 64, vocab, iters=1, warmup=1)
    t0 = rd_perf.measure_prefill_ms(m0, M, vocab, iters=20, warmup=8)
    del m0
    rd_perf.cleanup_memory()

    print("\n==== REAL full-model prefill TTFT (M=2048, Llama-3.2-1B-Instruct) ====")
    print(f"  1/8 mixed      : {t8:.3f} ms")
    print(f"  r=0 pure-NVFP4 : {t0:.3f} ms   ({t0/t8:.4f}x vs 1/8)")
    verdict = "FASTER" if t0 < t8 else "SLOWER"
    print(f"  => real r=0 is {verdict} than 1/8 e2e by {abs(1-t0/t8)*100:.1f}% "
          "(this number was a 1/8-fallback before the fix)")


if __name__ == "__main__":
    main()
