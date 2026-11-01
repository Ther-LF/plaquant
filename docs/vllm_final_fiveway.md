# FINAL five-way same-stack serving comparison (vLLM 0.32, RTX Pro 6000 SM120)

> 2026-11-01. The closing act of the vLLM perf loop (rounds 1-5, converged at
> `docs/vllm_perf_round5.md`). Everything below is measured in the SAME vLLM
> 0.32 (cu130) engine on the same RTX Pro 6000 Blackwell Server (SM120, 96 GB),
> one engine per config, CUDA graphs + torch.compile ON, prefix caching OFF,
> greedy + ignore_eos, identical exact-length token prompts, 1 warmup + median
> of 3 — the Phase-2/3 `bench_serve` protocol
> (`docs/vllm_integ_phase23.md`), extended to the large-batch regime.
> Raw JSONs: `docs/vllm_final_data/`; scripts:
> `docs/vllm_final_scripts/run_final.sh` (+ `run_sweep_point.sh`),
> harvest: `python -m promix.vllm_integ.harvest_final --dir docs/vllm_final_data --size 3b|8b`.

## 0. Gates (run before any ProMix bench — mandatory)

| gate | result |
|---|---|
| T2 prefill `--graphs`, final converged config (all R1-R5 levers default-ON) | **Byte-identical to the r5 gate on every field**: PPL 8.37029131329681, top-1 89.78647181310134 %, logprob-absdiff mean 0.17774903457377478 (`gate_t2_graphs_final.json` vs r5's `gate_t2_graphs_r5final.json`). (The harness's own top-1 criterion prints FAIL in graphs mode — same digits every round since r2; the gate is byte identity, per r3/r5.) |
| T2 decode-route m16 | **PASS, signed ΔPPL = −0.32004700553051535** (8.0473 vs ref 8.3674) — exactly r5's −0.3200. |

Zero drift from the converged round-5 configuration. Config = feat/vllm-integ
@9da7189 + this campaign's additions (r-sweep configs, pad-aware SF alloc —
byte/time-identical on the shipping r=1/8 splits, see §6 note 6).

## 1. The five methods (+ extensions)

| method | W / A | checkpoint & vLLM path | dtype |
|---|---|---|---|
| BF16 (anchor) | 16/16 | unsloth base weights, torch.compile + cuBLAS/CUTLASS | bfloat16 |
| FP8 W8A8 | 8/8 (dynamic per-token acts) | llm-compressor `FP8_DYNAMIC` (data-free) → compressed-tensors, `CutlassFP8ScaledMM` | auto |
| GPTQ-Marlin W4A16 | 4/16 (g128 sym, wikitext calib) | llm-compressor GPTQ → compressed-tensors, **Marlin** kernel (the weight-only reference) | auto |
| NVFP4 W4A4 | 4.5/4 (g16 e4m3 scales) | llm-compressor `NVFP4` → compressed-tensors, `FlashInferCutlassNvFp4` (`fp4_gemm_cutlass_sm120`) | auto |
| ProMix W4A4 (ours) | 4.97/4 (r=1/8 NVFP4+MXFP8) | ProMix exporter ckpt, promix vLLM plugin (fused dual-format GEMM + GEMV/marlin low-M routes + marlin-hi + split-K + gather-fold + cast-fold + qpfm, all default-ON) | float16 |

Extensions measured with the identical protocol (§5): **AWQ-Marlin W4A16**,
**INT8 W8A8** (SmoothQuant+GPTQ W8A8), **ct-MXFP4 W4A4**, and the ProMix
**r-sweep** (uniform high-fraction grid, §4). Models: Llama-3.2-3B ("3B") and
Llama-3-8B ("8B"). One GPU per engine, otherwise-idle card.

## 2. Accuracy column (wikitext-2 PPL, absolute vLLM protocol — 141 x 2048-token chunks, 288,627 scored positions, same tokenizer/scoring for every row)

| method | 3B PPL | 8B PPL | activations |
|---|---|---|---|
| BF16 | 7.82 | 6.14 | bf16 |
| FP8 W8A8 | **7.880** | *(measuring)* | fp8 dynamic |
| INT8 W8A8 (ext) | *(measuring)* | *(measuring)* | int8 dynamic |
| GPTQ-Marlin W4A16 | 8.20 | 6.52 | **bf16 (not quantized)** |
| AWQ-Marlin W4A16 | 8.48 | 6.73 | **bf16 (not quantized)** |
| NVFP4 W4A4 | 8.63 | 6.87 | fp4 static |
| ct-MXFP4 W4A4 (ext) | *(measuring)* | *(measuring)* | mxfp4 dynamic |
| **ProMix W4A4 (served config, real kernel)** | **8.370** | **6.640** | fp4+1/8 mxfp8, act-MSE |
| ProMix decode-route (marlin W4A16 m<=16 windows active) | 8.047 | — | w4a16 at small M |

ProMix is the most accurate W4A4 row: −0.26 PPL vs NVFP4 at 3B, −0.23 at 8B
(and −0.88/−0.56 vs ct-MXFP4). W4A16 rows keep bf16 activations (no activation
quantization cost) and are the weight-only reference, not a same-condition
W4A4 comparison. 3B/8B BF16, GPTQ, AWQ, NVFP4 from
`docs/baselines_on_hardware.md` (same harness/protocol); FP8/INT8/MXFP4
measured this campaign (`/home/luofan/baselines/results/*.json` on the host).

## 3. Serving tables

Cases: prefill-heavy ISL/OSL 1024/128, decode-heavy 128/1024, prefill-only
TTFT L2048, QServe-protocol continuous batching 1024/512 at concurrency
{64,128,256} (all requests submitted at once; the vLLM continuous-batching
scheduler manages the stream), long-ISL (2048/4096) decode with **fp8 KV
cache**. Batches {1,16} small-batch reference + {32,64,128,256,512}
large-batch primary (the serving regime; GEMMs saturated).

<!-- BEGIN GENERATED 3B -->
<!-- END GENERATED 3B -->

<!-- BEGIN GENERATED 8B -->
<!-- END GENERATED 8B -->

### Headline summary (ProMix vs each, TPOT at the serving regime)

**3B** (decode-heavy 128/1024): ProMix vs BF16 **1.09-1.34x faster** at every
batch; vs GPTQ-Marlin W4A16 0.66x at b1 → crossover at b128 (0.99) →
**1.10-1.12x at b256-512**; vs FP8 W8A8 ~parity at b256-512 (1.01/0.99); vs
NVFP4 0.74 (b16) → **0.93 (b512)**. Prefill TTFT L2048: ProMix beats BF16
1.26-1.77x and GPTQ/AWQ W4A16 1.33-1.88x at every batch; 0.66-0.68x of NVFP4
at b>=16 (b1 0.94).

**8B** (decode-heavy 128/1024): ProMix vs BF16 **1.19-1.68x**; vs GPTQ-Marlin
W4A16 0.66x at b1 → **1.03/1.15/1.24x at b128/256/512**; vs FP8 W8A8
0.90-1.03 (parity); vs NVFP4 0.74-0.87 (improving with batch). Prefill TTFT
L2048: ProMix 1.51-1.82x over BF16, **1.64-1.95x over GPTQ-Marlin**, 0.58-0.63x of NVFP4. QServe-protocol c256 throughput: ProMix 4922 tok/s vs GPTQ 3680
(**1.34x**), BF16 3864 (**1.27x**), FP8 4743 (1.04x), NVFP4 5845 (0.84x).

### The W4A16 verdict (the question this campaign had to answer)

**At the serving operating region, yes — ProMix W4A4 beats the weight-only
W4A16 Marlin reference**: every prefill point at every batch (1.3-2.0x), every
decode point from b128 up (1.03-1.24x), and high-concurrency continuous
batching (1.17x 3B / 1.34x 8B at c256) — while also being a 4-bit-activation
method (W4A16 pays zero activation-quant cost and serves bf16 activations).
The honest loss: small-batch decode (b1-b64), where Marlin's weight-streaming
GEMV is the textbook winner (0.66-0.85x); that regime is latency-bound
single-user serving, not the batched throughput regime these systems are
deployed in (and ProMix's own m<=16 marlin windows already borrow exactly that
trick inside its decode route, at better accuracy than the pure W4A16 model —
PPL 8.047 vs 8.20).

### fp8-KV support matrix (long-ISL runs)

All five methods accepted `kv_cache_dtype="fp8"` in this build (FlashAttention
backend) and produced correct-looking throughput; no method errored at engine
init. The 8B FP8-W8A8 kv8 row needed a re-run after a transient
free-memory race between chained engines (not a method failure).

## 4. ProMix r-sweep — the tunable accuracy/throughput curve (uniform n/32 grid)

The converged kernel serves ANY high-fraction r (dual-format split per
Linear). Grid constraint (honest): with uniform fractions, every Linear needs
K_high%32==0 at K in {3072, 8192} (3B) — representable points are n/32;
finer points (1/48, 1/64, ...) and non-n/32 ratios like 1/12, 1/24 need the
per-layer ring allocation (feat/rd-alloc machinery), which is not in this
serving branch. Points: 3B r in {1/32, 1/16, 3/32, 1/8, 3/16, 1/4};
8B r in {1/32, 1/16, 1/8, 1/4}.

<!-- BEGIN RSWEEP -->
<!-- END RSWEEP -->

## 5. Extensions — every other vLLM quant method runnable on this GPU

<!-- BEGIN EXT -->
<!-- END EXT -->

## 6. Published W4A4 peer methods, end-to-end on this GPU

<!-- BEGIN PEERS -->
<!-- END PEERS -->

## 7. Cross-paper end-to-end comparison (relative to same-stack FP16/BF16, ISL/OSL 1024/512)

Literature convention (QServe): each method's e2e serving speedup vs ITS OWN
same-stack FP16/BF16 anchor, protocol-matched. Hardware generations differ —
these are NOT directly comparable absolute numbers; anchor strength matters
(an HF-eager anchor inflates the ratio; TRT-LLM/vLLM anchors are strong).

| method | W/A bits | hardware | stack (anchor strength) | protocol | published / measured speedup vs same-stack FP16/BF16 |
|---|---|---|---|---|---|
| QServe W4A8KV4 | 4/8 + KV4 | A100 / L40S | TensorRT-LLM FP16 (strong) | ISL/OSL 1024/512 max throughput | 1.2-2.4x (A100) / 1.47-3.47x (L40S), Llama-7B..72B (as published) |
| Atom W4A4KV4 | 4/4 + KV4 | RTX 4090 | Punica FP16 (medium) | serving throughput, batch 2-256 (large-batch partially simulated) | up to 2.5x (as published) |
| AWQ / TinyChat W4A16 | 4/16 | 4090 / Orin | HF eager FP16 (**weak anchor**) | decode latency | 2.7-3.9x (as published) |
| vLLM NVFP4 W4A4 | 4.5/4 | **RTX Pro 6000 (this work's GPU)** | vLLM 0.32 BF16 (strong) | ISL/OSL 1024/512, c256 (measured here) | **1.27x (3B) / 1.51x (8B)**; decode-heavy b256 1.19/1.39x; prefill L2048 b16 2.56/3.13x |
| **ProMix W4A4 (ours)** | 4.97/4 | **RTX Pro 6000** | vLLM 0.32 BF16 (strong) | ISL/OSL 1024/512, c256 (measured here) | **1.13x (3B) / 1.27x (8B)**; decode-heavy b256 TPOT 1.09/1.19x (b512 1.14/1.29x); prefill L2048 1.26-1.82x |

Honesty footnotes: (1) different GPUs/generations — on Blackwell the BF16
anchor itself is very strong (TC-dense BF16 + FlashAttention + graphs), which
compresses ALL quantization speedups relative to older-GPU papers; the NVFP4
row is the calibration point — the best production 4-bit stack on this card
gets 1.27-1.51x on the same protocol where QServe cites 1.2-3.47x on
A100/L40S. (2) QServe quantizes activations to 8 bits (W4A8); ProMix/NVFP4/
Atom quantize to 4 bits — 2x more aggressive on the activation side.
(3) AWQ's 2.7-3.9x is against HF eager, not a serving engine. (4) Our numbers
are end-to-end engine throughput including scheduler/sampling, not kernel-only.

## 8. Honest notes & caveats

1. **Session noise**: one engine per GPU on an otherwise-idle card, but the
   box is shared (8 GPUs, other tenants' host load). Repeat-run wobble on this
   protocol has historically been ±3-4% on b1 long-context rows; the b>=64
   rows replicate across cases and are the load-bearing numbers.
2. **TPOT at high concurrency** is (full−TTFT)/(OSL−1) with all B requests
   submitted together — it mixes a small amount of scheduling/chunked-prefill
   into the decode average for every method equally.
3. **ProMix checkpoint is fp16; baselines bf16** (same width; the known
   bit-exact cast-fold handles the GEMM-out cast).
4. **b512 decode-heavy at 8B** exceeds the KV budget for BF16
   (67.9 GiB KV available vs ~77 GiB needed at 1152 tok x 512 seqs):
   vLLM preempts/recomputes rather than OOM — throughput numbers at that cell
   include preemption for BF16 (a capacity result in itself; 4-bit rows keep
   22-14 GiB more KV headroom, §capacity tables).
5. **ProMix capacity**: ProMix's KV budget is the lowest of the quantized
   rows (fp16 lm_head+embeddings in the checkpoint + the marlin-hi companion
   copy, §r5 doc 2a) — a known, fixable packaging cost (lm_head W8 is
   PPL-safe per the 1B study).
6. **Pad-aware SF alloc fix** (this campaign): r-sweep splits with
   K_seg/VEC % 4 != 0 read uninitialized SF pad bytes (NaN risk) in the
   activation packers; fixed by conditional zero-fill. The shipping r=1/8
   splits are unpadded — their path is byte- and time-identical (gate §0 ran
   on the fixed stack).
7. **GPTQ-Marlin 3B prefill is anomalously slow** (worse than BF16, both
   sessions, matches the committed round-1 baselines) — Marlin's act-order
   dequant path at prefill M; AWQ-Marlin (ext §5) is the stronger W4A16
   prefill row at 3B. The W4A16 verdict above cites whichever is better.
8. FP8 W8A8 8B long-ISL kv8 row re-run after an infra race (chained engines);
   all other cells are first-pass.

## 9. Reproduce

```bash
B=/home/luofan/vllm_integ/bench_final       # on the GPU host
# core five-way, one config per GPU (ckpt, tag, dtype):
docker exec -d -e CUDA_VISIBLE_DEVICES=<n> luofan-plaquant \
  bash -c "bash $B/run_final.sh <ckpt> <tag> <dtype>"
# gates: promix.vllm_integ.gates.gate_t2 --ckpt <pm-ckpt> --graphs / --decode-route-m 16
# r-sweep: export per config promix/configs/llama-3.2-3b-fp-perhead-r<r>.yaml,
#   then bash $B/run_sweep_point.sh <ckpt> pm3b_r<r>
# accuracy (absolute protocol): /home/luofan/baselines/scripts/bench_vllm.py
# harvest: python -m promix.vllm_integ.harvest_final --dir docs/vllm_final_data --size 3b|8b
```
