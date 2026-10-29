# RD large-M: the format / allocation levers and peer head-to-head at serving-scale M

**Branch** `feat/rd-largem` (off `origin/feat/rd-alloc`, + the `feat/rd-relat` interleaved
A/B machinery) · **HW** RTX PRO 6000 Blackwell (SM120, 96 GB), container `luofan-plaquant`,
dedicated idle GPU 2 (peers on idle GPU 3) · **Date** 2026-10-29

**Question.** All our W4A4 perf numbers so far were at M = 2048 (bs = 1 single-sequence
prefill) — the *least* compute-bound point, where the FP4/FP8 format lever measured only
~1.3× GEMM / ~26 % e2e and the measured-MSE allocation ~1 % e2e. W4A4 peers measure at much
larger effective M (QuaRot bs ≤ 64 · seq 2048, Atom batch 2–256, QServe continuous batching
ISL 1024). This round grows effective M via **batch** (seq fixed at 2048, batch 1→16,
M_eff 2048→32 768; Linears become more compute-bound, attention's per-sequence share fixed)
and re-measures, on the REAL packed fused-kernel 3B and 8B models:

1. the **format lever** (pure-NVFP4 endpoint vs the MXFP8 endpoint), e2e and GEMM-isolated;
2. the **allocation lever** (measured-MSE-1/8 vs uniform-1/8) e2e;
3. the **GEMM share** of prefill wall-clock;
4. *(added mid-round)* a formal **peer head-to-head at each M**: vLLM NVFP4 (GEMM-level and
   full-stack TTFT) and QuaRot INT4 (GEMM), plus the QServe/Atom non-viability record.

Harness: `promix/scripts/rd_largem.py` (batched interleaved A/B paired-ratio prefill +
cudaEvent GEMM decomposition; reuses rd_perf/rd_relat/rd_perfpuzzle machinery),
`promix/scripts/rd_largem_fp16.py` (same-stack FP16 vs W4A4 A/B),
`promix/eval/gemm_h2h_ours.py` / `gemm_h2h_vllm.py` (extended to M ∈ {2048…32768}),
`docs/rd_largem_data/quarot_gemm_largem.py`. Raw JSON: `docs/rd_largem_data/`.

Protocol notes. e2e = full-model prefill forward (`use_cache=False`, ids `(B, 2048)`),
cudaEvent, interleaved A/B with **paired per-iteration ratios** (contention-robust, the
rd_relat protocol). "FP8 endpoint" = per-Linear K_high = K−64 on all 7 targets
(avg FP8 fraction 0.983 (3B) / 0.989 (8B)): a **pure** K_low = 0 build is rejected by the
kernel (`pack_weight_mixed: K_low==0 … no single-phase MXFP8 path exists`) — the near-pure
endpoint is the honest FP8 end on this kernel. "FP4 endpoint" = K_high = 0 on all 7 targets
(the rd-kfix single-phase path). GEMM-isolated = cudaEvent brackets around every
`MixedQuantLinear._gemm` summed per forward (rd_perfpuzzle hooks). No OOM anywhere up to
batch 16 (M_eff 32 768) on the 96 GB part — the sweep is not memory-capped.

---

## 1. Format lever vs M — e2e flat, NOT growing (our stack)

t_FP8-endpoint / t_FP4-endpoint (paired-median; >1 = FP4 wins), in-model GEMM ratio, and
the FP4-endpoint GEMM share:

| model | M_eff | e2e FP8/FP4 | in-model GEMM FP8/FP4 | GEMM share (FP4 model) |
|---|---|---|---|---|
| 3B | 2 048 | 1.257 | 1.618 | 33.0 % |
| 3B | 4 096 | 1.265 | 1.694 | 34.0 % |
| 3B | 8 192 | 1.256 | 1.689 | 31.6 % |
| 3B | 16 384 | 1.234 | 1.668 | 29.6 % |
| 3B | 32 768 | **1.228** | 1.672 | 29.1 % |
| 8B | 2 048 | 1.338 | 1.732 | 39.1 % |
| 8B | 4 096 | 1.329 | 1.742 | 39.2 % |
| 8B | 8 192 | 1.291 | 1.737 | 35.4 % |
| 8B | 16 384 | 1.294 | 1.748 | 35.0 % |
| 8B | 32 768 | **1.296** | 1.883 | 35.3 % |

**The e2e format swing does NOT grow with M — it drifts slightly DOWN** (3B 1.257→1.228,
8B 1.338→1.296). Two reasons, both measured:

- The **in-model GEMM FP8/FP4 ratio is already large at M = 2048** (1.62–1.73) and almost
  flat in M (only the 8B 32 768 point rises to 1.88). It is much larger than the old 1.3×
  "stock-dispatcher" ratio because the fused kernel's Phase-1-only pure-NVFP4 path is fast
  while the near-pure-FP8 fused build leans on the slower MXFP8 phase — i.e. per-GEMM M was
  never the binding constraint *inside the fused kernel*; the microbench M-scaling
  (1.15→1.51 at 4096², rd_perfpuzzle) describes the stock pure dispatchers, not our
  production path.
- The **GEMM share of prefill does not grow with batch — it shrinks** (next section), so a
  flat GEMM ratio dilutes into a flat-to-falling e2e swing.

The M = 2048 numbers themselves are a useful correction: with the kfix single-phase path the
format swing at bs = 1 is ~26 % (3B) / ~34 % (8B) e2e — the earlier "~1 % at r=0" sweep rows
were the uniform-fallback artifact, as rd-perfpuzzle diagnosed.

## 2. Allocation lever vs M — ~0 % at every M (confirmed a pure accuracy lever)

t_uniform-1/8 / t_measured-MSE-1/8 (paired-median; >1 = measured faster). Measured
allocation realizes the same 0.125 avg FP8 fraction / 4.969 bits/weight as uniform.

| model | M_eff | unif/meas e2e | uniform-1/8 GEMM share | uniform whole (ms) |
|---|---|---|---|---|
| 3B | 2 048 | 1.0015 | 36.1 % | 54.6 |
| 3B | 4 096 | 0.9988 | 37.2 % | 85.4 |
| 3B | 8 192 | 0.9988 | 34.0 % | 178.1 |
| 3B | 16 384 | 0.9977 | 31.8 % | 380.9 |
| 3B | 32 768 | 0.9965 | 31.2 % | 759.6 |
| 8B | 2 048 | 0.9948 | 42.7 % | 83.8 |
| 8B | 4 096 | 0.9993 | 42.0 % | 163.1 |
| 8B | 8 192 | 0.9986 | 38.2 % | 338.8 |
| 8B | 16 384 | 0.9953 | 37.6 % | 692.7 |
| 8B | 32 768 | 0.9945 | 37.6 % | 1 385.9 |

**The allocation lever stays ≤ ±0.55 % at every M** — if anything the measured allocation
gets marginally *slower* at large M (its K_high counts concentrate FP8 rings in fewer,
larger segments; tile quantization is less kind there). Growing M does not unlock a speed
payoff from WHERE the FP8 rings go: at equal average budget, allocation remains an
**accuracy** lever (rd-sweep/rd-alloc conclusion), not a latency lever, at all serving M.

## 3. GEMM share of prefill vs M — shrinks, capping the e2e lever

Both Linears and attention scale ~linearly in batch (attention is O(B·S²) with S fixed), so
the share is roughly flat by construction — measured it *declines* a few points (3B 36→31 %,
8B 43→38 %) as the FP16 rest (SDPA, RMSNorm, RoPE, down_proj FP32 Hadamard,
casts, and the full-position lm_head+logits, which grows to multi-GB fp32 at M = 32 768)
scales slightly super-linearly. **Amdahl cap**: with GEMM ≤ 43 % of wall-clock, even an
infinitely fast quantized GEMM buys < 1.75×, and the realized format swing (sec. 1) is
1.23–1.34×. Batch does not relax the cap — the non-GEMM stack grows with it.

## 4. Peer head-to-head at each M (same shapes, same GPU)

### 4a. vLLM NVFP4 GEMM (FlashInferCutlassNvFp4, cutlass backend) vs our fused GEMM

Network-summed per-layer GEMM time (merged qkv + o_proj + merged gate_up + down, vLLM's own
decomposition; GEMM-only, median of 50, idle GPU):

| model | M | ours (ms) | vLLM (ms) | fp16 (ms) | ours/fp16 | vLLM/fp16 | **vLLM/ours** |
|---|---|---|---|---|---|---|---|
| 3B | 2 048 | 0.594 | 1.014 | 1.252 | 2.11× | 1.24× | **1.71** (we win) |
| 3B | 4 096 | 0.961 | 1.316 | 2.264 | 2.36× | 1.72× | **1.37** |
| 3B | 8 192 | 1.668 | 1.880 | 4.304 | 2.58× | 2.29× | **1.13** |
| 3B | 16 384 | 3.135 | 3.057 | 8.721 | 2.78× | 2.85× | **0.98** (cross) |
| 3B | 32 768 | 6.094 | 5.217 | 17.277 | 2.84× | 3.31× | **0.86** (vLLM wins) |
| 8B | 2 048 | 0.987 | 1.338 | 2.413 | 2.45× | 1.80× | **1.36** |
| 8B | 4 096 | 1.762 | 1.976 | 4.706 | 2.67× | 2.38× | **1.12** |
| 8B | 8 192 | 3.266 | 3.153 | 9.315 | 2.85× | 2.95× | **0.97** (cross) |
| 8B | 16 384 | 6.393 | 5.551 | 18.712 | 2.93× | 3.37× | **0.87** |
| 8B | 32 768 | 12.662 | 10.335 | 37.124 | 2.93× | 3.59× | **0.82** (vLLM wins) |

- The round-8 result ("we beat vLLM on 21/24 cells, median 1.37×, M ≤ 8192") **holds at
  M ≤ 8192** and **inverts at M ≥ 16 384**: our fused kernel saturates at ~1 000–1 200 TF
  while vLLM's cutlass fp4 keeps climbing to 1 250–1 460 TF (91 % of the 1.6 PF dense FP4
  peak) at M = 32 768. Per-shape, vLLM's win is concentrated in large-N `gate_up`
  (vLLM/ours 0.76–0.78 at 32 768) — the same large-N weakness rounds 6/19 flagged; we still
  win/tie o_proj and down up to 16 384.
- Including activation quant (vLLM `quant+gemm` vs our GEMM-only — conservative *against*
  us): vLLM q+g = 11.25 ms vs our gemm-only 12.66 ms at 8B/32 768; at 2 048 we win even
  giving away the pack (0.99 vs 1.49 ms).
- **Takeaway: the "beats vLLM NVFP4 GEMM 1.37×" claim is an M ≤ 8192 claim.** At serving
  M ≥ 16 384 the per-shape dispatch regime flips and vLLM's kernel is 1.16–1.22× faster
  summed; closing it needs a large-M/large-N config point (or stream-K/persistent variants)
  in our dispatch table.

### 4b. vLLM full-stack e2e prefill (TTFT, chunked prefill + FA2 + CUDA graphs)

vLLM 0.32, same protocol as `docs/baselines_on_hardware.md`, prefill L = 2048 × batch
{1,2,4,8,16}, median of 5 (data `docs/rd_largem_data/{nvfp4,bf16}_{3b,8b}.json`):

| model | batch (M_eff) | BF16 TTFT | NVFP4 TTFT | **NVFP4 speedup** | ours W4A4 whole* |
|---|---|---|---|---|---|
| 3B | 1 (2 048) | 45.4 ms | 33.6 ms | 1.35× | 54.6 ms |
| 3B | 2 (4 096) | 85.7 | 64.0 | 1.34× | 85.4 |
| 3B | 4 (8 192) | 160.6 | 85.0 | 1.89× | 178.1 |
| 3B | 8 (16 384) | 310.4 | 155.9 | 1.99× | 380.9 |
| 3B | 16 (32 768) | 608.3 | 284.1 | **2.14×** | 759.6 |
| 8B | 1 (2 048) | 93.3 | 43.1 | 2.16× | 83.8 |
| 8B | 2 (4 096) | 179.5 | 76.9 | 2.33× | 163.1 |
| 8B | 4 (8 192) | 350.6 | 138.8 | 2.53× | 338.8 |
| 8B | 8 (16 384) | 692.5 | 271.7 | 2.55× | 692.7 |
| 8B | 16 (32 768) | 1 501.7 | 524.1 | **2.87×** | 1 385.9 |

*ours = uniform-1/8 real-kernel transformers-stack prefill (sec. 2), shown for scale — not
protocol-identical (our harness computes full-position lm_head logits, FP16 SDPA, separate
pack, FP32 Hadamard; vLLM computes last-token logits only, FA2, fused quant, CUDA graphs).

**In a production serving stack the W4A4 format lever DOES grow with M** — vLLM NVFP4 goes
from 1.35×/2.16× (bs 1) to **2.14× (3B) / 2.87× (8B) at batch 16** over BF16-serving. The
contrast with our flat 1.23–1.34× swing localizes the deficit: it is NOT the GEMM (ours ≥
vLLM at M ≤ 8192) and NOT the format — it is our harness stack (full-position logits,
unfused pack, FP32 Hadamard, SDPA, no chunked prefill/graphs) whose cost also scales with M.

### 4c. QuaRot INT4 CUTLASS GEMM — non-competitive at every M (confirmed)

QuaRot's own `quarot._CUDA.matmul` (cutlass s4), built for SM120
(`/home/luofan/baselines/peers/quarot`), same shapes, GEMM-only
(data `docs/rd_largem_data/quarot_int4_gemm.json`):

| | M=2 048 | 8 192 | 32 768 |
|---|---|---|---|
| INT4/fp16 speed (3B shapes) | 0.15–0.24× | 0.20–0.23× | **0.23–0.24×** |
| INT4/fp16 speed (8B shapes) | 0.17–0.22× | 0.21–0.23× | **0.22–0.24×** |
| achieved TOPS | 46–87 | 74–91 | 88–92 (plateau) |

QuaRot INT4 is **4.2–6.8× SLOWER than FP16 at every M, and large M does not rescue it**: the
kernel plateaus at ~90 TOPS ≈ 11 % of the 814 TOPS INT8-rate ceiling (on SM120 s4 compiles
to the INT8 `IMMA.16832.S8.S8` pipeline — zero INT4 rate advantage,
`docs/hardware_format_peak.md`), and it is an unmaintained 2-launch INT4 path with no
Blackwell tuning.

### 4d. QServe / Atom — non-viable on Blackwell (reportable negative)

Per the committed peer audit (`docs/w4a4_peers_on_hardware.md` §4, re-confirmed unchanged):
**QServe**'s real W4A8 kernel does not build under CUDA 13 (`cub::Max` removed) — and is
W4A**8**, not a W4A4 peer; **Atom**'s INT4 lowers to INT8 IMMA on SM120 (same s4→S8 gate as
QuaRot/ResQ), no runnable speed path. On Blackwell the only W4A4 perf-viable peer family is
**NVFP4-based** (vLLM/TensorRT-class) — which is exactly the format family ProMix uses.

## 5. Is W4A4 ever e2e-competitive vs FP16 in our stack? (same-stack A/B)

`rd_largem_fp16.py`: plain FP16 transformers model vs the uniform-1/8 real-kernel model,
interleaved paired ratios, t_FP16/t_W4A4 (>1 = quantized wins):

| model | M_eff 2 048 | 4 096 | 8 192 | 16 384 | 32 768 |
|---|---|---|---|---|---|
| 3B | 0.978 | 1.188 | 1.132 | 1.131 | 1.138 |
| 8B | 1.240 | 1.240 | 1.269 | 1.277 | **1.288** |

(8B absolute: FP16 1 782.5 ms vs W4A4 1 384.1 ms at batch 16.) The W4A4-vs-FP16 e2e ratio
**does grow with model size and (mildly) with M** — 8B climbs 1.24→1.29 from batch 1 to 16 —
but in the swept range it **never crosses 1.3×, let alone 1.5×**: the GEMM saving
(fused/fp16 = 2.45–2.93× isolated) is Amdahl-diluted by the 57–62 % non-GEMM stack, and the
quant tax (pack ~6 %, FP32 Hadamard, casts) also scales with M. Extrapolating the flat
GEMM-share (~37.6 %) and flat format behavior, no batch size fixes this within the current
transformers-harness stack; the vLLM data in 4b shows the same model+format reaching 2.87×
when the stack is lean.

## Verdict

1. **The format lever does not grow with batch in our current stack.** e2e FP8→FP4 swing
   stays ~1.23–1.26× (3B) / ~1.29–1.34× (8B) from M = 2048 to 32 768; the in-model GEMM
   ratio is ~1.65–1.75 flat (1.88 at 8B/32k). The "ratio climbs to 2.05× at 16 384³" scaling
   belongs to square stock-dispatcher GEMMs, not to our per-layer shapes + fused kernel,
   and the GEMM share (31–43 %) actually *shrinks* with batch, Amdahl-capping e2e.
2. **The allocation lever is ~0 % at every M** (±0.55 %, slightly negative at 32k):
   measured-MSE ring placement is an accuracy lever only — at any serving scale.
3. **Same-stack W4A4 vs FP16 peaks at 1.29× (8B, batch 16)** — the best e2e number this
   stack has produced, rising with model size and M, but below the 1.3–1.5× competitive bar
   at every swept M. There is no M in 2k–32k where the current stack crosses it.
4. **Peers:** our fused GEMM beats vLLM NVFP4 1.36–1.71× at M = 2048 and holds to ~8192;
   **at M ≥ 16 384 vLLM overtakes (1.16–1.22× faster summed)** — a new, actionable kernel
   gap (large-M/large-N config/persistency, worst on gate_up). QuaRot INT4 is 4–7× slower
   than FP16 at all M (s4 = INT8 rate on Blackwell, kernel plateaus at ~90 TOPS);
   QServe (cub::Max, CUDA 13) and Atom (s4→INT8) don't run. The only live W4A4 perf race on
   this hardware is NVFP4-family vs NVFP4-family.
5. **W4A4 e2e-competitiveness at serving M is a stack property, not a format property.**
   vLLM NVFP4 reaches 2.14× (3B) / 2.87× (8B) TTFT over BF16 at batch 16 — the format DOES
   pay multi-× at serving scale when the non-GEMM stack is lean (chunked prefill, FA2,
   fused act-quant, CUDA graphs, last-token logits). Ours stays at 1.29× because pack, FP32
   Hadamard, SDPA and full-position logits scale with M alongside the GEMM. The
   data-mandated next steps are (a) stack work — fused pack-into-producer, FA2-class
   attention, last-token logits / graphed serving path — and (b) the new large-M GEMM gap
   vs vLLM (point 4); NOT more allocation tuning or small-M GEMM tuning.
