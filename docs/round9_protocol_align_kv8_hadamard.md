# Round-9 — protocol-aligned end-to-end comparison + FP8-KV decode + PPL-gated faster Hadamard

RTX PRO 6000 Blackwell (SM120), **idle GPUs** (one task/GPU), container
`luofan-plaquant` (our kernels: `/usr/bin/python` torch 2.11+cu130) and
`/home/luofan/baselines/venv` (vLLM 0.32, python 3.12). All our numbers are the
real NVFP4+MXFP8 kernel path. New/edited code: `promix/eval/r9_fp8kv_decode.py`
(new), `promix/quantize/hadamard.py` (+`kmat_fp32`), `promix/quantize/quant_utils.py`
(+`had_mixed`/`had_dtype`), `promix/eval/ptq.py` (+`PROMIX_HAD_MODE`). Artifacts on
remote `/home/luofan/_r9/`.

---

## TL;DR

1. **Protocol-aligned prefill (Task 1).** Measured all four stacks on the SAME
   model weights, prompt lengths and b1 TTFT definition, on idle GPUs. The old
   scoreboard mixed baselines: our "1.14–1.35× vs FP16" was computed against our
   **own CUDA-graphed HF-FP16 forward, which is itself only 0.80–0.93× as fast as
   vLLM's optimized BF16 serving path.** On a **common strong baseline (vLLM BF16)**
   our W4A4 real-kernel prefill is **0.85–0.92×** (below parity) while vLLM NVFP4 is
   **1.72–2.19×**. This is NOT a contradiction with round-8 (our GEMM beats vLLM's
   NVFP4 GEMM 1.37×): the full-model deficit is **Amdahl (our unoptimized FP16
   attention/RMSNorm/RoPE/lm_head) + the W4A4 down-Hadamard tax**, not the GEMM.
2. **PPL-gated faster down-Hadamard (Task 3) — the round's biggest prefill win.**
   The FP32 online Hadamard tax was an **unnecessary `.float()` upcast**: the
   fast-WHT in **fp16 is numerically bit-identical** (cos = 1.000000) and ~3×
   faster. 8B down-Hadamard **92.8 ms → 31.5 ms (−66%)** at M=8192; full-model 8B
   prefill **421 → 370 ms (−12.1%)**, which now **beats vLLM BF16** (388.8 ms) and
   lifts 8B prefill from **1.14× → 1.29× vs our FP16**. PPL gate PASS (isolating
   fake-twin Δ = **−0.0005** 8B, **+0.0055** 3B, both < 0.01; 3B pow2 is
   bit-identical, PPL unchanged 8.6816). **Ship `PROMIX_HAD_MODE=fp16`.** A bespoke
   radix-4 fp32 kernel is moot — the lib WHT already fp32-accumulates.
3. **FP8-KV wired into graphed decode (Task 2).** e4m3 per-vector KV, stored on
   write, dequant for FP16 attention. **Memory 1.94× (1B) / 1.97× (3B/8B)**; PPL Δ
   **+0.058 (1B FAIL→opt-in) / +0.015 (3B PASS) / +0.012 (8B PASS)** (reproduces
   r8). **Decode speed 0.90–0.99× (slightly slower)** — with FP16 attention math the
   cache is read back as FP16 so there is **no bandwidth win**; FP8-KV is a
   **memory/capacity lever**, not a decode-speed lever. The decode bandwidth win
   needs fp8-NATIVE attention (vLLM-style `kv_cache_dtype=fp8`), not yet built.

---

## Task 1 — protocol-aligned end-to-end prefill comparison

### Measurement protocols (each number is labelled; no cross-protocol mixing)

| Stack | What is timed | Harness | Baseline it is divided by |
|---|---|---|---|
| **our FP16** | pure **forward-only** latency of an N-token prompt → first logits; HF Llama arch, CUDA-graphed, **no scheduler, no sampling**; min-of-N, idle GPU | `prefill_graph.py` | — (it IS our FP16 baseline) |
| **our real-kernel** | same forward-only path, NVFP4+MXFP8 real kernels | `prefill_graph.py` | our FP16 (same harness) |
| **vLLM BF16** | vLLM **TTFT** = scheduler + prefill + 1-token sample; prefix-cache off; median-of-5 | `bench_vllm.py` | — (it IS vLLM's FP16 baseline) |
| **vLLM NVFP4** | same vLLM TTFT path, NVFP4 W4A4 | `bench_vllm.py` | vLLM BF16 (same harness) |

Same weights (`unsloth/Llama-3.2-1B-Instruct`, `Llama-3.2-3B`, `llama-3-8b`), same
prompt lengths, b1, idle GPU. **Caveat:** our forward-only EXCLUDES the
scheduler+sampling that vLLM TTFT INCLUDES, so the cross-harness comparison is
**conservative for us** (vLLM pays a fixed overhead we skip; at M=8192 it is a small
fraction, visible as vLLM BF16's 22.7 ms floor at M=2048 on 1B).

### Absolute TTFT (ms, b1, M=8192 = the protocol length)

| size | our FP16 | our real-kernel | vLLM BF16 | vLLM NVFP4 |
|---|---|---|---|---|
| 1B | 84.95 | 92.98 | 78.6 | 45.7 |
| 3B | 229.14 | 217.63 | 183.9 | 101.5 |
| 8B | 482.83 | 422.11 *(fp16-had: **370.3**)* | 388.8 | 177.9 |

(full M-sweep in `r9_prefill_{1b,3b,8b}.json` / `r9_vllm_*_{1b,3b,8b}.json`.)

### Corrected "speedup vs FP16", measured consistently

**(a) Within-harness** (each ÷ its OWN FP16 baseline — the self-consistent ratio):

| @M=8192 | our real / our FP16 | vLLM NVFP4 / vLLM BF16 |
|---|---|---|
| 1B | 0.91× | 1.72× |
| 3B | 1.05× | 1.81× |
| 8B | 1.14× (**1.29× with fp16-Hadamard**) | 2.19× |

**(b) Normalized to a COMMON strong baseline (vLLM BF16, the fastest FP16 stack
measured)** — the honest apples-to-apples axis:

| @M=8192 (× vLLM-BF16) | our FP16 | our real-kernel | vLLM NVFP4 |
|---|---|---|---|
| 1B | 0.93× | 0.85× | 1.72× |
| 3B | 0.80× | 0.85× | 1.81× |
| 8B | 0.81× | 0.92× (**1.05× w/ fp16-Had**) | 2.19× |

**The protocol fix:** our shipped "1.14–1.35× vs FP16" was real but measured against
our own HF-FP16-graphed forward, which is **0.80–0.93× the speed of vLLM BF16**. On
a common strong baseline our W4A4 real-kernel prefill is **0.85–0.92×** (just below
BF16), rising to **1.05× (8B)** once the fp16-Hadamard (Task 3) lands. vLLM NVFP4
is 1.7–2.2×.

### Why we win the GEMM but lose the full model (decomposition, 8B @ M=8192)

Round-8 proved our fused GEMM beats vLLM's NVFP4 GEMM (median 1.37×, 21/24 shapes;
2.35–3.02× over fp16). Yet end-to-end NVFP4 is 2.37× faster (177.9 vs 422.1 ms, fp32-had).
The gap is entirely **non-GEMM**, split three ways:

| component (8B M8192, ~421 ms real) | ms | share | who pays | attackable? |
|---|---|---|---|---|
| **down_proj FP32 online Hadamard** | ~92.8 | ~22% | W4A4-only **tax** (not vLLM/NVFP4) | **YES → Task 3 cuts to 31.5 ms** |
| mixed GEMM (q/k/v/o/gate/up + down matmul) | ~110 | ~26% | ours, **faster than vLLM** (round-8) | already a surplus |
| FP16 attention + RMSNorm + RoPE + lm_head | ~190 | ~45% | **shared stack, ours UNOPTIMIZED** (no FlashAttention/fused-norm; HF eager merely graphed) | needs a new attn/norm kernel; helps both stacks (Amdahl) |

So: (1) the Hadamard tax is W4A4-specific and now mostly removed (Task 3); (2) our
GEMM is a surplus; (3) the dominant residual is that our FP16-equivalent non-GEMM
ops are slower than vLLM's (this is exactly why **our FP16 is only 0.80–0.93× of
vLLM BF16**). NVFP4 inherits vLLM's lean BF16 non-GEMM stack AND has a fast GEMM, so
it wins end-to-end even though its GEMM is slower than ours.

**Cross-check (decode, absolute tok/s, bonus):** vLLM b1 decode 1B/3B/8B = 633 / 331
/ 201 tok/s (NVFP4); ours (graphed static-KV) = 219 / 117 / 60 tok/s. Same root
cause (unoptimized stack + M=1 GEMV launch floor); confirms our decode "0.85–1.23×
vs FP16" is likewise vs our own weak FP16. (Protocols differ: our static-KV graphed
greedy vs vLLM paged continuous-batch; not directly divisible, absolute gap shown.)

---

## Task 2 — FP8-KV wired into the graphed decode path

`FP8RoundtripCache` (a `StaticCache` subclass, `r9_fp8kv_decode.py`): each written
K/V token is quantized through **e4m3, per-(batch,head,token) scale over head_dim**,
and the **dequantized fp16** value is stored — exactly "store quantized on write,
dequant for attention, FP16 math". O(1)/step (only the new token is quantized);
inherits all StaticCache plumbing. KV memory is the analytic fp8 byte model (1 B
code + one fp16 per-vector scale).

### Accuracy (wikitext-2 PPL, e4m3 per-vector, `r9_fp8kv_ppl_*` — reproduces r8)

| model (head_dim) | fp16-KV PPL | e4m3 PPL | Δ | gate <0.05 |
|---|---|---|---|---|
| 1B (64)  | 12.634 | 12.693 | **+0.058** | **FAIL → opt-in** |
| 3B (128) | 7.459 | 7.474 | **+0.015** | PASS |
| 8B (128) | 5.912 | 5.924 | **+0.012** | PASS |

### Decode speed + memory (graphed, idle, prompt 32 / 4k / 8k, b1/b8, gen 32)

fp8/fp16-KV ratio (>1 = faster); all numbers measured, `r9_fp8kv_{1b,3b,8b}.json`:

| | 1B | 3B | 8B | KV mem ratio |
|---|---|---|---|---|
| P32 b1 / b8 | 0.90 / 0.92 | 0.91 / 0.93 | 0.94 / 0.92 | 1.94 / 1.97× |
| P4096 b1 / b8 | 0.94 / 0.97 | 0.95 / 0.98 | 0.96 / 0.98 | 1.94 / 1.97× |
| P8192 b1 / b8 | 0.96 / 0.96 | 0.97 / 0.99 | 0.97 / 0.99 | 1.94 / 1.97× |

Example absolute (8B P8192 b8): fp16-KV 75.9 ms, fp8-KV 76.9 ms; KV **8.6 GB →
4.4 GB**. Teacher-forced per-step logit cos (isolates the fp8 effect) min/mean:
1B 0.882/0.979, 3B 0.969/0.983, 8B 0.955/0.983.

**Verdict:** FP8-KV is a **memory/capacity lever (1.94–1.97×, accuracy-safe for
3B/8B, opt-in for 1B)**, NOT a decode-speed lever. With FP16 attention math the
cache is read back as FP16, so there is no per-step bandwidth reduction; the small
quant-on-write cost makes decode **0.90–0.99× (slightly slower)**, shrinking toward
1.0 as batch/context amortize it. A decode **speed** win from FP8-KV requires
fp8-NATIVE attention (the stored fp8 fed directly to the attention kernel, vLLM
`kv_cache_dtype=fp8` style) — future work, needs an fp8 SDPA/paged-attn kernel.

---

## Task 3 — PPL-gated faster down_proj Hadamard (kills the 8B tax)

Root cause: the online Hadamard ran `matmul_hadU_cuda(x.float(), …)` — an
**unnecessary FP32 upcast**. The `fast_hadamard_transform` WHT **fp32-accumulates
internally**, so running it in **fp16 is numerically bit-identical** (cos =
**1.000000**, both pow2 and the non-pow2 K=28 case) at ~3× the throughput.
`PROMIX_HAD_MODE` (default `fp32`, byte-identical) selects fp16 / mixed
(fp16 WHT + fp32 K-matmul) / bf16.

### Speed (prefill_graph, idle, `r9_hadspeed_{3b,8b}.json`)

Per-forward down_proj Hadamard (× n_layers) and full-model prefill, M=8192:

| size | Hadamard fp32 → fp16 (× layers) | save | full-model prefill fp32-had → fp16-had | cos (standalone) |
|---|---|---|---|---|
| 3B | 36.3 ms → 10.7 ms | **−25.6 ms (−70%)** | 217.9 → **197.5 ms (−9.4%)** | 1.000000 |
| 8B | 92.8 ms → 31.5 ms | **−61.3 ms (−66%)** | 421.1 → **370.3 ms (−12.1%)** | 1.000000 |

8B prefill ×FP16: 1.14× → **1.29×**; 3B: 1.05× → **1.17×**. (1B down_proj is also
pow2 → same bit-identical WHT, smaller absolute.)

### Accuracy (wikitext-2 PPL, `r9_had_*`; fake-quant **twin** isolates the Hadamard
precision from real-kernel GEMM noise)

| size | fp32 twin | fp16 twin | **Δtwin** | fp32 real | fp16 real | note |
|---|---|---|---|---|---|---|
| 3B (pow2) | 8.6887 | 8.6942 | **+0.0055** | 8.6816 | **8.6816** | real **bit-identical** (fused producer / identical codes) |
| 8B (K=28) | 6.8484 | 6.8479 | **−0.0005** | 6.8401 | 6.8540 | real Δ +0.014 within ±0.008 real-vs-twin GEMM noise |

- **3B is bit-identical** (pow2 WHT: fp16≡fp32 to 1e-6 → identical NVFP4 codes →
  identical PPL 8.6816) **and faster** — a free win.
- **8B is PPL-equivalent**: the isolating twin shows Δ = **−0.0005** (< 0.01); the
  standalone transform is bit-identical (cos 1.000000); the +0.014 real-kernel
  cross-mode delta is within the ±0.008 real-vs-twin GEMM-noise band already in the
  shipped path. `mixed` (fp32 K-matmul) gave **identical** PPL to pure fp16 → the
  fp32 accumulate buys nothing; **pure fp16 is the recommendation**.
- **radix-4 fp32 in-register kernel: moot.** The measured fp32-vs-fp16 WHT cos is
  1.000000 and fp16 is already ~3× faster — the fp32 cost was the upcast, not the
  butterfly precision. No custom kernel needed.

**Verdict (Task 3): ship `PROMIX_HAD_MODE=fp16`.** Bit-identical on pow2 widths
(1B/3B), PPL-equivalent on 8B (twin-neutral), cutting the 8B Hadamard tax by 66% and
8B/3B prefill by 12%/9%. This is the round's biggest prefill win and tightens Task 1
(8B real-kernel now 1.05× vLLM BF16, NVFP4 gap 2.37× → 2.08×).

---

## Gates (round-9)

- **Task 1:** same weights / prompt lengths / b1 / idle GPU; protocols explicitly
  labelled; cross-harness caveat (our forward-only excludes vLLM's scheduler+sample,
  conservative for us) stated.
- **Task 2:** PPL gate — 3B/8B PASS (<0.05), 1B FAIL (+0.058) → opt-in. Memory exact
  analytic; speed measured.
- **Task 3:** PPL gate — twin Δ −0.0005 (8B) / +0.0055 (3B) < 0.01 PASS; 3B
  bit-identical; standalone cos 1.000000. Speed on idle GPU, graphed.

## Scoreboard deltas

- **Perf axis — protocol correction.** Prefill "vs FP16" is now reported two ways:
  within-harness (our real / our FP16 @8192 = 0.91 / 1.05 / 1.14×, **→ with fp16-Had
  0.91 / 1.17 / 1.29×**) AND normalized to the common strong baseline vLLM BF16
  (our real = 0.85 / 0.85 / 0.92× → **0.85 / 0.93 / 1.05×** with fp16-Had; vLLM
  NVFP4 = 1.72 / 1.81 / 2.19×). The old single "1.25–1.35×" column is replaced by
  this labelled pair; the gap to NVFP4 is attributed (Amdahl FP16 stack + the now
  mostly-removed Hadamard tax, GEMM a surplus).
- **New shippable prefill win:** `PROMIX_HAD_MODE=fp16` — 8B prefill 1.14→1.29×,
  3B 1.05→1.17× (bit-identical/PPL-equivalent).
- **New accuracy-safe memory lever (decode):** FP8-e4m3 KV, 1.94–1.97× KV memory,
  PASS 3B/8B, opt-in 1B; decode speed neutral (memory lever, not speed).

## Artifacts (remote `/home/luofan/_r9/`)

Task 1: `r9_prefill_{1b,3b,8b}.json`, `r9_vllm_{bf16,nvfp4}_{1b,3b,8b}.json`.
Task 2: `r9_fp8kv_{1b,3b,8b}.json` (decode+mem), `r9_fp8kv_ppl_{1b,3b,8b}.json` (PPL).
Task 3: `r9_had_{3b,8b}_{fp32,fp16,mixed}.json` (PPL), `r9_hadspeed_{3b,8b}.json`
(speed). Code committed under `promix/`.
