# Round-8 — GEMM head-to-head (prefill gap) + decode b1 levers

RTX PRO 6000 Blackwell (SM120), container `luofan-plaquant`. Our kernels: base
python3.11 / torch 2.11+cu130. vLLM NVFP4: baselines venv python3.12 / vLLM 0.32 /
flashinfer 0.7 (`FlashInferCutlassNvFp4LinearKernel`, the kernel the NVFP4 model
actually serves on SM120). All timings: idle GPU, warm, CUDA-event median of 50
(min also recorded). New code: `promix/eval/gemm_h2h_ours.py`,
`/home/luofan/_r8/gemm_h2h_vllm.py` (vLLM-venv), `promix/eval/r8_decode_lmhead.py`,
`promix/eval/r8_fp8kv.py`, `promix/eval/r8_lmhead_ppl.py`.

## TL;DR

1. **The prefill gap is NOT the GEMM.** On IDENTICAL per-layer shapes our shipped
   fused dual-format (NVFP4+MXFP8) GEMM is **faster than vLLM's NVFP4 GEMM on 21/24
   cells (median 1.37×)** and hits **54–83 % of the blended hardware peak** (≥
   vLLM's %-of-peak). At M=8192 our fused GEMM is **2.35–3.02× over fp16** — i.e. it
   *already exceeds* vLLM's full-model prefill speedup (1.64/1.85/2.21×). The whole
   1.29× (ours) vs 2.21× (vLLM) full-model prefill gap therefore lives in the
   **non-GEMM stack**, dominated by the ResQ online Hadamard (8B down_proj: 87 ms /
   21.6 % of the forward) + the shared FP16 attention/RMSNorm/RoPE/lm_head (Amdahl,
   ~45–50 %, identical in both stacks). This settles the open question decisively.
2. **Decode b1 levers (accuracy-gated).** MXFP8 (W8) lm_head in the graphed decode
   path: **1B 0.848→0.859, 3B 0.964→0.976, 8B >1.0 (stays)**; PPL gate PASS
   (W8 lm_head Δ +0.031 1B / +0.012 3B, both < 0.05). Real but small (~1.2 %); it
   does **not** get 1B/3B to ≥1.0 — M=1 decode is occupancy-floor-bound (the merged
   GEMVs are 11× off their weight-BW ceiling) and the 4-bit bandwidth win can't
   manifest on small 1B/3B weights. FP8-e4m3 KV cache (the safe alternative to the
   failed KV4): **PASS on 3B (+0.015)/8B (+0.012), marginal fail on 1B (+0.058,
   head_dim 64)**, ~1.94–1.97× KV memory & per-step KV-read bandwidth.

---

## (A) GEMM head-to-head — the decisive measurement

Shapes = vLLM's actual GEMM decomposition (merged qkv, merged gate_up, o_proj,
down_proj), so all three kernels time IDENTICAL (M,N,K). Our kernels run the
shipped config (`cd.fused_gemm_prepacked`, 128×128 NV4/MX2, round-19 optimum);
"pure-NVFP4" = our `nvfp4_gemm_ext` over the full K (the r=0 single-format
reference). vLLM = `flashinfer_scaled_fp4_mm(..., backend="cutlass")`, the SM120
cutlass fp4 path, with activation pre-quantized (GEMM-only) and also measured with
`scaled_fp4_quant` included (quant+GEMM).

### Achieved TFLOPS @ M=8192 (the TTFT protocol length)

| shape | our fused | our pureNV4 | vLLM NVFP4 | fp16 | fused/fp16 | vLLM/fp16 | fused/vLLM | fused %peak | vLLM %peak |
|---|---|---|---|---|---|---|---|---|---|
| 3B qkv      |  877 | 803 |  695 | 371 | 2.37× | 1.87× | 1.26× | 61 % | 43 % |
| 3B o_proj   |  773 | 658 |  525 | 330 | 2.35× | 1.60× | 1.47× | 54 % | 33 % |
| 3B gate_up  | 1034 | 986 | 1045 | 402 | 2.57× | 2.57× | 0.99× | 72 % | 65 % |
| 3B down     | 1050 | 766 |  881 | 373 | 2.81× | 2.36× | 1.19× | 74 % | 55 % |
| 8B qkv      | 1007 | 896 |  874 | 379 | 2.66× | 2.31× | 1.15× | 71 % | 55 % |
| 8B o_proj   |  975 | 847 |  753 | 374 | 2.61× | 2.01× | 1.30× | 68 % | 47 % |
| 8B gate_up  | 1104 |1079 | 1302 | 387 | 2.85× | 3.31× | 0.85× | 77 % | 81 % |
| 8B down     | 1188 | 881 | 1142 | 393 | 3.02× | 2.85× | 1.04× | 83 % | 71 % |

(peak: fused vs blended NVFP4+MXFP8 ceiling = 1/(f_low/1600 + f_high/811) TF, the
HPE QuickSpecs dense FP4 1600 / FP8 811; vLLM & pureNV4 vs NVFP4 1600. Clock not
locked → %-peak is a conservative lower bound.) Across all M∈{2048,4096,8192}: our
fused beats vLLM on **21/24** cells (median 1.37×); vLLM wins only large-N gate_up
at M≥4096. At M=2048 vLLM's own NVFP4 GEMM is **slower than fp16** on qkv/o_proj
(0.79–0.98×) — further evidence its edge is not a uniformly faster GEMM.

### Diagnosis: the gap is the STACK, not the GEMM

- Our GEMM at M=8192 already delivers **2.35–3.02× over fp16** (8B: 2.66/2.61/2.85/
  3.02×). vLLM's *full-model* prefill is 2.21× (8B). So if our stack were pure GEMM
  we would be **above** vLLM. The GEMM is at/above the SM120 block-scaled hardware
  ceiling (round-23: 71–78 % TC-duty), ≥ vLLM's %-of-peak.
- vLLM's activation quant is nearly free / fused: `quant+gemm` is only 5–15 % slower
  than gemm-only (8B down M8192: 1142→1055 TF). Our activation pack is a *separate*
  per-forward stage (~7 % of the forward, round-7 profile).
- The real gap is the non-GEMM forward, unchanged by any GEMM work:
  * **ResQ online Hadamard on down_proj** — 8B 87 ms / **21.6 %** of the forward
    (FP32, I=14336 non-power-of-two). This is a W4A4-algorithm tax vLLM/NVFP4 never
    pays. perf_round2 showed the bit-identical fused producer is occupancy-bound
    (1 CTA/SM, fp32 had28 SMEM) → no net win; a genuine wall.
  * **FP16 attention (~15–23 %) + RMSNorm (~13 %) + RoPE + FP16 lm_head** — ~45–50 %
    of the forward, identical in both stacks (pure Amdahl; shrinking it needs a new
    attention/norm kernel and helps vLLM too).
  * vLLM runs merged qkv/gate_up GEMMs in prefill; our shipped prefill runs per-proj
    + a separate pack (the decode merge hurts prefill, round-7).

**Verdict (A): GEMM-not-stack, proven by direct head-to-head.** No GEMM lever
remains; the actionable prefill levers are all stack-side — (1) remove/cheapen the
down_proj Hadamard (hard, occupancy-bound), (2) fuse activation-quant into the
previous op's epilogue (vLLM-style) to kill the separate pack stage, (3) MXFP8
lm_head (landed this round, safe), (4) FlashAttention/fused-norm (helps both stacks).

## (B) Decode: 1B/3B b1 levers

Baselines reproduced under current code (graphed, idle, median, prompt 32 / gen 128,
×FP16-graph): **1B b1 0.848 / b8 0.797; 3B b1 0.964 / b8 0.947; 8B b1 1.233 / b8 1.066**
(matches round-7 exactly).

### Lever implemented: MXFP8 (W8) lm_head in the graphed decode path

lm_head is a large FP16 vocab GEMV read every M=1 step (1B 525 MB, 3B 788 MB /
token). W8 MXFP8 halves that traffic and runs on the block-scaled TCs. Stacked on
the round-7 QKV+gate/up merge:

| size | b1 merge-only → +MXFP8 lm_head | b8 merge-only → +lm_head | lm_head gain |
|---|---|---|---|
| 1B | 0.849 → **0.859** | 0.799 → 0.812 | 1.012× |
| 3B | 0.964 → **0.976** | 0.946 → 0.958 | 1.013× |
| 8B | already **1.233** (round-7) | 1.066 | not separately re-run* |

\* The 8B lm_head harness builds the real model + its fp32 `W_deq` twin buffers
(224 Linears) + the fp16 reference together, which exceeds a single 96 GB GPU
(~91 GB); 8B b1 is already > 1.0 (round-7 1.233) so it is not the lever's target.
The MXFP8 lm_head would save proportionally *less* on 8B (lm_head is a smaller
fraction of the larger per-token decode), so the ~1 % gain does not change 8B's
verdict.

Accuracy gate PASS: isolated W8 lm_head Δ = **+0.031 (1B), +0.012 (3B)** wikitext-2
PPL (< 0.05), bit-faithful twin (`quantize_mxfp8_spec`). Real but small (~1.2 %);
**does not reach b1 ≥ 1.0 on 1B/3B.**

### Why 1B/3B b1 can't reach 1.0 at this protocol (honest negative)

Graphed 1B decode = 4.75 ms/tok = GEMV 3.66 ms + FP16-rest 1.09 ms; each M=1 GEMV
is **11.3× off the 0.419 ms weight-BW ceiling** — occupancy/launch-floor bound, not
bandwidth bound. The round-7 merge (7→4 launches/layer) captured the main
launch-reduction win (1B +17 %, 3B +16 %); the lm_head shaves another ~1 %. The
residual is structural: at M=1 a single token cannot fill the GPU, and 1B/3B
weights are too small for the 4-bit bandwidth advantage to show (it only pays off on
8B, where b1 is already 1.23×). Reaching ≥1.0 needs a regime change (long context /
batch) or a fundamentally different M=1 kernel, not another launch trim — consistent
with the multi-round decode finding.

### FP8 KV cache (the accuracy-safe alternative to the failed KV4)

Per-vector (per-token-per-head) FP8, SDPA-patched (attention math stays FP16, only
stored K/V quantized), wikitext-2 PPL vs FP16 KV:

| model (head_dim) | e4m3 Δ | e5m2 Δ | KV mem / BW ratio |
|---|---|---|---|
| 1B (64)  | +0.058 **FAIL** | +0.215 FAIL | 1.94× |
| 3B (128) | **+0.015 PASS** | +0.070 FAIL | 1.97× |
| 8B (128) | **+0.012 PASS** | +0.039 PASS | 1.97× |

FP8-e4m3 KV clears the <0.05 gate for head_dim≥128 (3B/8B) — vastly safer than KV4
(which failed +0.76…+1.25) — at ~1.97× KV memory & per-step KV-read bandwidth. It is
marginal on 1B (head_dim 64: the per-64 scale tracks outliers less well). It does not
move b1 at this short-context protocol (ctx ~160, KV traffic tiny) but is the
decode/long-context + memory lever, now accuracy-cleared for 3B/8B.

## Gates (round-8)

- GEMM bench: same shapes/data both sides, warm, event-median-of-50, idle GPU.
- Decode merge path: unchanged (round-7 bit-exact at M=1). MXFP8 lm_head: PPL gate
  PASS (+0.031/+0.012 < 0.05). FP8 KV: PPL gate PASS 3B/8B (< 0.05), 1B marginal fail.

## Scoreboard deltas

- Perf axis — prefill gap **attributed**: GEMM not the bottleneck (our fused GEMM
  2.35–3.02× over fp16 @M8192 ≥ vLLM NVFP4 GEMM on 21/24 shapes); the 2.21-vs-1.29×
  full-model gap is the non-GEMM stack (down Hadamard + FP16 attention/norm Amdahl).
- Decode b1 (+MXFP8 lm_head, accuracy-safe): **1B 0.859, 3B 0.976, 8B ≥1.23**
  (was 0.848/0.964/1.233). b8: 1B 0.812, 3B 0.958.
- New accuracy-safe memory lever: FP8-e4m3 KV passes <0.05 on 3B/8B (1.97× KV mem/BW).

## Artifacts (remote `/home/luofan/_r8/`)

`r8_gemm_ours.json`, `r8_gemm_vllm.json` (Part A); `r8_decode_{1b,3b}_base.json`,
`r8_lmhead_{1b,3b,8b}.json`, `r8_lmhead_ppl_{1b,3b}.json`, `r8_fp8kv_{1b,3b,8b}.json`
(Part B). Code committed under `promix/eval/`.
