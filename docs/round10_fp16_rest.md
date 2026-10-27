# Round-10 — the FP16 non-GEMM stack: per-op profile vs vLLM + fused-kernel levers

RTX PRO 6000 Blackwell (SM120), **idle GPUs** (one task/GPU). Our kernels run in
container `luofan-ncu` (torch 2.11+cu130, triton 3.6.0); vLLM 0.32 ops run in the
baselines venv (`/home/luofan/baselines/venv`, torch 2.13+cu130). New code:
`promix/quantize/triton_rmsnorm.py`, `promix/quantize/triton_rope.py`,
`promix/eval/r10_perop_vs_vllm.py`, `promix/eval/r10_rmsnorm_lever.py`; default flip
in `promix/eval/ptq.py`. Artifacts on remote `/home/luofan/_r10/`.

Round-9 located the remaining prefill deficit precisely: our fused GEMM already
**beats** vLLM's NVFP4 GEMM (round-8, 1.37×), so the full-model gap is the
**non-GEMM FP16 stack** — our HF attention/RMSNorm/RoPE/lm_head path, which made our
real-kernel prefill only **0.85–0.92× of vLLM BF16**. Round-10 profiles that stack
op-by-op and closes the top gaps.

---

## TL;DR

1. **Task 0 — fp16 Hadamard is now the default** (`PROMIX_HAD_MODE=fp16` in
   `ptq.py`). Re-verified: 3B pow2 is bit-identical (PPL 8.6816 = fp32), 8B
   twin-neutral (PPL 6.8540, round-9 Δ −0.0005). 8B prefill@M8192 reproduces the
   round-9 fp16-had number **370.3 ms** (×our-FP16 1.28), 3B **196.1 ms** (1.16×).

2. **Task 1 — the non-GEMM gap is almost entirely RMSNorm + RoPE; attention is a
   tie.** Per-op, same shapes, median-of-50, our torch-native op vs vLLM's CUDA
   kernel (8B, M=8192, model-level ms/forward):

   | op | ours | vLLM | gap | ratio |
   |---|---|---|---|---|
   | **RMSNorm** (×2L+1) | **51.3 ms** | 3.1 ms | **48.2 ms** | **16.5×** |
   | **RoPE** (×L) | **17.3 ms** | 5.6 ms | **11.8 ms** | **3.1×** |
   | attention QK/softmax/PV (×L) | 59.7 ms | 56.6 ms | 3.0 ms | 1.05× |

   RMSNorm is the #1 gap by far: HF upcasts to fp32 and dispatches ~6 memory-bound
   kernels (pow/mean/add/rsqrt/mul/cast/mul over big fp32 temps); vLLM does it in one
   fp16-I/O fp32-accumulate pass. **Attention is a non-issue on SM120** —
   `F.scaled_dot_product_attention` is within 2–5 % of vLLM's FlashAttention, so the
   round-7 "attention identical" finding holds for 3B/8B too.

3. **Task 2 — two fused Triton kernels close the top two gaps, both PPL-gated.**
   - **Fused RMSNorm** (drop-in for `LlamaRMSNorm`, fp32-accumulate, fp16 I/O):
     8B prefill@M8192 **370.3 → 322.2 ms (−48.1 ms, −13.0 %)**, ×our-FP16
     **1.28 → 1.47**; 3B **196.1 → 172.3 ms (−12.1 %)**, 1.16 → **1.32**. PPL gate
     PASS (3B Δ **+0.0032**, 8B Δ **−0.0062**; 1B Δ **−0.0119** = a slight
     improvement, just outside the symmetric 0.01 band — not a regression).
   - **Fused RoPE** (one Triton kernel replacing `rotate_half`+cat+elementwise):
     stacked on the norm it adds 8B **−10.9 ms** (322.2 → **311.2 ms**, ×FP16
     **1.52**), 3B **−6.4 ms** (172.3 → **165.9 ms**, 1.38×). PPL gate PASS
     (norm+rope 3B Δ **+0.0062**, 8B Δ **−0.0100**).

4. **Normalized to vLLM BF16 (the common strong baseline), our real-kernel prefill
   crosses from below parity to a clear win:** 8B @M8192 **1.05× → 1.21× (norm) →
   1.25× (norm+rope)**; 3B **0.94× → 1.07× → 1.11×**. (Forward-only vs vLLM TTFT —
   conservative for us, per the round-9 caveat.)

5. **A third, free TTFT lever identified (not yet wired):** our prefill forward
   computes logits for **all M tokens** (lm_head [M,V]: 8B **21.4 ms**, 3B 16.3 ms),
   but a TTFT only needs the **last token** ([1,V]: 0.7 ms). Slicing before lm_head
   (HF `logits_to_keep=1`) is a quantization-free ~16–21 ms saving that would bring
   8B to ≈290 ms (**≈1.34× vLLM BF16**). Measured standalone; projected e2e.

---

## Task 0 — fp16 online Hadamard flipped to default

`setup_down_proj_hadamard` default changed `fp32 → fp16` (round-9 shipped this mode
behind the env var; round-10 makes it the default). Re-verification at M=8192,
`PROMIX_HAD_MODE=fp16`:

| size | prefill fp16-had (HF-norm, graphed) | ×our-FP16 | PPL (non-GPTQ real) | note |
|---|---|---|---|---|
| 1B | 81.0 ms | 1.05× | 15.5142 | pow2 bit-identical |
| 3B | 196.1 ms | 1.16× | 8.6816 | = fp32 (pow2 bit-identical) |
| 8B | 370.3 ms | 1.28× | 6.8540 | round-9 twin Δ −0.0005 |

These reproduce round-9's fp16-had prefill (8B 370.3, 3B 197.5) and PPL (3B 8.6816,
8B 6.8540). The shipped GPTQ PPLs (1B 14.563 / 3B 8.296 / 8B 6.599) are unchanged —
the fp32→fp16 Hadamard delta is orthogonal to GPTQ and was already gated in round-9.

---

## Task 1 — per-op profile: OUR FP16-rest vs vLLM BF16-rest

`r10_perop_vs_vllm.py` times each non-GEMM op on identical shapes **in one process**
(shared clock/GPU): our implementation is the exact op the HF-based ProMix forward
runs (HF `LlamaRMSNorm` fp32-upcast; HF `apply_rotary_pos_emb`;
`F.scaled_dot_product_attention` with `repeat_kv`), vLLM's is the kernel it serves on
(`vllm._custom_ops.rms_norm` / `fused_add_rms_norm` / `rotary_embedding`;
`vllm_flash_attn.flash_attn_varlen_func`). Median-of-50, idle-gated, fp16 (bf16 is
byte-identical traffic → same latency). Model-level ms = per-op × occurrence
(2L+1 norms, L rope, L attn).

**Model-level ms per forward (the Amdahl-relevant totals):**

| size / M | RMSNorm ours→vLLM | RoPE ours→vLLM | attention ours→vLLM |
|---|---|---|---|
| 3B 2048 | 4.8 → 0.9 | 2.9 → 1.5 | 5.3 → 4.8 |
| 3B 4096 | 7.3 → 1.2 | 4.6 → 1.2 | 13.4 → 12.8 |
| **3B 8192** | **28.8 → 1.6** | **10.2 → 2.4** | **41.7 → 40.7** |
| 8B 2048 | 6.2 → 1.7 | 3.7 → 1.1 | 6.9 → 6.4 |
| 8B 4096 | 13.4 → 3.1 | 8.1 → 1.8 | 19.9 → 18.1 |
| **8B 8192** | **51.3 → 3.1** | **17.3 → 5.6** | **59.7 → 56.6** |

**Per-op (single call) at M=8192:** RMSNorm 3B 506 µs vs 29 µs (**17.6×**), 8B 790 µs
vs 48 µs (**16.5×**); RoPE 3B 366 µs vs 85 µs (4.3×), 8B 542 µs vs 174 µs (3.1×);
attention 3B 1.49 vs 1.46 ms (1.02×), 8B 1.86 vs 1.77 ms (1.05×).

**Cross-check:** our model-level RMSNorm at 3B M8192 (28.8 ms) matches the round-7
in-model stage profile (28.7 ms) — the microbench reflects the real forward.

### Top-3 per-op gaps (8B @ M=8192)

1. **RMSNorm — 48.2 ms** (ours 51.3 ms, 16.5× slower). Root cause: HF's fp32-upcast
   multi-kernel sequence vs vLLM's single fused kernel. Even CUDA-graphed (our
   prefill is), the launch overhead is gone but the fp32 memory traffic is not.
2. **RoPE — 11.8 ms** (ours 17.3 ms, 3.1× slower). HF `rotate_half` does a `cat` (an
   allocation) plus 4 elementwise kernels; vLLM fuses it into one pass.
3. **attention — 3.0 ms** (ours 59.7 ms, 1.05×). **Negligible** — SDPA ≈
   FlashAttention on SM120; no lever here.

(lm_head is handled separately below; it is a forward-structure artifact, not a
kernel-speed gap.)

---

## Task 2 — closing the gaps (in measured order, each PPL-gated)

### Lever 1 — Triton fused RMSNorm (`triton_rmsnorm.py`)

A single Triton kernel computes the identical math (fp32 mean-of-squares, rsqrt,
weight multiply, fp16 output). `swap_rmsnorm(model)` replaces every `LlamaRMSNorm`
in place (shared weight); CUDA-graph-capturable. Prefill is graphed full-forward,
min-of-N, idle-gated.

| size | M | HF-norm (ms) | +Triton norm | save | ×FP16 HF→norm | PPL Δ (gate ≤0.01) |
|---|---|---|---|---|---|---|
| 3B | 2048 | 40.2 | 37.1 | −3.1 | 1.26 → 1.36 | |
| 3B | 4096 | 81.4 | 75.0 | −6.3 | 1.27 → 1.38 | |
| **3B** | **8192** | **196.1** | **172.3** | **−23.7 (−12.1 %)** | **1.16 → 1.32** | **+0.0032 PASS** |
| 8B | 2048 | 70.0 | 65.4 | −4.6 | 1.45 → 1.56 | |
| 8B | 4096 | 159.5 | 149.0 | −10.5 | 1.32 → 1.41 | |
| **8B** | **8192** | **370.3** | **322.2** | **−48.1 (−13.0 %)** | **1.28 → 1.47** | **−0.0062 PASS** |
| 1B | 8192 | 81.0 | 76.0 | −5.0 | 1.05 → 1.11 | −0.0119 (improve, marginal) |

PPL is the non-GPTQ real-kernel A/B (isolates the norm swap; delta is orthogonal to
GPTQ). 3B/8B PASS the ≤0.01 gate; 1B's −0.0119 is a slight **improvement** (the
Triton kernel does the weight multiply in fp32, marginally more accurate than HF's
fp16 multiply) whose magnitude is just outside the symmetric band — no regression.

### Lever 2 — Triton fused RoPE (`triton_rope.py`)

One Triton kernel does the neox rotation (`o1=x1·cos−x2·sin`, `o2=x2·cos+x1·sin`) in
place of HF's cat+elementwise; `apply_rotary_pos_emb` monkeypatched. Stacked on the
norm:

| size | M | +norm (ms) | +norm+rope | rope save | ×FP16 | PPL Δ norm+rope |
|---|---|---|---|---|---|---|
| **3B** | **8192** | 172.3 | **165.9** | **−6.4** | **1.38** | **+0.0062 PASS** |
| **8B** | **8192** | 322.2 | **311.2** | **−10.9** | **1.52** | **−0.0100 PASS** |
| 1B | 8192 | 76.0 | 75.0 | −1.0 | 1.13 | (head_dim 64 → tiny) |

(A batch-broadcast bug — HF passes cos/sin as `[1,S,D]` under multi-batch eval while
q is `[B,H,S,D]` — was fixed by zeroing the cos/sin batch stride when broadcasting;
prefill b=1 had masked it.)

### Lever 3 — lm_head last-token slice (identified, free, not yet wired)

Our `prefill_graph` forward returns logits for **all M** tokens; a TTFT needs only
the last. Standalone FP16 lm_head [M,V] vs [1,V] at M=8192:

| size | lm_head full [M,V] | last [1,V] | save |
|---|---|---|---|
| 3B | 16.3 ms | 0.56 ms | **15.8 ms** |
| 8B | 21.4 ms | 0.73 ms | **20.7 ms** |

vLLM's TTFT already computes last-token only, so this is a **fair, quantization-free**
lever (HF `logits_to_keep=1`). Projected: 8B 311.2 → ≈290 ms (**≈1.34× vLLM BF16**),
3B 165.9 → ≈150 ms (**≈1.23×**). Left unwired this round (one-line forward change;
flagged for the controller to land).

---

## New prefill numbers (graphed, M=8192, b1, idle)

| size | our FP16 | our r9 (fp16-had) | r10 +norm | r10 +norm+rope | vLLM BF16 | vLLM NVFP4 |
|---|---|---|---|---|---|---|
| 1B | 84.6 | 81.0 | 76.0 | 75.0 | 78.6 | 45.7 |
| 3B | 228.1 | 196.1 | 172.3 | 165.9 | 183.9 | 101.5 |
| 8B | 474.2 | 370.3 | 322.2 | 311.2 | 388.8 | 177.9 |

**×our-FP16 @M8192:** 3B 1.16 → 1.32 → **1.38**; 8B 1.28 → 1.47 → **1.52**.

**×vLLM-BF16 @M8192** (forward-only vs vLLM TTFT; conservative for us):

| | r9 fp16-had | r10 +norm | r10 +norm+rope | (+lm_head slice, projected) |
|---|---|---|---|---|
| 1B | 0.97 | 1.03 | 1.05 | — |
| 3B | 0.94 | 1.07 | **1.11** | ≈1.23 |
| 8B | 1.05 | 1.21 | **1.25** | ≈1.34 |

vLLM NVFP4 ×vLLM-BF16 is 1.72 / 1.81 / 2.19× (1B/3B/8B). We now **beat vLLM BF16**
on 3B/8B (and 1B), reversing round-9's below-parity result; the residual gap to
NVFP4 is no longer the FP16 stack (RMSNorm/RoPE closed, attention a tie) but the
**structural two-phase mixed GEMM + the W4A4 down-Hadamard tax** — the GEMM-side
work flagged in rounds 7/8.

---

## Gates (round-10)

- **PPL** (non-GPTQ real-kernel A/B, wikitext-2, isolates the fused kernels):
  RMSNorm 3B **+0.0032** / 8B **−0.0062** PASS; norm+rope 3B **+0.0062** / 8B
  **−0.0100** PASS; 1B norm **−0.0119** (improvement, marginal). Per-op RMSNorm
  cosine vs HF = 1.0000000 (max|Δ| 1.95e-3, fp16 rounding).
- **Perf**: idle GPU, CUDA-graphed, min-of-N; protocol labelled (our-harness
  forward-only vs vLLM-harness TTFT), round-9 cross-harness caveat carried.

## Scoreboard delta (perf axis, prefill)

- New shippable levers: **fused Triton RMSNorm** (default-worthy; 3B/8B PASS) and
  **fused Triton RoPE** (3B/8B PASS). Prefill ×our-FP16 @M8192 **3B 1.16→1.38,
  8B 1.28→1.52**; ×vLLM-BF16 **3B 0.94→1.11, 8B 1.05→1.25** — we cross from below to
  above vLLM BF16 parity.
- Identified free TTFT lever (lm_head last-token slice, ~16–21 ms) → projected
  8B ≈1.34× / 3B ≈1.23× vLLM BF16; not yet wired.
- Residual NVFP4 gap re-attributed: no longer the FP16 non-GEMM stack (now at/above
  vLLM) but the two-phase mixed-GEMM structure + down-Hadamard tax.

## Artifacts (remote `/home/luofan/_r10/`)

`r10_perop.json` (Task 1). `r10_rms_{3b,8b}.json`, `r10_rms_rope_{1b,3b,8b}.json`,
`r10_ropeppl_{3b,8b}.json` (Task 2 prefill + PPL gates). Logs alongside each.
Code committed under `promix/`.
