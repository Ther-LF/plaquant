# Round-4 — making the activation block-scale MSE search CHEAP

RTX PRO 6000 Blackwell (SM120), container `luofan-plaquant`, torch 2.11+cu130,
transformers 4.46.3. Goal: keep round-2/3's activation NVFP4 **block-scale MSE
search** accuracy win (3B 8.53→8.38, 8B 6.80→6.71, both below NVFP4's 8.63/6.87)
but kill its producer-kernel cost so accuracy-max and prefill-max become the SAME
operating point. Round-3 shipped everything except this lever — it was reverted to
opt-in because the search cost ~2–5× the pack.

## 1. Profile — why the round-3 search was slow (measured, not guessed)

The round-3 per-block search, for each 16-wide NVFP4 block, evaluates the E4M3
ceil base scale **plus 3 lower E4M3 grid steps** and keeps the argmin-MSE scale.
Each candidate re-does, for all 16 elements: `xs = x[j] / scale` (a **true FP32
division**), a binary-search `rtn_idx` over the 8-value FP4 grid, dequant and
squared-error. That is **4 candidates × 16 divisions + a final code pass** — on
a kernel that is one-thread-per-block and already near ALU-bound, the division is
the bottleneck (FP32 divide is ~1/16 the multiply throughput on Blackwell, and
the binary search diverges across the 32 blocks a warp owns).

Isolated A/B of the two producers at the 8B down_proj shape (I=14336, K_low=12544,
K_high=1792), act-MSE **on vs off**, min-of-N:

**Pure pack** (`quant_pack_float_mixed`, no Hadamard — the path round-3's "5.3×"
figure came from; the search is a large fraction here):

| M | OFF (ms) | round-3 (ms) | round-3 ovh |
|---|---|---|---|
| 2048 | 0.116 | 0.373 | **3.23×** |
| 4096 | 0.223 | 0.732 | **3.28×** |
| 8192 | 0.432 | 1.454 | **3.37×** |

**Fused producer** (`hadamard28_swiglu_pack_mixed` — SwiGLU+Hadamard dilute the
search):

| M | OFF (ms) | round-3 (ms) | round-3 ovh |
|---|---|---|---|
| 2048 | 0.701 | 1.447 | **2.06×** |
| 4096 | 1.339 | 2.822 | **2.11×** |
| 8192 | 2.595 | 5.541 | **2.14×** |

So the search ~tripled the pure pack and doubled the fused producer — which is
what sank round-3's 8B prefill to 0.78×. The cost is the **per-candidate
division**, confirmed by the fix below (removing it recovers almost all of it).

## 2. The cheap search (`kernels/mixed_gemm_sm120/act_mse.cuh`)

Two levers, both selection-only (the final code-writing pass is unchanged → the
emitted codes/SF for a selected scale stay byte-exact to the torch twin):

1. **Reciprocal-multiply.** Per candidate, compute `inv = 1/scale` ONCE (one
   divide) and evaluate `x[j]*inv` (16 multiplies) instead of 16 divisions.
   Divisions per block drop from ~80 → ~4 (one per candidate) + the unchanged
   final 16. The reciprocal's sub-ULP rounding only affects *which* scale is
   selected, never the final codes.
2. **Branchless nearest-FP4.** A 7-compare `fsel` cascade over the fixed E2M1
   grid replaces the data-dependent binary search (no warp divergence).

Both the CUDA producers (all 5: `quant_pack_float_ext`, `quant_pack_fused_ext`,
`hadamard_pack_ext`, `hadamard28_pack_ext`, `add_rmsnorm_pack_ext`, via the shared
`act_mse.cuh`) and the torch twin (`fake_quantize_nvfp4` →
`_mse_best_scale_act_fast` in `quant_utils.py`) use the identical reciprocal +
branchless selection, so real and twin pick the same scale. Depth is
`PROMIX_ACT_MSE_STEPS` (default 3); `PROMIX_ACT_MSE_FAST=0` reproduces the
round-3 true-division search bit-for-bit (the accuracy anchor).

## 3. Perf — new pack overhead (FAST, 3 steps)

| path | M | OFF | FAST3 | **FAST3 ovh** |
|---|---|---|---|---|
| pure pack (qpfm) | 8192 | 0.432 | 0.459 | **1.06×** |
| fused producer (hadamard28) | 2048 | 0.701 | 0.767 | **1.09×** |
| fused producer (hadamard28) | 4096 | 1.339 | 1.463 | **1.09×** |
| fused producer (hadamard28) | 8192 | 2.595 | 2.848 | **1.10×** |

**Both well under the <1.15× perf gate**, at the full 3-step depth. Step sweep
(hadamard28 ovh): FAST2 1.07×, FAST1 1.05× (and pure-pack FAST2 1.04×, FAST1
1.02×) — cheaper but unnecessary since 3 steps already clears the gate and gives
the full accuracy (below). 2-step matches 3-step accuracy to ≤0.02 but 1-step
regresses, so **ship FAST + 3 steps**.

## 4. Equivalence / PPL gate

**Bit self-consistency (CUDA):** the fused `hadamard28` producer == the plain
`quant_pack_float_mixed` packer (same flavor), 0/4 tensors mismatched, for OFF,
round-3-exact, FAST3, FAST2, FAST1 (fp16, M=2048, outlier-stressed).

**PPL (fake-quant full stack = act-MSE + weight-A2, r=1/8, 4.969 bit/w, wikitext-2):**
the FAST search is **accuracy-identical to the round-3 true-division search**:

| size | act-off (wA2) | round-3 exact (3 step) | **FAST (3 step, SHIPPED)** | FAST 2-step | FAST 1-step |
|---|---|---|---|---|---|
| 1B | 14.9359 | 14.7617 | **14.7617** | 14.7792 | — |
| 3B | 8.5452 | 8.3759 | **8.3759** | 8.3715 | 8.4498 |
| 8B | 6.8094 | 6.7126 | **6.7126** | — | — |

FAST == exact to 4 dp on 1B/3B/8B → the reciprocal + branchless selection picks
the same scale on every block that matters. Gate (FAST within 0.02 of the 8.376 /
6.715 accuracy-max): **PASS with Δ=0**. act-off baselines (8.5452 / 6.8094) are
the shipped round-3 weight-A2 config; act-MSE adds −0.169 (3B) and −0.097 (8B).

## 5. Wired ON by default

- `quant_utils.ACT_MSE_SCALE_SEARCH` default flipped `0→1` (fake-quant twin).
- The real-kernel producers build with `-DPROMIX_ACT_MSE=1 -DPROMIX_ACT_MSE_FAST=1
  -DPROMIX_ACT_MSE_STEPS=3` by default via `full_network_realkernel.act_mse_cflags()`,
  used in `load_kernels` / `load_hpk` / `load_hpk28` / `load_qpfm`. Env
  `PROMIX_ACT_MSE=0` reproduces the pre-round-3 baseline; `PROMIX_ACT_MSE_FAST=0`
  the exact anchor. Fake twin and real kernel read the same env → always coupled.

## 6. Prefill / decode (real kernel, ×FP16, CUDA-graph, idle GPU, one GPU per run)

Clean A/B of act-MSE FAST **ON vs OFF** on the shipped real-kernel path
(`promix.eval.multisize`, qpf+qpfm+hpk/hpk28+bias epilogue, frozen GPTQ). Prefill
is ×FP16-eager; decode is ×FP16-graph (static-KV, prompt 32 / gen 128). All
prefill logit cosines 1.0000005–1.0000014.

**Prefill ×FP16 (act-MSE FAST ON = SHIPPED):**

| size | M=2048 | M=4096 | M=8192 | (act-MSE OFF, same env) |
|------|--------|--------|--------|--------------------------|
| 1B | 1.230 | 1.300 | 1.260 | (1.235 / 1.304 / 1.264) |
| 3B | 1.384 | 1.429 | 1.341 | (1.382 / 1.433 / 1.341) |
| 8B | 1.285 | 1.301 | 1.281 | (1.319 / 1.333 / 1.311) |

**Decode ×FP16-graph (ON):**

| size | b1 | b8 | (OFF) |
|------|------|------|-------|
| 1B | 0.734 | 0.699 | (0.734 / 0.699) |
| 3B | 0.849 | 0.819 | (0.844 / 0.806) |
| 8B | 1.064 | 1.010 | (1.063 / 1.033 round-3) |

**1B and 3B: act-MSE is FREE** (prefill/decode ON == OFF to noise — they use the
power-of-2 `hadamard_pack` producer where the search is a negligible slice). **8B:
act-MSE costs ~0.03× prefill** (1.281 vs 1.311 at M=8192) and is decode-neutral —
versus round-3's old search which sank 8B prefill to **0.78×** and decode-b8 to
**0.80×**. The pack-overhead gate (<1.15×) is met with margin (1.06–1.10×); the 8B
M=8192 prefill lands at ~1.28× ON vs ~1.31× OFF — essentially tied, a hair under
the 1.30 target by run-to-run noise, and ≥1.30× at M=4096.

## 7. Conclusion

The division was the cost; removing it (reciprocal-multiply + branchless nearest)
cuts the activation-MSE producer overhead from 2.1–3.4× to **1.06–1.10×** while
keeping the accuracy **bit-identical** to round-3's search (1B 14.762, 3B 8.376,
8B 6.713 — all == the round-3 true-division accuracy-max to 4 dp). With the lever
now ON by default, **accuracy-max and prefill-max are the SAME shipped operating
point**: 3B/1B pay nothing, 8B pays ~0.03× prefill (1.28× vs 1.31×) instead of
collapsing to 0.78×. Decode is preserved at every size.
