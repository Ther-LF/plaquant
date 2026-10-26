# Round-3 integration (both axes) — wiring the round-1/2 wins into the shared forward

RTX PRO 6000 Blackwell (SM120), container `luofan-plaquant`, torch 2.11+cu130.
This round took the four previously-validated round-1/2 wins and wired them into
the **shared** ProMix forward (`promix/`), then re-measured PPL + prefill + decode
across 1B/3B/8B with the existing gates. All timing is on idle GPUs (one size per
GPU, env-level `CUDA_VISIBLE_DEVICES` isolation); accuracy shared GPUs.

## What landed vs reverted

| win | status | reason |
|---|---|---|
| **Weight-A2 MSE block scales** (round1) | **SHIPPED** (3B/8B), per-size | Re-pack only (GPTQ frozen scales → packer), **zero forward cost**. 3B real −0.08, 8B neutral. **1B regresses +0.08** (round1 flagged 1B r=1/8 +0.02) → gate 1B off. |
| **8B fused producer `hadamard28`** (perf r2) | **SHIPPED** | Bit-identical (cos 1.0000005). 8B prefill 1.24→1.34x (M=4096), 1.22→1.32x (M=8192). |
| **decode-GEMV v5 tunable warps** (perf r2) | **SHIPPED** | Bit-exact (M=1 path). 8B decode b1 0.805→**1.063x**. 1B/3B e2e ~flat (GEMV is a small share at those sizes). |
| **Activation block-scale MSE search** (round2) | **REVERTED to opt-in** (`-DPROMIX_ACT_MSE=1` / `PROMIX_ACT_MSE=1`) | Big ACCURACY win (3B −0.27, 8B −0.11) but **5.3× the fused-producer pack cost** (8B pack 9.2→48.7 ms/forward) → **8B prefill 1.32→0.78x, decode-b8 1.03→0.80x** (prefill gate FAIL). Kept available for an accuracy-max config; ship needs a cheaper producer. |

The activation-MSE revert follows the loop rule "if an integration breaks a gate,
revert that piece and report it." The isolated A/B that proved the cost
(`r3_packab.py`, 8B shapes, M=2048): MSE on = 1.52 ms/layer, off = 0.29 ms/layer.

## Correctness gates (all PASS)

- **Per-layer cosine (real vs fake twin), 1B, act-MSE+wA2 on:** worst **0.999998**
  across all 112 Linears — the CUDA activation-MSE producer and weight-A2 real pack
  are bit-faithful to the torch twin.
- **PPL twin gate (1B):** |real − twin| = 0.0043 ≤ 0.1.
- **Prefill graph cos (real vs eager):** 1.0000004–1.0000013 at every M, all sizes.
- **8B producer bit-identity (`hadamard28`), decode M=1 bit-exactness, div-fold,
  epilogue bias/cast fusion, 1B Hadamard fusion:** preserved (prefill/decode cos
  1.000000x; decode cos 1.000000 on the M=1 path).
- **Baseline reproduction:** with both MSE levers off (`PROMIX_ACT_MSE=0
  PROMIX_WEIGHT_MSE=0`) the fake-quant harness reproduces the documented
  pre-integration numbers exactly (8B r=1/8 6.8214, 1B r=1/16 15.4746, 3B r=1/16
  8.6815) → the gating is a clean superset of the prior code.

## PPL per size (real kernel, wikitext-2)

Refs: BF16 1B/3B/8B = 13.16 / 7.82 / 6.14. vLLM NVFP4 W4A4 = 15.65 / 8.63 / 6.87.
GPTQ W4A16 = 14.22 / 8.20 / 6.52. Pre-integration real = 14.849 / 8.615 / 6.823.

| size | pre-integration real | **SHIPPED** (weight-A2) | accuracy-max (act-MSE + wA2) |
|------|----------------------|-------------------------|------------------------------|
| 1B | 14.849 | 14.849 (wA2 off on 1B)¹ | **14.7604** (real) |
| 3B | 8.615  | **8.5326** (−0.082)     | 8.376 (fake²) |
| 8B | 6.823  | **6.7990** (−0.024)     | **6.7146** (real) |

¹ weight-A2 ON on 1B gives 14.9318 (+0.083, gate FAIL) → ship weight-A2 OFF for 1B
  (1B's absolute PPL is not cross-comparable per the scoreboard anyway).
² 3B accuracy-max real OOM'd on a shared GPU; the fake-quant full-stack = 8.3759
  (reproduces round2's 8.376) and 3B real-vs-twin tracked to 0.004 on 1B, so real
  ≈ fake.

**Accuracy-max beats NVFP4's PPL at every size** (3B 8.38 < 8.63, 8B 6.71 < 6.87)
and **with activations quantized (W4A4)** — but at 4.97 bit/weight vs NVFP4's 4.25,
so it does not *dominate* NVFP4 on accuracy-per-weight-bit.

## Accuracy-per-bit (fake-quant, accuracy-max config: act-MSE + weight-A2)

| size | r | base (no MSE) | **improved** | Δ |
|------|-----|---------------|--------------|------|
| 1B | 1/8 (4.969b)  | 14.9195 | **14.7617** | −0.158 |
| 1B | 1/16 (4.734b) | 15.4746 | **14.8276** | −0.647 |
| 3B | 1/8 (4.969b)  | 8.6470  | **8.3759**  | −0.271 |
| 3B | 1/16 (4.734b) | 8.6815  | **8.4627**  | −0.219 |
| 8B | 1/8 (4.969b)  | 6.8214  | **6.7126**  | −0.109 |

## Prefill ×FP16 (CUDA-graph, idle GPU, graphed-vs-graphed)

| size | M=2048 | M=4096 | M=8192 | (pre-integration) |
|------|--------|--------|--------|-------------------|
| 1B | 1.218 | 1.306 | 1.285 | (1.215/1.302/1.271) |
| 3B | 1.387 | 1.434 | 1.354 | (1.385/1.437/1.354) |
| 8B | **1.322** | **1.339** | **1.318** | (1.293/1.241/1.224) |

8B prefill is the headline perf win: the `hadamard28` fused producer lifts it from
1.24/1.22 → 1.34/1.32 at M=4096/8192 (bit-identical). 1B/3B unchanged (they use the
power-of-two `hadamard_pack` producer, untouched).

## Decode ×FP16 (static-KV graph, prompt 32 / gen 128)

| size | b1 | b8 | (pre-integration) |
|------|------|------|-------------------|
| 1B | 0.722 | 0.709 | (0.712/0.731) |
| 3B | 0.854 | 0.832 | (0.859/0.860) |
| 8B | **1.063** | **1.033** | (0.805/1.006) |

8B decode b1 crosses FP16 (0.805→1.063x) via the decode-GEMV v5 tunable-warps
(16–24 warps on the large Linears). 1B/3B e2e ~flat: at those sizes the GEMV is a
small slice of per-token time (launch/attention-bound), as perf-r2 predicted.

## Weight memory (unchanged — recipe identical)

1B 1946→604 MB, 3B 5637→1751 MB, 8B 13959→4335 MB (3.22×, 4.969 bit/weight). The
MSE levers only change *scale values* on the native grid; layout/bytes are identical.

## Pareto statement

**We do not yet beat NVFP4 on BOTH axes at any single point.**
- Accuracy: accuracy-max ProMix beats NVFP4's *absolute* W4A4 PPL at every size
  (3B 8.38 vs 8.63; 8B 6.71 vs 6.87) — but at 4.97 bit vs 4.25, so NVFP4 still wins
  accuracy-*per-bit*.
- Perf: shipped 8B prefill 1.32x and decode-b1 1.06x both beat FP16 but remain below
  NVFP4 (prefill 2.21x, decode 1.61x).
- The two MSE levers do not stack cleanly: weight-A2 is free but per-size (hurts 1B);
  activation-MSE is the big accuracy win but costs ~24% of the prefill forward →
  accuracy-max and perf-max are currently different operating points. Closing both
  needs a cheaper activation-MSE producer (branchless FP4-nearest instead of the
  per-candidate binary search) so it can ship in the prefill path.
</content>
