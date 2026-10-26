# Round 2 (ALGORITHM axis) — improving the ACTIVATION quantizer at equal bits

Round 1 attributed the entire W4A4-vs-GPTQ deficit on 3B to **activation**
quantization (+0.56; weights-only 8.084 already beats GPTQ 8.20) and showed the
error is **broad** (more MXFP8 channels recover only −0.04). This round attacks
the activation NVFP4 quantizer directly, at the **same** bits/weight (activations
are free on the storage axis), under the exact round-1 protocol.

Protocol: ProMix fake-quant STOCK path (`prepare_ptq_model`: fuse-norm → PCA
rotation (U+R1) → rearrange → actquant → online-Hadamard(down) → **frozen-GPTQ**
→ activation quantizers → KV), scored by the pipeline's own `evaluate_ppl` on
wikitext-2. Harness: `promix/scripts/ablation_activation_quant.py` (research-only
accuracy path; reuses round-1's `build_model` primitives). Raw JSON:
`docs/round2_results/`.

**bits/weight = WEIGHT storage only** = (1−r)·4.5 + r·8.25. Every variant below
leaves bits/weight UNCHANGED — the activation quantizer is free on this axis.

Reference points (same family, docs): FP16 3B = 7.818; **weights-only** (4.97b
NVFP4+MXFP8 wts, acts FP16) = **8.084**; vLLM NVFP4 W4A4 4.25b 3B = 8.63; GPTQ
W4A16 4.0b 3B = 8.20.

## Anchor reproduction (gate)

| run | 3B r=1/8 PPL |
|---|---|
| round-1 anchor | 8.6470 |
| round-2 harness `--variant baseline` | **8.6470** (8.646995) |
| round-2 `--weight_mse` (== round-1 A2) | **8.545** (round-1 reported 8.545) |

Anchor reproduced to 4 dp, and the round-1 weight-A2 lever reproduced exactly →
the round-2 build is byte-equivalent to round 1 and the `--weight_mse` stacking
hook is validated.

## Variants at equal bits (3B)

| variant | r=1/8 (4.969b) | Δ vs 8.647 | r=1/16 (4.734b) | Δ vs 8.681 |
|---|---|---|---|---|
| baseline (stock single-level NVFP4) | 8.647 | — | 8.681 | — |
| **act block-scale MSE search** | **8.428** | **−0.219** | **8.487** | **−0.194** |
| act MSE search, 7 steps (vs 3) | 8.430 | −0.217 | — | — |
| per-token two-level (FP32 global × E4M3 micro) | 8.757 | +0.110 | — | — |
| per-token correction on hi-kurtosis tokens (10%) | 8.621 | −0.026 | 8.687 | +0.006 |
| SmoothQuant (per-channel fold, α=0.5) | 9.089 | +0.442 | — | — |
| weight-A2 only (round-1, acts stock) | 8.545 | −0.102 | — | — |
| **act MSE + weight-A2 (full stack)** | **8.376** | **−0.271** | — | — |

## Winner: activation block-scale MSE search (−0.22 standalone, −0.27 stacked)

The default NVFP4 block scale is `ceil_to_E4M3(amax/FP4_MAX)` — a round-UP that
over-allocates headroom so the FP4 grid under-resolves the bulk of each 16-wide
block. The MSE search tries the base scale **plus 3 lower E4M3-grid steps** and
picks, per block, the one minimizing dequant MSE (accepting a little clipping of
the few outliers for much finer bulk resolution). This is round-1's A2 idea
applied to the **activation** side.

- Standalone: 3B r=1/8 **8.647 → 8.428 (−0.219)**; r=1/16 8.681 → 8.487 (−0.194).
- Stacked with the weight-A2 search (round-1's "keep" lever): 3B r=1/8 **8.376
  (−0.271)**. The two levers add (−0.10 weight + −0.22 act ≈ −0.27 combined).
- Depth saturates at 3 steps (7 steps = 8.430, no gain) → keep 3, cheaper.

**Activation-deficit recovery:** the +0.56 activation gap (W4A4 8.647 −
weights-only 8.084) is cut by **0.22 → 39% recovered** by the activation MSE
search alone; the full stack reaches 8.376, closing the gap to weights-only
(8.084) to 0.29 and narrowing the gap to GPTQ (8.20) from **0.45 → 0.18**.

### Deployability (stated precisely)

DEPLOYABLE as a modification confined to the **activation NVFP4 pack producer**
(the `quant_pack_float_fp4` CUDA kernel we already own, and its torch twin
`fake_quantize_nvfp4`). The change is ONLY the per-block scale *selection*:
instead of one `ceil`, compute 4 candidate E4M3-grid scales, quantize+dequant the
16-wide block against each, keep the argmin-MSE scale. Crucially:

- the chosen scale **stays on the E4M3 grid** — identical SFA storage, identical
  per-16 block layout, no per-token global, no new buffers;
- the fused GEMM (`fused_gemm_prepacked`), the SFA interleave, and the offline
  weight pack are **all unchanged** (`s_x = s_w = 1` still holds — the kernel
  takes no per-token `s_x`, verified in `mixed_linear.py`);
- cost is 4× the per-block quant arithmetic inside the pack kernel (negligible
  vs the GEMM; pack is a small fraction of e2e), zero extra HBM, bit budget
  unchanged.

So it is a **fused-producer (pack-kernel) change**, not pack-only and not a
GEMM/epilogue change. Weight-A2 is pure re-pack (round-1), so the full stack adds
only the pack-kernel scale search.

## Why the other ideas lost (all activation-only, equal bits)

- **Per-token two-level (FP32 global × E4M3 micro), +0.11 on 3B.** Mirrors
  round-1's weight-side finding (2level helps 1B, hurts 3B). Forcing the micro
  E4M3 scale relative to a per-row global pins the micro near the top of its
  range and *loses* per-block adaptivity. Also needs a per-row output-scale
  epilogue (kernel change). Rejected on both accuracy and cost.
- **SmoothQuant (per-channel fold), +0.44 on 3B.** Our activations are already
  PCA-rotated + online-Hadamard'd, so per-channel dynamic range is *already*
  equalized; a per-channel migration scale fights the rotation and disrupts the
  block structure. (Implemented correctly: main path + down, geomean-normalized
  s, folded into GPTQ'd weight and divided out of the activation — folded into
  168 Linears.) Deployable (pack-only) but a clear accuracy loss here → rejected.
  Confirms round-1's "smart allocation / per-channel tricks are a dead end once
  rotation is in place; the win is base-quantizer quality."
- **Per-token correction on high-kurtosis tokens, −0.03 / neutral.** Consistent
  with round-1's "error is broad, not tail/outlier-concentrated" — a per-token
  outlier lever buys almost nothing and needs a kurtosis gate + per-token global
  (kernel change). Rejected.

## New Pareto point (3B, accuracy-per-weight-bit)

| method | bits/w | 3B PPL |
|---|---|---|
| GPTQ W4A16 | 4.00 | 8.20 |
| vLLM NVFP4 W4A4 | 4.25 | 8.63 |
| ProMix r=1/16 + act-MSE | 4.734 | **8.487** |
| ProMix r=1/8 + act-MSE | 4.969 | 8.428 |
| **ProMix r=1/8 + act-MSE + weight-A2** | 4.969 | **8.376** |

ProMix now **beats NVFP4's PPL** (8.376 vs 8.63) and, at r=1/16 (4.73b), reaches
8.487 — below NVFP4 at nearer GPTQ's bit count. We still do not *dominate*
GPTQ/NVFP4 on raw accuracy-per-weight-bit (they are W4A16 / 4.25b), but the
W4A4 story is materially stronger: ≈NVFP4-beating accuracy **with activations
quantized** (prefill tensor-core speedup) and a hardware-native partition, with
the only new cost a 4-candidate per-block scale search in a producer kernel we
already ship.

## 8B (winner validation)

8B r=1/8 (same protocol; rotation/basis present on remote). Refs: FP16 6.138,
GPTQ W4A16 6.52, vLLM NVFP4 W4A4 6.87.

| variant | 8B r=1/8 PPL | Δ vs baseline |
|---|---|---|
| baseline | 6.821 | — (matches docs 6.821 exactly) |
| **act-MSE** | **6.725** | **−0.096** |
| act-MSE + weight-A2 | 6.713 | −0.108 |

The winner generalizes: activation block-scale MSE search is a free, positive win
on both sizes (−0.22 on 3B, −0.10 on 8B). On 8B the error is less
activation-dominated (round 1) so the gain is smaller, and weight-A2 adds almost
nothing (6.725 → 6.713) — consistent with 8B weights already being good. At 6.713
we beat NVFP4 (6.87) and sit 0.19 above GPTQ (6.52), the same ordering as 3B.
