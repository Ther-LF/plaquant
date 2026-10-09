# Adaptive High-Fraction via Energy Threshold (main path, per-layer)

**Date:** 2026-10-09
**Status:** Design approved, pending spec review
**Context:** ProMix / PLAQuant — SM120 NVFP4+MXFP8 mixed-precision quantization

## Problem

ResQ fixes the high-precision subspace at `high_fraction = 1/8` for every
layer and every quantization site. This value is arbitrary — it is not
derived from any measured property of the activations. Eigenvalue analysis
of Llama-3.2-1B (the `E-wikitext-512` bundle) shows the fixed 1/8 is both
*too small on average* and *badly matched layer-to-layer*:

- Shared main-path curve (`attn_mlp`): 1/8 captures only **60.7%** of
  activation variance; reaching 90% needs ~65% of channels.
- Per-layer `down_proj` curves: 1/8 captures anywhere from **24% to 82%**
  depending on the layer — the same fixed cut is wasteful in some layers and
  starved in others.

The high_fraction sweep confirms the lever works (FP, per-head o_proj):

| high_fraction | PPL |
|---|---|
| 1/8 | 14.92 |
| 1/4 | 14.43 |
| 1/2 | 14.03 |

(INT W4A4 baseline at 1/8 = 14.68; FP16 ≈ 9.8.)

## Key Insight (and what it rules out)

A threshold only beats a fixed fraction when it is applied to **multiple
curves of different shapes**, giving each a different cut. On a *single*
curve, "pick K to capture τ of energy" and "pick K = fraction × dim" both
produce one number — the threshold is just a renamed knob. Therefore the
adaptive scheme must be **per-layer** (or per-site) to have any teeth.

Where heterogeneous curves exist and are usable in the current W4A4-FP
config:

- **Main path (q/k/v/gate/up)** — shared U, but each layer's activation has
  its own per-channel variance *in the shared basis*. 16 distinct curves,
  derivable from already-collected per-layer covariances at ~zero extra cost.
  **This is the target of this spec.**
- down_proj — uses online Hadamard, not PCA; the quantizer sees the
  *post-Hadamard* (flattened) distribution, so the E-file (pre-Hadamard)
  eigenvalues do not describe what is quantized. Confounded — **out of scope**.
- value / key — FP16 in the W4A4 config (`k_bits = v_bits = 16`), not
  quantized. **Out of scope** until a KV4 variant.
- o_proj — per-head 64-dim; rounding K to a multiple of 32 leaves only
  {0, 32, 64}, too coarse to adapt. **Out of scope** until full_global.

## Goal

Replace the fixed 1/8 on the main path with a single physically-meaningful
threshold τ = the fraction of per-layer activation energy (variance) the
high-precision subspace must capture. Sweep τ to produce a
(average bit-width, PPL) **Pareto curve**, and show it dominates the
fixed-fraction baseline points (lower PPL at equal bit-width, or lower
bit-width at equal PPL).

## Design

### 1. Step 0 addition (`promix/quantize/basis.py`)

After the shared `attn_mlp` basis `U` is computed (full_shared branch),
compute each layer's per-channel variance in the shared basis:

```
var_i = diag( Uᵀ · (H_attn[i] + H_mlp[i]) · U )      for i in 0..n_layers-1
```

`H_attn[i]` and `H_mlp[i]` are already collected per layer (they are only
summed to form the shared covariance), so this adds a handful of matmuls and
**no extra forward passes**. Store each `var_i` (length = hidden_dim,
indexed in the same channel order as `U`, i.e. ascending average variance)
under a new key `layer.{i}.attn_mlp_var` in the eigenvalue bundle (`E-…bin`).

Backward compatibility: the new key is additive; existing consumers ignore
it. Bundles built before this change simply lack the key (adaptive mode then
errors with a clear message telling the user to re-run Step 0).

### 2. Offline K-derivation (new helper, e.g. `promix/quantize/adaptive.py`)

```
def derive_per_layer_khigh(var_curves, tau,
                           khigh_align=64, hidden_dim=2048):
    # khigh_align = lcm(32, 64) = 64 so BOTH MXFP8 (K_high%32=0) and
    # NVFP4 (K_low = hidden_dim - K_high, %64=0) constraints hold at once.
    # var_curves[i]: per-channel variance of layer i in shared-U order
    #   (ascending avg variance → high-variance channels are the TAIL)
    # Return list of K_high_i.
    for each layer i:
        v = var_curves[i]
        tail_energy_fraction[k] = sum(v[-k:]) / sum(v)   # last k channels
        K = smallest k such that tail_energy_fraction[k] >= tau
        K_high_i = round_up(K, khigh_align)              # → multiple of 64
        K_low_i  = hidden_dim - K_high_i                 # → multiple of 64
    return [K_high_i ...]
```

The high-precision channels are always the **contiguous tail** of the shared
ordering (U is variance-sorted), so no per-layer channel reorder is needed —
only the cut length K_high_i varies per layer. This round deliberately does
NOT let each layer pick a *different channel set* (that would need per-layer
reorder); it only varies the cut length. (Noted as future work.)

Rounding K_high up to a multiple of 64 satisfies MXFP8 (K_high % 32 = 0,
trivially) and NVFP4 (K_low = 2048 − K_high % 64 = 0, since 2048 % 64 = 0)
simultaneously.

### 3. Rotation handling

Set the main-path R1 to the 2048×2048 identity, so the fused transform is
`T = U × I = U` (pure PCA). This removes the fixed [main|high] block
boundary that a trained R1 would impose, letting the cut K_high_i vary
freely per layer. Justification: on Llama-3.2-1B, Step 1 rotation
optimization yields no measurable PPL improvement (random orthogonal R is
already optimal), so dropping R1 on the main path costs effectively nothing
here. (Larger models that benefit from R1 are out of scope for this round;
re-introducing a per-layer-blocked R1 is future work.)

Implementation: supply an identity R1 in the rotation bundle (or a flag that
skips the `U_attn @ R1` multiply and uses `U_attn` directly). Per-head R2
(value/o_proj) and the o_proj/down_proj paths are untouched.

### 4. PTQ config / plumbing (`promix/eval/ptq.py`)

The per-layer quantizer configuration loop already sets `high_bits_length`
per layer. Extend it so that, in adaptive mode, each main-path layer
(q/k/v/gate/up) receives its own `K_high_i` (and `K_low_i`) from the derived
list rather than the global `high_fraction × hidden_dim`.

New config knob (in the `quantize` block):

```
adaptive_high_fraction: true      # enable per-layer threshold mode
energy_threshold: 0.85            # τ
```

When `adaptive_high_fraction` is true:
- main path (q/k/v/gate/up): per-layer K_high_i from the derived list;
- o_proj: unchanged (per-head fixed);
- down_proj: unchanged (current behavior);
- KV: unchanged (FP16).

When false, behavior is exactly today's fixed `high_fraction` path
(no regression).

### 5. Average bit-width accounting

For the Pareto x-axis, report the main-path activation average bit-width:

```
avg_bits = mean over main-path layers of
           (K_low_i · 4 + K_high_i · 8) / hidden_dim
```

(NVFP4 = 4 bits/elem, MXFP8 = 8 bits/elem; block-scale overhead is a small
constant identical across configs and omitted.) Every other site (o_proj,
down_proj, KV, norms, embeddings) is identical between the fixed and
adaptive runs, so it does not affect the comparison and is excluded from the
x-axis.

### 6. Experiment protocol

- **Fixed baseline:** global fraction ∈ {1/8, 1/4, 1/2} → three
  (avg_bits, PPL) points. (1/8 and 1/4 and 1/2 PPLs already measured:
  14.92 / 14.43 / 14.03; recompute avg_bits under the main-path-only
  accounting for a fair x-axis.)
- **Adaptive:** τ ∈ {0.6, 0.7, 0.8, 0.85, 0.9, 0.95} → six
  (avg_bits, PPL) points.
- All runs: Llama-3.2-1B-Instruct, wikitext PPL, same U bundle, R1 = identity
  on the main path, nsamples=512, GPU 6/7 on Pro 6000.

### 7. Success criteria

The adaptive Pareto curve dominates the fixed-fraction points: at matched
main-path average bit-width, adaptive PPL ≤ fixed PPL (ideally strictly
lower), or at matched PPL, adaptive uses fewer average bits. A flat/overlapping
curve is a valid negative result — it would mean the per-layer *magnitude*
differences on the main path (same channel ordering, different cut length)
are too small to exploit, pointing the next increment at per-layer *channel
sets* or at down_proj/KV4.

## Out of Scope (future increments)

1. down_proj adaptive — requires resolving the Hadamard confound (quantize
   on post-Hadamard variance, or decide down_proj should not split at all).
2. value/key adaptive — requires the KV4 config (`k_bits/v_bits = nvfp4`).
3. o_proj adaptive — requires full_global (2048-dim) + its online rotation.
4. Per-layer *channel-set* selection on the main path (each layer picks a
   different high-channel set, needs per-layer reorder) — this round varies
   only the cut length on the shared ordering.
5. Re-introducing a per-layer-blocked trainable R1 (matters for 3B/8B).

## Files Touched

- `promix/quantize/basis.py` — emit `layer.{i}.attn_mlp_var` in full_shared.
- `promix/quantize/adaptive.py` — new; `derive_per_layer_khigh(...)`.
- `promix/quantize/rotation.py` — identity-R1 path for adaptive mode.
- `promix/eval/ptq.py` — adaptive config knobs; per-layer main-path K_high.
- `promix/configs/llama-3.2-1b-fp-adaptive.yaml` — new config for the sweep.
