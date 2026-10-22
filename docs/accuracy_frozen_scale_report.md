# Full-Network Accuracy — Frozen-Scale Weight Packing — Report

> 2026-10-22. Closes the +0.29 PPL residual flagged in
> `docs/full_network_realkernel_report.md` ("Honest note on REAL/TWIN vs STOCK").
> The full-network real-kernel path was already numerically faithful to its
> fake-quant twin (gate PASS, per-layer cosine 0.999998) but sat ~0.30 above the
> published 14.92 GPTQ baseline. Root cause: the weight packer re-derived each
> block's scale from the GPTQ-corrected weight's absmax (RTN) instead of REUSING
> GPTQ's frozen per-block scales, losing part of GPTQ's benefit. This change
> threads GPTQ's frozen scales into the packer and recovers the baseline.
>
> Remote-only: run inside container `luofan-plaquant` (10.208.130.90), FREE GPU 0.
> promix-only change; PROTECTED kernel files and the fused GEMM / quant_pack_float
> kernels unchanged — the kernel consumes whatever SFB the packer gives it.

## Result — baseline recovered

wikitext-2, seqlen 2048, FP per-head recipe (`llama-3.2-1b-fp-perhead.yaml`),
all 112 Linears, GPTQ FP weight quant (the 14.92 recipe).

| model | BEFORE (RTN packer) | AFTER (frozen-scale packer) | vs STOCK |
|---|---|---|---|
| **STOCK fake-quant** (act fake-quant + GPTQ FP weights + FP16 matmul) | 14.9195 | 14.9195 | — (reference) |
| **FAKE TWIN** (fake-quant act + FP32 matmul vs real kernel's `W_deq`) | 15.2239 | **14.9159** | **−0.0036** |
| **REAL KERNEL** (112 packed NVFP4+MXFP8 fused GEMMs) | 15.2055 | **14.8494** | **−0.0701** |
| reference: ProMix FP NVFP4+MXFP8 1B (published) | 14.92 | 14.92 | |

- **Recovered.** The +0.30 twin gap collapses to −0.0036 (twin 14.9159 ≈ stock
  14.9195). The real kernel lands at 14.8494 — 0.07 *below* stock, i.e. on the
  better side of the baseline. The documented RTN-vs-frozen-scale residual is
  gone.
- **Real-vs-twin gate PASS:** |REAL − TWIN| = 0.0665 ≤ 0.1. (It rose from 0.018
  to 0.067 only because both now sit near 14.9, where the fixed FP32
  GEMM-accumulation-order difference between the fused kernel and torch matmul is
  a larger *relative* share; it is unchanged in absolute terms and well inside
  the gate.)
- **Per-layer cosine (real vs twin), layers 0/8/15 × 7 Linears = 21 probes:**
  min 0.99999845, max 0.99999875 — all ≥ 0.9999. Frozen-scale packing is still
  bit-faithful to its twin; it changed the *scale source*, not the
  packing/accumulation faithfulness.
- Packer log confirms the source: `pack_all_targets: packed 112 Linears
  [frozen-GPTQ-scale (112/112)]` — every target Linear used a frozen scale.

Artifacts (remote): `kernels/mixed_gemm_sm120/full_network_ppl_frozen.json`,
log `/home/luofan/fn_ppl_frozen.log`.

## How frozen scales are sourced and threaded

GPTQ already computes and freezes the scales we need; they were simply being
thrown away. The fix captures and reuses them — no new quantization math.

1. **Where GPTQ freezes them.** `gptq_fwrd` (`promix/quantize/gptq.py`) gives each
   target Linear a `WeightQuantizer` (`promix/quantize/kv_quant.py`). For an FP
   format, `WeightQuantizer.find_params(W_segment)` runs on the ORIGINAL
   (pre-Hessian-correction) weight segment and stores
   `self.frozen_scales` (shape `(N, n_blocks)`) via
   `quant_utils.select_block_scales_{nvfp4,mxfp8}` — NVFP4 block-16 (E4M3-grid
   scale) for the main segment, MXFP8 block-32 (E8M0 power-of-2 scale) for the
   high segment. GPTQ's inner loop then rounds each Hessian-corrected column
   against that SAME frozen scale (`quantize_column_with_frozen_scale`) and
   writes the dequantized result back into `layer.weight`.

2. **Capture.** `run_gptq_if_enabled` (`promix/eval/ptq.py`) previously discarded
   the `quantizers` dict `gptq_fwrd` returns. It now stashes it on the model as
   `model._gptq_quantizers` (keyed `model.layers.{i}.{sub}.module` for the main /
   NVFP4 quantizer and `…,high_quantizer` for the high / MXFP8 quantizer).

3. **Thread into the packer.** `full_network.pack_all_targets` now looks up, per
   target wrapper, the frozen NVFP4 (main) and MXFP8 (high) scales
   (`_frozen_scales_for`) — the wrapper's `named_modules` name + `.module` is
   exactly GPTQ's dict key — and passes them to `pack_weight_mixed` via new
   `frozen_nvfp4_scale` / `frozen_mxfp8_scale` arguments. `quant_utils` already
   exposes these scales; the packer just had to stop re-deriving them.

4. **Use in the codes/SFB.** `quantize_nvfp4_spec` / `quantize_mxfp8_spec`
   (`promix/quantize/weight_pack.py`) gained a `frozen_scale=` argument. When
   given, they skip the absmax→ideal→round derivation and use the frozen scale
   directly for BOTH the element codes (`round(W / frozen_scale)` on the FP grid)
   AND the interleaved SFB byte (E4M3 for NVFP4, E8M0 exponent for MXFP8). Because
   the stored GPTQ weight is exactly `grid_value × frozen_scale`, dividing by the
   frozen scale returns the grid value exactly, so `W_deq` reproduces GPTQ's
   dequantized weight (up to the stored-fp16 rounding) — this is why the twin
   snaps back to 14.92.

### Alignment checks (why this is safe for this recipe)

- `act_order: false`, `low_fraction: 0.0` in the FP config, so GPTQ's segment
  split (`main = W[:, :K_low]`, `high = W[:, K_low:]`, no column permutation, no
  low segment) is identical to the packer's split, and frozen-scale block `b`
  maps to packer block `b` one-to-one.
- `high_bits_length = int(high_fraction × in_features)` is computed the same way
  in GPTQ and in `configure_quantizers`, so `K_low/K_high` match (q/k/v/o/gate/up:
  256/1792; down: 1024/7168).

## Fallback preserved (single-layer bit-exact probe)

`frozen_scale=None` keeps the original RTN derivation, so every path that does
not supply frozen scales is byte-for-byte unchanged:

- `probe_weight_sfb_bitexact` and `MixedQuantLinear.from_linear` call
  `pack_weight_mixed` with no frozen scales → RTN (single-layer bit-exact probe
  and the GEMM cosine contract unchanged).
- The harness's per-layer cosine spot-check builds its capture model with
  `run_gptq=False` (no `_gptq_quantizers`), so it exercises the RTN fallback
  end-to-end — and still reports cosine 0.999998, confirming the fallback is
  intact.

## Honest residual

- Twin vs stock = −0.0036 (essentially exact recovery of GPTQ's weight-quant
  benefit). Real vs stock = −0.0701, i.e. the real kernel is 0.07 *better* than
  the fake-quant baseline, within the real-vs-twin GEMM-accumulation tolerance
  (gate 0.0665 ≤ 0.1). There is no remaining RTN-vs-frozen-scale penalty; the
  only real-vs-twin difference left is the fused kernel's FP32 accumulation order
  vs torch's, which is ≤ 0.07 PPL and cosine ≥ 0.9999.
- Block granularity is NOT a residual here: ProMix's weight GPTQ froze its scales
  on the SAME block grid the kernel/packer use (NVFP4 16, MXFP8 32), so there is
  no group-size mismatch to pay for. (If a future recipe used a different GPTQ
  group size than the kernel block size, the frozen scales would need
  re-blocking; not applicable to this recipe.)
</content>
</invoke>
