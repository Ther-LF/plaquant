# MXFP8 + NVFP4 Quantization Format Specification

**Status**: Authored Round 0 of RLCR loop (2026-06-23) per plan AC-1.
**Source of Truth**: this document is the single source of truth for MXFP8 and NVFP4 numerical behavior. All derived implementations MUST match this spec bit-for-bit:
- Fake quantizer (`promix/quantize/quant_utils.py`)
- Real activation packer (`promix/inference/quant_ops.py`)
- Real weight packer (`promix/inference/weight_packer.py`)
- Kernel epilogue (`kernels/mixed_gemm_sm100/`)
- Reference Python implementation (used for the AC-1 fake-vs-real equivalence test)

If any of these disagree with each other, this spec is what they all reconcile to.

---

## 1. Element Format

### MXFP8 (high path)

- **Element dtype**: FP8 E4M3 (default) — 1 sign bit, 4 exponent bits, 3 mantissa bits, bias = 7
  - Representable range: ±[2^-9 (subnormal min) , 448] (E4M3 has no Inf; max value is finite 448)
  - 256 representable values total (signed)
- **Alternative**: FP8 E5M2 (1+5+2, bias=15) — wider range up to ±57344 but 4× coarser mantissa. Available as a config knob for early outlier-prone activations; default is E4M3.

### NVFP4 (low path)

- **Element dtype**: FP4 E2M1 — 1 sign bit, 2 exponent bits, 1 mantissa bit, bias = 1
  - 16 representable values total (signed): {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
  - No NaN / Inf encodings in E2M1

---

## 2. Block Size (number of K-direction elements sharing one block scale)

| Format | Block size | Source |
|---|---|---|
| MXFP8 | **32** elements per block | OCP MX standard; matches `tcgen05.mma.kind::mxf8f6f4` instruction K extent |
| NVFP4 | **16** elements per block | NVIDIA NVFP4 variant of `kind::mxf4nvf4` (the VS=16 variant; not the OCP MXFP4 VS=32 variant) |

Block boundary alignment: K_high MUST be a multiple of 32; K_low MUST be a multiple of 16 (and per AC-3 also satisfy the NVFP4 instruction K extent of 64 elements, so K_low MUST be a multiple of 64 in practice).

---

## 3. Block Scale Dtype

| Format | Block scale dtype | Encoding |
|---|---|---|
| **MXFP8** | **FP8 E8M0** | 8-bit unsigned exponent only (no mantissa, no sign). Each E8M0 value represents `2^(exp - 127)`. Power-of-2 scales only. OCP MX standard. |
| **NVFP4** | **FP8 E4M3** | Standard FP8 with 4-bit exponent + 3-bit mantissa (bias=7). NOT power-of-2; ~3-bit mantissa precision in the scale. NVIDIA NVFP4 variant. |

**This is one of the points that the original idea draft had mistyped** (some lines said MXFP8 scale was E4M3 — that is wrong; MXFP8 = E8M0, NVFP4 = E4M3). All derived code uses the values in this table.

---

## 4. Scale Axis

Scales are computed along the **K (contraction) axis** for both A and B matrices, since the MMA reduces along K.

- **A scale tensor shape**: `(M, K_total / block_size)` — each row of A has `K_total / block_size` block scales
- **B scale tensor shape**: `(N, K_total / block_size)` — each row of B (B is N-major in the standard CUTLASS layout) has `K_total / block_size` block scales
- **Stride**: scale tensors are stored as separate buffers from element data (NOT interleaved); strides are row-major in the (per-row, per-block) order. TMA descriptors load A_scale and B_scale as their own tiles.

(Layout is finalized in Section 9; this section only specifies the logical shape and axis.)

---

## 5. Scale Composition (which scales the MMA consumes vs which the epilogue handles)

This is the most-easily-mishandled section: there are several scales floating around the kernel, and the epilogue MUST NOT double-apply scales the MMA already consumed.

### What `tcgen05.mma.kind::mxf8f6f4` consumes (high path, MXFP8)

- A_high (M × K_high) packed FP8 elements
- B_high (N × K_high) packed FP8 elements
- A_high_block_scale (M × K_high/32) FP8 E8M0 scales
- B_high_block_scale (N × K_high/32) FP8 E8M0 scales

The instruction outputs an FP32 partial inner product **with both A and B block scales already multiplied in**:
```
acc_high[m, n] += sum_k( A_high[m, k] * A_block_scale[m, k/32] ) * ( B_high[n, k] * B_block_scale[n, k/32] )
```
Result: TMEM accumulator C contains the de-block-scaled FP32 partial sum, up to FP8 element rounding error.

### What `tcgen05.mma.kind::mxf4nvf4` consumes (low path, NVFP4)

Same pattern, but FP4 E2M1 elements with FP8 E4M3 block scales over 16-element blocks:
```
acc_low[m, n] += sum_k( A_low[m, k] * A_block_scale[m, k/16] ) * ( B_low[n, k] * B_block_scale[n, k/16] )
```

### What the epilogue handles (everything else)

After both phases accumulate into the same TMEM C, the epilogue applies the remaining scales:

```
out[m, n] = (acc_high[m, n] + acc_low[m, n])  // FP32, both phases summed
          * s_x_token[m]                       // per-token activation scale (FP16 or FP32)
          * s_w_channel[n]                     // per-channel weight scale (FP16 or FP32)
          * g_w                                // optional NVFP4 global FP32 scale (per tensor; see Section 10)
```

Then `out` is cast to FP16 and stored to gmem.

**Critical**: the block scales (E8M0 for MXFP8, E4M3 for NVFP4) are **already inside** `acc_high` and `acc_low`. The epilogue MUST NOT multiply them again — that would double-count and corrupt the output.

The fake-vs-real equivalence test (Section 12) explicitly checks this composition.

---

## 6. Rounding Mode

### Element quantization (FP16/FP32 input → MXFP8 / NVFP4 element)

- **Round-to-nearest-even** (RNE; ties-to-even). This is the IEEE 754 default.
- Applied per-element after dividing by the chosen block scale: `q = round_RNE(x / block_scale * 2^bias_offset)`.

### Block scale computation

For a block of `block_size` consecutive K-direction elements:
1. Compute `block_max = max_abs(block)` (largest absolute value in the block)
2. Compute the "ideal" scale: `ideal_scale = block_max / max_format_value` where `max_format_value` is the largest representable element value (MXFP8 E4M3: 448; NVFP4 E2M1: 6)
3. **Round the scale UP to the nearest representable scale value** (i.e., toward larger magnitude; smallest representable scale ≥ ideal_scale):
   - MXFP8 E8M0 scale: `chosen_scale = 2^ceil(log2(ideal_scale))` — the smallest power-of-2 ≥ ideal_scale.
   - NVFP4 E4M3 scale: `chosen_scale = smallest_representable_FP8_E4M3_above_or_equal(ideal_scale)` — the smallest E4M3 value ≥ ideal_scale.
4. Edge case: if `block_max == 0`, set `chosen_scale = smallest_positive_representable_scale` (E8M0: 2^-127; E4M3: 2^-9 subnormal-min). All-zero quantized block.

The "round scale UP" rule guarantees `block_max / chosen_scale ≤ max_format_value`, so elements never saturate after the divide-and-round step. The minor tradeoff is a slightly conservative scale (worst case ~1× the ideal), costing under 1 effective mantissa bit on average.

> **Reconciliation note (Round 1)**: the original plan AC-1 said "rounded down to the nearest representable scale value", which conflicted with both the math expressed here (ceil / above-or-equal) and with the safety property the rule is meant to enforce. The plan AC-1 line and this spec section have been reconciled to "round UP" as the canonical rule. Rounding DOWN would push some elements above `max_format_value` after division, forcing saturation and silently corrupting the quantized representation; that would defeat the whole point of a block scale.

---

## 7. Saturation Behavior

- During element quantization: out-of-range inputs **clamp** to ±max_format_value (E4M3: ±448; E2M1: ±6). No overflow to Inf. (E2M1 has no Inf encoding anyway.)
- The "round scale up" rule (Section 6) makes saturation rare but not impossible (e.g., if the post-block-scale value lands exactly at the boundary).
- Reason for clamp instead of overflow: NVFP4 has no Inf encoding, and in MXFP8 E4M3 there is no Inf either (max is 448 finite). Clamp is the only well-defined behavior on this hardware.

---

## 8. NaN / Inf Handling

- **Input NaN**: propagates as NaN in the FP8 element if the format encodes it; E4M3 has a single NaN encoding. NVFP4 E2M1 does NOT encode NaN — input NaN at quantization time is **undefined behavior** and a calibration check rejects NaN inputs (raises a clear error before quantization runs).
- **Input Inf**: similarly, NVFP4 cannot represent Inf; calibration check rejects Inf input. MXFP8 E4M3 does not represent Inf either; Inf input is rejected.
- **Block scale NaN/Inf**: never produced by the scale computation in Section 6 (since input is rejected before reaching the scale computation).

The calibration check is part of the fake quantizer (`promix/quantize/quant_utils.py`); it runs before the first quantization step in PTQ Step 1 (rotation optimization) and before each forward pass in real-FP path.

---

## 9. Packing Layout (memory layout for both data and scale tensors)

### Element data

Standard contiguous packing along K, then padding to MMA atom boundary:
- A_high: (M, K_high) row-major; each row aligned to multiple of 32 elements (MXFP8 instruction K extent)
- B_high: (N, K_high) row-major; same alignment
- A_low: (M, K_low) row-major; each row aligned to multiple of 64 elements (NVFP4 instruction K extent)
- B_low: (N, K_low) row-major; same alignment

FP8 packing: 1 byte per element (8 bits per E4M3 element).
FP4 packing: 2 elements per byte (4 bits per E2M1 element); low-nibble is the lower-K element.

### Block scale tensors

Stored as **separate buffers** from element data (NOT interleaved):
- A_high_block_scale: shape `(M, K_high / 32)` of FP8 E8M0 (1 byte each), row-major
- B_high_block_scale: shape `(N, K_high / 32)` of FP8 E8M0
- A_low_block_scale: shape `(M, K_low / 16)` of FP8 E4M3
- B_low_block_scale: shape `(N, K_low / 16)` of FP8 E4M3

Rationale for separate buffers (not interleaved):
- Cleaner TMA descriptor: one descriptor per tensor, no per-element stride math for the block scale tile
- Lower pack/unpack overhead in fake quantizer (PyTorch can compute block scales as a separate tensor without strided gathers)
- Matches NVIDIA's NVFP4 reference implementation in Transformer Engine

Alignment: scale tensor row strides should be multiples of 16 bytes for TMA boundary alignment; padding rows (zero-fill) is acceptable.

### Storage size estimate

For Llama-3.2-1B q_proj at high_fraction=1/8 (K_total=2048, K_high=256, K_low=1792, N=2048):
- A_high: M × 256 bytes
- B_high: 2048 × 256 = 0.5 MB (constant per layer)
- A_high_block_scale: M × 8 bytes (256/32)
- B_high_block_scale: 2048 × 8 = 16 KB
- A_low: M × 896 bytes (1792/2)
- B_low: 2048 × 896 = 1.75 MB
- A_low_block_scale: M × 112 bytes (1792/16)
- B_low_block_scale: 2048 × 112 = 224 KB

For 1B model total weight storage (dominated by B_high + B_low across all layers): ~150-200 MB. This compares against ~1 GB FP16 baseline — about 5-7× weight compression, minus a few % for block scales.

---

## 10. NVFP4 Global FP32 Scale Policy

**Decision (per plan DEC-2)**: **NO global FP32 scale by default**.

Rationale:
- ResQ rotation pre-conditioning (PCA + Stiefel R + down_proj Hadamard) flattens outliers across blocks; a per-tensor global scale would mostly capture redundant magnitude information.
- Simpler packer (no global scalar to track) and simpler epilogue (one fewer multiply).
- Matches plan's "Goal" framing (FP path's dequant formula is `out = s_x_token · s_w_channel · acc`, no extra terms).

**Override path**: if AC-2 (PPL non-regression) fails on any model, revisit by enabling global FP32 scale per Transformer Engine convention. To enable:
1. Compute per-tensor `g_w = max_abs(B_dequantized) / 6.0` (NVFP4 max element value) at PTQ packing time and store as one FP32 scalar per weight tensor
2. In epilogue: `out *= g_w`
3. Re-run AC-2.2 / AC-2.3 PPL eval to compare with-vs-without-global-scale

If override is enabled later, this section will be updated to RESOLVED-WITH-GLOBAL-SCALE and the spec re-derived implementations will track.

---

## 11. GPTQ Interaction (Hessian-based weight rounding)

GPTQ rounds each weight to the nearest representable value AFTER conditioning on already-quantized columns via the Hessian-inverse update. With block scales, "representable" means:
- For a given column k in a block: `representable_set = {q * block_scale | q in {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}}` for NVFP4; analogous set with E4M3 representable values for MXFP8.
- GPTQ rounds **per element to the representable_set under the CURRENT block_scale** (block scale is computed once per block from the unrounded weights, then held fixed during the GPTQ rounding pass within that block).

This means GPTQ does NOT re-search the block scale jointly with elements. The scale is determined by the un-GPTQ-corrected block_max (Section 6); GPTQ then rounds elements within that scale. This is a deliberate simplification — joint search would require iterating block scale + element rounding to convergence, which is expensive and not needed for this paper's claim.

The fake-vs-real equivalence test (Section 12) verifies that fake quantizer + GPTQ produces the same packed bytes as the real packer.

---

## 12. Fake-vs-Real Equivalence Test

The single regression test that all derived implementations must pass:

```python
# tests/test_quantize_format_equivalence.py (to be implemented in Round 1+)

def test_mxfp8_equivalence():
    # Random input shape (1024, 256) — 1024 rows × 256 cols (one MXFP8 block per row)
    x_fp16 = torch.randn(1024, 256, dtype=torch.float16)
    
    # Path 1: fake quantizer (PyTorch reference)
    x_q_fake, scale_fake = fake_mxfp8_quantize(x_fp16)  # promix/quantize/quant_utils.py
    
    # Path 2: real packer (CUDA / pybind binding)
    x_q_real, scale_real = real_mxfp8_pack(x_fp16)  # promix/inference/quant_ops.py
    
    # Path 3: spec-derived Python reference (this spec, Section 6)
    x_q_ref, scale_ref = spec_reference_mxfp8(x_fp16)  # implemented from Section 6
    
    # All three must produce bit-identical outputs
    assert torch.equal(x_q_fake, x_q_real), "fake != real"
    assert torch.equal(x_q_fake, x_q_ref), "fake != spec reference"
    assert torch.equal(scale_fake, scale_real), "scale fake != real"
    assert torch.equal(scale_fake, scale_ref), "scale fake != spec reference"

def test_nvfp4_equivalence():
    # Same shape with NVFP4 (16-element blocks, FP8 E4M3 scale)
    # ... analogous
```

If any pair disagrees, the disagreement source must be identified before any kernel work proceeds. Most likely failure modes:
- Rounding mode mismatch (RNE vs round-to-zero)
- Block scale rounding direction mismatch (round up vs round to nearest)
- Saturation policy mismatch (clamp vs overflow)
- Endianness / packing-order mismatch in FP4 (low nibble vs high nibble first)

---

## Summary Table (one-page reference)

| Property | MXFP8 | NVFP4 |
|---|---|---|
| Element dtype | FP8 E4M3 (default) / E5M2 (alt) | FP4 E2M1 |
| Element max value | 448 (E4M3) / 57344 (E5M2) | 6 |
| Block size (K-direction) | 32 | 16 |
| Block scale dtype | FP8 E8M0 (power-of-2) | FP8 E4M3 (with mantissa) |
| Scale rounding direction | round UP (toward larger magnitude; smallest representable ≥ ideal) | round UP (toward larger magnitude; smallest representable ≥ ideal) |
| Element rounding | RNE (ties-to-even) | RNE (ties-to-even) |
| Saturation | clamp ±448 (E4M3) | clamp ±6 |
| NaN/Inf | E4M3 has NaN; rejected at calibration | no NaN/Inf in E2M1; rejected at calibration |
| Packing layout | data + scale separate buffers | data + scale separate buffers |
| Global FP32 scale | N/A | NO by default (DEC-2); override if AC-2 fails |
| Bytes per element (data) | 1 | 0.5 (2 per byte) |
| Bytes per scale (per block) | 1 | 1 |
| MMA instruction | `tcgen05.mma.kind::mxf8f6f4` | `tcgen05.mma.kind::mxf4nvf4` (VS=16) |
| Atom (CUTLASS 4.5) | `SM100_MMA_MXF8F6F4_*` | `SM100_MMA_MXF4NVF4_*` |
| Effective bits/element | 8.25 | 4.5 |

---

## Cross-References

- Plan: `.humanize/plans/plaquant-sm100-fp8-nvfp4.md` — AC-1 positive/negative tests, broader context
- **CUTLASS atom reference memo**: `docs/specs/cutlass-sm100-atom-references.md` — inline excerpts of the SM100 atom traits this spec depends on. The CUTLASS submodule (`third_party/cutlass`, pinned commit `cb37157db`) is not initialized in every checkout, so the memo serves as the reproducible defensive copy. Run `git submodule update --init --recursive third_party/cutlass` to fetch the full source if in-tree verification is needed.
- CUTLASS upstream (canonical source): <https://github.com/NVIDIA/cutlass> at commit `cb37157db50d0528c4aea99feb37946ec278e3d9`. Files of interest: `include/cute/atom/mma_traits_sm100.hpp` (atom traits) and `include/cute/arch/mma_sm100_umma.hpp` (PTX wrapper macros).
- Reference INT path: `kernels/mixed_gemm_l20/mixed_gemm_l20.cu` — the SM80 PLAQuant implementation that this spec's MXFP8/NVFP4 path replaces
- ResQ algorithm: `project-resq/` (submodule) — the upstream rotation method whose noise model R must be retrained against (per plan AC-2)

---

## Changelog

- **Round 0** (2026-06-23): initial spec authored from plan AC-1 positive tests; covers all 11 numerical dimensions plus equivalence test design. NVFP4 global FP32 scale defaulted to NO per DEC-2 with documented override path.
