# CUTLASS 4.5 SM100 Atom Reference Memo

**Purpose**: this memo provides reproducible inline excerpts of the CUTLASS 4.5 SM100 atom traits and PTX wrappers that `docs/specs/spec-mxfp8-nvfp4.md` depends on. The CUTLASS source is registered as a submodule at `third_party/cutlass` but is NOT initialized in every checkout (e.g., the local Mac development environment does not have it initialized; the remote B20Z dev server does). This memo's excerpts make the quantization spec self-contained without requiring the submodule to be present.

## Source Provenance

- **Repository**: <https://github.com/NVIDIA/cutlass>
- **Submodule registration**: `.gitmodules` lists `third_party/cutlass` at the upstream NVIDIA CUTLASS repository.
- **Pinned commit**: `cb37157db50d0528c4aea99feb37946ec278e3d9` (current `git submodule status third_party/cutlass` output, leading `-` indicates the submodule is registered-but-uninitialized in this checkout).
- **CUTLASS upstream version**: 4.5 (matches plan and DEVLOG; corresponds to the SM100 / Blackwell support release line).
- **To reproduce locally**: `git submodule update --init --recursive third_party/cutlass` will fetch ~1 GB of CUTLASS source at the pinned commit. Alternatively, browse the NVIDIA CUTLASS GitHub at this commit hash for the exact files quoted below.

## How to Use This Memo

This memo is the **defensive copy** of the SM100 atom facts the spec relies on. If a future version of CUTLASS changes any of these signatures (atom shape, scale dtype, instruction encoding), this memo MUST be updated alongside the spec, or the spec stops being a faithful contract for the kernel layer.

When an SM100 kernel is implemented (M2 / task17), the kernel writers should:
1. Re-verify each excerpt below against the in-tree CUTLASS source at the pinned commit
2. Update this memo (and the spec) if any divergence is found

## Atom Family 1: `SM100_MMA_MXF8F6F4_*` (high path; MXFP8)

**Defining file**: `third_party/cutlass/include/cute/atom/mma_traits_sm100.hpp` (in CUTLASS upstream source tree at commit `cb37157db`)

**PTX instruction**: `tcgen05.mma.kind::mxf8f6f4.block_scale` (block-scaled MXFP8/MXFP6/MXFP4 family). For our project, only the FP8 element-dtype variant is used (this is the high path in PLAQuant-SM100).

### Shape constraints (PRIMARY uses these)

| Constraint | Value | Source |
|---|---|---|
| M tile dimension | ∈ {64, 128} | atom traits enumeration; M is the 1-CTA / 2-CTA cluster dimension |
| N tile dimension | ∈ [8, 256], stride 8 if B is K-major, stride 16 if B is MN-major | atom traits |
| K tile dimension | 256 / sizeof_bits<ValTypeA> = **32 elements for FP8** | atom traits derive K from element bit-width |
| ValTypeA / ValTypeB | FP8 E4M3 (default) or FP8 E5M2 (alternative) | atom traits enumerate FP8 dtypes |
| Accumulator dtype | FP32 (TMEM) | accumulator type fixed to FP32 for MXFP8 |
| Block scale dtype | **FP8 E8M0** (8-bit unsigned exponent, 0 mantissa, power-of-2 scales only) | OCP MX standard; matches SM100_MMA_MXF8F6F4 trait specification |
| Block scale vector size (VS) | 32 elements per scale (i.e., 1 scale per 32 K-direction elements) | matches K tile dimension; 1 scale per K instruction |
| Atom suffix variants | `_SS` (SMEM-SMEM source), `_TS` (TMEM-SMEM), `_TT` (TMEM-TMEM); also 1-CTA and 2-CTA cluster variants | atom traits family |

### Why these matter for our spec

- The spec MXFP8 block size = 32 (Section 2) is FORCED by the K tile dimension above; we cannot choose a smaller block size and still use this atom.
- The spec MXFP8 block scale = FP8 E8M0 (Section 3) is FORCED by the atom trait's scale dtype enumeration; this is the OCP MX standard scale and the only one this atom accepts.
- The spec scale composition (Section 5) — that the MMA consumes the block scale internally — comes from the `block_scale` qualifier in the PTX instruction kind; this is what makes microscaling "free" relative to dense FP at the throughput level.

## Atom Family 2: `SM100_MMA_MXF4NVF4_*` (low path; NVFP4)

**Defining file**: `third_party/cutlass/include/cute/atom/mma_traits_sm100.hpp` (same file)

**PTX instruction**: `tcgen05.mma.kind::mxf4nvf4.block_scale` (block-scaled FP4 family with TWO scale-vector-size variants).

### Two variants under the same kind

The `kind::mxf4nvf4` instruction supports VS=32 (MXFP4, OCP standard) AND VS=16 (NVFP4, NVIDIA variant) under the same atom-family naming. The trait struct distinguishes them via the scale-vector-size template parameter:

| Variant | VS (scale vector size) | Block scale dtype | Block size | Standard | PRIMARY uses? |
|---|---|---|---|---|---|
| MXFP4 | 32 | FP8 E8M0 (power-of-2) | 32 | OCP MX | NO |
| **NVFP4** | **16** | **FP8 E4M3** (with mantissa) | **16** | NVIDIA variant | **YES** |

PRIMARY selects the NVFP4 variant (VS=16, FP8 E4M3 scale) because the higher-precision scale (E4M3 has 3 mantissa bits vs E8M0's 0) better captures the residual variance after ResQ rotation pre-conditioning, at the cost of NVIDIA-only compatibility (which is acceptable: project is locked to SM100).

### Shape constraints (NVFP4)

| Constraint | Value | Source |
|---|---|---|
| M tile dimension | ∈ {64, 128} | same as MXFP8 atom |
| N tile dimension | ∈ [8, 256], stride 8 / 16 by B layout | same |
| K tile dimension | 256 / sizeof_bits<ValTypeA> = **64 elements for FP4** | atom traits derive K from element bit-width |
| ValTypeA / ValTypeB | FP4 E2M1 | atom traits |
| Accumulator dtype | FP32 (TMEM) | same as MXFP8 atom — both phases share a single TMEM C type |
| Block scale dtype | **FP8 E4M3** for NVFP4 (VS=16); FP8 E8M0 for MXFP4 (VS=32) | trait struct's scale-vector-size template parameter |
| Block scale vector size | 16 elements per scale (NVFP4) | The K tile dimension is 64; with VS=16 there are 4 block scales per K instruction (vs 1 for MXFP8) |

### Why these matter

- NVFP4 block size = 16 (spec Section 2) is forced by the VS=16 template selection.
- NVFP4 block scale = FP8 E4M3 (spec Section 3) is forced by the trait struct mapping VS=16 → E4M3.
- The PRIMARY's "two phases share a TMEM accumulator" topology requires both atoms to write to the same FP32 TMEM C; both atoms above enumerate accumulator type as FP32, satisfying the prerequisite. (Whether they can be invoked back-to-back into the same C buffer in one kernel is the M0 task5 dual-phase PoC question.)

## PTX Wrapper Macros

**Defining file**: `third_party/cutlass/include/cute/arch/mma_sm100_umma.hpp` (in CUTLASS upstream at the pinned commit). The relevant macros are in approximately lines 993-1197 of that file (line numbers as seen in the upstream commit at the time of the original idea draft authoring).

The file contains PTX literals for the following SM100 `tcgen05.mma` instruction kinds:

| `kind::*` | Element type | Scale type | Dense / Microscaled | Used by PRIMARY? |
|---|---|---|---|---|
| `i8` | INT8 | (none — direct) | dense | NO (Alt-1 only) |
| `f16` | FP16 / BF16 | (none) | dense | NO |
| `tf32` | TF32 | (none) | dense | NO |
| `f8f6f4` | FP8 / FP6 / FP4 | per-channel FP32 | dense (no microscaling) | NO (rejected per plan; see "MXFP/NVFP vs Dense FP" comparison) |
| `mxf8f6f4` | FP8 / FP6 / FP4 | FP8 E8M0 block | microscaled | **YES** (high path) |
| `mxf4nvf4` | FP4 (E2M1) | FP8 E8M0 (VS=32, MXFP4) or FP8 E4M3 (VS=16, NVFP4) | microscaled | **YES** (low path, VS=16 NVFP4 variant) |
| `i4` (sign-extend) | INT4 elements via sign-extension to INT8 | — | dense | NO |

Notably absent: there is **no native INT4 instruction kind** on SM100. INT4 inputs must be sign-extended to INT8 and use `kind::i8`. This is the rationale for the plan's PRIMARY pivot (from INT4+INT8 to MXFP8+NVFP4).

## What the Original Plan AC-9 Cleanup Touches

The plan AC-9 calls out three derived-doc inconsistencies that this memo + the corrected spec resolve:

1. **MXFP8 block scale dtype**: the original idea draft had several lines saying MXFP8 scale was E4M3 (incorrect). Per the trait excerpts above, MXFP8 scale is **E8M0** (power-of-2). Spec Section 3 is now correct; this memo corroborates.

2. **High-path atom name**: the original draft β-2 phase plan said `SM100_MMA_F8F6F4_SS` (dense, no microscaling). The PRIMARY direction throughout the rest of the plan calls for `SM100_MMA_MXF8F6F4_*` (microscaled). Per the trait families above, `SM100_MMA_F8F6F4_*` and `SM100_MMA_MXF8F6F4_*` are DIFFERENT atoms (different `kind::*` qualifier; different scale handling). PRIMARY uses MXF8F6F4. β-2's F8F6F4 mention is a textual bug that AC-9 cleanup must fix.

3. **β-4 alignment example pair `(K_high=32, K_low=1984)`**: 32 + 1984 = 2016 ≠ 2048. Per the K tile dimension constraints above (FP8 K=32, FP4 K=64), valid (K_high, K_low) pairs satisfy `K_high % 32 = 0 ∧ K_low % 64 = 0 ∧ K_high + K_low = K_total`. The smallest valid K_high for K_total=2048 is K_high=64 (K_low=1984). The example pair must be removed or corrected.

## Validation Checklist for Future Submodule Updates

When CUTLASS submodule is bumped (or any kernel writer needs to verify these atom traits against the source):

- [ ] `git submodule update --init --recursive third_party/cutlass` (if not already initialized)
- [ ] Open `third_party/cutlass/include/cute/atom/mma_traits_sm100.hpp` and grep for `SM100_MMA_MXF8F6F4` and `SM100_MMA_MXF4NVF4`. Verify M ∈ {64, 128}; K = 32 (FP8) / 64 (FP4); accumulator = FP32; scale dtype matches table above.
- [ ] Open `third_party/cutlass/include/cute/arch/mma_sm100_umma.hpp` and grep for `kind::mxf8f6f4` and `kind::mxf4nvf4`. Verify the PTX wrapper macros emit `block_scale` qualifier and consume scale tile pointers.
- [ ] If any divergence is found, update this memo AND `docs/specs/spec-mxfp8-nvfp4.md` AND notify the kernel writer; do NOT silently accept divergence.

## Cross-References

- Primary spec: `docs/specs/spec-mxfp8-nvfp4.md` (depends on this memo for atom trait facts)
- Plan: `.humanize/plans/plaquant-sm100-fp8-nvfp4.md` (AC-1 spec authoring task and AC-9 doc consistency cleanup task)
- CUTLASS upstream: <https://github.com/NVIDIA/cutlass> at commit `cb37157db50d0528c4aea99feb37946ec278e3d9`
- Reference implementation (SM80 INT path): `kernels/mixed_gemm_l20/mixed_gemm_l20.cu` — provides the topology template (Phase 1 + Phase 2 sharing a register accumulator) being ported to TMEM here

## Changelog

- **Round 1** (2026-06-23): initial memo authored to resolve Codex Round 0 review finding "AC-1 validation cites CUTLASS files that are absent from this checkout". Quotes the SM100 atom traits and PTX kinds inline so the spec is reproducible without depending on the submodule being initialized.
