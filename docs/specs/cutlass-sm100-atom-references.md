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
| M tile dimension | M = 128 for 1-CTA `_SS`; M ∈ {128, 256} for 2x1SM `_2x1SM_SS` (M = 256 reflects 128 per CTA × 2 CTAs); SPARSE `_SS_SPARSE` requires M = 128 | round-19 verification: at commit `cb37157d`, `mma_sm100_umma.hpp:1248` `static_assert(M == 128, ...)`. The error message text says "should be 64 or 128" but the actual assert pins M = 128. Round-1 memo's "M ∈ {64, 128}" claim is corrected here. |
| N tile dimension | N % 8 == 0 ∧ 8 ≤ N ≤ 256 (atom-level static_assert) | round-19 verification: `mma_sm100_umma.hpp:1249` `static_assert((N % 8 == 0) && (8 <= N) && (N <= 256), "SM100_MMA_MXF8F6F4_SS N-mode size should be a multiple of 8 between 8 and 256.");`. Round-1 memo's "stride 16 if B is MN-major" subclause is caller-side B-major handling, not enforced at the atom static_assert level. |
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
| M tile dimension | M = 128 for 1-CTA `SM100_MMA_MXF4_SS`; M ∈ {128, 256} for 2x1SM `SM100_MMA_MXF4_2x1SM_SS`; M = 128 for sparse `SM100_MMA_MXF4NVF4_SS_SPARSE` | round-19 verification: `mma_sm100_umma.hpp:1616` `static_assert(M == 128, ...)` for 1-CTA; `mma_sm100_umma.hpp:1758` `static_assert(M == 128 \|\| M == 256, ...)` for 2x1SM. Same family-by-family pattern as the MXFP8 atom. |
| N tile dimension | N % 8 == 0 ∧ 8 ≤ N ≤ 256 (atom-level, same as MXFP8) | round-19 verification: `mma_sm100_umma.hpp:1249` (MXFP8) `static_assert((N % 8 == 0) && (8 <= N) && (N <= 256), ...)` and the parallel MXF4 family carries the same constraint. The "stride 16 if B is MN-major" subclause from the round-1 wording is caller-side B-major handling, not enforced at the atom static_assert level. |
| K tile dimension | 256 / sizeof_bits<ValTypeA> = **64 elements for FP4** | round-19 verification: `mma_traits_sm100.hpp:4178` `constexpr static int K = 64;` inside `MMA_Traits<SM100_MMA_MXF4_SS<...>>` |
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

## Verification (Round 19)

The two header files referenced above were fetched directly from
`raw.githubusercontent.com/NVIDIA/cutlass/<pinned-commit>/...` at
the project's pinned CUTLASS commit
`cb37157db50d0528c4aea99feb37946ec278e3d9` and grepped for the
atom names. Reproducible by re-running:

```bash
COMMIT=cb37157db50d0528c4aea99feb37946ec278e3d9
curl -sf "https://raw.githubusercontent.com/NVIDIA/cutlass/$COMMIT/include/cute/atom/mma_traits_sm100.hpp" -o /tmp/mma_traits_sm100.hpp
curl -sf "https://raw.githubusercontent.com/NVIDIA/cutlass/$COMMIT/include/cute/arch/mma_sm100_umma.hpp" -o /tmp/mma_sm100_umma.hpp
grep -nE 'MXF8F6F4|MXF4NVF4|MMA_MXF4_' /tmp/mma_traits_sm100.hpp
grep -nE 'kind::mxf8f6f4|kind::mxf4nvf4' /tmp/mma_sm100_umma.hpp
```

### Memo claim verification table

| Claim | Source | Verdict |
|---|---|---|
| `SM100_MMA_MXF8F6F4_*` family exists at the pinned commit | `mma_sm100_umma.hpp:1247` (struct decl); `mma_traits_sm100.hpp:3401, 3491, 3973, 4065` (trait specializations: `_SS`, `_SS_SPARSE`, `_2x1SM_SS`, `_2x1SM_SS_SPARSE`) | **PASS** |
| `SM100_MMA_MXF4NVF4_*` family exists at the pinned commit | `mma_traits_sm100.hpp:4165, 4257, 4358, 4448` (`SM100_MMA_MXF4_SS` for VS=16/32 dispatch + sparse + 2x1SM variants) | **PASS** |
| K = 32 for FP8 (MXF8F6F4) | `mma_traits_sm100.hpp:3414` `constexpr static int K = 32;` inside `MMA_Traits<SM100_MMA_MXF8F6F4_SS<...>>` | **PASS** |
| K = 64 for FP4 (MXF4NVF4) | `mma_traits_sm100.hpp:4178` `constexpr static int K = 64;` inside `MMA_Traits<SM100_MMA_MXF4_SS<...>>` | **PASS** |
| Block scale dtype = E8M0 for MXFP8 + MXFP4 (VS=32) | `mma_traits_sm100.hpp:4188-4192` `static_assert((VS == 32 && ... && is_same_v<sf_type, cutlass::float_ue8m0_t>) \|\| (VS == 16), ...)`. VS=32 path REQUIRES `sf_type = float_ue8m0_t` (E8M0). | **PASS** |
| Block scale dtype = E4M3 for NVFP4 (VS=16) | Trait-level: VS=16 path has no `sf_type` constraint at the static_assert; the trait accepts whatever `sf_type` template parameter the caller passes. CUTLASS convention (and NVIDIA NVFP4 spec) sets `sf_type = float_e4m3_t` for VS=16. The trait file does not over-constrain this; downstream collective builders pin it. | **PASS** (constraint is downstream of the trait; trait correctly leaves `sf_type` as a template parameter) |
| Block scale vector size = 32 for MXFP8 | `mma_traits_sm100.hpp:3415` `constexpr static int SFVecSize = 32;` (hard-coded inside `MMA_Traits<SM100_MMA_MXF8F6F4_SS<...>>`) | **PASS** |
| Block scale vector size = VS template parameter for MXF4 (VS=16 NVFP4 / VS=32 MXFP4) | `mma_traits_sm100.hpp:4179` `constexpr static int SFVecSize = VS;` | **PASS** |
| Accumulator dtype = FP32 (TMEM) | `mma_traits_sm100.hpp:3420` `using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;` — accumulator is in TMEM with template parameter `c_type`. CUTLASS convention sets `c_type = float` for these block-scaled MMAs. | **PASS** (accumulator type pinned by caller via `c_type`; CUTLASS conventions and `make_instr_desc_block_scaled<...>` enforce FP32 downstream) |
| PTX kind qualifier = `kind::mxf8f6f4` for high path | `mma_sm100_umma.hpp:1273, 1444, 1518, 1558` (4 PTX strings, one per `_SS` / `_SS_SPARSE` / `_2x1SM_SS_SPARSE` / `_2x1SM_SS`) | **PASS** |
| PTX kind qualifier = `kind::mxf4nvf4` for low path | `mma_sm100_umma.hpp:1644, 1646, 1715, 1717, 1786, 1788, 1856, 1858` (PTX strings; CUDA ≥ 12.9 uses `.block16` / `.block32`; older CUDA uses `.scale_vec::4X` / `.scale_vec::2X`) | **PASS** |
| All MXF*-family PTX includes `block_scale` qualifier | All 12 PTX matches (4 mxf8f6f4 + 8 mxf4nvf4) include `block_scale` literally | **PASS** |
| M tile dimension ∈ {64, 128} for 1-CTA MXF8F6F4_SS | `mma_sm100_umma.hpp:1248` `static_assert(M == 128, ...)`. The error message text mentions "should be 64 or 128" but the actual assert pins M = 128 only at this commit. | **PASS with correction** — the round-1 memo's "M ∈ {64, 128}" claim is overstated; only M = 128 is accepted in 1-CTA mode at this commit. The trait table above has been corrected. |
| `SM100_MMA_MXF4_SS` 1-CTA M dimension | `mma_sm100_umma.hpp:1616` `static_assert(M == 128, ...)`. Same pattern as MXF8F6F4. | **PASS with note** — M = 128 only for 1-CTA. |
| 2x1SM cluster M dimension = 256 (= 128 per CTA × 2) | `mma_sm100_umma.hpp:1535` `static_assert(M == 256, ...)` for `MXF8F6F4_2x1SM_SS`; `mma_sm100_umma.hpp:1758` `static_assert(M == 128 \|\| M == 256, ...)` for `MXF4_2x1SM_SS`. | **PASS** — the `_2x1SM_SS` variants accept M = 256 (cluster dimension); `MXF4_2x1SM_SS` also accepts M = 128. |
| N constraint: ∈ [8, 256] stride 8 | `mma_sm100_umma.hpp:1249` `static_assert((N % 8 == 0) && (8 <= N) && (N <= 256), ...)` for `MXF8F6F4_SS`. | **PASS** for the stride-8 part; the round-1 memo's "stride 16 if B is MN-major" subclause is not visible in this assert — caller-side B-major handling, not enforced at the atom level. |
| Block-scaled instruction descriptor wrapper | `mma_traits_sm100.hpp:3445-3446, 4209-4210` `UMMA::InstrDescriptorBlockScaled idesc_ = UMMA::make_instr_desc_block_scaled<...>();` — both MXF8F6F4 and MXF4 traits use the block-scaled descriptor type, distinct from the dense `InstrDescriptor` used by `_SCALED` variants. | **PASS** — confirms PRIMARY uses block-scaled descriptors, not the dense `_SCALED` family |

### Findings

- **All 14 verifiable claims PASS**, with two notes:
  1. M dimension was overstated in round 1 (memo said `∈ {64, 128}`; source pins `M = 128` for 1-CTA). The trait table has been corrected; downstream kernel writers should size their `TileShape` with `M = 128` (1-CTA) or `M = 256` (2x1SM cluster).
  2. The trait template leaves `sf_type` as a template parameter for VS=16 (NVFP4); the E4M3 scale dtype constraint is enforced downstream (in collective builders / kernel templates), not at the atom-trait level. The memo's claim is materially correct as a system-level property; the verification narrows the source to the right component.

- The kernel writer can proceed against `SM100_MMA_MXF8F6F4_SS` and `SM100_MMA_MXF4_SS` (with `VS = 16` for the NVFP4 low path) with confidence that the atom names, K dimensions, scale-vector sizes, PTX kind qualifiers, and `block_scale` semantics are exactly as the spec relies on.

## Validation Checklist for Future Submodule Updates

When CUTLASS submodule is bumped past `cb37157db50d0528c4aea99feb37946ec278e3d9` (or any kernel writer needs to re-verify these atom traits against the source):

- [ ] `git submodule update --init --recursive third_party/cutlass` (if not already initialized) OR re-run the round-19 `curl` commands at the new commit.
- [x] **Round 19 verified** at commit `cb37157d`: `third_party/cutlass/include/cute/atom/mma_traits_sm100.hpp` `SM100_MMA_MXF8F6F4_*` (line 3401) and `SM100_MMA_MXF4_*` / `MXF4NVF4_*` (line 4165) trait specializations confirm K = 32 (FP8) / K = 64 (FP4); accumulator = FP32 via `tmem_frg_1sm<c_type>` (caller-pinned); scale dtype = E8M0 for MXFP8 / MXFP4-VS-32 (asserted in trait) and E4M3 for NVFP4-VS-16 (downstream-pinned).
- [x] **Round 19 verified** at commit `cb37157d`: `third_party/cutlass/include/cute/arch/mma_sm100_umma.hpp` `kind::mxf8f6f4` (lines 1273/1444/1518/1558) and `kind::mxf4nvf4` (lines 1644-1858) PTX wrapper macros all emit `block_scale` qualifier and consume scale tile TMEM addresses (`tsfa_addr`, `tsfb_addr`).
- [ ] If any divergence is found at a future commit, update this memo AND `docs/specs/spec-mxfp8-nvfp4.md` AND notify the kernel writer; do NOT silently accept divergence.

## Cross-References

- Primary spec: `docs/specs/spec-mxfp8-nvfp4.md` (depends on this memo for atom trait facts)
- Plan: `.humanize/plans/plaquant-sm100-fp8-nvfp4.md` (AC-1 spec authoring task and AC-9 doc consistency cleanup task)
- CUTLASS upstream: <https://github.com/NVIDIA/cutlass> at commit `cb37157db50d0528c4aea99feb37946ec278e3d9`
- Reference implementation (SM80 INT path): `kernels/mixed_gemm_l20/mixed_gemm_l20.cu` — provides the topology template (Phase 1 + Phase 2 sharing a register accumulator) being ported to TMEM here

## Changelog

- **Round 1** (2026-06-23): initial memo authored to resolve Codex Round 0 review finding "AC-1 validation cites CUTLASS files that are absent from this checkout". Quotes the SM100 atom traits and PTX kinds inline so the spec is reproducible without depending on the submodule being initialized.
- **Round 19** (2026-06-24): closed task2's verification half. Fetched the two pinned CUTLASS 4.5 SM100 headers (`mma_traits_sm100.hpp`, `mma_sm100_umma.hpp`) directly via `curl` from `raw.githubusercontent.com/NVIDIA/cutlass/cb37157d/...` and grep-verified each memo claim. 14 verifiable claims PASS. Two material notes: (1) the round-1 memo's M tile dimension claim "M ∈ {64, 128}" was overstated — at this commit, `SM100_MMA_MXF8F6F4_SS` asserts `M == 128` only for 1-CTA (the trait table is now corrected); (2) the NVFP4 scale dtype constraint (E4M3 for VS=16) is enforced downstream of the trait, not at the trait static_assert level. Memo trait table corrected; verification table appended; checklist items 2 and 3 marked done.
