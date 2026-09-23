# In-Register Fused FP8 NVFP4 GEMM For SM120 Blackwell

## Original Idea

Port PLAQuant single-kernel mixed-precision fused GEMM from SM80 (INT4+INT8) to SM120 (RTX Pro 6000 Blackwell). SM120 uses mma.sync.aligned (not tcgen05/TMEM) but natively supports MXFP8 block-scaled (m16n8k32, E4M3, UE8M0 scale per 32 K) and NVFP4 block-scaled (m16n8k64, E2M1, UE4M3 scale per 16 K). Both atoms share identical FP32[4] accumulator registers with SM80_16x8_Row CLayout — enabling direct in-register accumulation across phases (same pattern as existing SM80 fused kernel). Key differences from the original SM100 plan: no TMEM, no WGMMA, 99KB SMEM (not 228KB), no cluster multicast, GDDR7 1.8TB/s (not HBM3e 8TB/s). The algorithmic layer (ResQ PCA + variance split + Stiefel rotation) is unchanged in structure but requires re-running the full pipeline (Step 0/1/2) because: (1) o_proj PCA is upgraded to full global hidden_dim, (2) quantization noise model changes from INT to FP. CUTLASS already provides SM120 CollectiveBuilder support (examples 79/87). Hardware: 8x RTX Pro 6000 Blackwell Server Edition (SM120, 96GB GDDR7 each, CUDA 13.0). Target: single-kernel FP8+FP4 fused GEMM beating both FP16 cuBLAS and 2-launch FP baseline on LLM shapes (Llama 1B/3B/8B).

## Algorithm: ProMix/PLAQuant (ResQ-FP variant)

### Overview

ProMix uses PCA to find high-variance channels in LLM activations, routes them to higher precision (FP8), and routes the remaining low-variance channels to lower precision (FP4). A learnable orthogonal rotation further smooths the distribution within each sub-space. The full pipeline produces quantized weights + rotation matrices that are fused into the model at PTQ time (zero runtime overhead for rotation).

### Three-Step Pipeline

**Step 0: PCA Basis Computation** (`promix/quantize/basis.py`, ~20 min/model)
- Collects activation covariance matrices from calibration data (512 wikitext samples)
- Eigendecomposes to find variance-sorted basis vectors U
- Produces: U file (PCA basis), E file (eigenvalues), initial random R file
- For o_proj: collects full hidden_dim × hidden_dim covariance (our improvement over ResQ's per-head 64×64)
- Format-agnostic: same U works for INT or FP quantization (the basis captures data statistics, not quantizer properties)
- BUT: o_proj uses `o_proj_pca: full_global` (our change) → **must re-run Step 0** when switching from per-head to global

**Step 1: Rotation R Optimization** (`promix/quantize/optimize_rotation.py`, ~34 min/1B)
- Loads model with trainable orthogonal R matrices (R1 for hidden_dim, R2 for head_dim)
- Forward pass injects quantization noise via ActQuantWrapper
- SGDG optimizer (Stiefel manifold SGD via Cayley transform) updates R to minimize loss
- **Critical**: the noise model MUST match the final quantizer — if we use FP8/FP4 in inference, Step 1 must train with FP8/FP4 fake quant noise (not INT noise)
- **Old R is invalid**: the previous R was trained against INT4/INT8 noise + per-head U for o_proj; switching to FP8/FP4 noise + global U means R must be retrained
- For 1B model: rotation optimization shows no PPL improvement (random orthogonal R already optimal); 3B/8B may benefit

**Step 2: PTQ Evaluation** (`promix/eval/ptq.py`, ~3 min/model)
- Loads model → fuses U into weights → fuses R into weights → adds ActQuantWrapper → evaluates PPL
- After fusion, U and R are absorbed into Linear weights (zero runtime cost)
- ActQuantWrapper splits activation by variance, applies fake quantization, concatenates back, does matmul

### Algorithm Changes from ResQ

| Aspect | ResQ (original) | ProMix/PLAQuant (ours) |
|--------|-----------------|------------------------|
| o_proj PCA | per-head 64×64 block-diagonal | **full global hidden_dim (e.g., 2048×2048)** |
| o_proj input rotation | only U (no R) | **U_oproj_g + trainable R_oproj** (closes implementation gap) |
| o_proj quantization | per-group scale (groupsize=64) | **same MXFP8+NVFP4 block-scale as all other layers** |
| K/V cache quantization | INT4 (W4A4KV4) | **NVFP4 (W4A4KV4-FP)** — ResQ has INT KV4 baseline to compare against |
| Quantization format | INT4 (low) + INT8 (high) | **NVFP4 E2M1 (low) + MXFP8 E4M3 (high)** |
| Block scaling | per-channel/per-token only | **per-block (32 for FP8, 16 for FP4) + per-token/per-channel** |
| high_fraction | Fixed 1/8, shared U+R across layers | **Phase 1: full_shared + fixed 1/8 (ResQ baseline); Phase 2: per_layer U+R + adaptive high_fraction (core contribution)** |
| rotation_granularity | full_shared (1 global U + 1 global R1) | **Phase 1: full_shared; Phase 2: per_layer (per-layer U_i + per-layer R1_i)** |
| down_proj | INT: no split (single precision); online Hadamard | **FP4-only (no split) — Hadamard already flattens outliers** |
| Kernel | 2-launch INT or SM80 fused INT | **SM120 single-kernel fused FP** |

### Key Algorithm Design Decisions

**1. o_proj gets trainable R_oproj (NEW vs current code)**
- Current code: o_proj input only gets U_oproj_g (PCA), no R rotation — an implementation gap
- Our change: train an independent R_oproj in U_oproj_g's [low|main|high] sub-space (block-diagonal, same pattern as R1)
- Step 1 trains R1 (hidden_dim, for q/k/v/gate/up) + R2 (per-head, for V) + **R_oproj (hidden_dim, for o_proj input)**
- Step 2 fuses: `W_oproj_new = R1.T @ W_oproj @ U_oproj_g @ R_oproj`
- Rationale: PCA sorts variance globally, but sub-space still has residual outliers; R further smooths them (same logic as R1 for other layers)

**2. down_proj: FP4-only (no high/low split)**
- down_proj input is intermediate_size (8192 for 1B) — too large for full PCA, uses online Hadamard instead
- Hadamard rotation already flattens outliers to near-uniform distribution → no high-variance channels to protect
- Therefore: down_proj entire activation goes NVFP4 (no MXFP8 high path)
- Simplifies kernel (down_proj uses NVFP4-only path) and matches the "Hadamard + FP4" natural pairing
- Note: this differs from q/k/v/gate/up/o_proj which DO have high/low split (they use PCA, not Hadamard)

**3. KV4 quantization is in-scope (not future work)**
- ResQ paper provides W4A4KV4 INT baselines: 1B=11.96, 3B=8.70, 8B=7.04
- Our W4A4KV4-FP must not regress vs these baselines
- K/V use per-head PCA (H_value, H_key_pos) — NOT global, because K/V are used per-head in attention (cross-head split would break Q@K^T correspondence and RoPE)
- K/V quantized to NVFP4 with per-head groupsize=head_dim (preserves per-head blocking)
- FP KV4 configs already exist: `promix/configs/llama-3.2-{1b,3b}-mxfp8-nvfp4-kv4.yaml`

**4. Adaptive high_fraction is a core contribution (not future work)**
- **Phase 1**: `rotation_granularity="full_shared"` + fixed 1/8 (matches ResQ baseline, establishes correctness; reuses existing Step 1 code which only supports full_shared)
- **Phase 2**: switch to `rotation_granularity="per_layer"` — each layer has its own U (Step 0 already supports this) and its own R1 (Step 1 needs modification: remove `assert rotation_granularity=="full_shared"`, mount R1 per-layer as `model.layers[i].R1_1` instead of global `model.R1_1`)
- **Per-layer adaptive high_fraction**: each layer's K_high_i derived from that layer's eigenvalue cumulative distribution (e.g., channels capturing 90% cumulative variance). No R1 block-structure contradiction because R1_i's block-diagonal is sized to that layer's high_dim_i.
- Constraints per layer: K_high_i % 32 = 0 (MXFP8 block), K_low_i % 64 = 0 (NVFP4 atom)
- Kernel already parameterized by (K_high, K_low) per call → no kernel change for per-layer variable ratio
- Acceptance: per-layer variable ratio does not regress PPL vs fixed 1/8, ideally improves (< -0.2 PPL); average bit-width does not increase (reallocate high quota from insensitive layers to sensitive layers)

### Why 1/8 as Starting Point (high_fraction = 0.125)

- ResQ paper uses 1/8 and provides published PPL baselines we can compare against (W4A4: 1B=14.72, 3B=10.21, 8B=6.72; W4A4KV4: 1B=11.96, 3B=8.70, 8B=7.04)
- Alignment: K_high = K_total × 0.125 satisfies MXFP8 K%32=0 for all Llama hidden_dims (2048→256, 3072→384, 4096→512, 8192→1024, 14336→1792)
- NVFP4 alignment: K_low = K_total × 0.875 satisfies K%64=0 for all Llama dims (2048→1792, 3072→2688, 4096→3584, 8192→7168, 14336→12544)
- 1/8 is the **Phase 1 baseline**; Phase 2 makes it adaptive per-layer

### Existing Implementation Status (what's already done)

| Component | Status | Notes |
|-----------|--------|-------|
| `fake_quantize_mxfp8()` | ✅ Implemented (quant_utils.py:660-707) | Follows spec Section 6 |
| `fake_quantize_nvfp4()` | ✅ Implemented (quant_utils.py:710-751) | Follows spec Section 6 |
| `ActQuantizer.forward` FP dispatch | ✅ Implemented (quant_utils.py:182-224) | Routes string bits to FP fake quant |
| `configure_quantizers` FP support | ✅ Implemented | Accepts high_bits="mxfp8", low_bits="nvfp4" |
| `optimize_rotation.py` FP noise hook | ✅ Implemented (line 240-250) | Passes FP format strings to configure_quantizers |
| `basis.py` o_proj_global mode | ✅ Implemented (line 106, 302-403) | Collects H_oproj and stores as o_proj_global key |
| FP configs (1B/3B) | ✅ Exist | `promix/configs/llama-3.2-{1b,3b}-mxfp8-nvfp4.yaml` |
| Format spec | ✅ Complete | `docs/specs/spec-mxfp8-nvfp4.md` (297 lines + SM120 addendum) |
| SM80 fused kernel | ✅ Working | 1.12-1.19x speedup, architecture template |
| **Reference oracle for bit-match** | ❌ Not done | Needed for AC-3 |
| **Step 0/1/2 actually run with FP** | ❌ Not done | U/R files not generated with FP configs yet |
| **SM120 kernel** | ❌ Not done | The main deliverable |
| **End-to-end inference with SM120** | ❌ Not done | Depends on kernel |

## Primary Direction: SM120 Hand-Written CuTe Fused Kernel

### Rationale

The SM80 fused kernel (`kernels/mixed_gemm_l20/mixed_gemm_l20.cu`) is CUTLASS 2.x (`cutlass::gemm::device::Gemm`, `int32_t` accumulator) where dual-phase works by instantiating two separate `Mma` objects sharing one `FragmentC`. SM120 block-scaled FP MMA is only reachable via CUTLASS 3.x CuTe `CollectiveBuilder` (monolithic mainloop) — the SM80 2.x pattern does NOT mechanically port. This is **new kernel authoring**, not a port: hand-forking a SM120 CuTe mainloop that issues both `mxf8f6f4...m16n8k32` and `mxf4nvf4...m16n8k64` into the same register tensor. The SM80 kernel serves as an architectural *inspiration* (dual-phase + SMEM union pattern), not a code template.

### Approach Summary

Author a new fused SM120 kernel (`kernels/mixed_gemm_sm120/`) using CUTLASS 3.x CuTe:
- **Dual-phase in-register shared accumulator**: hand-written mainloop issuing NVFP4 MMA (Phase 1, K_low) then MXFP8 MMA (Phase 2, K_high) into the same FP32[4] register fragment (both atoms share `SM80_16x8_Row` CLayout — confirmed in `mma_traits_sm120.hpp`)
- **SMEM union reuse**: Phase 2 overwrites Phase 1's SMEM buffers (pattern from SM80 kernel, feasible within 99KB SMEM)
- **FP4 <<2 shift** after `ld.matrix b4x16_p64` (SM120-specific bit placement)
- **Single epilogue**: `out = acc * s_x_token * s_w_channel → FP16` (block scales consumed inside MMA)

Architecture decision (resolved): **in-register shared accumulator** (not EVT workspace-add). Rationale: matches draft's topology claim; GDDR7's low bandwidth (1.8TB/s) maximizes the value of avoiding the extra (M,N) buffer write that EVT-add requires. The v5 EVT-workspace pattern is a documented fallback if in-register proves infeasible.

### Two Parallel Workstreams

1. **Algorithm validation** (can start immediately, no SM120 hardware dependency for fake quant):
   - Run Step 0 with o_proj_pca=full_global → new U
   - Run Step 1 with FP noise model (high_bits="mxfp8", low_bits="nvfp4") → new R
   - Run Step 2 fake-FP PPL eval on 1B/3B/8B
   - Gate: PPL non-regression vs INT baseline

2. **Kernel development** (requires SM120 hardware):
   - Verify SM120 atom compilation
   - Build dual-phase shared-accumulator PoC
   - Full-tiled single-phase MXFP8 + NVFP4 GEMMs
   - 2-launch baseline
   - Fused kernel + performance characterization

These two streams converge at end-to-end integration (real kernel + real quantized weights → PPL + latency).

### Kernel Data Flow (Single Launch)

```
Single kernel launch:
┌─ TMA/cp.async load A_low (NVFP4 packed) → SMEM_A_low
├─ TMA/cp.async load B_low (NVFP4 packed) → SMEM_B_low
├─ TMA/cp.async load SF_A_low, SF_B_low   → SMEM (E4M3 scales)
│
├─ Phase 1 Mainloop (K_low / 64 iterations):
│   ├─ SMEM → RF: ld.matrix b4x16_p64 (FP4 in low bits)
│   ├─ FP4 shift left 2 bits (SM120-specific: place data in MMA-expected position)
│   └─ mma.sync.aligned.kind::mxf4nvf4.block_scale.m16n8k64
│       → acc_fp32[0..3] += A_nvfp4 @ B_nvfp4 (block-scale applied internally by MMA)
│
├─ [SMEM reuse via union — overwrite low buffers]
│
├─ TMA/cp.async load A_high (MXFP8) → SMEM_A_high  
├─ TMA/cp.async load B_high (MXFP8) → SMEM_B_high
├─ TMA/cp.async load SF_A_high, SF_B_high → SMEM (E8M0 scales)
│
├─ Phase 2 Mainloop (K_high / 32 iterations):
│   ├─ SMEM → RF: ld.matrix (FP8, no shift needed)
│   └─ mma.sync.aligned.kind::mxf8f6f4.block_scale.m16n8k32
│       → acc_fp32[0..3] += A_mxfp8 @ B_mxfp8 (SAME accumulator registers!)
│
└─ Epilogue:
    out[m,n] = acc_fp32[m,n] * s_x_token[m] * s_w_channel[n]
    cast to FP16, store to gmem
```

### Objective Evidence

- `kernels/mixed_gemm_l20/mixed_gemm_l20.cu:177-181` — SMEM union pattern (LOW/HIGH share via union)
- `kernels/mixed_gemm_l20/mixed_gemm_l20.cu:204-252` — Dual-phase in-register accumulation pattern
- `kernels/mixed_gemm_l20/mixed_gemm_l20.cu:47-50, 69-72` — INT atom shapes match SM120 FP atom shapes
- `kernels/mixed_gemm/mixed_gemm.cu:114-136` — SM90 CollectiveBuilder pattern (reusable API)
- `docs/specs/spec-mxfp8-nvfp4.md` — Complete format spec (Sections 1-14, including SM120 addendum)
- `promix/quantize/quant_utils.py:660-751` — FP fake quantizers (already implemented)
- `promix/quantize/optimize_rotation.py:240-250` — FP noise model hook (already implemented)
- `promix/quantize/basis.py:302-403` — o_proj global PCA (already implemented)
- `docs/mixed-gemm-optimization-ideas.md:56-86` — Cross-phase prefetch analysis (8-13% gain)
- CUTLASS upstream `examples/79_blackwell_geforce_gemm/` — SM120 NVFP4 GEMM reference
- CUTLASS upstream `examples/87_blackwell_geforce_gemm_blockwise/` — SM120 FP8 blockwise GEMM reference
- CUTLASS `include/cute/atom/mma_traits_sm120.hpp` — SM120 atom CLayout = SM80_16x8_Row (confirmed)
- CUTLASS `include/cute/arch/mma_sm120.hpp` — PTX instruction wrappers with register definitions

### Known Risks

- CUTLASS SM120 CollectiveBuilder may not support dual-phase topology natively — may need manual MMA + custom mainloop
- Block scale operands add to register pressure (SM80 kernel uses 90-96 registers)
- SM120 has only 99KB SMEM — tile size limited vs SM100's 228KB
- FP4 left-shift (<<2) after ld.matrix is SM120-specific — missing it produces wrong output
- Shared GPU environment (GPU 0-5 occupied by vLLM) — thermal/power throttling risk during benchmarks
- 2-launch FP may already be fast enough on SM120 that single-kernel fusion shows marginal speedup (same concern as SM100 plan)
- R_oproj adds trainable parameter to Step 1 — increases training cost and complexity vs current code that has no o_proj R

### Optimization Phases (all in-scope ACs, NOT future work)

**Phase 1 — Correctness baseline (fixed 1/8)**:
- Validate W4A4-FP PPL vs ResQ INT W4A4 baseline (1B=14.72, 3B=10.21, 8B=6.72)
- Validate W4A4KV4-FP PPL vs ResQ INT W4A4KV4 baseline (1B=11.96, 3B=8.70, 8B=7.04)
- Validate fused kernel correctness + beats 2-launch per-shape
- Kernel uses initial 64×64 tile

**Phase 2 — Algorithm-side ACs**:
- Adaptive per-layer high_fraction (eigenvalue cumulative threshold or grid search); constraints K_high%32=0, K_low%64=0
- R_oproj training (closes o_proj rotation gap)
- GPTQ weight rounding for FP formats (adapt Hessian-based rounding to FP representable set)

**Phase 3 — Kernel-side ACs**:
- Tile shape exploration: 64×64 (initial), 64×128, 128×64, 128×128 — different shapes optimal for different M regimes
- Stage count tuning: 3/4/5 pipeline stages for latency hiding
- Cross-phase prefetch: overlap Phase 2 data loading with Phase 1 tail compute (8-13% estimated gain)
- Mixed precision pipeline interleaving: instead of sequential (all FP4 → all FP8), interleave K-tiles for better ILP
- Per-shape config table: different GEMM shapes may need different tile/stage configs

### Clean Slate Policy (no artifact reuse)

**All prior RLCR artifacts are deleted and must be regenerated from scratch in the new SM120 environment:**
- Format spec `docs/specs/spec-mxfp8-nvfp4.md` — REWRITE (was SM100-flavored)
- CUTLASS atom memo — REGENERATE (was SM100 atoms; need SM120 atoms from `mma_traits_sm120.hpp`)
- All U/R/E bin files — RECOMPUTE (Step 0/1 must run fresh on SM120 with FP configs)
- Existing fake quantizer code — REVALIDATE against new reference oracle (don't assume correctness carries over)
- This avoids stale SM100 assumptions contaminating the SM120 implementation

## Alternative Directions Considered

### Alt-1: Bandwidth-Optimized Tile Strategy
- Gist: Focus on GDDR7 1.8TB/s constraint; design tiles/stages for SM120 memory hierarchy. 128×128+5 stages = 48KB SMEM.
- Objective Evidence: `kernels/mixed_gemm_l20/` tile configs, `docs/mixed-gemm-optimization-ideas.md:56-86`
- Why not primary: Speculative without hardware measurements; direct port first, tile-tune after.

### Alt-2: CuTe DSL Python Kernel
- Gist: Use CUTLASS Python CuTe DSL for rapid prototyping.
- Objective Evidence: exploratory, no concrete precedent in this repo.
- Why not primary: No Python kernel precedent; doubles effort if C++ conversion needed.

### Alt-3: Algorithm-Kernel Co-Design
- Gist: Co-optimize ResQ pipeline for FP: adaptive high_fraction, FP-aware rotation, block-scale alignment.
- Objective Evidence: `promix/quantize/quant_utils.py` FP infrastructure, `optimize_rotation.py` FP hooks, `basis.py` eigenvalues
- Why not primary: Complementary work; kernel is critical path. Algorithm validation happens in parallel.

### Alt-4: End-to-End Inference Integration
- Gist: Full pipeline: weight packing + online activation quant + kernel dispatch + epilogue dequant on `promix/inference/`.
- Objective Evidence: `promix/inference/` infrastructure (~536 LOC), format spec, ptq.py FP branch
- Why not primary: Depends on working kernel; integration is M3 work.

## Hardware Environment

- **Server**: `luofan@10.208.130.90` — 8x RTX Pro 6000 Blackwell Server Edition
- **Architecture**: SM120, compute capability 12.0
- **Memory**: 96 GB GDDR7 per GPU, ~1.8 TB/s bandwidth
- **CUDA**: 13.0 (nvcc V13.0.88)
- **Driver**: 580.159.03
- **Development GPUs**: 6 and 7 (GPU 0-5 occupied by vLLM)
- **PyTorch**: needs installation (not present; requires torch 2.7+ with cu128 or cu129)
- **CUTLASS**: submodule registered at `third_party/cutlass`, needs `git submodule update --init`
- **SM clock**: max 2430 MHz (attempt lock for benchmarks)

## Synthesis Notes

The project has two independent critical paths that converge: (1) algorithm validation (Step 0/1/2 with FP — proves PPL doesn't regress), and (2) kernel development (SM120 fused GEMM — proves performance wins). These can execute in parallel since fake-quant PPL eval doesn't need the SM120 kernel, and the kernel doesn't need validated PPL numbers. They converge at end-to-end integration where real weights + real kernel produce both PPL and latency numbers. Future optimizations (adaptive ratio, tile tuning, pipeline interleaving) are independent additive contributions that build on the validated baseline.
