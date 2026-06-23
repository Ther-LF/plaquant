# PLAQuant-SM100: Single-Kernel FP8+NVFP4 Mixed-Precision via tcgen05.mma

## Goal Description

Port the PLAQuant single-kernel multi-precision GEMM topology from SM80 (INT4+INT8) to SM100 native (MXFP8 high path + NVFP4 low path) on B20Z Blackwell, while keeping the ResQ algorithm (PCA basis + variance-based channel split + learnable Stiefel rotation R) as the source of the mixed-precision data layout.

The end result must satisfy three hard criteria across **1B / 3B / 8B Llama models** and **two quantization configurations (W4A4 and W4A4KV4)**, measured at **two layers (kernel-level micro benchmarks AND end-to-end real inference)**:

1. **Precision non-regression**: wikitext PPL ≤ current INT4+INT8 W4A4 baseline (quant-vs-quant, not vs FP16)
2. **Faster than FP16 cuBLAS**: end-to-end and kernel-shape-sweep latency > torch.matmul FP16 (5th-gen Tensor Core cuBLAS) on the LLM-realistic shape subset
3. **Faster than 2-launch FP**: single-kernel fused beats sequential MXFP8 GEMM + NVFP4 GEMM in BOTH timing modes — eager (no CUDA Graph; the deployment-realistic case where launch overhead reduction is part of fused's benefit) AND CUDA Graph (the paper-defensible topology-only case where Python launch overhead is OUT of the timing window). Both measured separately. (the PLAQuant claim's hard evidence)

Both kernel-level and end-to-end measurements are required. Only end-to-end without micro hides where speedup comes from; only micro without end-to-end hides whether Python-side quantization pack/dequant overhead consumes the kernel's gains.

The algorithmic side reuses ResQ (PCA basis with promoted **global** PCA on o_proj input, learnable R), with the quantizer changed from INT to FP. R must be retrained because the noise model changed.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- **AC-1: Quantization Format Specification (Single Source of Truth)**
  - A formal spec document exists covering MXFP8 and NVFP4 in full numerical detail, before any kernel or quantizer is written. The fake quantizer, real packer, and kernel must all be derived from this single document.
  - Positive Tests:
    - **Element format**: MXFP8 = E4M3 (default, dynamic-range-trading mantissa) or E5M2 (alternative, range-trading mantissa); NVFP4 = E2M1 (16 representable values).
    - **Block size**: MXFP8 = 32 along K, NVFP4 = 16 along K.
    - **Block scale dtype**: MXFP8 block scale = **FP8 E8M0** (8-bit power-of-2 exponent, 0 mantissa bits — NVIDIA Hopper/Blackwell + OCP MX standard). NVFP4 block scale = **FP8 E4M3** (4-bit exponent + 3-bit mantissa — NVIDIA NVFP4 variant).
    - **Scale axis**: along K (contraction dimension); A scales indexed by (M, K/block); B scales indexed by (N, K/block); explicit shape and stride documented.
    - **Scale composition**: which scales are consumed by the MMA instruction (block scales applied internally by `tcgen05.mma.kind::mxf8f6f4` and `kind::mxf4nvf4`) and which remain for the epilogue (per-token activation scale `s_x` + per-channel weight scale `s_w` + optional NVFP4 global FP32 scale `g_w`). The epilogue formula MUST NOT double-apply block scales already consumed by MMA.
    - **Rounding mode**: round-to-nearest-even (ties-to-even); applied at element quantization step. Block scale itself is computed by `block_scale = max_abs(block) / max_format_value`, then **rounded UP** (toward larger magnitude; smallest representable scale ≥ the ideal value) to the nearest representable scale value (E8M0 = power-of-2 via `ceil(log2(...))`; E4M3 = smallest representable FP8 ≥ ideal). Round-UP is required so that elements never overflow `max_format_value` after dividing by the chosen scale; round-DOWN would silently saturate outlier elements. (Round 1 correction: an earlier version of this line said "rounded down" — that was a bug.)
    - **Saturation behavior**: out-of-range values clamp to ±max_format_value (no overflow to Inf during quantization).
    - **NaN/Inf handling**: input NaN propagates as NaN in the FP8/FP4 element if the format encodes it; otherwise documented as undefined behavior with a calibration check rejecting NaN inputs.
    - **Packing layout**: scale tensor stored as a separate buffer (NOT interleaved with element data) — explicit shape/stride documented; choice rationale captured (typically separate buffer enables cleaner TMA descriptor + lower pack/unpack overhead).
    - **NVFP4 global FP32 scale policy**: explicit YES/NO decision with rationale (DEC-2 outcome). If YES, spec includes the per-tensor FP32 scalar position in epilogue; if NO, spec captures the calibration test that demonstrates ResQ-rotated activations don't need it.
    - **GPTQ interaction**: GPTQ Hessian-based weight rounding interacts with block scales — spec must define whether GPTQ rounds within a block (per-element representable) or across blocks (per-block representable), and how this composes with block-scale selection.
    - **Fake-vs-real equivalence test**: spec includes a unit test that compares fake quantizer output against a Python reference implementation derived from the spec; output must match bit-for-bit.
  - Negative Tests:
    - Spec missing any of the eleven dimensions above (element format, block size, scale dtype, scale axis, scale composition, rounding mode, saturation, NaN/Inf, packing layout, NVFP4 global scale, GPTQ interaction) is rejected
    - Two derived implementations (fake quantizer and real packer) producing different bit-patterns for the same input is rejected
    - Epilogue formula that double-applies block scales already consumed by `tcgen05.mma` is rejected

- **AC-2: Algorithm-Level PPL Non-Regression on 1B/3B/8B**
  - All three model sizes pass for both W4A4 and W4A4KV4 configurations. Failure on any one combination = AC fails.
  - Reference values (current ResQ INT4+INT8 baselines, locked at this plan's commit time):
    - W4A4 (KV16): 1B=11.70, 3B=8.61, 8B=6.99
    - W4A4KV4: 1B=11.96, 3B=8.70, 8B=7.04
  - Tolerance: float noise allows ±0.01; PPL > baseline + 0.05 is regression.
  - AC-2.1: INT-path global PCA smoke test (precondition for AC-2.2)
    - Three explicit bands resolve all PPL outcomes:
      - **Pass band** (≤ 11.70 + 0.10): smoke test passes, proceed to AC-2.2
      - **Investigate band** (11.70 + 0.10 to 11.70 + 0.30): proceed BUT run additional diagnostics — layer-wise output cosine vs per-head PCA, per-channel variance scatter plot, U×R unitarity check; if diagnostics show clean implementation, accept and proceed; if diagnostics flag an anomaly, fix before proceeding
      - **Stop-and-debug band** (> 11.70 + 0.30): STOP, treat as implementation bug first (the math says global PCA is a strict superset of per-head; > +0.30 regression cannot come from the design). If multiple debug rounds fail to identify the bug, trigger Alt-4 fallback (per-head PCA + cross-head rearrange)
    - Positive: PPL ≤ 11.70 + 0.10 (Pass band)
    - Negative: PPL > 11.70 + 0.30 with no implementation bug found = design failure → Alt-4 fallback
  - AC-2.2: Fake-FP W4A4 PPL on 1B/3B/8B
    - Positive: 1B ≤ 11.70 + 0.05, 3B ≤ 8.61 + 0.05, 8B ≤ 6.99 + 0.05
    - Negative: any model > baseline + 0.05 = fail
  - AC-2.3: Fake-FP W4A4KV4 PPL on 1B/3B/8B
    - Positive: 1B ≤ 11.96 + 0.05, 3B ≤ 8.70 + 0.05, 8B ≤ 7.04 + 0.05
    - Negative: any model > baseline + 0.05 = fail
  - AC-2.4: Layer-wise output cosine sanity (o_proj microscaled vs INT per-group)
    - Positive: layer-wise output cosine ≥ 0.99 across all attention layers on 1B
    - Negative: any layer cosine < 0.99 triggers o_proj fallback path investigation (Alt-4)

- **AC-3: Kernel Numerical Equivalence Across Shape Sweep**
  - The new fused SM100 kernel and its 2-launch FP baseline must be numerically equivalent (NOT bit-exact — quantization is lossy; we measure cosine + relative + absolute error against an FP32 dual-phase reference) across the entire shape sweep.
  - Shape sweep matrix derived from **actual Llama 1B/3B/8B layer dimensions** (not synthetic round numbers):
    - **M (active token count)** ∈ {1, 16, 64, 128, 256, 512, 1024, 2048, 4096}; M ∈ {1, 16} is decode regime (typically batch×1), M ∈ {64...} is prefill regime (batch×seq_len). MMA atom requires M ∈ {64, 128} so M ∈ {1, 16} requires padding/grouped-GEMM strategy (see AC-4 decode regime classification).
    - **(K_total, N) pairs** = union over models × layer types:
      - 1B: q/o (2048, 2048); k/v (2048, 512); gate/up (2048, 8192); down (8192, 2048)
      - 3B: q/o (3072, 3072); k/v (3072, 1024); gate/up (3072, 8192); down (8192, 3072)
      - 8B: q/o (4096, 4096); k/v (4096, 1024); gate/up (4096, 14336); down (14336, 4096)
    - **(K_high, K_low) splits** subject to (K_high % 32 == 0) ∧ (K_low % 64 == 0): for each K_total, evaluate the 1/8 default (e.g., 2048 → 256/1792, 3072 → 384/2688, 4096 → 512/3584, 8192 → 1024/7168, 14336 → 1792/12544) plus 1/4 and 1/2 ratios (β-4 search space). The (1/8, 1/4, 1/2) sweep characterizes both PRIMARY default and Alt-4 fallback options.
  - Positive Tests:
    - For every (M, K_total, N, K_high, K_low) tuple in the sweep above, fused-kernel output cosine vs FP32 reference ≥ 0.9999; max abs error ≤ 1e-2; mean abs error ≤ 1e-3; **mean relative error ≤ 1e-3** (added per Codex round-2 optional improvement; absolute thresholds alone don't catch scale-dependent error patterns)
    - Same correctness checks pass for the 2-launch FP baseline (taken under the same harness)
  - Negative Tests:
    - Any sweep tuple with cosine < 0.9999 = fail
    - Any tuple where fused and 2-launch FP outputs differ by more than reference noise (cosine < 0.99999 between fused and 2-launch) = fail (would mean fused implements different math than baseline)

- **AC-4: Kernel Performance vs FP16 cuBLAS (Shape-Sweep Characterization)**
  - Required on the LLM-hit shape subset (the (M, K_total, N, K_high, K_low) tuples enumerated in AC-3). Other sweep tuples may be slower but must be documented.
  - **M is active token count, NOT batch size**. M = batch × tokens_per_request_in_step. **Batched decode is a normal regime**: at batch=64 single-step decode, M = 64 × 1 = 64, which sits exactly at the MMA atom minimum M ∈ {64, 128}; at batch=128 decode, M=128 also fits naturally. The only true corner case is **batch=1 single-user decode**, where M=1 doesn't fit the MMA atom.
  - **Threshold hardness (per user decision)**: speedup > 1.0× is the **HARD bar** (cannot be slower than FP16); 1.5× / 2.0× targets are **directional** (recorded in RESULTS.md as goals; not failing acceptance if missed by an epsilon).
  - Positive Tests:
    - **Standard regime (M ≥ 64, covers all batched decode + all prefill)**: every LLM-hit shape fused-vs-FP16 > 1.0× — this is the HARD bar
    - **Directional targets** (recorded but not hard-failing): at M=128 LLM-hit geomean fused-vs-FP16 ≥ 1.5×; at M ≥ 2048 LLM-hit geomean ≥ 2.0×
    - **Single-user decode (M=1, batch=1)**: results documented per shape; if fused < 1.0×, report explicitly classifies "single-user decode edge case: needs M-padding to 64 / GEMV kernel" — see DEC-3
    - `kernels/mixed_gemm_sm100/RESULTS.md` records all sweep speedups + heatmap + per-regime classification
  - Negative Tests:
    - Any standard-regime (M ≥ 64) LLM-hit shape below 1.0× without a documented attribution (e.g., specific tile-shape / N=512 alignment / register pressure cause) = fail
    - Missing characterization of any LLM-hit shape regime = fail
    - For M=1 (batch=1 single-user decode): fused may be < 1.0× but the report MUST explicitly classify and name a mitigation candidate — silent failure (no classification, no mitigation) is rejected
    - Conflating batch with M in the report = fail (must use active-token-count M for the GEMM shape, with separate prose linking M to batch × seq_len contexts)

- **AC-5: Kernel Performance vs 2-Launch FP Baselines (PLAQuant Claim)**
  - This is the most-load-bearing perf criterion: it directly measures the value of single-kernel fusion over sequential 2-launch.
  - **Two baseline variants are measured separately, each answering a different question** (per user decision):
    - **Variant E — Eager 2-launch** (the deployment-realistic comparator): two block-scaled GEMM calls invoked from Python in eager mode (no CUDA Graph), with the strongest feasible accumulation path (priority below). Python launch overhead IS in the timing window. **This is what most PyTorch users / vLLM with dynamic batching actually see.** Reducing launch overhead is a legitimate PLAQuant advantage and beating Variant E proves the deployment-level benefit.
    - **Variant G — CUDA Graph 2-launch** (the topology-only comparator): same two GEMMs but launched as a single pre-recorded CUDA Graph; Python launch overhead is OUT of the timing window. **This is the paper-defensible "GPU-side topology benefit only" baseline.** Beating Variant G proves fused has TMEM accumulator-sharing benefit beyond just launch-overhead reduction.
  - Both variants use the same accumulation path between phase 1 (MXFP8) and phase 2 (NVFP4), in this priority order:
    1. **Preferred**: GEMM2 with `beta=1` semantics directly accumulating into GEMM1's FP32 output (CUTLASS `epilogue_op` with `Beta != 0`); no third add kernel
    2. **Acceptable**: GEMM1 produces FP32 C; GEMM2 fused-epilogue add into the same C buffer (avoids a separate add kernel via epilogue fusion)
    3. **Last resort, must be documented**: separate FP32 add kernel as a third launch (only if neither path 1 nor path 2 is supported by the available CUTLASS atoms — explicit memo required)
    Variant E and Variant G use the SAME accumulation path; the only difference is whether the launches are wrapped in a CUDA Graph.
  - **Threshold hardness (per user decision)**: > 1.0× is the **HARD bar** for BOTH variants (fused must beat both Eager and CUDA Graph 2-launch; the PLAQuant claim is the project's core contribution); 1.05× geomean against Variant G is **directional** (recorded as goal; not auto-failing if missed by epsilon, since Variant G is the harder comparator).
  - Positive Tests:
    - **Every** standard-regime (M ≥ 64) LLM-hit shape: fused/Variant-E > 1.0× single-shape — HARD bar (deployment-level claim)
    - **Every** standard-regime (M ≥ 64) LLM-hit shape: fused/Variant-G > 1.0× single-shape — HARD bar (paper claim of TMEM topology benefit)
    - **Directional target**: LLM-hit shape subset (M ≥ 64) geomean fused/Variant-E ≥ 1.10× and fused/Variant-G ≥ 1.05× (recorded but not auto-failing)
    - `RESULTS.md` records per-shape speedups for BOTH variants separately, with heatmaps, and the chosen accumulation path (1/2/3 above)
  - Negative Tests:
    - Any standard-regime (M ≥ 64) LLM-hit shape with fused/Variant-E ≤ 1.0× = fail (deployment-level PLAQuant claim doesn't hold for that shape; triggers tile-shape / SMEM tuning iteration)
    - Any standard-regime (M ≥ 64) LLM-hit shape with fused/Variant-G ≤ 1.0× = fail (paper-level PLAQuant claim of TMEM topology benefit doesn't hold for that shape; same triggering)
    - Geomean against Variant-G < 1.0× after tuning iterations = PLAQuant single-kernel topology claim cannot stand; trigger Alt-1 (path α) fallback per user decision. Note: Variant-E might still be winnable via launch-overhead reduction even if Variant-G is lost, but that is a weak fallback claim.
    - Reporting only one variant (collapsing both into a single "vs 2-launch" number) = fail; the two variants answer different questions and must be reported separately
    - Single-user decode (M=1) not subject to the > 1.0× bar (M=1 doesn't fit MMA atom anyway); per-shape numbers still reported for both variants

- **AC-6: End-to-End PPL Non-Regression on Real-FP Path**
  - Real-FP path (i.e., kernel-actual-execution) must reproduce fake-FP PPL within float noise. **Both W4A4 and W4A4KV4 configurations are in scope.**
  - AC-6.1: **W4A4 real-FP PPL** on 1B/3B/8B within ±0.05 of corresponding fake-FP PPL from AC-2.2
  - AC-6.2: **W4A4KV4 real-FP PPL** on 1B/3B/8B within ±0.05 of corresponding fake-FP PPL from AC-2.3 (subject to DEC-4 confirming KV4 is in this phase)
  - Positive Tests:
    - 1B/3B/8B real-FP W4A4 PPL within ±0.05 of fake-FP W4A4 PPL
    - 1B/3B/8B real-FP W4A4KV4 PPL within ±0.05 of fake-FP W4A4KV4 PPL (if DEC-4 = include)
  - Negative Tests:
    - Any model × config combination with real-vs-fake mismatch > 0.10 = real path has correctness bug (kernel implements different math than fake quantizer), STOP and debug

- **AC-7: End-to-End Latency on Real-FP Path**
  - All three model sizes × both quantization configurations (W4A4 / W4A4KV4) must satisfy both performance bars at the end-to-end level.
  - Activation quantization (per-token MXFP8 / NVFP4 quant + block scale compute) IS included in the latency window; weight packing is OFFLINE (one-time PTQ artifact, NOT in the latency window).
  - **Workload regimes** (Codex round-2 required): both prefill (long context) and decode (single-step generation) are measured separately, NOT collapsed into a single "end-to-end" number:
    - Prefill regime: seq_len ∈ {512, 2048}, batch=1, measured as time-to-first-token
    - Batched decode regime (the typical serving scenario): batch ∈ {64, 128}, single-step generation latency, M = batch fits MMA atom
    - Single-user decode (edge case): batch=1, single-step generation, M=1; reported but not subject to hard > 1.0× bar (M=1 doesn't fit MMA atom — see DEC-3)
  - **Both quantization configurations are in scope (per user decision DEC-4)**: W4A4 and W4A4KV4 both required.
  - AC-7.1: **W4A4 latency** on 1B/3B/8B across all three regimes
  - AC-7.2: **W4A4KV4 latency** on 1B/3B/8B across all three regimes
  - **2-launch FP comparator at end-to-end follows AC-5's two variants**: Eager (real-FP forward path running fused kernel vs an eager-mode forward path running 2-launch — the deployment-realistic comparison) AND CUDA Graph (forward pass wrapped in a CUDA Graph for both fused-path and 2-launch-path — the topology-only comparison).
  - Positive Tests:
    - **Prefill regime (HARD)**: 1B/3B/8B × {W4A4, W4A4KV4} end-to-end latency vs FP16 cuBLAS > 1.0×; vs 2-launch FP **Eager** > 1.0×; vs 2-launch FP **CUDA Graph** > 1.0×. Directional targets ≥ 1.5× / ≥ 1.10× / ≥ 1.05× geomean respectively recorded in RESULTS.md.
    - **Batched decode regime (HARD)**: 1B/3B/8B × {W4A4, W4A4KV4} × batch ∈ {64, 128} end-to-end latency vs FP16 cuBLAS > 1.0×; vs both 2-launch FP variants > 1.0×. Same MMA atom constraints as prefill so kernel reuse is direct.
    - **Single-user decode (M=1)**: 1B/3B/8B × {W4A4, W4A4KV4} latency vs FP16 cuBLAS reported per (model, config); HARD bar > 1.0× IF a mitigation (M-padding / GEMV) is implemented; else documented underperformance per DEC-3
    - Per-stage timing breakdown reported: kernel time, activation quant time, dequant/epilogue time, non-GEMM model time (e.g., attention softmax, RMSNorm, embedding)
  - Negative Tests:
    - Any (model, config, regime ∈ {prefill, batched-decode}) combination below 1.0× vs FP16 = fail
    - Any (model, config, regime ∈ {prefill, batched-decode}) combination at-or-below 1.0× vs **either** 2-launch FP variant = fail (the PLAQuant claim must hold end-to-end at both deployment-level AND topology-level)
    - Missing per-stage breakdown when end-to-end fails (un-attributable failure) = fail
    - Collapsed regimes (failing to separate prefill / batched-decode / single-user-decode) = fail
    - Reporting only one quantization config (W4A4 only without W4A4KV4 numbers) = fail per DEC-4
    - Reporting only one of the two 2-launch FP variants (collapsing Eager and CUDA Graph into one number) = fail; both variants must be reported separately

- **AC-8: Reproducibility Artifacts**
  - All measurements must be reproducible from scratch given the artifacts.
  - Positive Tests:
    - `kernels/mixed_gemm_sm100/RESULTS.md` includes: raw CSV of all sweep numbers, env versions (CUDA, CUTLASS, PyTorch), commit hash, GPU SKU, clock/power mode, warmup/iter counts, CUDA Graph use, profiler command lines
    - All generated rotation artifacts (U-*.bin, E-*.bin, R-*.bin) include their source commit + dataset + nsamples in metadata
  - Negative Tests:
    - Numbers reported without the corresponding raw CSV = fail
    - Rotation artifacts without traceable provenance = fail

- **AC-9: Draft Documentation Consistency Cleanup**
  - The original draft has three confirmed inconsistencies that must be reconciled in derived docs.
  - Positive Tests:
    - In all derived documents (configs, code comments, RESULTS.md), MXFP8 block scale is consistently described as **FP8 E8M0** (not E4M3 — that is NVFP4's scale dtype)
    - In all derived documents, the high-path atom is consistently `SM100_MMA_MXF8F6F4_*` (microscaled), not `SM100_MMA_F8F6F4_*` (dense)
    - The β-4 alignment example pair `(K_high=32, K_low=1984)` is removed or corrected — 32 + 1984 = 2016 ≠ 2048; valid pairs at the smallest K_high satisfy K_high % 32 = 0 AND (2048 - K_high) % 64 = 0, the smallest of which is K_high=64 (K_low=1984)
  - Negative Tests:
    - Any new spec/code/doc still containing the inconsistent statements = fail

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The implementation includes all of: (a) formal MXFP8/NVFP4 quantization spec document; (b) `promix/quantize/basis.py` extended with `o_proj_pca: full_global` mode; (c) MXFP8 + NVFP4 fake quantizers in `promix/quantize/quant_utils.py`; (d) FP-aware rotation training in `promix/quantize/optimize_rotation.py`; (e) FP-aware GPTQ in `promix/quantize/gptq.py`; (f) new `kernels/mixed_gemm_sm100/` module with the fused single-kernel implementation AND TWO 2-launch FP baseline variants (eager + CUDA Graph); (g) shape-sweep `benchmark.py` with four-column output (fused / eager-2-launch / CUDA-Graph-2-launch / FP16) plus per-stage profile; (h) FP packing path in `promix/inference/weight_packer.py` and FP activation quantization in `promix/inference/quant_ops.py`; (i) FP forward path in `promix/inference/real_forward.py` with EVT epilogue fusing dequant; (j) full set of W4A4/W4A4KV4 configs for 1B/3B/8B; (k) reproducibility artifacts (RESULTS.md, raw CSVs, profiler outputs); (l) β-4 per-operator variable high/low ratio extension as an algorithm-side independent contribution.

### Lower Bound (Minimum Acceptable Scope)
The implementation includes: (a) formal MXFP8/NVFP4 spec; (b) basis.py global PCA mode; (c) fake quantizers and end-to-end fake-FP PPL on 1B at pass; (d) one fused single-kernel implementation with shape-sweep correctness on at least the LLM-hit subset; (e) BOTH 2-launch FP baselines (eager + CUDA Graph), timed under the same harness; (f) shape-sweep micro benchmark report; (g) real-FP forward path on 1B with end-to-end PPL and latency vs FP16 and BOTH 2-launch variants; (h) extension to 3B/8B end-to-end (mandatory per user's coverage requirement); (i) RESULTS.md reproducibility artifacts. β-4 is OUT of lower bound (it is an extension, not a core deliverable).

### Allowed Choices
- Can use:
  - CUTLASS 4.5 CollectiveBuilder (preferred path) for the fused kernel if it can express dual-phase shared TMEM accumulation
  - Hand-written CuTe DSL (fallback) if CollectiveBuilder cannot express the topology
  - Hand-written PTX (last resort) for specific TMA descriptor / `tcgen05.mma` instruction sequences not covered by CUTLASS abstractions (Known Risk: CUTLASS 4.5 SM100 examples are sparse)
  - CUDA Graph or pre-recorded sequence for the 2-launch FP baseline timing window
  - Either (a) FP32 global scale per tensor for NVFP4 (Transformer Engine convention) or (b) no global scale (block-scale-only), with a documented rationale in the AC-1 spec
  - Per-operator high/low ratio (1/16, 1/8, 1/4, 1/2 etc., subject to alignment) as an algorithm-side knob, including o_proj-specific raised ratio if Alt-4 fallback triggers
- Cannot use:
  - Any path that bypasses the AC-5 vs-2-launch comparison (the PLAQuant claim is non-negotiable)
  - Verifying only on 1B and inferring 3B/8B (user requires all three)
  - Python-orchestrated 2-launch baseline (must be GPU-resident; Python launch overhead in the timing window invalidates the comparison)
  - SM89/SM90 hardware compatibility constraints (project is locked to SM100; this enables MXFP8/NVFP4 native paths)
  - Dropping per-stage attribution when end-to-end fails (un-attributable failure is rejected)

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

```
┌──────────────────────────────────────────────────────────────────┐
│ Algorithm side (ResQ-FP)                                         │
│                                                                  │
│   x_input ──> RMSNorm fused ──> rotation R1 (hidden_dim, full   │
│            global, fused into next Linear weight at PTQ)         │
│                                                                  │
│   For q/k/v/gate/up/down: quantizer routes top-K variance       │
│       channels to MXFP8, rest to NVFP4 (high_fraction=1/8)       │
│                                                                  │
│   For o_proj: NEW global hidden_dim PCA replaces per-head 64×64; │
│       no per-group scale; same MXFP8/NVFP4 split as other layers │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ Kernel side (kernels/mixed_gemm_sm100/)                          │
│                                                                  │
│   ┌───────── TMEM FP32 accumulator (M × N) ────────────┐         │
│   │   Phase 1: tcgen05.mma.kind::mxf8f6f4 (MXFP8)      │         │
│   │       A_high (FP8 E4M3) × B_high (FP8 E4M3)        │         │
│   │       block scale (FP8 E8M0) per 32 K-elements     │         │
│   │   Phase 2: tcgen05.mma.kind::mxf4nvf4 (NVFP4)      │         │
│   │       A_low (FP4 E2M1) × B_low (FP4 E2M1)          │         │
│   │       block scale (FP8 E4M3) per 16 K-elements     │         │
│   │       same TMEM C, scaleC=true (FP32 add)          │         │
│   └─────────────────────┬──────────────────────────────┘         │
│                         ▼                                        │
│             EVT epilogue: dequant + cast to FP16 output          │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                 FP16 output (M × N)
```

Scale composition (the formula `out = s_x_token · s_w_channel · acc` in the draft is shorthand and CAN double-count if treated naively — AC-1 spec authoring resolves this precisely):
- `tcgen05.mma.kind::mxf8f6f4` and `kind::mxf4nvf4` apply the **block scales internally** during the MMA. The output of a single MMA call into TMEM is already block-scale-applied: `acc[m,n] += sum_k(A[m,k] * scale_A[m, k/block_size]) * (B[n,k] * scale_B[n, k/block_size])`. The FP32 accumulator therefore stores the true (de-block-scaled) inner product up to FP8/FP4 element rounding error.
- The epilogue therefore handles only the **remaining scales**: per-token activation scale `s_x_token` (1 per row, FP8 or FP16, computed at activation quantization time), per-channel weight scale `s_w_channel` (1 per column, FP16 or FP32, captured at weight packing time), and OPTIONALLY for NVFP4 the global FP32 scale `g_w` (per tensor, DEC-2 outcome).
- Final output: `out[m,n] = (acc_high[m,n] + acc_low[m,n]) * s_x_token[m] * s_w_channel[n] * g_w` cast to FP16. AC-1's fake-vs-real equivalence test verifies this composition is consistent between fake quantizer and kernel epilogue.

The key bet: shared TMEM accumulator across two heterogeneous MMA atoms is the SM100-native re-expression of the SM80 shared-register-accumulator pattern in `kernels/mixed_gemm_l20/mixed_gemm_l20.cu`. This is technically an unconfirmed feasibility claim until M0's dual-phase PoC (task5) validates it.

**CollectiveBuilder fallback ladder** (Codex round-2 optional improvement; gives explicit progression rather than ad-hoc):
1. Attempt CUTLASS 4.5 `CollectiveBuilder<KernelTmaWarpSpecialized, ...>` for both phases sharing one TMEM C — this is the cleanest API path
2. If CollectiveBuilder cannot express the dual-phase shared-accumulator pattern (likely, since SM100 examples don't include this case), drop to **CuTe DSL** (manual layout / TMA descriptor / WGMMA dispatch); cite specific CuTe primitives used
3. If CuTe still cannot express the kind transition mid-kernel, fall back to **hand-written PTX** for the `tcgen05.mma.kind::*` instruction sequence; isolate the PTX block to one well-commented file
The progression must be documented in code comments + RESULTS.md, not chosen ad-hoc; each fallback step has a documented attempt log.

For decode-regime shapes (M ∈ {1, 16}), the SM100 MMA atom requires M ∈ {64, 128}. The plan accepts that fused may underperform here and requires the failure to be classified and documented (AC-4 negative test). Mitigation candidates if decode is critical (DEC-3): (a) M-padding to 64 with padding mask in epilogue, (b) grouped GEMM batching multiple decode requests under the M=64 atom, (c) separate GEMV kernel for M=1 not using MMA atom.

### Relevant References

- `kernels/mixed_gemm_l20/mixed_gemm_l20.cu` — SM80 PLAQuant fused kernel (571 LOC); architecture template for SM100 port; specifically the dual-phase shared-accumulator pattern
- `kernels/mixed_gemm_l20/benchmark.py` — three-column micro benchmark methodology (fused / 2-launch INT / FP16); the SM100 benchmark.py mirrors this structure
- `kernels/mixed_gemm/mixed_gemm.cu` — SM90 CollectiveBuilder + WGMMA + EVT (410 LOC); CUTLASS 3.x reference for SM100 collective patterns
- `third_party/cutlass/include/cute/atom/mma_traits_sm100.hpp` — SM100 atom definitions, including the two PRIMARY atoms `SM100_MMA_MXF8F6F4_*` and `SM100_MMA_MXF4NVF4_*`
- `third_party/cutlass/include/cute/arch/mma_sm100_umma.hpp` — full PTX literals for `tcgen05.mma.kind::*` instructions
- `promix/quantize/basis.py` — current per-head + hidden-dim PCA computation; needs `o_proj_pca: full_global` mode addition
- `promix/quantize/optimize_rotation.py` — Stiefel-manifold rotation training; FP noise model swap
- `promix/inference/real_forward.py` — current INT real-forward path (line 149: o_proj skip comment); FP branch addition
- `promix/inference/weight_packer.py` — current INT packer (line 34-36: groupsize > 0 skip); FP branch addition
- `tests/test_mixed_gemm.py` — existing kernel correctness tests; FP variants are mechanical extension
- `project-resq/` — original ResQ algorithm reference (submodule)

## Dependencies and Sequence

### Milestones

1. **M0 — Format spec + minimal kernels (gating: AC-1, partial AC-3)**
   - Phase A: Author MXFP8/NVFP4 quantization spec doc (AC-1)
   - Phase B: Minimal single-phase MXFP8 kernel + minimal single-phase NVFP4 kernel; **correctness vs FP32 reference (cosine ≥ 0.9999) is the gate**, not perf
   - Phase C: Minimal **dual-phase shared TMEM accumulator PoC** (Codex's missing piece; β-0 originally only had two single-phase kernels) — this is the highest-risk feasibility unknown and must be retired before M2 starts
   - Phase D: Quick-and-dirty 2-launch FP baseline (Python-orchestrated single-point) for early signal — its number is informational, NOT a comparator at this stage; the AC-5 comparator is the optimized GPU-resident version built in M2 Phase C
   - Phase E: M0 acceptance:
     - All three kernels (MXFP8 single, NVFP4 single, dual-phase PoC) **compile and run** with cosine ≥ 0.9999 vs FP32
     - Single-phase MXFP8 and NVFP4 each match FP16 within 0.9× single-point (i.e., NOT slower than 90% of FP16 — single-phase kernels are not expected to win 1.5× at this stage; what matters is they aren't broken)
     - **Dual-phase PoC successfully reuses TMEM across heterogeneous atoms** — this is the binary go/no-go for M2; if it cannot be made to work after a documented attempt with CUTLASS CollectiveBuilder + CuTe DSL fallback, escalate to Alt-1 (path α) or extend timebox per DEC-5
     - 2-launch FP single-point latency is recorded as M2 ceiling estimate (the optimized version in M2 must beat or match this)

2. **M1 — Algorithm sanity (gating: AC-2)**
   - Phase A: Add `o_proj_pca: full_global` mode to `promix/quantize/basis.py`
   - Phase B: INT-path smoke test (AC-2.1) — run global PCA on o_proj with INT W4A4 quantizer on 1B; PPL must be ≤ 11.70 + 0.10
   - Phase C: Add MXFP8/NVFP4 fake quantizers in `promix/quantize/quant_utils.py`; FP-aware rotation noise in `promix/quantize/optimize_rotation.py`; FP-aware GPTQ in `promix/quantize/gptq.py`
   - Phase D: Re-run Step 0 (PCA) + Step 1 (rotation) + Step 2 (PTQ) for 1B with new FP fake quantizer; verify AC-2.2 for 1B
   - Phase E: Layer-wise output cosine sanity AC-2.4 on 1B (microscaled o_proj vs INT per-group)
   - Phase F: Repeat for 3B and 8B; verify AC-2.2 + AC-2.3 for all three sizes
   - Hard gate: AC-2 must pass for all (1B/3B/8B) × (W4A4/W4A4KV4) combinations before M2 begins

3. **M2 — Fused SM100 kernel + characterization (gating: AC-3, AC-4, AC-5)**
   - Phase A: New `kernels/mixed_gemm_sm100/` module; CUTLASS 4.5 CollectiveBuilder dual-phase (or CuTe fallback)
   - Phase B: Bit-exact correctness AC-3 across full shape sweep
   - Phase C: Optimized GPU-side 2-launch FP baseline (CUDA Graph or pre-recorded sequence; same timing harness)
   - Phase D: Shape-sweep `benchmark.py` with **four-column** output (fused / eager-2-launch / CUDA-Graph-2-launch / FP16) + per-stage profile
   - Phase E: AC-4 (vs FP16) + AC-5 (vs 2-launch FP) characterization, RESULTS.md generation
   - Hard gate: AC-5 geomean ≥ 1.05× on LLM-hit shapes; if not, tile-shape / SMEM tune; if still not, trigger Alt-1 fallback

4. **M3 — End-to-end real-FP integration (gating: AC-6, AC-7)**
   - Phase A: FP weight packer in `promix/inference/weight_packer.py`; remove `groupsize > 0` skip
   - Phase B: FP activation quantization in `promix/inference/quant_ops.py` (per-token MXFP8/NVFP4)
   - Phase C: FP forward path in `promix/inference/real_forward.py`; o_proj joins common path
   - Phase D: EVT epilogue dequant fusion (CUTLASS EVT pattern from `kernels/mixed_gemm/mixed_gemm.cu`)
   - Phase E: AC-6 (real PPL ≈ fake PPL) + AC-7 (latency vs FP16 + vs 2-launch FP) on 1B/3B/8B
   - Per-stage timing breakdown for any failing model

5. **M4 — Reproducibility + closeout (gating: AC-8, AC-9)**
   - Phase A: RESULTS.md complete (raw CSVs, env, commits, profiler outputs)
   - Phase B: Rotation artifact metadata (commit + dataset + nsamples)
   - Phase C: Documentation consistency fixes for AC-9 in any new derived doc

6. **M5 (Optional, OUT of lower bound) — β-4 per-operator variable ratio**
   - Per-Linear high_fraction tuning subject to (K_high % 32 = 0) ∧ ((K_total - K_high) % 64 = 0) constraints
   - Static heuristic vs eigenvalue-cumulative threshold vs gridsearch — choose one based on M3 results
   - Independent algorithmic contribution; not a core deliverable

### Dependencies

- M0 must complete before M2 (kernel feasibility unknown otherwise — dual-phase TMEM PoC is the binary go/no-go)
- M1 1B fake-FP sanity (AC-2.2 for 1B) must complete before M3 (no point integrating if PPL fails on 1B)
- **M1 (3B/8B PPL) and M2 (kernel work) can parallelize after 1B fake-FP sanity passes** (Codex round-2 optional improvement) — they touch independent surfaces (algorithm-side rotation/PCA/quantizer vs kernel-side CUTLASS/CuTe). Re-serialize at M3 entry where both must be ready.
- M2 must complete before M3 Phase A (no kernel = no real path)
- M2 Phase C (optimized 2-launch FP baseline) is on the critical path for AC-5 — without it the PLAQuant claim has no comparator
- M3 depends on M1's basis.py global PCA AND M2's fused kernel + M2's 2-launch FP baseline
- M5 is a leaf, depends on M3 completion (uses real-FP path infrastructure)

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Author quantization format spec doc (`docs/spec-mxfp8-nvfp4.md`): MXFP8/NVFP4 element format, block size, scale dtype (E8M0/E4M3), scale axis, rounding, saturation, NaN/Inf, packing layout, NVFP4 global FP32 scale policy | AC-1 | analyze | - |
| task2 | Read CUTLASS 4.5 SM100 examples + `mma_traits_sm100.hpp` + `mma_sm100_umma.hpp` to confirm `tcgen05.mma.kind::mxf8f6f4` and `kind::mxf4nvf4` API surface; produce a brief feasibility memo about CollectiveBuilder dual-phase support | AC-3 | analyze | task1 |
| task3 | Implement minimal single-phase MXFP8 GEMM kernel (M=128, N=128, single tile); cosine ≥ 0.9999 vs FP32; ≥ 1.5× FP16 single-point | AC-3, AC-4 | coding | task2 |
| task4 | Implement minimal single-phase NVFP4 GEMM kernel; cosine ≥ 0.9999; ≥ 1.5× FP16 single-point | AC-3, AC-4 | coding | task2 |
| task5 | **Implement minimal dual-phase shared TMEM accumulator PoC** (the highest-risk feasibility unknown — Codex's "must validate before M2") | AC-3, AC-5 | coding | task3, task4 |
| task6 | Implement Python-orchestrated 2-launch FP baseline (single-point) for M0 early data; record latency target | AC-5 | coding | task3, task4 |
| task7 | Add `o_proj_pca: full_global` mode to `promix/quantize/basis.py` | AC-2.1 | coding | task1 |
| task8 | INT-path smoke test on 1B with global PCA on o_proj; verify PPL within ±0.10 of 11.70 | AC-2.1 | coding | task7 |
| task9 | Add MXFP8 + NVFP4 fake quantizers in `promix/quantize/quant_utils.py`; route via `bits=mxfp8` and `bits=nvfp4` | AC-2.2 | coding | task1 |
| task10 | Update `promix/quantize/optimize_rotation.py` quant noise model from INT round to FP cast | AC-2.2 | coding | task9 |
| task11 | Update `promix/quantize/gptq.py` round-to-nearest from INT to FP-nearest-representable | AC-2.2 | coding | task9 |
| task12 | Create `promix/configs/llama-3.2-1b-mxfp8-nvfp4.yaml` (and 3B/8B variants); re-run Step 0 (PCA) on each model with global o_proj PCA | AC-2.2 | coding | task7, task9 |
| task13 | Run Step 1 (rotation re-train) on 1B/3B/8B with new FP noise model | AC-2.2 | coding | task10, task12 |
| task14 | Run Step 2 (PTQ eval) for fake-FP W4A4 on 1B/3B/8B; verify AC-2.2 thresholds | AC-2.2 | coding | task13 |
| task15 | Run W4A4KV4 variant for 1B/3B/8B; verify AC-2.3 thresholds | AC-2.3 | coding | task13 |
| task16 | Layer-wise output cosine sanity check (microscaled o_proj vs INT per-group) on 1B | AC-2.4 | coding | task14 |
| task17 | Implement fused SM100 dual-phase kernel in new `kernels/mixed_gemm_sm100/` module using validated PoC pattern from task5 | AC-3, AC-4, AC-5 | coding | task5 |
| task18a | Implement Eager 2-launch FP baseline (Python-orchestrated `mxfp8_gemm()` + `nvfp4_gemm()` with GEMM2 beta=1 accumulation; no CUDA Graph; deployment-realistic comparator for AC-5 Variant E) | AC-5 | coding | task17 |
| task18b | Wrap the same two GEMMs into a CUDA Graph / pre-recorded sequence (Python launch overhead OUT of timing window; topology-only comparator for AC-5 Variant G) | AC-5 | coding | task18a |
| task19 | Implement `kernels/mixed_gemm_sm100/benchmark.py` with **four-column** output (fused / eager-2-launch / CUDA-Graph-2-launch / FP16 cuBLAS) + per-stage profile across full shape sweep | AC-3, AC-4, AC-5 | coding | task17, task18a, task18b |
| task20 | Run shape-sweep correctness verification (cosine ≥ 0.9999 across all sweep tuples) | AC-3 | coding | task19 |
| task21 | Run shape-sweep performance characterization; generate RESULTS.md with heatmaps + per-stage profile + per-shape attribution | AC-4, AC-5, AC-8 | coding | task20 |
| task22 | Verify AC-4 thresholds (vs FP16) on LLM-hit shape subset | AC-4 | analyze | task21 |
| task23 | Verify AC-5 thresholds for BOTH variants (vs eager 2-launch AND vs CUDA Graph 2-launch) on LLM-hit shape subset; if either variant's per-shape > 1.0× HARD bar fails, trigger tile-shape tune iteration; report both variants in characterization | AC-5 | analyze | task21 |
| task24 | Implement FP weight packer in `promix/inference/weight_packer.py`; delete `groupsize > 0` skip; o_proj joins common path | AC-6 | coding | task14, task17 |
| task25 | Implement FP per-token activation quantization in `promix/inference/quant_ops.py` | AC-6, AC-7 | coding | task17 |
| task26 | Implement FP forward path in `promix/inference/real_forward.py`; remove o_proj fake-quant skip | AC-6 | coding | task24, task25 |
| task27 | Implement EVT epilogue fusing dequant (FP path: `out = s_x_token · s_w_channel · acc` + cast) | AC-7 | coding | task17 |
| task28 | Run real-FP end-to-end PPL on 1B/3B/8B; verify AC-6 within ±0.05 of fake-FP | AC-6 | coding | task26 |
| task29 | Run real-FP end-to-end latency on 1B/3B/8B; verify AC-7 vs FP16 + vs 2-launch FP | AC-7 | coding | task27, task28 |
| task30 | Generate per-stage attribution (kernel/quant/dequant/non-GEMM) for every failing model size | AC-7, AC-8 | coding | task29 |
| task31 | Finalize RESULTS.md with reproducibility section: raw CSVs, env versions, commit hashes, profiler outputs, rotation artifact metadata | AC-8 | coding | task29, task30 |
| task32 | Documentation consistency cleanup: ensure all derived docs use MXFP8=E8M0 (not E4M3), `SM100_MMA_MXF8F6F4_*` (not F8F6F4), and corrected β-4 alignment example pairs | AC-9 | analyze | task31 |

## Claude-Codex Deliberation

### Agreements

- The PRIMARY direction (single-kernel SM100 FP8+NVFP4 fused) preserves PLAQuant's system-level contribution and aligns with B20Z native MMA support
- ResQ algorithm reuse (PCA + variance split + Stiefel R) is a sound algorithmic foundation; FP rotation R must be retrained because the noise model changed
- TWO 2-launch FP baseline variants are measured separately — eager (deployment-realistic, includes Python launch overhead) AND CUDA Graph (paper-defensible topology-only) — both required for the PLAQuant single-kernel claim to fully stand
- Quantization format spec must be authored before implementation; fake/real path equivalence depends on it
- Decode-regime shapes (M=1/16) are a real risk given SM100 MMA atom M ∈ {64, 128}; plan must explicitly handle them or document failure regime
- Per-stage attribution (kernel time / quant time / dequant time / non-GEMM time) is required for any failing end-to-end measurement so failures are debuggable
- A dual-phase shared TMEM accumulator PoC must be validated before committing to the fused kernel (Phase β-0 originally had only single-phase PoCs; that is insufficient)

### Resolved Disagreements

- **Topic**: Strength of "global PCA ≥ per-head PCA" mathematical argument
  - Claude position: Strict superset, top-K variance capture is monotone — variance argument is sufficient justification
  - Codex position: Variance capture alone does not prove PPL non-regression after split / scaling / packing / re-trained R
  - Resolution: Keep global PCA as PRIMARY (strict-superset variance argument is correct AS variance argument), but require AC-2.1 INT-path smoke test as the empirical bridge — this is what the draft already does. The mathematical argument is a sufficient reason to TRY global PCA; the smoke test is the empirical proof. Both are needed; neither alone is enough.
  - Rationale: Variance ≠ PPL, but variance-superset gives global PCA a mathematical floor of "no worse than per-head" that empirical PPL test can confirm. The explicit smoke-test gate guards against implementation bugs masquerading as design failure.

- **Topic**: 2-launch FP baseline strength
  - Claude position (in draft): Python-orchestrated `mxfp8_gemm() + nvfp4_gemm() + FP32 add` (eager mode)
  - Codex position: too weak — Python launch overhead inside the timing window inflates fused's apparent advantage
  - Initial Codex-driven resolution: switch to GPU-resident CUDA Graph baseline only
  - **Final resolution after user pushback**: measure BOTH variants and report separately:
    - **Variant E (Eager, no CUDA Graph)** — the deployment-realistic comparator. Reducing launch overhead is a legitimate PLAQuant advantage; excluding it from the baseline hides a real benefit users would see in PyTorch eager mode and in vLLM with dynamic batching where CUDA Graph is not always applicable.
    - **Variant G (CUDA Graph)** — the paper-defensible topology-only comparator. Beating it proves fused has TMEM-accumulator-sharing benefit beyond just launch-overhead reduction.
  - Rationale: User correctly observed that the Codex-driven "strongest baseline only" framing was over-rotated. The two variants answer different questions (deployment vs topology) and BOTH are interesting. Marginal cost is one extra harness wrapping (CUDA Graph capture); both share the same accumulation path. Plan AC-5 now mandates both variants with separate per-shape > 1.0× HARD bars, and `benchmark.py` outputs four columns (fused / eager-2-launch / CUDA-Graph-2-launch / FP16 cuBLAS) instead of three.

- **Topic**: β-0 should include dual-phase shared TMEM PoC, not only two single-phase kernels
  - Claude position (in draft): β-0 has minimal MXFP8 single-phase + minimal NVFP4 single-phase
  - Codex position: This validates atom availability but not the heterogeneous-phase shared-accumulator topology that is the core feasibility unknown
  - Resolution: Added Phase C "minimal dual-phase shared TMEM accumulator PoC" to M0 as task5; M2 is gated on its success. This addresses the highest-risk feasibility assumption before committing to the full fused kernel.
  - Rationale: Codex correctly identifies that the topology is the bet, not the atoms.

- **Topic**: "Any fail = project incomplete" gating
  - Claude position: Per user's explicit requirement, all (1B/3B/8B) × (W4A4/W4A4KV4) × (precision/perf) must pass
  - Codex position: This is a brittle research gate; one hard regime (e.g., decode M=1) can fail the whole project even if main contribution holds
  - Resolution: AC-2 / AC-6 / AC-7 keep the strict bar (per user requirement). AC-4 explicitly carves out a documented "decode regime" (M ∈ {1, 16}) where below-1.0× is acceptable IF the failure is classified and accompanied by mitigation candidates (padding / grouped GEMM / separate kernel). This preserves user's coverage requirement at the model-size level while acknowledging that some shape regimes inherently don't fit SM100 MMA atoms. The hard gate moves from "all shapes" to "all model sizes × all configs", with documented shape exceptions.
  - Rationale: User's intent (1B/3B/8B all valid + W4A4 + W4A4KV4) is preserved at the deliverable level. Codex's concern about un-mitigated brittleness is addressed at the shape level by requiring explicit failure classification.

- **Topic**: NVFP4 quantization recipe complexity (FP32 global scale)
  - Claude position (draft): Use simple per-token + per-channel + per-block scale; no global FP32 scale
  - Codex position: NVIDIA Transformer Engine NVFP4 spec recommends FP32 global scale + 2D weight scaling; simple recipe may not match hardware behavior exactly
  - Resolution: AC-1 spec must explicitly choose YES/NO on global FP32 scale with documented rationale. If NO is chosen (simplest path, matches draft), spec must include a fake-vs-real equivalence test confirming bit-pattern match between fake quantizer and real packer/kernel. If real-vs-fake mismatches > tolerance during M3, NVFP4 global scale becomes a forced revisit.
  - Rationale: Forces explicit decision rather than implicit one; equivalence test catches the bug class Codex flagged.

- **Topic**: Internal draft inconsistency (MXFP8 scale dtype, β-2 atom name, β-4 example pair)
  - Claude position: These are textual bugs in the draft; they don't affect the design
  - Codex position: They MUST be reconciled before code derives from them, otherwise quantizer/kernel/packer disagree
  - Resolution: AC-9 added as a documentation consistency criterion. The original draft remains as committed; AC-9 mandates that all DERIVED documents (spec, configs, code, RESULTS.md) use the canonical interpretation: MXFP8=E8M0, atom=`SM100_MMA_MXF8F6F4_*`, β-4 valid alignment minimum K_high=64 not 32.
  - Rationale: User instruction explicitly preserves draft content; consistency is enforced at the derived-doc layer.

### Round 2 Resolutions (Codex Round 2 review of Candidate v1)

Round 2 raised 8 REQUIRED_CHANGES; all addressed in revised plan:
1. **2-launch baseline strength** (was: GEMM1 + GEMM2 + third FP32 add): revised to use **strongest feasible accumulation path** (priority: GEMM2 with `beta=1` semantics → epilogue-fused add → last resort separate add kernel with documentation). After user pushback during Phase 6, AC-5 was further revised to measure BOTH **eager** and **CUDA Graph** baseline variants separately (not "CUDA Graph only" as the Codex round-2 fix originally pushed). Eager is the deployment-realistic comparator (PLAQuant gets credit for launch-overhead reduction); CUDA Graph is the paper-defensible topology-only comparator (proves TMEM accumulator-sharing benefit alone). Captured in revised AC-5.
2. **Shape sweep matrix** (was: synthetic N ∈ {2048, 4096, 8192} with mostly K=2048): revised to **derive from actual Llama 1B/3B/8B layer dimensions** including 3B intermediate=8192/hidden=3072 and 8B intermediate=14336/hidden=4096, plus all q/k/v/o/gate/up/down combinations. Captured in AC-3 shape sweep matrix.
3. **Workload regime split** (was: blended end-to-end): revised to **prefill / decode separated explicitly** with seq_len + batch documented. Captured in AC-7 workload regimes.
4. **AC-4/AC-5 threshold consistency** (was: AC-5 50% shapes >1.05× was lenient): tightened to **every prefill-regime LLM-hit shape > 1.0× single-shape; geomean ≥ 1.05×**. Captured in AC-5 positive tests.
5. **Scale-composition spec** (was: epilogue formula under-specified): added explicit decomposition of which scales MMA consumes vs which epilogue handles, with double-counting prohibition. Captured in AC-1 + Feasibility Hints.
6. **AC-2.1 ambiguity** (was: ±0.10 / +0.30 with no middle): revised to **three explicit bands** (Pass / Investigate / Stop-and-debug). Captured in AC-2.1.
7. **W4A4KV4 explicit coverage** (was: implicit only in AC-2.3): revised to **AC-6.1/6.2 and AC-7.1/7.2 explicit sub-criteria** with DEC-4 confirmation hook.
8. **M0 single-phase 1.5× gate** (was: each minimal kernel ≥ 1.5× FP16): revised to **correctness-first gating** (cosine ≥ 0.9999), with single-phase perf required only ≥ 0.9× FP16 since single-phase isn't expected to win at M0; **dual-phase PoC is the binary go/no-go** for M2.

Round 2 OPTIONAL_IMPROVEMENTS adopted: relative-error threshold added to AC-3; M1/M2 parallelization noted in Dependencies; CollectiveBuilder fallback ladder documented in Feasibility Hints.

Round 2 OPTIONAL_IMPROVEMENTS deferred: explicit time-box for CollectiveBuilder (the fallback ladder substitutes a quality-gate ladder for a time-box; rule prohibits time estimates anyway); Alt-4 fallback acceptance — left as "investigation triggered" since exact deliverable depends on debug findings.

### Round 3 Convergence Verdict

Round 3 (Codex final-pass review): **CONVERGED**. No remaining REQUIRED_CHANGES; no NEW_DISAGREE introduced by round-2 fixes. Round 3 confirmed all 8 round-2 required changes are properly addressed.

### Convergence Status

- Final Status: `converged` (after Phase 6 user decisions DEC-1, DEC-3, DEC-4, AC-thresholds resolved; DEC-2, DEC-5 defaulted to Claude recommendation with override paths)

## Pending User Decisions

> Status snapshot: 4 of 6 decisions resolved by user in Phase 6; DEC-2 and DEC-5 defaulted to Claude's recommendation (user is free to override during RLCR loop).

- DEC-1: FP16 baseline comparator strength → **RESOLVED: torch.matmul (per draft)**
  - User Decision: `torch.matmul` (FP16, auto-dispatches to cuBLAS / 5th-gen TC). Deployment-realistic for PyTorch users.
  - Implication: AC-4 / AC-7 measurements use `torch.matmul`. cuBLASLt + CUDA Graph variant is OUT of scope for the comparator.
  - Decision Status: RESOLVED

- DEC-2: NVFP4 global FP32 scale policy → **DEFAULT: NO global FP32 scale (Claude recommendation, override-able)**
  - Claude Default: Block-scale only — simpler packer, simpler epilogue, ResQ rotation pre-conditioning makes block-scale alone likely sufficient
  - Override path: AC-1 fake-vs-real equivalence test + AC-2.2 W4A4 PPL on 1B; if PPL fails, revisit with global FP32 scale added
  - Decision Status: DEFAULT (RESOLVED with override-on-failure escape)

- DEC-3: Decode regime explicit handling → **RESOLVED with user clarification**
  - User Clarification: "decode 也可以组成 batch — batch=64 decode M=64 完美对齐 MMA atom"; the prior framing of "decode regime is a problem" was overstated. Real serving runs batched decode (batch ∈ {16, 64, 128, ...}) so M ∈ {16, 64, 128, ...}. Only batch=1 single-user decode is the true M=1 corner case.
  - Implication: AC-4 / AC-7 distinguish three regimes — prefill (large M), batched decode (M ∈ {64, 128, ...}, fits MMA atom, HARD > 1.0× bar), single-user decode (M=1, edge case, mitigation listed but not hard-failing). Plan rewritten accordingly.
  - Decision Status: RESOLVED

- DEC-4: W4A4KV4 in this phase or deferred milestone → **RESOLVED: include both (per draft)**
  - User Decision: Both W4A4 and W4A4KV4 are in this phase. AC-2 / AC-6 / AC-7 all evaluate both configs.
  - Implication: 6-cell acceptance matrix (1B/3B/8B × W4A4/W4A4KV4) at PPL + 6-cell at latency.
  - Decision Status: RESOLVED

- DEC-5: Hand-written PTX/CUTE acceptable as fallback → **DEFAULT: yes, with documented fallback ladder (Claude recommendation, override-able)**
  - Claude Default: CollectiveBuilder → CuTe DSL → hand-written PTX, in that priority order, with a documented attempt log at each level (rule prohibits time-box estimates so we use a quality-gate ladder instead)
  - Override path: user can specify "CollectiveBuilder only, fail loudly if it doesn't cover" if the maintenance burden of CuTe / PTX is not acceptable
  - Decision Status: DEFAULT

- AC-4 / AC-5 acceleration thresholds → **RESOLVED: > 1.0× HARD, 1.5× / 2.0× / 1.05× geomean DIRECTIONAL**
  - User Decision: > 1.0× is the hard bar (cannot be slower than baseline); 1.5× at M=128, 2.0× at M=2048, 1.05× geomean are written in RESULTS.md as goals but missing them by epsilon does not fail acceptance
  - Implication: AC-4 and AC-5 positive/negative tests rewritten to separate HARD bar (must pass) from DIRECTIONAL goals (recorded but not auto-failing)
  - Decision Status: RESOLVED

- DEC-6: Activation quantization in latency window from day one → **DEFAULT: include (per draft; not explicitly contested)**
  - Decision Status: DEFAULT (RESOLVED via AC-7 explicit "activation quantization IS in the latency window" wording)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", "M0/M1/M2", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead (e.g., function `validate_mxfp8_format` not `ac1_validate_format`; commit message "fp8 quantizer correctness sweep" not "task20")

### Draft Inconsistency Reconciliation (AC-9 detail)
The original committed draft (`.humanize/ideas/resq-mixed-precis-20260621-154239.md`) contains three textual inconsistencies that must NOT propagate to derived documents:
- **MXFP8 block scale dtype**: Lines 90, 143, 234 say "E4M3"; the correct value is **FP8 E8M0**. Line 155 (the format comparison table) and Line 320 (Known Risk reference) have it correct.
- **High-path atom name**: Lines 248 (β-2 Phase 1) and 307 (Objective Evidence) say `SM100_MMA_F8F6F4_*` (dense); PRIMARY uses `SM100_MMA_MXF8F6F4_*` (microscaled). Lines 73, 78, 143 have it correct.
- **β-4 alignment example**: Line 288 lists `(K_high=32, K_low=1984)` — these don't sum to 2048 (32 + 1984 = 2016). The minimum valid K_high under (K_high % 32 == 0) ∧ ((2048 - K_high) % 64 == 0) is K_high=64.

All AC-1 spec, fake quantizer, real packer, kernel template, and RESULTS.md outputs must use the canonical interpretation, not the draft's bugged statements.

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# PLAQuant-SM100: Single-Kernel FP8+NVFP4 Mixed-Precision via tcgen05.mma

## Goal

Make the mixed-precision fused GEMM kernel (the PLAQuant system design) actually used in LLM inference, **simultaneously across 1B/3B/8B model sizes**, satisfying three hard criteria — each measured at two layers (kernel-level micro benchmark + end-to-end real inference):

1. **Precision non-regression** — wikitext PPL does not regress relative to the existing INT4+INT8 W4A4 baseline (quant-vs-quant comparison, NOT vs FP16; vs FP16 is too strict to be realistic)
2. **Faster than FP16 cuBLAS** — the kernel on the LLM-realistic shape sweep AND end-to-end real inference both beat torch.matmul FP16 (5th-gen TC cuBLAS)
3. **Faster than 2-launch FP** — the single fused kernel launch beats "two separate launches (independent FP8 GEMM for the high path + independent NVFP4 GEMM for the low path, FP32 sequential accumulation)" — winning at both kernel-micro and end-to-end levels; this is the core claim of the PLAQuant single-kernel system contribution

Why measure at both layers: **only running end-to-end leaves it unclear where the perf gain comes from** (which layer / which shape contributed?); **only running micro leaves it unclear whether Python-side quant pack / dequant overhead consumed the gain**. Both layers must pass for the project to be considered complete.

On the algorithmic side, the design uses ResQ's PCA + variance-split + learnable rotation as the mechanism that produces the "high / low precision channel" data layout.

## Final Acceptance Criteria

Hard criteria are split into two layers — **kernel layer (micro benchmark)** and **end-to-end layer (real inference)** — and each layer must be measured on 1B/3B/8B × W4A4/W4A4KV4. **Any single criterion failing on any combination = project incomplete**:

### End-to-end layer (real inference)

| Dimension | Threshold | 1B | 3B | 8B |
|---|---|---|---|---|
| **Precision (PPL non-regression) — W4A4** | wikitext PPL ≤ current INT4+INT8 W4A4 baseline | ≤ 11.70 | ≤ 8.61 | ≤ 6.99 |
| **Precision (PPL non-regression) — W4A4KV4** | same but with 4-bit KV cache | ≤ 11.96 | ≤ 8.70 | ≤ 7.04 |
| **Perf vs FP16 cuBLAS** | end-to-end latency speedup > 1.0× (target 1.5-2×) | ≥ 1.0× | ≥ 1.0× | ≥ 1.0× |
| **Perf vs 2-launch FP — Eager** | fused vs eager (no CUDA Graph) Python-orchestrated MXFP8 GEMM + NVFP4 GEMM, GEMM2 beta=1 accumulation; deployment-realistic comparator | > 1.0× | > 1.0× | > 1.0× |
| **Perf vs 2-launch FP — CUDA Graph** | fused vs CUDA-Graph-recorded sequential 2-launch (Python overhead OUT of timing window); paper-defensible topology-only comparator | > 1.0× | > 1.0× | > 1.0× |

### Kernel layer (micro benchmark, shape sweep)

Measures only the fused kernel itself, independent of the model — the goal is to know **at which (M, N, K_high, K_low) shapes fused beats 2-launch / FP16, and which shapes it loses on**. Looking only at end-to-end numbers, a win doesn't tell you which shape contributed and a loss doesn't tell you which layer dragged things down — **performance gains cannot be attributed**.

| Dimension | Threshold | Measurement |
|---|---|---|
| **Correctness** | every sweep shape cosine ≥ 0.9999 vs FP32 reference | no shape may distort across the entire sweep |
| **vs FP16 cuBLAS (per-shape)** | LLM-hit shape subset (1B/3B/8B layers × batch ∈ {1, 16, 64, 128}) fused/FP16 ≥ 1.0×; target ≥ 1.5× at M=128, ≥ 2× at M=2048 | per-shape speedup table + heatmap |
| **vs 2-launch FP — Eager (per-shape)** | LLM-hit shape subset fused/eager-2-launch > 1.0× (deployment-realistic; no CUDA Graph) | per-shape speedup table + heatmap |
| **vs 2-launch FP — CUDA Graph (per-shape)** | LLM-hit shape subset fused/CUDA-Graph-2-launch > 1.0× (paper-level topology-only claim) | per-shape speedup table + heatmap |
| **Characterization report** | `kernels/mixed_gemm_sm100/RESULTS.md` lists: shape regimes where fused wins / loses, per-stage profile (TMA / WGMMA / accumulate / store), attribution | β-2 deliverable; without this report β-2 is not considered passed |

Threshold definitions:
- **Precision non-regression** ≠ "slightly better" or "approximately equal" — any single PPL regression vs baseline = not met. **Float noise tolerance is ±0.01**; > +0.05 is treated as regression.
- **Perf vs FP16** compares against the real deployment baseline (torch.matmul FP16, dispatching to 5th-gen TC cuBLAS); minimum bar 1.0×, target 1.5-2×.
- **Perf vs 2-launch FP** is the core claim of the PLAQuant paper. Two variants reported separately: **Eager** (deployment-realistic: Python eager mode, launch overhead in window — beating it proves the deployment-level benefit including launch reduction) and **CUDA Graph** (paper-level: pre-recorded sequence, launch overhead out of window — beating it proves the TMEM-topology benefit alone). **This criterion does NOT depend on FP16 comparison; it is the hard proof of the PLAQuant claim.** Failing to win on Variant-E means deployment-level claim doesn't stand; failing to win on Variant-G means topology-level paper claim doesn't stand. Both must hold.
- **Kernel-layer and end-to-end layer cannot substitute for each other** — running only end-to-end without a shape sweep leaves the source of perf gain unknown; running only micro without end-to-end leaves it unknown whether Python-side dequant / quant pack consumed kernel gains. Both layers must pass for project completion.
- **Coverage** is NOT "verify on 1B and infer 3B/8B work the same way", it is **all three sizes must be measured** — the 8B model in particular is more likely to expose any subtle differences between global PCA and per-head due to cross-head statistical variation.

Every phase verification point (β-0..β-3) must trace directly to one cell of the two tables above; any phase verification without a corresponding acceptance row does not count as verification.

## Verified Current State (measured + source-code verified)

**Algorithm side (completed):**
- ResQ PTQ pipeline reproduces the paper's W4A4 / W4A4KV4 PPL on 1B/3B/8B (fake-quant path: quant→dequant→FP16 GEMM simulation)
- Step 0 PCA basis computation / Step 1 rotation optimization / Step 2 PTQ eval all run end-to-end under `promix/`

**Kernel layer (B20Z micro benchmark + CUTLASS 4.5 source check):**

| Comparison | Result |
|---|---|
| Current SM80 fused INT4+INT8 vs CUTLASS 2-launch INT (both on SM80 mma.sync) | 0.98-1.15× |
| Current SM80 fused INT4+INT8 vs FP16 cuBLAS (SM100 5th-gen TC) | **0.04-0.32×** (fused is 3-25× slower) |
| Current fused peak TFLOPS | 58.5 |
| FP16 cuBLAS peak TFLOPS | 1434.6 |

In other words: the "1.0-1.15× speedup" in earlier DEVLOG was vs INT's own 2-launch baseline, which is a different thing from the final goal (vs FP16 cuBLAS); **the real opponent — FP16 cuBLAS on SM100 — is 3-25× faster than the current kernel**.

**SM100 (Blackwell) native MMA type support (full enumeration from `include/cute/atom/mma_traits_sm100.hpp`):**

| Type | instruction kind | Supported | atom family |
|---|---|---|---|
| FP16 / BF16 | `kind::f16` | ✓ | SM100_MMA_F16BF16_* |
| TF32 | `kind::tf32` | ✓ | SM100_MMA_TF32_* |
| INT8 | `kind::i8` | ✓ | SM100_MMA_S8_* |
| FP8 / FP6 / FP4 (dense, no microscaling) | `kind::f8f6f4` | ✓ | SM100_MMA_F8F6F4_* |
| **MXFP8 / MXFP6 / MXFP4 microscaled** (block=32, FP8 E8M0 scale) | `kind::mxf8f6f4.block_scale` | ✓ | **SM100_MMA_MXF8F6F4_*** |
| MXFP4 (block=32, FP8 E8M0 scale) | `kind::mxf4nvf4` (VS=32) | ✓ | SM100_MMA_MXF4_* |
| **NVFP4 microscaled** (block=16, FP8 E4M3 scale) | `kind::mxf4nvf4` (VS=16) | ✓ | **SM100_MMA_MXF4NVF4_*** |
| **INT4** | — | ✗ | **does not exist** — Blackwell removed native INT4 |

The two atoms PRIMARY uses are bolded: high path `SM100_MMA_MXF8F6F4_*`, low path `SM100_MMA_MXF4NVF4_*`.

**End-to-end (observed):**
- `promix/inference/` is wired up with the real INT4+INT8 GEMM forward path; 1B pipeline does not crash
- 1B real-quant end-to-end is ~9× slower than FP16 (latency measured)
- BUT real-quant PPL has **never been measured**, so "PPL stays consistent after switching kernel" has no data backing it

## What Will Change (subsequent modifications, owned by PRIMARY)

Per the β-path decision, the items below will **no longer remain as-is**:

- The current INT4+INT8 real path under `promix/inference/` is **swapped wholesale to the MXFP8 + NVFP4 microscaled native path** (unified to one data model; **no dense FP fallback retained** — see "MXFP/NVFP vs Dense FP comparison" in PRIMARY for rationale)
- The ResQ quantizer is **changed from INT4/INT8 fake quant to MXFP8 (E4M3 + 32-element FP8 block scale) + NVFP4 (E2M1 + 16-element FP8 E4M3 block scale) fake quant**; PCA basis / high-low split / rotation architecture remain unchanged, but **R must be retrained** (per-block scale model differs from the original per-channel model)
- `kernels/mixed_gemm_l20/` is kept as SM80 legacy code + correctness oracle, **no longer the real inference path**; a new `kernels/mixed_gemm_sm100/` uses **`tcgen05.mma.kind::mxf8f6f4` + `kind::mxf4nvf4`** (**both microscaled**, not mixed with dense `f8f6f4`), sharing a TMEM FP32 accumulator
- 5th-gen Tensor Cores currently unused → the new kernel uses the native microscaled channel, targeting ~2-4× FP16 speedup (MXFP8 ~3 PFLOPS / NVFP4 ~6 PFLOPS theoretical ceiling)
- **The o_proj per-group quantization (groupsize=64) structure is replaced wholesale**: on the algorithm side, the o_proj input PCA is upgraded from per-head 64×64 to hidden_dim 2048×2048 global PCA (see PRIMARY "Companion algorithm-side change"); after the global rotation, the o_proj input is sorted by global variance and split directly into top-256 high / bottom-1792 low, **the per-head group concept ceases to exist** — the per-group scale is NOT absorbed into the block scale, it disappears from the model entirely. On the system side, o_proj follows the same code path as q/k/v/gate/up/down: per-row + per-channel + per-block scale, **no longer skipped, no longer specially handled**
- Python-side dequant's 9× end-to-end slowdown → resolved at the end of β-2 by fusing dequant into the kernel via EVT epilogue (the FP path dequant formula is simpler than INT path: only `out = s_x_token · s_w_channel · acc`, no INT-style shift/zero/colsum terms)
- The existing 1B/3B/8B PTQ rotation files break down by stored content as follows:
  - **`rotation/U-*.bin` partial regeneration** — the hidden_dim 2048×2048 PCA for q/k/v/gate/up is reusable (independent of quantization format); **but the o_proj input PCA is upgraded from per-head 64×64 to hidden_dim 2048×2048 global**, requiring a fresh basis.py run to generate new U files (cost: 1B ~20min, 3B ~30min, 8B ~45min; same order of magnitude as the original basis computation). See "Companion algorithm-side change: o_proj input PCA upgrade" below for rationale.
  - **`rotation/E-*.bin` partial regeneration** — eigenvalues are recomputed alongside U
  - **`rotation/R-*.bin` invalidated, must be retrained** — `optimize_rotation.py` trains against quantization noise; the original R was trained under the INT4/INT8 noise model + per-head U; both assumptions change after switching to MXFP8/NVFP4 + global U (cost: 1B ~1min, 3B ~5min, 8B ~10min; per-model retrain cost is manageable)
  - **Runtime composition logic unchanged** — Step 2's `rotation.py` line 194 computes `U_combined = U_pca @ R_new` in memory and applies it to the weight; this code does not change in PRIMARY; only the source tensors fed into the composition differ (new U + new R). Note: the "rotation actually applied to the weight" is the U×R composition, not the U file itself — the file-level split does not imply a mathematical-level split

## Primary Direction: PLAQuant-SM100 Single-Kernel FP8+NVFP4 Pipeline

### Rationale

The core contribution of our prior work (PLAQuant) is that "GEMMs of different precisions can be single-launch accumulated inside one kernel". It was realized on SM80 by sharing a register accumulator across two `mma.sync` atoms (INT4×INT4 + INT8×INT8). The ResQ algorithm naturally produces a data layout fitting this architecture (channel-variance-based high/low split, 1/8 high + 7/8 low), so the system layer + algorithm layer align naturally. But moving to B20Z (SM100 Blackwell) broke two things: (1) INT4 native does not exist on Blackwell, no CUTLASS atom either; (2) FP16 cuBLAS already runs on 5th-gen Tensor Cores at ~1.4 PFLOPS, while our SM80 path is ~58 TFLOPS — a 25× gap.

The correct response is **to port PLAQuant's single-kernel architecture from INT4+INT8 to SM100 native FP8+NVFP4**:
- System layer: fully preserve PLAQuant's single-kernel two-phase shared-accumulator topology; upgrade from register accumulator to TMEM accumulator
- Algorithm layer: ResQ → ResQ-FP; PCA basis / high-low split / rotation R all retained; only the quantizer is swapped from INT to FP

This path preserves PLAQuant's paper claim (system-level single-kernel multi-precision) + upgrades to the SM100 truly-native channel (theoretical ~3-4× FP16 speedup ceiling) + lands at a novel GEMM form (FP8+FP4 mixed PLAQuant). It has a **genuine algorithmic story compared to a plain SM100 INT8 degraded version (path α)**, and has a genuine hardware-generation advantage compared to plain SM90 INT4 validation (path γ).

### Approach Summary

**Core topology (unchanged vs SM80 PLAQuant):**

```
                        ┌──────────────────────────────────────┐
                        │   TMEM FP32 accumulator (M × N)      │
                        │   M ∈ {64, 128}, N ∈ [8, 256]        │
                        └──────────┬─────────┬─────────────────┘
                                   │ accumulate(scaleC=true)
       ┌───────────────────────────┘         └──────────────────────┐
       │ Phase 1: MXFP8 high path            │ Phase 2: NVFP4 low path
       │   tcgen05.mma.kind::mxf8f6f4        │   tcgen05.mma.kind::mxf4nvf4
       │   K_iter = K_high / 32              │   K_iter = K_low / 64
       │   A_high (FP8 E4M3, M×K_high)       │   A_low (NVFP4, M×K_low)
       │     + block_scale_A (FP8, 1/32)     │     + block_scale_A (FP8, 1/16)
       │   B_high (FP8 E4M3, N×K_high)       │   B_low (NVFP4, N×K_low)
       │     + block_scale_B (FP8, 1/32)     │     + block_scale_B (FP8, 1/16)
       │   TMA pull A/B + 2 scale tiles      │   TMA pull A/B + 2 scale tiles
       └─────────────────────────────────────┴──────────────────────┘
                                   │
                            ┌──────▼──────┐
                            │EVT epilogue │  s_x·s_w·acc → FP16
                            │ (done in β-2)│  (FP path has no shift/zero/colsum terms)
                            └──────┬──────┘
                                   ▼
                            Output (M×N) FP16
```

CUTLASS atom constraints (source: `include/cute/atom/mma_traits_sm100.hpp`):
- `SM100_MMA_MXF8F6F4_SS` (high path): M ∈ {64, 128}, N ∈ [8, 256], K = 256/sizeof_bits<A> = **32 elements** for FP8, **one FP8 (E8M0) block scale per 32 elements**
- `SM100_MMA_MXF4NVF4_SS` (low path): M ∈ {64, 128}, N ∈ [8, 256], K = 256/sizeof_bits<A> = **64 elements** for FP4, block scale vector size **VS=16 → NVFP4** (**NOT VS=32 MXFP4**), block scale is FP8 E4M3 (with 3-bit mantissa, ~1 bit more precision than MXFP4's E8M0 power-of-2 scale). Rationale: the low path carries 7/8 of the channels and is the quantization workhorse, so precision matters more; NVFP4 is not cross-vendor compatible (NVIDIA-only), but with the project hardware target locked to SM100 this does not matter
- **Both phases use microscaled atoms** — this is the key PRIMARY design (see "Why both phases use microscaled atoms" below)
- **The C (accumulator) layout does not depend on K** — it depends only on M and N. So choosing the same M, N for both phases → sharing one TMEM C region → direct FP32 accumulation. This is the natural extension of the SM80 shared-register-accumulator idea to TMEM.

**Background: MXFP / NVFP vs Dense FP comparison**

Plain FP (FP8 E4M3 / FP4 E2M1) is a per-element format with sign + exp + mantissa, with the scale being per-tensor or per-channel shared. Microscaled FP (MXFP / NVFP) adds one more level of structure: **every N consecutive elements share one FP8 block-level scale, and the per-element FP encoding only carries "the relative position within this small block"**. Specifics:

| Format | Element | block_size | Block scale | Effective bits/element | Standard |
|---|---|---|---|---|---|
| FP8 (dense) | E4M3 / E5M2 | — (per-channel) | FP32 (per channel, amortized) | 8 | NVIDIA early |
| **MXFP8** | E4M3 / E5M2 | 32 | FP8 E8M0 (power-of-2) | **8.25** | OCP MX |
| FP4 (dense) | E2M1 (16 values) | — | not practically usable | 4 | theoretical only |
| **MXFP4** | E2M1 | 32 | FP8 E8M0 | **4.25** | OCP MX |
| **NVFP4** | E2M1 | **16** | FP8 **E4M3** (with mantissa) | **4.5** | NVIDIA variant |

Per-dimension comparison:

| Dimension | Dense FP wins | MXFP/NVFP wins | Note |
|---|---|---|---|
| Precision | — | ✓ (almost all LLM workloads) | LLM weights/activations naturally have channel-wise outliers; per-block scale local adaptation is far better than per-channel; for synthetic uniformly-distributed data the two are equivalent |
| Throughput (SM100 native) | — | ≈ tied | Both are a single tcgen05.mma instruction; microscaled reads one extra block scale tile (SMEM), bandwidth overhead < 1% |
| Storage | ✓ | — | 0.25-0.5 bit/element of block scale metadata extra; negligible compared to the 12-bit FP16 → FP4 savings |
| Hardware support | ✓ (SM89 / SM90) | — (SM100 only) | SM89 does not support MXFP at all; SM90 partially supports mxfp8 but mxf4nvf4 still starts at SM100; on older GPUs MXFP runs in software emulation and is extremely slow |
| Software ecosystem | ✓ | — (~2 years behind) | Dense FP8 has been a default path in PyTorch / vLLM / TRT-LLM since 2022; MXFP/NVFP serialization and model hub compatibility are still being added |
| Standard fragmentation | neutral | slightly worse | NVFP4 (NVIDIA, block=16) vs MXFP4 (OCP, block=32) are incompatible; cross-vendor deployment requires the MX standard |

**Trade-off for this project (LLM PTQ + SM100 + outlier-dominated weights/activations)**: on all important dimensions, microscaled at least ties dense FP and on most dimensions is better; the only cost (0.25-0.5 bit/element + software maturity) is negligible for our workload. **So PRIMARY selects microscaled, with no reason to retain a dense f8f6f4 fallback.** This is a natural consequence of locking the project to SM100 — once SM89/SM90 hardware compatibility is given up, the last constraint forcing "must use dense FP" also disappears.

The only side effect that β-1 must handle explicitly: **the current ResQ rotation R was trained under the INT per-channel scale model**; after switching to per-block scale, the quantization noise distribution changes and R must be retrained (cannot be carried over directly). In theory, per-block is a refinement of per-channel (more information), so the optimal R is at least no worse than the old one; but R retraining is not free and goes into β-1 workload.

**Why both phases use microscaled atoms (key design decision):**

The ResQ algorithm uses per-group quantization on o_proj (groupsize=head_dim=64), with an independent scale per (row, group) and per (column, group). This feature is a dead-end in the INT path — a single scale-per-row epilogue cannot express per-group accumulation, so in the original ResQ implementation o_proj has always been skipped by weight_packer and left as fake quant.

If we choose `kind::f8f6f4` (dense FP8, no microscale) for the high path, o_proj would have the same problem on the high path: FP8 is also single-scale-per-row × single-scale-per-column.

**Key insight: FP microscaling is finer-grained than ResQ per-group; the goal is NOT to "absorb" per-group — instead, per-group becomes redundant under microscaling.**

| Quantization granularity (K direction) | INT path | FP microscaled path |
|---|---|---|
| Common layers (q/k/v/gate/up/down) | per-channel weight + per-token activation | per-block (MXFP8=32, NVFP4=16) |
| o_proj special case | **per-group=64** (captures per-head statistical differences) | **per-block (MXFP8=32, NVFP4=16)** — finer than per-head |

NVFP4 block=16 is 4× finer than ResQ groupsize=64; **microscaling blocks provide strictly finer scale granularity than per-group**. In other words, per-group is a coarser version of microscaling; it exists in the INT path because per-block is impossible there; the FP path goes directly to the finer per-block, no need for per-group.

More importantly: keeping per-group + 1/8 ratio is **fundamentally infeasible alignment-wise**. groupsize=64 with high_fraction=1/8 means 8 high channels + 56 low channels per head, but MXFP8 instruction K extent = 32, so the per-head 8-channel high segment **cannot fill one instruction**, and the 56-channel low segment is not a multiple of 16 or 64.

**Companion algorithm-side change: upgrade o_proj input PCA from per-head to full hidden_dim global**. In current ResQ, o_proj input uses per-head 64×64 PCA (block-diagonal U); this is the only special case where q/k/v/gate/up all use hidden_dim 2048×2048 global PCA. The original choice of per-head was for structural symmetry with R2 (head_dim per-head rotation, an internal attention constraint), **but R2 acts on V and U acts on the o_proj input — they are two independent rotations at different points in the data flow and can be chosen independently**. After switching to global, three things naturally hold:

- **No head structure to rearrange** — after global variance sorting, directly slice top-256 high / bottom-1792 low; K_high=256 divides 32 ✓, K_low=1792 divides 16 ✓
- **Magnitude homogeneity within an MXFP8 block-of-32** — the global top 256 is a monotonically decreasing variance sequence, the magnitudes of 32 adjacent channels are nearly the same, block scale utilization is maxed; no longer the per-head splitting case where "4 heads' high segments are packed into one block, magnitude spans many ×"
- **Mathematically strictly no worse than per-head** — per-head is the block-diagonal subset of O(2048), global is the full manifold; top-K variance capture global ≥ per-head (tied when there is no cross-head correlation, strictly better when there is); so INT path PPL is expected to be unchanged or slightly improved (**should not regress**)

R2 stays per-head, attention internal computation flow does not change. Cost: `basis.py` adds an `o_proj_pca: full_global` mode; the PCA matrix grows from 32×(64×64) to 1×(2048×2048), one-time PTQ-time cost (seconds-scale); fused into the o_proj weight at PTQ → zero inference overhead. R must be retrained against the new U (the original R was the optimum for per-head U).

Risks and fallback path: although global ≥ per-head mathematically, whether PPL stays even/slightly-improved must be verified by the β-1 INT-path smoke test (drop the PCA mode change, keep the quantizer on INT, measure INT W4A4 PPL); expected within ±0.1. If a regression > 0.3 is measured, **suspect implementation bug first, not design choice**; only if multiple debug rounds fail to identify the bug, trigger Alt-4 fallback (revert to per-head PCA + cross-head rearrange — the original ResQ design path).

**File-level map for algorithm-side changes (ResQ → ResQ-FP):**

| File | Change | Nature |
|---|---|---|
| `promix/quantize/basis.py` | Add `o_proj_pca: full_global` mode (default per_head for backward compatibility); hidden_dim PCA path unchanged | Modify one section |
| `promix/quantize/rotation.py` | **Unchanged** — rotation fusion is format-independent | Reuse |
| `promix/quantize/fuse_norm.py` | **Unchanged** — RMSNorm fusion is format-independent | Reuse |
| `promix/quantize/hadamard.py` | **Unchanged** | Reuse |
| `promix/models/loader.py` | **Unchanged** | Reuse |
| `promix/quantize/quant_utils.py` | `ActQuantizer` adds `bits=mxfp8` and `bits=nvfp4` paths; legacy INT path retained (coexist) | Extend |
| `promix/quantize/optimize_rotation.py` | Quant noise simulation switches from INT round to FP cast (mxfp8 / nvfp4) | Modify one section |
| `promix/quantize/gptq.py` | Round-to-nearest changed to FP nearest-representable (Hessian terms unchanged) | Modify one section |
| `promix/inference/quant_ops.py` | Add `quantize_activation_mxfp8_per_token`, `quantize_activation_nvfp4_per_token` (output includes block_scale tensor); legacy INT pack/shift functions retained | Add new functions |
| `promix/inference/weight_packer.py` | Add MXFP8 + NVFP4 packing path; **o_proj follows the same code path as q/k/v** (no per-group structure after global PCA); remove the existing `groupsize > 0: continue` skip; legacy INT packer retained | Add new branch |
| `promix/inference/real_forward.py` | Add FP branch; dequant formula simplified to `out = s_x_token · s_w_channel · acc` (FP path has no shift/zero/colsum terms; o_proj has no group concept and shares the dequant with other layers); **o_proj no longer skipped**, follows the same path as other layers | Add new branch |
| `kernels/mixed_gemm_l20/` | **Retained as correctness oracle**, **no longer the real inference path** | Frozen |
| `kernels/mixed_gemm_sm100/` (new) | dual-phase `mxf8f6f4 + mxf4nvf4`; TMEM shared accumulator + EVT epilogue | New |
| `promix/configs/llama-3.2-{1b,3b,8b}-mxfp8-nvfp4.yaml` (new) | high_bits=mxfp8, low_bits=nvfp4 markers; works with β-4 per-op variable ratio | New |

### Phase Plan

**β-0 Investigation + minimal PoC kernel (Phase A in M0)**
- Read CUTLASS 4.5 SM100 examples (especially `examples/61_hopper_gemm_with_topk_and_softmax/` and the SM100 collection)
- Confirm NVFP4 block scale memory layout (interleaved within tile vs separate buffer)
- Write ~50-80 LOC minimal kernel: M=128, N=128, single-phase FP8 GEMM; goal is just to compile + run; **does not connect to ResQ data, does not handle dequant**
- Same for a minimal NVFP4 single-phase kernel
- **Critical: use the two minimal kernels to assemble a 2-launch FP baseline** (Python sequencing of mxfp8_gemm() + nvfp4_gemm() + FP32 accumulation); run a single point (M=128, N=2048, K_h=256, K_l=1792) to measure latency
- Acceptance (a): both single-phase kernels compile + cosine ≥ 0.9999
- Acceptance (b): each single-phase kernel runs at ≥ 1.5× FP16 single-point speedup (note: revised in plan body to ≥ 0.9× minimum because single-phase isn't expected to outperform; the original spec is documented here for context)
- Acceptance (c): **2-launch FP baseline latency is recorded as the comparator that β-2 fused must beat** — the purpose of testing this early: if 2-launch is already close to the FP16 cuBLAS performance ceiling, β-2 fused will struggle to open up a meaningful gap even if written (PLAQuant single-kernel claim risk, see Known Risks). This data determines whether β-2 needs aggressive tile-shape-level tuning or can ship a simple version

**β-1 Algorithm sanity check**
- Add MXFP8 (E4M3 + 32-element FP8 block scale) + NVFP4 (E2M1 + 16-element FP8 block scale) fake quantizers in `promix/quantize/quant_utils.py` (PyTorch implementation; no kernel needed)
- Add `o_proj_pca: full_global` mode to `promix/quantize/basis.py` (see "Companion algorithm-side change" in PRIMARY)
- Create new config `promix/configs/llama-3.2-1b-mxfp8-nvfp4.yaml` (high_bits=mxfp8, low_bits=nvfp4 markers, o_proj_pca=full_global)
- **Re-run 1B Step 0** to generate new U/E bin (global PCA on o_proj input); the q/k/v/gate/up hidden_dim PCA part is identical to the old file
- Step 1: run rotation optimization (quant noise simulated with MXFP8/NVFP4)
- Step 2: fake quant PPL evaluation, **o_proj follows the same microscaled path as other layers (no longer skipped)**
- **Acceptance (a-pre): INT-path smoke test of o_proj global PCA** — without changing the quantizer, only swap basis.py's o_proj PCA from per-head to full_global; run INT W4A4 PPL and compare against the ResQ baseline 11.70 (1B); **expect within ±0.1**. This isolates "PCA mode change" from "FP quant change" — combined regression cannot be bisected. On failure (PPL > baseline + 0.3), **always suspect implementation bug first, not the design** (mathematically global PCA is the strict superset of per-head and should not regress); only after multiple debug rounds without identifying a bug, trigger Alt-4 fallback (per-head + cross-head rearrange)
- Acceptance (a): **all layers (including o_proj) pass microscaled fake quant**; PPL does not regress vs current INT4+INT8 W4A4 baseline — **1B ≤ 11.70**, **3B ≤ 8.61**, **8B ≤ 6.99** (KV4 config: 1B ≤ 11.96 / 3B ≤ 8.70 / 8B ≤ 7.04). Note: the threshold is "non-regression" not "+0.5 buffer" — float noise allows ±0.01; > +0.05 is considered regression
- Acceptance (b): **swap o_proj alone back to fake-int per-group**; layer-wise output cosine of microscaled o_proj vs INT per-group ≥ 0.99 — proves NVFP4 microscaling is not significantly worse than INT per-group
- Acceptance (c): repeat (a-pre)(a)(b) on 3B and 8B — **all three models must pass for β-1 to be considered passed**; "1B passed but 3B not measured" is not acceptable

**β-2 SM100 fused kernel**
- Create new `kernels/mixed_gemm_sm100/`, mirroring `kernels/mixed_gemm_l20/`
- Use CUTLASS 4.5 CollectiveBuilder for the dual-phase kernel:
  - Phase 1: `SM100_MMA_MXF8F6F4_SS`, FP8 E4M3 × FP8 E4M3 → FP32 (TMEM C)
  - Phase 2: `SM100_MMA_MXF4NVF4_SS`, NVFP4 × NVFP4 → FP32 (same TMEM C, scaleC=true accumulation)
  - Shared KernelTmaWarpSpecialized schedule + TMA descriptors
- TileShape starting point: 128×128×{32,64} (high K = 32, low K = 64; single-instruction K extent)
- ThreadblockShape: 128×128 (M=128, N=128 occupies the 1-CTA cluster ceiling)
- pybind11 expose `fused_fp8_nvf4_gemm(A_high_fp8, B_high_fp8, A_low_nvf4, B_low_nvf4, scales_low)`
- **Also implement TWO 2-launch FP baseline variants as comparators** — both share one standalone `mxfp8_gemm()` + one standalone `nvfp4_gemm()` with GEMM2 beta=1 accumulation; difference is timing harness:
  - **Variant E (Eager)**: invoked from Python in eager mode (no CUDA Graph) — deployment-realistic
  - **Variant G (CUDA Graph)**: same two GEMMs wrapped in a pre-recorded CUDA Graph — Python launch overhead OUT of timing window; paper-defensible topology-only comparator
  - **Both must be written**; the AC-5 PLAQuant claim requires fused to win on BOTH (compare with `baseline_cutlass_mixed_gemm` in `kernels/mixed_gemm_l20/mixed_gemm_l20.cu` which plays a similar role for the INT path)
- Write `kernels/mixed_gemm_sm100/benchmark.py`, methodology mirroring `kernels/mixed_gemm_l20/benchmark.py`: **four-column** output per row (fused / eager 2-launch / CUDA Graph 2-launch / FP16 cuBLAS), sweeping the (M, N, K_high, K_low) matrix:
  - **M sweep**: 1, 16, 64, 128, 256, 512, 1024, 2048, 4096 (full batch / context-length spectrum)
  - **N sweep**: 2048, 4096, 8192 (q/k/v vs gate/up_proj output dimensions)
  - **K_high/K_low ratio sweep**: high ratios 1/16, 1/8 (default), 1/4, 1/2 (matching β-4 per-op variable ratio search domain)
  - **down_proj shapes**: (M, 2048, 1024, 7168), since K_total is large and sensitivity differs
  
- Verify bit-exact (cosine ≥ 0.9999) vs reference FP32 dual-phase accumulation
- Acceptance (a): **correctness** — every row in the entire shape sweep cosine ≥ 0.9999
- Acceptance (b): **vs FP16 cuBLAS shape-sweep characterization** — output per-shape speedup table + heatmap; require fused/FP16 ≥ 1.5× at M=128, ≥ 2× at M=2048 on the "LLM inference real-hit shape" subset (i.e., the (M, N, K_high, K_low) tuples for 1B/3B/8B layers, common batch ∈ {1/16/64}); other shape regressions are allowed but must be **explicitly recorded in the characterization report**
- Acceptance (c): **vs 2-launch FP shape-sweep characterization for BOTH variants** — separate per-shape speedup tables for fused/eager-2-launch AND fused/CUDA-Graph-2-launch; **this is the most central criterion** — directly quantifies the gain of single-kernel topology over sequential 2-launch, **not depending on any FP16 comparison**. Require fused > 1.0× on LLM-hit shapes for BOTH variants (HARD); target ≥ 1.10× geomean for eager (launch reduction + topology) and ≥ 1.05× geomean for CUDA Graph (topology only). If fused loses on Variant G across all LLM shapes, the topology-only PLAQuant claim has failed; if it loses on Variant E too, the deployment-level claim also fails — both indicate kernel design needs revisiting (tile shape, SMEM allocation, warp specialization)
- **Deliverable**: `kernels/mixed_gemm_sm100/RESULTS.md` containing: (1) shape-sweep four-column latency table (fused / eager 2-launch / CUDA Graph 2-launch / FP16); (2) three speedup heatmaps (fused/FP16, fused/eager-2-launch, fused/CUDA-Graph-2-launch); (3) which shape regimes fused wins / loses for each variant; (4) per-stage profile (TMA load / WGMMA compute / accumulate / store proportions) for performance attribution

**β-3 End-to-end integration + speed verification**
- `promix/inference/weight_packer.py` adds FP8/NVFP4 weight packing path
- `promix/inference/quant_ops.py` adds FP8/NVFP4 activation per-token quant (FP8 doesn't need shift/zero; NVFP4 needs block scale computation)
- `promix/inference/real_forward.py` adds FP-path branch
- EVT epilogue folds dequant into the kernel (using the SM90 EVT pattern; see `kernels/mixed_gemm/mixed_gemm.cu` lines 49-92)
- Run 1B real-FP end-to-end PPL eval (matching fake-FP PPL, and ≤ 11.70)
- Run 1B real-FP end-to-end latency with three comparators: vs FP16 cuBLAS, vs 2-launch FP path (using β-2's standalone mxfp8_gemm + nvfp4_gemm sequenced from Python), vs existing fake-quant baseline; output all three datasets
- **Acceptance (1B)**: PPL ≤ 11.70 ✓, latency vs FP16 ≥ 1.0× ✓, latency vs 2-launch FP > 1.0× ✓
- **3B and 8B repeated end-to-end**: same three datasets per model; all three models must pass for β-3 to be considered passed; "only 1B passed" is not acceptance
- If end-to-end performance fails but β-2 micro shape sweep shows the kernel itself wins, **trigger attribution investigation**: is Python-side quant pack overhead consuming the kernel gain (→ EVT epilogue fuse at the end of β-3)? Or do the layer shapes mismatch the LLM-shape subset of the micro sweep (→ rerun sweep adding missing shapes)? Project is not considered complete before the cause is localized

**β-4 Algorithm-side extension: per-operator variable high/low ratio (after β-3, advanceable independently)**

The current ResQ paper applies a uniform `high_fraction = 1/8` (i.e., K_high : K_low = 1 : 7) to every operator. This ratio was chosen for simplicity and is not optimal for each operator — within the same model, attention's q/k/v have different sensitivity to quantization noise than MLP's gate/up/down; down_proj, with its larger intermediate_size and longer accumulation path, is often more sensitive than the attention projections. A fixed 1/8 is conservative on sensitive operators and wasteful on insensitive ones.

β-4 promotes high_fraction from a model-level constant to a **per-Linear independent knob**, with the search space bounded by two hardware constraints:
- K_high must be a **multiple of 32** (FP8 `kind::f8f6f4` single-instruction K extent = 32 elements)
- K_low must be a **multiple of 64** (NVFP4 `kind::mxf4nvf4` single-instruction K extent = 64 elements; ≥ block scale vector size = 16 is already covered by the 64 multiple)
- K_high + K_low = layer.K_total (fixed)

Example feasible (K_high, K_low) pairs for Llama-3.2-1B q_proj (K_total = 2048):
- (64, 1984), (64, 1920), (128, 1920), (128, 1856) ... up to (2048, 0)
- I.e., K_high ∈ {64, 128, 192, ..., 2048} ∩ {x : x % 32 == 0 ∧ (2048 - x) % 64 == 0}

Work items:
- **Refactor config schema**: allow yaml's `quantize.high_fraction: 0.125` (scalar) to alternatively be `quantize.high_fraction_per_layer: {q_proj: 0.10, k_proj: 0.10, v_proj: 0.10, o_proj: 0.20, gate_proj: 0.05, up_proj: 0.05, down_proj: 0.15}` (dict form, with backward-compatible scalar broadcast)
- **Per-layer quantizer and weight packer adaptation**: existing ActQuantWrapper is already instantiated per-layer; passing different high_fraction does not require architecture change
- **Search strategy (candidates)**:
  - Static heuristic: fixed by operator type (attention vs MLP; down_proj high, others low)
  - Variance cumulative threshold: pick the top-k channels with cumulative variance ≥ τ from the PCA basis eigenvalue spectrum as high; choose k independently per layer (subject to hardware alignment round-up)
  - Learnable / grid search: with a fixed total bit budget, search the per-layer high_fraction combination minimizing PPL (expensive but precise)
- **Kernel does not need to change**: the β-2 SM100 kernel is already parameterized by (M, N, K_high, K_low) (see test_cases in `kernels/mixed_gemm_l20/benchmark.py`, which already enumerates different K_high/K_low combinations); β-4 just passes different K configs to different layer instances
- **Acceptance**: prove on 1B/3B/8B that per-op variable ratio at minimum doesn't regress PPL vs fixed 1/8, ideally improves it slightly (< -0.2 PPL); average bit-width does not increase (i.e., not trading PPL by piling up high; instead reallocating high quota from insensitive layers to sensitive layers)

This step expands the project from "PLAQuant system layer + ResQ fixed algorithm" to "PLAQuant system layer + ResQ-Adaptive algorithm"; an independent publishable algorithmic contribution, not dependent on SM100 native (also applies on the SM80 INT4+INT8 kernel — just with the K-alignment constraint becoming INT4 K%64=0 / INT8 K%32=0).

### Objective Evidence

- `kernels/mixed_gemm_l20/mixed_gemm_l20.cu` (571 LOC, SM80 PLAQuant kernel): architecture preserved as topology template for the SM100 port; the Phase split / shared accumulator design migrates directly.
- `kernels/mixed_gemm/mixed_gemm.cu` (410 LOC, SM90 CollectiveBuilder + WGMMA + EVT): CUTLASS 3.x pattern reference; the SM100 collective form just swaps the atom on this base.
- `include/cute/atom/mma_traits_sm100.hpp`: full SM100 atom enumeration (21 atom families); `SM100_MMA_MXF8F6F4_*` and `SM100_MMA_MXF4NVF4_*` are the two PLAQuant-SM100 dual-phase directly uses.
- `include/cute/arch/mma_sm100_umma.hpp` lines 993-1197: PTX literals for `tcgen05.mma.kind::i8 / f16 / tf32 / f8f6f4 / mxf8f6f4 / mxf4nvf4`; confirms INT4 does not exist while FP8/NVFP4 native exist.
- Atom shape constraints (directly from `mma_traits_sm100.hpp`): M ∈ {64, 128}, N stride 8 (B K-major) or 16 (B MN-major) in [8, 256], K derived from 256/sizeof_bits<ValTypeA> (FP8 K=32, FP4 K=64).
- B20Z micro measurements: FP16 cuBLAS peak 1434 TFLOPS (= 5th-gen TC at work); current SM80 fused 58 TFLOPS (25× gap); theoretical SM100 FP8 ~3 PFLOPS, NVFP4 ~6 PFLOPS.
- `promix/quantize/basis.py`: PCA basis computation is data-format-agnostic; the q/k/v/gate/up hidden_dim PCA is directly reusable by ResQ-FP; the o_proj part must switch to global mode and be recomputed (see "Companion algorithm-side change: o_proj input PCA upgrade" in PRIMARY, ~20-45 min/model one-time cost).
- `promix/configs/llama-3.2-1b-resq.yaml`, `llama-3.2-1b-w4a4kv4.yaml`, etc.: existing yaml defines `high_bits` / `low_bits` / `high_fraction` parameters; ResQ-FP only needs to extend the enum to allow FP8 / NVFP4 markers; the schema does not change.
- `kernels/mixed_gemm_l20/setup.py` already contains `-gencode arch=compute_100,code=sm_100` (commit f9afacd); new `kernels/mixed_gemm_sm100/setup.py` uses the same template.
- `tests/test_mixed_gemm.py`: existing INT-path correctness tests; writing the FP counterparts for ResQ-FP is mechanical.

### Known Risks

- **NVFP4 quantization PPL risk**: NVFP4 has larger numeric range than INT4 but fewer mantissa bits (FP4 E2M1 has only 1 mantissa bit); ResQ's PCA + rotation + variance-based split was designed for INT and whether it remains effective on FP4 must be verified by β-1. If 1B fake-FP PPL regresses > 1, design-level adjustments are needed (raise high_fraction? switch to FP6+FP4?)
- **o_proj global PCA implementation risk**: mathematically global PCA is strictly no worse than per-head (the block-diagonal subset of full O(2048)), but implementation must handle: covariance matrix going from 64×64 to 2048×2048 (memory and numerical stability), eigenvectors sorted by variance, correct composition with R2 (per-head, internal to attention). If β-1 (a-pre) measures INT-path global PCA PPL regressing > 0.3 vs per-head, **it is almost certainly a bug, not a design failure** — sanity tools: layer-wise output cosine per-head vs global, per-channel variance scatter plot, U×R composition unitarity check. Only after multiple debug rounds without finding a bug, trigger Alt-4 fallback (revert to per-head + cross-head rearrange).
- ~~MXFP8 E8M0 scale precision risk~~: previously concerned E8M0 (power-of-2 only) scale would lose precision on outlier-heavy blocks, but **ResQ's PCA + Stiefel R + (down_proj) Hadamard three-layer rotation specifically scatters single-channel outliers into per-block-smooth Gaussian-like distributions** — when entering GEMM, the worst case "31 small values + 1 outlier 100 in one block" does not exist. E8M0 is sufficient on ResQ-rotated data (per-element FP8 mantissa fine-tunes within the ±√2 range; the inter-block optimal scale is also close to a power-of-2). **ResQ's rotation pre-conditioning + microscaling's per-block scale adaptation are a natural pairing** — risk withdrawn. Residual caveat: if overall β-1 PPL fails, diagnosis is still needed, but E8M0 is no longer the primary suspect.
- **TMEM learning curve**: tcgen05's TMEM programming model (descriptor addressing, warp specialization, persistent kernel) is completely different from SM80's register fragments; CUTLASS 4.5 abstracts some away but many low-level details remain; β-0 minimal PoC pre-pays this learning cost.
- **CUTLASS 4.5 SM100 documentation sparsity**: SM100 examples on the CUTLASS GitHub are far fewer than SM80/SM90; reading traits / collective source directly is required to understand the API. But `mma_sm100.hpp` / `mma_traits_sm100.hpp` comments are relatively complete.
- **Mixed FP8+NVFP4 single-kernel public examples are nearly zero**: CUTLASS samples include standalone FP8 GEMM and standalone NVFP4 GEMM, but **a mixed kernel where both phases share a TMEM accumulator has no off-the-shelf reference** — this is exactly the PLAQuant paper claim; meaning we need to assemble it ourselves.
- **NVFP4 block scale memory bandwidth and layout**: every 16 FP4 elements carries one FP8 scale; the layout of the scale tensor relative to data tensor (interleaved vs separate) at weight packing affects TMA descriptor design; β-0 PoC must resolve this.
- **Algorithm-side retraining cost**: rotation R was trained against INT loss; must be refit against FP loss; 1B ~1 min per run, 3B ~5 min, 8B ~10 min; multi-experiment cost is small but a budget is required.
- **2-launch FP may already be close to the single-kernel ceiling (PLAQuant paper claim risk)**: SM100's TMA + warp specialization + large SMEM make the overhead between two independent FP launches much lower than the SM80 era. If micro sweep shows fused only 1.01-1.04× faster than 2-launch, **the quantitative evidence for the PLAQuant single-kernel system contribution is weak** — paper-writable but the claim is not hard. Mitigation: in the β-0 minimal PoC, **also write the 2-launch FP baseline** (not only fused); measure fused vs 2-launch at a single point early; if that comparison is already < 1.05×, adjust tile shape / SMEM allocation / warp specialization strategy of the fused kernel ahead of time to open up a gap. If even after maximum tuning it remains < 1.05×, trigger the Alt-1 (path α) fallback to at least preserve the deliverable.
- **B20Z idle reaper**: the cluster does not check GPU activity but portal page activity (we have hit this); β-2 kernel build + benchmark spans multiple sessions; portal must stay open + `bring_up_remote.sh` recovery script is required.

## Alternative Directions Considered

### Alt-1: Path α — Pure W8A8 SM100 Native (INT4 sign-extend → INT8)

- Gist: do not change the algorithm; run the existing ResQ INT4+INT8 natively on SM100 — INT8 path uses `tcgen05.mma.kind::i8` directly; INT4 path sign-extends 4-bit data to INT8 and goes through `kind::i8` (the same approach as on SM90 where INT4 native is also missing). The result is effectively W8A8 (the INT4 2× compression vanishes), but the PTQ pipeline doesn't change at all, and a ~2× FP16 speedup is plausible quickly. Treated as "PRIMARY's fallback if it fails" or "short-term data path".
- Objective Evidence:
  - SM90 git history commits `987d3c9`, `5992d48`: confirm SM90 also lacks INT4 native; sign-extend to INT8 is a known engineering practice
  - `SM100_MMA_S8_*` atom family is fully available (4 sub-variants)
  - The existing PTQ pipeline does not change at all; existing 1B/3B/8B PPL data carries over directly
  - Drawback: PLAQuant's "single-kernel multi-precision" degenerates to "single-kernel single-precision" under α; no system-level paper contribution, just using SM100 native
- Why not primary: usable as fallback but not a worthwhile goal — keeping ResQ's INT8/INT4 split but having no INT4 channel in hardware effectively splits and rejoins the data; not a new system and not a new algorithm.

### Alt-2: Profile-Driven End-to-End Bottleneck Map

- Gist: instrument every stage of `promix/inference/real_forward.py` with `torch.cuda.Event` and output per-layer + per-stage latency-share JSON report. Under PRIMARY this is phase-acceptance infrastructure — after β-1 fake quant, see which layer's PPL regressed the most; after β-3 end-to-end, see where the remaining gap is. Does not produce speedup itself.
- Objective Evidence:
  - `promix/inference/benchmark.py` lines 17-37: existing `measure_latency` helper using CUDA events, but only measures total end-to-end duration
  - `tests/bench_h20_peak.py` lines 24-77: cold/hot CUDA event pattern
  - `promix/inference/real_forward.py` lines 57-143: 10 stages with no timing checkpoints
  - `promix/inference/quant_ops.py` lines 6-38, 51-64: 4096 calls per forward, zero instrumentation
- Why not primary: does not solve the core problem of "the kernel itself can't beat FP16". Use it inside PRIMARY as a phase-acceptance tool.

### Alt-3: Path γ — Defer SM100, Validate PLAQuant on H100/L40 with Native INT4

- Gist: leave the existing INT4+INT8 kernel alone; find hardware that still has INT4 native MMA (A100 SM80, L40 SM89, H100 SM90 all have `mma.sync.m16n8k64.s4s4.s32`); run micro + end-to-end to prove that PLAQuant design does achieve speedup on appropriate hardware. Treat B20Z as next-gen prototype for β. Short term, the paper can claim "PLAQuant achieves X× FP16 speedup on Hopper/Ada"; Blackwell is filled in later.
- Objective Evidence:
  - `kernels/mixed_gemm_l20/mixed_gemm_l20.cu`: the existing SM80 kernel already supports this
  - DEVLOG history 1B PPL=14.72 + 1.12-1.19× speedup were measured on H20 (SM90a) and have not been re-measured on H100/L40
  - Switching the remote container from H20 to B20Z is an infrastructure constraint; locally, the 8× L20 (SM89) lacks nvcc and cannot compile kernels; γ requires switching/borrowing H100/L40 compute
- Why not primary: a research strategy, not a technical direction; does not solve the long-term Blackwell problem; just kicks the can.

### Alt-4: Fallbacks If o_proj's Two-Layer Assumptions Fail

- Gist: PRIMARY makes two "use-the-alt-only-if-the-assumption-fails" dependencies on o_proj: (1) input PCA upgrades from per-head to global, assuming INT path PPL stays flat or improves slightly; (2) NVFP4 microscaled (block=16) replaces ResQ per-group=64, assumed ≥ per-group. Each assumption has its own fallback:
  - **β-1 (a-pre) fails** (global PCA regresses INT path > 0.3, and debug rules out a bug): revert to **per-head PCA + cross-head rearrange** (the original ResQ approach) — 8 hi + 56 lo per head, concat as [all-heads hi (256) | all-heads lo (1792)], K aligned ✓; cost: MXFP8 block-of-32 spans 4 heads sharing one scale; NVFP4 block=16 mostly within-head but every 7th block crosses a head boundary. R1+R2 rotation smoothing inter-head statistics is the precondition for this fallback to stand.
  - **β-1 (b) fails** (NVFP4 microscaled o_proj layer-wise cosine deviates from INT per-group > 0.01): fallback options (a) raise o_proj's high_fraction (β-4 per-op variable ratio provides this knob, from 1/8 to 1/4 or 1/2); (b) add per-group scale indexing in the epilogue for group-aware dequant; (c) accept o_proj reverting to fake quant as future work.
  - The two fallbacks are independent and can be triggered separately; simultaneous fail probability is extremely low, but each must be handled in flow.
- Objective Evidence:
  - `promix/quantize/basis.py`: o_proj currently uses per-head 64×64 PCA; q/k/v/gate/up use hidden_dim 2048×2048 global PCA; switching o_proj to global only adds a new branch, hidden_dim path is reused directly
  - `promix/inference/weight_packer.py` lines 34-36: `if wrapper.quantizer.groupsize > 0: continue` — explicit skip in legacy INT path; this skip is removed once PRIMARY is complete
  - `promix/inference/real_forward.py` line 149: comment `"Skips layers with per-group quantization (o_proj) — they keep fake quant."` — same as above, becomes obsolete after PRIMARY
  - `inference/forward_pass.py` lines 78-86, 215-226 (legacy path): ResQ per-group activation quant implementation, groupsize=head_dim=64; this logic can serve as a reference for cross-head rearrange design when fallback (a-pre) is triggered
  - `tests/test_mixed_gemm.py` `o_proj_dir()` fixture: full test data available for β-1 acceptance point (b) cosine comparison
- Why not primary: PRIMARY makes two "global PCA + microscaled subsumption" assumptions on o_proj to minimize complexity (no need for cross-head rearrange, no group-aware kernel); these alts are contingencies, **enabled only when the corresponding β-1 acceptance fails**.

### Alt-5: Custom Op + FX Graph Rewriter Integration

- Gist: register the kernel as a `torch.library.custom_op` + use FX graph rewriting to automatically replace `nn.Linear`, eliminating the `install_real_forward()` lambda monkey-patching. An architecture / maintainability improvement that lets any HF model benefit automatically. Under PRIMARY this is post-β-3 code-quality polish, not on the critical path.
- Objective Evidence:
  - `promix/inference/real_forward.py` lines 146-164: current lambda pattern
  - Only 2 call sites (`real_forward.py:159` and `benchmark.py:181`)
  - `kernels/mixed_gemm_l20/setup.py` lines 9-31: pybind11 already exposes `torch::Tensor` signatures, fully compatible with `torch.library`
  - No existing `torch.library` / `torch.fx` usage anywhere in the repo — entirely new infra
- Why not primary: does not solve the perf problem. Wrapping a slow kernel in a pretty op API is pointless; add the wrapper after PRIMARY achieves speedup.

## Synthesis Notes

After completing the missing micro benchmark vs FP16 cuBLAS and verifying the CUTLASS 4.5 SM100 atom catalog, the problem was relocated twice: first from "Python overhead dominates" to "the kernel itself is 25× slower"; second from "SM100 INT4+INT8 native upgrade" to "SM100 has no INT4, must migrate to FP8+NVFP4". The final PRIMARY (PLAQuant-SM100 FP8+NVFP4 single-kernel) preserves the project's two core contributions — the system layer's single-kernel multi-precision topology and the algorithmic layer's PCA + variance-split + learnable rotation — while only swapping the concrete-implementation layer of "INT4+INT8 atom" for the SM100-equivalent "FP8+NVFP4 atom". The M, N alignment (the 8×8 tile-sharing principle in SM80 PLAQuant) is naturally inherited on SM100 to the TMEM accumulator level.

The phase pipeline β-0 → β-1 → β-2 → β-3 has hard ordering: if the algorithm doesn't work (β-1 PPL regression), no time should be spent on the kernel (β-2); if the kernel doesn't work (β-2 vs FP16 < 1×), no time should be spent on integration (β-3). Each phase has a quantifiable go/no-go acceptance point, avoiding "betting everything on the end and only finding out at the end that the road was wrong".

Alt-1 (α) is the safety net for when PRIMARY truly hits a wall (preserve deliverable, scale back the paper claim). Alt-2 (profile) is a phase acceptance tool inside PRIMARY (can be wired in at the end of β-1 to attribute PPL regression to specific layers and guide tuning). Alt-3 (γ) is a parallel research strategy (run the existing INT version on H100/L40 for paper data; doesn't affect the Blackwell main line). Alt-4 (o_proj) and Alt-5 (FX) are coverage and code-quality polish after PRIMARY runs end-to-end; not on the critical path.

If only one thing can be started right now, it is β-0: write two minimal SM100 single-phase kernels (one FP8, one NVFP4) and run a micro vs FP16 cuBLAS, turning "SM100 native theoretical ~2-4× FP16 speedup" from theory into measurement. This step has very low cost (no ResQ data, no dequant, ~80 LOC each) but determines whether all subsequent phases are worth doing — if even the minimal FP8 kernel cannot deliver 1.5× FP16, the entire β path needs to be reassessed.

--- Original Design Draft End ---
