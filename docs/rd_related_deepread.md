# Deep-Read Competitive Analysis: the W4A4 / FP4 / Mixed-Precision Cluster

**Scope.** Full-paper reads (method + experiments + tables, not abstracts) of the eight works
closest to ProMix that were NOT yet deep-read (ARCQuant was already covered by the controller),
plus full confirmation reads of the two local PDFs (Oltron, SHMQ). Sources: arXiv full text via
MCP (versioned IDs noted per section) and the two local PDFs under `~/Downloads/`.

**ProMix reference point** (see `docs/rd_related_work.md` §B.0): ResQ-lineage W4A4(+KV4), PCA
basis + rotation, error-driven per-ring (64-channel) high/low partition, **NVFP4 (low) +
MXFP8 (high)**, **single-launch dual-format fused GEMM** with shared FP32 accumulator on real
SM120 (RTX PRO 6000), vLLM-targeted. Llama-3.1-8B wiki gap-to-own-FP16 **+0.46** at ~4.5 avg
bits (ARCQuant, same protocol class: +0.63).

**Extraction template per paper:** (1) activation bit-width (the key filter: true W4A4?);
(2) numeric format; (3) precision-assignment granularity; (4) rotation/PCA/Hadamard;
(5) kernel reality (fused dual-format GPU kernel vs segmented GEMM vs ASIC vs none);
(6) accuracy as gap-to-its-own-FP16 on shared models; (7) speedup + hardware; (8) precise
ProMix differentiation both ways.

**Cross-paper caveat.** Gap-to-own-FP16 is the only number comparable across papers, and even
that is soft: context length, eval harness, and KV treatment differ (e.g., MosaicQuant reports
LLaMA3-8B FP16 wiki = 8.6 where most papers report ~6.2; FGMP uses Wikitext-103, not -2).
Gaps below always cite the paper's own FP16 row.

---

## 1. MicroMix (arXiv 2508.02343v2) — THE closest "mixed microscaling + Blackwell kernel" rival

- **Authors/venue:** Liu, Meng, Luo, Zhao, Zhang (Tianjin U.) — same group as ARCQuant and SharQ. Code public.
- **(1) Activation bits:** mixed per channel — MXFP4 / MXFP6 / MXFP8, weights quantized to the
  *same* bit-width as their activation channel (§3.1). FP4 channels dominate (p4 > 50%,
  Fig. 3), but **average ~5.2–5.5 bits/element** (Table 1: 5.51 for Llama3.1-8B, 5.22 for
  Qwen2.5-32B). So: W-mixed/A-mixed with 4-bit majority, NOT iso-bit with W4A4 methods.
- **(2) Format:** OCP MX family, block 32, E8M0 scale: MXFP4 E2M1, MXFP6 E3M2, MXFP8 E4M3
  (§4.1; format-variant ablation Table 5 shows E/M split barely matters).
- **(3) Granularity:** per-channel, three groups per layer with **layer-adaptive proportions**
  p4/p6/p8. Criterion: closed-form quantization-error thresholds T(4), T(6) derived so MXFP4/6
  error stays below the INT8 error upper bound (Def. 1, Eq. 6–8); channels sorted by absolute
  mean (Eq. 9) and partitioned offline from calibration data (proportions are stable across
  datasets/batches, Fig. 3). Motivation 1 explicitly argues against Atom-style fixed
  high-precision counts — i.e., they already own "layer-adaptive channel ratios".
- **(4) Rotation:** **none.** Pure channel reorder (Atom/RPTQ style), fused into the
  quantization kernel. No Hadamard, no PCA, outliers are isolated, not Gaussianized.
- **(5) Kernel:** co-designed Blackwell kernel suite, but **segmented, not fused-dual-format**:
  "matrices of specific data types invoke their corresponding GEMM kernels … The process is
  highly decoupled" (§3.2) — i.e., per layer the three precision groups run as separate
  MXFP-GEMM invocations (CUTLASS-based, dequant fused into MMA), plus a fused
  reorder-and-quantize kernel (<20% of kernel time, Table 6). Their custom GEMM beats CUTLASS
  by 2.6–4.0× (W4A4) at small M (Table 7) — note this is vs an unoptimized CUTLASS baseline at
  small shapes.
- **(6) Accuracy (gap to own FP16, Table 1):** Llama3.1-8B wiki **6.72 vs 6.24 = +0.48 @
  5.51 avg bits**; Qwen2.5-32B 5.56 vs 5.02 = +0.54 @ 5.22 bits (paper text says "0.46",
  arithmetic says 0.54). Zero-shot ≥98% of FP16 on both; MMLU 62.65 vs 65.24 (8B). Beats
  QuaRot/Atom/FlatQuant/AMXFP4 rows at those budgets.
- **(7) Speedup/HW:** RTX 5070Ti laptop 2.45–2.93× and RTX 5090 2.29–3.38× vs TensorRT-FP16
  (up to 1.74× vs TRT-FP8); **RTX PRO 6000** (our exact GPU): decode throughput 1.82–3.02× vs
  Atom, prefill −85% vs Atom/QuaRot (Fig. 7); end-to-end ~2× vs HF FP16.
- **(8) ProMix differentiation:**
  - *What we do that they don't:* true low-bit budget (ours ~4.5 avg bits vs their 5.2–5.5 —
    at 8B we get a comparable wiki gap, +0.46 vs +0.48, for ~1 bit/element less); PCA+rotation
    error shaping (their channels keep raw outlier structure, which is why they must spend FP6/FP8
    on ≥40% of channels); **single-launch** dual-format GEMM with shared accumulator vs their
    3-kernel segmented path; error-driven per-ring allocation measured on output error, vs their
    per-element error-threshold heuristic on abs-mean-sorted channels; NVFP4 (FP8-scaled, finer
    g16) for the 4-bit bulk vs coarser MXFP4 (E8M0, g32).
  - *What they do that we don't:* a third precision level (FP6); demonstrated 32B-scale and
    code/math/MoE (Mixtral) evals; TensorRT-grade kernel baselines; KV-cache quantization table.
  - *Scoop risk:* **HIGH, confirmed** — they own the "mixed MX formats + Blackwell kernel +
    layer-adaptive channel allocation" narrative and publish RTX PRO 6000 numbers. Bounded by:
    no rotation, no single-launch fusion, ≥0.7 bit higher budget, error-threshold (not
    output-error) allocation. Must-compare head-to-head at iso-bits; emphasize bits/accuracy
    Pareto, kernel-launch structure, and PCA-rotated partition.

## 2. FGMP (arXiv 2504.14152v1, NVIDIA + UC Berkeley) — same NVFP4+FP8 pairing, ASIC-only (confirmed)

- **(1) Activation bits:** yes, weights AND activations, mixed NVFP4/FP8 per block; evaluated
  at 70/80/90% of blocks in FP4 (effective ~4.4–5.2 bits).
- **(2) Format:** **NVFP4** (g16, E2M1 + FP8 E4M3 scale) low; **FP8 (no microscaling)** high.
  Identical pairing-in-spirit to ProMix's NVFP4+MXFP8.
- **(3) Granularity:** 1-D blocks of 16 along the dot-product dim, **unstructured** interleave
  of FP4/FP8 blocks in both W and A (Fig. 2b); one metadata bit per block. Assignment =
  Fisher-information-weighted perturbation score I'_L(v) = Σ g_i²(Δ_{ph→pl}v_i)² (Eq. 8), with
  a **single global threshold across all layers** (Eq. 10) so sensitive layers automatically get
  more FP8 blocks (Fig. 7); plus sensitivity-weighted per-block scale clipping for weights
  (§3.3). Activations classified **on the fly** by a hardware post-processing unit using
  calibrated per-channel Fisher (§4.2).
- **(4) Rotation:** none.
- **(5) Kernel:** **NO GPU kernel — ASIC, confirmed.** VMAC datapath with FP4×FP4, FP8×FP8,
  FP4×FP8, FP8×FP4 dot-product units + mixed-precision activation-quantization PPU; SystemC/HLS,
  synthesized in a 5 nm process at 1 GHz (§4.3, Table 4: FGMP datapath 10356 µm², 3.5× area of
  FP8-only). The paper itself states software implementations of dynamic per-block precision
  selection "may be impractical" (§1) — i.e., the authors concede this does not run on GPUs.
- **(6) Accuracy:** Llama-2-7B Wikitext-103: <1% PPL degradation vs an all-FP8 baseline with
  70% FP4 blocks; lm-eval avg 0.656 @70% FP4 = exactly BF16's 0.656; 90% FP4 = 0.647 (Table 3).
  MMLU: 58–89% less degradation than FP8→FP4 (Table 2). Weight-only FP4+SW-clip: 5.13 vs BF16
  5.06 (Table 1). Models: Llama-2 7B/13B, GPT3 1.3B/8.3B/22B, Nemotron-4 15B (no Llama-3/Qwen).
- **(7) Efficiency:** energy, not speed: 14% datapath energy saving vs FP8 with <1% PPL loss;
  NVFP4 datapath −33% energy vs FP8; 30% weight-memory saving @70% FP4. No GPU, no latency.
- **(8) ProMix differentiation:**
  - *Us not them:* runs on **shipping SM120 silicon** — our structured 64-ch ring partition is
    exactly the restructuring that turns their unstructured per-16-block idea into tensor-core
    executable tiles with one kernel launch; PCA+rotation before partition; MXFP8 (block-scaled)
    high format vs their scalar FP8; modern models + end-to-end serving.
  - *Them not us:* Fisher (loss-gradient) weighted criterion with a *global cross-layer
    threshold* — this is prior art for "error/sensitivity-driven cross-layer adaptive budgets",
    so ProMix's allocation claim must be phrased as *per-ring output-error-measured allocation
    executable on GPU tiles*, not "first sensitivity-driven budget"; dynamic per-block activation
    routing (we are static per-ring); hardware energy accounting.
  - *Scoop risk:* HIGH on format pairing + allocation concept, **nil on systems claim**
    (ASIC-only confirmed from §4 + §5.4). Must-cite-and-differentiate; not runnable as baseline.

## 3. CoQuant (arXiv 2604.26378v1) — joint weight–activation weighted-PCA subspace (beats ResQ)

- **(1) Activation bits:** W4A4+KV4 mixed: 1/8 of channels in 8-bit subspace → **4.5/4.5/4.5
  avg bits** (Table 1 header), exactly the ResQ protocol.
- **(2) Format:** **INT** (per-token asymmetric activations, per-channel symmetric weights,
  GPTQ for weights — §4.1). No FP4/NVFP4 anywhere.
- **(3) Granularity:** rank-r subspace (r = d/8), **fixed ratio across all layers** (admitted
  limitation: "we adopt a fixed high-precision subspace ratio across layers"). Criterion: the
  paper's core contribution — high-precision subspace = top-r eigenvectors of the **joint
  matrix M = λ_X Σ_X + λ_W Σ_W** (weighted sum of activation and weight second-moment
  matrices), derived in closed form from a first-order output-error model under isotropic
  quantization noise (Eq. 10–13); fused Σ_W across Q/K/V and gate/up so one shared basis per
  block (Eq. 14); extended to KV cache (V paired with W_O, K paired with Q).
- **(4) Rotation:** yes — ResQ-style: U_h = P_h R_h with random orthogonal rotations inside
  each subspace to Gaussianize; the AQNM noise assumption explicitly leans on this.
- **(5) Kernel:** **none.** Fake quant on a single A100; "dedicated kernel implementation is
  needed to fully validate practical inference speedups" (Limitations).
- **(6) Accuracy (Table 1, gap to own FP16):** Llama-3.2-1B wiki 11.60 vs 9.75 = **+1.85**
  (ResQ +2.28, QuaRot +4.64); Llama-3.2-3B 8.93 vs 7.81 = +1.12 (ResQ +1.37); Qwen2.5-7B 8.70
  vs 6.85 = **+1.85** (ResQ +2.10; QUIK/GPTQ explode); Qwen2.5-14B 6.48 vs 5.29 = +1.19. No
  Llama-3-8B. Ablation Table 2: activation-only (=ResQ) 12.03 / weight-only 13.23 / joint 11.60
  on 1B. At 3-bit+6-bit (avg 3.375) the margin grows: 28.71 vs ResQ 39.81 (Table 3). Layer-wise
  isolated k_proj error −30~60% vs ResQ (Fig. 2).
- **(7) Speedup/HW:** none.
- **(8) ProMix differentiation:**
  - *Us not them:* formats (NVFP4/MXFP8 vs INT — their INT gaps are 3–4× ours in absolute PPL
    class), any kernel at all, Blackwell deployment, per-layer-adaptive and per-ring (64-ch)
    error-driven budgets vs their one fixed global 1/8 subspace, larger-model + serving evidence.
  - *Them not us:* a *principled closed-form* basis-selection criterion that provably (under
    their noise model) beats activation-variance-only PCA — this attacks the exact axis "our
    subspace criterion is better than ResQ variance". If ProMix's ring ordering is still
    activation-variance/PCA-eigenvalue ordered, CoQuant's joint-covariance ordering is a direct,
    cheap upgrade they published first. KV-cache subspace treatment is also cleanly worked out.
  - *Scoop risk:* **HIGH on the allocation-criterion axis, zero on systems.** Action: (a)
    must-cite; (b) run the ablation "ring order by Σ_X eigenvalue vs by λ_XΣ_X+λ_WΣ_W vs by our
    measured output-error" — if our error-driven ranking ≥ joint-covariance ranking, CoQuant
    becomes a supporting citation rather than a threat; (c) emphasize ours is *measured* error
    per ring (no isotropic-noise assumption) and feeds a cross-layer budget, which CoQuant
    explicitly lacks.

## 4. dMX (arXiv 2606.04115v2, AMD) — differentiable per-layer MXFP format assignment

- **(1) Activation bits:** W+A quantized; each layer gets ONE format (weights and activations
  tied: "tying weights and activations bit-widths reduces the problem to a single learnable
  scalar per layer", §2.1). MXFP4 layers are truly A4, MXFP8/6 layers are A8/A6.
- **(2) Format:** OCP MXFP4/6/8; continuous E(2+β)M(1+β) parameterization of the format, β∈[0,2]
  learned per layer, temperature-annealed sigmoid discretization (Eq. 8), target-average-bit
  regularizer (Eq. 10).
- **(3) Granularity:** **per-layer** — the coarsest in this cluster. Gradient-based (SpinQuant
  infra, 3200 FineWeb samples, 400 steps), jointly optimizes rotations and bit-widths; beats a
  KL-divergence greedy layer selector at intermediate budgets (Fig. 5).
- **(4) Rotation:** yes — SpinQuant-style learned rotations trained jointly.
- **(5) Kernel:** **none.** Brevitas fake quant; no latency numbers at all.
- **(6) Accuracy (Table 1/2, gap to own BF16):** Llama-3.2-1B: dMX @4.57 avg bits wiki 11.02 vs
  8.94 = **+2.08** (uniform MXFP4 +2.74); @5.11 +1.66; @6.04 +0.89. SmolLM2/Qwen3-1.7B similar.
  8B-scale only as Appendix-D plots (no table numbers in text). Useful finding (Appendix E,
  Tables 3–7): **down_proj and v_proj are the layers most consistently kept high-precision**
  across all models — independent corroboration of our layer-sensitivity structure.
- **(7) Speedup/HW:** none.
- **(8) ProMix differentiation:* we are intra-layer (64-ch rings) + dual-format within one GEMM;
  dMX cannot express "7/8 of a layer in FP4, 1/8 in FP8" at all — a layer is all-FP4 or all-FP8.
  Gaps at matched budgets are far above ours. They do bring: gradient-based *global* budget
  allocation capturing cross-layer error interaction (our per-ring greedy/measured allocation
  could inherit this as future work), and the down_proj/v_proj sensitivity tables to cite.
  *Scoop risk:* LOW. Must-cite as the per-layer-granularity representative + for sensitivity
  structure.

## 5. MosaicQuant (arXiv 2606.15652v1) — unified INT4 with sparse 4-bit residual + fused kernel

- **(1) Activation bits:** true **W4A4** (also W4A8 track), INT; activations dense 4-bit after
  outliers are migrated into weights (Eq. 3–4); **no high-precision channels anywhere** —
  anti-mixed-precision stance (like ARCQuant, but on the weight side): dense 4-bit base +
  **sparse 4-bit residual** blocks compensating weight quantization error (10% block budget ≈
  +0.4 bit weight-side).
- **(2) Format:** INT4 (GPU INT4 tensor cores).
- **(3) Granularity:** weight **blocks aligned to GEMM tiles**; selection via activation-aware
  Δ-Hessian score S_B = Σ h_j R_ij² − Σ h_j(R−Q(R))², h_j = (2/N)ΣX²_nj (Alg. 1), provably
  optimal top-K under diagonal Hessian (Thm 4.1). Blocks beat channels (ablation: channel-level
  compensation 13.1 PPL vs block-level 10.8 on LLaMA3-8B, §5.4). 10% budget recovers 55–63% of
  output distortion (Obs. 3).
- **(4) Rotation:** equivalent-transformation **outlier migration** (SmoothQuant-style scaling
  framing; they compare against, and beat, QuaRot/SpinQuant rather than building on Hadamard).
- **(5) Kernel:** **yes — ZipperEngine**: fuses the sparse 4-bit residual blocks INTO the dense
  INT4 GEMM (shared activation tiles, accumulate into same output tiles), overlapped
  quant/dense/sparse/reduce pipeline; removes the naive +30% separate-sparse-kernel overhead
  (1.9× kernel-level, Fig. 3). Same-format dense+sparse fusion, NOT dual-format. **RTX 4090**
  (Ada, INT4) — not Blackwell, not FP4.
- **(6) Accuracy (Table 1 W4A4, gap to own W16A16 — NB their FP16 wiki numbers are high,
  protocol differs):** LLaMA3-8B 10.8 vs 8.6 = **+2.2** (SpinQuant +2.8, ResQ +3.0, QuaRot
  +4.5); Qwen3-8B 11.2 vs 9.7 = +1.5 (ResQ +4.5!); Qwen3-32B 8.8 vs 7.6 = +1.2. 0-shot losses
  1.3–3.1 pp. Sparse branch costs 8.5–9.5% memory, 5.3–6.2% latency.
- **(7) Speedup/HW:** up to **1.24× vs W16A16** end-to-end; 1.11–1.20× vs ResQ throughput on
  Qwen3-8B (RTX 4090, TensorRT-LLM baseline in fig).
- **(8) ProMix differentiation:* INT vs our FP4 formats; Ada vs Blackwell; weight-side-only
  compensation (activations get NO protection beyond migration — hence gaps 3–5× ours in the
  same wiki-gap metric, protocol caveat applies); their fused dense+sparse single-pipeline
  kernel is the closest *kernel-philosophy* relative to our single-launch dual-format GEMM —
  cite it as the INT/sparse analogue. They beat ResQ/Atom throughput by integrating the extra
  path into the GEMM, which independently validates our core systems thesis (fallback paths
  must be fused into the main GEMM to pay off). *Scoop risk:* MEDIUM-LOW (format, HW, and
  precision structure all differ; the "fused compensation kernel" narrative overlaps).

## 6. SharQ (arXiv 2606.26587v1) — activation N:M sparsity + FP4, same group as ARCQuant/MicroMix

- **(1) Activation bits:** **all-FP4 W4A4** — no high-precision anything. Activations
  decomposed ONLINE into a 4:8 semi-structured sparse FP4 "backbone" (outliers) + dense FP4
  residual defined relative to the *quantized* backbone (Eq. 8), so the dense path compensates
  both mask loss and sparse-path quant error. Training-free AND calibration-free.
- **(2) Format:** NVFP4 primary (4:8 in-pairs sparsity); generalizes to HiF4 (2:4) and MXFP4
  (Table 4).
- **(3) Granularity:** element-wise dynamic N:M mask per activation — this is **sparsity
  routing, not precision allocation**; weights stay uniform FP4 (single shared payload with
  path-specific scale views: sparse g32 scales + duplicated dense g16 scales).
- **(4) Rotation:** none (cites Shao et al. 2025 that global transforms hurt NVFP4 block
  statistics — same anti-rotation stance as ARCQuant).
- **(5) Kernel:** yes, serious engineering on Blackwell: fused preparation kernel
  (RMSNorm + mask-gen + quant + sparse compression + residual construction in one operator);
  sparse NVFP4 GEMM + dense NVFP4 GEMM **accumulating into the sparse output** (no separate
  merge). That's **two GEMM launches + one fused prep** per linear — total GEMM work ≈
  0.7 (sparse) + 1.0 (dense) = **~1.7× the FLOP-time of plain dense NVFP4** (their Fig. 4b:
  sparse = 0.66–0.72× dense; prefill overhead +11.2% vs NVFP4). vLLM integration.
- **(6) Accuracy (Table 1, gap to own FP16):** Llama-3.1-8B wiki **6.73 vs 6.25 = +0.48**
  (plain NVFP4 +0.69 → recovers ~30% of wiki gap; 56% of the 0-shot gap: 71.55 vs 72.50, NVFP4
  70.32); MMLU 63.76 vs 65.24. Qwen2.5-7B 7.15 vs 6.85 = +0.30 (NVFP4 +0.44). Also MoE
  (Qwen3-30B-A3B), VLM (Qwen3-VL-8B), video (Wan2.2). Ablation Table 5: dense+dense 6.63,
  SharQ 6.73, sparse-only 82.95.
- **(7) Speedup/HW:** RTX 5090 + vLLM: 2.2–2.4× latency vs FP16, **1.2–1.4× throughput vs
  FP8**; 4×5090 video gen 1.23×.
- **(8) ProMix differentiation:* mechanism is orthogonal (outlier *sparsity routing* vs our
  *subspace precision partition*); it spends ~1.7× dense-FP4 compute to recover the gap where
  we spend ~1.125× (one 1/8-K MXFP8 ring inside the same launch); our 8B wiki gap +0.46 beats
  their +0.48 (and our 0-shot story should be checked head-to-head); we use rotation+PCA, they
  are transform-free. They beat us on: zero calibration, MoE/VLM/video breadth, N:M sparse
  tensor-core utilization (a hardware path we don't touch), and a shipped vLLM path on 5090.
  *Scoop risk:* MEDIUM — same lab's "recover the NVFP4 gap on Blackwell with a clever second
  path + fused prep kernel" pitch, adjacent headline numbers; mechanism clearly distinct.
  **Must-compare** (code public, Blackwell-native, same eval suite).

## 7. DualQuant (arXiv 2610.13767v1) — calibration-free dual-scale W4A4 across INT4/MX/NVFP4

- **(1) Activation bits:** true W4A4, uniform (no mixed precision at all).
- **(2) Format:** INT4 (G=64), MXINT4, MXFP4 (G=32), NVFP4 (G=16) — "first dual-scaling method
  to formally support MX and NV block formats" (scales projected onto format-admissible sets).
- **(3) Granularity:** per-row (α) + per-column (β) weight scales via block coordinate descent
  on weight-reconstruction MSE with closed-form updates and convergence proof (Alg. 1, Thm 1);
  the column scale β doubles as a **weight-derived activation per-channel rescaling**
  (SmoothQuant-like, but with zero calibration data; Pearson +0.83~0.97 vs SmoothQuant λ=0
  scales, Table 5). No channel/bit selection anywhere.
- **(4) Rotation:** none — and importantly, **adversarial evidence against rotation on NVFP4**:
  "QuaRot … diverges on NVFP4" (PPL >1000, Table 1) because "Hadamard rotation redistributes
  activation magnitude in a way that collapses NVFP4's narrow-range E4M3 block scale" (§5.1,
  Table 2 caption). SINQ+ also diverges on NVFP4.
- **(5) Kernel:** real packed **W4A4 INT4 (G=64) path on RTX 4090** built on the
  Nunchaku/SVDQuant GEMM backend; β applied inside the activation-quant kernel (4.3–6.1% of GPU
  time); packed PPL 6.74 vs fake-quant 6.73 (±0.01). **No FP4/Blackwell kernel** — NVFP4/MX
  results are fake-quant on GH200.
- **(6) Accuracy (Table 1, gap to own BF16, NVFP4 rows):** Llama-3.1-8B wiki **6.83 vs 5.62 =
  +1.21** (best calibration-free; SmoothQuant +1.70, RTN +3.42); Llama-3.2-1B 10.91 vs 8.67 =
  +2.24; Llama-2-7B 5.54 vs 5.11 = +0.43. NVFP4 0-shot 8B: beats even SmoothQuant on all five
  tasks (Table 2). MXFP4 8B +1.73.
- **(7) Speedup/HW:** RTX 4090 INT4: 1.39–2.46× vs BF16, memory 2.51–2.61×.
- **(8) ProMix differentiation:* no mixed precision, no rotation, no Blackwell FP4 kernel;
  accuracy a full PPL-class behind mixed-precision methods (+1.21 vs our +0.46 at 8B). Value to
  us: (a) strong *calibration-free uniform-NVFP4* baseline row; (b) their dual-scale weight
  treatment is composable with ProMix (could pre-condition our NVFP4 rings); (c) **their
  QuaRot-diverges-on-NVFP4 result is the sharpest published version of the anti-rotation
  critique** — ProMix must explicitly state why our rotation survives NVFP4 (rotation confined
  within variance-sorted PCA subspaces; block scales fitted post-rotation; empirically our gap
  beats all transform-free NVFP4 methods). *Scoop risk:* LOW; must-cite.

## 8. Oltron (DAC'24, local PDF 3649329.3656221.pdf) — ASIC, fp8/12/16 outlier channels (controller's read CONFIRMED)

- **(1) Activation bits:** activations mixed: normal channels INT4, salient channels fp8/fp12
  /fp16 chosen per layer by an adaptive budget search (Alg. 1, 3σ init, MSE-greedy); avg ~4.1
  bits in the headline config. Weights INT4 via GPTQ.
- **(2) Format:** INT4 + scalar fp8/fp12(/fp16) — **no block scaling, no NVFP4/MX**.
- **(3) Granularity:** channel; TOBE (Tile-wise Outlier-Balanced Encoding) reorders salient
  channels evenly across tiles for regular memory access; inter-layer ratios adaptive.
- **(4) Rotation:** none (reordering only; compile-time reorder elimination through the graph).
- **(5) Kernel:** **ASIC, definitively** — weight-stationary systolic array, hybrid
  efficient/flexible PEs (flexible PE = int4×fp8-E5M2 via shifter; two PEs pair up for fp12),
  reconfigurable decoders; Verilog RTL synthesized with Synopsys DC on **TSMC-28nm**, CACTI
  memories, cycle-level simulator (§5.1). Baselines are OliVe/OLAccel accelerators, not GPUs.
- **(6) Accuracy (Table 2, gap to own FP16):** LLaMA-7B wiki 11.67 vs 5.68 = **+5.99** at A4.1
  avg (pre-rotation-era quality); LLaMA-65B 5.82 vs 3.53 = +2.29; OPT models near-lossless-ish
  (OPT-30B +0.95). Beats OmniQuant/OliVe rows.
- **(7) Perf:** 1.9×/3.6× speedup and 1.6×/1.9× energy vs OliVe/OLAccel (iso-area, simulated).
- **(8) Verdict (unchanged from controller):** shares only the abstract shape "inter+intra-layer
  adaptive channel-granular mixed precision + tile-balanced reorder + HW co-design". Not FP4,
  not block-scaled, not GPU, not rotation, accuracy not competitive. **Cite-only** as the ASIC
  ancestor of channel-granular mixed precision; zero scoop risk.

## 9. SHMQ (EMNLP 2025 Industry, local PDF 2025.emnlp-industry.175.pdf) — static INT W4.8A8 (controller's read CONFIRMED)

- **(1) Activation bits:** **A8 always** (INT8, static per-group g=128 symmetric). Headline
  config "W4.8A8" = W4A8 + 20% of weights W8A8. **Not W4A4** — fails the key filter.
- **(2) Format:** INT only. No FP4/MX/NVFP4.
- **(3) Granularity:** hierarchical: inter-layer W8A8 ratio via Fisher-approximated layer
  sensitivity (Eq. 7–9), intra-layer channel selection via OBS-style Hessian sensitivity +
  Manhattan norm (Eq. 10–12), then **decoupled identification/permutation** — sensitive and
  insensitive clusters each re-sorted by magnitude so static per-group scales see flat
  distributions (§3.2.3, Fig. 4).
- **(4) Rotation:** none (permutation only; explicitly positions against dynamic quant, and
  cites rotation works as the other family).
- **(5) Kernel:** layer-wise GPU mixed W4A8/W8A8 GEMM timing (Table 3: 1.83–4.21× vs FP16,
  QUIK-style split) + **edge NPU deployment** (Novatek NT98690, 2.77–2.81× vs INT16). No fused
  dual-format anything; static-scale efficiency is the whole point.
- **(6) Accuracy (Table 1, gap to own FP16):** near-lossless *at A8*: LLaMA2-7B wiki 5.58 vs
  5.47 = +0.11; LLaMA3.1-8B 6.60 vs 6.24 = +0.36; Qwen2.5-7B-I 0-shot 75.58 vs 75.71 = −0.13 pp.
  Beats dynamic QUIK/Atom and static MixLLM at W4.8A8.
- **(7) Speedup:** above; representative-layer, not end-to-end serving.
- **(8) Verdict (unchanged):** different problem (static INT8-activation edge deployment).
  Overlaps only in Hessian-ish hierarchical allocation + permutation vocabulary. **Cite-only**;
  zero scoop risk for a W4A4-FP4-Blackwell paper.

---

## Summary matrix (key filter columns)

| Work | A bits (real) | Format | Assign granularity | Rotation/PCA | Kernel reality | 8B-class wiki gap (own FP16) | HW | Scoop risk |
|---|---|---|---|---|---|---|---|---|
| **ProMix (ours)** | 4 (7/8 K) + 8 (1/8 K) | NVFP4 + MXFP8 | 64-ch rings, error-driven, per-layer budgets | PCA + rotation | **single-launch dual-format fused GEMM** | **+0.46** (L3.1-8B) | RTX PRO 6000 (SM120) | — |
| ARCQuant | 4 (+residual ch, FP4) | NVFP4 only | top-S outlier chans → residual K-append | no (anti) | standard CUTLASS NVFP4 (K+S) | +0.63 | 5090 / PRO 6000 | HIGH (done) |
| MicroMix | mixed 4/6/8 per ch (avg 5.2–5.5) | MXFP4/6/8 | per-channel, layer-adaptive ratios (err thresholds) | no (reorder) | **segmented 3-GEMM** + fused reorder-quant | +0.48 @5.51b | 5070Ti/5090/**PRO 6000** | **HIGH** |
| FGMP | mixed 4/8 per 16-block | NVFP4 + FP8 | per-block, Fisher, global threshold | no | **ASIC 5nm** (no GPU) | ~0 vs FP8 @70%FP4 (L2-7B, wiki-103) | ASIC sim | HIGH concept / nil systems |
| CoQuant | 4 + 8 (1/8 subspace) | INT | rank-d/8 subspace, joint W+A weighted PCA, fixed ratio | **yes (ResQ-style)** | none (fake quant) | n/a (Qwen2.5-7B +1.85) | A100 | **HIGH (criterion axis)** |
| dMX | per-layer 4 or 6/8 | MXFP4/6/8 | **per-layer**, learned (differentiable+anneal) | yes (SpinQuant) | none | 1B +2.08 @4.57b | none | LOW |
| MosaicQuant | 4 (dense, migrated) | INT4 | weight blocks (Δ-Hessian top-K, 10%) | migration (scaling) | fused dense+sparse INT4 GEMM (ZipperEngine) | +2.2 (L3-8B, own protocol) | RTX 4090 | MED-LOW |
| SharQ | 4 (sparse FP4 + dense FP4 paths) | NVFP4/HiF4/MXFP4 | element N:M mask (dynamic), no precision mixing | no (anti) | sparse GEMM + dense GEMM (accumulate) + fused prep; ~1.7× dense-FP4 work | **+0.48** (L3.1-8B) | **RTX 5090 + vLLM** | MED |
| DualQuant | 4 uniform | INT4/MXINT4/MXFP4/NVFP4 | per-row+col scales only (no mixing) | no (QuaRot **diverges** on NVFP4) | packed INT4 on 4090; FP4 = fake quant | +1.21 (NVFP4 L3.1-8B) | 4090 / GH200 | LOW |
| Oltron | mixed int4 + fp8/12/16 chans | INT4 + scalar fp | channel, TOBE tiles, adaptive ratios | no | **ASIC TSMC-28nm** | +5.99 (L1-7B) | ASIC sim | nil |
| SHMQ | **8** (static INT8) | INT | layer ratio + channel (Hessian), permute | no | W4A8/W8A8 split GEMM; edge NPU | +0.36 at **A8** | GPU layer-wise + NT98690 | nil |

## ProMix defensible novelty after full reads

After reading every rival in full, **no paper combines (i) true ~4.5-bit W4A4 activations+weights,
(ii) a PCA-rotated subspace partition, (iii) NVFP4-low + MXFP8-high heterogeneous block-scaled
formats, and (iv) a single-launch dual-format fused GEMM with a shared FP32 accumulator on
shipping Blackwell silicon** — and none matches our 8B gap-to-FP16 (+0.46) at our bit budget.
The two components that ARE individually claimed elsewhere and must be positioned carefully:
(a) *layer-adaptive, error/sensitivity-driven precision budgets* — MicroMix (error thresholds,
per-channel), FGMP (Fisher, per-block, global cross-layer threshold), SHMQ/Oltron (Hessian,
inter+intra-layer) all predate us at the concept level, so our allocation claim must be scoped
as "per-64-ch-ring allocation driven by *measured output error*, co-designed so rings map 1:1
onto tensor-core tile boundaries of one fused kernel"; (b) *a better-than-variance subspace
criterion* — CoQuant's joint weight–activation weighted PCA directly occupies this ground with
a closed-form solution that beats ResQ; our counter is that error-driven ring ranking subsumes
covariance heuristics (needs the ablation in §3 above to be in the paper). On the kernel side
the field splits into segmented multi-GEMM (MicroMix), same-format dense+sparse fusion
(MosaicQuant, SharQ), standard-GEMM reformulations (ARCQuant, DualQuant), and ASIC (FGMP,
Oltron) — the single-launch *dual-format* (two scale granularities, two element formats, one
accumulator) GEMM remains uniquely ours and is also the direct rebuttal to FGMP's "impractical
in software" concession. Finally, DualQuant's published QuaRot-diverges-on-NVFP4 result and
ARCQuant/SharQ's anti-rotation stance make a short "why rotation inside PCA subspaces survives
block-scaled FP4" subsection mandatory.

## Ranked must-compare list (runnable W4A4-on-Blackwell head-to-head first)

1. **MicroMix** — must-compare #1. Public code, native Blackwell kernels, publishes RTX PRO
   6000 numbers (our GPU). Compare: accuracy at iso-average-bits, and kernel latency
   single-launch vs their segmented 3-GEMM at production shapes. (Known local blocker: CUDA-13
   validation failure in earlier smoke — fix or report.)
2. **ARCQuant** — already smoke-PASSED (controller); the uniform-NVFP4+residual-channels
   accuracy/latency baseline. Keep.
3. **SharQ** — must-compare #3. Public code, RTX 5090 + vLLM, NVFP4 W4A4, same eval harness;
   head-to-head on Llama-3.1-8B gap (+0.48 vs +0.46) and on compute cost of the recovery path
   (their sparse+dense ≈1.7× dense-FP4 work vs our 1.125×).
4. **DualQuant** — accuracy-baseline row (calibration-free uniform NVFP4, fake-quant
   reproducible); their packed kernel is 4090/INT4 so no Blackwell kernel comparison; cite the
   QuaRot-on-NVFP4 divergence and rebut.
5. **MosaicQuant** — INT4/Ada; include as the "unified 4-bit, weight-side compensation, fused
   kernel" accuracy row if cheap (code availability unverified); otherwise cite.
6. **CoQuant** — not a systems baseline (fake-quant INT), but **mandatory ablation**: ring/basis
   ordering by activation PCA vs joint weighted PCA vs our measured output error, same budget.
7. **dMX** — optional accuracy row for the per-layer-granularity straw man; cite for
   down_proj/v_proj sensitivity corroboration.
8. **FGMP** — cite-and-differentiate only (ASIC, 5nm synthesis; no runnable artifact).
9. **Oltron / SHMQ** — cite-only (ASIC INT / static A8 INT); zero head-to-head value.
