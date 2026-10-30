# Related Work & Baseline Readiness (rd-related)

Date: 2026-10-30. Hardware context: 8× NVIDIA RTX PRO 6000 Blackwell Server (SM120, cc 12.0),
CUDA 13.0, container `luofan-plaquant` (torch 2.11.0+cu130). Baselines workspace:
`/home/luofan/baselines` (vLLM venv, peers, logs, results — see
`docs/baselines_on_hardware.md` for round-1 measured numbers).

Two parts:
- **Part A** — runnability inventory: which W4A4 / activation-quant peers CAN run on our SM120
  box (no benchmarks run here; inputs/sizes not yet fixed).
- **Part B** — related-work completeness survey (2024–2026), grouped, with ProMix
  differentiation and scoop-risk verdicts. Built by snowballing: seed list → ResQ forward
  citations (`docs/rd_related_data/resq_forward_citations.md`, 38 quant-relevant citers) →
  related-work + baseline lists of each close paper → survey-hub taxonomies. Hops stopped when
  new papers stopped adding W4A4 / mixed-precision-allocation works (converged after ~2 hops).

---

## Part A — Peer-baseline runnability on SM120 (RTX PRO 6000)

### A.1 Comparison policy (user directive)

We do **NOT** integrate peers into vLLM and we do **NOT** port/optimize peer code. Each peer is
recorded at its **highest-performance available implementation that runs on SM120 as shipped**:

1. If the method is vLLM-native → use vLLM's implementation (same-strong-stack, fairest).
2. Else → the original repo's fastest path, as shipped (minimal compat patches only).
3. If nothing runs efficiently on Blackwell → that **is** the method's honest deployment
   ceiling on this hardware, recorded as-is — not a gap we fix.

The only method WE integrate into vLLM is ProMix itself. The headline head-to-head is
**ProMix-in-vLLM vs vLLM-NVFP4 (modelopt / compressed-tensors W4A4-FP4)** — both on the same
serving stack. The fairness split (same rule as our accuracy scoreboard): **W4A4
(activations quantized) vs W4A16 (weight-only) are different classes and are never mixed in a
peer comparison** — W4A16 rows are context only.

### A.2 vLLM 0.32.0 native quantization inventory

Install: `/home/luofan/baselines/venv` (uv, py3.12, **inside container** `luofan-plaquant`;
the venv's python symlinks resolve only in-container — use
`docker exec -i luofan-plaquant /home/luofan/baselines/venv/bin/python`). vLLM 0.32.0 cu130
wheel, torch 2.13.0+cu130. SM120 verified end-to-end in round 1 (BF16/FP8/GPTQ/AWQ/NVFP4 all
ran, no fallbacks).

Registry (`QUANTIZATION_METHODS`, 30 entries):
`auto_awq, auto_gptq, awq, awq_marlin, compressed-tensors, deepseek_v4_fp8, experts_int8,
fbgemm_fp8, fp8, fp8_per_block, fp8_per_channel, fp8_per_tensor, fp_quant, gpt_oss_mxfp4,
gptq, gptq_marlin, humming, inc, int8_per_channel_weight_only, modelopt, modelopt_fp4,
modelopt_mixed, modelopt_mxfp8, moe_wna16, mxfp4, mxfp8, nvfp4_per_token, online, quark,
torchao`

W4A4 / FP4 / mixed-precision relevant, first-class in vLLM:

| vLLM method | Format | Act-quant? | SM120 status (this box) |
|---|---|---|---|
| `modelopt_fp4` (ModelOptNvFp4Config) | **NVFP4 W4A4** (E2M1 g16, FP8-E4M3 scales) | **yes, A4 static tensor-group** | **WORKS, measured** — `FlashInferCutlassNvFp4LinearKernel`, JIT `fp4_gemm_cutlass_sm120`; full PPL/TTFT/decode table in baselines doc |
| `compressed-tensors` → `CompressedTensorsW4A4Fp4` | **NVFP4 W4A4** | **yes** | llm-compressor NVFP4 checkpoints load & run (round-1 nvfp4 runs used this route + FlashInfer kernel) |
| `compressed-tensors` → `W4A4Mxfp4`, `W4A8Fp8`, `W4A8Int`, `W8A8Mxfp8`, `W8A8Int8/Fp8` | MXFP4 W4A4; W4A8 (FP8 or INT8 act); W8A8 | yes | schemes present in 0.32.0; not yet exercised on this box (checkpoint needed per scheme) |
| `fp_quant` (FPQuantConfig) | **FP-Quant / Quartet: rotation-based MXFP4/NVFP4 W4A4** (QuTLASS kernels) | **yes** | registered, but `qutlass` **not installed** in the venv → blocker is one pip/source install (QuTLASS supports Blackwell). The ONLY rotation-based W4A4 method that is vLLM-native — high-value same-stack peer if we install it |
| `mxfp4` / `gpt_oss_mxfp4` | MXFP4 (gpt-oss W4, MoE) | partial | present; gpt-oss-specific |
| `modelopt` (FP8), `fp8*` | FP8 W8A8 | yes | WORKS, measured (CutlassFP8ScaledMM) |
| `modelopt_mixed` (ModelOptMixedPrecisionConfig) | ModelOpt mixed-precision checkpoints (per-layer formats) | depends on ckpt | present; relevant precedent: vLLM already has plumbing for per-layer mixed format checkpoints |
| `gptq`/`gptq_marlin`, `awq`/`awq_marlin` | INT4 **W4A16** | **no (A=BF16)** | WORKS, measured — Marlin. Context only (weight-only class) |
| `nvfp4_per_token`, `online` | online/dynamic quant config | yes | present (online quantization path), untested |

Not in vLLM (no integration exists in 0.32.0): **QuaRot, SpinQuant, Atom, QServe/QoQ, ResQ,
ARCQuant** (ARCQuant is upstreamed into **TensorRT-LLM** (PR #11333) and ships its own
vLLM integration in-repo — not in upstream vLLM 0.32), **MicroMix, FGMP, OffQ, KurTail,
WUSH** (WUSH/FP-Quant share QuTLASS lineage; only `fp_quant` is upstream).

### A.3 Peer smoke-test status (original repos, as shipped)

All four classic W4A4-family peers were already stood up under `/home/luofan/baselines/peers/`
(each with RESULTS.md, logs, patches; commits pinned there). Status, including this round's
re-checks:

- **QuaRot** (spcl/QuaRot @5008669): fake-quant accuracy path **runs** (W4A4KV4 PPL measured
  1B/3B/8B: 25.996 / 11.227 / 8.860; own-FP16 deltas +12.84/+3.41/+2.72). Real INT4 CUTLASS
  kernels **build and are bit-exact on SM120** after a one-line arch flag, but `s4` MMA does
  not exist on Blackwell — ptxas lowers to INT8 SASS; measured **~5–6× SLOWER than FP16**.
  Honest ceiling: accuracy-only; no competitive speed path on this hardware.
- **SpinQuant** (facebookresearch/SpinQuant @8f47aa3): **runs** (fake-quant only; repo has NO
  GPU kernels). W4A4KV4 Hadamard PPL 1B/3B/8B = 18.623/10.092/7.882; learned-rotation 1B
  16.143, 3B 9.246. Honest ceiling: no speed path at all (ExecuTorch mobile export only).
- **Atom** (efeslab/Atom @7e3618b): fake-quant accuracy **runs** after transformers-4.48
  patches. W4A4 (~4.2 bits, 128 INT8 keeper ch, KV4) PPL 1B/3B/8B = 18.400/9.699/**7.570**
  (8B completed after RESULTS.md was written; log `w4a4_8b.log`). Kernels: documented build
  fails (gcc-11 hardcode, CUDA-13 Thrust move, sm_86 hardcode); forced sm_120 build compiles
  and launches BUT SASS is `IMMA...S8` — the "INT4" path silently runs on INT8 tensor cores
  (same `s4` lowering as QuaRot). Honest ceiling: accuracy-only on Blackwell.
- **QServe/QoQ** (mit-han-lab omniserve @02b2925 + lmquant @69f3473): **W4A8KV4, not W4A4**
  (different class). Serving kernels do NOT build on CUDA 13: `cub::Max` removed (fused
  attention); core W4A8 GEMMs alone DO compile for sm_120. Fake-quant QoQ pipeline runs, but
  out-of-box 1B config produced degenerate PPL (tied-embedding pitfall). Honest ceiling:
  blocked as shipped on CUDA13/SM120.
- **ResQ** (reference implementation inside our repo): runs (our promix pipeline reproduces
  1B PPL 14.68 vs paper 14.72 on this box).

New peers surfaced by Part B worth a runnability row (minimal smoke attempted this round —
see A.4 and the addendum section at the end for outcomes):

- **ARCQuant** (github.com/actypedef/ARCQuant): NVFP4 W4A4; paper deploys on RTX 5090 **and
  RTX PRO 6000** (our exact GPU) with vLLM integration. Highest-priority new must-compare.
- **MicroMix** (github.com/lwy2020/MicroMix): MXFP4/6/8 channel-wise mixed precision with a
  co-designed Blackwell kernel (RTX 5070Ti/5090, sm_120 class).
- **WUSH** (github.com/IST-DASLab/WUSH): adaptive-transform W4A4 (MXFP4), fused GPU impl.
- **FP-Quant/QuTLASS**: vLLM-native (`fp_quant`) — blocker is only the missing `qutlass`
  package in the venv.

### A.4 Readiness matrix

Columns per the agreed framing: act-quant class first; "best impl" = highest-performance
available form we would use (no porting by us).

| Method | Class (A-quant?) | Best available impl | Builds on this box? | Smoke test | Honest perf ceiling on SM120 | Blocker |
|---|---|---|---|---|---|---|
| **vLLM NVFP4 (modelopt / ct-W4A4Fp4)** | **W4A4** (A4 NVFP4) | vLLM-native (FlashInfer CUTLASS sm120) | yes (wheel) | **PASS — full runs measured** (PPL+TTFT+decode, 1B/3B/8B) | Best prefill of all round-1 baselines (8B TTFT@8192 177ms vs BF16 392ms) | none — THE same-stack head-to-head |
| **FP-Quant (Quartet/QuTLASS)** | **W4A4** (MXFP4/NVFP4 + rotation) | vLLM-native (`fp_quant`) | venv lacks `qutlass` (PyPI `qutlass` is a 0.0.0 placeholder → SOURCE build from IST-DASLab/qutlass required; `fp-quant` 0.3.2 pip-installable) | NOT RUN | expected real FP4 TC speed (QuTLASS targets Blackwell) | one source build of qutlass (CUDA13 compat untested) + checkpoint |
| **ARCQuant** | **W4A4** (NVFP4 + residual channels) | original repo (own CUTLASS kernels; also upstreamed into TensorRT-LLM PR#11333) | **YES** — CUDA13, `CMAKE_CUDA_ARCHITECTURES 120a`, only fix: pass `-Dpybind11_DIR` | **PASS** — `bench_nvfp4` GEMM runs (0.146ms @ default shape, GPU6); `agemm.so` imports with full API (matmul, reorder_quantize_w/x, rmsnorm_quantize_x, i4 KV-cache ops) | paper: 2.0–2.5× vs FP16 prefill on PRO 6000; +3–9% over plain NVFP4 | none found — READY (accuracy eval additionally needs its requirements.txt env + model) |
| **MicroMix** | **W4A4~A6/A8 mixed** (MXFP4/6/8 channels) | original repo (Blackwell mgemm kernels) | **YES** — CUDA13, 120a, same pybind11 fix; `mixedgemm.so` + bench built | **PARTIAL FAIL** — kernels launch (97–112 TFLOPS reported) but built-in `--validate` reports GEMM VAL FAILED for 4x4, 6x6, 8x4 on CUDA 13 (repo pins CUDA 12.8) | paper: 2.29–3.38× vs TRT-FP16 on RTX 5090 | numerics wrong under CUDA 13 as shipped — honest ceiling until upstream supports CUDA 13 (or a CUDA 12.8 env is used) |
| **QuaRot** | **W4A4** (INT4, KV4) | original repo fake-quant (accuracy); kernels build but s4→INT8 | yes (patched, pinned) | **PASS** (PPL 1B/3B/8B done) | **no speed path** — INT4 kernels ~5–6× slower than FP16 (no s4 MMA on Blackwell) | hardware (s4 silently lowered to INT8) |
| **SpinQuant** | **W4A4** (INT4, KV4) | original repo fake-quant | yes (patched) | **PASS** (PPL 1B/3B done + 8B Hadamard) | no GPU kernels exist — accuracy-only | none for accuracy; no speed artifact |
| **Atom** | **W4A4** (~4.2b, INT8 keepers, KV4) | original repo fake-quant | accuracy yes; kernel forced-build launches but is INT8-SASS | **PASS** (PPL 1B/3B/8B=18.40/9.70/7.57) | accuracy-only (s4→INT8 lowering) | hardware + CMake bit-rot |
| **QServe/QoQ** | W4A8KV4 (NOT W4A4 — context) | original repo | **NO** (CUDA13: `cub::Max` removed; W4A8 GEMMs alone compile) | fake-quant runs; 1B config degenerate | blocked as shipped | CUDA-13 API break in fused attention |
| **ResQ** (ancestor) | **W4A4** (INT, 1/8 ch INT8) | our promix reimplementation | yes | **PASS** (1B PPL 14.68) | fake-quant (ProMix provides the real-kernel path) | — |
| GPTQ / AWQ (context) | W4A16 (A=BF16) | vLLM-native (Marlin) | yes | **PASS** (measured) | strong decode; weight-only class — not a peer | — |
| FP8 (context ceiling) | W8A8 | vLLM-native | yes | **PASS** (measured) | accuracy ceiling reference | — |
| SVDQuant | W4A4 (NVFP4, diffusion) | nunchaku engine | not attempted | — | diffusion-focused, not LLM-serving | out of scope as LLM baseline |

**Bottom line (Part A):** when inputs/sizes are fixed we can immediately run: vLLM-NVFP4
(primary head-to-head), GPTQ/AWQ/FP8/BF16 (context), **ARCQuant original kernels (smoke
PASSED on this box)**, and the four classic W4A4 peers in their honest best-available forms
(QuaRot/SpinQuant/Atom = accuracy-only fake-quant; QServe = blocked, cite paper numbers).
MicroMix builds but mis-validates under CUDA 13 as shipped. Remaining one-step setup:
`qutlass` source build (FP-Quant same-stack W4A4).

---

## Part B — Related-work completeness survey (2024–2026)

### B.0 What ProMix is (for closeness scoring)

ProMix = ResQ-lineage W4A4(+KV4): PCA basis + learned rotation; **error-driven per-layer
high/low channel partition** (high-variance 1/8 → high precision, rest → 4-bit); formats
**NVFP4 (low) + MXFP8 (high)**; **single-launch dual-format fused GEMM** on SM120 with shared
FP32 accumulator (hand kernel + twin-CollectiveMma co-design, at/above CUTLASS TC duty,
beats CUTLASS pure-FP4 endpoints at production shapes); real vLLM-targeted serving path.
Closeness axes: (1) activations quantized to 4-bit? (2) error/Hessian-driven precision
partition? (3) FP4/NVFP4 block-scaled formats? (4) dual-format fused kernel / HW alignment?

Verdicts: **must-cite** (appears in related work), **must-compare** (baseline in experiments),
**scoop-risk** (overlaps our claimed novelty; needs explicit differentiation).

### B.1 Group 1 — W4A4 weight+activation (TRUE peer group)

| Work | Venue/date | Format & bits | Allocation mechanism | Kernel? | Closeness | Verdict |
|---|---|---|---|---|---|---|
| **ARCQuant** (2601.07475) | arXiv 2026-01, code | **NVFP4 W4A4**, uniform; top-S outlier channels get *quantized residual channels* appended along K (same NVFP4) | absmax channel reorder (Atom-style) + threshold τ=M/8; S per layer (adaptive, ≤512) | **no new GEMM needed** — standard CUTLASS NVFP4 GEMM on K+S; fused reorder+RMSNorm+quant kernel; vLLM integration; **RTX 5090 + RTX PRO 6000** | VERY HIGH | **must-compare + HIGH scoop risk** (see B.6) |
| **MicroMix** (2508.02343) | arXiv 2025-08, code | **MXFP4/MXFP6/MXFP8 per-channel mix** (W and A same bit per channel; avg ~5b) | per-layer quantization-error thresholds pick which channels get FP6/FP8 | **yes — co-designed Blackwell GEMM** (arbitrary MXFP4/6/8 channel combos, BF16 out); RTX 5070Ti/5090 | VERY HIGH | **must-compare + HIGH scoop risk** (see B.6) |
| **FGMP** (2504.14152, NVIDIA+UCB) | arXiv 2025-04 | **NVFP4 low + FP8 high**, W AND A, per-BLOCK (g16) | Fisher-information-weighted per-block sensitivity; runtime act classifier | **ASIC datapath augmentation** (prototype HW; no GPU kernel — no heterogeneous-group MMA on real GPUs) | VERY HIGH (format pairing identical) | **must-cite + differentiate** (we make NVFP4+FP8-class mixing run on REAL SM120 silicon via channel partition + dual-phase kernel) |
| **QuaRot** (2404.00456) | NeurIPS'24 | INT4 W4A4KV4 uniform, Hadamard | none (uniform) | INT4 CUTLASS (Ampere s4; dead on Blackwell) | HIGH (ancestor) | must-cite + must-compare (accuracy; already measured on-box) |
| **SpinQuant** (2405.16406) | 2024 | INT4 W4A4KV4, learned (Cayley) rotations | none (uniform) | none | HIGH (ancestor; our Step-1 optimizer lineage) | must-cite + must-compare (measured on-box) |
| **Atom** (2310.19102) | MLSys'24 | INT4 + 128 INT8 keeper channels, KV4 (~4.2b) | absmax channel reorder, fixed keeper count | Ampere INT4 kernels (dead on Blackwell) | HIGH (mixed-precision channel ancestor) | must-cite + must-compare (measured on-box) |
| **QUIK** (2310.09259) | 2023 | INT4 + outlier cols INT8/FP16 | per-column outlier selection | CUTLASS INT4 | MED-HIGH | must-cite |
| **ResQ** (2412.14363) | our ancestor | INT4 + 1/8 ch INT8 (4.5b), PCA+rotation | activation-variance PCA subspace | fake-quant | — | ancestor (ProMix = FP/Blackwell/fused-kernel/adaptive extension) |
| **CoQuant** (2604.26378) | arXiv 2026-04, code | INT 4.5b (1/8 ch 8-bit), W4A4KV4 — same recipe as ResQ | **joint weight+activation weighted-PCA** closed form picks the high subspace | none (fake-quant, A100) | VERY HIGH algorithmically | **must-cite + must-compare (accuracy); scoop risk on "better-than-ResQ subspace selection"** (see B.6) |
| **OffQ** (2606.07116, Huawei) | arXiv 2026-06 | INT **W4A4KV4 UNIFORM** (no mixed precision) | top-1-PCA concentrates outliers into 1 channel → absorbed as additive offset | deployment-friendly (uniform grid) | HIGH (anti-thesis: removes the need for a high-precision part) | must-cite (+compare if code) — the "uniform is enough" counterpoint |
| **KurTail** (2503.01483) | 2025 | INT W4A4KV4, kurtosis-optimized rotation | none (uniform) | none | MED | must-cite (rotation objective alternative) |
| **WUSH** (2512.00956, IST-DASLab) | arXiv 2025-12, code | **MXFP4 W4A4** (also INT) | closed-form near-optimal non-orthogonal blockwise transform (Hadamard backbone + second-moment) | fused GPU impl, 5.8× per-layer vs BF16 | HIGH (transform SOTA for FP4 W4A4) | **must-cite**, candidate accuracy baseline; counters "rotation can't help FP4" |
| **FP-Quant / Quartet (QuTLASS)** | 2025, in vLLM as `fp_quant` | MXFP4/NVFP4 W4A4 + online Hadamard | none (uniform) | **QuTLASS Blackwell kernels, vLLM-native** | HIGH (the same-stack rotation-FP4 peer) | **must-compare** (vLLM-native) |
| **FlatQuant** (2410.09426) | 2024 | INT W4A4, learnable affine per-layer transforms | none (uniform) | custom kernel (Ampere-class) | MED-HIGH | must-cite |
| **DuQuant** (2406.01721) | NeurIPS'24 | INT W4A4, rotation+permutation | none | none | MED | must-cite |
| **OSTQuant** (ICLR'25) | 2025 | INT W4A4, orthogonal+scaling transforms | none | none | MED | must-cite |
| **PrefixQuant** (2410.05265) | 2024 | INT W4A4 static | prefix-token outlier isolation | none | MED | must-cite |
| **MosaicQuant** (2606.15652) | arXiv 2026-06 | INT, W4A4/W4A8: dense 4-bit base + sparse 4-bit residual BLOCKS (weight side); per-token A quant | output-distortion-driven residual block selection | **ZipperEngine: sparse residual fused INTO dense 4-bit GEMM** (RTX 4090, 1.24× vs FP16) | HIGH on "fused compensation kernel", weight-side focus | must-cite + differentiate (their mixed part is weights-only; our high/low split covers activations too) |
| **SharQ** (2606.26587, same group as ARCQuant) | arXiv 2026-06, code | **FP4 W4A4** (NVFP4/HiF4/MXFP4) + input-adaptive N:M activation sparsity | online sparse(outlier)–dense decomposition; sparse-FP4 + dense-FP4 GEMM share one FP4 weight payload | fused prep kernel (mask+residual+LN); RTX 5090, 2.2–2.4× vs FP16 | HIGH (another compensation-style NVFP4 W4A4 with kernels) | must-cite; candidate compare |
| **DualQuant** (2610.13767; Sharify — ResQ co-author) | arXiv 2026-10 | **NVFP4/MX-family W4A4**, calibration-free | dual-scale weight-reconstruction MSE → per-channel act rescaling (closed-form BCD) | real packed W4A4 path on RTX 4090, 2.46× vs BF16 | HIGH (ResQ-adjacent authors, NVFP4 W4A4, calibration-free counterpoint) | must-cite; candidate compare |
| **GyRot** (2607.27694, KAIST) | arXiv 2026-07 | INT4 W4A4, rotation+group co-design | coarse-rotation/fine-grouping + harmonic permutation | **ASIC accelerator** (not GPU) | MED | must-cite (HW co-design group) |
| **JustQuant** (2609.33601) | arXiv 2026-09 | plain INT4 A4 via QAT/distillation (no rotation/SVD/smoothing/mixed precision) | none — pushes complexity into training | plain operators | MED (counter-claim to our whole operator family) | must-cite (address the PTQ-vs-QAD framing) |

### B.2 Group 2 — Mixed-precision / bit-allocation (the algorithmic peer group)

The MXPLM survey's own taxonomy (2510.16805, Fig. 4) splits frameworks into MPW (weights
mixed, A=FP16), MPW+UPA (A uniform), MPW+MPA (both mixed). **Its MPW+MPA set is essentially
{Atom, QUIK, ResQ}** — the category ProMix lives in is near-empty in the literature. Most
"mixed-precision quantization" papers are weight-only (Group 5).

| Work | A-quant? | Allocation granularity & driver | Closeness / verdict |
|---|---|---|---|
| **SHMQ** (EMNLP-Industry'25: "Beyond Dynamic Quantization… Static Hierarchical Mix-precision") | **A8 static** (W4.8A8 — NOT W4A4) | **unified Hessian derivation for inter-layer (coarse) + intra-layer (fine) precision allocation** + sensitive-channel permutation; 2.86× speedup | HIGH algorithmic overlap with our error-driven hierarchical allocation. BUT: INT, A8, no rotation/PCA, no FP4, no dual-format kernel. **must-cite; medium scoop on the allocation idea, none on W4A4/NVFP4/kernel** |
| **Oltron** (DAC'24, DOI 10.1145/3649329.3656221) | yes (outlier-aware act quant, OliVe-lineage) | **inter-/intra-layer adaptation of outlier ratio/precision, software-hardware co-design** | *Caveat: full text paywalled (ACM); assessed from title/venue/metadata + citing text.* It is an **ASIC accelerator** co-design (DAC architecture track), INT-based, pre-ResQ. Conceptually the closest prior claim to "per-layer adaptive outlier precision + co-design", but no GPU path, no FP formats, no PCA subspace. **must-cite; conceptual-overlap scoop LOW-MEDIUM in practice** (different substrate + format; our kernel runs on commodity Blackwell) |
| **dMX** (2606.04115, AMD/Brevitas) | yes (W AND A in MXFP) | **per-LAYER differentiable format assignment** (MXFP4/6/8), annealed to HW formats, budget-regularized | HIGH on the per-layer adaptive-bitwidth axis (= our adaptive high_fraction Phase-2). KEY difference: per-layer uniform format needs NO new kernel; our INTRA-layer channel split is what requires (and gets) the fused dual-format kernel. **must-cite** |
| **HCDCMQ** (Neurocomputing 2026, no arXiv) | unverified (paywalled) | Hessian-aware channel determinism-decomposition + counterfactual multi-agent optimization for channel-wise MP PTQ | Title-level: channel-wise Hessian-driven MP = our partition driver family. **must-cite; verify W4A4-or-not before any comparison claim** |
| **MicroMix / FGMP / ARCQuant** | (see Group 1) | channel-threshold / Fisher-block / residual-channel | the W4A4 allocation peers — see B.6 |
| SliM-LLM (2405.14917) | **no (W-only)** | Hessian/salience group-wise W bit allocation | context only |
| CMPQ (2410.13056) | **no (W-only)** | activation-statistics-driven channel-wise W precision | context only |
| MixLLM (survey §3.2.1) | A8 (W-mixed A8) | output-feature salience W4/W8 + system co-design | context; must-cite as MPW+UPA neighbor |
| SqueezeLLM / SpQR / OWQ / PB-LLM / LLM-MQ / APTQ / LLM-PQ / BitMoD / BlockDialect etc. (survey Fig. 4) | **no (W-only)** | various (sensitivity, NAS, codebooks) | context bucket, cite via survey |
| AlignQuant (2610.07457) | A8 (INT8 act) | tile-aligned W precision allocation, prefill/decode-phase-aware, loss-gradient-weighted | adjacent (HW-aligned MP), A8 not A4; must-cite in HW-alignment discussion |
| SensitivityGuided MPQAT / Joint Pruning+MPQ (2606.07819) / Multi-bitwidth codebooks (2606.12876) / component-sensitivity (2609.26173) | mixed | QAT/pruning-side | cite-optional |

### B.3 Group 3 — Rotation / transform methods

QuaRot, SpinQuant (Group 1, ancestors); **KurTail** (kurtosis objective); **OSTQuant**
(orthogonal+scaling); **FlatQuant** (affine per-layer); **DuQuant**; **OptRot** (2512.24124,
Microsoft: data-free fusible rotations minimizing weight 4th-moment — beats Hadamard/SpinQuant
for WEIGHT quant and W4A8; **honestly reports it performs WORSE at W4A4**); **WUSH**
(near-optimal data-adaptive non-orthogonal transforms, W4A4 MXFP4, fused GPU impl);
WUSH-KV (2609.38121, KV variant); **GyRot** (rotation×grouping co-design, ASIC).

**The live scientific dispute ProMix must address:** the comprehensive-eval survey
(2507.17417) measures that **rotation helps INT4 but adds little for MXFP4/NVFP4**; ARCQuant
goes further — Hadamard **hurts** NVFP4 by destroying block-isolation (propagates outlier
magnitude into all blocks' local dynamic range). WUSH counters that *adaptive* (second-moment,
non-orthogonal) transforms DO help FP4. ProMix sits exactly on this fault line: our PCA-U +
learned-R is data-adaptive (WUSH-side), and our high/low split means rotation operates
*within* variance-sorted subspaces rather than globally smearing outliers (partially immune to
the ARCQuant critique). Our own FP ablation (weights-only FP beats INT; activation-side gap)
is direct evidence in this debate. → related-work section must engage 2507.17417 + ARCQuant +
WUSH explicitly.

### B.4 Group 4 — NVFP4 / FP4 on Blackwell (format & serving ecosystem)

- **vLLM 0.32** NVFP4 W4A4 serving (modelopt + compressed-tensors; FlashInfer
  `fp4_gemm_cutlass_sm120`) — our same-stack baseline (Part A).
- **TensorRT Model Optimizer / TensorRT-LLM** NVFP4 PTQ recipe (per-tensor second-level scale
  + g16 E4M3) — the "default NVFP4" every 2026 paper compares to; cite.
- **QuTLASS / FP-Quant (Quartet)** — open Blackwell FP4 kernels + rotation; vLLM `fp_quant`.
- **CUTLASS block-scaled SM120 GEMMs** (examples 79b/79c) — our kernel substrate; cite.
- **ARCQuant / MicroMix / WUSH** — method papers WITH Blackwell kernels (Group 1).
- **QUADS** (2607.15810): NVFP4 for RL ROLLOUT in MoE (quantization-error alignment across
  dual sides) — training/RL side, not PTQ serving. cite-optional (shows NVFP4 breadth).
- **ReSET** (2606.13233): NVFP4 reasoning accuracy via step-aware temperature scaling —
  orthogonal (decoding-side fix). cite-optional.
- **SVDQuant** (2411.05007): NVFP4 W4A4 for DIFFUSION models, low-rank outlier branch
  (ResQ-adjacent decomposition) + Nunchaku engine. must-cite (decomposition + NVFP4 precedent).
- NVIDIA NVFP4 pretraining (2509.25149) — training-side NVFP4 evidence; cite-optional.
- **JustQuant** — "NVFP4/rotations are deployment burdens" counter-position (Group 1).

### B.5 Group 5 — Context-only (NEVER in peer comparisons)

- **Weight-only W4A16**: GPTQ, AWQ (+Marlin), SqueezeLLM, SpQR, OWQ, PB-LLM, SliM-LLM, CMPQ,
  OptRot (weight-focused), MosaicQuant's mixed part, dMX-weights-only configs, survey's MPW
  bucket. Report alongside, never as W4A4 peers (activations unquantized ⇒ unfair).
- **KV-cache-only**: RotateKV (2501.16383), KVLinC (2510.05373), "Don't Waste Bits" KV
  (2604.04722), WUSH-KV. Different axis; cite in KV4 subsection only.
- **Attention-internal precision routing**: TileMix (2608.17336, FP16/INT8 score tiles) —
  attention scores, not linear-layer GEMM. cite-optional.
- **Surveys (hubs)**: 2510.16805 (MXPLM taxonomy: MPW / MPW+UPA / MPW+MPA — positions ProMix
  in the near-empty MPW+MPA cell) and 2507.17417 (same-ground evaluation; the
  rotation-vs-FP4-format finding). Both must-cite.

### B.6 ProMix differentiation vs the closest works (honest)

**1. ARCQuant — the sharpest competitor.** Same format family (NVFP4 A4W4), same GPU (RTX
PRO 6000), **ACL 2026 Main**, code public, **upstreamed into TensorRT-LLM**, kernels smoke-
tested working on our box, and its motivation section argues *against* both of our
pillars: (a) rotations break NVFP4 block isolation; (b) dual-format mixed precision
"precludes hardware-accelerated pipelines" because tensor cores lack heterogeneous-group-size
MMA. Differentiation: (i) **our fused dual-phase kernel is precisely the artifact ARCQuant
claims impractical** — one launch, NVFP4(g16)+MXFP8(g32) phases sharing an in-register FP32
accumulator, measured at/above CUTLASS TC duty and below the two-launch ideal envelope; their
hardware-uniformity premise is empirically refuted on their own named GPU. (ii) Their
compensation adds S residual channels (extra K-dim FLOPs, both operands duplicated for
outlier cols) and compensates only the top-S channels at effective 8-bit-equivalent
resolution; ours gives the high subspace a genuinely wider-dynamic-range format (MXFP8) after
PCA concentration, with KV4 and learned rotation. (iii) Accuracy claim must be settled
empirically → **must-compare** (their code on our box). Scoop risk: **HIGH** on "accurate
NVFP4 W4A4 serving on Blackwell with outlier handling"; our unique claims remain (PCA/error-
driven partition, dual-format fused kernel, adaptive per-layer fraction).

**2. MicroMix — the co-design twin.** "Co-designed mixed-precision algorithm + Blackwell
kernel, channel-level precision allocation" is literally our elevator pitch. Differentiation:
(i) formats — MX family only (g32 throughout, incl. MXFP4; no NVFP4 g16/E4M3-scale, which is
the accuracy-critical format at 4-bit) vs our NVFP4+MXFP8 heterogeneous-group pairing; (ii)
allocation driver — local quantization-error thresholds vs our global PCA-rotated
variance/error subspace with learned rotation (ResQ lineage, provably concentrates energy);
(iii) their channels cost ~5 avg bits for near-FP16; our production point is 4.5b with 1/8
high fraction + KV4; (iv) kernel — theirs executes channel segments via per-format-pair
kernels (src: w4a4/w4a6/w4a8/w6a6/w8a8.cu — segmented, g32 throughout); ours fuses two
different scale-granularity formats (g16+g32) in ONE mainloop with a shared accumulator.
As shipped it also mis-validates on CUDA 13 (addendum). Scoop risk: **HIGH** on the
co-design framing; must-compare once numerically valid on this box.

**3. FGMP — the format twin on the wrong substrate.** NVFP4+FP8 W/A mixed precision,
Fisher-weighted selection — but per-BLOCK with runtime reassignment, realizable only in their
proposed ASIC datapath. ProMix's channel-partition (contiguous after reorder) is exactly the
restructuring that makes NVFP4+FP8-class mixing executable on *shipping* Blackwell tensor
cores. Cite prominently; scoop risk MEDIUM (concept published; GPU realization + subspace
method are ours).

**4. CoQuant — the subspace-selection successor.** Improves ResQ by selecting the
high-precision subspace with joint W+A weighted PCA (closed form), beating ResQ by ~0.2-0.4
PPL on the same 4.5b INT recipe. Overlaps our planned Phase-2 "error-driven (not just
variance) partition": the *selection-criterion* novelty is partially taken for the
fixed-fraction case. Remaining ProMix-unique: per-layer ADAPTIVE fraction, FP formats,
hardware alignment (K_high%32), fused kernel, real serving. Must-cite + adopt-or-compare
(their criterion could even be plugged into our Step 0). Scoop risk: **MEDIUM-HIGH on the
allocation-criterion axis only.**

**5. MosaicQuant — the fused-compensation kernel neighbor.** Unified-format compensation
(4-bit residual) fused into the dense GEMM pipeline — rhymes with both ARCQuant (residual) and
our fused dual-phase execution, but weight-side, INT, Ada GPU. Cite; differentiate on
activation-side high/low and FP formats.

**6. Oltron + SHMQ — the "inter/intra-layer adaptation" prior art.** Both claim
Hessian/sensitivity-driven inter+intra-layer precision adaptation with HW awareness (Oltron:
ASIC co-design, DAC'24; SHMQ: static W4.8A8 INT, EMNLP-Ind'25). Honest reading: **the
abstract idea "allocate precision per layer AND within layer, driven by a sensitivity metric,
co-designed with hardware" is published — twice.** ProMix cannot claim that idea as novel.
What remains defensible: the *specific* mechanism (PCA-rotated variance/error subspace
partition on ACTIVATIONS+weights at W4A4), the *formats* (NVFP4+MXFP8 on real Blackwell), the
*kernel* (single-launch dual-format fused GEMM, bit-exact, at CUTLASS TC parity), and
*measured serving* integration. Related-work must cite both and frame ProMix as the first to
realize error-driven intra-layer dual-FP-format mixing on commodity Blackwell at W4A4.

### B.7 Completeness verdict — what we were missing

Our old citation set (QuaRot, SpinQuant, Atom, QServe, ResQ, GPTQ/AWQ) was 2023–2024-only.
Missing and now cataloged:

1. **The 2025–26 Blackwell FP4 W4A4 wave (the real competitive set):** ARCQuant, MicroMix,
   FGMP, WUSH, FP-Quant/QuTLASS, vLLM/TRT NVFP4 recipes. ← biggest gap; two of these share
   our exact GPU and pitch.
2. **ResQ successors on subspace selection:** CoQuant, OffQ (+ KurTail objective line).
3. **Per-layer/channel allocation:** dMX, SHMQ, HCDCMQ, Oltron, SliM-LLM/CMPQ (W-only
   context), AlignQuant/TileMix (tile-level).
4. **Survey hubs:** 2510.16805 (we sit in its near-empty MPW+MPA cell — a positioning gift),
   2507.17417 (rotation-vs-FP4 finding we must engage).
5. **Counter-positions to address:** ARCQuant (rotation hurts NVFP4), JustQuant (no rotation
   needed at all), OptRot (rotations trade W-vs-A quality at W4A4), 2507.17417 (rotation
   ineffective for FP4).

Snowball convergence: ARCQuant/MicroMix/MosaicQuant/CoQuant/dMX baseline lists and both
surveys re-surface the same closed set (QuaRot, SpinQuant, Atom, QUIK, ResQ, SmoothQuant,
GPTQ/AWQ, FlatQuant, OSTQuant, SVDQuant, MicroMix, FGMP) — no new W4A4/allocation works
appeared on the second hop, so the landscape above is complete as of 2026-10.

---

## Addendum — new-peer smoke attempts (ARCQuant, MicroMix), 2026-10-30

Both repos cloned (shallow) to `/home/luofan/baselines/peers/{arcquant,micromix}/`, cutlass
submodules shallow-initialized, built **inside container `luofan-plaquant`** (nvcc 13.0,
cmake 4.2.3, pybind11 3.1.0). Only deviation from as-shipped: `-Dpybind11_DIR=$(python3 -m
pybind11 --cmakedir)` (their CMake `find_package(pybind11)` doesn't see pip-installed
pybind11). Build logs: `peers/{arcquant,micromix}/kbuild_sm120.log`.

**ARCQuant** (actypedef/ARCQuant, shallow @HEAD 2026-10; cutlass pinned ec8daf64):
- Status upgrade since paper: **ACL 2026 Main**; merged into **TensorRT-LLM (PR #11333)**;
  vLLM integration listed, "vLLM-based efficiency evaluation scripts to be released".
- Kernels CMake targets `CMAKE_CUDA_ARCHITECTURES 120a` natively. Build: **SUCCESS** on
  CUDA 13 (README asks for 12.8 — not needed).
- Smoke: `./build/bench_nvfp4` → "GEMM completed in 0.145673 ms / nvfp4 gemm finished"
  (GPU 6). `import agemm` → OK; API: `matmul, reorder_quantize_w, reorder_quantize_x,
  rmsnorm_quantize_x, init/append/batch_decode kv f16+i4`. **RUNNABLE as shipped.**
- Kernel-structure note (repo inspection): src = nvfp4.cu (standard block-scaled GEMM) +
  reorder/rmsnorm-quantize fusion + down.cu + flashinfer glue — consistent with the paper's
  "standard CUTLASS GEMM on K+S, complexity moved into a fused quantize kernel".

**MicroMix** (lwy2020/MicroMix, shallow @HEAD; vendored cutlass a1aaf230):
- mgemm CMake also targets 120a. Build: **SUCCESS** on CUDA 13.
- Smoke: `./build/bench_mxf4f6f8 128 4096 4096 10 <cfg> --validate` launches and reports
  97–112 TFLOPS, **but `GEMM VAL FAILED` for all tried configs (4x4, 6x6, 8x4)** —
  result mismatch vs its own reference. Repo pins CUDA 12.8; numerics under CUDA 13 are
  wrong as shipped. Per policy (no porting), recorded as the method's honest ceiling on this
  box until upstream supports CUDA 13 (or we provision a CUDA 12.8 env, which is a setup
  choice, not a code fix).
- Kernel-structure note (repo inspection): mgemm/src has per-format-pair kernels
  (w4a4.cu, w4a6.cu, w4a8.cu, w6a6.cu, w8a8.cu) — channel groups are executed by
  per-precision GEMM kernels (segmented execution), NOT a single dual-format mainloop with a
  shared accumulator. Keeps our single-launch heterogeneous-group-size (g16+g32) fused kernel
  claim intact.

**Procedure note:** the DeepSeek websearch helper referenced in the task
(`kernels/mixed_gemm_sm120/tools/websearch.py` + `.ds_key`) does not exist on the box (not in
git history of any branch/worktree); the survey used arXiv API + Semantic Scholar + OpenReview
directly instead.
