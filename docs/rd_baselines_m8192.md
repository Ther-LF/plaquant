# W4A4-on-Blackwell head-to-head @ M = 8192 (rd-bench)

**The first real performance side-by-side**: every runnable W4A4 competitor kernel actually
RUN on our box, same shapes, same GPU, same session — vs ProMix's single-launch fused
NVFP4+MXFP8 GEMM.

**HW** RTX PRO 6000 Blackwell Server (SM120, cc 12.0), container `luofan-plaquant`
(torch 2.11 cu130, nvcc 13.0), **GPU 2**, box otherwise idle except a third-party job on
GPU 7 (see variance notes). **Date** 2026-10-30. **Branch** `feat/rd-bench` (off
`origin/feat/rd-alloc`), remote workdir `/home/luofan/plaquant-bench`.

**Protocol.** Quantized-Linear **GEMM-only** latency at **M = 8192** on the per-projection
(unmerged) shapes — 3B: q/k/v/o 3072×3072, gate/up 8192×3072, down 3072×8192; 8B: 4096 /
14336 analogues. cudaEvent, median of 50 (15 warmup) for all Python-path methods; SharQ via
its own shipped CUDA harness (`bench_sharq_linear`, 50 warmup / 200 iters avg). Activation
quant / prep / pack kernels are EXCLUDED for every method (vLLM quant+GEMM additionally
recorded). Each method runs **as shipped** (no porting/optimization by us; only build-system
compat fixes already recorded in `docs/rd_related_work.md`). The canonical numbers below are
one **serialized back-to-back suite** (`promix/scripts/rd_bench_m8192_all.sh`) so all
methods share the same clock/power regime — the per-method fp16 cuBLAS reference agreed
within ~1 % across the whole suite. Raw JSON: `docs/rd_baselines_m8192_data/`.

---

## 1. Head-to-head table (median ms @ M = 8192; lower is better)

| model·shape (N×K) | fp16 | **ProMix fused** | vLLM-NVFP4 | ARCQuant S=512 | ARCQuant S=0 | MicroMix pureFP4 | MicroMix mix-5.5b | SharQ |
|---|---|---|---|---|---|---|---|---|
| 3B qkvo 3072×3072 | 0.474 | **0.2024** | 0.3102 | 0.2203 | 0.2011 | 0.2095 | 0.3533 | 0.2265 |
| 3B gate_up 8192×3072 | 1.025 | 0.4416 | 0.5274 | **0.4302** | 0.3878 | 0.4108 | 0.8155 | 0.6020 |
| 3B down 3072×8192 | 1.110 | 0.4106 | 0.4877 | **0.3917** | 0.3666 | 0.4034 | 0.6405 | 0.5386 |
| 8B qkvo 4096×4096 | 0.739 | **0.2859** | 0.3831 | 0.2890 | 0.2668 | 0.3081 | 0.4626 | 0.3791 |
| 8B gate_up 14336×4096 | 2.490 | 0.8681 | 0.8665 | **0.8471** | 0.7921 | 0.8593 | 1.5606 | 1.3082 |
| 8B down 4096×14336 | 2.450 | 0.7999 | 0.8905 | **0.7441** | 0.7383 | 0.7261 | 1.1213 | 1.2065 |

Bold = fastest among the **outlier-protected** W4A4 configs (ProMix, ARCQuant S=512,
MicroMix mix, SharQ). ARCQuant S=0 and MicroMix pureFP4 are that method's *unprotected*
plain-FP4 floor (= no outlier handling), shown for context.

**Network-proxy sum** (4×qkvo + 2×gate_up + 1×down, mission shapes — NB real GQA k/v are
narrower, so this over-weights kv):

| model | fp16 | ProMix | vLLM | ARCQuant S=512 | MicroMix 5.5b | SharQ |
|---|---|---|---|---|---|---|
| 3B | 5.056 | **2.103** | 2.783 | 2.133 | 3.685 | 2.649 |
| 8B | 10.387 | 3.680 | 4.156 | **3.594** | 6.093 | 5.339 |

**ProMix speedup over each peer (per-shape range / proxy-sum):**

- vs **vLLM-NVFP4**: 1.00–1.53× (3B sum 1.32×, 8B sum 1.13×) — wins 5/6 shapes, ties 8B gate_up.
- vs **ARCQuant (S=512)**: 0.93–1.09× (3B sum 1.01×, 8B sum 0.98×) — **effective parity**;
  ProMix wins the qkvo shapes, ARCQuant slightly wins gate_up/down (3–7 %).
- vs **MicroMix (mix ≈5.5b)**: 1.40–1.85× (sums 1.75× / 1.66×).
- vs **SharQ**: 1.12–1.51× (sums 1.26× / 1.45×).
- vs **fp16 cuBLAS**: 2.32–3.06×.

## 2. Format / bits / validation per method

| Method | Impl run | Format (A & W) | Effective weight bits | Correctness @ this bench |
|---|---|---|---|---|
| **ProMix** (ours) | fused dual-format kernel, `codesign_fused_ext` via `MixedQuantLinear`, 1 launch | NVFP4 (g16) on 7/8 K + MXFP8 (g32) on 1/8 K | 4.97 incl. scales | **validated**: cos 0.9963 vs fp32 ref here; bit-exact vs quantized oracle in prior gates (rounds 13–19) |
| **vLLM-NVFP4** | vLLM 0.32 `FlashInferCutlassNvFp4LinearKernel` (fp4_gemm_cutlass_sm120), baselines venv | uniform NVFP4 W4A4 | 4.5 | **validated**: cos 0.9910 (after fixing OUR harness — see §4) |
| **ARCQuant** | original repo `agemm` (CUTLASS NVFP4 on K+S), built CUDA13/120a | NVFP4 W4A4 + S quantized residual channels appended along K | 4.5·(K+S)/K: S=512 → 5.25 (K=3072), 5.06 (4096), 4.78 (8192), 4.66 (14336) | **validated**: cos 0.9910–0.9917 |
| **MicroMix** | original repo `mixedgemm` (segmented MXFP4/6/8 GEMMs), built CUDA13/120a | per-channel MXFP4/6/8 mix (W=A bits per channel) | pureFP4 4.25; mix (K/2,K/4,K/4) ≈ 5.75 incl. g32 scales (paper avg 5.5 elem-bits) | **NOT validated**: its own `--validate` FAILS under CUDA 13 (repo pins 12.8). Our independent probe: cos 0.9860 (FP4) / 0.9891 (mix) vs fp32 — consistent with MXFP4-level quant error, i.e. launches and is *approximately* correct, but per policy recorded as unvalidated as shipped |
| **SharQ** | original repo `bench_sharq_linear` (CUTLASS sparse-NVFP4 + dense-NVFP4), built CUDA13/120a this round | NVFP4 W4A4; activation = 4:8 sparse backbone + dense residual, 2 GEMMs on one shared FP4 weight payload | ~4.5–5 (single payload + dual scale views); **GEMM work ≈ 1.85–2.07× dense FP4 measured** | **validated** via its own correctness suite: kernel ≈ its pseudo reference (rel_rmse 0.030; total-path vs BF16 error identical to reference) |

Cosine values are run sanity-checks on random data, not accuracy comparisons (input
distributions differ per method; accuracy head-to-head is the PPL scoreboard, not this doc).

## 3. Runnability outcomes (this round)

- **vLLM-NVFP4** — RAN. venv `/home/luofan/baselines/venv` inside container.
- **ARCQuant** — RAN (smoke had already PASSED). S is calibration-adaptive per layer
  (threshold 0.125·max, repo `utilize.py`; paper caps S ≤ 512); without running its
  calibration we bracket with S ∈ {0, 256, 512} — its actual deployed GEMM lies between the
  S=0 and S=512 rows (≤4 % spread on large-K shapes, ≤10 % on 3072).
- **MicroMix** — RAN at M=8192 (timing + our cosine probe), but **flagged**: correctness not
  validated by its own harness under CUDA 13 as shipped. Its mixed config is the method's
  real operating point (paper avg 5.2–5.5 bits); the segmented 3-GEMM execution costs it
  1.4–1.9× vs ProMix at equal-or-higher bit budget.
- **SharQ** — NEW this round: cloned `actypedef/SharQ`, cutlass submodule init, built
  `sharq_ops` + benches with CUDA 13 / sm120a in-container (only deviation: `-Dpybind11_DIR`,
  same as ARCQuant/MicroMix). Smoke PASS; correctness suite PASS. Measured
  sparse-main + residual fused-accum = **1.85–2.07×** its own dense NVFP4 GEMM time at
  M=8192 (worse than the paper's ~1.7× estimate at these shapes).
- **ProMix** — RAN from the bench clone (`.torchext` isolated JIT cache; cutlass headers
  synced at pinned cb37157). Cross-check vs committed `docs/rd_largem_data` per-shape
  M=8192 rows (o_proj/down identical shapes): within ±5 %.
- Not re-run (unchanged verdicts, `docs/rd_related_work.md`): QuaRot/Atom INT4 → s4 lowers
  to INT8 on SM120, 4–7× slower than fp16 (measured in rd-largem); SpinQuant no kernels;
  QServe no CUDA-13 build and W4A8 class; FP-Quant/QuTLASS still needs a qutlass source build.

## 4. Honest notes & caveats

1. **Our old vLLM GEMM harness had two correctness bugs** (inherited from round-8
   `gemm_h2h_vllm.py`): (a) activation quantized with the *inverse* global scale → x_fp4 all
   zeros; (b) weight scale double-swizzled (`scaled_fp4_quant` defaults to swizzled,
   then `swizzle_blockscale` again). Latency is shape-driven so prior vLLM *timings* remain
   roughly valid, but all-zero operands measurably flatter the kernel on this power-limited
   part (buggy run: 0.46 ms @ 3B down vs 0.49 ms fixed). Fixed here
   (`rd_bench_m8192_vllm.py`, cos 0.991); prior rd-largem vLLM rows should be re-taken with
   the fix before publication.
2. **No clock locking** (no sudo on this box) and a third-party 527 W job on GPU 7 during
   parts of the session. Standalone re-runs showed up to ±10 % swing (and ~2× on one run
   that overlapped our own compile+bench on a sibling GPU). The canonical table is one
   serialized suite with per-method fp16 references agreeing within ~1 %; treat differences
   <5 % as parity.
3. **Shapes are the mission-specified unmerged per-projection shapes.** vLLM's deployed
   decomposition merges qkv and gate_up into wider GEMMs where its kernel fares better: on
   merged shapes at M=8192 (rd-largem) ours/vLLM was 1.13× (3B) / 0.97× (8B). The unmerged
   qkvo (N=3072/4096) is where FlashInfer-cutlass is weakest and ProMix/ARCQuant strongest.
4. **ARCQuant parity is the headline threat and the headline rebuttal.** Their plain CUTLASS
   NVFP4 build (S=0) is the fastest pure-FP4 GEMM measured on this box (faster than vLLM's
   FlashInfer path AND our own single-format NVFP4 ext). With protection on, ProMix
   (4.97 b, single launch, PCA+rotation) and ARCQuant (4.7–5.25 b effective, K-extension)
   are within ±7 % per shape, ~1–2 % on the network sum — so the perf argument between the
   two is a wash at M=8192, and the differentiation rests on accuracy per bit (our 8B wiki
   gap +0.46 vs their +0.63), KV4, and the adaptive-fraction story.
5. MicroMix's `paper_avg` mix here fixes (KN,KS,KO)=(K/2,K/4,K/4) ≈ its published 5.5-bit
   average; its real per-layer proportions require its calibration pipeline. Its pure-FP4
   row is faster but is not the accuracy configuration the paper reports.
6. SharQ pays its ≥1.7× FLOP-time by design (recovery mechanism = second GEMM; measured
   1.85–2.07× here); at M=8192 this prices it 1.12–1.51× behind ProMix, consistent with the
   rd-deepread analysis (their recovery path costs ~2× dense-FP4, ours ~1.125×).

## Reproduce

```bash
# remote, inside container luofan-plaquant, from /home/luofan/plaquant-bench
bash promix/scripts/rd_bench_m8192_all.sh 2   # arg = GPU index (must be idle)
```

Per-method scripts: `promix/eval/rd_bench_m8192_{ours,vllm,arcquant,micromix,sharq}.py`.
SharQ build recipe: see §3 (repo at `/home/luofan/baselines/peers/sharq/SharQ`).
