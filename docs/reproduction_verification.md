# Baseline reproduction verification

Date: 2026-10-26. Purpose: before comparing ProMix against the on-hardware baselines in
`docs/baselines_on_hardware.md`, confirm that our GPTQ-W4A16, AWQ-W4A16, FP8-W8A8 and
NVFP4-W4A4 reproductions (vLLM 0.32.0 + llm-compressor 0.14.0.1 on RTX PRO 6000 / SM120)
match the original papers / official docs in **accuracy** and **performance**, so that any
gap is a known protocol difference and not a mis-implementation or mis-measurement.

All our numbers are the committed ones in `docs/baselines_on_hardware.md`; the raw JSON and
quantized checkpoints were re-inspected on the GPU host (`/home/luofan/baselines`). All
checkpoint parameters below were read from the actual `config.json → quantization_config`
of the served checkpoints, not from the launcher comments.

## 0. Models and harness actually used (verified)

| Size | Checkpoint (verified path) | Type |
|---|---|---|
| 1B | `unsloth/Llama-3.2-1B-Instruct` | **Instruct** |
| 3B | `unsloth/Llama-3.2-3B` | base |
| 8B | `unsloth/llama-3-8b` | base (**Llama-3**, not 3.1) |

Note the 1B is an **Instruct** model while 3B/8B are **base** models. Instruct models have
higher raw WikiText PPL (they are chat-tuned, not LM-tuned), which is why 1B PPL (13.16) is
far above 3B/8B and why its % quant deltas look larger. This is consistent as long as ProMix
is evaluated on the same 1B-Instruct checkpoint (it is — the ResQ/ProMix pipeline uses
Llama-3.2-1B-Instruct).

**PPL harness** (`bench_vllm.py`): WikiText-2-raw-v1 **test** split, model's own HF
tokenizer, `add_special_tokens=False`, rows joined with blank lines, token stream cut into
**non-overlapping 2048-token chunks** (141 chunks, 288,627 scored tokens), NLL scored from
position 1 via vLLM `prompt_logprobs`, PPL = exp(mean NLL). This is exactly the "standard
WikiText perplexity with sequence length 2048" used by the reference papers below
(e.g. arXiv 2506.12044 §Table 4 states: "We follow the standard implementation of WikiText
perplexity that sets the sequence length to 2048"). 0 missing logprob slots in every run.

### Harness sanity check (the most important check)

If our BF16 baseline is off, every delta is suspect. It is **not**:

| Model | Our BF16 PPL | External reference (same seqlen-2048 WikiText2) | Source |
|---|---|---|---|
| Llama-3-8B | **6.138** | **6.14** | arXiv 2506.12044 Table 4; arXiv 2602.02546 (both list FP16 6.14) |
| Llama-3.2-3B | **7.818** | **7.82** | arXiv 2501.04052v3 (RaZeR) Table 1, FP16 column |
| Llama-3.2-1B-Instruct | 13.16 | no clean public ref (instruct) | — (internally consistent) |

Our BF16 8B (6.138) and 3B (7.818) match published FP16 WikiText2 to the second decimal.
**The PPL harness is correct.** (Some papers quote Llama-3-8B FP16 as ~5.75 instead of 6.14;
that is the AutoGPTQ/legacy concat protocol. Our 6.14 matches the lm-eval / per-2048-chunk
family, which is the stricter and more common modern protocol.)

---

## 1. GPTQ-W4A16

**Verified recipe** (checkpoint `quantization_config`): INT4, `group_size=128`,
`symmetric=true`, `actorder="static"` (activation reordering on, static group assignment,
Marlin-compatible), `dampening_frac=0.01`, `lm_head` excluded. Calibration: WikiText-2-raw-v1
**train**, 256 samples × 2048 tokens. Served by vLLM `MarlinLinearKernel`.

| | 1B-Instruct | 3B | 8B |
|---|---|---|---|
| Our GPTQ PPL | 14.22 | 8.20 | 6.52 |
| Our BF16 PPL | 13.16 | 7.82 | 6.14 |
| **Our Δ** | +1.06 (+8.1%) | +0.38 (+4.9%) | **+0.38 (+6.2%)** |

**Paper/doc claim (accuracy).** GPTQ (Frantar et al., 2022) predates Llama-3, so there is no
original-paper Llama-3 number; the right reference is modern W4-g128 reproductions on
Llama-3-8B at seqlen-2048 WikiText2:

- arXiv 2506.12044 Table 4 (FP16 6.14): **GPTQ-W4 = 6.86 (+0.72)**, AWQ-W4 6.56, NF4 6.56.
- arXiv 2501.04052v3 (RaZeR) Table 1, Llama-3.2-3B (FP16 7.82): GPTQ-W4A16 = 8.29 (+0.47);
  our 3B GPTQ is 8.20 (+0.38) — **better than** the RaZeR GPTQ point.

**Verdict: OK (matches / slightly better than published).** Our 8B GPTQ Δ (+0.38) is *smaller*
than the 2506.12044 GPTQ-W4 point (+0.72) and our 3B Δ (+0.38) is smaller than RaZeR's (+0.47).
The suspicion that "+0.38 looks larger than the typical +0.05..0.15" comes from Llama-2-era
expectations; **Llama-3 is well-documented to be much harder to quantize** (this is the thesis
of "How Good Are Low-Bit Quantized LLaMA3 Models?"), with W4-g128 deltas of +0.3..+0.7 on 8B.
Our GPTQ is at the good end of that range. No fix needed; `actorder` is on and group_size is
128, i.e. the recommended recipe.

**Performance.** W4A16 Marlin is weight-only, so it wins **decode** (memory-bound): 1B
663 tok/s vs 435 BF16 (1.5×), 8B 218 vs 87 (2.5×). It does *not* speed up prefill
(compute-bound) — 8B TTFT@8192 441 ms vs 392 BF16 (slightly slower), as expected for a
dequant-to-BF16 GEMM. This matches the known Marlin behaviour (decode/throughput win,
prefill neutral-to-negative). **OK.**

---

## 2. AWQ-W4A16

**Verified recipe** (checkpoint `quantization_config`): INT4, `group_size=128`,
**`symmetric=true`**, `actorder=null`, `lm_head` excluded. Pipeline:
`AWQModifier` (scale search) + `QuantizationModifier(scheme="W4A16")`. Calibration:
WikiText-2-raw-v1 train, 256 × 2048. Served by vLLM `MarlinLinearKernel`.

| | 1B-Instruct | 3B | 8B |
|---|---|---|---|
| Our AWQ PPL | 15.16 | 8.48 | 6.73 |
| Our Δ vs BF16 | +2.00 (+15.2%) | +0.66 (+8.4%) | +0.59 (+9.6%) |

**Paper/doc claim (accuracy).** AWQ (Lin et al., 2023) uses **asymmetric (zero-point) INT4**
with group 128 — e.g. the AMD/official AWQ model cards specify "Group 128 / **Asymmetric** /
UINT4 weights". Reference Llama-3-8B seqlen-2048 WikiText2: arXiv 2506.12044 Table 4
**AWQ-W4 = 6.56 (+0.42)**; RaZeR Table 1 Llama-3.2-3B best W4A16 ≈ 8.1-8.3.

**Protocol difference / suspicion CONFIRMED.** Our AWQ checkpoint is **symmetric**
(`symmetric:true`), which is *not* the canonical AWQ recipe — AWQ's accuracy benefit largely
comes from asymmetric (zero-point) INT4. Symptoms consistent with this:
1. Our AWQ (+0.59 on 8B) is **worse than** the published asymmetric AWQ-W4 (+0.42).
2. Our AWQ is **worse than our own GPTQ** at every size (8B 6.73 > 6.52; 3B 8.48 > 8.20;
   1B 15.16 > 14.22). In the literature AWQ is normally ≈ or slightly better than GPTQ.
   This inversion is the fingerprint of symmetric AWQ.

**Corrected run (asymmetric, canonical AWQ).** Re-quantized with
`QuantizationModifier(scheme="W4A16_ASYM")` (confirmed `symmetric=False, group_size=128`),
WikiText-2 128×2048 calibration, same PPL harness:

| | 1B-Instruct | 8B |
|---|---|---|
| Our AWQ symmetric (committed, 256×2048 calib) | 15.16 (+2.00 / +15.2%) | 6.73 (+0.59 / +9.6%) |
| **Our AWQ asymmetric (corrected, 128×2048 calib)** | **14.53 (+1.37 / +10.4%)** | **6.59 (+0.45 / +7.3%)** |
| Published AWQ-W4 ref (seqlen-2048 WikiText2) | — | 6.56 (+0.42) |

Switching to the canonical **asymmetric** W4A16 recipe improved AWQ at both sizes, even with
*fewer* calibration samples (128 vs 256):
- 1B: 15.16 → **14.53** (−0.63). AWQ (14.53) now sits just above GPTQ (14.22) instead of well
  below it, restoring the expected AWQ≈GPTQ ordering.
- 8B: 6.73 → **6.59** (−0.14), which now **matches the published AWQ-W4 reference 6.56 within
  noise** (Δ +0.45 vs the paper's +0.42) and is right next to our GPTQ (6.52).

This is direct confirmation that the symmetric setting — not a measurement bug — caused the
inflated AWQ numbers. (The 128-vs-256 sample difference only works *against* the corrected
run, so the true asymmetric gain is if anything slightly larger.) Corrected checkpoints:
`/home/luofan/baselines/quant_fix/{Llama-3.2-1B-Instruct,llama-3-8b}-AWQ-W4A16-asym`; raw PPL
JSON: `results/fix_awqasym_{1b,8b}.json`.

**Verdict: explained by protocol (symmetric vs asymmetric) → corrected.** The committed AWQ
number is a valid *symmetric-W4A16* reproduction but is **not** the canonical AWQ the paper
reports; the corrected asymmetric run is the apples-to-apples AWQ baseline. Recommend
re-generating 3B AWQ the same way for consistency before publishing an AWQ column.

**Performance.** Same Marlin path as GPTQ, so TTFT/decode are effectively identical to GPTQ
(1B 673 vs 663 tok/s decode; 8B 220 vs 218). Symmetric-vs-asymmetric does not change the
served kernel or its speed. **OK.**

---

## 3. FP8-W8A8

**Verified recipe** (checkpoint `quantization_config`): FP8 e4m3, **weights per-channel
(static)**, **activations per-token dynamic** (`strategy:"token", dynamic:true`), `lm_head`
excluded. Served by vLLM `CutlassFP8ScaledMMLinearKernel`. (Note: the launcher comment and
`baselines_on_hardware.md` call this "dynamic per-tensor"; the checkpoint shows it is actually
**per-token activations + per-channel weights** — a doc wording fix, not a measurement error.)

| | 1B-Instruct |
|---|---|
| Our FP8 PPL | 13.30 |
| Our BF16 PPL | 13.16 |
| Our Δ | +0.14 (+1.1%) |
(FP8 only run on 1B.)

**Paper/doc claim.** FP8 W8A8 dynamic is treated as near-lossless: Neural Magic / Red Hat's
large study ("Give Me BF16 or Give Me Death?", arXiv 2411.02355) reports <1% quality impact
across Llama FP8, and the official `neuralmagic/*-FP8-dynamic` cards report ~99.9% recovery.

**Verdict: OK (matches, near-lossless).** +1.1% on a 1B *instruct* model is at the expected
level for FP8-dynamic (per-token activation scales, per-channel weight scales, no KV quant).
No implementation concern. Recommend adding 3B/8B FP8 for completeness (expected Δ < +0.1 PPL).

**Performance.** FP8 gives the best prefill after NVFP4 (1B TTFT@8192 59 ms vs 77 BF16, ~1.3×)
and a decode win (537 vs 435 tok/s). Consistent with FP8 tensor-core throughput on SM120.
Weight size 1.52 GB = ½ BF16 (2.49), exactly as expected. **OK.**

---

## 4. NVFP4-W4A4

**Verified recipe** (checkpoint `quantization_config`, `format:"nvfp4-pack-quantized"`):
weights E2M1, `group_size=16`, fp8-e4m3 micro-scales, symmetric; **activations** E2M1
`group_size=16`, fp8-e4m3 scales, `dynamic:"local"` with a `static_minmax` observer
(calibrated per-tensor global scale + runtime per-16 micro-scale); `lm_head` excluded.
Served by vLLM `FlashInferCutlassNvFp4LinearKernel` (`fp4_gemm_cutlass_sm120`). This is a true
**W4A4** baseline (unlike GPTQ/AWQ which are W4A16).

| | 1B-Instruct | 3B | 8B |
|---|---|---|---|
| Our NVFP4 PPL | 15.65 | 8.63 | 6.87 |
| Our BF16 PPL | 13.16 | 7.82 | 6.14 |
| **Our Δ** | +2.49 (+18.9%) | +0.81 (+10.4%) | +0.73 (+11.9%) |

**Paper/doc claim (accuracy).** NVFP4 W4A4 reference — arXiv 2501.04052v3 (RaZeR) Table 1,
same seqlen-2048 WikiText2:

| Model | FP16 (ref) | NVFP4 **W4A4** (ref) | Ref Δ | Ours |
|---|---|---|---|---|
| Llama-3.2-3B | 7.82 | **8.60** | +0.78 | 8.63 (+0.81) |
| Llama-3.1-8B | 6.24 | **6.95** | +0.71 | — |
| Llama-3-8B (ours) | 6.14 | — | — | 6.87 (+0.73) |

**Verdict: OK (bullseye match).** Our 3B NVFP4 W4A4 (8.63) lands on RaZeR's NVFP4 W4A4 (8.60),
Δ +0.81 vs +0.78. Our 8B NVFP4 Δ (+0.73) matches RaZeR's Llama-3.1-8B NVFP4 W4A4 Δ (+0.71)
— the small absolute gap is just Llama-3 vs Llama-3.1. The activation scheme (static global
+ dynamic per-16 micro-scale, lm_head excluded) is the standard NVFP4 recipe. No concern.

**Performance.** NVFP4 is the prefill winner on SM120: TTFT@8192 b1 beats BF16 by 1.6× (1B),
1.9× (3B), **2.2× (8B)** and beats W4A16 Marlin at every size. This is consistent with FP4
tensor cores doubling math throughput for compute-bound prefill. **Caveat on "2-4×" claims:**
NVIDIA's headline NVFP4 "2-4×" figures are **cross-generation** (B200 FP4 vs H100/H200 FP8/BF16)
and are *not* comparable to our same-stack SM120 ratio; our ~2× prefill-vs-BF16 on one
RTX PRO 6000 is the correct, apples-to-apples number and is in line with the ~2× math-throughput
expectation. Decode is limited by the extra activation quantize/scale per layer, so NVFP4
trails W4A16 at b1 decode but still beats BF16 from 3B up. **OK.**

---

## 5. Summary verdicts

| Baseline | Our 8B Δ (PPL) | Published 8B ref Δ | Verdict |
|---|---|---|---|
| GPTQ-W4A16 | +0.38 | +0.47..+0.72 | **OK** — ours slightly better than refs; recipe (g128, act-order) is correct |
| AWQ-W4A16 (committed, sym) | +0.59 | +0.42 (asym) | **MISMATCH (recipe) → FIXED**: committed run is *symmetric*; canonical AWQ is *asymmetric* |
| AWQ-W4A16 (corrected, asym) | **+0.45** | +0.42 (asym) | **OK** — 8B 6.59 matches published 6.56 within noise; 1B 15.16→14.53 |
| FP8-W8A8 | (1B +0.14) | <1% / ~lossless | **OK** — near-lossless as documented |
| NVFP4-W4A4 | +0.73 | +0.71 (3.1-8B) | **OK** — bullseye match to RaZeR NVFP4 W4A4 |

### 3-line bottom line
1. **Trustworthy.** The PPL harness is validated (BF16 8B=6.14, 3B=7.82 match published FP16
   to 2 dp), and GPTQ, FP8, NVFP4 reproductions all match published WikiText2 deltas within
   noise — GPTQ is even slightly better than references, NVFP4 is a bullseye vs RaZeR.
2. **One recipe fix, now done:** AWQ was quantized **symmetric** (non-canonical); the paper's
   AWQ is **asymmetric**. Re-running asymmetric fixed it — 8B 6.73→**6.59** (matches published
   6.56) and 1B 15.16→**14.53**. Swap the AWQ column to these corrected numbers and regenerate
   3B AWQ asymmetric (expected ≈8.3-8.4) before publishing. No other baseline needs a re-run.
3. **Performance claims hold** on the same SM120 stack (NVFP4 ~2× prefill vs BF16, Marlin
   ~1.5-2.5× decode, FP8 ~1.3× prefill); do **not** compare these to NVIDIA's cross-generation
   "2-4×" FP4 headlines. ProMix W4A4 must beat the NVFP4 W4A4 accuracy (8B +0.73 / 3B +0.81)
   at equal-or-better prefill speed — that is the real bar.
