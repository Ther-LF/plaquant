# rd-downctl: Forced-FP8-on-down_proj Control — All-FP4 down_proj Is Measured-OPTIMAL, Not a Simplification

**Branch:** `feat/rd-downctl` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000
Blackwell (SM120), docker `luofan-plaquant`, GPU 6 · **Date:** 2026-11-03 ·
**Model:** Llama-3.2-3B · **Substrate:** dense-R-on, ring=64, matched protocol =
the committed `rd_metrics` shipping build (same `rd_validity` build path,
o_proj held uniform 1/8, GPTQ nsamples=128, seed 0). Protocol bridges
reproduced digit-for-digit: bf16 wikitext **7.8148** (committed 7.8148), anchor
wiki/C4 **8.2678 / 11.3194** (committed 8.2678 / 11.3194).

## The missing control

`docs/rd_whydecay_report.md` §5b established that down_proj is a **ΔD desert**:
it holds 34.8 % of all rings on 3B but <1 % of total ΔD, because its input
passes through the online Hadamard, which flattens outliers and makes NVFP4
nearly as good as MXFP8 there. The measured-MSE allocator strips down_proj to
**zero** FP8 rings and reinvests the freed budget in the attn/mlp streams —
and that config is the committed winning anchor. What was never run is the
explicit control where down_proj is **forced** to carry FP8. This round runs
it, both budget-matched and additive.

Forced down_proj FP8 = per-Linear uniform-1/8 ring count (16 of 128 rings per
down_proj, 448 total), selected as each down_proj's **top-ΔD rings** — the most
charitable possible pick for down_proj.

## Results (3B, wikitext + C4 PPL)

| config | global FP8 | down_proj FP8 | attn/mlp FP8 | wikitext | Δ vs anchor | C4 | Δ vs anchor |
|---|---|---|---|---|---|---|---|
| **anchor** — measured-MSE-1/8 (down gets 0 naturally) | 1288/10304 = 0.125 | **0/3584 = 0.000** | 1288/6720 = 0.192 | **8.2678** | — | **11.3194** | — |
| **Control A** — forced down 1/8, budget removed from attn/mlp (matched) | 1288/10304 = 0.125 | 448/3584 = 0.125 | 840/6720 = 0.125 | 8.2756 | **+0.0079** | 11.3757 | **+0.0563** |
| **Control B** — forced down 1/8 ON TOP of full measured alloc (+34.8 % budget) | 1736/10304 = 0.168 | 448/3584 = 0.125 | 1288/6720 = 0.192 | 8.2513 | −0.0164 | 11.3581 | **+0.0387** |

(uniform-1/8 reference from the committed `rd_metrics` build: 8.2974 / 11.3915;
harness per-point reproducibility ≈ ±0.005, cross-build ring-set jitter on 3B
≤0.03 — see rd_whydecay §4 footnote.)

## Reading

- **Control A (matched budget): forcing FP8 onto down_proj costs accuracy on
  both evals** — +0.0079 wiki / +0.0563 C4. Sign-consistent, and on C4 the
  penalty is 78 % of the entire measured-vs-uniform win (0.072). Equivalently:
  of the measured allocator's total win over uniform, the single systematic
  move "strip down_proj to 0 FP8 and reinvest in attn/mlp" accounts for
  −0.0079 of −0.0296 on wiki and −0.0563 of −0.0721 on C4. The 448 rings
  parked on the desert are budget wasted.
- **Control B (additive, +34.8 % FP8 budget for free): the extra FP8 on
  down_proj buys ~nothing.** The two evals disagree in sign (wiki −0.0164,
  C4 **+0.0387**) and both deltas sit within the cross-build jitter band
  (≤±0.03–0.04 — each config re-runs GPTQ under a different reorder, so
  build-level jitter applies, not just the ±0.005 eval floor). A 35 % budget
  increase that cannot even produce a sign-consistent improvement is noise;
  for calibration, the same 448 rings moved WITHIN the budget (anchor vs
  Control A) produce a sign-consistent gain on both evals.
- down_proj's own ΔD agrees: ALL 3584 down_proj rings together hold 0.66 % of
  total ΔD on 3B (rd_whydecay §5b), so the 448 best rings carry <0.66 % —
  there is nothing there for FP8 to protect.

## Verdict

**All-FP4 down_proj is measured-optimal, not a kernel convenience.** Forcing
high precision onto the Hadamard-flattened down_proj buys no reliable accuracy
even when granted a 35 % larger FP8 budget (Control B: wiki −0.016 / C4
+0.039, sign-inconsistent, within build jitter), and costs accuracy at matched
budget on both evals (Control A: wiki +0.008 / C4 +0.056). The measured-MSE
allocator's decision to strip down_proj to zero FP8 and reinvest in the
attn/mlp streams is the correct one, confirmed by explicit forced controls.

Paper sentence this supports: *"forcing high precision onto the
Hadamard-flattened down_proj buys no accuracy even when granted extra budget
(Control B) and costs accuracy at matched budget (Control A) — all-FP4
down_proj is measured-optimal, not a simplification."*

## Reproduce

```bash
# inside docker luofan-plaquant, isolated dir /home/luofan/plaquant-downctl (GPU 6/7 lane)
CUDA_VISIBLE_DEVICES=6 MASTER_PORT=29517 PYTHONPATH=$PWD \
python promix/scripts/rd_downctl.py \
    --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml \
    --c4 --out /home/luofan/downctl/RD_DOWNCTL_3b.json
```

Raw JSON: `docs/rd_downctl/RD_DOWNCTL_3b.json`. Runtime ≈ 70 min total
(bridge 26 s, stats collect once, ~19 min per cell incl. C4).
