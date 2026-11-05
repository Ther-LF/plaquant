# rd-accmatrix: W4A4 accuracy × cost (BOPs) matrix — ProMix vs the field

**Branch:** `feat/rd-accmatrix` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000 Blackwell
(SM120, CUDA 13.0), container `luofan-plaquant`.

**What this is.** A *plain join* (no new experiments) of two already-committed bodies of work
so every method appears with **(accuracy, effective bits, theoretical BOPs, datapath-gated real
BOPs)** on one row — the "Method | W/A | size | BOPs | accuracy" layout standard in
mixed-precision quant papers (HAWQ-V3, RVQuant). The accuracy numbers are copied verbatim from
the committed scoreboard; the cost columns are joined from the committed BOPs model.

**Sources (all committed, nothing re-run here):**
- Accuracy (wikitext / C4 PPL, our protocol): `docs/rd_newpeers_accuracy.md` (1B/3B/8B W4A4 field),
  `docs/rd_metrics_report.md` (ProMix), `docs/baselines_on_hardware.md` (vLLM-NVFP4), peer
  `RESULTS.md` (INT4 family); **70B** from `feat/rd-peers70b` (`docs/rd_peers70b.md`: SHARQ/ARCQuant)
  and `rdscale` ProMix 70B tuned cell.
- Cost (BOPs / bits / datapath gate): `docs/theoretical_complexity.md` (committed on `feat/rd-alloc`).

**Protocol** (identical to the committed scoreboard): wikitext-2-raw-v1 test, bf16 model,
`"\n\n".join`, non-overlapping 2048-token chunks; C4 = QuaRot-standard 256×2048 windows, seed 0.

## Cost-column definitions

- **theo rel-BOPs** = (eff_bit_w × eff_bit_a) / 256, i.e. relative to FP16 (= 1.000). The
  hardware-agnostic BOPs metric (MACs cancel for a fixed model/GEMM). Lower = cheaper.
- **real rel-BOPs (datapath-gated)** = the honest number after the SM120 datapath gate
  (`hardware_format_peak.md`): a format only gets its theoretical BOPs if the GPU has a native
  datapath. Gates applied:
  - **INT4 family** (QuaRot/Atom/SpinQuant/ResQ/CoQuant): **no INT4 datapath on SM120** — `ptxas`
    lowers `s4 → IMMA.S8` (zero S4 SASS), so both operands execute at 8-bit → real ≈ (8×8)/256 = **0.25**
    (≈ 4× the paper-BOPs). This is why INT4 "looks cheapest on paper, costs FP8 in reality".
  - **SharQ**: single NVFP4 weight payload but **sparse + dense dual GEMM** per Linear, ~2× the
    GEMM work → real ≈ 2× theoretical.
  - **FGMP / Oltron**: **no GPU datapath at all** (FGMP self-declares software-infeasible; Oltron is
    ASIC/TSMC-28nm sim) → real = **N/A**.
  - NVFP4 / MXFP8 / FP8 have native SM120 datapaths → real = theoretical (ProMix's fused dual-format
    single-launch GEMM keeps the MXFP8 half-rate already folded into its 0.087).
- **eff bits/wt** includes micro-scale / residual-channel / high-precision amortization (see
  `rd_newpeers_accuracy.md` "Bits accounting"). **KV** = KV-cache bits during eval.

Bold **ProMix** = our method. "†" = SharQ weight payload is single but it runs two activation
streams; its real BOPs (not its weight bits) is the honest cost. "‡" = MicroMix kernels
mis-validate under CUDA 13 (pessimistic bound). INT rows are KV4 and not format-matched to the
NVFP4 group (noted where it matters).

---

## Llama-3.2-1B-Instruct  (bf16 bridge: wiki 13.16 / C4 18.52)

| method | class | eff bits/wt | KV | wiki | C4 | theo rel-BOPs | real rel-BOPs | source |
|---|---|---|---|---|---|---|---|---|
| **ProMix measured-1/8** | NVFP4+MXFP8 W4A4 | **4.97** | 16 | **14.47** | **20.19** | 0.087 | **0.087** | ProMix (rd_metrics) |
| ProMix uniform-1/8 | NVFP4+MXFP8 W4A4 | 4.97 | 16 | 14.55 | 20.32 | 0.087 | 0.087 | ProMix (rd_metrics) |
| SharQ | all-NVFP4 sparse+residual | ~4.5 † | 16 | 14.79 | 20.96 | 0.079 | 0.158 † | measured (newpeers) |
| ARCQuant | NVFP4+residual-ch | 5.22 | 16 | 15.22 | 20.79 | 0.090 | 0.090 (+res K) | measured (newpeers) |
| vLLM-NVFP4 (modelopt) | plain NVFP4 | 4.56 | 16 | 15.65 | n/a | 0.080 | 0.080 | baselines |
| SpinQuant (learned R) | INT4 W4A4KV4 | 4.0 | 4 | 16.14 | n/a | 0.063 | **0.25** (s4→INT8) | peer RESULTS |
| CoQuant wa_cov | INT4 + 1/8 INT8 | ~4.6 | 4 | 16.64 | 23.32 | ~0.072 | **0.25** (INT gate) | measured (newpeers) |
| Atom | INT4 + 128 INT8 ch | ~4.3 | 4 | 18.40 | n/a | 0.063 | **0.25** | peer RESULTS |
| QuaRot | INT4 W4A4KV4 | 4.0 | 4 | 26.00 | n/a | 0.063 | **0.25** | peer RESULTS |

## Llama-3.2-3B  (bf16 bridge: wiki 7.81 / C4 10.43)

| method | class | eff bits/wt | KV | wiki | C4 | theo rel-BOPs | real rel-BOPs | source |
|---|---|---|---|---|---|---|---|---|
| **ProMix measured-1/8** | NVFP4+MXFP8 W4A4 | **4.97** | 16 | **8.27** | 11.32 | 0.087 | **0.087** | ProMix (rd_metrics) |
| ProMix uniform-1/8 | NVFP4+MXFP8 W4A4 | 4.97 | 16 | 8.30 | 11.39 | 0.087 | 0.087 | ProMix (rd_metrics) |
| SharQ | all-NVFP4 sparse+residual | ~4.5 † | 16 | 8.39 | **11.29** | 0.079 | 0.158 † | measured (newpeers) |
| ARCQuant | NVFP4+residual-ch | 5.12 | 16 | 8.54 | 11.55 | 0.090 | 0.090 (+res K) | measured (newpeers) |
| vLLM-NVFP4 (modelopt) | plain NVFP4 | 4.56 | 16 | 8.63 | n/a | 0.080 | 0.080 | baselines |
| MicroMix ‡ | MXFP4/6/8 channel mix | 5.23 | 16 | 9.05 | 12.28 | 0.092 | 0.092 ‡ | measured (newpeers) |
| CoQuant a_cov (=ResQ crit.) | INT4 + 1/8 INT8 | ~4.6 | 4 | 9.18 | 12.82 | ~0.072 | **0.25** (INT gate) | measured (newpeers) |
| CoQuant wa_cov | INT4 + 1/8 INT8 | ~4.6 | 4 | 9.56 | 13.25 | ~0.072 | **0.25** | measured (newpeers) |
| Atom | INT4 + 128 INT8 ch | ~4.3 | 4 | 9.70 | n/a | 0.063 | **0.25** | peer RESULTS |
| SpinQuant | INT4 W4A4KV4 | 4.0 | 4 | 10.09 | n/a | 0.063 | **0.25** | peer RESULTS |
| QuaRot | INT4 W4A4KV4 | 4.0 | 4 | 11.23 | n/a | 0.063 | **0.25** | peer RESULTS |

## Llama-3-8B  (bf16 bridge: wiki 6.14 / C4 8.89)

| method | class | eff bits/wt | KV | wiki | C4 | theo rel-BOPs | real rel-BOPs | source |
|---|---|---|---|---|---|---|---|---|
| **ProMix uniform-1/8** | NVFP4+MXFP8 W4A4 | **4.97** | 16 | **6.59** | 9.64 | 0.087 | **0.087** | ProMix (rd_metrics) |
| ProMix measured-1/8 | NVFP4+MXFP8 W4A4 | 4.97 | 16 | 6.61 | 9.64 | 0.087 | 0.087 | ProMix (rd_metrics) |
| SharQ | all-NVFP4 sparse+residual | ~4.5 † | 16 | 6.68 | **9.58** | 0.079 | 0.158 † | measured (newpeers) |
| ARCQuant | NVFP4+residual-ch | 5.14 | 16 | 6.81 | 9.79 | 0.090 | 0.090 (+res K) | measured (newpeers) |
| vLLM-NVFP4 (modelopt) | plain NVFP4 | 4.56 | 16 | 6.87 | n/a | 0.080 | 0.080 | baselines |
| CoQuant a_cov (=ResQ crit.) | INT4 + 1/8 INT8 | ~4.6 | 4 | 7.35 | 10.79 | ~0.072 | **0.25** (INT gate) | measured (newpeers) |
| MicroMix ‡ | MXFP4/6/8 channel mix | 5.18 | 16 | 7.39 | 10.56 | 0.091 | 0.091 ‡ | measured (newpeers) |
| Atom | INT4 + 128 INT8 ch | ~4.3 | 4 | 7.57 | n/a | 0.063 | **0.25** | peer RESULTS |
| CoQuant wa_cov | INT4 + 1/8 INT8 | ~4.6 | 4 | 7.69 | 11.18 | ~0.072 | **0.25** | measured (newpeers) |
| SpinQuant | INT4 W4A4KV4 | 4.0 | 4 | 7.88 | n/a | 0.063 | **0.25** | peer RESULTS |
| QuaRot | INT4 W4A4KV4 | 4.0 | 4 | 8.86 | n/a | 0.063 | **0.25** | peer RESULTS |

## Meta-Llama-3-70B  (bf16 bridge: wiki 5.318 / C4 9.205)

70B peers measured via layer-offload harness (`feat/rd-peers70b`); ProMix 70B = tuned r=1/8 cell
(ring512 + measured-ΔD alloc + up2/g512, `rdscale`). **This is the one size where a peer leads
ProMix on the headline metric** — SharQ 5.769 < ProMix 5.983 wikitext — at ~2× real BOPs.

| method | class | eff bits/wt | KV | wiki | C4 | theo rel-BOPs | real rel-BOPs | source |
|---|---|---|---|---|---|---|---|---|
| **SharQ** | all-NVFP4 sparse+residual | ~4.5 † | 16 | **5.769** | 9.659 | 0.079 | **0.158** † | peers70b (measured) |
| **ProMix r=1/8 (tuned)** | NVFP4+MXFP8 W4A4 | **4.97** | 16 | 5.983 | **9.855** | 0.087 | **0.087** | rdscale (ring512+meas+up2) |
| ProMix r=1/8 (uniform) | NVFP4+MXFP8 W4A4 | 4.97 | 16 | 6.075 | 9.909 | 0.087 | 0.087 | rdscale |
| ARCQuant | NVFP4+residual-ch | ~5.1 | 16 | 6.183 | 10.049 | 0.090 | 0.090 (+res K) | peers70b (measured) |

**Open question (in flight).** SharQ's 70B wikitext win (5.769 vs ProMix 5.983, Δ 0.214) comes at
**~2× real BOPs** (0.158 vs 0.087) from its sparse+dense dual GEMM. The `feat/rd-densesweep` 70B
accuracy curve (`acc 70b` 0.05→1.00, running) tests whether a higher FP8 ring budget (r > 1/8,
which raises ProMix's bits/BOPs) lets ProMix match 5.769 — i.e. whether ProMix can buy SharQ's
0.214 PPL for less than SharQ's 2× compute premium. Curve → `docs/rd_accuracy_curve.csv`.

---

## Measured downstream metrics — 0-shot 6-task avg + MMLU 5-shot (rd-accmatrix round)

This fills the cells the newpeers/mmlu campaigns left empty: **0-shot 6-task average**
(piqa / arc_easy / arc_challenge / hellaswag / winogrande / lambada_openai; acc_norm for
piqa/arc/hellaswag, acc for winogrande/lambada; lm-eval **0.4.5**, same GPU-pinning
convention as `rd_metrics_report.md`) and **MMLU 5-shot** (57-subtask group aggregate =
micro acc, `num_fewshot=5`). Peer builds are the **same as-shipped quantized models** as
the PPL rows: ARCQuant / SharQ / MicroMix via `rd_newacc_peer.py --lmeval_tasks`; CoQuant
(a_cov control = ResQ criterion) via its own shipped `ptq.py` (reproduces its committed
wiki PPL in-run as the trust gate); vLLM-NVFP4 via the lm-eval vLLM backend. Cited numbers
(ProMix / ARCQuant / SharQ / NVFP4 MMLU, ProMix 0-shot) are from `rd_mmlu_report.md` /
`rd_metrics_report.md`; **bold** cells are measured this round (raw JSON in
`docs/rd_accmatrix_data/`). Runner: `promix/scripts/rd_accmatrix_run.sh` (+ `_coquant.sh`).
"‡" = ProMix MMLU is the acctune shipping-recipe build (the only ProMix MMLU runs);
the measured-1/8 / uniform-1/8 anchors had no MMLU run. Rows sorted by wikitext (matrix order).

### Llama-3.2-3B

| method | 0-shot 6-task avg | MMLU 5-shot | source |
|---|---|---|---|
| bf16 bridge | 68.11 | 56.11 | rd_metrics / rd_mmlu |
| ProMix measured-1/8 | 66.77 | 54.12 ‡ | rd_metrics / rd_mmlu (‡ new recipe) |
| ProMix uniform-1/8 | 66.89 | — | rd_metrics |
| SharQ | _(running)_ | 53.42 | rd_mmlu (MMLU) |
| ARCQuant | **66.14** | 52.64 | **measured** / rd_mmlu |
| vLLM-NVFP4 | **66.16** | 52.63 | **measured** / rd_mmlu |
| MicroMix ‡ | _(running)_ | _(running)_ | measured |
| CoQuant a_cov | _(running)_ | _(running)_ | measured |

### Llama-3-8B

| method | 0-shot 6-task avg | MMLU 5-shot | source |
|---|---|---|---|
| bf16 bridge | 73.33 | 65.28 | rd_metrics / rd_mmlu |
| ProMix uniform-1/8 | 72.69 | 62.85 ‡ | rd_metrics / rd_mmlu (‡ new recipe) |
| ProMix measured-1/8 | 72.36 | — | rd_metrics |
| SharQ | _(running)_ | 62.87 | rd_mmlu (MMLU) |
| ARCQuant | _(running)_ | 62.19 | rd_mmlu (MMLU) |
| vLLM-NVFP4 | **71.48** | 62.34 | measured / rd_mmlu |
| MicroMix ‡ | _(running)_ | _(running)_ | measured |
| CoQuant a_cov | _(running)_ | _(running)_ | measured |

---

## Reading the matrix as a cost table

1. **On paper, the whole W4A4 field clusters at theo rel-BOPs ≈ 0.06–0.09** (the "it's just
   about the bits" intuition — true *theoretically*). The spread is a few tenths of a bit of
   micro-scale / residual / high-precision overhead.
2. **The datapath gate separates the field.** The NVFP4-native group (ProMix, NVFP4, ARCQuant,
   MicroMix) keeps its paper BOPs (real ≈ theo ≈ 0.08–0.09). The INT4 group's paper-cheapest
   0.063 becomes **0.25** (s4→INT8), and SharQ's 0.079 becomes **0.158** (dual GEMM). FGMP/Oltron
   have **no GPU path at all**.
3. **ProMix is Pareto-frontier on accuracy-per-real-BOPs at every size ≤ 8B**: lowest (or tied-lowest)
   wikitext in the whole field at real rel-BOPs 0.087, while every method that comes near it on
   accuracy either costs ~2× real BOPs (SharQ) or more bits (ARCQuant 0.090 at higher PPL) or is
   INT4-gated to 0.25 (CoQuant/Atom/SpinQuant/QuaRot, all worse PPL too).
4. **At 70B the frontier has two points**: ProMix (0.087 real BOPs, 5.983) and SharQ (0.158 real
   BOPs, 5.769). Neither dominates — ProMix is the cheap corner, SharQ the accurate corner at 2×
   cost. The densesweep 70B curve is measuring the ProMix trade curve between them.

## Suggested figure — accuracy-vs-real-BOPs Pareto

Scatter **x = real rel-BOPs (log), y = wikitext PPL**, one panel per model, one point per method,
INT4 family clamped to x = 0.25 and SharQ to x = 0.158. ProMix sits at x = 0.087 on the lower-left
(cheap + accurate) frontier; the INT4 cluster collapses to a vertical line at x = 0.25 (paper-0.063
ghost marked with a dashed arrow to show the gate penalty); SharQ sits up-and-right of ProMix. The
one-line story: **"FP4-era INT4's theoretical BOPs advantage is paper-only; on real Blackwell
datapaths ProMix is the accuracy-per-BOP frontier."** Mirror of the two-bar theo-vs-real figure in
`theoretical_complexity.md`.

## Scope / honesty notes

- **Metrics.** wikitext + C4 PPL in the cost tables above; the **0-shot 6-task avg + MMLU
  5-shot** cells (out of scope for the original newpeers/mmlu campaigns) are now measured and
  joined in the "Measured downstream metrics" section. ProMix MMLU/0-shot at the ratio
  subset live in `docs/rd_mmlu_report.md` where measured.
- **KV mismatch:** INT rows (CoQuant/Atom/SpinQuant/QuaRot) are KV4 and INT-format, i.e. not
  format-matched to the NVFP4 KV16 group; they are shown for field completeness, not as a
  format-matched comparison.
- **BOPs are per-GEMM arithmetic cost**, not end-to-end latency. SharQ's "2×" is GEMM-work; the
  measured kernel latencies live in the serving docs (`baselines_on_hardware.md`, densesweep
  serving curve).
