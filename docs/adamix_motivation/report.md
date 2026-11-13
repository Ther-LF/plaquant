# AdaMix Motivation — report (CPU run, 2026-11-13)

Scope delivered this run (CPU-only, all 8 Pro6000 GPUs occupied → untouched):
**inventory** + **E1a** local controlled diagnostic (3B) + **E2 allocation
MATH** (no PPL). GPU work — E1b full-model PPL, E2-PTQ 72-config matrix, E4
kernel timing — is deferred. All numbers trace to
`runs/e1a_3b/` (manifest `run_id=e1a_Llama-3.2-3B_G512`) and `runs/e2math_*/`.

Model Llama-3.2-3B; shared PCA basis `U-wikitext-512-Llama-3.2-3B.bin`
(sha256 `3a402945c1d33f8b31b7`); deployed NVFP4(bs16)/MXFP8(bs32) fake-quant on
BOTH X and W; RTN, NO GPTQ; G=512; wikitext-2 fit/diag disjoint windows, 2048
tokens each; o_proj excluded. See `protocol.yaml` and `manifest.json`.

## E1a — representation × precision (4×2), 3B

Mean over all eligible layers, **fit** split. act_mse / out_mse = linear output
MSE / out_rel_sq = output relative squared error. q/k/v (attn) and gate/up (mlp)
use the SHIPPED shared PCA basis as P; down uses a LOCAL full PCA (isolated probe,
NOT the shipped 256-block-PCA+Hadamard path — flagged).

### q/k/v (attn-input)
| rep | policy | act_mse | out_mse | out_rel_sq |
|---|---|---|---|---|
| R0 (I) | U4 | 7.41e-3 | 6.67e-3 | 8.45e-3 |
| R0 | LDelta | 5.94e-3 | 5.33e-3 | 6.87e-3 |
| R1 (ring) | U4 | 7.48e-3 | 6.64e-3 | 9.18e-3 |
| R1 | LDelta | 6.03e-3 | 5.17e-3 | 7.31e-3 |
| R2 (PCA) | U4 | 7.36e-3 | **1.18e-2** | 1.07e-2 |
| R2 | LDelta | **3.42e-3** | 3.35e-3 | **4.19e-3** |
| R3 (PCA+ring) | U4 | 7.48e-3 | 8.24e-3 | 1.01e-2 |
| R3 | LDelta | **3.42e-3** | **2.56e-3** | **3.97e-3** |

### gate/up (mlp-input)
| rep | policy | act_mse | out_mse | out_rel_sq |
|---|---|---|---|---|
| R0 | U4 | 7.46e-3 | 2.23e-3 | 1.23e-2 |
| R0 | LDelta | 5.85e-3 | 1.74e-3 | 9.84e-3 |
| R2 | U4 | 7.37e-3 | 2.94e-3 | 1.42e-2 |
| R2 | LDelta | 3.39e-3 | 1.10e-3 | 6.45e-3 |
| R3 | U4 | 7.49e-3 | 2.44e-3 | 1.34e-2 |
| R3 | LDelta | 3.39e-3 | 1.04e-3 | 6.34e-3 |

### down (LOCAL PCA probe — caveat below)
| rep | policy | act_mse | out_mse | out_rel_sq |
|---|---|---|---|---|
| R0 | U4 | 6.68e-5 | 3.99e-4 | 1.61e-2 |
| R0 | LDelta | 5.70e-5 | 3.39e-4 | 1.38e-2 |
| R2 | U4 | 7.14e-5 | 5.42e-4 | 1.69e-2 |
| R2 | LDelta | **8.94e-6** | **6.81e-5** | **2.31e-3** |
| R3 | U4 | 7.67e-5 | 4.98e-4 | 1.71e-2 |
| R3 | LDelta | 1.03e-5 | 8.88e-5 | 2.52e-3 |

Diagnostic generalization (R3 LDelta, attn+mlp): fit out_rel_sq 4.914e-3 vs
**diag 4.960e-3** → masks selected on fit do NOT overfit. (Spearman of fit-vs-diag
per-group Delta ranking = 0.989, §E2 below.)

## E2 — allocation MATH (rep frozen; no PPL), 3B

Per-group `variance`, `d4=mse_fp4`, `d8=mse_fp8`, `signed_delta=d4-d8` cached in
`group_stats.csv`. K=196 FP8 groups of 1288 total (rho_actual=0.152; see rounding
note). Policies A-pos (per-layer top-k_l by variance), B-local (per-layer top-k_l
by Δ), C-global (global top-K by Δ), D-var (global top-K by variance).

| rep | Spearman(var,Δ) global | A∩B /196 | A∩C /196 | Spearman(fit,diag Δ) |
|---|---|---|---|---|
| R0 (I) | 0.871 | 111 (57%) | 91 | 0.980 |
| R1 (ring) | 0.882 | 114 (58%) | 91 | 0.982 |
| R2 (PCA) | 0.998 | 196 (100%) | 141 | 0.989 |
| R3 (PCA+ring) | 0.996 | 195 (99.5%) | 141 | 0.989 |

Byte accounting (R3): A/B 1.4753 GB weight + 0.1473 GB scale; C/D 1.4877 GB +
0.1465 GB — equal FP8 group count (196) but DIFFERENT bytes (out_features differ
across layers), confirming §3.3 "equal groups ≠ equal memory". Exchange identity
(move one FP8 group i→j ⇒ ΣMSE changes by Δ_i−Δ_j) holds by construction on the
cached d4/d8 (independent-group regime; no re-GPTQ, no scale coupling here).

## Answers to §9 questions (honest, scoped)

**Which transforms improve/worsen error, and is it consistent?**
Under **uniform FP4 (U4)**, PCA and ring rotation do **not** improve and the
linear **output** error gets **worse** (attn R2 U4 out_mse 1.18e-2 vs R0 6.67e-3;
every projection shows R2/R3 U4 out_rel_sq ≥ R0). So "PCA alone helps" is
**false** here — reported as a negative, consistent across q/k/v/gate/up/down.
Representation barely moves activation MSE under U4 (all ≈7.4e-3) and does not
change clipping (clip_fp4 0.055→0.060).

**Does representation change the EFFECT of precision selection? (Motivation 3.1)**
Yes, strongly. Under **LDelta** (one fixed FP8 group/layer by measured Δ), PCA
roughly **halves activation MSE** (7.4e-3→3.4e-3) and cuts output rel-sq-error
~40–55% vs both R0 U4 and R0 LDelta, across all projections (down even ~6×, with
the local-PCA caveat). The ring alone (R1) adds little; the gain is from PCA.
E2-math pins the mechanism: PCA makes per-group **variance an almost-perfect
proxy for measured Δ** (Spearman 0.87→0.998) and makes variance-selection
**coincide** with Δ-selection (A∩B 57%→100%). In the raw basis the selection
criterion has teeth (they disagree on 43% of groups); under PCA it does not.

**A→B (layer-internal selection) vs B→C (cross-layer quota):** on R3, A→B is
nearly a no-op (A∩B=195/196) because var≈Δ; the real mask change is the
per-layer→global **quota** reallocation (A/B vs C/D overlap only 141/196, ~28% of
groups move across layers). In the raw basis R0 that reallocation moves even more
(A∩C=91/196). PPL impact of these mask changes is **not measured** (GPU).

**C→D (Δ vs variance scoring):** on R2/R3 C≈D (overlap 195/196) — we therefore
**cannot** claim measured-format error always beats variance; on the PCA
representation they coincide. On R0 they diverge more. Per §5 guidance this is
reported as-is, not spun as a Δ win.

**Equal groups → memory/latency:** equal FP8 group count gives DIFFERENT weight
bytes (1.4753 vs 1.4877 GB) — do not call it iso-memory. Latency not measured
(E4, GPU).

**Writable in Motivation now:** (1) representation alone does not improve — and
can worsen — quantized linear error under uniform FP4; (2) representation must be
chosen **jointly** with precision allocation: PCA roughly halves mixed-precision
linear error AND turns cheap variance into a sufficient allocation statistic.
**Hypotheses needing GPU before claiming:** that these linear-error gains move
full-model PPL (E1b/E2-PTQ); that cross-layer quota reallocation (B→C) helps PPL;
cross-model (8B) generality. **Refuted here:** "PCA alone improves quality".

## Caveats / limits (do not over-read)
- **U4 vs LDelta differ in bit budget** — same-column (within one precision)
  comparisons isolate representation; U4↔LDelta differences include added bits.
- **down_proj R2/R3 use a LOCAL full PCA**, not the shipped 256-block-PCA+online-
  Hadamard; its strong numbers are an isolated change-of-basis probe only.
- **No PPL** (GPU). Linear output MSE ≠ model task loss.
- **rho_actual=0.152≠0.125**: G=512 on 3072-dim layers → 6 groups →
  k_l=round(0.75)=1 (rho 0.167); down 16 groups → k_l=2 (0.125). Honest per §2.1.
- **G=512 is coarse** on hidden (6 groups); finer G is E3, not run.
- Single calibration seed (seed0); 3 GPUs-free seeds are an E2-PTQ follow-up.
- RTN single-level weight quant (deployed two-level weight refinement off) for a
  clean control.

## Next CPU step
8B E1a + E2-math (same two commands, 8B checkpoint/basis; ~2.5× slower, CPU-doable
once its basis is located). Everything else (E1b/E2-PTQ/E4) needs a free GPU.
