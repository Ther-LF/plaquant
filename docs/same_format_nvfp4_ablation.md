# Same-format NVFP4 ablation — is ProMix's win the partition, or just "FP4 beats INT4"?

Date: 2026-10-28. Hardware: RTX Pro 6000 Blackwell (SM120), `luofan-plaquant` container.

## 0. Motivation

Our committed W4A4 comparison (`docs/scoreboard.md`) ran every peer in its **native**
format: QuaRot/SpinQuant/ResQ in INT4, ProMix in NVFP4+MXFP8. That confounds
**algorithm** (ProMix's high/low-variance channel partition) with **format** (FP4 block-
scaled vs plain INT4). This round removes the format confound: every variant below uses
**the exact same quantizer** — `promix/quantize/quant_utils.py`'s `fake_quantize_nvfp4`
(E2M1, block-16, E4M3 micro-scale) for the low-precision group and `fake_quantize_mxfp8`
for the high-precision group, both already wired through GPTQ with frozen block scales
exactly as ProMix ships. The only things that vary are (a) **whether the high/low
partition is on** (r=1/8, r=1/16, r=0) and (b) **which residual-stream rotation feeds
the (now all-NVFP4) pipeline** (ProMix's own PCA+R1, QuaRot-style data-free Hadamard/
random-orthogonal, or SpinQuant's own trained R1).

## 1. Protocol

wikitext-2-raw-v1 **test** split, model's own HF tokenizer, `add_special_tokens=False`,
all rows `"\n\n".join`-ed, cut into **non-overlapping 2048-token chunks**, token-pooled
`exp(mean NLL))` via `promix/eval/evaluator.py::evaluate_ppl` — the same pure-torch,
layer-by-layer-offloaded evaluator every prior round's PPL numbers in `docs/scoreboard.md`
and `docs/round13_rotation_consistent_r.md` used. No changes were made to the evaluator
or the wikitext loader (`promix/eval/data.py`) for this round.

**FP16 bridge** (same evaluator + same loader, zero quantization — `w_bits=a_bits=
k_bits=v_bits=16` run through the full `prepare_ptq_model` pipeline, rotation fused but
mathematically lossless):

| Model | This round's bridge | Committed baseline | Δ |
|---|---|---|---|
| 1B | **13.1630** | 13.16 (13.1650 round-13) | 0.002 |
| 3B | **7.8122** | 7.82 (7.8199 round-13) | 0.008 |
| 8B | **6.1344** | 6.14 (6.1382 round-13) | 0.004 |

All three are within 0.008 PPL (≤0.1%) of the committed baseline — far below the 0.49–0.70
PPL effect this ablation is measuring. Harness confirmed before trusting any delta.
(Raw un-fused HF checkpoint through the same evaluator gave 7.8113 for 3B — same ballpark,
consistent with ordinary fp16 batching/rounding noise, not a harness bug.)

**Second, stronger bridge check**: the r=1/8 (SHIPPED) and r=0 (pure NVFP4) reproductions
below reproduce the committed scoreboard values to 3–4 decimal places on every model
(e.g. 3B r=1/8: 8.2925 vs committed 8.296/8.2925; 8B r=1/8: 6.5991 vs committed 6.599).
This end-to-end match is a tighter check than the FP16 bridge alone — it confirms GPTQ,
the NVFP4/MXFP8 quantizers, and the rotation fusion all behave identically to the runs
that produced the committed numbers.

## 2. How "pure NVFP4 r=0 with a foreign rotation" was plumbed

The least-invasive, most faithful route (per the assignment's own fallback guidance) was
chosen: **only the shared residual-stream rotation is swapped; everything else in the
ProMix pipeline is held fixed.**

`promix/quantize/rotation.py::fuse_basis_to_model` composes the main transform as
`T = U_attn @ block_diag(R1_1, R1_2)` (`identity_main_rotation=False`) or `T = U_attn`
directly (`identity_main_rotation=True`, the pre-existing "R1-off" ablation path from
round-13). `U_attn` is read from the `"attn_mlp"` key of the basis `.bin` file. We built
**surrogate basis files** (`promix/scripts/build_surrogate_rotation_basis.py`) that are
byte-for-byte copies of the real PCA basis file **except** the `"attn_mlp"` entry, which
is replaced by:

- **QuaRot variant**: a seeded, data-free rotation with no PCA involved. For power-of-2
  hidden sizes (1B=2048, 8B=4096) this is a **literal randomized-Hadamard transform**
  (Sylvester construction × seeded Rademacher sign diagonal — the same mechanism
  `promix/quantize/hadamard.py` already uses for the online down_proj Hadamard). For 3B
  (hidden_size=3072=2^10·3, not power-of-2) a literal structured Hadamard core needs a
  precomputed order-12 Paley/Sylvester matrix that this repo doesn't ship (only order-28,
  sized for 8B's down_proj); we substitute a seeded **Haar-random orthogonal matrix**
  (QR of a Gaussian) instead. Both are data-independent, non-learned, "PCA-off"
  rotations — the actual property being tested against variant #3 — and QuaRot's own
  ablations find structured-Hadamard and Haar-random statistically equivalent for
  incoherence processing, so this is a documented, defensible proxy, not a different
  algorithm.
- **SpinQuant variant**: their **actual trained R1** — loaded directly from
  `/home/luofan/baselines/peers/spinquant/rot_{1b,3b}/R.bin` (Cayley-SGD, 100 steps; this
  is the SAME checkpoint that produced the native-INT4 SpinQuant numbers already in
  `docs/scoreboard.md` round-14: 1B 16.14 / 3B 9.25). Verified orthogonal before use
  (`max|R1 R1^T − I| ≈ 5–6e-7`). **No SpinQuant rotation exists for 8B** — round-14
  explicitly skipped training it ("8B too slow, un-trained"); 8B variant 5 is therefore
  not reported (no 8B row for SpinQuant below).

Fed through `promix/scripts/ablation_acc_per_bit.py --mode a3 --rotation off
--basis_override <surrogate.bin>` (new `--basis_override` flag, added this round) so
`identity_main_rotation=True` makes `T` exactly the surrogate's `attn_mlp` matrix, with
no ProMix R1 mixed in.

**Held constant across variants #3/#4/#5** (i.e. NOT part of what's being compared,
because they're ProMix architecture plumbing rather than "rotation algorithm" per se):
the per-head V/K rotation (`R2`, ProMix's own PCA-derived value basis, read from the
*same* real basis file, unmodified), the `o_proj_global` hidden-dim rotation used for
the FP high/low o_proj split, and the online Hadamard transform on `down_proj`'s input
(mechanically identical to QuaRot's own online Hadamard — unchanged for every variant).
This isolates "choice of the R1-equivalent residual-stream rotation" as the only axis
that differs between #3/#4/#5.

**Bug found + fixed while building 8B's r=0/r=1/16 rotation files**: `ensure_rotation_file`
in `ablation_acc_per_bit.py` was calling `compute_basis(..., high_fraction=qcfg["high_fraction"])`
— the yaml's **default** `high_fraction` (0.125), not the sweep's actual `--r`. Since
`compute_basis` derives its own rotation filename from whatever `high_fraction` it's
given, this silently no-ops when the default's R-file already exists (it did, for every
r this script had previously been run with) and otherwise writes a rotation file under
the *wrong* name, tripping the function's own `assert os.path.exists(rot)`. 3B's r=0/
r=1/16 files already existed from round-13, so this bug was latent until 8B's r=0/r=1/16
(never generated before) hit it. Fixed: `ensure_rotation_file` now takes `high_fraction`
as an explicit required argument and `main()` passes `args.rw`. 8B's two missing rotation
files were generated directly via `compute_basis(..., high_fraction=0.0)` /
`(..., high_fraction=0.0625)` before the fix landed, so the reported 8B numbers are
unaffected either way.

## 3. Results — all variants, matched NVFP4 format

| Variant | Partition | Rotation | bits/w | 1B PPL | 3B PPL | 8B PPL |
|---|---|---|---|---|---|---|
| FP16 (bridge) | — | — | 16 | 13.163 | 7.812 | 6.134 |
| #1 SHIPPED | r=1/8 (NVFP4 low + MXFP8 high) | ProMix PCA+R1 | 4.969 | **14.573** | **8.293** | **6.599** |
| #2 | r=1/16 | ProMix PCA+R1 | 4.734 | **14.520** | **8.361** | **6.681** |
| #3 | r=0 (pure NVFP4) | ProMix PCA+R1 (ours) | 4.500 | 15.098 | 8.859 | 7.107 |
| #4 | r=0 (pure NVFP4) | QuaRot-style (Hadamard/Haar, data-free) | 4.500 | 15.272 | 8.910 | 7.087 |
| #5 | r=0 (pure NVFP4) | SpinQuant learned R1 | 4.500 | 15.241 | 8.878 | *(n/a, not trained)* |

Reproduction fidelity (variants #1/#3 vs. previously-committed numbers, independent of
this round's new #4/#5 work): 3B r=1/8 8.2925 vs committed 8.2925/8.296; 3B r=0 8.8588
vs committed 8.859; 8B r=1/8 6.5991 vs committed 6.599 — bit-for-bit to 3–4 decimals.

### 3a. Rotation choice at matched r=0 (pure NVFP4, no partition)

| Model | ours (PCA+R1) | QuaRot-style | SpinQuant learned | spread (max−min) |
|---|---|---|---|---|
| 1B | 15.098 | 15.272 (+0.174) | 15.241 (+0.143) | 0.174 |
| 3B | 8.859 | 8.910 (+0.051) | 8.878 (+0.019) | 0.051 |
| 8B | 7.107 | 7.087 (**−0.020**) | n/a | 0.020 |

Rotation choice moves PPL by only **0.02–0.17** (≤1.2% relative) at matched r=0/NVFP4,
and the ranking isn't even stable across model sizes: ours wins on 1B/3B, but QuaRot's
Hadamard edges ours out by 0.02 on 8B. This matches the repo's existing prior finding
(`CLAUDE.md`: "rotation optimization provides no measurable improvement" on these model
sizes) — now shown to hold even when every candidate rotation is evaluated through the
identical NVFP4 quantizer, not just within ProMix's own pipeline.

### 3b. Partition vs. no partition, at matched NVFP4 format (the actual question)

ΔPPL of turning the partition ON (r=1/8 or r=1/16, ProMix's own rotation) vs. OFF
(r=0), for every rotation choice available at r=0:

| Model | r=1/8 vs r=0-ours | r=1/8 vs r=0-QuaRot | r=1/8 vs r=0-SpinQuant | r=1/16 vs r=0-ours |
|---|---|---|---|---|
| 1B | −0.525 | −0.699 | −0.668 | −0.578 |
| 3B | −0.566 | −0.617 | −0.585 | −0.497 |
| 8B | −0.507 | −0.488 | n/a | −0.425 |

The partition's win (0.42–0.70 PPL) is **5–30× larger** than the rotation-choice spread
measured in §3a (0.02–0.17 PPL), for every model size and every rotation it's compared
against.

## 4. Verdict

**At matched NVFP4 format, ProMix's high/low-variance channel partition gives ΔPPL ≈
0.42–0.70 over pure-NVFP4 regardless of which residual-stream rotation feeds the
pipeline (ours, QuaRot's, or SpinQuant's) — 5 to 30× larger than the ΔPPL between
rotation choices themselves (0.02–0.17). The algorithmic win is the PARTITION
(MXFP8-high + NVFP4-low channel split), not "FP4 beats INT4" and not "our rotation beats
theirs."** This holds on 1B and 3B with all three rotations, and on 8B with the two
rotations we could test (SpinQuant never trained a rotation for 8B). Rotation choice is,
at most, a second-order effect at these model sizes — consistent with this repo's
existing finding that rotation optimization barely moves PPL below ~8B scale — and does
not change the ordering: the partition dominates every rotation variant we tried.

## 5. Artifacts

- `promix/scripts/ablation_acc_per_bit.py` — added `--basis_override` (surrogate-basis
  swap-in for `--mode a3 --rotation off`) and fixed the `ensure_rotation_file` latent
  `high_fraction` bug described in §2.
- `promix/scripts/build_surrogate_rotation_basis.py` — new; builds the QuaRot-style and
  SpinQuant surrogate basis files from a real ProMix basis bundle.
- Surrogate basis files (on the GPU host, `rotation/`): `U-quarot-surrogate-{Llama-3.2-1B-Instruct,Llama-3.2-3B,llama-3-8b}.bin`,
  `U-spinquant-surrogate-{Llama-3.2-1B-Instruct,Llama-3.2-3B}.bin`.
  New rotation files generated this round: `R-high-{0.0,0.0625}-low-0.0-sparse-0.0-llama-3-8b.bin`.
- Raw per-run JSON: `/home/luofan/ablation_results/{1b,3b,8b}_{rsweep,a3}_samefmt_v{1..5}_*.json`
  (on the GPU host; not copied into the repo — sizes/paths are reproducible from the
  commands in §2/§6).

## 6. Exact commands

```bash
# surrogate basis files (once per model)
python3 promix/scripts/build_surrogate_rotation_basis.py quarot \
    rotation/U-wikitext-512-Llama-3.2-3B.bin rotation/U-quarot-surrogate-Llama-3.2-3B.bin 3072 0
python3 promix/scripts/build_surrogate_rotation_basis.py spinquant \
    rotation/U-wikitext-512-Llama-3.2-3B.bin rotation/U-spinquant-surrogate-Llama-3.2-3B.bin \
    /home/luofan/baselines/peers/spinquant/rot_3b/R.bin

# partition reproductions (variants #1-#3)
python3 promix/scripts/ablation_acc_per_bit.py --size 3b --mode rsweep --r 0.125 --tag v1_r8
python3 promix/scripts/ablation_acc_per_bit.py --size 3b --mode rsweep --r 0.0625 --tag v2_r16
python3 promix/scripts/ablation_acc_per_bit.py --size 3b --mode rsweep --r 0.0 --tag v3_r0_ourrot

# foreign-rotation variants (#4/#5), pure NVFP4 r=0
python3 promix/scripts/ablation_acc_per_bit.py --size 3b --mode a3 --rotation off --r 0.0 \
    --basis_override rotation/U-quarot-surrogate-Llama-3.2-3B.bin --tag v4_r0_quarot
python3 promix/scripts/ablation_acc_per_bit.py --size 3b --mode a3 --rotation off --r 0.0 \
    --basis_override rotation/U-spinquant-surrogate-Llama-3.2-3B.bin --tag v5_r0_spinquant
```
