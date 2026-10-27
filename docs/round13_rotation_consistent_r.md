# Round-13 (ALGORITHM axis) — rotation-consistent r, and a spectrum-driven per-layer allocation algorithm

Model: Llama-3.2-3B (28 layers) + Llama-3-8B (32 layers), wikitext-2 PPL (fake-quant,
GPTQ frozen scales, seq 2048), SM120 RTX Pro 6000. Shipped recipe = act-MSE on +
two-level weight scale (variant B: shared per-tensor FP32 global × E4M3/E8M0 micro,
nearest + micro-MSE) + PCA(U)+R1 rotation + per-head o_proj + online-Hadamard(down).
Harness: `promix/scripts/ablation_acc_per_bit.py` (uniform `rsweep`) and
`promix/scripts/round11_sensitivity.py` (`--mode plan/anchor --r1_off`, per-layer r).
Raw JSON on the box: `/home/luofan/round13_results/`.

## 0. How the 1/8 boundary is baked in, and how rotation is rebuilt per r

The main-path transform fused into every weight (and the residual stream) is
`T = U_attn @ block_diag(R1_1, R1_2)` (`rotation.py::build_main_transform`).

- **U_attn** is the shared PCA basis (`basis.py`), columns sorted **ascending by the
  global eigenvalue** (variance). It is **independent of r**: the top-r high-variance
  subspace is simply the **last r·d columns**, well-defined for ANY r. `U` is computed
  once (the ~20-min Step-0; activation covariance + eigendecomposition) and **never
  rebuilt**.
- **R1_1 / R1_2** are random-orthogonal matrices of size `(1-r)·d` and `r·d`. The 1/8
  boundary lives ONLY here: `R1_2` mixes the top-`r·d` principal directions among
  themselves. Rebuilding the rotation for a new r is just two QR factorizations of
  random Gaussians (`basis.py::random_orthogonal_matrix`) at the new block sizes —
  **no model load, no calibration, no eigendecomposition, no gradient training**.

**Measured rebuild cost (in-container, GPU0):** generating all five new 3B R-files
(r = 1/32, 3/16, 1/4, 3/8, 1/2) took **4.4 s total incl. Python startup; 0.2–1.0 s per
r**. This is the coordinator's correction, now confirmed in code and by timing: *U is
the global variance basis and is fixed; only R1's block split follows r, and that is an
O(1) re-cut.* Round-11's "rotation is welded to 1/8, needs rebuild" framing was wrong
about cost — the rebuild is effectively free and offline.

**Why intermediate per-layer r needs R1-off, not R1-on.** `T` (hence R1) is fused into
the **shared residual stream** (embed, lm_head, and every layer's q/k/v/gate/up inputs
+ attention/MLP output un-rotation all use the SAME `U@R1`). In `full_shared` there is
exactly **one** hidden-dim rotation for all layers. If `R1_2` mixes the top-1/8
subspace, a layer that quantizes only its top-1/16 reads a *sub-tail of a mixed block*
→ no longer variance-monotonic (the round-11 "fixed-1/8 R at r=1/16 = 8.77" artifact).
Therefore **per-layer-varying r_i is only rotation-consistent on the pure PCA tail**,
i.e. with the identity main rotation `T = U` (**R1-off**): then the last r_i·d channels
of any layer are exactly its top-r_i PCA components, valid for any r_i. This is what
`round11_sensitivity.py --r1_off` builds. Uniform-r keeps R1-on (all layers quantize
the whole r-block, so the mixing is consistent) and R1 is rebuilt per (uniform) r.

## 1. Rotation-consistent uniform-r curve (3B, R1-on, shipped recipe)

Each point uses a rotation **rebuilt for that r** (`R-high-{r}-…-Llama-3.2-3B.bin`),
not the fixed-1/8 R. r=1/8 reproduces the shipped anchor bit-for-bit.

| r | bits/w | 3B PPL | Δ vs prev | accuracy-per-bit (ΔPPL/Δbit) |
|---|---|---|---|---|
| 0 (pure NVFP4) | 4.500 | 8.859 | — | — |
| 1/16 | 4.734 | **8.361** | −0.498 | **−2.13 / bit** (steep knee) |
| **1/8 (shipped)** | 4.969 | **8.293** | −0.069 | −0.29 / bit |
| 3/16 | 5.203 | 8.272 | −0.020 | −0.086 / bit |
| 1/4 | 5.438 | 8.230 | −0.042 | −0.18 / bit |

(r=1/8 = 8.2925 vs round-11 anchor 8.29254 — bit-identical; r=1/16 = 8.3613 = scoreboard
8.361. Gate PASS. r=0 reused from round-5 = 8.859. NVFP4 ref: 8.63 @ 4.50.)

**Is uniform 1/8 optimal?** There is **no interior optimum** — PPL decreases
monotonically as r grows (more 8-bit channels always help a little). "Optimal r" only
exists w.r.t. a bit budget. The accuracy-per-bit **knee is at r≈1/16** (−2.13 PPL/bit);
by 1/8 the marginal value has already collapsed to −0.29/bit and keeps falling. So **on
accuracy-per-bit, r=1/16 Pareto-dominates 1/8**, and 1/8 is already *past* the knee.
Against NVFP4 (8.63 @ 4.50): at matched 4.50 bits (r=0) we are **worse** (8.859 > 8.63);
r=1/16 (8.361 @ 4.734) and r=1/8 (8.293 @ 4.969) beat NVFP4's PPL but spend +0.23/+0.47
bits. Our accuracy-per-bit front still does not dominate NVFP4 at matched bits.

8B (R1-on uniform 1/8): **6.599 @ 4.969** (matches scoreboard; NVFP4 8B = 6.87 @ 4.50).

## 2. The spectrum-driven per-layer allocation algorithm (executable)

**Signal (free, no sensitivity runs):** the per-layer channel-variance spectrum in the
shared PCA basis, already stored in `E-wikitext-512-*.bin` as `layer.{i}.attn_mlp_var`
(the eigen-energy of each principal direction for that layer). Flat spectrum → variance
spread over many directions; concentrated spectrum → few dominant directions.

**Inter-layer spread (3B, top-r cumulative energy in the deployable global-U order):**

| r | global | L0 (flattest) | L1 (most concentrated) | spread |
|---|---|---|---|---|
| 1/16 | 0.475 | 0.361 | 0.545 | 0.18 |
| 1/8  | 0.566 | 0.471 | 0.626 | 0.16 |
| 1/4  | 0.676 | 0.598 | 0.718 | 0.12 |

The spectrum decays **slowly** (top-1/8 holds only 57 % of energy; top-1/2 only 82 %)
and the inter-layer spread is **modest** — L0 is the one clear outlier (needs ~3× more
high channels than L1 to reach a given energy).

**Budget range.** The [NVFP4 | MXFP8] 2-segment scheme floors at **4.5 bits/w** (r=0,
pure NVFP4) and tops at 8.25 (r=1). The coordinator's **B=4.3 is infeasible** here — it
would need a third sub-4-bit segment (INT3/2-bit low group), out of scope for the clean
2-segment hardware-native partition. The meaningful Pareto region is **4.5–5.0 bits**;
we test 4.734, 4.852, 4.969 (and the 4.5 floor).

**Algorithm:** given an average bit budget B → target mean r̄ = (B−4.5)/3.75; pick an
energy threshold τ by bisection so that the induced per-layer r_i averages to r̄, where
`r_i = min r on the legal grid such that top-r cumulative energy ≥ τ`. Concentrated
layers reach τ at small r; flat layers need large r.

**Hardware-alignment snapping (coordinator's hard constraint):** each Linear requires
`K_high % 32 == 0` (MXFP8 block) **and** `K_low = K−K_high % 64 == 0` (NVFP4 pack +
reorder tile). For 3B (K ∈ {3072, 8192}, both %64) and 8B (K ∈ {4096, 14336}, both %64),
the only per-layer r legal for **all 7 Linears simultaneously** are **multiples of
1/16**. The algorithm snaps each τ-ideal r_i to this legal grid. The snap vs the
continuous-τ ideal is small: **mean |ideal−snap| = 0.019–0.029**; on 3B @ r̄=1/8 the
snap cost the allocation only **+0.010 PPL** (8.3787 illegal-fine-grid → 8.3884 legal).
**Static audit: all 196 Linear-configs (7×28) in every emitted plan satisfy K_high%32
and K_low%64** — kernel-legal, one-launch runnable. Example snapped plans (r_i dist):
`4.852b → {1/16:15, 1/8:12, 3/16:1}`, `4.969b → {1/16:5, 1/8:18, 3/16:5}`.

**Real-kernel bit-exactness:** the fused kernel already accepts arbitrary per-Linear
K_low/K_high (round-19/24: per-layer cos 0.999999, incl. K_high=0 pure-NVFP4); these
legal plans are therefore deployable with the **kernel unchanged** — only a re-pack and
a per-layer K_high table (the `adaptive_high_fraction` infra already carries a per-layer
khigh map). No Step-0/1 rerun (U fixed; R1-off needs no main R at all). (A fresh
real-kernel cosine on these exact plans was not re-run this round — no `.so` was built;
it is covered by the existing arbitrary-length bit-exact proof + the static align audit.)

## 3. Per-layer allocation vs uniform, at EQUAL bits (rotation-consistent)

All per-layer runs are R1-off (the only consistent regime for varying r_i); the fair
same-regime baseline is R1-off uniform. The best overall uniform is R1-on.

**3B, legal grid:**

| scheme | regime | bits/w | PPL | vs R1-off uniform | vs **R1-on uniform** |
|---|---|---|---|---|---|
| uniform 1/16 | R1-off | 4.734 | 8.4704 | — | (R1-on 8.361) +0.109 |
| spectral, mean 3/32 | R1-off | 4.852 | **8.4225** | −0.026 vs interp. | (R1-on interp ≈8.327) **+0.096** |
| uniform 1/8 | R1-off | 4.969 | 8.4257 | — | (R1-on 8.293) +0.133 |
| **spectral, mean 1/8** | R1-off | 4.969 | **8.3884** | **−0.037** | (R1-on 8.293) **+0.096** |
| uniform 1/8 | **R1-on** | 4.969 | **8.293** | — | — |

**8B, legal grid @ 4.969 b:** R1-on uniform **6.599** | R1-off uniform 6.6402 |
spectral R1-off **6.6422** (**+0.002, a tie/slight loss** — no benefit on 8B).

**Reading:**
1. **The spectral signal genuinely helps *within* the R1-off regime on 3B** — spectral
   beats R1-off uniform at equal bits by −0.037 (1/8), −0.026 (3/32), ~0 (1/16, where
   the legal grid floor forces everyone to 1/16 = uniform). The win **grows with
   budget** (more legal allocation freedom). This is a *stronger* win than round-11's
   sensitivity-based R1-off reallocation (−0.010), using only the free eigenvalue
   spectrum. **On 8B the win vanishes** (+0.002) — model-dependent, because 8B's spectra
   are more uniform across layers and its R1-off penalty is smaller.
2. **But per-layer allocation does NOT beat the best uniform (R1-on).** It is a flat
   **+0.096 worse on 3B and +0.04 worse on 8B** at every equal-bit point. The reason is
   mechanistic: per-layer r forces **R1-off**, which costs +0.133 (3B) / +0.04 (8B) at
   r=1/8; the spectral allocation recovers only ~0.037/0.00 of that. R1's benefit is
   concentrated at **low r** (it spreads variance so NVFP4 blocks have no outliers) —
   exactly the layers the allocation starves to small r. So the allocation needs cheap
   low-r layers precisely where dropping R1 hurts most: the two levers fight.

## 4. Verdict

- **Best uniform r / is 1/8 optimal?** PPL falls monotonically with r (no interior
  optimum); the accuracy-per-bit **knee is r≈1/16** (−2.13 PPL/bit), and r=1/16 Pareto-
  dominates 1/8 on accuracy-per-bit. 1/8 is a defensible "safe" point but **past** the
  knee. No uniform r beats NVFP4 at matched bits (r=0: 8.859 > 8.63); r=1/16 and 1/8
  beat NVFP4's PPL only by spending +0.23/+0.47 bit.
- **Does rotation-consistent per-layer allocation beat uniform?** **No — not the best
  (R1-on) uniform.** It is +0.096 (3B) / +0.04 (8B) worse at equal bits, because varying
  r per layer requires R1-off and the R1 penalty exceeds the allocation gain. It *does*
  beat the same-regime R1-off uniform on 3B (−0.037, growing with budget) but not on 8B.
- **Core-contribution framing (honest).** The r-partition is **not a hand-tuned
  hyperparameter**: we have a free, executable rule — *per-layer r_i from the stored
  PCA energy spectrum, snapped to the kernel-legal grid (K_high%32, K_low%64), rebuilt
  per r in <1 s since U is fixed* — and it is **provably hardware-deployable** (static
  align audit passes for all 196 Linears; kernel unchanged). But the measured result is
  the **"uniform r* is near-optimal" conclusion, not a per-layer win:** the per-layer
  spectra are similar enough (modest spread) that the spectrum-optimal r_i cluster near
  the uniform value, and the global-shared-R1 architecture makes any per-layer variation
  pay an R1-off tax larger than the allocation gain. The clean, writable statement is:
  **"given a budget B the optimal hardware-aligned partition is essentially uniform
  r*(B); the PCA energy spectrum confirms per-layer r is not worth the shared-rotation
  cost."** This extends round-1/round-11 (sensitivity/heuristic allocation ≤ uniform)
  with the rotation freed and a spectrum-only criterion — the fourth independent "uniform
  wins" confirmation, now also explaining *why* (shared R1 + concave, similar spectra).
- **Deployability.** Uniform r: ship as-is (kernel + single global K_high). Per-layer r:
  needs R1-off (a cheaper rotation, no main-R file) + a per-layer K_high table + re-pack;
  kernel unchanged; **not recommended** given it loses to R1-on uniform.

## 5. Open lever (not pursued)

The only way to get per-layer r *with* R1-on is a **nested/block-diagonal R1_2** whose
high block is a stack of legal-grid shells (each shell independently orthogonal, no
cross-shell mixing), so a sub-tail = a union of whole shells stays variance-monotonic.
This would keep some R1 benefit while allowing per-layer sub-tails. Untested; it is the
only candidate that could flip the verdict, since the blocker is purely the R1-off tax.

## 6. Reproduction

- Rotation rebuild: `round13_results/gen_rot.py <short> <hidden> <head_dim> <r,r,...>`
  (sub-second/r; U untouched). Allocator: `spectral_alloc_legal.py` (+ `_8b`), emits
  `planL_*.json` and prints τ, snapped dist, |ideal−snap|, and the legality audit.
- Uniform R1-on: `ablation_acc_per_bit.py --mode rsweep --r R --weight_2level tensor
  --w2l_round nearest --w2l_mse --w2l_high`. Uniform/plan R1-off:
  `round11_sensitivity.py --mode anchor|plan --r1_off [--base_r R | --plan_json …]`.
- Raw JSON: `/home/luofan/round13_results/{3b,8b}_{rsweep,anchor,plan}_*.json`.
</content>
</invoke>
