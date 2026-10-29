# Loss-Weighted Rate-Distortion Allocation (Fisher proxy)

**Branch** `feat/rd-lossweight` · **date** 2026-10-29 · spec §9 fallback / §4.5
refinement · follows `docs/rd_ron_report.md`

## Question

The shipped allocator ranks 64-rings by **pure local ΔD** = D_nvfp4 − D_mxfp8 of
the rotated activation — implicitly assuming every channel's quantization error
costs the final LM loss equally. It doesn't, and `docs/rd_ron_report.md` found
the failure concretely: on **8B the pure-ΔD dense-limit pick was WRONG** (measured
6.6072 vs uniform 6.5882, gain **−0.019**). This work tests the refinement: weight
each ring's ΔD by its **sensitivity to the final cross-entropy loss**, and ask
whether that beats pure-ΔD — especially whether it can flip 8B's negative gain
positive.

## Method — the Fisher-diagonal proxy (exact form)

Local MSE ≠ final-loss impact. A first-order estimate of the LM-loss change from a
channel's activation error `ε_c` is `ΔL ≈ ½ Σ_c H_cc ε_c²`, and the empirical
Fisher diagonal `F_c = E_t[(∂L/∂a_c)²]` is the standard positive-semidefinite
proxy for the loss curvature `H_cc` w.r.t. that activation. We therefore:

1. Run **one backward pass of the LM cross-entropy loss per calibration
   sequence** (`model(ids, labels=ids).loss.backward()`), on the *same* dense-R-on
   rotated fp16 substrate the ΔD stats come from.
2. A tensor grad-hook on the **input of each target Linear** (the identical hook
   point `rd_validity.collect_stats` uses for ΔD, so Fisher channels line up 1:1
   with ΔD channels — for down_proj this is post-online-Hadamard) captures
   `∂L/∂a_c` and accumulates
   
       F_c = (1/T) · Σ_t (∂L/∂a_c)²          # per channel, over T calib tokens

   Grads are `nan_to_num`'d (fp16 backward) and accumulated in fp32 on CPU. To
   keep the activation graph alive with minimal memory we freeze all params and
   re-enable grad only on the embedding. Budget: **4 sequences × 1024 tokens =
   4096 tokens** (a proxy needs little — gradients average fast).
3. Per-ring weight is the Fisher summed over the ring's channels, and the
   **loss-weighted score** is the product:

       w_j = Σ_{c ∈ ring j} F_c            score_j = w_j · ΔD_j

   This is exactly the Fisher-proxy estimate of the LM-loss reduction from
   upgrading ring *j* FP4→FP8 (w_j = ring-summed curvature, ΔD_j = ring-mean
   per-channel MSE reduction). The raw Fisher is comparable across layers (same
   loss, same tokens), so a ring in a loss-sensitive layer can outrank a
   higher-ΔD ring in an insensitive one — the whole point. `allocate_by_budget`
   then ranks **all** rings globally by `score_j` and upgrades the top 1/8.
   (Normalization of `w` is a global positive rescale → ranking-invariant → not
   applied; the `normalize` flag in `ring_deltaD_weighted` exists only for
   inspection.)

Code (opt-in, zero-regression — pure-ΔD path byte-identical):
`promix/quantize/adaptive.py::{ring_fisher_weight, ring_deltaD_weighted}`,
driver `promix/scripts/rd_lossweight.py`, 5 new tests in `tests/test_adaptive.py`
(incl. a flip-the-pick unit case; 16/16 pass).

## Protocol

Matched **1/8** FP8-ring budget, **dense-R-on** shipping regime (shipped dense R1,
`ring_r1=False`, reorder-before-quant K), reusing `rd_validity`'s proven
build/collect/allocate/eval unchanged. Three allocations: **uniform** (shipped),
**pure-ΔD** (`measured_alloc`, what rd_ron used), **loss-weighted**
(`allocate_by_budget` on `w_j·ΔD_j`). o_proj held at uniform 1/8 in all three.

**Gates reproduced exactly** before trusting the new number:

| gate | 3B | 8B |
|---|---|---|
| bf16 bridge (= floor 7.82 / 6.14) | 7.8148 | 6.1368 |
| uniform (= rd_ron dense-R-on) | 8.2974 | 6.5882 |
| pure-ΔD dense-limit (= rd_ron) | 8.2678 | 6.6072 |

All four match `docs/rd_ron_report.md` to the digit → the harness is sound and the
loss-weighted PPL is comparable like-for-like.

## Results (wikitext-2 PPL, matched 1/8, dense-R-on)

| model | bf16 | uniform | pure-ΔD (Δ vs uniform) | **loss-weighted** (Δ vs uniform) | **lw vs pure-ΔD** | ring-pick overlap |
|---|---|---|---|---|---|---|
| **3B** | 7.815 | 8.2974 | 8.2678 (**−0.0296**) | **8.2743** (−0.0231) | **+0.0066 (worse)** | 60.2% |
| **8B** | 6.137 | 6.5882 | 6.6072 (**+0.0190**) | **6.5933** (+0.0051) | **−0.0139 (better)** | 62.7% |

Fisher is strongly non-uniform on both models (3B CV 4.51, max/mean 38.5; 8B CV
4.30, max/mean 35.2), and reweights ~38–40% of the picks (overlap ≈ 60–63%), so
loss-weighting is a genuine change, not a no-op.

## Verdict

**Loss-weighting does NOT uniformly beat pure-ΔD — it is model-dependent in
exactly the predicted direction:**

- **8B (the key test): loss-weighting HELPS and substantially repairs the
  mispick.** Pure-ΔD was **+0.0190 worse** than uniform (the known negative gain);
  loss-weighting is **−0.0139 better than pure-ΔD**, cutting the regression to
  **+0.0051** vs uniform — it recovers ≈73% of the gap. This directly confirms the
  diagnosis: 8B's local-MSE ring ranking is **loss-misaligned**, and the Fisher
  proxy correctly re-ranks toward the loss-sensitive rings. **But it does NOT fully
  flip 8B positive** — the loss-weighted allocation is still 0.005 PPL *above*
  uniform, i.e. allocation remains ≈ break-even on 8B, not a win.

- **3B: loss-weighting slightly HURTS vs pure-ΔD** (+0.0066), though it still beats
  uniform (−0.0231). On 3B the local-MSE ranking was already well-aligned with
  loss impact (pure-ΔD overturned uniform cleanly), so layering the noisy 4k-token
  Fisher estimate on top only perturbs a good ranking.

**Net:** the refinement is a **partial, honest success**. It behaves exactly as the
"local MSE ≠ final-loss impact" hypothesis predicts — it fixes the model where
pure-ΔD was loss-misaligned (8B: +0.019 regression → +0.005) and mildly degrades
the model where pure-ΔD was already loss-aligned (3B). But the Fisher proxy is not
strong enough to turn 8B allocation into a net win over uniform: 8B stays ≈
break-even. **Shipping recommendation unchanged by this experiment** — 3B: ship
pure-ΔD (best); 8B: ship uniform (allocation, even loss-weighted, does not beat
it). The value here is diagnostic: it isolates *why* 8B's pure-ΔD failed (loss
misalignment, now measured and largely corrected), which is the right lever for
future work (stronger sensitivity models — true diagonal Hessian / GPTQ
error-weighted, more tokens, or per-ring second-order terms).

## Raw numbers

3B: uniform 8.2974 / pure-ΔD 8.2678 / loss-weighted 8.2743 (bf16 7.8148).
8B: uniform 6.5882 / pure-ΔD 6.6072 / loss-weighted 6.5933 (bf16 6.1368).
JSON on remote: `/home/luofan/RD_LOSSW_3B.json`, `/home/luofan/RD_LOSSW_8B.json`.
Fisher: 4 seqs × 1024 tok; 3B 168 units / 10304 rings, 8B 192 units / 17408 rings.
