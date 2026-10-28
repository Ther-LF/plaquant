# Rate-Distortion Allocation — Validity Gate Report (Task 3)

**Verdict: PASS.** On Llama-3.2-3B, R1-off (pure PCA), at matched average
FP8 ring-fraction = 1/8, the measured-MSE ΔD allocation gives **lower wikitext-2
PPL than both the variance proxy and the shipped uniform split**. The signal the
whole direction depends on (spec §9) is real: ranking 64-channel rings by
measured quantization distortion `ΔD = D_fp4 − D_fp8` allocates FP8 better than a
variance/energy proxy, which here is actually *worse* than uniform.

## Results

| Config | Allocation signal | Avg FP8 frac | wikitext-2 PPL |
|---|---|---|---|
| bf16 bridge | — (no quant, no rotation) | — | **7.8148** |
| (A) uniform | fixed 1/8 per Linear (shipped) | 0.1250 | **8.3924** |
| (B) variance-proxy | per-ring activation variance, `derive_khigh` @ τ=0.2696 | 0.1250 | **8.4711** |
| (C) measured-MSE | per-ring `ΔD` (measured), `allocate_by_budget` | 0.1250 | **8.3124** |

Gate (plan Task 3 / spec §9):
- `PPL(C)=8.3124 < PPL(B)=8.4711` → **True** (measured-MSE beats the variance proxy)
- `PPL(C)=8.3124 ≤ PPL(A)=8.3924` → **True** (measured-MSE ≥ uniform accuracy)
- → **PASS**. Measured-MSE ΔD ranking tracks PPL; proceed to Task 4+ (cost model,
  operating points, 64-ring rotation integration). Task 3b (loss-weighted ΔD) is
  **not** warranted — the unweighted local MSE already buys accuracy.

### Why this is a meaningful test, not a tautology
- All three allocations are at the **identical** budget: 1288 FP8 rings of 10304
  total reallocated rings (plus o_proj held at uniform 1/8), realized FP8 fraction
  = 93184 / 745472 channels = **0.1250** for all three (printed and confirmed).
- The variance and measured-MSE allocations **disagree on 75% of their FP8 rings**
  (ring overlap = 319 / 1288 = 24.8%). They are genuinely different allocations,
  so the PPL gap reflects the allocation signal, not noise.
- **The variance proxy made things worse than uniform** (8.4711 vs 8.3924). This is
  exactly the spec §5/§9 prediction: `error ∝ variance` is a first-order assumption
  that misses heavy-tailed / outlier rings whose NVFP4 block scale gets blown out —
  reallocating by variance pulls FP8 toward high-variance-but-already-fine rings and
  away from the clipping-prone ones. The measured MSE sees the real clipping and
  allocates correctly.

## Method (what was run)

Regime: **R1-off** (identity main rotation; `--r1_off`) — the cheapest setting that
tests the signal, reusing the proven `round11_sensitivity.py` R1-off build and the
`ablation_acc_per_bit.py` reorder + GPTQ-edriven machinery. No new quantization
logic.

1. **Collect** R1-off-rotated calibration activations (16 wikitext-2 train seqs,
   ≤4096 tokens/Linear) via forward-pre hooks on each reallocated Linear's inner
   `nn.Linear` (post online-Hadamard for down_proj). Per ring (64 ch) compute
   measured `D_fp4`, `D_fp8` with the deployment fake-quant (`fake_quantize_nvfp4`
   / `_mxfp8`, act-MSE scale search on) → `ΔD`; also per-ring activation variance.
2. **Allocate** at matched budget (1288 FP8 rings over q/k/v/gate/up/down_proj;
   o_proj fixed at 1/8 in all three as a controlled constant):
   - uniform: `K_high = round(1/8·d)` per Linear.
   - variance: `adaptive.derive_khigh` on each Linear's ascending per-ring variance,
     single global τ bisected to the budget; highest-variance rings chosen.
   - measured-MSE: `adaptive.allocate_by_budget([ring_deltaD…], target=0.125)` →
     `masks_to_khigh`.
3. **Apply** each allocation by permuting weight columns so the chosen FP8 rings sit
   in the tail, setting `reorder_before_quant` + `_column_order` on the wrapper and
   `_edriven_khigh` on the inner Linear (so GPTQ's high split matches), and feeding
   the per-Linear K_high into `configure_quantizers` via the new opt-in
   `model._rd_khigh_override` seam (zero-regression when absent).
4. **Evaluate** wikitext-2 PPL (bf16 model, `add_special_tokens=False`, 2048 non-
   overlapping chunks, token-pooled exp(mean NLL)) per `docs/baselines_on_hardware.md`.

The bf16 bridge (no rotation, no quant) reproduces 3B = **7.8148** (target 7.82),
so the eval protocol is trusted before any delta is read.

### Reproduce

```bash
# remote (docker luofan-plaquant, Pro 6000; GPU picked from an idle index):
cd /home/luofan/plaquant && git fetch origin && git checkout feat/rd-alloc && git pull
docker exec -d -w /home/luofan/plaquant \
  -e CUDA_VISIBLE_DEVICES=<idle gpu> -e PYTHONPATH=/home/luofan/plaquant \
  -e HF_HOME=/home/luofan/hf_cache luofan-plaquant bash -c \
  'python promix/scripts/rd_validity.py \
     --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml --r1_off \
     --target_mean_fp8 0.125 > /home/luofan/RD_VALIDITY.log 2>&1'
# then poll /home/luofan/RD_VALIDITY.log  (full JSON -> /home/luofan/RD_VALIDITY.json)
```

Notes:
- `rd_validity.py` resolves to the existing `U-wikitext-512-Llama-3.2-3B.bin` +
  `R-high-0.125-…-Llama-3.2-3B.bin` artifacts (per-head basis) and the local 3B
  model; the mxfp8-nvfp4 config's own `-mxfp8nvfp4-` basis was never generated.
- Each PTQ(GPTQ)+PPL build is ~11 min; three configs + bf16 bridge + collect ≈ 55 min.

## Raw log markers

```
BF16_BRIDGE ppl=7.8148 (29s)
ALLOC units=168 total_rings=10304 budget_fp8_rings=1288 (target_frac=0.1250)
VARIANCE tau=0.2696 fp8_rings=1288
MEASURED fp8_rings=1288
VAR∩MSE ring overlap=319/1288 (24.8%)
RESULT A_uniform   ppl=8.3924 avg_fp8=0.1250 (93184/745472 ch)
RESULT B_variance  ppl=8.4711 avg_fp8=0.1250 (93184/745472 ch)
RESULT C_measured  ppl=8.3124 avg_fp8=0.1250 (93184/745472 ch)
GATE  C<B: True  C<=A: True  -> PASS
```

## Caveats / honesty notes

- The absolute R1-off uniform number (8.3924) is slightly above the R1-on shipped
  3B anchor (8.30) — expected, since R1-off drops the learned/structured rotation.
  The gate is a **relative** comparison, all three in the same R1-off regime, so
  this does not affect the verdict.
- o_proj was held at uniform 1/8 across A/B/C (its per-head PCA uses the reorder
  path already; re-selecting its rings would entangle the comparison). The signal is
  measured on the q/k/v/gate/up/down_proj units that were reallocated. A follow-up
  (Task 6) that unifies the 64-ring rotation base can bring o_proj into the pool.
- The C-vs-A margin (8.3124 vs 8.3924, −0.08 PPL) is modest but consistent and in
  the predicted direction; the C-vs-B margin (−0.16) is larger. The decisive
  qualitative finding is that the **variance proxy is a net loss** while measured MSE
  is a net win at the same budget — the exact distinction the method is built on.
