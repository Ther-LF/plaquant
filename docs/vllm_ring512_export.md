# vLLM serving on the ring512+measured substrate (new accuracy anchors)

> 2026-11-03, branch `feat/vllm-integ` ("vllm-ring512" commits). Carries the
> rd-acctune accuracy anchors (docs/rd_accuracy_tuning.md: 1B 14.169 / 3B
> 8.218 / 8B 6.564 fake-quant wikitext) into the vLLM serving stack: the
> serving checkpoints previously exported from the dense-R1 substrate are
> re-exported on the promoted **ring512 R-on substrate + global measured-ΔD
> allocation + per-model acctune stack**, re-gated (T0/T1/T2), and A/B-served
> against the dense checkpoints. Raw JSONs: `docs/vllm_ring512_data/`.
> Checkpoints: `/home/luofan/vllm_ring512/promix-{1b,3b,8b}-ring512` (remote
> ICT90).

## What changed

1. **Exporter** (`promix/vllm_integ/export.py`): new `--substrate ring
   --ring_size 512 --alloc measured --gptq_nsamples N --percdamp P
   --resolve-paths` knobs. The build goes through the proven `rd_validity`
   machinery (`build_base` → `collect_stats` → `measured_alloc` →
   `build_eval`), i.e. the exact code path that produced the acctune anchors,
   then packs/re-references/exports as before (shared-g per merged group).
   Per-model recipe (docs/rd_accuracy_tuning.md):
   - 1B: ring512+measured + act-MSE up2 + GPTQ nsamples 1024
   - 3B: + weight-MSE d7 + GPTQ 512 + percdamp 0.025
   - 8B: + weight-MSE d7 + GPTQ 512
2. **Per-prefix K_high + runtime gathers generalized.** The measured
   allocation yields a per-layer, per-merged-prefix K_high (including
   **K_high = 0**) and, in principle, a per-Linear ring reorder. The exporter
   now emits `column_order` at the merged module name for ANY
   reorder-before-quant Linear (identity orders dropped) +
   `reordered_prefixes` in `quantization_config`; the plugin applies the
   gather-folded pack (the o_proj r3 lever, generalized) for any reordered
   prefix. **Empirically the ring512 measured allocation selects exactly the
   eigenvalue-tail rings for qkv/gate_up/down on all three models, so every
   non-o_proj order is identity → zero new runtime gathers** (the
   `reordered_prefixes` list contains only the o_proj entries).
3. **Act-MSE up2 in the CUDA producers** (`act_mse.cuh`): new
   `PROMIX_ACT_MSE_STEPS_UP` compile knob — N upper-E4M3 candidates evaluated
   after the lower walk, byte-increment == torch `_next_higher_grid`, same
   candidate order and first-wins ties as the fake-quant twin; default 0 is
   byte-identical to the shipped search. Threaded through both
   `act_mse_cflags()` (container + venv). The plugin now **defaults the
   act-MSE env from the checkpoint's `act_mse` block** (setdefault), so
   serving a ring512 (up2) checkpoint reproduces its exporter reference
   without manual env plumbing.
4. **Pure-NVFP4 (K_high==0) serving-route parity**
   (`promix/quantize/mixed_linear.py`, plugin): the measured allocation gives
   every down_proj (and a few 1B qkv/gate_up) K_high=0. The pure-NVFP4 path
   previously lost the decode-GEMV, Marlin-W4A16 and had_pack routes — first
   serve A/B showed decode b16 TPOT +10%. Now: marlin needs no high companion
   (covers the whole weight), GEMV runs with Kh=0 (kernels tolerate it),
   had_pack admitted. Probe (`docs/vllm_ring512_data/PUREFP4_PROBE.log`):
   GEMV/marlin cos ≈ 1.0000 vs the exact-activation W4A16 reference at
   M=1..32 on the real exported down_proj. This flipped decode b16 to a WIN.

## Allocation outcome (ring512, global measured-ΔD, budget 12.5 %)

| model | qkv K_high | gate_up K_high | down_proj | o_proj (held uniform) |
|---|---|---|---|---|
| 1B | 512×14, 0×2 | 512×15, 0×1 | **0 (pure NVFP4) ×16** | 256 |
| 3B | 512×23, 1024×5 | 512×25, 1024×3 | **0 ×28** | 384 |
| 8B | 512/1024 mix | 512/1024 mix | **0 ×32** | 512 |

The budget migrates from the (online-Hadamard-flattened) down_proj input to
the attention/MLP hidden-stream inputs — down_proj goes pure NVFP4 everywhere.

## Accuracy: before → after (wikitext-2)

| model | fake-quant anchor (acctune) | per-proj-g real kernel | **shared-g reference (T2 target)** | old dense shared-g | Δ shared-g |
|---|---|---|---|---|---|
| 1B | 14.1694 | 14.1735 | **14.2736** | — (first 1B export) | — |
| 3B | 8.2180 | 8.2142 | **8.2785** | 8.3646 | **−0.086** |
| 8B | 6.5642 | — (dual-pack OOM, skipped as on dense) | **6.6269** | 6.6363 | **−0.009** |

- Per-projection-g real-kernel PPL lands on the fake-quant anchors (3B
  −0.004, 1B +0.004) — the kernel path reproduces the acctune recipe.
- Shared-g re-quantization cost: 3B +0.064, 1B +0.100 (dense was +0.071).
- vLLM-scored (gate T2, graphs): 1B 14.2670 / 3B 8.2824 / 8B 6.6283.

## Gates (all three checkpoints)

| gate | 1B | 3B | 8B |
|---|---|---|---|
| T0 packed-buffer torch.equal | PASS 64/64 | PASS 112/112 | PASS 128/128 |
| T1 per-layer bit-exact (+ split-vs-merge) | PASS 64/64 | PASS 112/112 | PASS 128/128 |
| T2 \|ΔPPL\| ≤ 0.1 vs own shared-g ref (graphs) | **0.0134** | **0.0012** | **0.0000** |
| T2 top-1 (known fp16-logit-tie class, bar waived since phase 1) | 87.3 % | 90.1 % | 90.0 % |
| T2 decode-route m16 (W4A16 windows incl pure-FP4 marlin, full wikitext) | — | signed ΔPPL −0.2481 (8.0331 vs ref 8.2812), PASS | — |

**Rotation-only sanity** (ring512 T = U·ring_R1 fused, NO quantization, vs
plain fp16 — the lossless-rotation gate): 3B 7.8119 vs 7.8113 (+0.0006), 8B
6.1347 vs 6.1346 (+0.0001), 1B 13.1628 vs 13.1537 (+0.0091 — dense-R1 control
13.1631, i.e. the small 1B drift is substrate-INDEPENDENT fp16 roundoff).
The ring512 rotation fuses identically through the export.

## Serving parity (same-session A/B, one GPU, graphs+compile ON, final-campaign protocol)

Decode-heavy 128/1024 TPOT ms; prefill TTFT L2048 ms. `ring512` vs `dense`
(same session, back-to-back; raw: `pm3b_ring512_v2 / pm3b_dense_v2 /
pm8b_ring512 / pm8b_dense_rerun`).

| point | 3B ring512 | 3B dense | Δ | 8B ring512 | 8B dense | Δ |
|---|---|---|---|---|---|---|
| decode b16 | **5.10** | 5.47 | **−6.8 %** | **7.80** | 8.13 | **−4.1 %** |
| decode b128 | 11.64 | 11.34 | +2.6 % | 16.55 | 15.98 | +3.6 % |
| decode b256 | 18.56 | 18.20 | +2.0 % | 25.59 | 24.95 | +2.6 % |
| prefill L2048 b16 | 398.9 | 385.7 | +3.4 % | 828.9 | 800.8 | +3.5 % |
| prefill L2048 b128 | 3197.7 | 3081.9 | +3.8 % | 6872.0 | 6726.7 | +2.2 % |
| prefill L2048 b256 | 6481.9 | 6186.0 | +4.8 %* | 14421.1 | 14014.1 | +2.9 % |

*dense 3B b256 prefill varied 6186–6529 across sessions (same ckpt, same GPU);
the other points reproduced within ~0.3 % across two sessions.

**Attribution** (per-merged-Linear microbench, M=8192, pack+GEMM+cast,
`SHAPE_PROBE*.log`): qkv Kh 384→512 +2.6 %, gate_up +4.5 %, the few Kh=1024
layers +8 %, down_proj pure-FP4 −5.2 % → weighted ≈ +1.8 % Linear-level. The
ring granularity (512-channel allocation atom vs the dense 384 = 1/8·3072)
carries a small structural large-batch cost; the pure-FP4 down_proj pays it
back at decode (marlin whole-weight route): **decode b16 is faster, large
batch is +2–4 %** — in exchange for the accuracy column moving −0.086 (3B) /
−0.009 (8B) shared-g and a first-ever 1B serving checkpoint.

Against the five-way tables' original (cross-session) ProMix rows the new
points sit within the tables' own session spread except the +2-4 % cells
above; the five-way serving tables keep the dense-campaign numbers with a
note, and the accuracy column moves to the new shared-g anchors.

## Reproduce

```bash
# container base env, one GPU, per model (3B shown; see doc head for 1B/8B knobs)
PROMIX_ACT_MSE_STEPS_UP=2 PROMIX_WEIGHT_MSE_STEPS=7 MASTER_PORT=29610 \
python -m promix.vllm_integ.export \
  --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml --resolve-paths \
  --substrate ring --ring_size 512 --alloc measured \
  --gptq_nsamples 512 --percdamp 0.025 \
  --out /home/luofan/vllm_ring512/promix-3b-ring512
# gates (venv): gate_t0 / gate_t1_capture + (container) gate_t1_replay /
# gate_t2 --graphs — PROMIX_ACT_MSE_STEPS_UP now auto-defaults from the ckpt.
# serve A/B: promix.vllm_integ.bench_serve --batches 16,128,256
#   --cases 128:1024 --prefill-len 2048 --gpu-mem-util 0.90
```
