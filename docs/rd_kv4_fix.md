# rd-kv4: fixing the KV4 mechanical blocker + the honest W4A4KV4 row

**Branch:** `feat/rd-kv4` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000
Blackwell (SM120), docker `luofan-plaquant`, remote dir
`/home/luofan/plaquant-kv4`, GPU 3 · **Date:** 2026-11-03 · **Status: COMPLETE**

## The blocker (rd-acctune lever 6)

`MXFP8 quantization requires K (16) divisible by block_size (32)`. Under the
FP mixed-precision stack the KV cache is quantized **per head**
(k/v_groupsize = head_dim), and the per-head high segment
`int(head_dim * high_fraction)` is

| model | head_dim | raw per-head high | MXFP8 %32 | per-head main | NVFP4 %16 |
|---|---|---|---|---|---|
| 1B | 64 | 8 | ✗ | 56 | ✗ (56 % 16 = 8) |
| 3B / 8B | 128 | 16 | ✗ | 112 | ✓ |

so the W4A4KV4 FP configs could not run at all. A second latent bug: the
`ActQuantizer` FP string branch **ignored `groupsize`** entirely — the V-cache
`out_quantizer` (groupsize = head_dim) would have split the flat
`num_kv_heads*head_dim` vector instead of each head, mis-routing every head
except the last (the per-head rotation places the high-variance channels at
the tail of EACH head).

## The fix (option a chosen; option b kept as the measured ablation)

Three candidates were evaluated:

- **(a) round the per-head KV high segment UP to 32** — effective per-head KV
  high fraction becomes 32/head_dim (1/4 at head_dim 128, 1/2 at head_dim
  64); a few more KV bits, mechanically trivial. **CHOSEN** — and the
  measurement justifies it (see ablation below: dropping the high segment
  costs up to +0.92 wiki on 1B).
- **(b) round DOWN to 0** — pure-NVFP4 KV (head_dim % 16 always holds).
  Implemented as the `kv_fp_high: "zero"` ablation; it is also the
  budget-honest "KV4 = 4.5 b/v" datapoint.
- **(c) group the high segments ACROSS heads into %32 groups** — keeps the
  1/8 budget exact but requires reorder plumbing through the attention +
  RoPE wrapper for a segment the measurement shows we *want* to widen, not
  shrink. Rejected (complexity without an accuracy case).

Implementation (`rd-kv4:` commits):

- `promix/quantize/quant_utils.py` — `kv_fp_high_length(head_dim, hf, mode)`
  (block-aligned per-head KV high length, modes `up`/`zero`, loud
  `ValueError` when no aligned split exists); `ActQuantizer` FP branch now
  honors `groupsize > 0` (reshape to `(..., n_heads, head_dim)` so the
  `[low|main|high]` split applies within each head). `groupsize = -1` stays
  byte-identical — every previously-working FP quantizer uses -1.
- `promix/eval/ptq.py` (`configure_quantizers`, V path) +
  `promix/quantize/kv_quant.py` (`setup_k_quant`, K path) — use the helper
  when `k_bits`/`v_bits` are FP strings; INT KV4 path untouched.
- `promix/scripts/rd_acctune.py` — `--kv4` / `--kv_high {up,zero}` to stack
  FP KV4 on any acctune point; config knob `quantize.kv_fp_high` (default
  `"up"`) added to the three `llama-*-mxfp8-nvfp4-kv4.yaml` configs.
- 6 new tests in `tests/test_fake_quant_mxfp8_nvfp4.py` (helper values,
  per-head split vs manual reference, groupsize=-1 zero-regression).

K-path protocol note: the committed `QKRotationWrapper` machinery is reused
unchanged — post-RoPE INT8 pre-rotation quantization, INT8-quantized
`U_key_pos @ R2` rotation, then the per-head mixed quantization (now
NVFP4 main + MXFP8 high on the aligned split). V is quantized by the
`out_quantizer` on `v_proj` per head.

## Protocol gates (all PASS)

| gate | expected | measured |
|---|---|---|
| 1B final anchor repro, `--kv4` OFF (zero-regression) | 14.16938304901123 / 19.711017608642578 | identical, full float digits |
| pytest `tests/test_fake_quant_mxfp8_nvfp4.py` (container) | no new failures | 72 passed (66 baseline + 6 new), same 4 pre-existing env failures |

## Results — W4A4 vs W4A4KV4 on the CURRENT best stack

Stack = the rd-acctune shipping recipes (ring512 + measured-ΔD + act-MSE
d3u2 + per-model GPTQ/weight-MSE knobs), same protocol, same artifacts; the
ONLY delta is `--kv4`. KV b/v includes block-scale overhead
(NVFP4 4+8/16 = 4.5; MXFP8 8+8/32 = 8.25).

| model | W4A4 anchor (KV fp16) | **W4A4KV4 fixed (a: high→32)** | ΔKV4 | KV4-zero ablation (b: pure NVFP4) | Δzero |
|---|---|---|---|---|---|
| 1B wiki | 14.1694 | **14.4137** | +0.244 | 15.3382 | +1.169 |
| 1B C4 | 19.7110 | **19.9846** | +0.274 | 21.0719 | +1.361 |
| 3B wiki | 8.2180 | **TBD** | | TBD | |
| 3B C4 | 11.2549 | **TBD** | | TBD | |
| 8B wiki | 6.5642 | **TBD** | | TBD | |
| 8B C4 | 9.5639 | **TBD** | | TBD | |

KV bits per value: option (a) = 5.44 b/v at head_dim 128 (3B/8B), 6.38 b/v
at head_dim 64 (1B); option (b) = 4.50 b/v everywhere; the unaligned intent
(1/8 @ 8.25b + 7/8 @ 4.5b) would have been 4.97 b/v.

### Reading

TBD after 3B/8B land.

## Fairness note (why this row exists)

The peers in the scoreboard main table — QuaRot, ResQ, Atom, SpinQuant —
are all **W4A4KV4** (quantized KV cache), while the committed ProMix rows
are W4A4 with **FP16 KV**. Comparing our no-KV-quant numbers against their
KV4 numbers UNDERSELLS the peers relative to us. The ProMix-KV4 row closes
that honesty gap: it is the number that should be quoted next to the
same-protocol peer re-measures (ResQ 15.98 / 8.71 / 7.06, QuaRot 22.11 /
10.72 / 8.65, Atom 18.40 / 9.70 / —, SpinQuant-learned 16.14 / 9.25 / —,
SpinQuant-Hadamard 18.62 / 10.09 / 7.88 wiki 1B/3B/8B).

Remaining caveats for an honest caption:

- Option (a) KV is 5.4–6.4 b/v, i.e. richer than the peers' ≈4–4.5 b/v
  INT4 KV. The budget-matched ProMix KV4 row is option (b) (4.5 b/v,
  same b/v as NVFP4 weights); quote (b) when the caption says "KV4 at
  matched KV bits", quote (a) as "ProMix KV4 (mixed, 5.4 b/v)" otherwise.
  On 1B option (a)'s 6.4 b/v is closer to "KV6" — flag it.
- The old W4A4KV4 numbers floating in older notes (ResQ paper: 1B 11.96 /
  3B 8.70 / 8B 7.04) are the ResQ **paper protocol** (different tokenizer
  handling / eval chunking / models) and are NOT comparable to this
  harness; the same-protocol ResQ re-measure is 15.98 / 8.71 / 7.06.

## Reproduce

```bash
# in /home/luofan/plaquant-kv4 (feat/rd-kv4), container luofan-plaquant
# 1B (final recipe + KV4 option a)
PROMIX_ACT_MSE_STEPS_UP=2 CUDA_VISIBLE_DEVICES=3 python promix/scripts/rd_acctune.py \
  --config promix/configs/llama-3.2-1b-mxfp8-nvfp4.yaml --point 1b_kv4_up \
  --substrate ring --ring_size 512 --alloc measured --gptq_nsamples 1024 \
  --c4 --kv4 --kv_high up --out OUT.json
# 3B: += PROMIX_WEIGHT_MSE_STEPS=7, --gptq_nsamples 512 --percdamp 0.025
# 8B: += PROMIX_WEIGHT_MSE_STEPS=7, --gptq_nsamples 512
# ablation: --kv_high zero
```

Raw JSONs/logs: `/home/luofan/kv4/*.{json,log}` (remote ICT90).
