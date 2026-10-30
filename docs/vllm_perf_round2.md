# vLLM perf-optimization loop — ROUND 2: small/mid-M GEMM configs + pack-launch fusion

**Status: DONE.** Round-1's "next-biggest overhead" was the small/mid-M GEMM regime
(decode b8-b16 + chunked-prefill small-M, 0.53-0.61x of vLLM-NVFP4). Round 2 attacked it
with the two planned levers and shipped both green ones:

1. **Per-shape small-M GEMM configs** (`cd_smallm_ext.cu`, 128x64 tile, prepacked-SF,
   capture-safe): torch.equal to the shipping kernel, 1.15-1.65x GEMM-only in measured
   per-(N,K) M-windows (biggest at M=64-384, the chunked-prefill regime).
2. **Single-launch mixed activation pack** (`quant_pack_float_mixed` wired): bit-exact,
   1.10-1.16x pack, one fewer launch per Linear.

Same-session serve A/B (levers on/off, same box, same engine):
**3B decode b8 TPOT −6.7/−9.5 %, b16 −7.0/−7.3 %** (pm/NVFP4 0.51-0.59 → **0.54-0.62**);
8B b8/b16 −2.3/−2.6 %; prefill TTFT b1 −5 % both sizes; nothing regressed (b1 decode
within its known ±3 % bimodal noise, gates bit-identical). Two more levers were built,
measured, and honestly rejected: the fused Hadamard+pack down_proj producer (loses to
the FHT+qpfm chain) and 1-byte decode scales (bit-exact but 0.92-0.98x — r1's +8-12 %
estimate refuted).

Raw JSONs: `docs/vllm_perf_round2_data/`; full set on the remote under
`/home/luofan/vllm_integ/bench_r2/`.

---

## 1. What changed

| file | change |
|---|---|
| `kernels/mixed_gemm_sm120/cd_smallm_ext.cu` | NEW ext: co-design dual-format GEMM at 4 configs (cfg0 128x128 NV4/MX2 = shipping cross-check; cfg1 128x64 NV4/MX2; cfg2 128x64 NV6/MX2; cfg3 128x64 NV6/MX3) through the SAME prepacked-SF / cached-workspace / capture-safe entry contract as `cd_fused_gemm_prepacked`. PROTECTED mainloop headers #included unchanged (config-only instantiation, the cd_cfg/config_dispatch pattern). Key enabler: the CUTLASS interleaved SF atom (`Sm1xxBlockScaledConfig<SFVecSize>`) depends ONLY on SFVecSize, not the tile — the load-time weight SF and the qpf-emitted activation SF are byte-compatible with every config (verified in `sm100_blockscaled_layout.hpp`, then torch.equal-gated). |
| `promix/quantize/mixed_linear.py` | `_SMALLM_TABLE` (measured per-(N,K) M-windows) + `_smallm_cfg` dispatch inside `_gemm`; `gemv_route()` helper (single source of truth for the r1 decode-GEMV route, used by forward and the plugin); `forward_had_pack` / `can_had_pack` (lever-2a entry, default-unused — measured negative); `cd_sm` ctor arg. |
| `promix/vllm_integ/kernels.py` | `load_smallm()` (PROMIX_SMALLM=0 disables), `load_qpfm()` (PROMIX_QPFM=0), `load_had_pack()` (DEFAULT-OFF, PROMIX_HAD_PACK=1 opts in; docstring records the measured negative). |
| `promix/vllm_integ/plugin.py` | passes qpfm/cd_sm (+ hpk/hpk28/had28 when opted in) into `MixedQuantLinear`; down_proj apply branches to `forward_had_pack` only when opted-in AND on the GEMM route (GEMV routes keep the torch Hadamard — they consume fp16 activations). |
| `kernels/mixed_gemm_sm120/hadamard_pack_ext.cu`, `hadamard28_pack_ext.cu` | SWIGLU template flag + `hadamard_pack_mixed` / `hadamard28_pack_mixed` entries (already-activated input x, no SwiGLU) — built for lever 2a; numerics green; perf negative; kept as prior art. |
| `promix/vllm_integ/gemm_probe.py`, `pack_probe.py` | NEW graph-replay probes: per-shape x M config bench + bit gates (SFA-padding-aware), pack-fusion bit gates + chain bench. |

## 2. Lever 1 — small-M GEMM configs (probe: `r2_gemm_probe{,_rerun}.json`)

All four configs are **torch.equal** to the shipping kernel on every shape x M (the
128-row SFA atom padding is uninitialized memory and excluded; data bytes + consuming
GEMM outputs compared). Graph-replay timing, two independent runs agree (first run had
transient-contention outliers on the 8B rows; rerun on an idle GPU below):

GEMM-only µs, shipping (c0 128x128) → best 128x64, at the win window:

| shape (N,K) | window | examples (M: c0 → c1) |
|---|---|---|
| 3B qkv (5120,3072) | M 64..256 | 128: 19.7→13.4 (1.47x), 256: 20.0→16.5 (1.22x) |
| 3B o (3072,3072) | M 8..384 | 16: 19.6→16.2 (1.20x), 96: 19.4→13.1 (1.49x), 384: 19.9→15.0 (1.33x) |
| 3B down (3072,8192) | M 8..384 | 16: 41.1→33.7 (1.22x), 64-256: 40.9→25.6 (1.60x) |
| 8B qkv (6144,4096) | M 96..128 | 128: 24.1→16.4 (1.47x); 192+ loses (2-wave) |
| 8B o (4096,4096) | M 32..256 | 96-128: 23.6→15.9 (1.49x) |
| 8B down (4096,14336) | M 32..256 | 96-128: 66.6→40.9 (1.63x) |
| gate_up (both) | never | N/128 CTAs already fill the machine |

Mechanism: below the 128-row M-tile the shipping config runs only N/128 CTAs (24-48 of
188 SMs for qkv/o/down); 128x64 doubles the CTA count. The windows close above where
`ceil(M/128)*ceil(N/64)` exceeds one 188-SM wave, and below where the doubled A/SF
re-reads stop amortizing (M<~32-64 on qkv-class). Stage variants (NV6/MX2, NV6/MX3)
are within noise of NV4/MX2 — cfg1 is wired.

Honest bounds: at M=8-16 (decode b8/b16) the wins are limited to 3B o/down (1.15-1.22x);
the big 1.45-1.65x wins live at M=64-384 — the chunked-prefill / mid-batch regime.
At M<=16 the dual-format GEMM is latency-bound on its serial K-loop (down: 59 k-iters,
~3.4x above the weight-traffic floor) — no config in the SM120-viable space fixes that
(TK=256 is smem-impossible with MX stages>=2: union > 101376 B). The named fix is
split-K / stream-K in the fused glue (R3 lever, see §5).

## 3. Lever 2 — pack fusion (probe: `r2_pack_probe.json`)

* **qpfm (single-launch mixed pack): WIRED.** Bit-exact to the two-launch qpf path on
  every shape x M (data bytes + GEMM output torch.equal; the raw SFA compare is only
  meaningful at M%128==0 where the atom padding vanishes — both buffers' padding is
  uninitialized `torch.empty`). 1.10-1.16x on the pack step (e.g. 9.7→8.4 µs at decode
  M, 30.5→27.5 µs at M=2048), i.e. one fewer launch per Linear per forward.
* **Fused Hadamard+pack down_proj producer: MEASURED NEGATIVE, default-off.** The
  one-CTA-per-row hpk/hpk28 kernels lose to the `fast_hadamard_transform` fp16 kernel +
  qpfm chain: 3B K=8192 fused 16.5-105 µs vs chain 12.2-87.9 µs (0.82-0.97x); 8B
  K=14336 fused 55.6-634 µs vs chain 13.0-208 µs (0.26-0.34x). Numerics were green
  (bit-exact vs the fp32-trait reference; GEMM cos >= 0.99998 vs the shipping fp16-FHT
  chain) — the loss is kernel efficiency (FHT's dedicated butterfly kernel is far
  faster than the hpk transform stage; the saved fp16 round-trip doesn't pay for it).
  The down_proj chain therefore stays FHT + qpfm (qpfm still improves it: 13.5→12.2 µs
  at decode M).
* **RMSNorm+quant producer fusion (vLLM-NVFP4's lever): NOT DONE, scoped.** Our
  `promix_linear` is an opaque custom op — inductor's rms_norm+quant fusion passes
  cannot see the pack inside it. Doing this right needs either (a) splitting the op
  into quant + gemm custom ops plus a custom post-grad fusion pass matching vLLM's
  `fused_add_rms_norm -> promix_quant` (graph-structure change to the R1-validated
  capture path), or (b) model surgery replacing the decoder-layer norms. Both are
  R3-sized; the in-repo `add_rmsnorm_pack_ext.cu` (round-26) is the ready kernel, and
  `docs/vllm_quant_tricks_survey.md` has the vLLM patterns to mirror
  (silu_and_mul_nvfp4_quant / rms_norm_dynamic_per_token_quant custom ops + the
  rms_quant_fusion / act_quant_fusion compile passes, with file:line refs).

## 3b. Secondary lever — 1-byte decode scales (probe: `r2_sf8_probe.json`)

Built and wired behind `PROMIX_DECODE_SF8` (`decode_gemv_w4a16_sf8` / `_mb_sf8`: stream
the checkpoint's raw E4M3/UE8M0 scale bytes, fold the per-tensor g in-kernel).
**Bit-exact everywhere** (torch.equal vs the fp32-scale r1 kernels, all 8 shapes,
M=1..16 — `float(E4M3)*g` and `ldexpf(g,e-127)` reproduce the precomputed fp32
products exactly). **Measured SLOWER: 0.92-0.98x** under graph replay at M=1/2/4 on
every shape — the in-kernel scale decode (E4M3 cvt, ldexpf) costs more than the ~18%
low-segment byte saving; R1's "+8-12% decode-b1" estimate is REFUTED. DEFAULT-OFF;
kernels + probe kept as the documented negative.

## 4. Numerics gates (3B, mandatory)

| gate | result |
|---|---|
| T2 prefill path, post-graphs (`--graphs`) | **Byte-identical to the R1/Phase-2 baseline**: PPL 8.37029131329681 vs ref 8.3674, |dPPL|=0.0029; top-1 89.786% = the known Phase-1 engine-glue regime, every digit equal to `vllm_perf_round1_data/gate_t2_graphs.json` — qpfm + the new wiring are bit-invisible on the prefill path. |
| T2 through the decode route (`--decode-route-m 16`) | **PASS: PPL 8.3603 vs ref 8.3674 → signed ΔPPL = −0.0071 (improvement direction).** Exercises qpfm at every Linear + the 128x64 GEMM on 3B o/down at M=8..16 + the GEMV tail routes. NOTE the R1 doc's −0.108 for this gate is NOT the comparison point: R1's gate ran BEFORE its own threshold re-fit (all M≤16 through the W4A16 GEMV, which skips activation quant → big PPL gain); under the final shipped per-shape crossovers (M≤2-4 GEMV) most 16-token steps run the fused W4A4 GEMM ≈ the reference quantization, so ≈0 delta with a small GEMV-tail improvement is the expected green result. |
| T2 decode-route M=1 subset (`--decode-route-m 1 --graphs --max-chunks 30`) | **PASS — byte-identical to the R1 result** (PPL 8.462834401645244 vs subset ref 8.8014, signed ΔPPL −0.339, every digit equal to `vllm_perf_round1_data/gate_t2_decode_m1.json`): the M=1 GEMV path is untouched by this round, bit-provenly. |
| T2 decode-route m16 CONTROL (same gate, `PROMIX_SMALLM=0 PROMIX_QPFM=0`) | **Byte-identical to the levers-on run** (PPL 8.360270763492986, every digit; top-1, mean/max diffs all equal) — levers 1+2 are bit-invisible END-TO-END on the decode route, and the delta vs R1's −0.108 is definitively the route-threshold config (R1's gate predated its own threshold re-fit), not this round's changes. |

## 5. Scoreboard — serve bench

### 5a. Same-session A/B (the clean attribution): levers ON vs OFF (`PROMIX_SMALLM=0
PROMIX_QPFM=0`), same box/GPU/engine stack, back-to-back (`promix_*_ctl.json`)

3B decode TPOT (ms) off → on, with pm/NVFP4 ratio (NVFP4 = Phase-3 baseline):

| case | TPOT off → on | Δ | pm/nv off → on |
|---|---|---|---|
| isl1024_osl128_b8 | 7.00 → **6.34** | **−9.5 %** | 0.533 → **0.582** |
| isl128_osl1024_b8 | 6.51 → **6.07** | **−6.7 %** | 0.505 → **0.541** |
| isl1024_osl128_b16 | 7.46 → **6.93** | **−7.0 %** | 0.587 → **0.622** |
| isl128_osl1024_b16 | 6.85 → **6.35** | **−7.3 %** | 0.540 → **0.583** |
| b4 (both) | 6.27/5.89 → 5.98/5.84 | −4.7/−0.8 % | 0.548/0.522 → 0.573/0.526 |
| b1 (both) | 4.45/4.26 → 4.30/4.25 | −3.5/−0.2 % (bimodal ±3 % noise; M=1 path bit-identical) | ~0.73-0.76 |
| prefill L2048 b1/b4/b8/b16 | — | −5.5/−1.6/+0.1/+0.1 % | |

8B: decode b8/b16 −2.3 to −2.6 % (qpfm only — the 8B small-M windows start at M=32, so
decode b8/16 GEMMs are unchanged, exactly as the probe predicted), b1/b4 ≈ 0,
prefill b1 −4.9 %.

The 3B b8/16 gain (−6.7..−9.5 %) comes from cfg1 on o/down at M=8..16 (probe: −3.4 and
−7.4 µs/layer) + qpfm (−1.3 µs x4/layer) + 4 fewer launches/layer under graph replay.

### 5b. Three-way vs the Round-1 baselines (`promix_{3b,8b}.json` vs
`vllm_perf_round1_data/` promix + Phase-3 nvfp4/bf16)

| | decode b8 pm/nv | decode b16 pm/nv | decode b1 pm/nv | prefill L2048 pm/nv (b1..b16) |
|---|---|---|---|---|
| 3B r1 → r2 | 0.53-0.54 → **0.54-0.58** | 0.56-0.59 → **0.58-0.62** | 0.76-0.77 → 0.74-0.76 | 0.59-0.81 → **0.60-0.89** |
| 8B r1 → r2 | 0.57-0.58 → 0.56-0.58 | 0.60-0.61 → 0.60-0.61 | 0.73-0.75 → 0.73 | 0.53-0.55 → **0.56-0.57** |

Caveat: the r1-promix rows were measured in an older session — the same-session
levers-off control today reads ~2-4 % slower than r1's rows (session drift), so 5a is
the honest per-lever delta; 5b is the loop-level trajectory. pm/bf16 decode:
3B b8/16 0.93-1.00 → **0.96-1.05**; 8B unchanged ~1.14-1.23; b1 1.26-1.33 (3B) /
1.58-1.65 (8B) unchanged.

Protocol note: the serve cases (b1-16, isl 1024/128) sample M≤16 decode and M≥1024
prefill — they barely touch M=64-384 where lever 1 wins 1.45-1.65x (graph-replay-timed,
§2). Chunked-prefill tails / mid-batch decode workloads in that regime get those wins
directly.

## 6. Next biggest overhead (feeds round 3)

1. **Decode b8/16 GEMM is structurally latency-bound, not config-bound.** At M≤16 the
   dual-format GEMM runs its serial K mainloop on only N/128..N/64 CTAs (3B qkv 19.7 µs
   vs 6.1 µs weight floor; down 33.7 vs 9.8): the config space is exhausted (128x64
   probed; TK=256 smem-impossible with MX stages≥2; stage variants within noise), the
   batched GEMV loses 2-3x at M=16 (LDS-bound per row), and sf8 scales lose 0.92-0.98x.
   The named lever is **split-K / stream-K in the fused dual-phase glue** (k-iter count
   24-112 split across CTAs) — a forked-kernel scheduler change with the SM90
   cooperative stream-K machinery as the template; alternatively a Marlin-style W4A16
   tensor-core small-M kernel.
2. **Producer-side quant fusion** (vLLM-NVFP4's rms_norm+quant / silu_mul+quant
   inductor fusions): requires splitting `promix_linear` into quant + gemm custom ops
   + a custom post-grad pass (or norm surgery). Kernels exist in-repo
   (`add_rmsnorm_pack_ext.cu`); see `docs/vllm_quant_tricks_survey.md` for the vLLM
   pattern file:lines. Removes 1-2 kernels + one fp16 [M,K] round-trip per norm site.
3. **o_proj column gather** (`x[..., col_order]`): a separate gather kernel + round
   trip per o_proj; fold the static permutation into the qpfm pack's column indexing.
4. **Structural +19 % weight bytes vs pure-FP4** (the MXFP8 ring) — accuracy-motivated,
   not a perf bug; bounds the decode ceiling vs NVFP4.

## 7. Reproduce

```bash
B=/home/luofan/vllm_integ/bench_r2
# probes (graph-replay timing + bit gates):
docker exec -w /home/luofan/plaquant-vllm-integ -e CUDA_VISIBLE_DEVICES=7 luofan-plaquant \
  /home/luofan/baselines/venv/bin/python -m promix.vllm_integ.gemm_probe --out $B/r2_gemm_probe.json
... -m promix.vllm_integ.pack_probe --out $B/r2_pack_probe.json   # sets PROMIX_HAD_PACK=1 itself
... -m promix.vllm_integ.sf8_probe  --out $B/r2_sf8_probe.json
# gates (3B): gate_t2 --graphs | --decode-route-m 16 | --decode-route-m 1 --graphs --max-chunks 30
# serve A/B: bench_serve ... (levers on) vs PROMIX_SMALLM=0 PROMIX_QPFM=0 (control)
# env switches: PROMIX_SMALLM=0 (128x128-only GEMM), PROMIX_QPFM=0 (two-launch pack),
#               PROMIX_HAD_PACK=1 (opt-in fused had+pack, measured slower),
#               PROMIX_DECODE_SF8=1 (opt-in 1-byte decode scales, measured slower)
```
