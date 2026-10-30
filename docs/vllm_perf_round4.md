# vLLM perf-optimization loop — ROUND 4: Marlin W4A16 small-M route + FP8 KV cache

**Status: DONE.** Round-3's "next biggest overhead" was the qkv/o-class decode
GEMM at M≤16 sitting 3x above its weight-byte floor (split-K proven unable to
win there). Round 4 shipped **lever 1: a Marlin W4A16 tensor-core route for
M=2..16** that reuses vLLM's own Marlin fp4 kernel for our NVFP4 low segment
(it runs on SM120, feasibility probe cos=1.0) plus a new MXFP8 high-segment
companion kernel, and **lever 2: FP8 KV cache** (engine flag, FlashInfer fp8
paths on SM120, scales 1.0).

Same-session default-config serve A/B (marlin ON vs OFF, split-K + gather ON
both sides): **3B decode b4 TPOT −25.4..−25.9%, b8 −15.5..−16.0%, b16
−2.0..−2.1%; 8B b4 −30.0..−30.9%, b8 −9.6..−12.2%** (b16 neutral after the
serve-measured window cap, §2c). FP8 KV on top (same-session A/B, long-ISL
decode points): **3B TPOT −4.8..−26.8%, 8B −2.0..−21.5%**, KV token capacity
×1.99. Three-way ratios (pm/nv = nv_ms/pm_ms): 3B decode b4 0.52→**0.78**, b8
0.55→**0.75**, b16 0.59→**0.67**; 8B b4 0.55→**0.81**, b8 0.63→**0.73**.
Decode accuracy moved in the IMPROVEMENT direction: T2 decode-route m16 signed
ΔPPL **−0.3227** (8.3674→8.0447 — W4A16 covers M=2..16 on every Linear). Raw
JSONs: `docs/vllm_perf_round4_data/`; full set on the remote under
`/home/luofan/vllm_integ/bench_r4/`.

---

## 1. What changed

| file | change |
|---|---|
| `promix/vllm_integ/marlin_w4a16.py` | NEW: load-time repack of the checkpoint NVFP4 low segment into vLLM's Marlin fp4 format. Our layout (uint8 (N,K_low/2) codes low-nibble=even-k + (N,K_low/16) raw E4M3 scales + per-tensor g) is EXACTLY the modelopt layout the marlin utils consume (nibble order verified against `rand_marlin_weight_nvfp4_like`; dequant convention fp4·E4M3·g reproduced by passing global_scale=g through the S0E5M3 re-encode + power-of-2 fold — feasibility cos=1.0). Apply = direct `torch.ops.vllm.marlin_gemm` (use_atomic_add=False, use_fp32_reduce=True), shared per-device workspace. |
| `kernels/mixed_gemm_sm120/decode_gemv_ext.cu` | NEW `decode_gemv_w8a16_hi_add`: MXFP8 high-segment GEMV companion (same dequant + fp32 accumulation order as the mb GEMV's high loop) that ADDS into Marlin's fp16 (M,N) output in place; reads the high columns of the full (M,K) activation in place (no x_hi slice copy); M≤32, `__launch_bounds__`-capped. |
| `promix/quantize/mixed_linear.py` | `_MARLIN_TABLE` measured per-shape M-windows + `marlin_route`/`_marlin_forward`/`attach_marlin`; route PRIORITY over the batched GEMV at M≥2 (same W4A16 numerics class, 1.2-3.1x faster); M=1 keeps the byte-proven single-row GEMV; `fp16_route()` = gemv∨marlin for the plugin's producer branches. |
| `promix/vllm_integ/plugin.py` | marlin prepack in `process_weights_after_loading` (PROMIX_MARLIN=0 skips); per-shape hi_add warp heuristic (32 if N≥4096 else 16); `_apply_eager` o_proj/down_proj branches gate on `fp16_route` (marlin consumes fp16 → torch gather / torch Hadamard run first, exactly like the GEMV routes). |
| `promix/vllm_integ/marlin_probe.py` | NEW probe: numerics gate (cos vs fp32 dequant reference + cross-check vs mb GEMV) + full-route graph-replay A/B per shape × M (current pack+table-GEMM+cast vs slice+marlin+hi_add; o shapes additionally gather-chain vs gather+marlin) + hi_add warp sweep. |
| `promix/vllm_integ/gates/gate_t2.py` | `--kv-cache-dtype` passthrough (lever 2 gates). |

Env switches: `PROMIX_MARLIN=0` (disable route + skip the repack memory),
`PROMIX_MARLIN_MMAX` (cap/disable the window; the b16 localization knob).

## 2. Lever 1 — Marlin W4A16 small-M tensor-core route (SHIPPED)

### 2a. Why option (a) "reuse vLLM's marlin" won

The r3 brief's cheapest path worked outright: `marlin_gemm` with
`b_q_type=float4_e2m1f` **is compiled for and runs on SM120** (the
`is_fp4_marlin_supported` gate is cc≥75), and our checkpoint low-segment
buffers ARE modelopt-NVFP4-format, so the whole offline repack is ~15 lines of
vLLM util calls at load. No new TC kernel was written for the low segment; the
only new CUDA is the ~100-line high-segment companion. Honest memory cost: the
repack is a second copy of the low-segment bytes (fused prefill GEMM keeps the
original layout) — measured **+1.33 GiB on 3B, +3.36 GiB on 8B** of lost KV
budget (≈ +44% quantized-weight bytes).

### 2b. Probe (graph-replay, idle GPU; `r4_marlin_probe{,_m2}.json`)

Numerics: `MARLIN_GATE_ALL_PASS` — cos vs the fp32 dequant reference ≥
0.9999998 at every shape × M (fp16-scale dequant in marlin: fp4×E4M3 products
are ≤6-significant-bit → exact in fp16; accumulate fp32).

Full-route µs (current W4A4 route with r2/r3 tables + qpfm pack + cast → slice
+ marlin + hi_add); o rows are the gather-inclusive serve chains:

| shape | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|
| 3B qkv (5120,3072) | 16.0→13.0 (1.23x) | 21.9→14.0 (1.56x) | 30.3→16.4 (**1.84x**) | 30.2→22.2 (1.36x) |
| 3B o (chain) | 1.21x | 30.3→20.7 (1.47x) | 28.2→22.9 (1.23x) | 0.96x — stop |
| 3B gate_up | 41.8→20.0 (**2.09x**) | 53.3→22.9 (**2.33x**) | 50.6→29.5 (1.71x) | 1.11x |
| 3B down | 33.9→17.6 (1.93x) | 48.4→19.0 (**2.56x**) | 42.5→24.0 (1.77x) | 1.22x |
| 8B qkv (6144,4096) | 34.3→16.5 (**2.08x**) | 35.5→18.9 (1.88x) | 35.3→23.7 (1.49x) | 1.07x |
| 8B o (chain) | 1.40x | 35.1→21.4 (1.64x) | 35.4→24.5 (1.45x) | 1.12x |
| 8B gate_up | 81.5→32.3 (**2.52x**) | 104.8→37.1 (**2.83x**) | 93.4→50.6 (1.85x) | 1.19x |
| 8B down | 56.5→23.2 (2.44x) | 77.6→25.4 (**3.05x**) | 58.4→34.2 (1.71x) | 1.02x — stop |

Two structural findings:
* The Marlin low GEMM is near-flat in M (3B qkv: 9.4µs at M=2 → 12.1µs at
  M=32 — the "M=16 as cheap as M=1" promise delivered) and close to the
  weight-byte floor.
* The **hi_add companion is the new bottleneck**: it scales ~linearly in M
  (the batched-GEMV weakness in miniature — per-row staging+FMA over only
  K_high bytes) and closes every window by M=16-24 even though marlin stays
  flat to 32. At 3B qkv M=16 the 384-wide high segment costs 12.2µs vs 11.0µs
  for the 2688-wide marlin low — 7x the per-byte cost. A TC (or marlin-fp8)
  high companion is the named r5 lever (§6).
* Marlin also beats the **M=2..4 batched GEMV routes** everywhere (1.2-3.1x)
  — same W4A16 numerics class — so the route takes priority at M≥2; M=1 keeps
  the round-12 single-row GEMV (byte-proven, and still the right kernel there).

### 2c. Serve-level window correction (honest probe/serve discrepancy)

First 8B serve A/B ran windows (2,16): decode-b16 TPOT read **+4.0/+4.6%**
although the probe showed 1.07-1.19x kernel-level wins at M=16. Localization:
`PROMIX_MARLIN_MMAX=8` restores the control digits exactly (10.263/9.584 vs
ctl 10.268/9.581) while a same-session windows-(2,16) rerun reproduces the
regression (10.580/9.904) — the M=16 route itself, not session noise. Root
cause not isolated (probe = idle-GPU back-to-back graph replay; engine = full
graph with interleaved attention/norm work); the 8B windows ship capped at
m_hi=8. 3B keeps m_hi=16 (serve b16 −2.0/−2.1%). Encoded in `_MARLIN_TABLE`
with the discrepancy documented.

## 3. Lever 2 — FP8 KV cache (SHIPPED as a serve flag)

Zero plugin work needed: `kv_cache_dtype="fp8"` works out of the box with the
promix method (vLLM default scales 1.0, no checkpoint calibration;
`PromixConfig.get_quant_method` returns None for attention). On SM120 the
engine keeps the FLASHINFER backend and resolves
`decode_backend=xqa, kv_cache_dtype=fp8e4m3` — the same backend the fp16-KV
runs use, so the A/B isolates the KV dtype. KV token capacity ×1.99 (3B
638k→1,269k; 8B 501k→996k tokens at mml 8192).

## 4. Numerics gates (3B, mandatory)

| gate | result |
|---|---|
| T2 prefill `--graphs` (DEFAULT config, marlin on) | **Byte-identical to the r2/r3 baseline** — PPL 8.37029131329681, top-1 89.78647181310134%, every digit equal. Marlin never routes at prefill M; the repack is load-time only. (The harness's top-1-vs-eager-reference criterion prints FAIL at 89.79% in graphs mode — same digits every round since r2; the round gate is byte-identity, per r3 §5.) |
| T2 decode-route m16 (marlin active M=2..16) | **PASS: signed ΔPPL = −0.3227** (8.0447 vs ref 8.3674) — W4A16 replaces W4A4 on every Linear at the decode-route Ms, PPL improves by 0.32. Same phase-adaptive-accuracy effect as the r1 GEMV routes, now across the whole M=2..16 window. |
| T2 decode-route m16 CONTROL (`PROMIX_MARLIN=0`) | **Byte-identical to r3's final config** (PPL 8.367113659407261, every digit) — levers-off control green. |
| T2 decode m1 `--graphs` | **Byte-identical to r1/r2/r3** (PPL 8.462834401645244) — M=1 GEMV untouched. |
| T2 prefill `--graphs` + `--kv-cache-dtype fp8` | signed ΔPPL **+0.0276** vs the fp16-KV graphs run (8.3979 vs 8.3703); |d| vs reference 0.0305 ≤ 0.1 clause **PASS**. |
| T2 decode m1 `--graphs` + fp8 KV | 8.4854 vs fp16-KV 8.4628 → fp8-KV cost **+0.0226**; still −0.316 vs the W4A4 reference. PASS. |

## 5. Scoreboard — serve bench (all same-session A/B)

### 5a. Lever 1 (marlin) ON vs OFF (both sides: split-K + gather ON)

| case | 3B TPOT off → on | 8B TPOT off → on |
|---|---|---|
| isl1024_osl128 b4 | 5.966 → **4.450 (−25.4%)** | 10.072 → **7.053 (−30.0%)** |
| isl1024_osl128 b8 | 6.153 → **5.201 (−15.5%)** | 9.439 → **8.529 (−9.6%)** ¹ |
| isl1024_osl128 b16 | 6.759 → **6.620 (−2.1%)** | 10.222 → 10.263 (+0.4%) ¹ |
| isl128_osl1024 b4 | 5.823 → **4.313 (−25.9%)** | 9.884 → **6.835 (−30.9%)** |
| isl128_osl1024 b8 | 5.884 → **4.944 (−16.0%)** | 9.098 → **8.149 (−10.4%)** ¹ |
| isl128_osl1024 b16 | 6.173 → **6.051 (−2.0%)** | 9.538 → 9.584 (+0.5%) ¹ |
| b1 (both cases) | +0.2% / +0.2% | +2.0% / −0.1% ² |
| prefill TTFT L2048 | −0.1..+4.3% (b1 noise) ² | −1.3..+0.2% |

¹ 8B b8/b16 rows are the FINAL-config runs (windows capped m_hi=8:
`pm8b_mmax8` vs same-session `pm8b_b16ctl`); the first-session windows-(2,16)
run read b8 −12.0/−12.2% but b16 +4.0/+4.6% (§2c). b1/b4 rows from
`pm8b_r4{,ctl}` (unaffected by the cap).
² M=1 paths are byte-identical (m1 gate); the 8B isl1024-b1 +2.0% and 3B
prefill-b1 +4.3% single-run wobbles are inside r2's documented b1 ±3% band.

The b4 −25..−31% is the largest single-round decode win of the loop so far;
mechanism: at M=4 the control runs the worst-case kernels (plain 128×128
W4A4 GEMM on qkv/gate_up/down — the r2/r3 windows only start at M=8-32 —
plus the M=4 GEMV on o), while marlin serves all four near the weight floor.

### 5b. Lever 2 (fp8 KV) ON vs OFF (long-ISL decode points, default levers both)

| case | 3B TPOT fp16→fp8 KV | 8B TPOT fp16→fp8 KV |
|---|---|---|
| isl2048 b1 / b8 / b16 | −4.8% / −15.2% / **−19.5%** | −2.0% / −8.2% / **−13.9%** |
| isl4096 b1 / b8 / b16 | −6.6% / −20.5% / **−26.8%** | −3.3% / −13.4% / **−21.5%** |
| prefill TTFT (isl2048/4096 points) | — | −1.4..−1.7% |

Grows with ISL×batch exactly as the survey predicted. HONEST note: the NVFP4
baseline gains even MORE from the same flag (nv own deltas −2.3..−33.7%) — its
faster linears make attention/KV a larger fraction — so the pm/nv ratio at
long ISL dips ~0.03-0.07 when BOTH engines run fp8 KV (e.g. 3B isl2048 b16
0.717→0.655). fp8 KV is a recommended serve flag for promix (free TPOT +
double KV capacity at +0.03 PPL), not a ratio-vs-rival lever.

### 5c. Three-way vs nvfp4/bf16 (same-session, standard cases, fp16 KV;
pm/nv = nv_ms/pm_ms, higher is better)

| decode TPOT | b1 | b4 | b8 | b16 |
|---|---|---|---|---|
| 3B r3 → r4 | 0.72-0.74 → **0.78** | 0.515-0.525 → **0.78-0.79** | 0.549-0.551 → **0.74-0.75** | 0.59-0.61 → **0.64-0.67** |
| 8B r3 → r4 | 0.71-0.73 → **0.76** | 0.545-0.557 → **0.81** | 0.625-0.634 → **0.73** ³ | 0.660-0.675 → **0.66-0.68** |

³ 8B b8/b16 ratios computed from the first-session pm8b_r4 run (b8 8.33/8.03);
the final-config b8 (8.53/8.15) gives ~0.72. bf16/pm: 3B 1.06-1.40, 8B
1.21-1.68 (pm beats bf16 decode everywhere). Prefill TTFT nv/pm: 3B
0.57-0.81, 8B 0.54-0.57 — prefill remains the biggest relative gap
(structural: +19% weight bytes + act-quant producer chain).

## 6. Next biggest overhead (feeds round 5)

1. **TC/marlin-class HIGH-segment companion** — the hi_add GEMV is now the
   small-M bottleneck (7x the per-byte cost of the marlin low at M=16; it
   closes the windows at M=16-24 and is ~35-50% of the route at M=8-16).
   Options: marlin fp8 weight-type call for the MXFP8 segment (needs group-32
   UE8M0→fp16 scale handling — check `marlin_gemm` 8-bit group support), an
   m16n8k16 TC mini-kernel, or folding the high segment into the marlin launch
   epilogue. Opening M=16-32 cleanly would also lift 3B/8B b16 (today −2%/0%)
   and chunked-prefill tails.
2. **8B M=16 probe/serve discrepancy** — kernel-level 1.07-1.19x wins that
   invert to +4% TPOT in the full graph. Profile the captured decode graph
   (nsys on graph replay) to find what the probe misses (L2/SM contention with
   attention kernels? marlin CTA count at N=28672?). Resolving it unlocks the
   capped windows.
3. **Prefill ratio 0.54-0.58 (8B)** — unchanged structural bound (+19% weight
   bytes vs pure-FP4 + per-linear act-quant chain); norm-fuse opt-in exists
   (r3, measured decode-negative), attention-output quant fusion (survey §6.3)
   still blocked on the o_proj reorder fold.
4. **fp8-KV accuracy headroom** — scales default 1.0 cost +0.028 PPL (3B);
   checkpoint-calibrated per-tensor scales (BaseKVCacheMethod path) would
   likely halve it if it ever matters.

CONVERGENCE: NOT converged — this round moved decode b4 by −25..−31%, b8 by
−10..−16%, and long-ISL serving by up to −27%; lever 6.1 has a concrete
>2% expected value (b16 + route-interior gains), so the loop should continue.

## 7. Reproduce

```bash
B=/home/luofan/vllm_integ/bench_r4
# feasibility + probe (graph-replay timing + numerics gates):
docker exec -w /home/luofan/plaquant-vllm-integ -e CUDA_VISIBLE_DEVICES=7 luofan-plaquant \
  /home/luofan/baselines/venv/bin/python -m promix.vllm_integ.marlin_probe --out $B/r4_marlin_probe.json
# gates (3B): bench_r4/run_gates_r4.sh  (marlin + kv8 suites)
# serve: bench_r4/run_serve_r4.sh <ckpt> <tag>  (marlin A/B + kv8 A/B)
#        bench_r4/run_baselines_r4.sh <nv-ckpt> <bf-ckpt> <tag>
# env: PROMIX_MARLIN=0 (off), PROMIX_MARLIN_MMAX=<m> (window cap; 8 = 8B final)
# fp8 KV: --extra-llm-kwargs '{"kv_cache_dtype":"fp8"}' (bench) / --kv-cache-dtype fp8 (gate)
```
