# vLLM perf-optimization loop — ROUND 5: MXFP8-marlin high companion + prefill decomposition & cast fold

**Status: DONE.** Round-4's two named fronts: (1) the hi_add GEMV companion was
the small-M bottleneck of the Marlin W4A16 route (linear in M, 7x the marlin
low's per-byte cost at M=16, closed every window by M=16-24); (2) the
structural prefill gap (pm/nv 0.54-0.58 at 8B) had never been decomposed.
Round 5 shipped **lever 1: an MXFP8-marlin tensor-core high companion**
(vLLM's own marlin kernel natively runs our checkpoint high-segment layout —
fp8 codes + e8m0 group-32 scales; near-flat in M like the low GEMM) which
reopens the route windows to M=32-48 and RESOLVES the r4 8B M=16 probe/serve
discrepancy, and **lever 2: a per-forward prefill decomposition** (new
`prefill_profile` probe, pm vs nv side by side under graphs) whose #1
actionable — the per-Linear bf16→fp16 GEMM-output cast, 112/128 `direct_copy`
launches = 3.3/6.8 ms per 3B/8B L2048 step — was eliminated by wiring the
round-26 **cast-fold epilogue** (bit-identical by construction), plus an
honest structural-floor account of everything that remains.

Same-session serve A/B (r5 levers ON vs r4-equivalent control): **3B decode
b16 TPOT −9.2/−9.9%, b8 −1.5/−4.0%; 8B b16 −13.8/−14.5%, b8 −8.6/−9.2%**
(the r4 8B b16 regression is not just gone — it inverted into the biggest b16
win of the loop); **prefill TTFT L2048 3B −4.6..−15.5%, 8B −9.3..−13.5%**.
Three-way: 8B decode pm/nv b8 0.73→**0.79**, b16 0.66-0.68→**0.81-0.82**;
8B prefill L2048 pm/nv 0.54-0.57→**0.60-0.66**. Raw JSONs:
`docs/vllm_perf_round5_data/`; full set on the remote under
`/home/luofan/vllm_integ/bench_r5/`.

---

## 1. What changed

| file | change |
|---|---|
| `promix/vllm_integ/marlin_w4a16.py` | NEW `MarlinHigh` + `prepack_marlin_high`: load-time repack of the checkpoint high segment (uint8 E4M3 codes (N,K_high) + uint8 UE8M0 scales (N,K_high/32)) into marlin's NATIVE MXFP8 format — exactly vLLM's `prepare_mxfp8_layer_for_marlin` recipe (`pack_fp8_to_int32` + `gptq_marlin_repack` num_bits=8 + e8m0 scale permute + `mxfp8_marlin_process_scales`). Per-tensor g rides the companion's add alpha (e8m0 only holds powers of two). `PROMIX_MARLIN_HI=0` restores the r4 hi_add-only route. |
| `kernels/mixed_gemm_sm120/decode_gemv_ext.cu` | NEW `axpy_bf16_into_half`: ONE fused kernel for Y(fp16) += g·H(bf16) (fp32 opmath), replacing the torch `H.to(fp16)` + `add_(alpha=)` pair (−2 µs/call route-level). |
| `promix/quantize/mixed_linear.py` | `_MARLIN_HI_TABLE` (per-shape route-level crossover hi_add→marlin-hi) + `_hi_marlin_route`; `_MARLIN_TABLE` windows re-measured with the companion (M up to 32-48; `marlin_route` requires the companion above the hi_add M=32 kernel bound); `_gemm` routes the no-window GEMM default through the cast-fold bias epilogue (`_bias_or_zero` cached zero bias). |
| `promix/vllm_integ/kernels.py` | `load_cd_bias()` (PROMIX_CAST_FOLD=0 disables) — loads the round-26 `codesign_fused_bias_ext` (epilogue `half(float(bf16(acc))+float(bias))`, bit-identical to the bf16 store + `.to(fp16)` + bias chain; PROTECTED mainloop untouched). |
| `promix/vllm_integ/plugin.py` | prepacks the marlin high companion at load (skipped under PROMIX_MARLIN_HI=0); passes `cd_bias` into MixedQuantLinear. |
| `promix/vllm_integ/marlin_hi_probe.py` | NEW probe: numerics gates (full route vs fp32 dequant ref + vs the r4 hi_add route) + graph-replay benches (hi companion vs hi_add per M; full route r4 vs r5 vs the W4A4 tables at M=2..64; o gather chains). |
| `promix/vllm_integ/castfold_probe.py` | NEW probe: torch.equal bit gate (fold vs GEMM+cast) + graph-replay chain bench at M=64..2048. |
| `promix/vllm_integ/prefill_profile.py` | NEW lever-2 probe: one L-token prefill step under graphs → kernel-time buckets (GEMM/pack/hadamard/attention/norm/cast/host-gap) from torch.profiler, pm and nv side by side; dumps the raw kernel table for audit. |

Env switches: `PROMIX_MARLIN_HI=0` (r4 hi_add-only), `PROMIX_CAST_FOLD=0`
(bf16 store + torch cast), plus all prior-round switches unchanged.

## 2. Lever 1 — MXFP8-marlin high companion (SHIPPED)

### 2a. Feasibility findings (the r4 brief's option (a) won, with one twist)

* `MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]` and vLLM 0.32 ships a
  NATIVE MXFP8 marlin path (`apply_mxfp8_marlin_linear`): fp8 weights +
  **e8m0 group-32 scales — byte-for-byte our checkpoint high-segment layout**,
  so the repack is again ~15 lines of vLLM utils. No new TC kernel.
* TWIST (measured, not documented anywhere): the e8m0-scale marlin kernels
  are compiled for **bf16 activations only** — fp16-activation group-32 fp8
  kernels do not exist in the wheel (`Invalid thread config` for every
  K_high × M; bf16 works everywhere). The companion therefore consumes a bf16
  slice-cast of the high columns. Numerics class: the W4A4 reference
  quantizes the same slice to E4M3 (3-bit mantissa), bf16 keeps 8 — still
  strictly finer; the decode gate moved +0.0027 PPL vs the r4 fp16-exact
  hi_add class and remains −0.320 vs the W4A4 reference (§4).
* g cannot ride e8m0 (powers of two only) → folded into the companion's
  accumulate step; the fused `axpy_bf16_into_half` makes cast+scale+add ONE
  kernel (route −2 µs/call vs the torch pair).
* Memory (honest): a second copy of the high-segment bytes + scales,
  ~0.35 GiB (3B) / ~0.85 GiB (8B) of lost KV budget (analytic Σ N·K_high;
  the 3B measured allocated delta +0.41 GiB agrees; the 8B reading is
  confounded by the gpu-mem-util KV autoscale).

### 2b. Probe (graph-replay, idle GPU; `r5_marlin_hi_probe2.json`)

Numerics: `MARLIN_HI_GATE_ALL_PASS` — full-route cos vs the fp32 dequant
reference ≥ 0.9999970 at every shape × M (r4 hi_add route: 0.9999998; the
~3e-6 delta is exactly the bf16 high-activation class).

Hi segment only (best hi_add warp → marlin companion, µs): the companion is
near-flat in M, hi_add linear — e.g. 8B gate_up M8 31.6→10.0, M16 51.6→11.5,
M32 111.8→13.4 (**3.0-4.5x**); 3B qkv M16 11.7→7.8, M32 22.1→8.7. Route-level
crossovers (slice-cast included) encoded in `_MARLIN_HI_TABLE`: gate_up 2-4,
qkv/down 8, o 16.

Full route vs the best W4A4 GEMM tables (r2+r3 included), probe `≥5%` rule —
the windows the companion REOPENS (r4 closed them all by M=8-16):

| shape | M=8 | M=16 | M=32 | M=48 | window |
|---|---|---|---|---|---|
| 3B qkv | 1.64x | 1.43x | 1.28x | 1.09x | (2,48) was (2,16) |
| 3B gate_up | 2.28x | 1.93x | 1.35x | 1.02 stop | (2,32) was (2,16) |
| 3B down | 1.92x | 1.63x | 1.49x | 1.09x | (2,48) was (2,16) |
| 3B o (gather chain) | 1.11x | 1.00 stop | — | — | (2,8) unchanged |
| 8B qkv | 1.63x | 1.50x | 1.30x | 1.08x | (2,48) was (2,8) |
| 8B o (gather chain) | 1.40x | 1.28x | 1.14x | 0.95 stop | (2,32) was (2,8) |
| 8B gate_up | 2.84x | 2.36x | 1.73x | 1.35x | (2,48) was (2,8) |
| 8B down | 2.18x | 1.63x | 1.23x | 0.99 stop | (2,32) was (2,8) |

### 2c. The r4 8B M=16 probe/serve discrepancy — RESOLVED

r4 shipped the 8B windows capped at m_hi=8 because kernel-level M=16 wins
(1.07-1.19x, hi_add era) inverted to +4% TPOT in the serve graph. With the
companion the M=16 route is 1.28-2.36x at the kernel level (a margin the
serve-graph effect cannot eat) and the same-session serve A/B (control =
exactly the r4 shipping config: `PROMIX_MARLIN_HI=0 PROMIX_CAST_FOLD=0
PROMIX_MARLIN_MMAX=8`) reads **8B decode b16 TPOT −13.8/−14.5%** — the
discrepancy did not reproduce once the hi segment stopped being the
bottleneck. The 8B windows now ship at the probe values (no cap);
`PROMIX_MARLIN_MMAX` remains the kill switch.

## 3. Lever 2 — prefill decomposition + cast fold (SHIPPED) and the structural floor

### 3a. The decomposition (prefill_profile, L2048 b1, graphs ON; ms per step)

| bucket | pm 3B | nv 3B | pm 8B | nv 8B |
|---|---|---|---|---|
| linear GEMMs | 13.04 | 10.36 | 26.22 | 20.37 |
| act pack/quant (standalone) | 4.35 | ~0.4 | 6.82 | ~0.5 |
| GEMM-out cast (direct_copy) | **3.29** | 0 | **6.79** | 0 |
| silu_mul (inductor) | 0.65 | (fused in quant) | 3.44 | 2.37 (silu+quant fused) |
| Hadamard (fht + had28 matmul) | 0.80 | 0 | 2.52 | 0 |
| norms (± fused quant) | 0.73 | 0.41 | 1.55 | 1.33 |
| attention (flash + kv cache) | 4.21 | 4.05 | 5.76 | 5.73 |
| kernel total | 27.93 | 15.94 | 54.42 | 31.64 |
| wall (b1, incl. engine host) | 40.99 | 32.94 | 67.84 | 36.94 |

(nv quant rides producers: `*_rms_norm_scaled_fp4_quant` triton +
`silu_mul_cvt_fp16_to_fp4` + tiny `cvt_fp16_to_fp4` — the survey §6.1 pattern,
confirmed in the kernel table.)

### 3b. Shipped: the GEMM-output cast fold (the #1 actionable)

The validated bf16 GEMM forces a `D.to(fp16)` `direct_copy` per Linear —
112/128 launches, 3.3/6.8 ms per step = 12% of pm kernel time. The round-26
`codesign_fused_bias_ext` already stores `half(float(bf16(acc)) + float(bias))`
— for the bias-less Llama Linears (cached zero bias) that is **bit-identical**
to the two-step store+cast. Wired as the no-window GEMM default (the r2/r3
small-M/split-K windows keep their measured exts + the then-small cast; a real
bias still takes epilogue priority everywhere).

* `castfold_probe`: **torch.equal on every shape × M** + GEMM-chain
  1.03-1.34x (8B gate_up M2048 617→478 µs).
* Post-fold profile: pm 3B kernels 27.9→**24.65 ms** (cast bucket 4.17→0.76),
  pm 8B 54.4→**47.8 ms** (10.76→3.00; the remainder is the inductor
  silu_mul, not a cast).
* Serve: the −9..−15.5% TTFT rows in §5a are this lever (marlin never routes
  at prefill M).

### 3c. The honest structural floor (8B L2048, post-fold)

What remains of the prefill gap, item by item (kernel-level, per step):

1. **Fused GEMM 27.0 ms vs nv 20.4 ms = 1.33x.** Theory for the dual-phase
   design: the MXFP8 ring runs 1/8 of the union k-tiles at half TC rate
   ⇒ ~1.13x; +19% weight bytes is negligible at compute-bound M=2048. The
   ~0.18x residual is kernel-stack headroom beyond that model — but the
   kernel line was CONCLUDED at round 19 (config space exhausted at the
   production operating point vs the CUTLASS reference; vLLM's production
   fp4 GEMMs are evidently faster than that reference). STRUCTURAL for this
   loop unless the kernel line is reopened. ≈ +5.4 ms incl. theory part.
2. **Act pack 7.0 ms vs nv's ~1.0 ms incremental fused quant.** Our pack
   carries the act-MSE scale search (PROMIX_ACT_MSE, 3 steps — deliberate
   accuracy work, part of the gated numerics) and runs standalone because the
   producer fusions measured NET-NEGATIVE at serve (r3 norm-fuse) or at the
   kernel level (r2 had+pack). STRUCTURAL-BY-CHOICE. ≈ +6 ms.
3. **Hadamard 2.5-2.8 ms** (fht + the 28×28 matmul for K=14336): the ResQ
   online rotation, no nv equivalent. STRUCTURAL. ≈ +2.8 ms.
4. **silu_mul 3.44 vs nv 2.37 fused**: a SiluAndMul→pack fusion could save
   ~1 ms (sub-2% of wall); same r3-negative pattern risk. Parked, honest.
5. **Host gap**: pm 13-16 ms vs nv 5.3 ms at 8B b1 — the opaque-op Python
   body per Linear at prefill sizes the engine does not graph-capture.
   Largely overlapped/amortized at b≥4 (the TTFT ratio improves with batch).
   Plugin-structural; micro-trims possible but sub-2% at the batched
   operating points.

Floor estimate: with items 1-3+5 kept structural, the 8B prefill kernel-level
floor is ≈ nv 31.6 + 5.4 + 6 + 2.8 ≈ 46 ms vs 47.8 measured — i.e. the
shipped configuration is within ~4% of its decomposed structural floor; the
serve ratio ceiling under this structure is ≈ 0.66-0.69 vs **0.60-0.66
measured** (b1 trails on the host gap). The remaining non-structural levers
(silu fusion, host micro-trims) are each sub-2% of wall.

## 4. Numerics gates (3B, mandatory — all green)

| gate | result |
|---|---|
| T2 prefill `--graphs`, lever-1 config AND final config (marlin-hi + cast fold) | **Byte-identical to the r2/r3/r4 baseline** — PPL 8.37029131329681, top-1 89.78647181310134%, every digit equal (two separate runs: `gate_t2_graphs_r5.json`, `gate_t2_graphs_r5final.json`). The cast fold is bit-invisible end-to-end, as constructed; marlin never routes at prefill M. (The harness's own top-1 criterion prints FAIL at 89.79% in graphs mode — same digits every round since r2; the round gate is byte identity, per r3 §5.) |
| T2 decode-route m16 (companion active at M≥8-16) | **PASS: signed ΔPPL = −0.3200** (8.0473 vs ref 8.3674). vs r4's −0.3227: the +0.0027 is exactly the bf16 high-activation class (proven by ctlA). Final config byte-identical to the lever-1-only run (cast fold bit-invisible on this route too). |
| T2 m16 CONTROL A (`PROMIX_MARLIN_HI=0 PROMIX_MARLIN_MMAX=16`) | **Byte-identical to r4's m16 run** (8.044671299500335, every digit) — the r4 route is exactly recoverable; the m16 delta is definitively the companion. |
| T2 m16 CONTROL B (`PROMIX_MARLIN=0`) | **Byte-identical to r3's final-config digits** (8.367113659407261) — levers-off control green across three rounds. |
| T2 decode m1 `--graphs`, lever-1 AND final config | **Byte-identical to r1-r4** (8.462834401645244) — M=1 GEMV untouched. |
| Kernel probes | `MARLIN_HI_GATE_ALL_PASS` (cos ≥ 0.9999970 all shapes × M, cross vs hi_add route attached); `CASTFOLD_BIT_PASS` (torch.equal all shapes × M). |

## 5. Scoreboard — serve bench (same-session A/B)

### 5a. r5 levers ON vs r4-equivalent control (ctl = `PROMIX_MARLIN_HI=0
PROMIX_CAST_FOLD=0 PROMIX_MARLIN_MMAX=16(3B)/8(8B)` — the r4 shipping config)

| case | 3B TPOT off → on | 8B TPOT off → on |
|---|---|---|
| isl1024_osl128 b8 | 5.204 → **5.125 (−1.5%)** | 8.454 → **7.725 (−8.6%)** |
| isl1024_osl128 b16 | 6.621 → **6.011 (−9.2%)** | 10.238 → **8.824 (−13.8%)** |
| isl128_osl1024 b8 | 4.940 → **4.742 (−4.0%)** | 8.138 → **7.391 (−9.2%)** |
| isl128_osl1024 b16 | 6.056 → **5.454 (−9.9%)** | 9.567 → **8.180 (−14.5%)** |
| b1 / b4 (both cases) | +3.4/0.0% / +2.1/−0.2% ¹ | +1.3/+0.1% / 0.0/−1.4% ¹ |
| TTFT (case rows, b4-b16) | −3.9..−16.3% | −6.0..−14.2% |
| prefill L2048 b1/b4/b8/b16 | **−4.6/−12.3/−15.5/−15.4%** | **−9.3/−12.2/−13.5/−13.3%** |

¹ b1 paths are byte-identical (m1 gate) and the 3B isl1024 b1 +3.4% / b4
+2.1% rows sit inside the r2/r3-documented long-context placement wobble
(±3-4%); the isl128 twins read 0.0..−0.2%. The signal rows (b16, b8, all
prefill) replicate across both cases.

### 5b. Long-ISL decode points with fp8 KV ON (mml 8192, isl 2048/4096;
r5 kv8 run vs the r4 kv8 run — cross-session, the levers' effect at long ISL)

| case | 3B TPOT r4 → r5 | 8B TPOT r4 → r5 |
|---|---|---|
| isl2048 b8 / b16 | −4.0% / **−9.3%** | −8.9% / **−16.5%** |
| isl4096 b8 / b16 | −3.4% / **−7.6%** | −8.3% / **−14.3%** |
| b1 (both) | −0.7/0.0% | −0.2/0.0% |

### 5c. Three-way vs nvfp4/bf16 (same-session fresh baselines, standard cases,
fp16 KV; pm/nv = nv_ms/pm_ms, higher is better)

| decode TPOT | b1 | b4 | b8 | b16 |
|---|---|---|---|---|
| 3B r4 → r5 | 0.78 → 0.77 | 0.78-0.79 → 0.75 ² | 0.74-0.75 → 0.72 ² | 0.64-0.67 → **0.71-0.73** |
| 8B r4 → r5 | 0.76 → 0.73 | 0.81 → **0.81-0.82** | 0.73 → **0.79** | 0.66-0.68 → **0.81-0.82** |

² The r4→r5 trajectory mixes ~±3% session drift on BOTH sides (today's nv
rows run faster than r4's); §5a is the honest per-lever delta. The b16
columns — the round's target — moved decisively on both sizes. pm/bf16
decode: 3B 1.16-1.40, 8B **1.47-1.67** (pm beats bf16 decode everywhere).

Prefill TTFT pm/nv, L2048 sweep: 3B 0.67-0.89 (b1 0.89), **8B
0.60-0.66** (was 0.54-0.57) — the cast fold closed ~1/4 of the 8B prefill
gap; the remainder is decomposed in §3c.

## 6. Convergence assessment

**This round: NOT converged by the loop rule** — decode b16 moved −9..−14.5%,
8B b8 −8.6..−9.2%, prefill TTFT −9..−15.5%, all ≫2%.

**But the lever list is now exhausted down to sub-2%/structural items:**

* Decode: M=1..48 is covered by byte-proven GEMV (M=1) + marlin W4A16 TC
  routes near the weight floor (M=2..48, both segments now tensor-core);
  b4-b16 pm/nv sits at 0.72-0.82 with the remaining distance ≈ the +19%
  weight bytes + act-exact numerics choices. No named decode lever remains
  with ≥2% expected value.
* Prefill: decomposed to its floor (§3c). Remaining non-structural items —
  SiluAndMul→pack (~1.5%), host-gap micro-trims (sub-2% at b≥4), reopening
  the concluded kernel line for the 1.33x-vs-theory-1.13x GEMM residual —
  are each sub-2% or contradict a concluded/measured-negative line. The
  shipped config is within ~4% of the decomposed kernel-level floor.
* fp8-KV accuracy headroom (r4 §6.4) remains an accuracy, not perf, item.

**Recommendation to the controller: treat round 6 as the convergence round —
expected total improvement from the remaining list is <2%. The loop can close
and the final campaign can run on this configuration** (levers: marlin W4A16
+ MXFP8-marlin companion + cast fold + split-K + gather-pack + qpfm defaults
ON; fp8 KV as the recommended serve flag; norm-fuse/had-pack/sf8 documented
negatives OFF).

## 7. Reproduce

```bash
B=/home/luofan/vllm_integ/bench_r5
# probes (graph-replay timing + numerics/bit gates):
docker exec -w /home/luofan/plaquant-vllm-integ -e CUDA_VISIBLE_DEVICES=7 luofan-plaquant \
  /home/luofan/baselines/venv/bin/python -m promix.vllm_integ.marlin_hi_probe --out $B/r5_marlin_hi_probe2.json
... -m promix.vllm_integ.castfold_probe --out $B/r5_castfold_probe.json
# prefill decomposition (pm + nv, 3B + 8B):  bench_r5/run_prof_r5.sh, run_prof_r5b.sh (post-fold)
# gates (3B): bench_r5/run_gates_r5.sh (lever-1 + controls), run_gates_r5b.sh (final config)
# serve: bench_r5/run_serve_r5.sh <ckpt> <tag> <ctl_mmax>   (ctl_mmax 16=3B, 8=8B r4 windows)
#        bench_r5/run_baselines_r5.sh <nv-ckpt> <bf-ckpt> <tag>
# env: PROMIX_MARLIN_HI=0 (r4 hi_add route), PROMIX_CAST_FOLD=0 (torch cast),
#      PROMIX_MARLIN_MMAX=<m> (window cap)
```
