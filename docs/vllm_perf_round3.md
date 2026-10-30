# vLLM perf-optimization loop — ROUND 3: split-K for the latency-bound decode GEMM + producer-side quant fusion

**Status: DONE.** Round-2's "next-biggest overhead" was the structurally
latency-bound decode-b8/b16 GEMM regime (serial K mainloop on 24-96 CTAs,
2-3.4x above the weight-byte floor, config space exhausted). Round 3 attacked
it with **split-K / stream-K in the fused dual-phase glue** (lever 1, SHIPPED),
built the **producer-side RMSNorm→quant_pack fusion** through an op split + an
out-of-tree inductor pass (lever 2, bit-exact, measured NET-NEGATIVE at decode
→ default-off, honest negative), and shipped a bonus **o_proj gather→pack fold**
(lever 3, bit-exact, 1.2-2.0x on its chain).

Same-session default-config serve A/B (split-K + gather ON, vs both OFF):
**8B decode b8/b16 TPOT −9..−11%** (pm/NVFP4 0.60-0.61 → **0.63-0.68**),
3B b8/16 −1..−4%; prefill TTFT −2..−9% at b8/16; b1/b4 unchanged (norm-fuse,
the one lever that regressed them, is off by default). Raw JSONs:
`docs/vllm_perf_round3_data/`; full set on the remote under
`/home/luofan/vllm_integ/bench_r3/`.

---

## 1. What changed

| file | change |
|---|---|
| `kernels/mixed_gemm_sm120/codesign_splitk_kernel.hpp` | NEW forked glue (fork of the round-14 `codesign_fused_kernel.hpp`, PROTECTED original untouched): drives the CUTLASS **SM90 stream-K scheduler** (`PersistentTileSchedulerSm90StreamK`, used directly — the SM120 selector maps to the CLC-based SM100 variant, which is dynamic-persistent and N/A) over ONE **union k-tile domain** `[NV k-tiles \| MX k-tiles]` mapped per work unit onto the two phases; `clear(accumulators)` in the glue for MX-only units (the NV collective owns ClearAccum); the shared FP32 accumulator is reduced ONCE over both phases by the scheduler's stock workspace fixup (reserved StreamkBarrier0/1 — no collision with the phase barriers at effective 14/15). The round-14 fork had preserved every stream-K glue hook (`get_work_k_tile_start/count`, `fixup`, `compute_epilogue`, `valid_warpgroup_in_work_tile`) — the fork compiled and passed its bit gate on the first build. |
| `kernels/mixed_gemm_sm120/cd_splitk_ext.cu` | NEW ext `fused_gemm_prepacked_sk(..., cfg, splits, decomp)`: cfg 0=128x128 / 1=128x64 NV4/MX2; decomp 0=heuristic / 1=split-K(splits) / 2=stream-K / 3=data-parallel (serial cross-check). Same prepacked-SF / cached-workspace / capture-safe contract as `cd_smallm_ext`; `gemm.initialize()` per call enqueues the stream-K workspace memset on the stream → re-zeroed on every CUDA-graph replay by construction. |
| `kernels/mixed_gemm_sm120/vllm_rms_quant_pack_ext.cu` | NEW ext (lever 2): ONE kernel = **bit-exact replica of vLLM csrc fused_add_rms_norm** (vendored `_f16Vec` vec8 loops, `cub::BlockReduce<float,1024>`, vLLM's num_tokens-dependent block-size rule — all REQUIRED for the exact reduction order) + the verbatim qpf NVFP4/MXFP8 pack reading the rounded fp16 row from SMEM; also emits fp16 `xn` (decode-GEMV routes consume fp16) + functional `residual_out`. `do_pack=false` (GEMV-routed M) skips the pack phase. BLOCK-templated `__launch_bounds__` (the 1024-thread decode variant is capped at 64 regs; the 256-thread prefill variant is not). |
| `kernels/mixed_gemm_sm120/quant_pack_fused_ext.cu` | lever 3: `quant_pack_float_mixed_gather(x, gidx, ...)` — the o_proj static column reorder folded into the pack's load (`GATHER` template flag; default path byte-untouched). |
| `promix/vllm_integ/ops.py` | **op split**: `torch.ops.vllm.promix_rms_quant_pack` (norm+pack producer, 6 outputs) + `torch.ops.vllm.promix_gemm_packed` (consumes packs on the GEMM route / `xn` on the GEMV route, per captured M); `_sfa_numel` symbolic SFA sizing for the fake impls; registry entries carry `mql`/`vrq`. |
| `promix/vllm_integ/fusion_pass.py` | NEW out-of-tree inductor pass `PromixNormPackFusionPass`: manual fx rewrite of `vllm_ir.fused_add_rms_norm → promix_linear` into `promix_rms_quant_pack → promix_gemm_packed` (manual walk, not the pattern matcher — the per-site int handle has no clean wildcard), with fake-mode meta propagation; installed by wrapping `PostGradPassManager.configure` (DEFAULT-OFF, see §3). |
| `promix/quantize/mixed_linear.py` | `_SPLITK_TABLE` (measured per-shape M-windows, list of (m_lo, m_hi, cfg, splits, decomp)) + `_splitk_cfg` dispatch (priority over `_SMALLM_TABLE` inside its windows); `cd_sk` ctor arg; `can_gather_pack`/`forward_gather_pack`. |
| `promix/vllm_integ/kernels.py` | `load_splitk()` (PROMIX_SPLITK=0 disables), `load_norm_pack()` (DEFAULT-OFF, PROMIX_NORM_FUSE=1 opts in; docstring records the measured negative). |
| `promix/vllm_integ/plugin.py` | wires cd_sk; o_proj apply branches to `forward_gather_pack` on the GEMM route (PROMIX_GATHER_PACK=0 disables); registers mql/vrq with the op registry; installs the fusion pass at import (no-op by default). |
| `promix/vllm_integ/{splitk,norm_pack,gather_pack}_probe.py` | NEW graph-replay probes: bit gates + per-shape x M benches for each lever. |

## 2. Lever 1 — split-K/stream-K dual-phase GEMM (SHIPPED)

Probe (`r3_splitk_probe.json`, graph-replay timing, idle GPU): the
**data-parallel decomposition is `torch.equal` to the shipping kernel on every
shape x M** (the forked glue is bit-sound); split decompositions reorder the
FP32 partial reduction → cosine 0.99999988..1.0.

GEMM-only µs at the decode/mid-M points, r2-best (min of shipping 128x128 and
the r2 cfg1) → best split variant:

| shape | M=8 | M=16 | M=32..512 |
|---|---|---|---|
| 8B down (4096,14336) | 65.2→43.9 (**1.48x**) | 66.1→39.7 (**1.66x**) | 1.07-1.35x (stream-K c0) |
| 8B gate_up (28672,4096) | 91.6→82.8 (1.11x) | 88.4→79.5 (1.11x) | 1.09-1.21x to M=256 |
| 3B down (3072,8192) | 35.9→31.5 (1.14x) | 34.5→29.7 (1.16x) | M64-128: 25.8→21.2/20.0 (1.22-1.30x) |
| qkv (both), o (both), 3B gate_up | **no win** (0.8-1.0x) — honest negative, no window | | |

Mechanism: down_proj is the K-heavy serial-latency case (56-112 union k-tiles
on 24-64 CTAs); splitting K across the 188-SM grid cuts the serial chain ~4x
and the FP32 fixup traffic is trivial. qkv-class (K=3072/4096 → 22-32 k-tiles)
does NOT win: the serial loop is already short, and the full-grid prologue +
reduction overhead exceeds the saved iterations — at M≤16 qkv remains at
19.8/24.8 µs vs its ~6 µs weight floor (the r4 lever, §6). Encoded windows:
`_SPLITK_TABLE` in `mixed_linear.py`.

## 3. Lever 2 — producer-side RMSNorm→quant_pack fusion (BUILT, BIT-EXACT, default-OFF)

The vLLM-NVFP4 trick (survey §6.1): no standalone act-quant launch — quant
rides the producing norm. Three parts, all landed:

1. **Op split** (`promix_rms_quant_pack` + `promix_gemm_packed`) so the graph
   can see the pack; the gemm op keeps the r1 capture-safe per-M GEMV/GEMM
   routing (consumes `xn` on GEMV routes).
2. **Out-of-tree pass registration**: wrapping `PostGradPassManager.configure`
   works (55 decoder sites rewritten); patterns are matched on
   `torch.ops.vllm_ir.fused_add_rms_norm` (0.32's pre-lowering IR ops).
   CAVEAT discovered: vLLM's **AOT compile cache key does not include
   out-of-tree passes** — a cached unfused graph silently bypasses the fusion;
   opt-in runs need `VLLM_DISABLE_COMPILE_CACHE=1`.
3. **Bit-exact fused kernel**: round-26's arn kernel failed its gate on ATen
   reduction order; the vLLM-path reference is vLLM's OWN CUDA kernel, which IS
   exactly replicable (vendored `_f16Vec`/cub semantics + the num_tokens-dependent
   block-size rule). `norm_pack_probe`: **torch.equal to
   `torch.ops._C.fused_add_rms_norm` (norm + residual) AND to qpfm (codes +
   consuming-GEMM bytes) at every M/hidden/weight — NORM_PACK_GATE_ALL_PASS.**

Why it is default-OFF (the honest negative, two layers deep):

* Micro (probe): after porting qpfm's intra-row parallel decomposition and
  BLOCK-templated launch bounds, the fused kernel beats the
  [CUDA-norm + qpfm] chain 1.04-1.12x at M≤64 and ~parity at larger M.
* E2E: the REAL baseline norm is not the CUDA kernel — the engine lowers
  `fused_add_rms_norm` to inductor-NATIVE triton (`custom_ops: [none]`,
  `ir_op_priority: native`), fused with neighboring elementwise ops. First
  serve A/B caught the fused op **packing at GEMV-routed M where the baseline
  never packs at all** (decode b1-b4 TPOT +5..+11%); the `do_pack=false`
  branch fixed most of it, but the same-session isolation
  (`iso{3b,8b}_{nf,nonf}.json`, split-K+gather ON in both) still measures
  **decode TPOT +3.8/+4.0% (3B b1/b4 isl1024) and +1.0/+1.6% (8B)** with ~0
  contribution at b8/16, against only −1..−2% prefill TTFT. Decode-dominated
  serving loses net → DEFAULT-OFF, kept as opt-in prior art
  (PROMIX_NORM_FUSE=1 + VLLM_DISABLE_COMPILE_CACHE=1).
* Accuracy when opted in: T2 graphs gate PPL 8.3661 vs ref 8.3674
  (|d|=0.0013, better than the baseline's 0.0029) — the native→CUDA-replica
  norm swap is ulp-level; NOT byte-identical to the shipping graph (and a
  cache-off levers-off control shows recompilation alone drifts digits:
  8.37029→8.3705, inductor combo-kernel autotuning), so the claim is the
  |dPPL|≤0.1 clause, not byte identity.

## 4. Lever 3 — o_proj gather→pack fold (SHIPPED)

`gather_pack_probe`: **bit-exact** (codes + consuming-GEMM outputs torch.equal
vs gather-then-pack) and wins everywhere on its chain: 3B o_proj 1.22x (M=4)
→ 1.75x (M=2048), 8B 1.25x → 1.98x — the [M,K] fp16 gather round trip and its
launch disappear into the pack's indexed load. GEMV routes (decode M≤4) keep
the torch gather (they consume fp16). Default-ON (PROMIX_GATHER_PACK=0
disables).

## 5. Numerics gates (3B, mandatory)

| gate | result |
|---|---|
| T2 prefill `--graphs` (split-K on, norm-fuse off = DEFAULT config) | **Byte-identical to the r2 baseline** — PPL 8.37029131329681, top-1 89.78647181310134%, every digit equal to `vllm_perf_round2_data/gate_t2_graphs.json`. Split-K never routes at prefill M; gather is bit-exact. |
| T2 decode-route m16 (split-K active on 3B down M≤16) | **PASS: signed ΔPPL = −0.0003** (8.3671 vs ref 8.3674). The shift vs r2's 8.3603 is exactly the split-K FP32 reduction order (see control). |
| T2 decode-route m16 CONTROL (`PROMIX_SPLITK=0`) | **Byte-identical to r2's m16 run** (PPL 8.360270763492986, every digit) — levers-off control green; the m16 delta is definitively split-K's reduction order. |
| T2 decode m1 `--graphs` | **Byte-identical to r1/r2** (PPL 8.462834401645244, every digit) — the M=1 GEMV path untouched. |
| norm-fuse opt-in gates | T2 graphs \|d\|=0.0013 PASS; decode m1 graphs signed −0.3382 PASS (§3). |

## 6. Scoreboard — serve bench

### 6a. Same-session A/B, intermediate full-lever config (`promix_*_v2{,ctl}.json`:
split-K + gather + norm-fuse ON vs all r3 levers OFF)

| case | 3B TPOT Δ | 8B TPOT Δ |
|---|---|---|
| isl1024_osl128 b8 / b16 | +0.5% / −1.1% | **−10.3% / −9.0%** |
| isl128_osl1024 b8 / b16 | **−3.3% / −4.1%** | **−10.7% / −9.8%** |
| b1 / b4 | +3.5..+1.0% / +4.0..+0.6% | +3.2..+1.0% / +3.6..−0.3% |
| prefill TTFT L2048 b8/b16 | −4.2% / −8.8% | −4.0% / −4.3% |

The b1/b4 regressions are the norm-fuse lever (isolation in §3) → it is
default-off; the FINAL default-config A/B (split-K + gather only,
`pm{3b,8b}_final{,ctl}.json`) is the shipping scoreboard: [FINAL_AB]

### 6b. Three-way vs the round-1 nvfp4/bf16 baselines (TPOT, pm/nv = nv/pm ms)

| | decode b8 | decode b16 | decode b1 | decode b4 |
|---|---|---|---|---|
| 3B r2 → r3 | 0.54-0.58 → [3B_B8] | 0.58-0.62 → [3B_B16] | 0.74-0.76 → [3B_B1] | → [3B_B4] |
| 8B r2 → r3 | 0.56-0.58 → [8B_B8] | 0.60-0.61 → [8B_B16] | 0.73 → [8B_B1] | → [8B_B4] |

### 6c. Mid-M / chunked-prefill serve points (the regime the main protocol
barely samples)

[MIDM]

## 7. Next biggest overhead (feeds round 4)

1. **qkv/o-class decode GEMM at M≤16 is the remaining latency floor** — split-K
   does NOT pay there (K too short: 22-32 union k-tiles; full-grid prologue +
   fixup overhead ≥ saved serial iterations; measured 0.8-1.0x). 3B qkv sits at
   19.8 µs vs a ~6.1 µs weight floor, 8B qkv 24.8 vs ~8. The named lever is the
   **Marlin-style W4A16 tensor-core small-M kernel** (M=8..32): stream the
   packed weights once at HBM BW like the GEMV, but feed 16x8x16 HMMA so M=16
   is as cheap as M=1 (the batched LDS-bound GEMV loses 2-3x at M=16; Marlin
   proves near-optimal to batch 32). Numerics note: W4A16 at M≤16 SKIPS
   activation quant (like the shipped M≤4 GEMV routes) — decode-route gate
   moves in the improvement direction, not byte-identical.
2. **FP8 KV cache** (survey §6.7, near-free): we serve fp16 KV; halves KV
   read BW + capacity; TPOT gain grows with ISL/batch. Gate with T2; baselines
   must get the same flag.
3. **Norm-fuse revisit** (optional): the kernel + pass are bit-exact and ready;
   the loss is vs inductor-fused native norms at decode. If vLLM's native norm
   lowering changes (or on prefill-heavy serving), re-measure. A compile-cache
   key hook for out-of-tree passes would also de-landmine opt-in.
4. **Structural +19% weight bytes vs pure-FP4** (MXFP8 ring) — unchanged bound.

## 8. Reproduce

```bash
B=/home/luofan/vllm_integ/bench_r3
# probes (graph-replay timing + bit gates):
docker exec -w /home/luofan/plaquant-vllm-integ -e CUDA_VISIBLE_DEVICES=7 luofan-plaquant \
  /home/luofan/baselines/venv/bin/python -m promix.vllm_integ.splitk_probe --out $B/r3_splitk_probe.json
... -m promix.vllm_integ.norm_pack_probe --out $B/r3_norm_pack_probe.json
... -m promix.vllm_integ.gather_pack_probe --out $B/r3_gather_pack_probe.json
# gates (3B): bench_r3/run_gatesA.sh (split-K), run_gatesB.sh (norm-fuse opt-in, cache off)
# serve: bench_r3/run_final.sh <ckpt> <tag>   (default config vs PROMIX_SPLITK=0 PROMIX_GATHER_PACK=0)
# env switches: PROMIX_SPLITK=0, PROMIX_GATHER_PACK=0,
#               PROMIX_NORM_FUSE=1 (+VLLM_DISABLE_COMPILE_CACHE=1) opt-in (measured decode negative)
```
