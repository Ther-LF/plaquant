# vLLM perf-optimization loop — ROUND 1: decode GEMV wired into the ProMix vLLM path

**Status: DONE.** The #1 overhead named by `docs/vllm_integ_phase23.md` (decode routed
through the 128×128 prefill fused GEMM) is fixed: decode-size batches now route through
the phase-adaptive **W4A16 weight-streaming GEMV** (round-12 kernel + a NEW batched
variant), per captured CUDA-graph size. Numerics gates GREEN (decode-route PPL actually
**improves** by 0.11-0.34). Decode TPOT at b1 drops **30-31 %** on 3B **and** 8B;
ProMix/NVFP4 decode-b1 throughput ratio moves **0.52-0.54x → 0.73-0.77x**;
ProMix/BF16 decode b1 is now **1.26-1.33x (3B) / 1.58-1.65x (8B)**. No regression at
any other point (b8/b16 and all prefill rows reproduce the Phase-2/3 baseline).

Raw JSONs: `docs/vllm_perf_round1_data/` (promix rows + gates + graph-timed probe);
full set on the remote under `/home/luofan/vllm_integ/bench_r1/`.

---

## 1. What changed

| file | change |
|---|---|
| `kernels/mixed_gemm_sm120/decode_gemv_ext.cu` | NEW `decode_gemv_w4a16_mb`: batched (M ≤ 16) weight-only W4A16 GEMV. Warp-per-output-row with per-lane fp32 `acc[MB]`, K-tiled smem staging of the (M,K) activation (`KT%256` keeps every lane's k-walk the exact strided sequence of the M=1 kernel → **per-row bit-identical to `decode_gemv_w4a16`**, torch.equal-proven on all 8 merged shapes), float4 LDS (the first scalar-LDS version was LDS-bound and 2× slower at M≥8). |
| `promix/quantize/mixed_linear.py` | M-based decode route in `MixedQuantLinear.forward`: M=1 → round-12 GEMV; 2 ≤ M ≤ min(`PROMIX_DECODE_M_MAX`, per-shape crossover) → batched GEMV; larger M → fused W4A4 GEMM (unchanged). `PROMIX_DECODE_MODE=w4a4` keeps the legacy quant-act GEMV at M=1 (M>1 w4a4 falls through to the fused GEMM, which IS W4A4). |
| `promix/vllm_integ/plugin.py` | Decode buffers derived at **load time** from the existing checkpoint — no exporter/checkpoint change: the GEMV streams the same packed code bytes the fused GEMM uses (checkpoint `weight_lo_packed` == `nv.pack_fp4` == the GEMV's even-k-low-nibble layout; `weight_hi_bytes` is raw FP8); scales become raw fp32 effective scales with the per-tensor g folded (`E4M3(sf_lo)·g`, `2^(sf_hi−127)·g`) — exactly `pack_weight_mixed`'s `W_*_dec`/`S_*_dec` recipe. `MixedQuantLinear` now gets `dg`. |
| `promix/vllm_integ/kernels.py` | `load_decode_gemv()` (JIT, cached; `PROMIX_DECODE_GEMV=0` restores the Phase-2/3 all-fused behavior). |
| `promix/vllm_integ/gates/gate_t2.py` | `--decode-route-m N`: caps `max_num_batched_tokens` so the full wikitext PPL is scored **through the decode route**; signed-dPPL gate (W4A16 may legitimately improve). |
| `promix/vllm_integ/decode_probe.py` | NEW probe: bit/cos gates + GEMV-vs-fused crossover bench under **CUDA-graph replay timing**. |

### CUDA-graph safety
The route branches on the flattened M **in Python inside the opaque
`torch.ops.vllm.promix_linear` op**. vLLM pads decode batches to its fixed capture
sizes, so M is static per captured graph: the branch is evaluated once at capture and
the chosen kernels are recorded — no on-device branching, no dynamic shapes. Verified:
the full default pipeline (Dynamo → inductor → 51 PIECEWISE + 51 FULL graphs) captures
and replays with the GEMV kernels inside the decode-size graphs; the post-graphs T2
re-gate reproduces the Phase-2 numbers **exactly** (PPL |Δ|=0.0029, top-1 89.786 %).

### Route thresholds — the eager-timing trap (r1 lesson)
The first threshold table was fit on **eager** timings, where the fused path pays
~50-80 µs of host/launch overhead per call that graph replay erases; it picked
M≤16 thresholds and **regressed decode b8/b16 by 15-30 %** in the serve bench.
Re-probed under CUDA-graph replay (capture 10 iterations/graph, replay):

| shape (N,K) | graph-timed fused→GEMV ratio M=1 / 2 / 4 / 8 | m_max |
|---|---|---|
| 3B qkv (5120,3072) | 2.13 / 2.05 / 1.48 / 0.90 | 4 |
| 3B o (3072,3072) | 2.92 / 2.05 / 1.50 / 0.90 | 4 |
| 3B gate_up (16384,3072) | 1.69 / 1.28 / 0.94 / 0.51 | 2 |
| 3B down (3072,8192) | 2.44 / 1.60 / 1.12 / 0.63 | 4 |
| 8B qkv (6144,4096) | 1.74 / 1.21 / 0.80 / 0.46 | 2 |
| 8B o (4096,4096) | 2.82 / 1.95 / 1.35 / 0.81 | 4 |
| 8B gate_up (28672,4096) | 1.67 / 1.30 / 0.88 / 0.48 | 2 |
| 8B down (4096,14336) | 2.20 / 1.43 / 1.00 / 0.55 | 2 |

GEMV wins at M=1 (1.7-2.9×) and M=2 (1.2-2.1×) on EVERY shape; at M=4 only where the
margin survives the extra fp32→fp16 cast; at M≥8 the fused GEMM always wins (the GEMV
is LDS/FMA-bound per row, ~linear in M; the fused GEMM is flat below its 128-row
tile). Encoded heuristic: `m_max = 4 if N*K ≤ 17e6 or (N*K ≤ 26e6 and K ≥ 2N) else 2`.

## 2. Numerics gate (mandatory, ran FIRST)

| gate | result |
|---|---|
| T2 post-graphs re-gate (3B, prefill path, decode route wired) | **PPL 8.3703 vs ref 8.3674, \|Δ\|=0.0029 — identical to Phase 2** (top-1 89.786 % = the known Phase-1 root-caused engine-glue regime, unchanged). Decode wiring changed nothing on the prefill path and capture works. |
| T2 **through the decode route** (3B, full wikitext, `--decode-route-m 16`, every step a ≤16-token chunk through the W4A16 GEMV for all 4 kinds × all layers) | **PPL 8.2594 vs shared-g ref 8.3674 → signed ΔPPL = −0.108 (IMPROVEMENT), PASS.** As predicted (round-12): W4A16 decode removes activation quantization, so the decode route is *more* accurate than the W4A4 reference. |
| T2 decode-route M=1 subset (`--decode-route-m 1 --graphs --max-chunks 30`) | **PPL 8.4628 vs subset ref 8.8014 → signed ΔPPL = −0.339 (IMPROVEMENT), PASS** — the M=1 GEMV path scored under CUDA-graph replay; additionally torch.equal-bit-identical per row to the batched kernel gated above. |
| Kernel probe (all 8 merged shapes) | batched-vs-M=1 **bit-identical** (torch.equal); GEMV vs fp32 `x@W_deq.T` cos ≥ 0.99999988. |

## 3. Scoreboard — three-way serve bench (same Phase-2/3 protocol, idle box, GPU 7)

ProMix re-measured with the decode route; NVFP4/BF16 rows are the Phase-3 baselines
(same stack, same protocol, cross-checked then). "was" = Phase-3 ProMix.

### 3B decode TPOT (ms) and gen-throughput ratios

| case | TPOT was → now | pm/nvfp4 was → now | pm/bf16 was → now |
|---|---|---|---|
| isl1024_osl128_b1 | 6.12 → **4.23 (−31 %)** | 0.54 → **0.76** | 0.89 → **1.26** |
| isl128_osl1024_b1 | 5.93 → **4.09 (−31 %)** | 0.53 → **0.77** | 0.92 → **1.33** |
| isl1024_osl128_b4 | 6.42 → 6.04 | 0.52 → 0.55 | 0.94 → 0.99 |
| isl128_osl1024_b4 | 6.07 → 5.69 | 0.51 → 0.54 | 0.97 → 1.03 |
| b8 (both cases) | 6.75/6.24 → 6.74/6.24 | unchanged | unchanged |
| b16 (both cases) | 7.22/6.59 → 7.24/6.61 | unchanged | unchanged |

### 8B decode TPOT (ms) and gen-throughput ratios

| case | TPOT was → now | pm/nvfp4 was → now | pm/bf16 was → now |
|---|---|---|---|
| isl1024_osl128_b1 | 10.17 → **7.02 (−31 %)** | 0.52 → **0.73** | 1.12 → **1.58** |
| isl128_osl1024_b1 | 9.72 → **6.84 (−30 %)** | 0.53 → **0.75** | 1.16 → **1.65** |
| b4 | 10.11/9.88 → 10.27/9.74 | ~unchanged (±2 %, noise) | ~unchanged |
| b8 | 10.33/10.03 → 10.39/10.06 | unchanged | unchanged |
| b16 | 11.09/10.45 → 11.12/10.48 | unchanged | unchanged |

Prefill TTFT rows (both sizes) reproduce Phase-3 within noise (±2-6 % on the
bimodal-prone b1 points) — the prefill path is untouched.

### Honest summary vs the 1.7× gap
The round closed the decode-b1 chunk of the gap: pm/nvfp4 0.52-0.54 → **0.75-0.77**
(NVFP4's decode-b1 lead shrank from ~1.9× to ~1.3×). At b4 the 3B ratio improved
slightly (0.51-0.52 → 0.54-0.55); **b8/b16 decode and b4-b16 prefill are unchanged**
(still 0.53-0.61× of NVFP4) — the GEMV route cannot win there (measured, above), so
that part of the gap needs a different lever.

## 4. Next biggest overhead (feeds round 2)

Ranked by measured remaining gap:

1. **Small/mid-M GEMM regime (decode b8-b16 AND chunked-prefill small-M): still
   0.53-0.61× of NVFP4.** At M=8..512 we run the single 128×128 co-design config +
   2 standalone pack launches per Linear, while vLLM-NVFP4 runs tuned FlashInfer fp4
   GEMMs with quant fused into the producing norm. Two sub-levers: (a) wire the
   per-shape `config_dispatch`/small-M configs (round-2's cfg10 class) into the
   plugin; (b) fuse activation quant+pack into the producer (RMSNorm epilogue), the
   Phase-2/3 follow-up #2. These also lift the prefill TTFT rows (0.55-0.64×).
2. **fp32 decode-scale traffic in the GEMV (+~2 bits/weight on the low segment ≈
   +18 % weight bytes/token at M≤4):** the decode buffers store per-block scales as
   fp32 (round-12 design); storing the checkpoint's 1-byte E4M3/UE8M0 scales and
   decoding in-kernel cuts decode-b1 TPOT by an estimated further ~8-12 %.
3. **Batched-GEMV scaling (M=2..16):** per-row LDS/FMA cost makes GEMV ~linear in M;
   a weight-reuse-in-register / half2 variant could push the crossover from 2-4 to
   8-16 and attack the b8/b16 rows from the GEMV side instead of lever 1.
4. **Structural +24 % bits of the MXFP8 ring** (accuracy-motivated; not a round-2
   target).

## 5. Reproduce

```bash
# probe (bit/cos gates + graph-timed crossover):
docker exec -w /home/luofan/plaquant-vllm-integ -e CUDA_VISIBLE_DEVICES=7 luofan-plaquant \
  /home/luofan/baselines/venv/bin/python -m promix.vllm_integ.decode_probe
# decode-route PPL gate (3B):
... gates.gate_t2 --ckpt /home/luofan/vllm_integ/promix-llama-3.2-3b --decode-route-m 16
# serve bench (promix rows; baselines reused from docs/vllm_integ_phase23_data):
... bench_serve --model /home/luofan/vllm_integ/promix-llama-3.2-3b --tag promix_3b \
    --dtype float16 --out bench_r1/promix_3b.json
# env switches: PROMIX_DECODE_GEMV=0 (disable route), PROMIX_DECODE_M_MAX (cap, def 16),
#               PROMIX_DECODE_MODE=w4a4 (legacy quant-act GEMV at M=1)
```
