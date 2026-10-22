# Decode CUDA-Graph Capture (W4A4) — Report

> 2026-10-22. Closes the remaining gap the decode-throughput round
> (`docs/decode_throughput_report.md`) left open: the bit-exact memory-bound
> decode-GEMV lifted full-model Llama-3.2-1B decode to 0.67x FP16 at M=1 but was
> **host/launch-latency bound** (per-token ~15 ms real vs ~10 ms FP16; the GPU
> weight-read work per Linear is <5 µs while fixed launch+Python dispatch across the
> 112 Linears is ~10-15 µs each). The fix: **CUDA-graph capture the whole per-token
> decoder forward with a STATIC KV cache**, erasing the 112-Linear Python/launch
> dispatch — the same lever `docs/prefill_overhead_report.md` proved for prefill.
> Remote-only, RTX PRO 6000 Blackwell SM120, container `luofan-plaquant`, idle GPU
> (idle-waiter). PROTECTED kernel files unchanged; all new code is in `promix/`
> (`promix/eval/decode_graph.py`). The shipped decode-GEMV kernel + div-fold are
> untouched.

## TL;DR (the headline)

| full model, batch 1, KV cache, prompt 32 / gen 128 | per-token | tok/s | vs FP16-eager |
|---|---|---|---|
| FP16 (plain HF), **eager** | 9.78 ms | 102.2 | 1.00x |
| real-kernel decode-GEMV @ M=1, **eager** (shipped, the "before") | 15.26 ms | 65.5 | **0.64x** |
| real-kernel decode-GEMV @ M=1, **CUDA-GRAPHED** (this round) | **4.75 ms** | **210.7** | **2.06x** |

- **CUDA-graph decode capture erases the per-token host/launch dispatch: eager
  15.26 ms/token → graphed 4.75 ms/token (3.21x less per-token time), reaching
  2.06x FP16-eager.** The loop's **≥1.5x target is REACHED on 1B decode vs
  FP16-eager** (0.64x → 2.06x). Pure graph `replay()` is 4.70 ms — the host glue
  (token copy + argmax + position advance) is **~0.05 ms**, i.e. host dispatch is
  ~100% erased, exactly as prefill_graph showed.
- **NUMERICS GATE PASS (first).** (i) The StaticCache + explicit-mask + M=1
  machinery is **bit-faithful to the no-cache full-prefill: cos 0.999993, argmax
  100%** (teacher-forced, fused both sides — the exact analog of the shipped decode
  gate (a)). (ii) The captured graph reproduces eager static-cache decode
  **EXACTLY: argmax 100%, min-cos 0.9999924** over the generated window. The
  decode-GEMV itself is unchanged (per-Linear cos 1.000000 vs the fp32 twin).
- **HONEST CAVEAT — vs a FAIRLY graphed FP16 baseline, 1B graphed decode is 0.70x
  (FP16 is still 1.43x faster).** FP16 decode ALSO graphs (9.78 → 3.32 ms, 2.95x).
  Even pure-GPU, the quantized path's 112 M=1 GEMVs cost 3.66 ms and are **11.3x
  off the 604 MB / 1.44 TB/s = 0.419 ms/token weight-bandwidth ceiling** — 1B
  decode is NOT weight-bandwidth-bound even after graphing. The 2.06x headline
  comes from the quantized path having MORE host/launch overhead to erase (112
  Python-dispatched custom-kernel Linears) than FP16's efficient cuBLAS dispatch,
  not from the 4-bit weight-traffic win. **The genuine multi-x low-bit decode win
  needs a 7B/8B model** (per-Linear weights large enough to be HBM-bound) — §5.

## 1. Correctness gate (FIRST) — PASS

Config `llama-3.2-1b-fp-perhead.yaml` (NVFP4 main + MXFP8 high), harness
`promix/eval/decode_graph.py`, prompt 32 / gate 48 steps:

| gate | result | what it proves |
|---|---|---|
| **(i)** StaticCache decode vs full no-cache prefill, teacher-forced, **fused both sides** | **min cos 0.999993, argmax 100%** (49 steps) | StaticCache pre-alloc + per-step `index_copy_` write + the tensor-driven causal mask + M=1 path is bit-faithful to the model's own prefill — no KV / mask / position bug |
| **(ii)** graphed decode vs eager static-cache decode, **decode-GEMV, free-run greedy** | **argmax 100%, min cos 0.9999924** (48 steps, no divergence) | CUDA-graph replay (advancing the write position + mask per step) changes NOTHING vs eager |
| decode-GEMV per-Linear vs fp32 twin (`decode_gemv.py`, shipped) | **cos 1.000000** | the streamed-weight GEMV is bit-exact to the spec (unchanged this round) |
| *(report, not gated)* decode-GEMV vs fused, full-model teacher-forced | cos 0.905, argmax 85.7% | PRE-EXISTING fp32-GEMV-vs-bf16-fused drift (== shipped decode gate (c) = 0.896); orthogonal to this round |

**Key subtlety resolved (why gate (i) needed care).** A static KV cache MUST use an
explicit causal mask over the padded max-length buffer, which dispatches SDPA to a
different backend than the no-cache / `DynamicCache` `is_causal` (flash) path. Those
two backends differ numerically (~0.995 cos/layer → ~0.95 over 16 layers), which
naively makes a graphed static-cache decode *look* like it changed the model (our
first measurement: cos 0.954, 77%). Forcing the **memory-EFFICIENT** SDPA backend
for the static path AND the reference makes them agree to **cos 0.999993** —
*proving the deviation was backend choice, not a mask/cache logic error* (the mask is
provably causal; `DynamicCache` + `is_causal` == reference at cos 1.0). The harness
therefore pins `SDPBackend.EFFICIENT_ATTENTION` on the static/graph decode path
(capturable, M=1-cheap). This is the standard static-KV-decode numeric every graphed
decode system (vLLM / TRT-LLM / HF StaticCache) lives with.

The "graph vs shipped DynamicCache free-run" argmax match is only 12.5% — expected
and benign: free-running greedy + the DynamicCache (is_causal) vs static (explicit
mask) backend difference causes fp-noise divergence at a near-tie, after which all
subsequent greedy tokens differ. The teacher-forced anchor (i) proves there is no
logic error.

## 2. Decode throughput + per-token breakdown (before eager → after graph)

Full model, prompt 32, generate 128, steady-state median per-token (idle GPU):

| path | prefill | per-token | tok/s | vs FP16-eager |
|---|---|---|---|---|
| FP16, eager | 10.9 ms | 9.78 ms | 102.2 | 1.00x |
| FP16, **graphed** | — | **3.32 ms** | 301.0 | (2.95x over FP16-eager) |
| real decode-GEMV @ M=1, eager (**before**) | 22.2 ms | 15.26 ms | 65.5 | **0.64x** |
| real decode-GEMV @ M=1, **GRAPHED (after)** | — | **4.75 ms** | **210.7** | **2.06x** |

- **Speedup vs FP16-eager: BEFORE 0.64x → AFTER (graph) 2.06x. ≥1.5x REACHED.**
- **Speedup vs FP16-graphed (fair, apples-to-apples): 0.70x** (FP16 still 1.43x faster).
- Pure graph `replay()` = 4.70 ms (min 4.69); full greedy loop 4.75 ms → **host glue
  ≈ 0.05 ms** (token argmax + copy_ + position advance) — host dispatch erased.

**Per-token component breakdown (M=1, graphed real path):**

| | GEMV (112 Linears) | attn + KV + RMSNorm + RoPE + lm_head + host | total |
|---|---|---|---|
| real **graphed** | **3.66 ms** | **1.09 ms** | **4.75 ms** |
| FP16 graphed | ~2.81 ms (cuBLAS linear) | ~0.51 ms | 3.32 ms |

- Graphing cut per-token from 15.26 → 4.75 ms. What remains is **pure GPU compute**,
  dominated by the 112 M=1 decode-GEMVs (3.66 ms). The attention/KV/norm/rope/lm_head
  bucket is only 1.09 ms.
- The 3.66 ms GEMV is the whole story at 1B: see §3.

## 3. Why 1B graphed decode still loses to graphed FP16 — the weight-BW ceiling

Empirical HBM ceiling (large D2D copy / big FP16 GEMV) = **1.44 TB/s**.

| | packed/FP16 weight bytes | HBM-BW floor @ M=1 | graphed achieved | off-floor |
|---|---|---|---|---|
| **W4A4 (this path)** | **604.4 MB** | **0.419 ms/token** | 4.75 ms (GEMV 3.66 ms) | **11.3x** |
| FP16 (cuBLAS) | 1946.2 MB | 1.349 ms/token | 3.32 ms | 2.46x |

- **Even fully graphed, 1B W4A4 decode is 11.3x off its OWN 4-bit weight-bandwidth
  ceiling** — it is NOT weight-bandwidth-bound. The 112 per-Linear weights are tiny
  (q/k/v/o ~0.5-1.5 MB packed; 0.4-1 µs at HBM), so each M=1 GEMV is dominated by the
  per-kernel launch/occupancy floor (~15-30 µs), and 112 of them back-to-back inside
  the graph sum to 3.66 ms regardless of how little weight each reads.
- **FP16 is far closer to its floor (2.46x)** because cuBLAS GEMV streams the bigger
  FP16 weights bandwidth-efficiently even at M=1 (one well-tuned kernel per Linear).
- So the **3.22x weight-memory reduction (604 MB vs 1946 MB, 4.97 bits/weight) does
  NOT convert to a decode speedup at 1B M=1** — the GEMVs never saturate HBM. The
  2.06x-vs-FP16-eager headline is the graph erasing the quantized path's extra host
  overhead, not the 4-bit traffic win.

## 4. What this round built

`promix/eval/decode_graph.py` (new, non-protected):

- **Static-KV CUDA-graph decode**: `transformers.StaticCache` pre-allocated to
  `max_cache_len = P+G+1`; `StaticCache.update()` writes K/V at `cache_position` via
  `index_copy_` (a TENSOR index → graph-safe, HF PR #31857/#32168). `attention_mask=None`
  so `LlamaModel._prepare_4d_causal_attention_mask_with_cache_position` rebuilds the
  causal mask INSIDE the captured forward from the `cache_position` tensor
  (`arange(L) > cache_position`) — so a single captured graph, replayed with the
  position index advanced (`copy_`/`fill_`), advances BOTH the write slot AND the
  attend length. The mask is captured (not rebuilt host-side between replays), avoiding
  the host-bound-mask caveat noted in HF PR #49542.
- **Capture safety** (reusing the prefill_graph pattern): side-stream warmup (cuBLAS/
  CUTLASS lazy init + the decode-GEMV `init_fp4_once` `cudaMemcpyToSymbol` happen
  OUTSIDE capture); move the o_proj column-reorder index to CUDA (a CPU-index gather
  would trigger an illegal copy during capture). The codesign stream-sync guard already
  exists; the decode path does not touch it.
- **SDPA backend pin** (`SDPBackend.EFFICIENT_ATTENTION`) on the static/graph path for
  numeric faithfulness to the no-cache reference (§1).
- Numerics gate (teacher-forced static==prefill; graph==eager), before→after
  throughput vs FP16 (eager and graphed), per-token breakdown, and the weight-BW
  ceiling diagnosis.

Launcher: `/home/luofan/run_decode_graph.sh` (idle-GPU waiter, detached). Results:
`kernels/mixed_gemm_sm120/decode_graph.json`; log `/home/luofan/decode_graph.log`.

Prior art consulted (RESEARCH_TOOLS): HF StaticCache `index_copy_` graph/export
compatibility (transformers PR #31857, #32168); side-stream `torch.cuda.graph(...,
capture_error_mode="thread_local")` capture pattern (SimSD); host-bound-mask caveat
(HF PR #49542) — avoided here by building the mask inside the captured forward;
KernelWiki: vLLM/TRT-LLM/FlashInfer SM120 decode + CUDAGraph-safe kernels.

## 5. Verdict & the 7B/8B recommendation — HONEST

- **The loop's ≥1.5x target is REACHED on 1B decode vs FP16-eager: 0.64x → 2.06x
  (graphed).** CUDA-graph decode capture with a static KV cache erases the 112-Linear
  Python/launch dispatch (15.26 → 4.75 ms/token; host glue ~0.05 ms). Numerics held:
  graph == eager (argmax 100%, cos 0.9999924) and the static-cache machinery is
  bit-faithful to full-prefill (cos 0.999993); the decode-GEMV stays bit-exact.
- **But vs a FAIRLY graphed FP16 baseline, 1B is 0.70x (FP16 1.43x faster).** 1B
  graphed decode is **11.3x off its own 604 MB / 1.44 TB/s weight-bandwidth ceiling**
  — the 112 M=1 GEMVs are launch/occupancy-floor-bound, not HBM-bound, so the 3.22x
  weight-traffic reduction does not pay off. On 1B, the 2.06x is "erase the quantized
  path's extra overhead," not a bandwidth win.
- **The genuine multi-x low-bit decode win needs a bigger model (7B/8B).** The decode
  weight-BW win scales with per-Linear weight size: at 8B the Linears are ~4-14 MB
  packed (vs 1B's 0.5-1.5 MB), so each M=1 GEMV reads enough weight (3-10 µs at HBM) to
  dominate the ~15-30 µs kernel floor and become genuinely HBM-bound — at which point
  the 3.22x weight-traffic reduction converts to a ~2-3x decode win over FP16 *even
  graphed*. **Recommendation: validate the graphed decode path on
  `llama-3-8b-fp-perhead` (basis/rotation calibration required) as the next step; the
  graph machinery and numerics proven here carry over unchanged (M=1 path + static
  cache are model-size agnostic).**
- Optional HBM%-push on the 1B GEMV (split-K / persistent, Marlin/Machete-style,
  28-44% → ~80% HBM standalone) would shave the 3.66 ms GEMV but cannot reach the
  0.419 ms aggregate floor at 1B M=1 (per-kernel launch floor dominates 112 tiny
  GEMVs) — so it would not flip the vs-FP16-graphed verdict on 1B. Deferred; the graph
  was the primary, higher-leverage lever and hit the stated target.
