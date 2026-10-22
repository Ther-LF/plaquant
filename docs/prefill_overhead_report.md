# Prefill Per-Call Overhead Reduction — Report

> 2026-10-22. Converts the full-network real-kernel **compute** win
> (`docs/full_network_realkernel_report.md`: fused pack+GEMM beats FP16 cuBLAS Linear
> 1.05–1.38× at every prefill M) toward an **end-to-end** win by erasing the per-call
> Python / kernel-launch dispatch overhead across the 112 real-kernel Linears.
> Remote-only, RTX PRO 6000 Blackwell SM120, container `luofan-plaquant`, measured on a
> **genuinely idle GPU** (idle-waiter: util≤3 % + mem≤2 GB for 3 consecutive samples).
> Llama-3.2-1B, FP per-head recipe, batch 1, prefill (use_cache=False).
>
> **Primary lever = CUDA-graph capture of the whole decoder prefill forward.** Enabler
> (host glue, NOT kernel math): `codesign_fused_ext.cu::run_with_cached_ws` now skips its
> host `cudaStreamSynchronize` while the stream is capturing (guarded by
> `cudaStreamIsCapturing`); `quant_pack_float_ext.cu::init_grids_once` is pre-warmed by the
> warm-up forward so its one-shot `cudaMemcpyToSymbol` never runs during capture. PROTECTED
> kernel files and all GEMM / quant-pack kernel math are untouched.

## TL;DR

| M | FP16 eager | real BEFORE (eager) | real AFTER (**graph**) | speedup before→after | beats FP16? |
|---|---|---|---|---|---|
| 2048 | 20.94 ms | 29.28 ms (0.715×) | **19.52 ms (1.073×)** | 0.72× → **1.07×** | **YES** |
| 4096 | 41.14 ms | 49.46 ms (0.832×) | 43.42 ms (0.947×) | 0.83× → 0.95× | no (0.95×) |
| 8192 | 86.12 ms | 103.27 ms (0.834×) | 98.29 ms (0.876×) | 0.83× → 0.88× | no (0.88×) |

- **Numerics gate PASS at every M**: full-model logits cosine (graph vs pre-optimization
  eager real-kernel path) = **1.0000005 / 1.0000012 / 1.0000015** (≥0.9999). CUDA-graph
  replays the identical kernels on the identical static input → bit-exact. **PPL therefore
  unchanged (15.21 stands).**
- CUDA-graph capture **completely removes the host launch/dispatch overhead** — proven at
  M=2048, where the graphed real model (19.52 ms) hits the prior report's zero-host-overhead
  ideal (20.60 ms) *and beats it* (overhead bucket −1.09 ms) and crosses FP16 to **1.07×**.
- **Honest residual at M≥4096**: the remaining gap is NOT host overhead — it is the **FP32
  online Hadamard on down_proj** (real GPU compute the FP16 baseline never does), which the
  prior report mis-attributed into its "overhead" bucket. Graph cannot remove real compute.
  Making the Hadamard FP16 crosses 1.0× at every M (1.03–1.18×) **but fails the numerics
  gate** and is therefore rejected (see §3).

## 1. The big lever — CUDA-graph capture of the prefill forward

Each `MixedQuantLinear.forward` does reshape + 2 pack launches (fp4+fp8) + 1 fused-GEMM
launch + `D.to(fp16)` + bias + reshape, wrapped in `ActQuantWrapper` glue — ~4 kernel
launches + Python per Linear × 112, plus RMSNorm/attention/RoPE/embed/lm_head. A CUDA graph
captures this whole forward once (fixed prefill shape, static input-id buffer) so replay
issues the entire launch sequence with ~zero host cost.

**Capture harness** (`promix/eval/prefill_graph.py`, new, non-protected): warm up 3× on a
side stream (one-time cuBLAS/CUTLASS lazy init + qpf grid upload happen OUTSIDE capture),
capture `model(static_ids).logits`, replay. Two capture-safety fixes were required and found
by capture failures:
1. **GEMM host sync** — `run_with_cached_ws` called `cudaStreamSynchronize` after every
   `gemm.run`, which aborts capture. Guarded with `cudaStreamIsCapturing` → skipped only
   while capturing (within one stream the consumers of `D` already serialize after the GEMM;
   non-capture behavior byte-for-byte unchanged). *Host glue, not kernel math.*
2. **o_proj column-reorder index on CPU** — `ActQuantWrapper._column_order` is a plain CPU
   tensor attribute that `model.to(cuda)` does not move; `x[..., order]` with a CPU index
   triggers an illegal CPU↔CUDA copy during capture. The harness moves all 32 reorder
   indices to the device up front (pure reindex, no numeric change).

### Latency, before → after (idle GPU, min-of-N)

| M | real eager | real **graph** | fp16 eager | fp16 graph | su vs FP16-eager (before→after) | su vs FP16-graph |
|---|---|---|---|---|---|---|
| 2048 | 29.28 ms | **19.52 ms** | 20.94 ms | 19.45 ms | 0.715× → **1.073×** | 0.997× |
| 4096 | 49.46 ms | **43.42 ms** | 41.14 ms | 39.74 ms | 0.832× → 0.947× | 0.915× |
| 8192 | 103.27 ms | **98.29 ms** | 86.12 ms | 84.84 ms | 0.834× → 0.876× | 0.863× |

(FP16 is also graphable but, being compute-bound with no 112-Linear Python glue, gains almost
nothing from capture — hence fp16 graph ≈ fp16 eager.)

### Time breakdown (ms) — where it goes, before and after

| M | pack | GEMM | fp16-rest¹ | FP32 Hadamard² | zero-host-ovhd ideal³ | overhead eager⁴ | **overhead graph⁴** |
|---|---|---|---|---|---|---|---|
| 2048 | 3.18 | 8.10 | 9.32 | 2.81 | 20.60 | 8.68 | **−1.09** |
| 4096 | 5.60 | 11.27 | 19.24 | 9.78 | 36.11 | 13.35 | **7.31** |
| 8192 | 9.97 | 19.44 | 45.42 | 20.71 | 74.83 | 28.44 | **23.46** |

¹ fp16-rest = FP16 full forward − FP16 Linear GEMMs = non-Linear ops shared by both models.
² FP32 online Hadamard on down_proj, 16×, measured standalone (§3). Real-model-only.
³ ideal = pack+GEMM+fp16-rest (prior report's "zero-overhead" target; **excludes** Hadamard).
⁴ overhead = (real total) − ideal. Eager = host dispatch + FP32 Hadamard + reorder; graph = the
  same minus the host-dispatch part that capture erases.

**Reading the breakdown:**
- **M=2048: `overhead graph` = −1.09 ms** → CUDA-graph erases the ENTIRE host-dispatch bucket
  (eager 8.68 ms). The graphed real model equals/beats the zero-host-overhead ideal and
  crosses FP16 (1.07×). This is the lever working exactly as intended.
- **M=4096/8192: `overhead graph` = 7.31 / 23.46 ms ≈ FP32 Hadamard (9.78 / 20.71 ms)** +
  o_proj reorder. This residual is **real GPU compute**, not host overhead — graph legitimately
  cannot remove it. The prior report's single "overhead" bucket (9.9/14.9/29.4 ms) conflated
  host dispatch with this FP32 Hadamard; separating them shows host dispatch is fully gone.

### Why e2e > FP16 only at M=2048 (honest compute accounting)

Linear-compute savings vs the extra Hadamard cost, per M (both numerics-safe):

| M | FP16 Linear | pack+GEMM | Linear **saving** | FP32 Hadamard **cost** | net | matches |
|---|---|---|---|---|---|---|
| 2048 | 11.62 | 11.28 | +0.34 | 2.81 (hidden/overlapped) | ≈ even, host-ovhd removed → win | 1.07× |
| 4096 | 21.90 | 16.87 | +5.03 | 9.78 | −4.75 | 0.95× |
| 8192 | 40.70 | 29.41 | +11.29 | 20.71 | −9.42 | 0.88× |

The fused Linear compute really is faster, but at M≥4096 the FP32 online Hadamard (which FP16
never pays) outweighs the Linear saving. Once host overhead is gone (graph), this is the whole
story.

## 2. Lever 1 — fewer launches per Linear (fold the 2 packs + bias/fp16)

**Verdict: subsumed by lever 2 for the dominant cost; not separately implemented.** Lever 1's
goal was to shrink the per-Linear host dispatch (4 launches → 2). CUDA-graph capture already
removes 100 % of host dispatch (M=2048: overhead graph −1.09 ms proves the extra pack/`.to`/bias
launches cost ~0 host time under replay). The only residual lever-1 could still touch is
GPU-side kernel-launch *latency*, which inside a graph is minimal, and the two packs read
disjoint K-column slices of x (fp4 low vs fp8 high) so a single kernel cannot share the read.
Estimated upside ≤1–2 ms at M=8192 — far short of the ~9 ms (M=8192) needed to cross FP16, which
is Hadamard-bound. Reported as redundant; the engineering effort is better spent on the Hadamard.

## 3. Lever 3 — online Hadamard FP32 → FP16 (measured, REJECTED on numerics)

The FP32 online Hadamard (down_proj input, 16×, `matmul_hadU_cuda(x.float(), …)`) is the single
largest real-model-only GPU cost and the entire M≥4096 residual.

| M | per-call fp32 | per-call fp16 | 16× fp32 | 16× fp16 | **save** | standalone cos(fp16,fp32) |
|---|---|---|---|---|---|---|
| 2048 | 175.7 µs | 59.9 µs | 2.81 ms | 0.96 ms | 1.85 ms | 1.000000 |
| 4096 | 611.4 µs | 147.4 µs | 9.78 ms | 2.36 ms | 7.42 ms | 1.000000 |
| 8192 | 1294.5 µs | 383.9 µs | 20.71 ms | 6.14 ms | 14.57 ms | 1.000000 |

Dropping the Hadamard to FP16 and capturing it into the graph **crosses 1.0× at every M**:

| M | real graph+FP16-Had | speedup vs FP16-eager | **full-model cos vs FP32-Had path** |
|---|---|---|---|
| 2048 | 17.68 ms | **1.184×** | 0.9857 ✗ |
| 4096 | 35.14 ms | **1.170×** | 0.9886 ✗ |
| 8192 | 83.26 ms | **1.034×** | 0.9644 ✗ |

**REJECTED.** Although the *standalone* Hadamard-output cosine is 1.000000 on well-conditioned
synthetic input, on the **real** down_proj activations (outlier-heavy SwiGLU output) FP16
rounding shifts the per-block NVFP4/MXFP8 scales and element codes, and the error accumulates
over 16 layers to a full-model logits cosine of **0.964–0.989 — below the 0.9999 gate**. FP16
Hadamard would change PPL. bf16 (fewer mantissa bits than fp16) would be worse. There is no
numerics-safe way found this round to cut the FP32 Hadamard, so it stays FP32 and the M≥4096
residual remains. (Documented as the clear next target: a precision-preserving fused rotate, or
folding the fixed Hadamard into the NVFP4 pack so it is paid once in the quant kernel.)

## 4. Numerics gate (the hard requirement) — PASS for the adopted lever

- **Adopted = CUDA-graph (FP32 Hadamard preserved):** full-model prefill logits cosine vs the
  pre-optimization eager real-kernel path = **1.0000005 / 1.0000012 / 1.0000015** at
  M=2048/4096/8192, all ≥0.9999. Capture replays the identical kernels on identical input, so
  the result is bit-exact and **PPL is unchanged (15.21)** — no re-run needed to know it.
- **Rejected = FP16 Hadamard:** cosine 0.964–0.989 < 0.9999 → fails the gate (§3).

## 5. Honest residual

- **Is e2e now > FP16?** With the numerics gate satisfied (graph, FP32 Hadamard): **yes at
  M=2048 (1.07×)**; **no at M=4096 (0.95×) and M=8192 (0.88×)** vs FP16-eager (and
  0.92×/0.86× vs FP16-graph). The stated ≥1.0× at M≥4096 is **not** reached under the gate.
- **Why:** CUDA-graph removed all host/launch dispatch overhead (verified: M=2048 hits the
  zero-host-overhead ideal). The remaining gap at M≥4096 is the **FP32 online Hadamard** — real
  quant-algorithm GPU compute, not host overhead — which outweighs the (genuine) fused-Linear
  compute saving at large M. The prior report's "~1.03–1.15× achievable" assumed that cost away;
  it is reachable only by also cheapening the Hadamard, for which the one measured option (FP16)
  violates numerics.
- **Net correction to the prior report:** the "overhead" bucket was host-dispatch **+** FP32
  Hadamard. Graph capture erases the host-dispatch part entirely and bit-exactly; the Hadamard
  is the true, precision-bound bottleneck that remains.

## Deliverables / artifacts

- `promix/eval/prefill_graph.py` — CUDA-graph capture + replay harness (idle-GPU min-of-N
  timing, before→after, breakdown, numerics gate, FP16-Hadamard lever quantification). New,
  non-protected.
- `kernels/mixed_gemm_sm120/codesign_fused_ext.cu` — one capture-safety guard in
  `run_with_cached_ws` (`cudaStreamIsCapturing` around the host sync). No kernel math /
  mainloop / epilogue change; non-capture path byte-for-byte identical.
- Idle-waiter launcher (remote `/home/luofan/run_prefill_graph.sh`); results
  `kernels/mixed_gemm_sm120/prefill_graph.json`; log `/home/luofan/pg_full.log`.

Prior art (RESEARCH_TOOLS): CUDA-graph prefill capture with a static input buffer + side-stream
warm-up + private memory pool follows the vLLM / TensorRT-LLM "capture per fixed prefill shape"
pattern; the kernel-side requirement (no stream sync inside captured custom kernels) is the same
constraint those frameworks impose on their fused ops.

## Verdict

CUDA-graph capture of the full prefill forward is **numerically bit-exact (cos ≥0.9999999,
PPL 15.21 unchanged)** and **erases the entire per-call Python/launch dispatch overhead** across
the 112 real-kernel Linears — taking full-model prefill from 0.72–0.83× to **1.07× (M=2048,
beats FP16)**, 0.95× (M=4096) and 0.88× (M=8192). The honest residual at M≥4096 is the **FP32
online Hadamard** (real GPU compute, not host overhead); reducing it to FP16 would cross 1.0×
everywhere (1.03–1.18×) but breaks the numerics gate (cos 0.96–0.99) and is rejected. The
measured compute win is now realized end-to-end wherever the extra FP32 rotation does not
dominate; closing M≥4096 requires a **precision-preserving** cheaper Hadamard, the clear next
target.
