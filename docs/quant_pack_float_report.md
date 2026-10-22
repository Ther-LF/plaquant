# quant_pack_float — Fused float-input activation quantize+pack kernel (SM120)

> 2026-10-22. The piece that turns the single-layer real-kernel path from
> "GEMM-fast but pack-bound" into "end-to-end faster than FP16". A new, non-protected
> CUDA kernel (`kernels/mixed_gemm_sm120/quant_pack_float_ext.cu`) that goes
> **FP16/BF16/FP32 activation → packed NVFP4/MXFP8 codes + interleaved SF in ONE pass**,
> bit-exact to the torch spec quantizer, replacing the plain-PyTorch float→code quantize
> that was 93–98 % of e2e. Remote-only: built/run in container `luofan-plaquant`
> (10.208.130.90), profiled in `luofan-ncu`, RTX PRO 6000 Blackwell SM120.
>
> PROTECTED kernel files unchanged (`b1_fused_ext.cu`, `fused_gemm_ext.cu`,
> `nvfp4_gemm_ext.cu`, `mxfp8_gemm_ext.cu`, `mixed_gemm.py`, `pure_presf_ext.cu`,
> `codesign_fused_kernel.hpp`, `codesign_sm120_bs_mma.hpp`, `b1_tma_ext.cu`). New kernel is
> a standalone TU; `mixed_linear.py` (non-protected promix) edited to call it.

## The problem

The single-layer integration (`docs/single_layer_realkernel_report.md`) was correct
(cosine 0.999999) and its fused GEMM beat FP16 1.45–2.10×, but the per-forward activation
quant+pack was done in **plain PyTorch** (`_round_to_nearest_value` broadcast over the FP
grids) and cost **16–65 ms — 93–98 % of e2e**, making e2e ~25× slower than FP16. The
round-18 `cd.quant_pack_fp4/fp8` only packs *already-quantized* codes (µs); the slow part
was the float→code quantize. There was no fast CUDA kernel for float → packed codes +
interleaved SF.

## What was built

`quant_pack_float_fp4(x, M, N, K[, row_stride, col_off])` and `..._fp8(...)` — one CUDA
kernel each that, per block (16 for NVFP4, 32 for MXFP8):
1. computes the block scale — **NVFP4**: block absmax → E4M3 scale **rounded UP**
   (hardware `__nv_cvt_float_to_fp8` RNE then bump-if-below == `_ceil_to_value`);
   **MXFP8**: block absmax → UE8M0 power-of-2 (`frexpf`-exact `ceil(log2)`);
2. quantizes each element — NVFP4: FP4 E2M1 code via the 8-value grid with ties-to-even +
   spec sign masking; MXFP8: FP8 E4M3 byte via hardware `__nv_cvt_float_to_fp8` (RNE);
3. writes **packed codes** (2×FP4/byte for NVFP4, 1 byte/FP8) + the block SF directly in
   the **CUTLASS interleaved atom layout** (`tile_atom_to_shape_SFA`, the ILV layout
   `fused_gemm_prepacked` / `launch_best_codesign` consume) — all in one pass.

Matches `promix.quantize.weight_pack.quantize_nvfp4_spec` / `quantize_mxfp8_spec` +
`cd.quant_pack_*(ilv=True)` **bit-for-bit**. `MixedQuantLinear._pack_activation` now calls
it with `row_stride=K, col_off=0 / K_low` so the two K-segments are read **in place** from
the wide activation — no pre-slice or fp32-convert copies.

### Design levers (and what each bought)
- **Bit-exact first (TDD gate), then speed** — probe below.
- **Hardware fp8 conversion intrinsics** instead of a 127-entry E4M3 **constant-memory grid
  binary search** (the original bottleneck: fp8 at 0.1–0.25 TB/s from divergent
  per-element constant loads). fp8 → 108 µs → and then → 34 µs. Verified still bit-exact.
- **Warp-cooperative MXFP8** (one 32-lane warp per 32-block, coalesced loads, warp-shuffle
  absmax) — fixes the low-occupancy latency floor on the tiny high-variance segments
  (K_high=256 → only ~16 K threads one-thread-per-block). fp8 34 µs → **9.9–28.8 µs**,
  1.06–1.46 TB/s. NVFP4 kept the simple one-thread-per-16-block design (already
  SM-bound/near-memory-bound; the warp scheme *hurt* it via half-idle lanes + scattered
  single-byte packed writes).
- **Strided in-place read** (`row_stride`/`col_off`) — removes the two `.contiguous()`
  slice copies + the fp32 cast per forward (~126 µs/layer of pure copy at M=2048). Pack at
  M=2048 dropped 0.50 → **0.19 ms**.

Prior art (RESEARCH_TOOLS): KernelWiki `pr-TensorRT-LLM-13628` ("Fuse FP8 1x128 quantize +
UE8M0 scale pack on SM100"), `pr-sglang-10101` (NVFP4 block-scaled, small-M), hw/nvfp4 —
all fuse the per-block absmax→scale→pack in one pass and use hardware block-scale casts;
this kernel follows the same pattern (fused absmax/scale/cast/pack, interleaved-SF write).

## 1. Bit-exact probe (GATE, before timing) — PASS

`qpf_probe.py`: for all 7 Llama-3.2-1B Linear shapes × M∈{2048,4096} × dtype∈{fp32,fp16,bf16},
`torch.equal` of the kernel's `{packed_data, interleaved_SF}` vs `quantize_*_spec` +
`cd.quant_pack_*(ilv=True)` — for **both** the contiguous-segment and the strided
(full-tensor, `col_off`) call paths.

**`QPF_BITEXACT_PASS`** — packed FP4/FP8 data AND interleaved SFA all byte-identical
(42 cells × 4 buffers). The hardware intrinsics reproduce the torch grid quantizer exactly
(incl. negative-rounds-to-zero → 0x80, E4M3 round-up ties, UE8M0 exponent). Because the SF
values and the pack/interleave path are identical, the fused-GEMM output is unchanged.

## 2. Pack latency + roofline (new vs torch)

Single-layer harness (`single_layer_realkernel.py`, 7 Linears summed), **free GPU**, fp32
synthetic activations:

| M | pack torch | **pack fused** | speedup | GEMM-only | pack/GEMM |
|---|---|---|---|---|---|
| 2048 | 16.20 ms | **0.193 ms** | **84×** | 0.503 ms | 0.38 |
| 4096 | 31.56 ms | **0.338 ms** | **93×** | 0.702 ms | 0.48 |
| 8192 | 65.52 ms | **0.621 ms** | **106×** | 1.213 ms | 0.51 |

Min-of-N micro-bench (`qpf_lat.py`, per-kernel, near-clean): fp4 down8192 **200 µs
(1.32 TB/s)**, qkv/down2048 53 µs (1.23–1.24 TB/s); fp8 down8192 **28.8 µs (1.46 TB/s)**,
qkv/down2048 9.9 µs (1.06 TB/s).

**ncu BW % of roofline** (`luofan-ncu`, down8192 kernel):

| kernel | DRAM % peak | SM throughput % | bound by |
|---|---|---|---|
| qpf4 (NVFP4) fp32 | **62.9 %** | **91.4 %** | SM (compute) — near-saturated |
| qpf8 (MXFP8) fp32 | 47.4 % | 66.9 % | — |
| qpf4 fp16 | 43.1 % | 92.5 % | SM (reads ½ bytes) |
| qpf8 fp16 | 26.9 % | 67.8 % | — |

NVFP4 is **compute(SM)-bound at ~91 %** — the per-element divide + grid round-to-nearest-even
+ E4M3-round-up scale. It is effectively at its achievable ceiling; pack is no longer the
bottleneck. (Pushing DRAM higher would need branchless FP4 rounding, but SM is the limiter
and the pack is already a small fraction of the GEMM — not worth the bit-exactness risk.)

## 3. Single-layer e2e vs FP16 (the real speedup now) — free GPU

| M | GEMM-only | GEMM/FP16 | pack fused | **e2e (pack+GEMM)** | FP16 cuBLAS | **e2e/FP16** |
|---|---|---|---|---|---|---|
| 2048 | 0.503 ms | 1.43× | 0.193 ms | 0.821 ms | 0.719 ms | **0.88×** |
| 4096 | 0.702 ms | 1.95× | 0.338 ms | 1.232 ms | 1.367 ms | **1.11×** |
| 8192 | 1.213 ms | 2.11× | 0.621 ms | 2.329 ms | 2.561 ms | **1.10×** |

**e2e now BEATS FP16 at M=4096 (1.11×) and M=8192 (1.10×)** — the prefill regime — up from
0.04× with the torch pack. Cosine vs fake-quant stays **0.999999** (whole-layer), vs fp16
0.9952 (the genuine NVFP4 4-bit loss). Before→after e2e: **~25× slower → faster than FP16.**

### Honest residual: M=2048
At M=2048 wall-clock e2e (0.821 ms) is 0.88× FP16 (0.719 ms). But pack+GEMM **compute**
(0.193+0.503 = 0.696 ms) is itself below FP16 — the 0.125 ms gap is per-call Python/launch
overhead (reshape + 2 kernel launches + `D.to(fp16)` + bias + reshape) that FP16's single
`x@Wᵀ` doesn't pay. This is the smallest GEMM where the fused advantage is thinnest
(consistent with round-18's qkv·2048 being the one losing shape). Fusing the two pack
launches into one and avoiding the output copy would close it; left as follow-up.

## Measurement note (shared box)
The box (10.208.130.90) was heavily contended throughout (all 8 GPUs 20–100 %). The fused
GEMM's persistent/cooperative scheduler is very contention-sensitive (grabs all SMs), so its
latency inflates 2–4× under any co-tenant and min-of-N can't recover it, whereas FP16 cuBLAS
tolerates contention. The **free-GPU** numbers above (captured by a waiter that polled for an
idle GPU and ran immediately) reproduce the report's clean GEMM/FP16 (GEMM/FP16 1.43–2.11×)
and are the fair comparison. ncu %-of-peak is contention-independent (serialized replay).

## Deliverables / artifacts
- `kernels/mixed_gemm_sm120/quant_pack_float_ext.cu` — new kernel + torch binding
  (`quant_pack_float_fp4/fp8`, strided in-place read).
- `promix/quantize/mixed_linear.py` — `MixedQuantLinear(qpf=...)` fast path (strided, fp16
  read, no copies); torch path kept as fallback.
- `promix/eval/single_layer_realkernel.py` — loads `qpf`, threads it through, phase-3 now
  reports pack torch→fused + e2e(fast)/FP16.
- Tests/bench: `qpf_probe.py` (bit-exact gate, contiguous + strided), `qpf_lat.py`
  (per-kernel min-of-N + ncu target), `qpf_e2e.py` (min-of-N e2e).
- Logs: `logs/_single_layer_FREE.log` (clean e2e), `logs/_qpf_probe5.log`
  (`QPF_BITEXACT_PASS`), `logs/_qpf_lat4.log`, `logs/_qpf_ncu.log`.

## Verdict
A bit-exact fused float→packed-codes+interleaved-SF kernel (84–106× faster than the torch
quantize, NVFP4 at 91 % SM / 63 % DRAM) drops the per-forward pack from 16–65 ms to
0.19–0.62 ms (0.16–0.51× of the GEMM). The single decoder layer's 7 real-kernel Linears now
run **end-to-end faster than FP16 (1.10–1.11× at M=4096/8192)** with cosine 0.999999 —
realizing the fused GEMM's 1.43–2.11× advantage at the layer level, which the torch pack had
hidden.
