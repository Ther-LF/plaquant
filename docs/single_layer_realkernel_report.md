# Single-Layer Real-Kernel Integration — Report

> 2026-10-22. First end-to-end integration step: wire the real packed NVFP4+MXFP8
> fused GEMM (SM120, RTX PRO 6000 Blackwell) into ONE Llama-3.2-1B decoder layer's
> 7 Linears, replacing the fake-quant path. Validates single-layer correctness
> (cosine) + prefill speedup. NO full network, NO PPL, NO decode.
>
> Remote-only: built/run in container `luofan-plaquant` (10.208.130.90), free GPU 1.
> PROTECTED kernel files unchanged — new code is all in `promix/`.

## Deliverables

- `promix/quantize/weight_pack.py` — `pack_weight_mixed` (offline weight packer)
  + `probe_weight_sfb_bitexact` + spec-matching `quantize_nvfp4_spec` /
  `quantize_mxfp8_spec` (emit codes+SF whose dequant is bit-identical to
  `quant_utils.fake_quantize_*`).
- `promix/quantize/mixed_linear.py` — `MixedQuantLinear` (holds offline-packed
  W_packed/SFB; forward = fold activation quant+pack → `fused_gemm_prepacked` → +bias).
- `promix/eval/single_layer_realkernel.py` — validation harness (bit-exact probe,
  per-Linear + whole-layer cosine, prefill latency, real layer-8 capture).
- Results JSON: `kernels/mixed_gemm_sm120/single_layer_realkernel{,_real}.json`;
  logs `kernels/mixed_gemm_sm120/logs/_single_layer{,_real}.log`.

## Kernel contract (as wired)

- **Weight (offline, once at load):** split `W(N,K)` → `[:K_low]` NVFP4 (block-16) |
  `[K_low:]` MXFP8 (block-32), `K_high = K/8`. Quantize with spec rounding, pack FP4
  codes (`nv.pack_fp4`), interleave weight SF via the kernel's own load-time
  prepackers `cd.prepack_sf_fp4_b` / `cd.prepack_sf_fp8_b` — exactly the layout both
  `fused_gemm` and `fused_gemm_prepacked` consume.
- **Activation (per-forward):** split x → quantize each segment (spec rounding) →
  `cd.quant_pack_fp4/fp8(ilv=True)` folds data-pack + SF-interleave in one pass →
  `cd.fused_gemm_prepacked(...)` → dequantized GEMM D(M,N). Scale convention matches
  fake-quant: per-block SF carries the full scale, no extra per-tensor s_x/s_w; +bias.
- **Config:** `fused_gemm_prepacked` runs the production 128×128 NV4/MX2 tile
  (SharedStorage 88064 B, NV stages=4 MX stages=2). Per round-24's dispatcher
  (`select_codesign`), **128×128 NV4/MX2 is the chosen config for every shape here at
  M≥2048** — so the path used == `launch_best_codesign`'s pick. Small K_high
  (qkv/k/v/o/gate/up: 256) and large K_high (down: 1024) both produce cosine 0.999999.

## 1. Bit-exact SF probe (GATE, before anything else) — PASS

All 7 Linear shapes, `torch.equal`:

| check | result |
|---|---|
| spec-quant deq == `fake_quantize_nvfp4` | True |
| spec-quant deq == `fake_quantize_mxfp8` | True |
| weight SFB_lo == `prepack_sf_fp4_b` | True (×7) |
| weight SFB_hi == `prepack_sf_fp8_b` | True (×7) |
| activation data (FP4 pack / FP8 bytes) == torch pack | True (×7) |
| activation SFA_lo == `prepack_sf_fp4_a` (ilv) | True (×7) |
| activation SFA_hi == `prepack_sf_fp8_a` (ilv) | True (×7) |

`BITEXACT_PROBE_PASS`. The offline weight SFB and the folded activation SFA are
byte-for-byte the layout the kernel reads.

## 2. Per-Linear + whole-layer cosine

Shapes (Llama-3.2-1B): q/o 2048×2048, k/v 512×2048, gate/up 8192×2048, down 2048×8192.
`cos(vs fakequant)` = MixedQuantLinear vs `quant_utils` NVFP4+MXFP8 fake-quant (the
gate, ≥0.999). `cos(vs fp16)` = NVFP4 4-bit quant drift (informational).

### Synthetic (M=2048) — COSINE_SYNTHETIC_PASS
All 7 Linears: `cos(vs fakequant) = 0.999999`; whole-layer `0.999999`;
`cos(vs fp16) ≈ 0.9952`.

### Real rotated activations (layer 8, M=2048) — COSINE_REAL_PASS
Captured real post-rotation/reorder/Hadamard activations + rotated weights from a
wikitext batch through the ProMix rotated model (hook on each layer-8 quantizer input).

| Linear | cos(vs fakequant) | cos(vs fp16) |
|---|---|---|
| q_proj | 0.999999 | 0.999223 |
| k_proj | 0.999999 | 0.999463 |
| v_proj | 0.999999 | 0.995256 |
| o_proj | 0.999999 | 0.987412 |
| gate_proj | 0.999998 | 0.998311 |
| up_proj | 0.999999 | 0.994125 |
| down_proj | 0.999999 | 0.989744 |
| **whole layer** | **0.999999** | **0.998808** |

All ≥0.999 vs fake-quant (gate met with margin). The `vs fp16` spread (0.987–0.999)
is the genuine NVFP4 4-bit loss, largest on high-variance projections (o_proj,
down_proj), as expected — and is NOT a gate.

## 3. Prefill latency (sum of 7 Linears, GPU 1, unlocked clock)

| M | GEMM-only | FP16 cuBLAS | **GEMM/FP16** | pack (torch-quant) | e2e (pack+GEMM) | fake-quant | e2e/FP16 | e2e/fakequant |
|---|---|---|---|---|---|---|---|---|
| 2048 | 0.513 ms | 0.743 ms | **1.45×** | 16.56 ms | 17.50 ms | 33.91 ms | 0.04× | 1.94× |
| 4096 | 0.713 ms | 1.383 ms | **1.94×** | 31.71 ms | 32.63 ms | 48.42 ms | 0.04× | 1.48× |
| 8192 | 1.205 ms | 2.510 ms | **2.08×** | 65.27 ms | 67.12 ms | 80.67 ms | 0.04× | 1.20× |

**The fused GEMM itself beats FP16 cuBLAS by 1.45×→2.08× (growing with M)** — the real
kernel win, consistent with round-24's co-design results. The e2e pack+GEMM is
dominated (93–98%) by the **activation quantization (float→FP4/FP8 code)** done in
plain PyTorch (`_round_to_nearest_value` broadcast over the 31-value E4M3 grid). This
is the honest per-forward cost as currently implemented; it is NOT the kernel's pack
step. Round-18's folded pack kernel reads *codes* and measures the SF-layout fold at
~µs; realizing the GEMM-only speedup end-to-end needs a **fused float-input quant+pack
CUDA kernel** (a new, non-protected kernel — the clear next step; not built here).

## 4. Honest residuals / boundaries

- **Pack bottleneck:** the torch float→code quantize dominates e2e. GEMM-only is the
  kernel's true contribution; e2e is gated by the missing fused quant+pack kernel.
  Even so, e2e already beats the full fake-quant path (1.2–1.9×) because fake-quant
  pays torch quant on both operands then an FP16 matmul.
- **Dispatcher:** used the production ceiling path `fused_gemm_prepacked` (fixed
  128×128 NV4/MX2), which equals `launch_best_codesign`'s selected config for every
  shape here at M≥2048. Did not separately compile a `DISPATCH_ENABLE_CODESIGN` TU
  (it re-interleaves SF internally and would be redundant/slower at these M). Small-M
  (512, where the dispatcher picks 128×64) is out of this step's prefill scope.
- **Numerics:** MixedQuantLinear is bit-faithful to the spec fake-quant scheme
  (`quant_utils`). Note the kernel's *helper* `nv.quantize_nvfp4` rounds the NVFP4
  block scale to *nearest* E4M3 whereas the spec rounds *up*; using the spec rounding
  (`quantize_nvfp4_spec`) is what lifts cos(vs fakequant) from 0.9954 → 0.999999.
- **Real capture:** built the rotated model from the INT-config rotation artifacts
  (U/R present) with FP quantizer routing forced on (the FP-specific `mxfp8nvfp4`
  basis/rotation artifacts are absent on the box). The high-variance-last rotation
  structure is preserved, so the single-layer isolation is faithful.
- PROTECTED kernel files / mainloop logic unchanged; all new code lives in `promix/`.

## Verdict

First real-inference integration works: 7 Linears of one decoder layer run the real
packed NVFP4+MXFP8 fused kernel with bit-exact SF layout and ≥0.999 cosine vs
fake-quant on both synthetic and real rotated activations. The fused GEMM is
1.45–2.08× faster than FP16 cuBLAS. Remaining work to make the layer end-to-end
faster than FP16: a fused float-input activation quant+pack CUDA kernel to replace the
torch quantize (currently 93–98% of e2e).
