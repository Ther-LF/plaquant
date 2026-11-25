# promix_dualphase_gemm

One fused GEMM over a weight matrix whose contraction (K) axis is split into two
block-scaled segments: a **low segment** quantized to NVFP4 (FP4 E2M1 elements,
one FP8 E4M3 scale per 16 elements) and a **high segment** quantized to MXFP8
(FP8 E4M3 elements, one E8M0 scale per 32 elements). The output is the sum of
the two segment GEMMs plus a bias, in float16. This is the prefill-path linear
layer of a mixed-precision (NVFP4+MXFP8) quantized LLM.

## Target hardware

NVIDIA GB10 (DGX Spark), compute capability **12.1** — a Blackwell-class
consumer/embedded part: 48 SMs, ~100 KB usable shared memory per block
opt-in, unified LPDDR5x memory (~273 GB/s), 4th/5th-gen tensor cores with
block-scaled MMA instructions (`mma.sync.*.block_scale` for
e2m1/e4m3 with ue4m3/ue8m0 scale factors, as exposed by CUTLASS SM120
block-scaled collectives). There is **no TMA-free tcgen05 path here**: SM 12.x
uses the SM120-style `mma.sync` block-scaled instructions, not SM100 `tcgen05`.
A CUDA kernel file must name the arch explicitly, e.g. first line
`// fk-cuda-cflags: -arch=sm_121a` (the build default would otherwise be
sm_100a, which does not run on this GPU).

## Signature

Inputs, in order (all CUDA tensors; axes `m`, `n`, `k_lo`, `k_hi` come from the
shape; `k_lo % 128 == 0`, `k_hi % 32 == 0`; the shape also carries the derived
axes `k_lo_b = k_lo/2`, `k_lo_sf = k_lo/16`, `k_hi_sf = k_hi/32` so packed
tensor shapes can be written directly against axis names):

- `a_lo`: `[m, k_lo_b]`, uint8 — activation low-segment NVFP4 codes, two FP4
  codes per byte: element `2*j` of the row is the **low nibble** of byte `j`,
  element `2*j+1` the high nibble.
- `sfa_lo`: `[m, k_lo_sf]`, uint8 — activation low-segment block scales,
  row-major: byte `[i, b]` is the FP8 E4M3 encoding of the scale of elements
  `16*b .. 16*b+15` of row `i`.
- `a_hi`: `[m, k_hi]`, uint8 — activation high-segment FP8 E4M3 element bytes.
- `sfa_hi`: `[m, k_hi_sf]`, uint8 — activation high-segment block scales,
  row-major: byte `[i, b]` is an E8M0 code (value `2**(byte - 127)`) scaling
  elements `32*b .. 32*b+31` of row `i`.
- `b_lo`: `[n, k_lo_b]`, uint8 — weight low-segment NVFP4 codes, same packing
  as `a_lo` (row r holds output channel r; the contraction runs along the row).
- `sfb_lo`: `[n, k_lo_sf]`, uint8 — weight low-segment E4M3 block scales,
  row-major like `sfa_lo`.
- `b_hi`: `[n, k_hi]`, uint8 — weight high-segment FP8 E4M3 element bytes.
- `sfb_hi`: `[n, k_hi_sf]`, uint8 — weight high-segment E8M0 block scales.
- `bias`: `[n]`, float16.

Output:

- `y`: `[m, n]`, float16.

Byte tensors (`a_lo`, `sfa_lo`, `a_hi`, `sfa_hi`, `b_lo`, `sfb_lo`, `b_hi`,
`sfb_hi`) may be delivered as int8 carrying the identical bit pattern (an
environment without a uint8 dtype); consumers must reinterpret the bits as
uint8 (`tensor.view(torch.uint8)`), never value-convert.

## Semantics

Decode rules:

- FP4 E2M1 code (one nibble): bits 0-2 index the magnitude table
  `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`, bit 3 is the sign (set = negative).
- FP8 E4M3: the standard OCP E4M3 format (what
  `tensor.view(torch.float8_e4m3fn).float()` decodes), exponent bias 7,
  byte `0x7F`/`0xFF` = NaN (never present in legal inputs).
- E8M0: `2.0 ** (int(byte) - 127)`.

Dequantized operands (conceptually float32):

```
A_lo[i, k] = fp4(a_lo nibble (i,k)) * e4m3(sfa_lo[i, k//16])     # [m, k_lo]
A_hi[i, k] = e4m3(a_hi[i, k])       * e8m0(sfa_hi[i, k//32])     # [m, k_hi]
B_lo[r, k] = fp4(b_lo nibble (r,k)) * e4m3(sfb_lo[r, k//16])     # [n, k_lo]
B_hi[r, k] = e4m3(b_hi[r, k])       * e8m0(sfb_hi[r, k//32])     # [n, k_hi]
```

Result:

```
acc[i, r] = sum_k A_lo[i, k] * B_lo[r, k]  +  sum_k A_hi[i, k] * B_hi[r, k]
y[i, r]   = float16( float32(acc[i, r]) + bias[r] )
```

The two segment products accumulate in float32 (tensor-core scaled-MMA
accumulation); the exact inner accumulation/rounding order is free — correctness
is judged within a tolerance calibrated on this freedom. (The timed
implementation rounds `acc` through bfloat16 before the bias add; that is
within the tolerance and is NOT part of the required semantics.)

## Legal inputs

- `a_lo` / `b_lo`: any bytes (all 256 values decode to legal FP4 pairs).
- `a_hi` / `b_hi`: any E4M3 byte except the NaN encodings `0x7F` and `0xFF`.
- `sfa_lo` / `sfb_lo`: E4M3 bytes of **positive, finite** scales in
  `[2**-6, 0.5]` (i.e. bytes `0x08 .. 0x30`, no sign bit, no NaN).
- `sfa_hi` / `sfb_hi`: bytes in `[103, 119]` (scales `2**-24 .. 2**-8`).
- `bias`: finite float16, magnitude <= 8.
- Finiteness: within these ranges every output element stays far inside the
  float16 finite range even for adversarial inputs (worst-case |dot| is a few
  times 1e4); generators must stay inside them.
- Representative data: quantize Gaussian activations/weights (std ~1 for A,
  ~0.08 for B) with per-block absmax scaling into the formats above; extreme
  groups may use all-max-magnitude codes with small scales, or sparse rows.
  Outputs must stay finite in float16 for legal inputs.

## Notes for implementations

- `m` is large (8192) and fixed per shape; `n = 8192`; `k = k_lo + k_hi` is
  **small** (2048-3072). The contraction is short: per 128x128 output tile
  there are only `k/128 = 16..24` K-iterations, so pipeline fill/drain and
  per-tile prologue overheads are first-order costs; the whole B matrix
  (~8-12 MB packed) fits in a fraction of L2.
- The kernel must be ONE launch for the whole result (both segments fused);
  a single kernel entry called per the signature above.
- Scale-factor tensors are given row-major. In the real pipeline the
  scale-factor layout is produced upstream for free (the producer kernel can
  store scales in any layout at no extra cost), so an implementation may
  derive a rearranged/interleaved copy of the **scale-factor tensors only**
  (never the code/element tensors) on first call and cache it across calls
  on the same inputs; the timed steady state should not re-derive it.
