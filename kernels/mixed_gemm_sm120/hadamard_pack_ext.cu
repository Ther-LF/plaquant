// hadamard_pack_ext.cu — FUSED SwiGLU + online Hadamard + NVFP4/MXFP8 quantize+pack
// for the ResQ down_proj input path (SM120, RTX PRO 6000 Blackwell).
//
// Replaces, per decoder layer, this four-op chain (per [M, I=8192] intermediate):
//
//   h   = silu(gate) * up                        bf16  [M, I]  -> HBM
//   h32 = h.float()                              fp32  [M, I]  -> HBM
//   y   = fast_hadamard_transform(h32) * 1/sqrt(I)   fp32 [M, I] -> HBM
//   (codes_lo, sfa_lo, codes_hi, sfa_hi) = qpf pack(half(y))
//
// with ONE kernel that reads gate/up once, keeps the fp32 Hadamard tile in
// registers/SMEM only, and writes ONLY the packed NVFP4 (low K) + MXFP8 (high K)
// codes and their CUTLASS-interleaved SFA.  No fp32 [M, I] tensor ever reaches
// HBM, and the 1/sqrt(I) is folded into the transform's final multiply (free).
//
// NUMERICS (hard gate).  Element-wise reference arithmetic (dtype T = the
// activation dtype, fp16 or bf16 — production gate/up GEMM outputs are fp16):
//   g = T load; s = T( g/(1+exp(-g)) );  h = T( s * u )   (two roundings!)
//   y = (sum_j +-x[j]) * float(1/sqrt(I));  q = half(y);  pack(q)
// and q feeds the SAME quantizer device code as quant_pack_float_ext.cu.
// We reproduce it exactly:
//   * silu = g/(1+exp(-g)) in fp32 with correctly-rounded division and the
//     correctly-rounded expf, ROUNDED TO T, then multiplied by u and rounded to
//     T again — this two-rounding order is what `F.silu(g) * u` does on
//     half/bfloat16 tensors.  Probes: bf16 0 mismatches / 30M; fp16 requires
//     accurate expf (the __expf fast path left 4/458752 code diffs, expf: 0);
//   * Hadamard: the reference library's own device functions
//     (`hadamard_mult_thread`, `hadamard_mult_warp`, `exchange_smem_pre`) are
//     copied VERBATIM below (BSD-3, (c) Tri Dao 2023 — see the attribution
//     block) and driven with the same traits (256 threads, kNElts=4,
//     kNChunks=8) and the same register layout, so the stage order, the
//     lane/chunk exchange and every fp32 sum are bit-identical;
//   * the 1/sqrt(I) multiply is the same fp32 `scale` the library folds into
//     store_output (production `fold_had_scale` path), applied before the
//     fp16 round — same as `xf.to(fp16)`;
//   * pack = verbatim qpf device code (thread per NVFP4 block-16, warp per
//     MXFP8 block-32, same E4M3 round-up scale, same RTN grid, same layout).
//
// PROTECTED files untouched: new standalone extension; only reads the stock
// Sm1xxBlkScaledConfig SFA atom layout (same as quant_pack_float_ext.cu).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "act_mse.cuh"

using namespace cute;

static inline void hpk_check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// --- Stock SM120 block-scaled config (config-only; read Sm1xxBlkScaledConfig). ---
template <class ElementAB, int Alignment, int Stages>
struct HpkStockBuilder {
  using ElementD = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using TileShape = Shape<_128,_128,_128>;
  using ClusterShape = Shape<_1,_1,_1>;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementAB, cutlass::layout::RowMajor, Alignment,
      ElementAB, cutlass::layout::ColumnMajor, Alignment,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<Stages>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
};

using HpkNV = HpkStockBuilder<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 32, 4>;
using HpkMX = HpkStockBuilder<cutlass::mx_float8_t<cutlass::float_e4m3_t>, 16, 2>;
using BSConfigNV = typename HpkNV::CollectiveMainloop::Sm1xxBlkScaledConfig;
using BSConfigMX = typename HpkMX::CollectiveMainloop::Sm1xxBlkScaledConfig;

static inline torch::TensorOptions u8opts(torch::Device dev) {
  return torch::TensorOptions().dtype(torch::kUInt8).device(dev);
}

// ====================== device helpers (verbatim from quant_pack_float_ext.cu) ==
__constant__ float cFP4_val_h[8];
static bool g_grids_ready_h = false;
static void init_grids_once_h() {
  if (g_grids_ready_h) return;
  float fp4[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  hpk_check_cuda(cudaMemcpyToSymbol(cFP4_val_h, fp4, sizeof(fp4)), "cpy fp4");
  g_grids_ready_h = true;
}

__device__ __forceinline__ float e4m3_to_float_h(unsigned char b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}
__device__ __forceinline__ void e4m3_ceil_h(float ideal, unsigned char& byte, float& val) {
  unsigned char b = (unsigned char)__nv_cvt_float_to_fp8(ideal, __NV_SATFINITE, __NV_E4M3);
  float v = e4m3_to_float_h(b);
  if (v < ideal && b < 0x7E) { ++b; v = e4m3_to_float_h(b); }
  byte = b; val = v;
}
__device__ __forceinline__ float to_float_h(const half v)          { return __half2float(v); }
__device__ __forceinline__ float to_float_h(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float_h(const float v)         { return v; }
// round a float to the activation dtype (the dtype `F.silu(gate)*up` produces
// on the reference path — the silu output and the product are each rounded to
// this dtype, exactly as PyTorch does elementwise on half/bfloat16 tensors).
__device__ __forceinline__ float round_h(const float v, const half)          { return __half2float(__float2half_rn(v)); }
__device__ __forceinline__ float round_h(const float v, const __nv_bfloat16) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ int rtn_idx_h(float absx, const float* v, int n) {
  int lo = 0, hi = n;
  while (lo < hi) { int mid = (lo + hi) >> 1; if (v[mid] < absx) lo = mid + 1; else hi = mid; }
  int insert = lo; if (insert > n - 1) insert = n - 1; if (insert < 0) insert = 0;
  int idx_hi = insert < 1 ? 1 : (insert > n - 1 ? n - 1 : insert);
  int idx_lo = idx_hi - 1; if (idx_lo < 0) idx_lo = 0;
  float vlo = v[idx_lo], vhi = v[idx_hi];
  float dlo = fabsf(absx - vlo), dhi = fabsf(vhi - absx);
  bool pick_hi = dhi < dlo;
  bool is_tie = fabsf(dhi - dlo) < 1e-12f;
  bool even_lo = (idx_lo & 1) == 0;
  if (is_tie) pick_hi = !even_lo;
  return pick_hi ? idx_hi : idx_lo;
}

// NVFP4 pack of 16 fp32 values (contiguous in the block) -> codes + SFA byte.
// Identical math to quant_pack_float_ext.cu's qpf4_kernel.
template <class LayoutSF>
__device__ __forceinline__ void pack_fp4_block(const float* xf, uint8_t* code_out,
                                               uint8_t* sf_out, LayoutSF layout,
                                               int r, int b, int K_low) {
  float bmax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) bmax = fmaxf(bmax, fabsf(xf[j]));
  float ideal = fmaxf(bmax / 6.0f, 0.001953125f);
  unsigned char sfb; float scale;
  e4m3_ceil_h(ideal, sfb, scale);
#if defined(PROMIX_ACT_MSE) && (PROMIX_ACT_MSE)
  promix_act_mse_select(xf, sfb, scale);   // round-4 cheap search (act_mse.cuh)
#endif
  sf_out[layout(make_coord(r, b * 16, 0))] = sfb;
  int obase = r * (K_low / 2) + b * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float xs0 = fminf(fmaxf(xf[2 * j]     / scale, -6.0f), 6.0f);
    float xs1 = fminf(fmaxf(xf[2 * j + 1] / scale, -6.0f), 6.0f);
    int i0 = rtn_idx_h(fabsf(xs0), cFP4_val_h, 8);
    int i1 = rtn_idx_h(fabsf(xs1), cFP4_val_h, 8);
    unsigned char c0 = (unsigned char)i0; if (xs0 < 0.0f && i0 > 0) c0 |= 0x8;
    unsigned char c1 = (unsigned char)i1; if (xs1 < 0.0f && i1 > 0) c1 |= 0x8;
    code_out[obase + j] = (c0 & 0x0F) | (c1 << 4);
  }
}


// ===================================================================================
// Hadamard core — COPIED VERBATIM from fast_hadamard_transform
// (https://github.com/Dao-AILab/fast-hadamard-transform,
//  csrc/fast_hadamard_transform_common.h, csrc/fast_hadamard_transform_cuda.cu),
// BSD 3-Clause License, Copyright (c) 2022, the respective contributors
// (Tri Dao).  Redistribution permitted with this notice retained.  Only the
// traits are fixed (I=8192 -> 256 threads / 4 elts / 8 chunks) and the two
// callers (`load_input`, `store_output`) are replaced by the fused SwiGLU
// producer and the quantize+pack consumer below; the transform code, its stage
// order and its arithmetic are unchanged.
// ===================================================================================
constexpr int kLog2(int v) { return v > 0 ? 1 + kLog2(v >> 1) : -1; }

// bits carried by one thread's elements (element-bit stages)
template <int kLogN, int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread(float x[kNChunks][1 << kLogN]) {
  constexpr int N = 1 << kLogN;
#pragma unroll
  for (int i = 0; i < kLogN; ++i) {
    const int stride = 1 << i;
#pragma unroll
    for (int j = 0; j < N / 2; ++j) {
      const int lo = j & (stride - 1);
      const int idx = (j - lo) * 2 + lo;
#pragma unroll
      for (int c = 0; c < kNChunks; ++c) {
        const float a = x[c][idx];
        const float b = x[c][idx + stride];
        x[c][idx] = a + b;
        x[c][idx + stride] = a - b;
      }
    }
  }
}

// lane-bit stages
template <int kLogWarpSize, int kStepStart, int kNChunks, int kNItems>
__device__ __forceinline__ void hadamard_mult_warp(float x[kNChunks][kNItems]) {
  constexpr int N = 1 << kLogWarpSize;
  int lane_id = threadIdx.x % N;
#pragma unroll
  for (int step = kStepStart; step < kLogWarpSize; ++step) {
    const int lane_mask = 1 << step;
    const float sign = (lane_id & lane_mask) ? -1.f : 1.f;
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNItems; ++i) {
        float x_val_other = __shfl_xor_sync(0xffffffffu, x[c][i], lane_mask);
        x[c][i] = sign * x[c][i] + x_val_other;
      }
    }
  }
}

// Pre=true: exchange before hadamard_mult_warp; Pre=false: after.
template <int kNChunks, int kChunksPerExchange, int kNElts, int kWarpSize, int kNWarps,
          bool Pre, typename vec_t>
__device__ __forceinline__ void exchange_smem_pre(float x_vals[kNChunks][kNElts], vec_t *smem) {
  constexpr int kNThreads = kWarpSize * kNWarps;
  constexpr int kNExchangePerVec = kNElts / (sizeof(vec_t) / sizeof(float));
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int row_t = threadIdx.x % kNWarps;
  const int col_t = threadIdx.x / kNWarps;
#pragma unroll
  for (int c0 = 0; c0 < kNChunks / kChunksPerExchange; ++c0) {
    __syncthreads();
#pragma unroll
    for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
#pragma unroll
      for (int r = 0; r < kNExchangePerVec; ++r) {
        smem[(c1 * kNExchangePerVec + r) * kNThreads + (Pre ? warp_id * kWarpSize + lane_id ^ warp_id : row_t * kWarpSize + col_t ^ row_t)] = reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r];
      }
    }
    __syncthreads();
#pragma unroll
    for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
#pragma unroll
      for (int r = 0; r < kNExchangePerVec; ++r) {
        reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r] = smem[(c1 * kNExchangePerVec + r) * kNThreads + (Pre ? row_t * kWarpSize + col_t ^ row_t : warp_id * kWarpSize + lane_id ^ warp_id)];
      }
    }
  }
}

// ---- fused-traits constants (I = 8192) ----
constexpr int kHpkThreads = 256;
constexpr int kHpkElts = 4;                       // fp32 elements per exchange vector
constexpr int kHpkLogNElts = 2;
constexpr int kHpkChunks = 8;                     // kNChunks = N / (kNElts * kNThreads)
constexpr int kHpkLogNChunks = 3;
constexpr int kHpkChunksPerExchange = 8;          // 32 KB / (16 B * 256)
constexpr int kHpkNWarps = kHpkThreads / 32;     // COUNT of warps (the exchange wants
                                                 // the count, hadamard_mult_warp the log)
constexpr int kHpkSmemFloats = 8192;              // exchange buffer (32 KB), reused for output

// MXFP8 block-32 pack from 32 values held by one warp (lane j holds q[j]);
// identical math to qpf8_kernel (warp absmax reduce, UE8M0 scale, E4M3 bytes).
template <class LayoutSF>
__device__ __forceinline__ void pack_fp8_block_warp(const float q,
                                                    uint8_t* __restrict__ code,
                                                    uint8_t* __restrict__ sf_out,
                                                    LayoutSF layout, int r, int b,
                                                    int K_high, uint32_t full_mask) {
  const int lane = threadIdx.x & 31;
  float a = fabsf(q);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(full_mask, a, o));
  float ideal = a / 448.0f;
  float yv = fmaxf(ideal, 1.1754944e-38f);
  int ex; float m = frexpf(yv, &ex);
  int e = (m == 0.5f) ? ex - 1 : ex;
  float sc = ldexpf(1.0f, e);
  if (lane == 0) {
    int sb = e + 127; sb = sb < 0 ? 0 : (sb > 254 ? 254 : sb);
    sf_out[layout(make_coord(r, b * 32, 0))] = (unsigned char)sb;
  }
  // NOTE: `sc`/`e` are only defined on lane 0 (the shuffle-reduced absmax lives
  // there); broadcast both so every lane scales with the SAME value.
  sc = __shfl_sync(full_mask, sc, 0);
  float xs = fminf(fmaxf(q / sc, -448.0f), 448.0f);
  code[(long)r * K_high + b * 32 + lane] =
      (unsigned char)__nv_cvt_float_to_fp8(xs, __NV_SATFINITE, __NV_E4M3);
}

// ===================================================================================
// Kernel: one CTA (256 threads) per row.  Registers hold the exact reference
// layout x_vals[c][i] <-> column c*1024 + tid*4 + i (chunk-major, as load_input
// builds it).  Phase 0 fills it with h = bf16(silu(gate)*up); phase 1 runs the
// library's transform unchanged; phase 2 writes q = half(x * scale) back to the
// exchange SMEM in column-linear order, syncs, and packs both the NVFP4 low and
// MXFP8 high segments from column-contiguous reads.
// ===================================================================================
// SWIGLU=false (vllm-perf r2): phase 0 is a plain load of an ALREADY-ACTIVATED
// row x (the vLLM down_proj input is SiluAndMul output) — `up` is ignored. The
// transform + scale + fp16-round + pack phases are byte-identical; numerics
// equal matmul_hadU_cuda(x.float(), None, 1, fold_scale=True).half() -> pack
// (fp32 transform of exact fp16 inputs — the library's fp32-trait stage order).
template <class TIn, class LayoutSF4, class LayoutSF8, int N, bool SWIGLU = true>
__global__ __launch_bounds__(kHpkThreads)
void hadamard_pack_kernel(uint8_t* __restrict__ code_lo, uint8_t* __restrict__ sf_lo,
                          LayoutSF4 layout4,
                          uint8_t* __restrict__ code_hi, uint8_t* __restrict__ sf_hi,
                          LayoutSF8 layout8,
                          const TIn* __restrict__ gate, const TIn* __restrict__ up,
                          int K_low, int K_high, float scale) {
  static_assert(N == 8192, "fused kernel traits are fixed for I = 8192");
  constexpr int kNThreads = kHpkThreads;
  constexpr int kNElts = kHpkElts;
  constexpr int kNChunks = kHpkChunks;
  constexpr int kChunksPerExchange = kHpkChunksPerExchange;
  constexpr int kLogNWarps = 3;
  using vec_t = float4;

  extern __shared__ __align__(16) float smem[];   // kNThreads*kNElts*kNChunks floats

  const int r = blockIdx.x;
  const int tid = threadIdx.x;
  const long rowoff = (long)r * N;

  float x_vals[kNChunks][kNElts];

  // ---- phase 0: h = bf16(silu(gate) * up) into the reference register layout --
  //      (SWIGLU=false: h = the already-activated input row, loaded exactly)
  {
    const TIn* gr = gate + rowoff;
    const TIn* ur = SWIGLU ? up + rowoff : nullptr;
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
      const int base = c * (kNThreads * kNElts) + tid * kNElts;   // 16 B aligned
      const float4 g4 = *reinterpret_cast<const float4*>(gr + base);
      const TIn* gh = reinterpret_cast<const TIn*>(&g4);
      if constexpr (SWIGLU) {
        const float4 u4 = *reinterpret_cast<const float4*>(ur + base);
        const TIn* uh = reinterpret_cast<const TIn*>(&u4);
#pragma unroll
        for (int i = 0; i < kNElts; ++i) {
          const float gf = to_float_h(gh[i]);
          const float uf = to_float_h(uh[i]);
          const float sv = __fdiv_rn(gf, 1.0f + expf(-gf));    // ATen SiLU (fp32)
          const float sb = round_h(sv, TIn{});                   // rounded SiLU output
          x_vals[c][i] = round_h(sb * uf, TIn{});                // rounded product
        }
      } else {
#pragma unroll
        for (int i = 0; i < kNElts; ++i) x_vals[c][i] = to_float_h(gh[i]);
      }
    }
  }

  // ---- phase 1: the reference transform, unchanged ---------------------------
  hadamard_mult_thread<kHpkLogNElts, kNChunks>(x_vals);
  hadamard_mult_warp<5, 0, kNChunks, kNElts>(x_vals);
  exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, 32, kHpkNWarps, true, vec_t>(
      x_vals, reinterpret_cast<vec_t*>(smem));
  hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
  exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, 32, kHpkNWarps, false, vec_t>(
      x_vals, reinterpret_cast<vec_t*>(smem));
  {
    float x_vals_transposed[kNElts][kNChunks];
#pragma unroll
    for (int c = 0; c < kNChunks; ++c)
#pragma unroll
      for (int i = 0; i < kNElts; ++i) x_vals_transposed[i][c] = x_vals[c][i];
    hadamard_mult_thread<kHpkLogNChunks, kNElts>(x_vals_transposed);
#pragma unroll
    for (int c = 0; c < kNChunks; ++c)
#pragma unroll
      for (int i = 0; i < kNElts; ++i) x_vals[c][i] = x_vals_transposed[i][c];
  }

  // ---- phase 2: scale + fp16 round, laid out column-linearly, then pack -------
  __syncthreads();
  // pad-33 addressing: value at linear column `col` lives at smem[col + col/32]
  // (bank-conflict-free, 16 B aligned), covering the FULL row (all N columns).
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    const int col = c * (kNThreads * kNElts) + tid * kNElts;
    float4 v;
    v.x = __half2float(__float2half_rn(x_vals[c][0] * scale));
    v.y = __half2float(__float2half_rn(x_vals[c][1] * scale));
    v.z = __half2float(__float2half_rn(x_vals[c][2] * scale));
    v.w = __half2float(__float2half_rn(x_vals[c][3] * scale));
    // cols here are 4-aligned but the pad makes them 4-strided -> scalar stores
    smem[col + 0 + (col + 0) / 32] = v.x;
    smem[col + 1 + (col + 1) / 32] = v.y;
    smem[col + 2 + (col + 2) / 32] = v.z;
    smem[col + 3 + (col + 3) / 32] = v.w;
  }
  __syncthreads();

  // Pack units (column-linear, identical math/order to quant_pack_float_ext.cu):
  //   unit u in [0, n_lo)  : NVFP4 block 16 at column 16u
  //   unit u in [n_lo, ..) : MXFP8 block 32 at column 32*(u - n_lo)
  // Pack units, collectively covering both segments with this CTA's 256 threads:
  //   NVFP4: n_lo = K_low/16 blocks, TWO per thread (32 contiguous columns each)
  //          -> threads [0, n_lo/2)
  //   MXFP8: n_hi = K_high/32 blocks, ONE per thread (a warp's 32 lanes cover
  //          exactly one block-32) -> the next n_hi threads
  // For the production split (K_low=7168 -> 224, K_high=1024 -> 32) this is a
  // clean 224 + 32 = 256 partition of the CTA.
  // Pack both segments from the linear q row in SMEM (identical math/order to
  // quant_pack_float_ext.cu: per-block absmax -> round-up E4M3 / UE8M0 scale ->
  // per-element grid RTN / E4M3 cast).
  //
  // NVFP4 (n_lo = K_low/16 blocks of 16): NO cross-lane cooperation is needed,
  // so threads [0, n_lo/2) each take 2 blocks (32 contiguous columns).
  // MXFP8  (n_hi = K_high/32 blocks of 32): the warp absmax reduce spans one
  // FULL warp, so a block must be warp-uniform: warp w takes blocks
  // w, w+8, w+16, ... (all 256 threads participate, all lanes of each warp hold
  // the 32 lanes of the same block).
  const int n_lo = K_low / 16;
  const int n_lo_threads = n_lo / 2;
  const int n_hi = (K_high > 0) ? K_high / 32 : 0;
  const int warp = tid >> 5, lane = tid & 31;
  if (tid < n_lo_threads) {
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
      const int b = tid * 2 + blk;
      if (b >= n_lo) break;
      const int c0 = b * 16;
      float xf[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) xf[j] = smem[c0 + j + (c0 + j) / 32];
      pack_fp4_block<LayoutSF4>(xf, code_lo, sf_lo, layout4, r, b, K_low);
    }
  }
  // MXFP8 packing uses ONLY the warps above the NVFP4 partition (threads
  // >= n_lo_threads), so no warp runs both loops; each such warp covers one
  // block-32 per iteration (32 lanes), blocks b = (warp - 7) + 8*k.
  const int hi_warp0 = n_lo_threads / 32;                // 7
  const int hi_warps = kHpkNWarps - hi_warp0;            // 1
  if (warp >= hi_warp0) {
    for (int b = warp - hi_warp0; b < n_hi; b += hi_warps) {
    const int c0 = K_low + b * 32;           // high segment starts at column K_low
    const int col = c0 + lane;
    const float q = smem[col + col / 32];    // linear column -> padded smem index
      pack_fp8_block_warp<LayoutSF8>(q, code_hi, sf_hi, layout8, r, b, K_high, 0xffffffffu);
    }
  }
}

// ===================================================================================
// Host entry points.  N must be 8192 (llama-3.2 intermediate; the kernel is
// templated so other power-of-2 sizes are a one-line instantiation).
// ===================================================================================
template <int N, class TIn, bool SWIGLU, class LayoutSF4, class LayoutSF8>
static void launch_hpk(const TIn* gate, const TIn* up,
                       uint8_t* code_lo, uint8_t* sf_lo, LayoutSF4 layout4,
                       uint8_t* code_hi, uint8_t* sf_hi, LayoutSF8 layout8,
                       int rows, int K_low, int K_high, float scale, cudaStream_t stream) {
  constexpr int kSmemBytes = (N + N / 32 + 32) * (int)sizeof(float);   // 8448 floats @ 8192
  auto kernel = &hadamard_pack_kernel<TIn, LayoutSF4, LayoutSF8, N, SWIGLU>;
  static bool attr_set = false;
  if (!attr_set) {
    hpk_check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kSmemBytes), "hpk smem attr");
    attr_set = true;
  }
  dim3 grid((unsigned)rows, 1, 1);
  kernel<<<grid, 256, kSmemBytes, stream>>>(
      code_lo, sf_lo, layout4, code_hi, sf_hi, layout8, gate, up, K_low, K_high, scale);
}

// fused(gate, up) -> {code_lo(M,K_low/2), sfa_lo, code_hi(M,K_high), sfa_hi}
std::vector<torch::Tensor> hadamard_swiglu_pack_mixed(torch::Tensor gate, torch::Tensor up,
                                                      int64_t M, int64_t N_gemm, int64_t K_low,
                                                      int64_t K_high) {
  TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "gate/up must be CUDA");
  TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gate/up dtype mismatch");

  TORCH_CHECK(K_low % 16 == 0, "K_low % 16");
  TORCH_CHECK(K_high % 32 == 0 || K_high == 0, "K_high % 32");
  TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(), "gate/up must be contiguous");
  init_grids_once_h();
  // N_gemm is the CONSUMER GEMM's N (down_proj's output width): the CUTLASS
  // interleaved SFA atom layout depends on it.  N (the row width) is always 8192.
  int Mi = (int)M, Ni = (int)N_gemm, Klo = (int)K_low, Khi = (int)K_high;
  auto dev = gate.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto layout4 = BSConfigNV::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Klo, 1));
  auto layout8 = BSConfigMX::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Khi > 0 ? Khi : 32, 1));
  auto code_lo_t = torch::empty({(int64_t)Mi * (Klo / 2)}, u8opts(dev));
  auto sf_lo_t   = torch::empty({(int64_t)size(filter_zeros(layout4))}, u8opts(dev));
  auto code_hi_t = torch::empty({(int64_t)Mi * Khi}, u8opts(dev));
  auto sf_hi_t   = torch::empty({(int64_t)size(filter_zeros(layout8))}, u8opts(dev));
  const float scale = 1.0f / std::sqrt(8192.0f);   // 1/sqrt(I), I = row width = 8192
  auto st = gate.scalar_type();
#define LAUNCH_HPK(TY)                                                                   \
  launch_hpk<8192, TY, true>(reinterpret_cast<const TY*>(gate.data_ptr()),               \
                       reinterpret_cast<const TY*>(up.data_ptr()),                       \
                       code_lo_t.data_ptr<uint8_t>(), sf_lo_t.data_ptr<uint8_t>(), layout4, \
                       code_hi_t.data_ptr<uint8_t>(), sf_hi_t.data_ptr<uint8_t>(), layout8, \
                       Mi, Klo, Khi, scale, stream)
  if (st == torch::kFloat16)       LAUNCH_HPK(half);
  else if (st == torch::kBFloat16) LAUNCH_HPK(__nv_bfloat16);
  else TORCH_CHECK(false, "hadamard_swiglu_pack_mixed: gate/up must be float16/bfloat16");
#undef LAUNCH_HPK
  hpk_check_cuda(cudaGetLastError(), "hadamard_pack launch");
  code_lo_t = code_lo_t.view({Mi, Klo / 2});
  code_hi_t = code_hi_t.view({Mi, Khi});
  return {code_lo_t, sf_lo_t, code_hi_t, sf_hi_t};
}

// vllm-perf r2: fused Hadamard + quant/pack of an ALREADY-ACTIVATED row (the
// vLLM down_proj input is SiluAndMul output). Same kernel, SWIGLU=false phase 0.
std::vector<torch::Tensor> hadamard_pack_mixed(torch::Tensor x, int64_t M, int64_t N_gemm,
                                               int64_t K_low, int64_t K_high) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(K_low % 16 == 0, "K_low % 16");
  TORCH_CHECK(K_high % 32 == 0 || K_high == 0, "K_high % 32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(K_low + K_high == 8192, "row width must be 8192");
  init_grids_once_h();
  int Mi = (int)M, Ni = (int)N_gemm, Klo = (int)K_low, Khi = (int)K_high;
  auto dev = x.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto layout4 = BSConfigNV::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Klo, 1));
  auto layout8 = BSConfigMX::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Khi > 0 ? Khi : 32, 1));
  auto code_lo_t = torch::empty({(int64_t)Mi * (Klo / 2)}, u8opts(dev));
  auto sf_lo_t   = torch::empty({(int64_t)size(filter_zeros(layout4))}, u8opts(dev));
  auto code_hi_t = torch::empty({(int64_t)Mi * Khi}, u8opts(dev));
  auto sf_hi_t   = torch::empty({(int64_t)size(filter_zeros(layout8))}, u8opts(dev));
  const float scale = 1.0f / std::sqrt(8192.0f);
  auto st = x.scalar_type();
#define LAUNCH_HPX(TY)                                                                   \
  launch_hpk<8192, TY, false>(reinterpret_cast<const TY*>(x.data_ptr()), nullptr,        \
                       code_lo_t.data_ptr<uint8_t>(), sf_lo_t.data_ptr<uint8_t>(), layout4, \
                       code_hi_t.data_ptr<uint8_t>(), sf_hi_t.data_ptr<uint8_t>(), layout8, \
                       Mi, Klo, Khi, scale, stream)
  if (st == torch::kFloat16)       LAUNCH_HPX(half);
  else if (st == torch::kBFloat16) LAUNCH_HPX(__nv_bfloat16);
  else TORCH_CHECK(false, "hadamard_pack_mixed: x must be float16/bfloat16");
#undef LAUNCH_HPX
  hpk_check_cuda(cudaGetLastError(), "hadamard_pack_x launch");
  code_lo_t = code_lo_t.view({Mi, Klo / 2});
  code_hi_t = code_hi_t.view({Mi, Khi});
  return {code_lo_t, sf_lo_t, code_hi_t, sf_hi_t};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hadamard_swiglu_pack_mixed", &hadamard_swiglu_pack_mixed,
        "FUSED silu(gate)*up -> Hadamard/sqrt(I) (fp32, in SMEM) -> fp16 -> "
        "NVFP4(low)+MXFP8(high) codes + interleaved SFA; "
        "returns {code_lo(M,K_low/2), sfa_lo, code_hi(M,K_high), sfa_hi}; "
        "N_gemm is the consumer GEMM's N (SFA layout), the row width is fixed 8192",
        pybind11::arg("gate"), pybind11::arg("up"), pybind11::arg("M"),
        pybind11::arg("N_gemm"), pybind11::arg("K_low"), pybind11::arg("K_high"));
  m.def("hadamard_pack_mixed", &hadamard_pack_mixed,
        "FUSED Hadamard/sqrt(8192) (fp32, in SMEM) -> fp16 -> NVFP4+MXFP8 pack of an "
        "already-activated row x (vLLM down_proj input); same returns as the SwiGLU entry",
        pybind11::arg("x"), pybind11::arg("M"), pybind11::arg("N_gemm"),
        pybind11::arg("K_low"), pybind11::arg("K_high"));
}
