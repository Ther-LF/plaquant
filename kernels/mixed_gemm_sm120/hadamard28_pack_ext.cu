// hadamard28_pack_ext.cu — FUSED SwiGLU + online Hadamard(28x512) + NVFP4/MXFP8
// quantize+pack for the ResQ down_proj input path when the intermediate width
// I is NOT a power of two (Llama-3.1-8B: I = 14336 = 28 * 512).
//
// The power-of-two fused producer lives in hadamard_pack_ext.cu (I = 8192).  For
// I = 14336 the reference ResQ path is:
//
//   h       = silu(gate) * up                          T=[M,14336]  (two roundings)
//   h32     = h.float()
//   t[g,:]  = fast_hadamard_transform(h32.view(28,512)[g], scale=1/sqrt(14336))  // per group
//   y[g,:]  = sum_{g'} had28[g,g'] * t[g',:]            // torch fp32 matmul, K=28
//   (codes_lo, sfa_lo, codes_hi, sfa_hi) = qpf pack(half(y))
//
// i.e. matmul_hadU_cuda(h.float(), had28, K=28, fold_scale=True).half() then the
// SAME qpf pack as quant_pack_float_ext.cu / hadamard_pack_ext.cu.
//
// BIT-IDENTICAL reproduction (the hard gate; the I=8192 fusion achieved 0 byte
// mismatch and this matches that standard):
//   * SwiGLU: silu = g/(1+exp(-g)) in fp32 (correctly-rounded div + expf),
//     rounded to T, times u, rounded to T again (two roundings) — identical to
//     hadamard_pack_ext.cu.
//   * H_512: the library picks kNThreads=32 for log_N=9 (fast_hadamard_transform
//     _cuda.cu launch table), kNElts=4, kNChunks=4, kNWarps=1 -> NO smem exchange.
//     The decomposition is  hadamard_mult_thread<2,4> ; hadamard_mult_warp<5,0,4,4>
//     ; (no exchange) ; transpose ; hadamard_mult_thread<2,4> ; transpose.  We run
//     exactly that sequence per 512-group inside one warp, with load_input's index
//     map  x_vals[c][i] <-> logical col (c*128 + lane*4 + i)  — so each group's
//     transform is bit-identical to hadamard_transform(.,scale).
//   * scale = 1/sqrt(14336) is applied at the transform store (store_output's
//     out*scale), BEFORE the had28 sum — exactly as HadamardTransform.apply(.,scale)
//     then had28 @ (the fold order matters for fp32 rounding).
//   * had28 sum accumulates g'=0..27 ASCENDING (verified: torch had28@x is
//     byte-identical to the ascending sequential sum; had28 entries are +-1, so
//     fma(+-1,x,acc)==acc+-x is one rounding == mul-add, only the order matters).
//   * pack = VERBATIM qpf device code (pack_fp4_block / pack_fp8_block_warp),
//     identical to hadamard_pack_ext.cu, so the quantizer half is already proven
//     byte-exact for this layout.
//
// PROTECTED files untouched; this is a NEW standalone extension that only reads
// the stock Sm1xxBlkScaledConfig SFA atom layout (same as quant_pack_float_ext.cu).

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

using namespace cute;

static inline void h28_check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// --- Stock SM120 block-scaled config (config-only; read Sm1xxBlkScaledConfig). ---
template <class ElementAB, int Alignment, int Stages>
struct H28StockBuilder {
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

using H28NV = H28StockBuilder<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 32, 4>;
using H28MX = H28StockBuilder<cutlass::mx_float8_t<cutlass::float_e4m3_t>, 16, 2>;
using BSConfigNV = typename H28NV::CollectiveMainloop::Sm1xxBlkScaledConfig;
using BSConfigMX = typename H28MX::CollectiveMainloop::Sm1xxBlkScaledConfig;

static inline torch::TensorOptions u8opts(torch::Device dev) {
  return torch::TensorOptions().dtype(torch::kUInt8).device(dev);
}

// ====================== device helpers (verbatim from quant_pack_float_ext.cu) ==
__constant__ float cFP4_val28[8];
__constant__ float cHad28[28 * 28];     // had28 matrix (+-1), row-major [g][g']
static bool g_grids_ready_28 = false;
static void init_grids_once_28(const float* had28_host) {
  float fp4[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  h28_check_cuda(cudaMemcpyToSymbol(cFP4_val28, fp4, sizeof(fp4)), "cpy fp4");
  h28_check_cuda(cudaMemcpyToSymbol(cHad28, had28_host, sizeof(float) * 28 * 28), "cpy had28");
  g_grids_ready_28 = true;
}

__device__ __forceinline__ float e4m3_to_float_28(unsigned char b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}
__device__ __forceinline__ void e4m3_ceil_28(float ideal, unsigned char& byte, float& val) {
  unsigned char b = (unsigned char)__nv_cvt_float_to_fp8(ideal, __NV_SATFINITE, __NV_E4M3);
  float v = e4m3_to_float_28(b);
  if (v < ideal && b < 0x7E) { ++b; v = e4m3_to_float_28(b); }
  byte = b; val = v;
}
__device__ __forceinline__ float to_float_28(const half v)          { return __half2float(v); }
__device__ __forceinline__ float to_float_28(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float round_28(const float v, const half)          { return __half2float(__float2half_rn(v)); }
__device__ __forceinline__ float round_28(const float v, const __nv_bfloat16) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ int rtn_idx_28(float absx, const float* v, int n) {
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

// NVFP4 pack of 16 fp32 values (already fp16-rounded) -> codes + SFA byte.
template <class LayoutSF>
__device__ __forceinline__ void pack_fp4_block28(const float* xf, uint8_t* code_out,
                                                 uint8_t* sf_out, LayoutSF layout,
                                                 int r, int b, int K_low) {
  float bmax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) bmax = fmaxf(bmax, fabsf(xf[j]));
  float ideal = fmaxf(bmax / 6.0f, 0.001953125f);
  unsigned char sfb; float scale;
  e4m3_ceil_28(ideal, sfb, scale);
  sf_out[layout(make_coord(r, b * 16, 0))] = sfb;
  int obase = r * (K_low / 2) + b * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float xs0 = fminf(fmaxf(xf[2 * j]     / scale, -6.0f), 6.0f);
    float xs1 = fminf(fmaxf(xf[2 * j + 1] / scale, -6.0f), 6.0f);
    int i0 = rtn_idx_28(fabsf(xs0), cFP4_val28, 8);
    int i1 = rtn_idx_28(fabsf(xs1), cFP4_val28, 8);
    unsigned char c0 = (unsigned char)i0; if (xs0 < 0.0f && i0 > 0) c0 |= 0x8;
    unsigned char c1 = (unsigned char)i1; if (xs1 < 0.0f && i1 > 0) c1 |= 0x8;
    code_out[obase + j] = (c0 & 0x0F) | (c1 << 4);
  }
}

// MXFP8 block-32 pack from 32 values held by one warp (lane j holds q[j]).
template <class LayoutSF>
__device__ __forceinline__ void pack_fp8_block_warp28(const float q,
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
  sc = __shfl_sync(full_mask, sc, 0);
  float xs = fminf(fmaxf(q / sc, -448.0f), 448.0f);
  code[(long)r * K_high + b * 32 + lane] =
      (unsigned char)__nv_cvt_float_to_fp8(xs, __NV_SATFINITE, __NV_E4M3);
}

// ===================================================================================
// Hadamard core — COPIED VERBATIM from fast_hadamard_transform (BSD-3, Tri Dao 2023).
// Only the two callers are replaced; the butterfly arithmetic is unchanged.
// ===================================================================================
template <int kLogN, int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_28(float x[kNChunks][1 << kLogN]) {
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

template <int kLogWarpSize, int kStepStart, int kNChunks, int kNItems>
__device__ __forceinline__ void hadamard_mult_warp_28(float x[kNChunks][kNItems]) {
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

// ---- fused-traits constants ----
constexpr int kH28Threads = 256;        // 8 warps (256 beat 512: fewer threads/CTA
                                         // -> better ILP at the forced 1 CTA/SM)
constexpr int kH28Warps   = 8;
constexpr int kGroups     = 28;         // 14336 / 512
constexpr int kGroupN     = 512;        // power-of-two sub-transform size
constexpr int kTsStride   = 513;        // pad the per-group stride (bank-conflict-free had28 read)

// Apply the library's N=512 transform (log_N=9 -> 32 threads, kNElts=4, kNChunks=4,
// kNWarps=1) to one 512-group whose SwiGLU producer output is already in x_vals,
// laid out as x_vals[c][i] <-> logical column (c*128 + lane*4 + i).  Bit-identical
// to fast_hadamard_transform(., scale) for dim=512.
__device__ __forceinline__ void transform_512(float x_vals[4][4]) {
  hadamard_mult_thread_28<2, 4>(x_vals);            // kLogNElts=2, kNChunks=4
  hadamard_mult_warp_28<5, 0, 4, 4>(x_vals);        // kLogWarpSize=5
  // kNWarps == 1 -> no smem exchange
  float xt[4][4];                                   // transpose [chunk][elt] -> [elt][chunk]
#pragma unroll
  for (int c = 0; c < 4; ++c)
#pragma unroll
    for (int i = 0; i < 4; ++i) xt[i][c] = x_vals[c][i];
  hadamard_mult_thread_28<2, 4>(xt);                // kLogNChunks=2, kNElts=4
#pragma unroll
  for (int c = 0; c < 4; ++c)
#pragma unroll
    for (int i = 0; i < 4; ++i) x_vals[c][i] = xt[i][c];
}

// ===================================================================================
// Kernel: one CTA (256 threads / 8 warps) per row r of the [M, 14336] intermediate.
//   Phase 0+1: each warp transforms groups g = warp, warp+8, warp+16, warp+24 (<28):
//     SwiGLU producer -> H_512 -> * scale, written to smem ts[g*kTsStride + col].
//   Phase 2: had28 (ascending g'=0..27) + fp16 round + pack, reading ts from smem.
//     NVFP4: thread owns 16-col blocks (within one group -> one had28 row g).
//     MXFP8: warp owns 32-col blocks (within one group).
//   (A 512-thread + fp16-reshuffle + coalesced-had28 variant was tried and was ~2x
//   SLOWER — at the forced 1 CTA/SM the extra threads/sync/second-pass hurt ILP.)
// ===================================================================================
template <class TIn, class LayoutSF4, class LayoutSF8>
__global__ __launch_bounds__(kH28Threads)
void hadamard28_pack_kernel(uint8_t* __restrict__ code_lo, uint8_t* __restrict__ sf_lo,
                            LayoutSF4 layout4,
                            uint8_t* __restrict__ code_hi, uint8_t* __restrict__ sf_hi,
                            LayoutSF8 layout8,
                            const TIn* __restrict__ gate, const TIn* __restrict__ up,
                            int K_low, int K_high, float scale) {
  constexpr int I = kGroups * kGroupN;              // 14336
  extern __shared__ __align__(16) float ts[];       // kGroups * kTsStride floats

  const int r = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const long rowoff = (long)r * I;

  // ---- phase 0+1: SwiGLU + H_512 + scale, per group, into smem ----------------
  const TIn* gr = gate + rowoff;
  const TIn* ur = up + rowoff;
#pragma unroll
  for (int gi = 0; gi < 4; ++gi) {
    const int g = warp + gi * kH28Warps;
    if (g >= kGroups) break;
    const int gbase = g * kGroupN;                  // logical column base of this group
    float x_vals[4][4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int col = c * 128 + lane * 4;           // logical col within the 512-group
      const int goff = gbase + col;                 // global column in the 14336 row
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float gf = to_float_28(gr[goff + i]);
        const float uf = to_float_28(ur[goff + i]);
        const float sv = __fdiv_rn(gf, 1.0f + expf(-gf));  // ATen SiLU (fp32)
        const float sb = round_28(sv, TIn{});               // rounded SiLU output
        x_vals[c][i] = round_28(sb * uf, TIn{});            // rounded product
      }
    }
    transform_512(x_vals);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int col = c * 128 + lane * 4;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ts[g * kTsStride + col + i] = x_vals[c][i] * scale;  // scale folded BEFORE had28
    }
  }
  __syncthreads();

  // ---- phase 2a: NVFP4 (low) — had28 (ascending) + fp16 + pack ----------------
  const int n_lo = K_low / 16;                      // number of NVFP4 blocks
  const int n_hi = (K_high > 0) ? K_high / 32 : 0;  // number of MXFP8 blocks
  for (int b = tid; b < n_lo; b += kH28Threads) {
    const int c0 = b * 16;                          // first column of the block
    const int g  = c0 / kGroupN;                    // output group (16 | 512 -> no crossing)
    const int j0 = c0 - g * kGroupN;                // position within the group
    const float* had_g = cHad28 + g * 28;
    float xf[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      const int j = j0 + t;
      float acc = 0.0f;
#pragma unroll
      for (int gp = 0; gp < 28; ++gp)               // ASCENDING, matches torch had28@x
        acc += had_g[gp] * ts[gp * kTsStride + j];
      xf[t] = __half2float(__float2half_rn(acc));   // fp16 round (== y.half())
    }
    pack_fp4_block28<LayoutSF4>(xf, code_lo, sf_lo, layout4, r, b, K_low);
  }

  // ---- phase 2b: MXFP8 (high) — had28 (ascending) + fp16 + pack ---------------
  // Warp owns 32-col blocks: block b in [0,n_hi) at columns [K_low+32b, +32).
  for (int b = warp; b < n_hi; b += kH28Warps) {
    const int c0 = K_low + b * 32;                  // global column base of the block
    const int col = c0 + lane;
    const int g = col / kGroupN;                    // output group (32 | 512, K_low%32==0)
    const int j = col - g * kGroupN;
    const float* had_g = cHad28 + g * 28;
    float acc = 0.0f;
#pragma unroll
    for (int gp = 0; gp < 28; ++gp)                 // ASCENDING
      acc += had_g[gp] * ts[gp * kTsStride + j];
    const float q = __half2float(__float2half_rn(acc));
    pack_fp8_block_warp28<LayoutSF8>(q, code_hi, sf_hi, layout8, r, b, K_high, 0xffffffffu);
  }
}

// ===================================================================================
// Host entry.  I is fixed at 14336 (= 28 * 512); K_low % 16 == 0, K_high % 32 == 0,
// K_low % 32 == 0 (so MXFP8 32-blocks never straddle a 512-group boundary).
// ===================================================================================
std::vector<torch::Tensor> hadamard28_swiglu_pack_mixed(torch::Tensor gate, torch::Tensor up,
                                                        torch::Tensor had28,
                                                        int64_t M, int64_t N_gemm, int64_t K_low,
                                                        int64_t K_high) {
  TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "gate/up must be CUDA");
  TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gate/up dtype mismatch");
  TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(), "gate/up must be contiguous");
  TORCH_CHECK(K_low + K_high == kGroups * kGroupN, "K_low+K_high must be 14336");
  TORCH_CHECK(K_low % 32 == 0, "K_low % 32 (MXFP8 block alignment vs 512-group)");
  TORCH_CHECK(K_high % 32 == 0 || K_high == 0, "K_high % 32");
  TORCH_CHECK(had28.is_cuda() == false || had28.is_cpu(), "had28 provided on cpu");

  auto had28_cpu = had28.to(torch::kCPU).to(torch::kFloat32).contiguous();
  TORCH_CHECK(had28_cpu.numel() == 28 * 28, "had28 must be 28x28");
  init_grids_once_28(had28_cpu.data_ptr<float>());

  int Mi = (int)M, Ni = (int)N_gemm, Klo = (int)K_low, Khi = (int)K_high;
  auto dev = gate.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto layout4 = BSConfigNV::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Klo, 1));
  auto layout8 = BSConfigMX::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Khi > 0 ? Khi : 32, 1));
  auto code_lo_t = torch::empty({(int64_t)Mi * (Klo / 2)}, u8opts(dev));
  auto sf_lo_t   = torch::empty({(int64_t)size(filter_zeros(layout4))}, u8opts(dev));
  auto code_hi_t = torch::empty({(int64_t)Mi * Khi}, u8opts(dev));
  auto sf_hi_t   = torch::empty({(int64_t)size(filter_zeros(layout8))}, u8opts(dev));
  const float scale = 1.0f / std::sqrt((float)(kGroups * kGroupN));   // 1/sqrt(14336)

  constexpr int kSmemBytes = kGroups * kTsStride * (int)sizeof(float);  // ~56.1 KB
  auto st = gate.scalar_type();
#define LAUNCH_H28(TY)                                                                      \
  do {                                                                                      \
    auto kernel = &hadamard28_pack_kernel<TY, decltype(layout4), decltype(layout8)>;        \
    static bool attr_set = false;                                                           \
    if (!attr_set) {                                                                        \
      h28_check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                          kSmemBytes), "h28 smem attr");                     \
      attr_set = true;                                                                       \
    }                                                                                        \
    dim3 grid((unsigned)Mi, 1, 1);                                                           \
    kernel<<<grid, kH28Threads, kSmemBytes, stream>>>(                                       \
        code_lo_t.data_ptr<uint8_t>(), sf_lo_t.data_ptr<uint8_t>(), layout4,                 \
        code_hi_t.data_ptr<uint8_t>(), sf_hi_t.data_ptr<uint8_t>(), layout8,                 \
        reinterpret_cast<const TY*>(gate.data_ptr()),                                        \
        reinterpret_cast<const TY*>(up.data_ptr()), Klo, Khi, scale);                        \
  } while (0)
  if (st == torch::kFloat16)       LAUNCH_H28(half);
  else if (st == torch::kBFloat16) LAUNCH_H28(__nv_bfloat16);
  else TORCH_CHECK(false, "hadamard28_swiglu_pack_mixed: gate/up must be float16/bfloat16");
#undef LAUNCH_H28
  h28_check_cuda(cudaGetLastError(), "hadamard28_pack launch");
  code_lo_t = code_lo_t.view({Mi, Klo / 2});
  code_hi_t = code_hi_t.view({Mi, Khi});
  return {code_lo_t, sf_lo_t, code_hi_t, sf_hi_t};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hadamard28_swiglu_pack_mixed", &hadamard28_swiglu_pack_mixed,
        "FUSED silu(gate)*up -> Hadamard(28x512)/sqrt(14336) (fp32, SMEM) -> fp16 -> "
        "NVFP4(low)+MXFP8(high) codes + interleaved SFA; I fixed 14336; "
        "had28 is the 28x28 +-1 matrix (same as hadamard_data.json)",
        pybind11::arg("gate"), pybind11::arg("up"), pybind11::arg("had28"),
        pybind11::arg("M"), pybind11::arg("N_gemm"), pybind11::arg("K_low"),
        pybind11::arg("K_high"));
}
