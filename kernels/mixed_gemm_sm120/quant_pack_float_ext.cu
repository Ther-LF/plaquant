// quant_pack_float_ext.cu — FUSED float-input activation quantize + pack + SF-interleave
// (SM120, RTX PRO 6000 Blackwell). ONE CUDA pass: FP16/BF16/FP32 activation ->
//   (1) per-block scale  (NVFP4: block-16 absmax -> E4M3 scale rounded UP;
//                         MXFP8: block-32 absmax -> UE8M0 power-of-2 scale)
//   (2) per-element FP4(E2M1)/FP8(E4M3) code via grid round-to-nearest-even
//   (3) packed codes (2xFP4/byte for NVFP4, 1 byte/FP8) written directly, AND the
//       block SF written in the CUTLASS interleaved atom layout (ILV) the dispatcher
//       (fused_gemm_prepacked / launch_best_codesign) consumes.
//
// This replaces the plain-PyTorch float->code quantize (16-65 ms, 93-98% of e2e in the
// single-layer harness) that dominated end-to-end latency. It is BIT-EXACT (torch.equal)
// to promix.quantize.weight_pack.quantize_nvfp4_spec / quantize_mxfp8_spec followed by
// codesign_fused_ext.cu's quant_pack_fp4/fp8(ilv=True) — same codes, same SF bytes, same
// interleaved layout — so the fused-GEMM output cosine stays 0.999999.
//
// PROTECTED files untouched: this is a NEW standalone extension. It reconstructs the SAME
// Sm1xxBlkScaledConfig SF atom layout as codesign_fused_ext.cu by instantiating the SAME
// stock SM120 block-scaled CollectiveBuilder config (128x128x128 tile), reading only its
// Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA — NO mainloop / forked-kernel code.

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

static inline void check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// ---------------------------------------------------------------------------
// Stock SM120 block-scaled config (mirrors codesign_fused_ext.cu's StockBuilder,
// config-only; we only read Sm1xxBlkScaledConfig for the SFA atom layout).
// ---------------------------------------------------------------------------
template <class ElementAB, int Alignment, int Stages>
struct StockBuilder {
  using ElementD = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
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

using NVbuild = StockBuilder<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 32, 4>;
using MXbuild = StockBuilder<cutlass::mx_float8_t<cutlass::float_e4m3_t>, 16, 2>;
using BSConfigNV = typename NVbuild::CollectiveMainloop::Sm1xxBlkScaledConfig;
using BSConfigMX = typename MXbuild::CollectiveMainloop::Sm1xxBlkScaledConfig;

static inline torch::TensorOptions u8opts(torch::Device dev) {
  return torch::TensorOptions().dtype(torch::kUInt8).device(dev);
}

// ---------------------------------------------------------------------------
// Device grids / helpers.
//   cFP4_val[i]: FP4 E2M1 positive grid {0,.5,1,1.5,2,3,4,6}; code == index.
// The E4M3 scale (NVFP4 round-UP) and E4M3 element bytes (MXFP8) are produced
// with the hardware fp8 conversion intrinsics (O(1), no constant-memory grid
// search) — the 127-entry grid binary search was the pack bottleneck (fp8 path
// at 0.1-0.25 TB/s). Bit-exactness vs quant_utils is re-verified by the probe.
// ---------------------------------------------------------------------------
__constant__ float cFP4_val[8];

static bool g_grids_ready = false;
static void init_grids_once() {
  if (g_grids_ready) return;
  float fp4[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  check_cuda(cudaMemcpyToSymbol(cFP4_val, fp4, sizeof(fp4)), "cpy fp4");
  g_grids_ready = true;
}

// E4M3 byte <-> float (hardware).
__device__ __forceinline__ float e4m3_to_float(unsigned char b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}
// Smallest E4M3 representable >= ideal (round UP). Returns byte + decoded value.
// Matches quant_utils._ceil_to_value over the E4M3 positive grid: hardware RNE
// cast, then bump to the next representable if it landed below ideal.
__device__ __forceinline__ void e4m3_ceil(float ideal, unsigned char& byte, float& val) {
  unsigned char b = (unsigned char)__nv_cvt_float_to_fp8(ideal, __NV_SATFINITE, __NV_E4M3);
  float v = e4m3_to_float(b);
  if (v < ideal && b < 0x7E) { ++b; v = e4m3_to_float(b); }   // 0x7E == 448 (max finite)
  byte = b; val = v;
}

// ---- element in-value conversion (fp16/bf16/fp32 -> float, lossless) ----
__device__ __forceinline__ float to_float(const half v)          { return __half2float(v); }
__device__ __forceinline__ float to_float(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(const float v)         { return v; }

// Replicate quant_utils._round_to_nearest_value over a sorted-ascending grid `v[0..n)`:
// nearest representable to absx with ties-to-even (by grid index), 1e-12 tie tolerance.
__device__ __forceinline__ int rtn_idx(float absx, const float* v, int n) {
  int lo = 0, hi = n;                                               // searchsorted(right=False)
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

// ===================================================================================
// Warp-cooperative fused float->code+pack+interleaved-SF kernels. One WARP per block
// (fp8: 32-lane warp == one 32-elem block; fp4: 32-lane warp == two 16-elem blocks).
// Lanes load one element each (fully coalesced), warp-shuffle reduce the block absmax,
// then each lane quantizes its own element. This keeps occupancy high even for the tiny
// high-variance segments (Kh=256 -> plenty of warps) where the old one-thread-per-block
// design was latency-bound at ~20% occupancy. Bit-exact (max-reduce is order-free; the
// per-element cast is unchanged).
// ===================================================================================

// MXFP8: one warp per 32-element block. `row_stride`/`col_off` let the kernel read a
// column slice of a wider (M, K_full) activation in place (no pre-slice/convert copy):
// element (r, c) of this segment is x[r*row_stride + col_off + c].
template <class TIn, class LayoutSF>
__global__ void qpf8_kernel(uint8_t* __restrict__ data_out, uint8_t* __restrict__ sf_out,
                            const TIn* __restrict__ x, LayoutSF layout,
                            int rows, int nblk, int K, int row_stride, int col_off) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (warp >= rows * nblk) return;
  int r = warp / nblk, b = warp % nblk;
  int base = r * row_stride + col_off + b * 32;
  float f = to_float(x[base + lane]);
  float a = fabsf(f);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  float ideal = a / 448.0f;                                         // / FP8_E4M3_MAX
  float y = fmaxf(ideal, 1.1754944e-38f);                          // net clamp_min 2^-126
  int ex; float m = frexpf(y, &ex);
  int e = (m == 0.5f) ? ex - 1 : ex;                                // ceil(log2(y)) exact
  float scale = ldexpf(1.0f, e);
  if (lane == 0) {
    int sb = e + 127; sb = sb < 0 ? 0 : (sb > 254 ? 254 : sb);
    sf_out[layout(make_coord(r, b * 32, 0))] = (unsigned char)sb;
  }
  float xs = fminf(fmaxf(f / scale, -448.0f), 448.0f);
  data_out[r * K + b * 32 + lane] = (unsigned char)__nv_cvt_float_to_fp8(xs, __NV_SATFINITE, __NV_E4M3);
}

// NVFP4: one thread per 16-element block (already near memory-bound; the warp-coop
// scheme that helps the tiny fp8 segments hurts here due to half-idle lanes +
// single-byte scattered packed writes, so fp4 keeps the simple per-block design).
template <class TIn, class LayoutSF>
__global__ void qpf4_kernel(uint8_t* __restrict__ data_out, uint8_t* __restrict__ sf_out,
                            const TIn* __restrict__ x, LayoutSF layout,
                            int rows, int nblk, int K, int row_stride, int col_off) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * nblk) return;
  int r = idx / nblk, b = idx % nblk;
  int base = r * row_stride + col_off + b * 16;
  float xf[16]; float bmax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) { float f = to_float(x[base + j]); xf[j] = f; bmax = fmaxf(bmax, fabsf(f)); }
  float ideal = fmaxf(bmax / 6.0f, 0.001953125f);                   // / FP4_E2M1_MAX, clamp_min 2^-9
  unsigned char sfb; float scale;
  e4m3_ceil(ideal, sfb, scale);                                     // round UP to E4M3 (hw)
#if defined(PROMIX_ACT_MSE) && (PROMIX_ACT_MSE)
  promix_act_mse_select(xf, sfb, scale);   // round-4 cheap search (act_mse.cuh)
#endif
  sf_out[layout(make_coord(r, b * 16, 0))] = sfb;
  int obase = r * (K / 2) + b * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float xs0 = fminf(fmaxf(xf[2 * j]     / scale, -6.0f), 6.0f);
    float xs1 = fminf(fmaxf(xf[2 * j + 1] / scale, -6.0f), 6.0f);
    int i0 = rtn_idx(fabsf(xs0), cFP4_val, 8);
    int i1 = rtn_idx(fabsf(xs1), cFP4_val, 8);
    unsigned char c0 = (unsigned char)i0; if (xs0 < 0.0f && i0 > 0) c0 |= 0x8;
    unsigned char c1 = (unsigned char)i1; if (xs1 < 0.0f && i1 > 0) c1 |= 0x8;
    data_out[obase + j] = (c0 & 0x0F) | (c1 << 4);                  // even k -> low nibble
  }
}

template <class LayoutSF, int FMT>
static void launch_qpf(torch::Tensor x, uint8_t* data_out, uint8_t* sf_out,
                       LayoutSF layout, int rows, int nblk, int K, int row_stride, int col_off,
                       cudaStream_t stream) {
  int tpb = 256;
  // fp8: one warp per block -> grid over rows*nblk warps. fp4: one thread per block.
  long units = (FMT == 8) ? (long)rows * nblk * 32 : (long)rows * nblk;
  dim3 grid((unsigned)((units + tpb - 1) / tpb));
  auto st = x.scalar_type();
#define LAUNCH(TY, PTR)                                                                 \
  do {                                                                                  \
    if (FMT == 8)                                                                       \
      qpf8_kernel<TY, LayoutSF><<<grid, tpb, 0, stream>>>(data_out, sf_out, PTR, layout, rows, nblk, K, row_stride, col_off); \
    else                                                                                \
      qpf4_kernel<TY, LayoutSF><<<grid, tpb, 0, stream>>>(data_out, sf_out, PTR, layout, rows, nblk, K, row_stride, col_off); \
  } while (0)
  if (st == torch::kFloat32)      LAUNCH(float, x.data_ptr<float>());
  else if (st == torch::kFloat16) LAUNCH(half, reinterpret_cast<const half*>(x.data_ptr<at::Half>()));
  else if (st == torch::kBFloat16)LAUNCH(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
  else TORCH_CHECK(false, "quant_pack_float: x must be float32/float16/bfloat16");
#undef LAUNCH
  check_cuda(cudaGetLastError(), "qpf launch");
}

// ---- entry points: float activation (M,K) -> {packed_data, interleaved_SF} ----
// If row_stride<0: x is the contiguous (M, K) segment. Otherwise x is a wider
// (M, row_stride) activation and this segment is columns [col_off, col_off+K)
// read in place (no pre-slice/convert copy) — the fast path used by MixedQuantLinear.
std::vector<torch::Tensor> quant_pack_float_fp4(torch::Tensor x, int64_t M, int64_t N, int64_t K,
                                                int64_t row_stride, int64_t col_off) {
  TORCH_CHECK(x.is_cuda(), "x CUDA"); TORCH_CHECK(K % 16 == 0, "K%16 (NVFP4 block-16)");
  init_grids_once();
  x = x.contiguous();
  int rs = (row_stride < 0) ? (int)K : (int)row_stride;
  int co = (row_stride < 0) ? 0 : (int)col_off;
  auto dev = x.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  int Mi = (int)M, Ni = (int)N, Ki = (int)K, nblk = Ki / 16;
  auto layout_SFA = BSConfigNV::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Ki, 1));
  auto data_out = torch::empty({(int64_t)Mi * (Ki / 2)}, u8opts(dev));
  auto sf_out = torch::empty({(int64_t)size(filter_zeros(layout_SFA))}, u8opts(dev));
  launch_qpf<decltype(layout_SFA), 4>(x, data_out.data_ptr<uint8_t>(), sf_out.data_ptr<uint8_t>(),
                                      layout_SFA, Mi, nblk, Ki, rs, co, stream);
  data_out = data_out.view({Mi, Ki / 2});
  return {data_out, sf_out};
}

std::vector<torch::Tensor> quant_pack_float_fp8(torch::Tensor x, int64_t M, int64_t N, int64_t K,
                                                int64_t row_stride, int64_t col_off) {
  TORCH_CHECK(x.is_cuda(), "x CUDA"); TORCH_CHECK(K % 32 == 0, "K%32 (MXFP8 block-32)");
  init_grids_once();
  x = x.contiguous();
  int rs = (row_stride < 0) ? (int)K : (int)row_stride;
  int co = (row_stride < 0) ? 0 : (int)col_off;
  auto dev = x.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  int Mi = (int)M, Ni = (int)N, Ki = (int)K, nblk = Ki / 32;
  auto layout_SFA = BSConfigMX::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Ki, 1));
  auto data_out = torch::empty({(int64_t)Mi * Ki}, u8opts(dev));
  auto sf_out = torch::empty({(int64_t)size(filter_zeros(layout_SFA))}, u8opts(dev));
  launch_qpf<decltype(layout_SFA), 8>(x, data_out.data_ptr<uint8_t>(), sf_out.data_ptr<uint8_t>(),
                                      layout_SFA, Mi, nblk, Ki, rs, co, stream);
  data_out = data_out.view({Mi, Ki});
  return {data_out, sf_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_pack_float_fp4", &quant_pack_float_fp4,
        "fused float->NVFP4 quantize+pack+interleaved-SF; returns {packed(M,K/2), physSFA}",
        pybind11::arg("x"), pybind11::arg("M"), pybind11::arg("N"), pybind11::arg("K"),
        pybind11::arg("row_stride") = -1, pybind11::arg("col_off") = 0);
  m.def("quant_pack_float_fp8", &quant_pack_float_fp8,
        "fused float->MXFP8 quantize+bytes+interleaved-SF; returns {bytes(M,K), physSFA}",
        pybind11::arg("x"), pybind11::arg("M"), pybind11::arg("N"), pybind11::arg("K"),
        pybind11::arg("row_stride") = -1, pybind11::arg("col_off") = 0);
}
