// quant_pack_fused_ext.cu — LEVER 1 (pack fusion): SINGLE-LAUNCH fused float->codes
// activation pack for the co-design dual-format GEMM. One CUDA launch produces BOTH
// the NVFP4 (low 7/8) AND the MXFP8 (high 1/8) packed activation segments + their
// CUTLASS-interleaved SFA, instead of the two separate quant_pack_float_fp4 /
// quant_pack_float_fp8 launches currently issued per Linear in MixedQuantLinear.
//
// WHY: the per-Linear activation pack is two kernel launches (224 launches / forward
// over 112 Linears). Each reads a disjoint column slice of the SAME fp16 activation
// and writes its own codes+SF. Fusing them into one launch (grid.z selects segment)
// halves the launch count and the kernel-tail/scheduling gaps between them. It is
// BIT-EXACT by construction: the per-block absmax reduce, scale, and per-element cast
// are the IDENTICAL device code as quant_pack_float_ext.cu (duplicated verbatim below
// — NOT the protected mainloop), so the codes + SF bytes + interleaved layout are the
// same bytes the dispatcher (fused_gemm_prepacked) already consumes.
//
// NOTE: a TRUE pack-into-GEMM-prologue fusion (quantize fp16 A inside the GEMM mainloop
// as it loads A into smem) is NOT feasible here: the SM120 block-scaled mainloop
// (codesign_fused_kernel.hpp / codesign_sm120_bs_mma.hpp, PROTECTED) uses block-scaled
// TMA that loads PRE-QUANTIZED FP4/FP8 codes + SF from gmem — there is no HW path to
// quantize-in-prologue without abandoning the block-scaled TMA load. The validated
// industry pattern (flashinfer PR-2233 "Fused RMSNorm + FP4 Quantization", sm120;
// sglang PR-17449 mxfp8 online quant) fuses quant into the PRODUCER (RMSNorm) epilogue,
// not the GEMM consumer prologue. This single-launch fusion is the in-scope intermediate.
//
// PROTECTED files untouched: NEW standalone extension; only reads Sm1xxBlkScaledConfig
// for the SFA atom layout, same as quant_pack_float_ext.cu.

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

static inline void qpfm_check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// --- Stock SM120 block-scaled config (config-only; read Sm1xxBlkScaledConfig). ---
template <class ElementAB, int Alignment, int Stages>
struct QpfmStockBuilder {
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

using QpfmNV = QpfmStockBuilder<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 32, 4>;
using QpfmMX = QpfmStockBuilder<cutlass::mx_float8_t<cutlass::float_e4m3_t>, 16, 2>;
using BSConfigNV = typename QpfmNV::CollectiveMainloop::Sm1xxBlkScaledConfig;
using BSConfigMX = typename QpfmMX::CollectiveMainloop::Sm1xxBlkScaledConfig;

static inline torch::TensorOptions u8opts(torch::Device dev) {
  return torch::TensorOptions().dtype(torch::kUInt8).device(dev);
}

// ====================== device helpers (verbatim from quant_pack_float_ext.cu) ======
__constant__ float cFP4_val_m[8];
static bool g_grids_ready_m = false;
static void init_grids_once_m() {
  if (g_grids_ready_m) return;
  float fp4[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  qpfm_check_cuda(cudaMemcpyToSymbol(cFP4_val_m, fp4, sizeof(fp4)), "cpy fp4");
  g_grids_ready_m = true;
}

__device__ __forceinline__ float e4m3_to_float_m(unsigned char b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}
__device__ __forceinline__ void e4m3_ceil_m(float ideal, unsigned char& byte, float& val) {
  unsigned char b = (unsigned char)__nv_cvt_float_to_fp8(ideal, __NV_SATFINITE, __NV_E4M3);
  float v = e4m3_to_float_m(b);
  if (v < ideal && b < 0x7E) { ++b; v = e4m3_to_float_m(b); }
  byte = b; val = v;
}
__device__ __forceinline__ float to_float_m(const half v)          { return __half2float(v); }
__device__ __forceinline__ float to_float_m(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float_m(const float v)         { return v; }

__device__ __forceinline__ int rtn_idx_m(float absx, const float* v, int n) {
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

// ====================== fused single-launch kernel ===================================
// grid.z == 0 -> NVFP4 low segment (thread per 16-block, cols [0, K_low));
// grid.z == 1 -> MXFP8 high segment (warp per 32-block, cols [K_low, K_low+K_high)).
// Both read the SAME wider (M, K) activation x in place via row_stride=K.
template <class TIn, class LayoutSF4, class LayoutSF8>
__global__ void qpf_mixed_kernel(
    uint8_t* __restrict__ data_lo, uint8_t* __restrict__ sf_lo, LayoutSF4 layout4,
    uint8_t* __restrict__ data_hi, uint8_t* __restrict__ sf_hi, LayoutSF8 layout8,
    const TIn* __restrict__ x, int rows, int K,
    int K_low, int K_high, int nblk_lo, int nblk_hi) {
  if (blockIdx.z == 0) {
    // ---- NVFP4 low: one thread per 16-element block ----
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * nblk_lo) return;
    int r = idx / nblk_lo, b = idx % nblk_lo;
    int base = r * K + b * 16;                       // col_off = 0
    float xf[16]; float bmax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) { float f = to_float_m(x[base + j]); xf[j] = f; bmax = fmaxf(bmax, fabsf(f)); }
    float ideal = fmaxf(bmax / 6.0f, 0.001953125f);
    unsigned char sfb; float scale;
    e4m3_ceil_m(ideal, sfb, scale);
#if defined(PROMIX_ACT_MSE) && (PROMIX_ACT_MSE)
    promix_act_mse_select(xf, sfb, scale);   // round-4 cheap search (act_mse.cuh)
#endif
    sf_lo[layout4(make_coord(r, b * 16, 0))] = sfb;
    int obase = r * (K_low / 2) + b * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float xs0 = fminf(fmaxf(xf[2 * j]     / scale, -6.0f), 6.0f);
      float xs1 = fminf(fmaxf(xf[2 * j + 1] / scale, -6.0f), 6.0f);
      int i0 = rtn_idx_m(fabsf(xs0), cFP4_val_m, 8);
      int i1 = rtn_idx_m(fabsf(xs1), cFP4_val_m, 8);
      unsigned char c0 = (unsigned char)i0; if (xs0 < 0.0f && i0 > 0) c0 |= 0x8;
      unsigned char c1 = (unsigned char)i1; if (xs1 < 0.0f && i1 > 0) c1 |= 0x8;
      data_lo[obase + j] = (c0 & 0x0F) | (c1 << 4);
    }
  } else {
    // ---- MXFP8 high: one warp per 32-element block ----
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= rows * nblk_hi) return;
    int r = warp / nblk_hi, b = warp % nblk_hi;
    int base = r * K + K_low + b * 32;               // col_off = K_low
    float f = to_float_m(x[base + lane]);
    float a = fabsf(f);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float ideal = a / 448.0f;
    float y = fmaxf(ideal, 1.1754944e-38f);
    int ex; float m = frexpf(y, &ex);
    int e = (m == 0.5f) ? ex - 1 : ex;
    float scale = ldexpf(1.0f, e);
    if (lane == 0) {
      int sb = e + 127; sb = sb < 0 ? 0 : (sb > 254 ? 254 : sb);
      sf_hi[layout8(make_coord(r, b * 32, 0))] = (unsigned char)sb;
    }
    float xs = fminf(fmaxf(f / scale, -448.0f), 448.0f);
    data_hi[r * K_high + b * 32 + lane] = (unsigned char)__nv_cvt_float_to_fp8(xs, __NV_SATFINITE, __NV_E4M3);
  }
}

template <class LayoutSF4, class LayoutSF8>
static void launch_qpf_mixed(torch::Tensor x,
                             uint8_t* data_lo, uint8_t* sf_lo, LayoutSF4 layout4,
                             uint8_t* data_hi, uint8_t* sf_hi, LayoutSF8 layout8,
                             int rows, int K, int K_low, int K_high,
                             int nblk_lo, int nblk_hi, cudaStream_t stream) {
  int tpb = 256;
  long units_lo = (long)rows * nblk_lo;          // one thread each
  long units_hi = (long)rows * nblk_hi * 32;     // one warp each (32 threads)
  long gx = ((units_lo > units_hi ? units_lo : units_hi) + tpb - 1) / tpb;
  dim3 grid((unsigned)gx, 1, 2);
#define LAUNCHM(TY, PTR)                                                                \
  qpf_mixed_kernel<TY, LayoutSF4, LayoutSF8><<<grid, tpb, 0, stream>>>(                 \
      data_lo, sf_lo, layout4, data_hi, sf_hi, layout8, PTR, rows, K, K_low, K_high,    \
      nblk_lo, nblk_hi)
  auto st = x.scalar_type();
  if (st == torch::kFloat32)       LAUNCHM(float, x.data_ptr<float>());
  else if (st == torch::kFloat16)  LAUNCHM(half, reinterpret_cast<const half*>(x.data_ptr<at::Half>()));
  else if (st == torch::kBFloat16) LAUNCHM(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
  else TORCH_CHECK(false, "quant_pack_float_mixed: x must be float32/float16/bfloat16");
#undef LAUNCHM
  qpfm_check_cuda(cudaGetLastError(), "qpf_mixed launch");
}

// entry: wider (M, K) activation -> {data_lo(M,K_low/2), sfa_lo, data_hi(M,K_high), sfa_hi}
std::vector<torch::Tensor> quant_pack_float_mixed(torch::Tensor x, int64_t M, int64_t N,
                                                  int64_t K_low, int64_t K_high) {
  TORCH_CHECK(x.is_cuda(), "x CUDA");
  TORCH_CHECK(K_low % 16 == 0, "K_low%16 (NVFP4 block-16)");
  TORCH_CHECK(K_high % 32 == 0, "K_high%32 (MXFP8 block-32)");
  init_grids_once_m();
  x = x.contiguous();
  int Mi = (int)M, Ni = (int)N, Klo = (int)K_low, Khi = (int)K_high, K = Klo + Khi;
  int nblk_lo = Klo / 16, nblk_hi = Khi / 32;
  auto dev = x.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto layout4 = BSConfigNV::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Klo, 1));
  auto layout8 = BSConfigMX::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Khi, 1));
  auto data_lo = torch::empty({(int64_t)Mi * (Klo / 2)}, u8opts(dev));
  auto sf_lo   = torch::empty({(int64_t)size(filter_zeros(layout4))}, u8opts(dev));
  auto data_hi = torch::empty({(int64_t)Mi * Khi}, u8opts(dev));
  auto sf_hi   = torch::empty({(int64_t)size(filter_zeros(layout8))}, u8opts(dev));
  launch_qpf_mixed(x, data_lo.data_ptr<uint8_t>(), sf_lo.data_ptr<uint8_t>(), layout4,
                   data_hi.data_ptr<uint8_t>(), sf_hi.data_ptr<uint8_t>(), layout8,
                   Mi, K, Klo, Khi, nblk_lo, nblk_hi, stream);
  data_lo = data_lo.view({Mi, Klo / 2});
  data_hi = data_hi.view({Mi, Khi});
  return {data_lo, sf_lo, data_hi, sf_hi};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_pack_float_mixed", &quant_pack_float_mixed,
        "SINGLE-LAUNCH fused float->NVFP4(low)+MXFP8(high) quantize+pack+interleaved-SF; "
        "returns {data_lo(M,K_low/2), sfa_lo, data_hi(M,K_high), sfa_hi}",
        pybind11::arg("x"), pybind11::arg("M"), pybind11::arg("N"),
        pybind11::arg("K_low"), pybind11::arg("K_high"));
}
