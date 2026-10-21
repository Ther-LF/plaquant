// codesign_overlap_ext.cu — round-20 STAGE-1 PoC harness for the cross-phase OVERLAP
// dual-format fused GEMM (codesign_overlap_kernel.hpp, de-unioned smem, no phase bars).
//
// Same data contract + shared-accumulator math as the SERIAL codesign_fused_ext.cu, so
// cosine must match (1.0). Stage counts are build macros (CD_NV_STAGES/CD_MX_STAGES):
// de-union doubles tensor smem, so the serial NV4/MX2 won't fit — this ext sweeps
// shallower configs to find what fits SM120's ~101 KB/block and measures overlap vs
// the serial kernel at mid-r. compat_report prints SharedStorageSize so a config that
// overflows is caught at build/initialize, not silently.
//
// PROTECTED files + the serial codesign_fused_kernel.hpp are UNTOUCHED.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"

#include "codesign_sm120_bs_mma.hpp"
#include "codesign_overlap_kernel.hpp"

using namespace cute;

#ifndef CD_NV_STAGES
#define CD_NV_STAGES 2
#endif
#ifndef CD_MX_STAGES
#define CD_MX_STAGES 1
#endif
// CTA TileShape macros (default production 128x128x128). A smaller TN (e.g. 64)
// halves B smem per stage -> lets the de-unioned overlap fit SM120's ~101 KB for a
// bit-exact run + same-tile/same-stage A/B vs the serial cd_cfg kernel.
#ifndef CDO_TM
#define CDO_TM 128
#endif
#ifndef CDO_TN
#define CDO_TN 128
#endif
#ifndef CDO_TK
#define CDO_TK 128
#endif

static inline void check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// activation/weight SF interleave: one thread per logical SF byte (efficient; r17).
template <class LayoutSF>
__global__ void cdo_fill_sf_block_kernel(uint8_t* __restrict__ phys,
                                         const uint8_t* __restrict__ logical,
                                         LayoutSF layout, int rows, int nblk, int VEC) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * nblk;
  if (idx >= total) return;
  int r = idx / nblk;
  int blk = idx % nblk;
  int off = layout(make_coord(r, blk * VEC, 0));
  phys[off] = logical[r * nblk + blk];
}

// fused quant/pack (data pack + interleaved SF) — identical to the serial ext (ILV path).
template <class LayoutSF, int VEC, bool PACK>
__global__ void cdo_quant_pack_kernel(uint8_t* __restrict__ data_out,
                                      uint8_t* __restrict__ sf_out,
                                      const uint8_t* __restrict__ data_in,
                                      const uint8_t* __restrict__ sf_in,
                                      LayoutSF layout, int rows, int nblk, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * nblk;
  if (idx >= total) return;
  int r = idx / nblk;
  int b = idx % nblk;
  uint8_t sfv = sf_in[r * nblk + b];
  int off = layout(make_coord(r, b * VEC, 0));
  sf_out[off] = sfv;
  int base = r * K + b * VEC;
  if (PACK) {
    int obase = r * (K / 2) + b * (VEC / 2);
#pragma unroll
    for (int j = 0; j < VEC / 2; ++j) {
      uint8_t lo = data_in[base + 2 * j] & 0x0F;
      uint8_t hi = data_in[base + 2 * j + 1] & 0x0F;
      data_out[obase + j] = lo | (hi << 4);
    }
  } else {
#pragma unroll
    for (int j = 0; j < VEC; ++j) data_out[base + j] = data_in[base + j];
  }
}

namespace {
template <class Policy> struct PolicyParts;
template <int S, int Sch, class CS, class KS>
struct PolicyParts<cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled<S, Sch, CS, KS>> {
  static constexpr int Stages = S;
  static constexpr int SchedStages = Sch;
  using ClusterShape = CS;
  using KernelSchedule = KS;
};
}

template <class Stock, bool ClearAccum>
using ForkMma = cutlass::gemm::collective::CodesignBlockScaledMma<
    PolicyParts<typename Stock::DispatchPolicy>::Stages,
    PolicyParts<typename Stock::DispatchPolicy>::SchedStages,
    typename PolicyParts<typename Stock::DispatchPolicy>::ClusterShape,
    typename PolicyParts<typename Stock::DispatchPolicy>::KernelSchedule,
    typename Stock::TileShape,
    typename Stock::ElementPairA, typename Stock::StridePairA,
    typename Stock::ElementPairB, typename Stock::StridePairB,
    typename Stock::TiledMma,
    typename Stock::GmemTiledCopyPairA, typename Stock::SmemLayoutAtomsA, typename Stock::SmemCopyAtomsA, typename Stock::TransformA,
    typename Stock::GmemTiledCopyPairB, typename Stock::SmemLayoutAtomsB, typename Stock::SmemCopyAtomsB, typename Stock::TransformB,
    ClearAccum>;

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
  using TileShape = Shape<Int<CDO_TM>, Int<CDO_TN>, Int<CDO_TK>>;
  using ClusterShape = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementAB, cutlass::layout::RowMajor, Alignment,
      ElementAB, cutlass::layout::ColumnMajor, Alignment,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<Stages>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
};

using NVbuild = StockBuilder<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 32, CD_NV_STAGES>;
using MXbuild = StockBuilder<cutlass::mx_float8_t<cutlass::float_e4m3_t>, 16, CD_MX_STAGES>;

using StockNV = NVbuild::CollectiveMainloop;
using StockMX = MXbuild::CollectiveMainloop;
using Epilogue = NVbuild::CollectiveEpilogue;

using ForkNV = ForkMma<StockNV, /*ClearAccum=*/true>;
using ForkMX = ForkMma<StockMX, /*ClearAccum=*/false>;

template <bool DualPhase>
using OverlapKernel = codesign_overlap::CodesignOverlapKernel<
    Shape<int,int,int,int>, ForkNV, ForkMX, Epilogue,
    cutlass::gemm::StaticPersistentScheduler, DualPhase>;

using GemmDual = cutlass::gemm::device::GemmUniversalAdapter<OverlapKernel<true>>;

using BSConfigNV = typename StockNV::Sm1xxBlkScaledConfig;
using BSConfigMX = typename StockMX::Sm1xxBlkScaledConfig;
using StrideA = typename OverlapKernel<true>::StrideA;
using StrideB = typename OverlapKernel<true>::StrideB;
using StrideC = typename OverlapKernel<true>::StrideC;
using StrideD = typename OverlapKernel<true>::StrideD;

static inline torch::TensorOptions u8opts(torch::Device dev) {
  return torch::TensorOptions().dtype(torch::kUInt8).device(dev);
}

template <class BSConfig, int VEC>
static torch::Tensor interleave_sfb(torch::Tensor SFB, int Ni, int Ki, cudaStream_t stream) {
  SFB = SFB.contiguous();
  auto dev = SFB.device();
  auto layout_SFB = BSConfig::tile_atom_to_shape_SFB(make_shape(1, Ni, Ki, 1));
  int64_t sz = (int64_t)size(filter_zeros(layout_SFB));
  auto phys = torch::empty({sz}, u8opts(dev));
  int nblk = Ki / VEC, tpb = 256, total = Ni * nblk;
  cdo_fill_sf_block_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
      phys.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(), layout_SFB, Ni, nblk, VEC);
  check_cuda(cudaGetLastError(), "interleave_sfb launch");
  return phys;
}

torch::Tensor cdo_prepack_sf_fp4_b(torch::Tensor SFB, int64_t N, int64_t K) {
  auto stream = at::cuda::getCurrentCUDAStream(SFB.device().index());
  auto phys = interleave_sfb<BSConfigNV, 16>(SFB, (int)N, (int)K, stream);
  check_cuda(cudaStreamSynchronize(stream), "prepack_b sync");
  return phys;
}
torch::Tensor cdo_prepack_sf_fp8_b(torch::Tensor SFB, int64_t N, int64_t K) {
  auto stream = at::cuda::getCurrentCUDAStream(SFB.device().index());
  auto phys = interleave_sfb<BSConfigMX, 32>(SFB, (int)N, (int)K, stream);
  check_cuda(cudaStreamSynchronize(stream), "prepack_b sync");
  return phys;
}

template <class BSConfig, int VEC, bool PACK>
static std::vector<torch::Tensor> quant_pack_impl(torch::Tensor data_in, torch::Tensor sf_in,
                                                  int Mi, int Ni, int Ki, cudaStream_t stream) {
  data_in = data_in.contiguous(); sf_in = sf_in.contiguous();
  auto dev = data_in.device();
  int nblk = Ki / VEC;
  int64_t dsz = PACK ? (int64_t)Mi * (Ki / 2) : (int64_t)Mi * Ki;
  auto data_out = torch::empty({dsz}, u8opts(dev));
  auto layout_SFA = BSConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Ki, 1));
  int64_t sfsz = (int64_t)size(filter_zeros(layout_SFA));
  auto sf_out = torch::empty({sfsz}, u8opts(dev));
  int total = Mi * nblk, tpb = 256;
  dim3 grid((total + tpb - 1) / tpb);
  cdo_quant_pack_kernel<decltype(layout_SFA), VEC, PACK><<<grid, tpb, 0, stream>>>(
      data_out.data_ptr<uint8_t>(), sf_out.data_ptr<uint8_t>(),
      data_in.data_ptr<uint8_t>(), sf_in.data_ptr<uint8_t>(), layout_SFA, Mi, nblk, Ki);
  check_cuda(cudaGetLastError(), "quant_pack launch");
  data_out = data_out.view({Mi, PACK ? Ki / 2 : Ki});
  return {data_out, sf_out};
}

std::vector<torch::Tensor> cdo_quant_pack_fp4(torch::Tensor codes, torch::Tensor SFA,
                                              int64_t M, int64_t N, int64_t K) {
  auto stream = at::cuda::getCurrentCUDAStream(codes.device().index());
  return quant_pack_impl<BSConfigNV, 16, true>(codes, SFA, (int)M, (int)N, (int)K, stream);
}
std::vector<torch::Tensor> cdo_quant_pack_fp8(torch::Tensor bytes, torch::Tensor SFA,
                                              int64_t M, int64_t N, int64_t K) {
  auto stream = at::cuda::getCurrentCUDAStream(bytes.device().index());
  return quant_pack_impl<BSConfigMX, 32, false>(bytes, SFA, (int)M, (int)N, (int)K, stream);
}

template <class Coll, class BSConfig>
static typename Coll::Arguments make_mainloop_args_pre(torch::Tensor A, torch::Tensor physSFA,
                                torch::Tensor B, torch::Tensor physSFB,
                                int Mi, int Ni, int Ki) {
  auto layout_SFA = BSConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Ki, 1));
  auto layout_SFB = BSConfig::tile_atom_to_shape_SFB(make_shape(Mi, Ni, Ki, 1));
  using ElemA = typename Coll::ElementA;
  using ElemSF = typename Coll::ElementSF;
  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {Mi, Ki, 1});
  StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {Ni, Ki, 1});
  typename Coll::Arguments m;
  m.ptr_A = reinterpret_cast<ElemA const*>(A.data_ptr<uint8_t>());
  m.dA = sA;
  m.ptr_B = reinterpret_cast<ElemA const*>(B.data_ptr<uint8_t>());
  m.dB = sB;
  m.ptr_SFA = reinterpret_cast<ElemSF const*>(physSFA.data_ptr<uint8_t>());
  m.layout_SFA = layout_SFA;
  m.ptr_SFB = reinterpret_cast<ElemSF const*>(physSFB.data_ptr<uint8_t>());
  m.layout_SFB = layout_SFB;
  return m;
}

static typename Epilogue::Arguments make_epi_args(torch::Tensor D, int Mi, int Ni) {
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {Mi, Ni, 1});
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {Mi, Ni, 1});
  using ElementC = typename Epilogue::ElementC;
  using ElementD = typename Epilogue::ElementD;
  return {
    {1.0f, 0.0f},
    reinterpret_cast<ElementC const*>(D.data_ptr()), sC,
    reinterpret_cast<ElementD*>(D.data_ptr()), sD
  };
}

template <class Gemm>
static void run_with_cached_ws(Gemm& gemm, typename Gemm::Arguments& arguments,
                               torch::Device dev, cudaStream_t stream,
                               torch::Tensor& ws_cache, bool& checked, const char* tag) {
  size_t ws = Gemm::get_workspace_size(arguments);
  if (!ws_cache.defined() || (size_t)ws_cache.numel() < std::max<size_t>(ws, 1)) {
    ws_cache = torch::empty({(int64_t)std::max<size_t>(ws, 1)}, u8opts(dev));
  }
  if (!checked) {
    auto st = gemm.can_implement(arguments);
    TORCH_CHECK(st == cutlass::Status::kSuccess, tag, " can_implement: ", cutlassGetStatusString(st));
    checked = true;
  }
  auto st = gemm.initialize(arguments, ws_cache.data_ptr(), stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, tag, " initialize: ", cutlassGetStatusString(st));
  st = gemm.run(stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, tag, " run: ", cutlassGetStatusString(st));
  check_cuda(cudaStreamSynchronize(stream), "run sync");
}

static torch::Tensor run_dual_pre(torch::Tensor A_lo, torch::Tensor physSFA_lo, torch::Tensor B_lo, torch::Tensor physSFB_lo,
                                  torch::Tensor A_hi, torch::Tensor physSFA_hi, torch::Tensor B_hi, torch::Tensor physSFB_hi,
                                  int Mi, int Ni, int Klo, int Khi, cudaStream_t stream, torch::Device dev) {
  auto D = torch::empty({Mi, Ni}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
  auto nv_args = make_mainloop_args_pre<ForkNV, BSConfigNV>(A_lo, physSFA_lo, B_lo, physSFB_lo, Mi, Ni, Klo);
  auto mx_args = make_mainloop_args_pre<ForkMX, BSConfigMX>(A_hi, physSFA_hi, B_hi, physSFB_hi, Mi, Ni, Khi);

  typename GemmDual::Arguments arguments;
  arguments.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  arguments.problem_shape = {Mi, Ni, Klo, 1};
  arguments.mainloop_nv = nv_args;
  arguments.mainloop_mx = mx_args;
  arguments.K_high = Khi;
  arguments.epilogue = make_epi_args(D, Mi, Ni);

  static GemmDual gemm;
  static torch::Tensor ws_cache;
  static bool checked = false;
  run_with_cached_ws(gemm, arguments, dev, stream, ws_cache, checked, "overlap");
  return D;
}

torch::Tensor cdo_fused_gemm_overlap_prepacked(torch::Tensor A_lo, torch::Tensor physSFA_lo, torch::Tensor B_lo, torch::Tensor physSFB_lo,
                                      torch::Tensor A_hi, torch::Tensor physSFA_hi, torch::Tensor B_hi, torch::Tensor physSFB_hi,
                                      int64_t M, int64_t N, int64_t K_lo, int64_t K_hi) {
  TORCH_CHECK(K_lo % 16 == 0 && K_hi % 32 == 0, "K_lo%16, K_hi%32");
  A_lo=A_lo.contiguous(); B_lo=B_lo.contiguous(); A_hi=A_hi.contiguous(); B_hi=B_hi.contiguous();
  physSFA_lo=physSFA_lo.contiguous(); physSFB_lo=physSFB_lo.contiguous();
  physSFA_hi=physSFA_hi.contiguous(); physSFB_hi=physSFB_hi.contiguous();
  auto dev = A_lo.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  return run_dual_pre(A_lo, physSFA_lo, B_lo, physSFB_lo, A_hi, physSFA_hi, B_hi, physSFB_hi,
                      (int)M, (int)N, (int)K_lo, (int)K_hi, stream, dev);
}

static std::string cdo_compat_report() {
  std::string s;
  s += "CODESIGN_OVERLAP_BUILD=OK\n";
  s += "  NV stages=" + std::to_string(CD_NV_STAGES) + " MX stages=" + std::to_string(CD_MX_STAGES) + "\n";
  s += "  SharedStorageSize(dual) = " + std::to_string((int)OverlapKernel<true>::SharedStorageSize) + " bytes (cap ~101376)\n";
  return s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepack_sf_fp4_b", &cdo_prepack_sf_fp4_b, "prepack NVFP4 weight SF");
  m.def("prepack_sf_fp8_b", &cdo_prepack_sf_fp8_b, "prepack MXFP8 weight SF");
  m.def("quant_pack_fp4", &cdo_quant_pack_fp4, "fused pack NVFP4 (interleaved SF); returns {packed, SF}");
  m.def("quant_pack_fp8", &cdo_quant_pack_fp8, "fused pack MXFP8 (interleaved SF); returns {bytes, SF}");
  m.def("fused_gemm_overlap_prepacked", &cdo_fused_gemm_overlap_prepacked, "OVERLAP dual-format fused GEMM (GEMM-only ceiling)");
  m.def("compat_report", &cdo_compat_report, "build/smem report");
}
