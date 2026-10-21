// cd_cfg_ext.cu — round-19 CO-DESIGN GRID CONFIG SEARCH harness.
//
// This is a MACRO-PARAMETERIZED copy of the codesign_fused_ext.cu config layer. It
// #includes the SAME protected mainloop headers (codesign_sm120_bs_mma.hpp /
// codesign_fused_kernel.hpp) with ZERO logic changes -- only the CollectiveBuilder
// CONFIG (TileShape, ClusterShape, mainloop stage counts) is swapped via -D macros so
// each config is a separate compile / separate named extension. The co-design mainloop
// algorithm (twin-CollectiveMma, shared FP32 accumulator, phase barriers, cooperative
// schedule) is identical to the shipping build. We keep COOPERATIVE (shared accumulator
// requires it) -- NO pingpong.
//
// Macros (all have defaults = the shipping 128x128x128 NV4/MX2 1x1x1 config):
//   CDCFG_TM/TN/TK   CTA TileShape
//   CDCFG_CM/CN/CK   ClusterShape
//   CDCFG_NVST       NVFP4 mainloop stages
//   CDCFG_MXST       MXFP8 mainloop stages
//
// Exposes: prepack_sf_{fp4,fp8}_{a,b}, fused_gemm_prepacked (GEMM-only ceiling),
//          single_nvfp4_gemm (r=0 path), compat_report (smem + tile report).
// Trimmed vs codesign_fused_ext.cu (no slow/quant_pack/non-prepacked fused_gemm) to
// cut per-config compile time. Pack is config-independent so GEMM-only latency is the
// correct discriminator across configs; e2e = (fixed pack delta from r19) + cd_gemmonly.

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
#include "codesign_fused_kernel.hpp"

using namespace cute;

#ifndef CDCFG_TM
#define CDCFG_TM 128
#endif
#ifndef CDCFG_TN
#define CDCFG_TN 128
#endif
#ifndef CDCFG_TK
#define CDCFG_TK 128
#endif
#ifndef CDCFG_CM
#define CDCFG_CM 1
#endif
#ifndef CDCFG_CN
#define CDCFG_CN 1
#endif
#ifndef CDCFG_CK
#define CDCFG_CK 1
#endif
#ifndef CDCFG_NVST
#define CDCFG_NVST 4
#endif
#ifndef CDCFG_MXST
#define CDCFG_MXST 2
#endif

static inline void check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

template <class LayoutSF>
__global__ void cfg_fill_sf_block_kernel(uint8_t* __restrict__ phys,
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
  using TileShape = Shape<Int<CDCFG_TM>, Int<CDCFG_TN>, Int<CDCFG_TK>>;
  using ClusterShape = Shape<Int<CDCFG_CM>, Int<CDCFG_CN>, Int<CDCFG_CK>>;

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

using NVbuild = StockBuilder<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 32, CDCFG_NVST>;
using MXbuild = StockBuilder<cutlass::mx_float8_t<cutlass::float_e4m3_t>, 16, CDCFG_MXST>;

using StockNV = NVbuild::CollectiveMainloop;
using StockMX = MXbuild::CollectiveMainloop;
using Epilogue = NVbuild::CollectiveEpilogue;

using ForkNV = ForkMma<StockNV, /*ClearAccum=*/true>;
using ForkMX = ForkMma<StockMX, /*ClearAccum=*/false>;

template <bool DualPhase>
using FusedKernel = codesign::CodesignFusedKernel<
    Shape<int,int,int,int>, ForkNV, ForkMX, Epilogue,
    cutlass::gemm::StaticPersistentScheduler, DualPhase>;

using GemmDual   = cutlass::gemm::device::GemmUniversalAdapter<FusedKernel<true>>;
using GemmSingle = cutlass::gemm::device::GemmUniversalAdapter<FusedKernel<false>>;

using BSConfigNV = typename StockNV::Sm1xxBlkScaledConfig;
using BSConfigMX = typename StockMX::Sm1xxBlkScaledConfig;
using StrideA = typename FusedKernel<true>::StrideA;
using StrideB = typename FusedKernel<true>::StrideB;
using StrideC = typename FusedKernel<true>::StrideC;
using StrideD = typename FusedKernel<true>::StrideD;

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
  cfg_fill_sf_block_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
      phys.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(), layout_SFB, Ni, nblk, VEC);
  check_cuda(cudaGetLastError(), "interleave_sfb launch");
  return phys;
}

template <class BSConfig, int VEC>
static torch::Tensor interleave_sfa(torch::Tensor SFA, int Mi, int Ni, int Ki, cudaStream_t stream) {
  SFA = SFA.contiguous();
  auto dev = SFA.device();
  auto layout_SFA = BSConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Ki, 1));
  int64_t sz = (int64_t)size(filter_zeros(layout_SFA));
  auto phys = torch::empty({sz}, u8opts(dev));
  int nblk = Ki / VEC, tpb = 256, total = Mi * nblk;
  cfg_fill_sf_block_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
      phys.data_ptr<uint8_t>(), SFA.data_ptr<uint8_t>(), layout_SFA, Mi, nblk, VEC);
  check_cuda(cudaGetLastError(), "interleave_sfa launch");
  return phys;
}

torch::Tensor cfg_prepack_sf_fp4_b(torch::Tensor SFB, int64_t N, int64_t K) {
  auto stream = at::cuda::getCurrentCUDAStream(SFB.device().index());
  auto phys = interleave_sfb<BSConfigNV, 16>(SFB, (int)N, (int)K, stream);
  check_cuda(cudaStreamSynchronize(stream), "prepack_b sync"); return phys;
}
torch::Tensor cfg_prepack_sf_fp8_b(torch::Tensor SFB, int64_t N, int64_t K) {
  auto stream = at::cuda::getCurrentCUDAStream(SFB.device().index());
  auto phys = interleave_sfb<BSConfigMX, 32>(SFB, (int)N, (int)K, stream);
  check_cuda(cudaStreamSynchronize(stream), "prepack_b sync"); return phys;
}
torch::Tensor cfg_prepack_sf_fp4_a(torch::Tensor SFA, int64_t M, int64_t N, int64_t K) {
  auto stream = at::cuda::getCurrentCUDAStream(SFA.device().index());
  auto phys = interleave_sfa<BSConfigNV, 16>(SFA, (int)M, (int)N, (int)K, stream);
  check_cuda(cudaStreamSynchronize(stream), "prepack_a sync"); return phys;
}
torch::Tensor cfg_prepack_sf_fp8_a(torch::Tensor SFA, int64_t M, int64_t N, int64_t K) {
  auto stream = at::cuda::getCurrentCUDAStream(SFA.device().index());
  auto phys = interleave_sfa<BSConfigMX, 32>(SFA, (int)M, (int)N, (int)K, stream);
  check_cuda(cudaStreamSynchronize(stream), "prepack_a sync"); return phys;
}

template <class Coll, class BSConfig>
static typename Coll::Arguments make_mainloop_args_pre(torch::Tensor A, torch::Tensor physSFA,
                                torch::Tensor B, torch::Tensor physSFB, int Mi, int Ni, int Ki) {
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
  return { {1.0f, 0.0f},
    reinterpret_cast<ElementC const*>(D.data_ptr()), sC,
    reinterpret_cast<ElementD*>(D.data_ptr()), sD };
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

torch::Tensor cfg_fused_gemm_prepacked(torch::Tensor A_lo, torch::Tensor physSFA_lo, torch::Tensor B_lo, torch::Tensor physSFB_lo,
                                       torch::Tensor A_hi, torch::Tensor physSFA_hi, torch::Tensor B_hi, torch::Tensor physSFB_hi,
                                       int64_t M, int64_t N, int64_t K_lo, int64_t K_hi) {
  A_lo=A_lo.contiguous(); B_lo=B_lo.contiguous(); A_hi=A_hi.contiguous(); B_hi=B_hi.contiguous();
  physSFA_lo=physSFA_lo.contiguous(); physSFB_lo=physSFB_lo.contiguous();
  physSFA_hi=physSFA_hi.contiguous(); physSFB_hi=physSFB_hi.contiguous();
  auto dev = A_lo.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  int Mi=(int)M, Ni=(int)N, Klo=(int)K_lo, Khi=(int)K_hi;
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
  static GemmDual gemm; static torch::Tensor ws_cache; static bool checked = false;
  run_with_cached_ws(gemm, arguments, dev, stream, ws_cache, checked, "cfg_fused");
  return D;
}

torch::Tensor cfg_single_nvfp4_gemm(torch::Tensor A, torch::Tensor SFA, torch::Tensor B, torch::Tensor physSFB,
                                    int64_t M, int64_t N, int64_t K) {
  A = A.contiguous(); B = B.contiguous(); SFA = SFA.contiguous(); physSFB = physSFB.contiguous();
  auto dev = A.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  int Mi=(int)M, Ni=(int)N, Ki=(int)K;
  auto D = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
  auto physSFA = interleave_sfa<BSConfigNV, 16>(SFA, Mi, Ni, Ki, stream);
  auto nv_args = make_mainloop_args_pre<ForkNV, BSConfigNV>(A, physSFA, B, physSFB, Mi, Ni, Ki);
  typename GemmSingle::Arguments arguments;
  arguments.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  arguments.problem_shape = {Mi, Ni, Ki, 1};
  arguments.mainloop_nv = nv_args;
  arguments.K_high = 0;
  arguments.epilogue = make_epi_args(D, Mi, Ni);
  static GemmSingle gemm; static torch::Tensor ws_cache; static bool checked = false;
  run_with_cached_ws(gemm, arguments, dev, stream, ws_cache, checked, "cfg_single");
  return D;
}

static std::string cfg_compat_report() {
  std::string s;
  s += "CDCFG_BUILD=OK\n";
  s += "  tile=" + std::to_string(CDCFG_TM) + "x" + std::to_string(CDCFG_TN) + "x" + std::to_string(CDCFG_TK) + "\n";
  s += "  cluster=" + std::to_string(CDCFG_CM) + "x" + std::to_string(CDCFG_CN) + "x" + std::to_string(CDCFG_CK) + "\n";
  s += "  stages NV=" + std::to_string(CDCFG_NVST) + " MX=" + std::to_string(CDCFG_MXST) + "\n";
  s += "  SharedStorageSize(dual) = " + std::to_string((int)FusedKernel<true>::SharedStorageSize) + " bytes\n";
  s += "  SharedStorageSize(single) = " + std::to_string((int)FusedKernel<false>::SharedStorageSize) + " bytes\n";
  s += "  MaxThreadsPerBlock = " + std::to_string((int)FusedKernel<true>::MaxThreadsPerBlock) + "\n";
  return s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepack_sf_fp4_b", &cfg_prepack_sf_fp4_b);
  m.def("prepack_sf_fp8_b", &cfg_prepack_sf_fp8_b);
  m.def("prepack_sf_fp4_a", &cfg_prepack_sf_fp4_a);
  m.def("prepack_sf_fp8_a", &cfg_prepack_sf_fp8_a);
  m.def("fused_gemm_prepacked", &cfg_fused_gemm_prepacked);
  m.def("single_nvfp4_gemm", &cfg_single_nvfp4_gemm);
  m.def("compat_report", &cfg_compat_report);
}
