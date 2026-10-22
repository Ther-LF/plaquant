// config_dispatch.hpp — ProMix SM120 fused-GEMM HEURISTIC CONFIG DISPATCHER (round-24).
//
// Crystallizes the round-24 per-shape x M grid-search (r24_presf_search.csv +
// r24_cd_search.csv) into an EXECUTABLE, CUTLASS-profiler-style template dispatcher for
// the three kernels: pure NVFP4, pure MXFP8, and the co-design dual-format GEMM.
//
//   select_<kind>(M, N, K[, K_high]) -> ConfigTag      // if-ladder heuristic (host, cheap)
//   launch_best_<kind>(...)                             // runtime dispatch -> compile-time
//                                                       //   specialization for the chosen tag
//
// Pattern mirrors CUTLASS's profiler-derived heuristic dispatch (a table of measured
// (problem -> best kernel) winners + an if-ladder selector + templated instantiation).
// Each ConfigTag's provenance (which measured shape x M it won) is documented inline.
//
// ZERO mainloop logic change: the co-design launcher #includes the SAME PROTECTED headers
// (codesign_sm120_bs_mma.hpp / codesign_fused_kernel.hpp) and only varies the
// CollectiveBuilder CONFIG template params -- identical to the cd_cfg_ext.cu approach. The
// pure launcher mirrors pure_presf_ext.cu's builder, parameterized over tile/stages/sched.
//
// SM120 CONFIG SPACE (measured round-19/24, honored here -- NOT guessed):
//   * smem cap ~101376 B/CTA. 256-wide tiles need explicit Stages to dodge
//     StageCountAutoCarveout -> Stages=1 -> static_assert(Stages>=2). NVFP4 256x128 fits
//     at Stages=3 (smem 83968); MXFP8 256x128 does NOT (153600 > cap).
//   * cooperative kernel requires Tile-M >= 128 (static_assert) -> 64-tall tiles illegal.
//   * SM120 has no cluster multicast -> ClusterShape fixed 1x1x1.
//
// Build knobs (define before #include to bound test-TU instantiation cost):
//   DISPATCH_ENABLE_NVFP4 / DISPATCH_ENABLE_MXFP8 / DISPATCH_ENABLE_CODESIGN
//     -> emit launch_best_* + its tag instantiations for that kind. If none defined, the
//        header is heuristic-only (enums + select_* + ConfigDesc table), instant to compile.

#pragma once

#include <cstdint>

namespace promix_dispatch {

// ------------------------------------------------------------------ problem taxonomy
enum class Kind { NVFP4, MXFP8, CODESIGN };

enum class ShapeClass { QKV, GATE, DOWN };

// Classify an LLM projection by its (N,K) aspect ratio:
//   GATE/up : N >> K   (e.g. N=8192,K=2048)   -- N-heavy
//   DOWN    : K >> N   (e.g. N=2048,K=8192)   -- K-heavy
//   QKV/o   : N ~ K    (e.g. N=2048,K=2048)   -- square-ish
constexpr ShapeClass classify(int N, int K) {
  return (N >= 2 * K) ? ShapeClass::GATE
       : (K >= 2 * N) ? ShapeClass::DOWN
                      : ShapeClass::QKV;
}

// ------------------------------------------------------------------ config tags (one per measured winner)
// Pure NVFP4 winners (r24_presf_search.csv). All cooperative unless noted.
enum class NvTag {
  T128x128_AUTO,   // tile 128x128x128, StageCountAutoCarveout(NV=4), sched auto
  T128x128_PP,     // tile 128x128x128, auto stages, PINGPONG schedule
  T256x128_S3,     // tile 256x128x128, explicit Stages=3 (NVFP4-only; MXFP8 overflows)
  T128x64_AUTO     // tile 128x64x128,  StageCountAutoCarveout(NV=6), sched auto
};
// Pure MXFP8 winners (256-wide illegal for MXFP8 -> only these three).
enum class MxTag {
  T128x128_AUTO,   // tile 128x128x128, StageCountAutoCarveout(MX=2), sched auto
  T128x128_PP,     // tile 128x128x128, auto stages, PINGPONG schedule
  T128x64_AUTO     // tile 128x64x128,  StageCountAutoCarveout(MX=3), sched auto
};
// Co-design dual-format winners (r24_cd_search.csv). Cooperative, shared FP32 accumulator.
enum class CdTag {
  T128x128_NV4MX2, // tile 128x128x128, NV stages=4 / MX stages=2  (production, M>=2048)
  T128x64_NV4MX2   // tile 128x64x128,  NV stages=4 / MX stages=2  (small-M M<=512)
};

struct ConfigDesc {
  int tm, tn, tk;        // CTA tile
  int stage_a, stage_b;  // pure: stage_a=mainloop stages(0=auto); codesign: NV / MX stages
  int sched;             // 0=auto(cooperative), 2=pingpong
  const char* tag;
  const char* provenance;
};

// ===================================================================
// HEURISTIC SELECTORS  (derived from round-24 measured best-config tables)
// ===================================================================

// --- pure NVFP4 (r-independent, full-K single format) -----------------------------------
constexpr NvTag select_nvfp4(int M, int N, int K) {
  ShapeClass s = classify(N, K);
  if (M <= 512) {
    // small-M: narrow 128x64 (NV6 stages) wins for square/K-heavy; gate prefers 256x128_s3
    return (s == ShapeClass::GATE) ? NvTag::T256x128_S3 : NvTag::T128x64_AUTO;
  }
  if (M <= 2048) {
    return NvTag::T256x128_S3;                 // wide tile + forced 3 stages wins all shapes
  }
  if (M <= 4096) {
    return (s == ShapeClass::DOWN) ? NvTag::T128x128_AUTO : NvTag::T128x128_PP;
  }
  // M > 4096
  return (s == ShapeClass::QKV) ? NvTag::T128x128_PP : NvTag::T256x128_S3;
}

// --- pure MXFP8 (r-independent, full-K single format) -----------------------------------
constexpr MxTag select_mxfp8(int M, int N, int K) {
  ShapeClass s = classify(N, K);
  if (M <= 512) return MxTag::T128x64_AUTO;
  if (M <= 2048) return (s == ShapeClass::GATE) ? MxTag::T128x128_PP : MxTag::T128x64_AUTO;
  if (M <= 4096) return (s == ShapeClass::DOWN) ? MxTag::T128x128_AUTO : MxTag::T128x128_PP;
  // M > 4096
  return (s == ShapeClass::QKV) ? MxTag::T128x128_PP : MxTag::T128x128_AUTO;
}

// --- co-design dual-format (NVFP4 low-K + MXFP8 high-K, shared accumulator) --------------
// Config optimum is shape x M dependent, essentially r-INDEPENDENT (round-19/24 verified
// across r{0,1/8,1/2}). Only 128-tall tiles are legal (cooperative); only 128x128 / 128x64
// fit the de-union-free union smem. 128x64 wins the small-M tile-quantization regime.
constexpr CdTag select_codesign(int M, int N, int K, int /*K_high*/) {
  ShapeClass s = classify(N, K);
  if (M <= 512 && s != ShapeClass::GATE) return CdTag::T128x64_NV4MX2;
  return CdTag::T128x128_NV4MX2;
}

// ConfigDesc lookup (for logging / provenance / the dispatcher test).
constexpr ConfigDesc desc(NvTag t) {
  switch (t) {
    case NvTag::T128x128_AUTO: return {128,128,128, 0,0, 0, "nvfp4:128x128_auto",  "down M4096 (best); baseline"};
    case NvTag::T128x128_PP:   return {128,128,128, 0,0, 2, "nvfp4:128x128_pp",    "qkv/gate M4096, qkv M8192"};
    case NvTag::T256x128_S3:   return {256,128,128, 3,0, 0, "nvfp4:256x128_s3",    "M2048 all; gate/down M8192"};
    case NvTag::T128x64_AUTO:  return {128, 64,128, 0,0, 0, "nvfp4:128x64_auto",   "qkv/down M512"};
  }
  return {128,128,128,0,0,0,"nvfp4:128x128_auto","fallback"};
}
constexpr ConfigDesc desc(MxTag t) {
  switch (t) {
    case MxTag::T128x128_AUTO: return {128,128,128, 0,0, 0, "mxfp8:128x128_auto",  "gate/down M4096-8192; baseline"};
    case MxTag::T128x128_PP:   return {128,128,128, 0,0, 2, "mxfp8:128x128_pp",    "qkv M4096-8192, gate M2048-4096"};
    case MxTag::T128x64_AUTO:  return {128, 64,128, 0,0, 0, "mxfp8:128x64_auto",   "all M512; qkv/down M2048"};
  }
  return {128,128,128,0,0,0,"mxfp8:128x128_auto","fallback"};
}
constexpr ConfigDesc desc(CdTag t) {
  switch (t) {
    case CdTag::T128x128_NV4MX2: return {128,128,128, 4,2, 0, "codesign:128x128_nv4mx2", "all shapes M>=2048; gate M512"};
    case CdTag::T128x64_NV4MX2:  return {128, 64,128, 4,2, 0, "codesign:128x64_nv4mx2",  "qkv/down M512"};
  }
  return {128,128,128,4,2,0,"codesign:128x128_nv4mx2","fallback"};
}

}  // namespace promix_dispatch

// ===================================================================
// EXECUTABLE LAUNCHERS  (only compiled when a DISPATCH_ENABLE_* is set)
// ===================================================================
#if defined(DISPATCH_ENABLE_NVFP4) || defined(DISPATCH_ENABLE_MXFP8) || defined(DISPATCH_ENABLE_CODESIGN)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <string>

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

namespace promix_dispatch {

using namespace cute;

static inline void dsp_check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

template <class LayoutSF>
__global__ void dsp_fill_sf_kernel(uint8_t* __restrict__ phys, const uint8_t* __restrict__ logical,
                                   LayoutSF layout, int rows, int K, int VEC) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * K;
  if (idx >= total) return;
  int r = idx / K, k = idx % K, nblk = K / VEC;
  phys[layout(make_coord(r, k, 0))] = logical[r * nblk + (k / VEC)];
}

// Per-BLOCK SF fill (one thread per (row, block)); matches cd_cfg_ext.cu exactly. Used for
// the co-design dual-format path where the SFA/SFB atom is addressed at block granularity.
template <class LayoutSF>
__global__ void dsp_fill_sf_block_kernel(uint8_t* __restrict__ phys, const uint8_t* __restrict__ logical,
                                         LayoutSF layout, int rows, int nblk, int VEC) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * nblk;
  if (idx >= total) return;
  int r = idx / nblk, blk = idx % nblk;
  phys[layout(make_coord(r, blk * VEC, 0))] = logical[r * nblk + blk];
}

}  // namespace promix_dispatch
#endif

// ----------------------------------------------------------------- PURE block-scaled launcher
#if defined(DISPATCH_ENABLE_NVFP4) || defined(DISPATCH_ENABLE_MXFP8)
namespace promix_dispatch {

template <int SCHED> struct PureSched {
  using K = cutlass::gemm::collective::KernelScheduleAuto;
  using E = cutlass::epilogue::collective::EpilogueScheduleAuto;
};
template <> struct PureSched<2> {   // pingpong
  using K = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using E = cutlass::epilogue::TmaWarpSpecialized;
};

template <class ElementAB, int Alignment, int VEC, int TM, int TN, int TK, int STAGES, int SCHED>
struct PureBlockScaledGemm {
  using ElementA = ElementAB; using ElementB = ElementAB;
  using ElementD = cutlass::bfloat16_t; using ElementC = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor; using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor; using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
  using ElementAcc = float;
  using Arch = cutlass::arch::Sm120; using Op = cutlass::arch::OpClassBlockScaledTensorOp;
  using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>; using ClusterShape = Shape<_1,_1,_1>;

  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, Op, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementAcc, ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      typename PureSched<SCHED>::E>::CollectiveOp;

  using StageArg = typename std::conditional<(STAGES > 0),
      cutlass::gemm::collective::StageCount<STAGES>,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epi::SharedStorage))>>::type;

  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, Op, ElementA, LayoutA, Alignment, ElementB, LayoutB, Alignment, ElementAcc,
      TileShape, ClusterShape, StageArg, typename PureSched<SCHED>::K>::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA; using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC; using StrideD = typename Gemm::GemmKernel::StrideD;
  using BSConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  static torch::Tensor prepack_b(torch::Tensor SFB, int64_t N, int64_t K) {
    SFB = SFB.contiguous(); auto dev = SFB.device();
    auto stream = at::cuda::getCurrentCUDAStream(dev.index());
    int Ni=(int)N, Ki=(int)K;
    auto layout = BSConfig::tile_atom_to_shape_SFB(make_shape(1, Ni, Ki, 1));
    int64_t sz = (int64_t)size(filter_zeros(layout));
    auto phys = torch::empty({sz}, torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    int total=Ni*Ki, tpb=256;
    dsp_fill_sf_kernel<<<(total+tpb-1)/tpb, tpb, 0, stream>>>(phys.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(), layout, Ni, Ki, VEC);
    dsp_check_cuda(cudaGetLastError(), "pure prepack_b");
    dsp_check_cuda(cudaStreamSynchronize(stream), "pure prepack_b sync");
    return phys;
  }

  static torch::Tensor run(torch::Tensor A, torch::Tensor SFA, torch::Tensor B, torch::Tensor physSFB,
                           int64_t M, int64_t N, int64_t K) {
    A=A.contiguous(); B=B.contiguous(); SFA=SFA.contiguous(); physSFB=physSFB.contiguous();
    auto dev=A.device(); auto stream=at::cuda::getCurrentCUDAStream(dev.index());
    int Mi=(int)M, Ni=(int)N, Ki=(int)K;
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {Mi,Ki,1});
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {Ni,Ki,1});
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {Mi,Ni,1});
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {Mi,Ni,1});
    auto layout_SFA = BSConfig::tile_atom_to_shape_SFA(make_shape(Mi,Ni,Ki,1));
    auto layout_SFB = BSConfig::tile_atom_to_shape_SFB(make_shape(Mi,Ni,Ki,1));
    auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    int64_t sfa_sz = (int64_t)size(filter_zeros(layout_SFA));
    auto phys_SFA = torch::empty({sfa_sz}, u8);
    { int total=Mi*Ki, tpb=256;
      dsp_fill_sf_kernel<<<(total+tpb-1)/tpb, tpb, 0, stream>>>(phys_SFA.data_ptr<uint8_t>(), SFA.data_ptr<uint8_t>(), layout_SFA, Mi, Ki, VEC);
      dsp_check_cuda(cudaGetLastError(), "pure fill SFA"); }
    auto D = torch::empty({M,N}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {Mi,Ni,Ki,1},
      { reinterpret_cast<typename ElementAB::DataType const*>(A.data_ptr<uint8_t>()), sA,
        reinterpret_cast<typename ElementAB::DataType const*>(B.data_ptr<uint8_t>()), sB,
        reinterpret_cast<typename ElementAB::ScaleFactorType const*>(phys_SFA.data_ptr<uint8_t>()), layout_SFA,
        reinterpret_cast<typename ElementAB::ScaleFactorType const*>(physSFB.data_ptr<uint8_t>()), layout_SFB },
      { {1.0f,0.0f}, reinterpret_cast<ElementC const*>(D.data_ptr()), sC,
        reinterpret_cast<ElementD*>(D.data_ptr()), sD } };
    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    auto wsb = torch::empty({(int64_t)ws}, u8);
    auto st = gemm.can_implement(args); TORCH_CHECK(st==cutlass::Status::kSuccess, "can_implement: ", cutlassGetStatusString(st));
    st = gemm.initialize(args, wsb.data_ptr(), stream); TORCH_CHECK(st==cutlass::Status::kSuccess, "initialize: ", cutlassGetStatusString(st));
    st = gemm.run(stream); TORCH_CHECK(st==cutlass::Status::kSuccess, "run: ", cutlassGetStatusString(st));
    dsp_check_cuda(cudaStreamSynchronize(stream), "pure run sync");
    return D;
  }
};

}  // namespace promix_dispatch
#endif

// ----------------------------------------------------------------- CO-DESIGN dual-format launcher
#if defined(DISPATCH_ENABLE_CODESIGN)
#include "codesign_sm120_bs_mma.hpp"
#include "codesign_fused_kernel.hpp"

namespace promix_dispatch {

namespace cd_detail {
template <class Policy> struct PolicyParts;
template <int S, int Sch, class CS, class KS>
struct PolicyParts<cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled<S, Sch, CS, KS>> {
  static constexpr int Stages = S;
  static constexpr int SchedStages = Sch;
  using ClusterShape = CS; using KernelSchedule = KS;
};
}  // namespace cd_detail

template <class Stock, bool ClearAccum>
using CdForkMma = cutlass::gemm::collective::CodesignBlockScaledMma<
    cd_detail::PolicyParts<typename Stock::DispatchPolicy>::Stages,
    cd_detail::PolicyParts<typename Stock::DispatchPolicy>::SchedStages,
    typename cd_detail::PolicyParts<typename Stock::DispatchPolicy>::ClusterShape,
    typename cd_detail::PolicyParts<typename Stock::DispatchPolicy>::KernelSchedule,
    typename Stock::TileShape,
    typename Stock::ElementPairA, typename Stock::StridePairA,
    typename Stock::ElementPairB, typename Stock::StridePairB,
    typename Stock::TiledMma,
    typename Stock::GmemTiledCopyPairA, typename Stock::SmemLayoutAtomsA, typename Stock::SmemCopyAtomsA, typename Stock::TransformA,
    typename Stock::GmemTiledCopyPairB, typename Stock::SmemLayoutAtomsB, typename Stock::SmemCopyAtomsB, typename Stock::TransformB,
    ClearAccum>;

// Co-design dual-format GEMM, parameterized on tile + per-format stages (mirrors cd_cfg_ext.cu).
template <int TM, int TN, int TK, int NVST, int MXST>
struct CodesignDualGemm {
  template <class ElementAB, int Alignment, int Stages>
  struct Stock {
    using ElementD = cutlass::bfloat16_t; using ElementC = cutlass::bfloat16_t;
    static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using ElementAcc = float; using Arch = cutlass::arch::Sm120;
    using Op = cutlass::arch::OpClassBlockScaledTensorOp;
    using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>; using ClusterShape = Shape<_1,_1,_1>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, Op, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, ElementAcc, ElementC, cutlass::layout::RowMajor, AlignC,
        ElementD, cutlass::layout::RowMajor, AlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, Op, ElementAB, cutlass::layout::RowMajor, Alignment,
        ElementAB, cutlass::layout::ColumnMajor, Alignment, ElementAcc, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCount<Stages>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  };

  using NVbuild = Stock<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 32, NVST>;
  using MXbuild = Stock<cutlass::mx_float8_t<cutlass::float_e4m3_t>, 16, MXST>;
  using Epilogue = typename NVbuild::Epi;
  using ForkNV = CdForkMma<typename NVbuild::Main, /*ClearAccum=*/true>;
  using ForkMX = CdForkMma<typename MXbuild::Main, /*ClearAccum=*/false>;

  template <bool DualPhase>
  using FusedKernel = codesign::CodesignFusedKernel<
      Shape<int,int,int,int>, ForkNV, ForkMX, Epilogue,
      cutlass::gemm::StaticPersistentScheduler, DualPhase>;
  using GemmDual   = cutlass::gemm::device::GemmUniversalAdapter<FusedKernel<true>>;
  using GemmSingle = cutlass::gemm::device::GemmUniversalAdapter<FusedKernel<false>>;
  using BSConfigNV = typename NVbuild::Main::Sm1xxBlkScaledConfig;
  using BSConfigMX = typename MXbuild::Main::Sm1xxBlkScaledConfig;
  using StrideA = typename FusedKernel<true>::StrideA; using StrideB = typename FusedKernel<true>::StrideB;
  using StrideC = typename FusedKernel<true>::StrideC; using StrideD = typename FusedKernel<true>::StrideD;

  template <class BSConfig, int VEC>
  static torch::Tensor interleave_sfa(torch::Tensor SFA, int Mi, int Ni, int Ki, cudaStream_t stream) {
    SFA = SFA.contiguous(); auto dev = SFA.device();
    auto layout = BSConfig::tile_atom_to_shape_SFA(make_shape(Mi,Ni,Ki,1));
    int64_t sz = (int64_t)size(filter_zeros(layout));
    auto phys = torch::empty({sz}, torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    int nblk=Ki/VEC, tpb=256, total=Mi*nblk;
    dsp_fill_sf_block_kernel<<<(total+tpb-1)/tpb, tpb, 0, stream>>>(phys.data_ptr<uint8_t>(), SFA.data_ptr<uint8_t>(), layout, Mi, nblk, VEC);
    dsp_check_cuda(cudaGetLastError(), "cd interleave_sfa");
    return phys;
  }
  template <class BSConfig, int VEC>
  static torch::Tensor interleave_sfb(torch::Tensor SFB, int Ni, int Ki, cudaStream_t stream) {
    SFB = SFB.contiguous(); auto dev = SFB.device();
    auto layout = BSConfig::tile_atom_to_shape_SFB(make_shape(1,Ni,Ki,1));
    int64_t sz = (int64_t)size(filter_zeros(layout));
    auto phys = torch::empty({sz}, torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    int nblk=Ki/VEC, tpb=256, total=Ni*nblk;
    dsp_fill_sf_block_kernel<<<(total+tpb-1)/tpb, tpb, 0, stream>>>(phys.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(), layout, Ni, nblk, VEC);
    dsp_check_cuda(cudaGetLastError(), "cd interleave_sfb");
    return phys;
  }

  template <class Coll, class BSConfig>
  static typename Coll::Arguments mainloop_args(torch::Tensor A, torch::Tensor pSFA, torch::Tensor B, torch::Tensor pSFB,
                                                int Mi, int Ni, int Ki) {
    auto lSFA = BSConfig::tile_atom_to_shape_SFA(make_shape(Mi,Ni,Ki,1));
    auto lSFB = BSConfig::tile_atom_to_shape_SFB(make_shape(Mi,Ni,Ki,1));
    using ElemA = typename Coll::ElementA; using ElemSF = typename Coll::ElementSF;
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {Mi,Ki,1});
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {Ni,Ki,1});
    typename Coll::Arguments m;
    m.ptr_A = reinterpret_cast<ElemA const*>(A.data_ptr<uint8_t>()); m.dA = sA;
    m.ptr_B = reinterpret_cast<ElemA const*>(B.data_ptr<uint8_t>()); m.dB = sB;
    m.ptr_SFA = reinterpret_cast<ElemSF const*>(pSFA.data_ptr<uint8_t>()); m.layout_SFA = lSFA;
    m.ptr_SFB = reinterpret_cast<ElemSF const*>(pSFB.data_ptr<uint8_t>()); m.layout_SFB = lSFB;
    return m;
  }
  static typename Epilogue::Arguments epi_args(torch::Tensor D, int Mi, int Ni) {
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {Mi,Ni,1});
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {Mi,Ni,1});
    using EC = typename Epilogue::ElementC; using ED = typename Epilogue::ElementD;
    return { {1.0f,0.0f}, reinterpret_cast<EC const*>(D.data_ptr()), sC, reinterpret_cast<ED*>(D.data_ptr()), sD };
  }

  // Dual-format fused GEMM (activation + weight SF interleaved internally).
  static torch::Tensor run(torch::Tensor A_lo, torch::Tensor SFA_lo, torch::Tensor B_lo, torch::Tensor SFB_lo,
                           torch::Tensor A_hi, torch::Tensor SFA_hi, torch::Tensor B_hi, torch::Tensor SFB_hi,
                           int64_t M, int64_t N, int64_t K_lo, int64_t K_hi) {
    auto dev=A_lo.device(); auto stream=at::cuda::getCurrentCUDAStream(dev.index());
    int Mi=(int)M, Ni=(int)N, Klo=(int)K_lo, Khi=(int)K_hi;
    auto pSFA_lo = interleave_sfa<BSConfigNV,16>(SFA_lo, Mi, Ni, Klo, stream);
    auto pSFB_lo = interleave_sfb<BSConfigNV,16>(SFB_lo, Ni, Klo, stream);
    auto pSFA_hi = interleave_sfa<BSConfigMX,32>(SFA_hi, Mi, Ni, Khi, stream);
    auto pSFB_hi = interleave_sfb<BSConfigMX,32>(SFB_hi, Ni, Khi, stream);
    auto D = torch::empty({Mi,Ni}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
    auto nv = mainloop_args<ForkNV,BSConfigNV>(A_lo, pSFA_lo, B_lo, pSFB_lo, Mi, Ni, Klo);
    auto mx = mainloop_args<ForkMX,BSConfigMX>(A_hi, pSFA_hi, B_hi, pSFB_hi, Mi, Ni, Khi);
    typename GemmDual::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = {Mi,Ni,Klo,1};
    args.mainloop_nv = nv; args.mainloop_mx = mx; args.K_high = Khi;
    args.epilogue = epi_args(D, Mi, Ni);
    GemmDual gemm;
    size_t ws = GemmDual::get_workspace_size(args);
    auto wsb = torch::empty({(int64_t)std::max<size_t>(ws,1)}, torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    auto st = gemm.can_implement(args); TORCH_CHECK(st==cutlass::Status::kSuccess, "cd can_implement: ", cutlassGetStatusString(st));
    st = gemm.initialize(args, wsb.data_ptr(), stream); TORCH_CHECK(st==cutlass::Status::kSuccess, "cd initialize: ", cutlassGetStatusString(st));
    st = gemm.run(stream); TORCH_CHECK(st==cutlass::Status::kSuccess, "cd run: ", cutlassGetStatusString(st));
    dsp_check_cuda(cudaStreamSynchronize(stream), "cd run sync");
    return D;
  }
};

}  // namespace promix_dispatch
#endif  // DISPATCH_ENABLE_CODESIGN

// ----------------------------------------------------------------- runtime dispatch entry points
#if defined(DISPATCH_ENABLE_NVFP4)
namespace promix_dispatch {
using NvG_128x128_auto = PureBlockScaledGemm<cutlass::nv_float4_t<cutlass::float_e2m1_t>,32,16, 128,128,128, 0,0>;
using NvG_128x128_pp   = PureBlockScaledGemm<cutlass::nv_float4_t<cutlass::float_e2m1_t>,32,16, 128,128,128, 0,2>;
using NvG_256x128_s3   = PureBlockScaledGemm<cutlass::nv_float4_t<cutlass::float_e2m1_t>,32,16, 256,128,128, 3,0>;
using NvG_128x64_auto  = PureBlockScaledGemm<cutlass::nv_float4_t<cutlass::float_e2m1_t>,32,16, 128, 64,128, 0,0>;

// Returns the selected tag name so the caller can verify provenance (test harness).
inline std::string tag_nvfp4(int M, int N, int K) { return desc(select_nvfp4(M,N,K)).tag; }

inline torch::Tensor launch_best_nvfp4(torch::Tensor A, torch::Tensor SFA, torch::Tensor B,
                                       torch::Tensor SFB, int64_t M, int64_t N, int64_t K) {
  // physSFB layout is config-dependent (interleave atom differs per tile) -> prepack with the chosen config.
  switch (select_nvfp4((int)M,(int)N,(int)K)) {
    case NvTag::T128x128_AUTO: { auto pb=NvG_128x128_auto::prepack_b(SFB,N,K); return NvG_128x128_auto::run(A,SFA,B,pb,M,N,K); }
    case NvTag::T128x128_PP:   { auto pb=NvG_128x128_pp::prepack_b(SFB,N,K);   return NvG_128x128_pp::run(A,SFA,B,pb,M,N,K); }
    case NvTag::T256x128_S3:   { auto pb=NvG_256x128_s3::prepack_b(SFB,N,K);   return NvG_256x128_s3::run(A,SFA,B,pb,M,N,K); }
    case NvTag::T128x64_AUTO:  { auto pb=NvG_128x64_auto::prepack_b(SFB,N,K);  return NvG_128x64_auto::run(A,SFA,B,pb,M,N,K); }
  }
  auto pb=NvG_128x128_auto::prepack_b(SFB,N,K); return NvG_128x128_auto::run(A,SFA,B,pb,M,N,K);
}
}  // namespace promix_dispatch
#endif

#if defined(DISPATCH_ENABLE_MXFP8)
namespace promix_dispatch {
using MxG_128x128_auto = PureBlockScaledGemm<cutlass::mx_float8_t<cutlass::float_e4m3_t>,16,32, 128,128,128, 0,0>;
using MxG_128x128_pp   = PureBlockScaledGemm<cutlass::mx_float8_t<cutlass::float_e4m3_t>,16,32, 128,128,128, 0,2>;
using MxG_128x64_auto  = PureBlockScaledGemm<cutlass::mx_float8_t<cutlass::float_e4m3_t>,16,32, 128, 64,128, 0,0>;

inline std::string tag_mxfp8(int M, int N, int K) { return desc(select_mxfp8(M,N,K)).tag; }

inline torch::Tensor launch_best_mxfp8(torch::Tensor A, torch::Tensor SFA, torch::Tensor B,
                                       torch::Tensor SFB, int64_t M, int64_t N, int64_t K) {
  switch (select_mxfp8((int)M,(int)N,(int)K)) {
    case MxTag::T128x128_AUTO: { auto pb=MxG_128x128_auto::prepack_b(SFB,N,K); return MxG_128x128_auto::run(A,SFA,B,pb,M,N,K); }
    case MxTag::T128x128_PP:   { auto pb=MxG_128x128_pp::prepack_b(SFB,N,K);   return MxG_128x128_pp::run(A,SFA,B,pb,M,N,K); }
    case MxTag::T128x64_AUTO:  { auto pb=MxG_128x64_auto::prepack_b(SFB,N,K);  return MxG_128x64_auto::run(A,SFA,B,pb,M,N,K); }
  }
  auto pb=MxG_128x128_auto::prepack_b(SFB,N,K); return MxG_128x128_auto::run(A,SFA,B,pb,M,N,K);
}
}  // namespace promix_dispatch
#endif

#if defined(DISPATCH_ENABLE_CODESIGN)
namespace promix_dispatch {
using CdG_128x128 = CodesignDualGemm<128,128,128, 4,2>;
using CdG_128x64  = CodesignDualGemm<128, 64,128, 4,2>;

inline std::string tag_codesign(int M, int N, int K, int Kh) { return desc(select_codesign(M,N,K,Kh)).tag; }

inline torch::Tensor launch_best_codesign(torch::Tensor A_lo, torch::Tensor SFA_lo, torch::Tensor B_lo, torch::Tensor SFB_lo,
                                          torch::Tensor A_hi, torch::Tensor SFA_hi, torch::Tensor B_hi, torch::Tensor SFB_hi,
                                          int64_t M, int64_t N, int64_t K_lo, int64_t K_hi) {
  switch (select_codesign((int)M,(int)N,(int)(K_lo+K_hi),(int)K_hi)) {
    case CdTag::T128x128_NV4MX2: return CdG_128x128::run(A_lo,SFA_lo,B_lo,SFB_lo,A_hi,SFA_hi,B_hi,SFB_hi,M,N,K_lo,K_hi);
    case CdTag::T128x64_NV4MX2:  return CdG_128x64::run(A_lo,SFA_lo,B_lo,SFB_lo,A_hi,SFA_hi,B_hi,SFB_hi,M,N,K_lo,K_hi);
  }
  return CdG_128x128::run(A_lo,SFA_lo,B_lo,SFB_lo,A_hi,SFA_hi,B_hi,SFB_hi,M,N,K_lo,K_hi);
}
}  // namespace promix_dispatch
#endif
