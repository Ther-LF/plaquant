// cd_smallm_ext.cu — vllm-perf round-2 lever 1: SMALL-M co-design GEMM configs.
//
// The vLLM plugin's fused dual-format GEMM (codesign_fused_ext.cu) is a single
// FIXED 128x128 NV4/MX2 config — the round-24 optimum for prefill M>=2048. At
// GEMM-routed decode/chunked-prefill sizes (M=8..512) that tile gives only
// N/128 CTAs (24-48 on 188 SMs for qkv/o/down) and the kernel runs 2.5-5x above
// the weight-traffic floor (r1 probe). The round-24 grid search (r24_cd_search)
// already found 128x64 NV4/MX2 wins the small-M regime (config_dispatch.hpp
// CdTag::T128x64_NV4MX2) but only exposed it through launch_best_codesign,
// whose per-call RAW-SF interleave + stream syncs make it unusable in the
// serving path (CUDA-graph capture + prepacked activation SF from
// quant_pack_float).
//
// This ext exposes the alternative configs through the SAME prepacked-SF,
// cached-workspace, capture-safe entry contract as cd_fused_gemm_prepacked:
//
//   fused_gemm_prepacked_cfg(..., alpha, cfg)
//     cfg 0 : 128x128 NV4/MX2  (== the shipping config; cross-check only)
//     cfg 1 : 128x64  NV4/MX2  (r24 small-M winner)
//     cfg 2 : 128x64  NV6/MX2  (deeper NVFP4 ring; fits the freed B-smem)
//     cfg 3 : 128x64  NV6/MX3  (deeper both)
//
// SF-layout compatibility: the CUTLASS interleaved SF atom
// (Sm1xxBlockScaledConfig<SFVecSize>) depends ONLY on SFVecSize (16/32), NOT on
// the tile shape — verified in cutlass/detail/sm100_blockscaled_layout.hpp
// (Blk_MN/Blk_SF fixed by Sm1xxBlockScaledBasicChunk). The physSFA/physSFB
// buffers produced by quant_pack_float / prepack_sf_*_b are therefore
// byte-compatible with every config here (gated by torch.equal vs cfg0 in
// promix/vllm_integ/gemm_probe.py).
//
// PROTECTED mainloop headers (codesign_sm120_bs_mma.hpp /
// codesign_fused_kernel.hpp) are #included UNCHANGED — this file only varies
// CollectiveBuilder CONFIG template params, identical to the cd_cfg_ext.cu /
// config_dispatch.hpp approach.

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

static inline void sm_check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
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
}  // namespace

template <class Stock, bool ClearAccum>
using SmForkMma = cutlass::gemm::collective::CodesignBlockScaledMma<
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

// ---- config-templated dual-format fused GEMM (prepacked SF, cached ws) ----
template <int TM, int TN, int NVST, int MXST>
struct SmallMGemm {
  template <class ElementAB, int Alignment, int Stages>
  struct Stock {
    using ElementD = cutlass::bfloat16_t;   // matches the shipping vLLM ext (bf16 out)
    using ElementC = cutlass::bfloat16_t;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using ElementAccumulator = float;
    using ArchTag = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
    using TileShape = Shape<Int<TM>, Int<TN>, _128>;
    using ClusterShape = Shape<_1,_1,_1>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, cutlass::layout::RowMajor, AlignmentC,
        ElementD, cutlass::layout::RowMajor, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementAB, cutlass::layout::RowMajor, Alignment,
        ElementAB, cutlass::layout::ColumnMajor, Alignment,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCount<Stages>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  };

  using NVbuild = Stock<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 32, NVST>;
  using MXbuild = Stock<cutlass::mx_float8_t<cutlass::float_e4m3_t>, 16, MXST>;
  using Epilogue = typename NVbuild::CollectiveEpilogue;
  using ForkNV = SmForkMma<typename NVbuild::CollectiveMainloop, /*ClearAccum=*/true>;
  using ForkMX = SmForkMma<typename MXbuild::CollectiveMainloop, /*ClearAccum=*/false>;
  using FusedK = codesign::CodesignFusedKernel<
      Shape<int,int,int,int>, ForkNV, ForkMX, Epilogue,
      cutlass::gemm::StaticPersistentScheduler, /*DualPhase=*/true>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<FusedK>;
  using BSConfigNV = typename NVbuild::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using BSConfigMX = typename MXbuild::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using StrideA = typename FusedK::StrideA;
  using StrideB = typename FusedK::StrideB;
  using StrideC = typename FusedK::StrideC;
  using StrideD = typename FusedK::StrideD;

  template <class Coll, class BSConfig>
  static typename Coll::Arguments mainloop_args(torch::Tensor A, torch::Tensor physSFA,
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

  static torch::Tensor run(torch::Tensor A_lo, torch::Tensor physSFA_lo,
                           torch::Tensor B_lo, torch::Tensor physSFB_lo,
                           torch::Tensor A_hi, torch::Tensor physSFA_hi,
                           torch::Tensor B_hi, torch::Tensor physSFB_hi,
                           int Mi, int Ni, int Klo, int Khi, float alpha) {
    auto dev = A_lo.device();
    auto stream = at::cuda::getCurrentCUDAStream(dev.index());
    auto D = torch::empty({Mi, Ni}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
    auto nv_args = mainloop_args<ForkNV, BSConfigNV>(A_lo, physSFA_lo, B_lo, physSFB_lo, Mi, Ni, Klo);
    auto mx_args = mainloop_args<ForkMX, BSConfigMX>(A_hi, physSFA_hi, B_hi, physSFB_hi, Mi, Ni, Khi);

    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {Mi, Ni, 1});
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {Mi, Ni, 1});
    using EC = typename Epilogue::ElementC;
    using ED = typename Epilogue::ElementD;

    typename Gemm::Arguments arguments;
    arguments.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    arguments.problem_shape = {Mi, Ni, Klo, 1};
    arguments.mainloop_nv = nv_args;
    arguments.mainloop_mx = mx_args;
    arguments.K_high = Khi;
    arguments.epilogue = { {alpha, 0.0f},
                           reinterpret_cast<EC const*>(D.data_ptr()), sC,
                           reinterpret_cast<ED*>(D.data_ptr()), sD };

    static Gemm gemm;
    static torch::Tensor ws_cache;
    static bool checked = false;
    size_t ws = Gemm::get_workspace_size(arguments);
    if (!ws_cache.defined() || (size_t)ws_cache.numel() < std::max<size_t>(ws, 1)) {
      ws_cache = torch::empty({(int64_t)std::max<size_t>(ws, 1)},
                              torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    }
    if (!checked) {
      auto st = gemm.can_implement(arguments);
      TORCH_CHECK(st == cutlass::Status::kSuccess, "smallm can_implement: ", cutlassGetStatusString(st));
      checked = true;
    }
    auto st = gemm.initialize(arguments, ws_cache.data_ptr(), stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "smallm initialize: ", cutlassGetStatusString(st));
    st = gemm.run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "smallm run: ", cutlassGetStatusString(st));
    // capture-safe host barrier (same contract as run_with_cached_ws)
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap == cudaStreamCaptureStatusNone)
      sm_check_cuda(cudaStreamSynchronize(stream), "smallm run sync");
    return D;
  }

  static std::string report() {
    return "tile=" + std::to_string(TM) + "x" + std::to_string(TN) +
           " nv=" + std::to_string(NVST) + " mx=" + std::to_string(MXST) +
           " smem=" + std::to_string((int)FusedK::SharedStorageSize) + "B";
  }
};

using Cfg0 = SmallMGemm<128, 128, 4, 2>;   // == shipping config (cross-check)
using Cfg1 = SmallMGemm<128, 64, 4, 2>;    // r24 small-M winner
using Cfg2 = SmallMGemm<128, 64, 6, 2>;    // deeper NVFP4 ring
using Cfg3 = SmallMGemm<128, 64, 6, 3>;    // deeper both

torch::Tensor fused_gemm_prepacked_cfg(torch::Tensor A_lo, torch::Tensor physSFA_lo,
                                       torch::Tensor B_lo, torch::Tensor physSFB_lo,
                                       torch::Tensor A_hi, torch::Tensor physSFA_hi,
                                       torch::Tensor B_hi, torch::Tensor physSFB_hi,
                                       int64_t M, int64_t N, int64_t K_lo, int64_t K_hi,
                                       double alpha, int64_t cfg) {
  TORCH_CHECK(K_lo % 16 == 0 && K_hi % 32 == 0, "K_lo%16, K_hi%32");
  A_lo = A_lo.contiguous(); B_lo = B_lo.contiguous();
  A_hi = A_hi.contiguous(); B_hi = B_hi.contiguous();
  physSFA_lo = physSFA_lo.contiguous(); physSFB_lo = physSFB_lo.contiguous();
  physSFA_hi = physSFA_hi.contiguous(); physSFB_hi = physSFB_hi.contiguous();
  int Mi = (int)M, Ni = (int)N, Klo = (int)K_lo, Khi = (int)K_hi;
  float a = (float)alpha;
  switch (cfg) {
    case 0: return Cfg0::run(A_lo, physSFA_lo, B_lo, physSFB_lo, A_hi, physSFA_hi, B_hi, physSFB_hi, Mi, Ni, Klo, Khi, a);
    case 1: return Cfg1::run(A_lo, physSFA_lo, B_lo, physSFB_lo, A_hi, physSFA_hi, B_hi, physSFB_hi, Mi, Ni, Klo, Khi, a);
    case 2: return Cfg2::run(A_lo, physSFA_lo, B_lo, physSFB_lo, A_hi, physSFA_hi, B_hi, physSFB_hi, Mi, Ni, Klo, Khi, a);
    case 3: return Cfg3::run(A_lo, physSFA_lo, B_lo, physSFB_lo, A_hi, physSFA_hi, B_hi, physSFB_hi, Mi, Ni, Klo, Khi, a);
    default: TORCH_CHECK(false, "cfg must be 0..3, got ", cfg);
  }
  return {};
}

static std::string smallm_report() {
  return "cfg0 " + Cfg0::report() + "\ncfg1 " + Cfg1::report() +
         "\ncfg2 " + Cfg2::report() + "\ncfg3 " + Cfg3::report();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_gemm_prepacked_cfg", &fused_gemm_prepacked_cfg,
        "dual-format fused GEMM at a small-M config (prepacked SF, cached ws, capture-safe); "
        "cfg 0:128x128/4/2 1:128x64/4/2 2:128x64/6/2 3:128x64/6/3");
  m.def("report", &smallm_report, "per-config tile/stages/smem report");
}
