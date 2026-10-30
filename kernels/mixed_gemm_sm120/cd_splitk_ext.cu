// cd_splitk_ext.cu — vllm-perf round-3 lever 1: SPLIT-K / STREAM-K dual-format
// fused GEMM for the latency-bound small-M (decode b8/b16) regime.
//
// r2 verdict: at M<=16 the fused GEMM is structurally latency-bound — the
// serial K mainloop (24..64 union k-tiles) runs on only N/128..N/64 CTAs and
// sits 2-3.4x above the weight-byte floor; the SM120-viable tile-config space
// is exhausted. This ext splits the K iteration space across CTAs with the
// CUTLASS SM90 stream-K scheduler through a forked glue
// (codesign_splitk_kernel.hpp) that maps ONE union k-tile domain
// [NV k-tiles | MX k-tiles] onto the two phases and reduces the shared FP32
// accumulator ONCE via the scheduler's stock workspace fixup.
//
// Entry contract matches cd_smallm_ext.fused_gemm_prepacked_cfg (prepacked
// interleaved SF in, cached workspace, capture-safe): the SF atom layout
// depends only on SFVecSize, so the same physSFA/physSFB buffers feed every
// config here.
//
//   fused_gemm_prepacked_sk(..., alpha, cfg, splits, decomp)
//     cfg    0: 128x128 NV4/MX2   1: 128x64 NV4/MX2
//     decomp 0: Heuristic  1: SplitK(splits)  2: StreamK  3: DataParallel
//            (DataParallel == serial per-tile K -> bit-identical cross-check)
//     reduction mode is always Deterministic (run-to-run stable).
//
// Workspace note: stream-K needs its reduction/barrier workspace re-zeroed
// before every launch; gemm.initialize() (called per run) enqueues that memset
// on the stream, so under CUDA-graph capture it is captured and replays with
// the kernel — capture-safe by construction.
//
// PROTECTED mainloop headers are #included UNCHANGED (config-only
// instantiation, same pattern as cd_smallm_ext.cu).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
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

#include "codesign_sm120_bs_mma.hpp"
#include "codesign_splitk_kernel.hpp"

using namespace cute;

static inline void sk_check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

namespace {
template <class Policy> struct SkPolicyParts;
template <int S, int Sch, class CS, class KS>
struct SkPolicyParts<cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled<S, Sch, CS, KS>> {
  static constexpr int Stages = S;
  static constexpr int SchedStages = Sch;
  using ClusterShape = CS;
  using KernelSchedule = KS;
};
}  // namespace

template <class Stock, bool ClearAccum>
using SkForkMma = cutlass::gemm::collective::CodesignBlockScaledMma<
    SkPolicyParts<typename Stock::DispatchPolicy>::Stages,
    SkPolicyParts<typename Stock::DispatchPolicy>::SchedStages,
    typename SkPolicyParts<typename Stock::DispatchPolicy>::ClusterShape,
    typename SkPolicyParts<typename Stock::DispatchPolicy>::KernelSchedule,
    typename Stock::TileShape,
    typename Stock::ElementPairA, typename Stock::StridePairA,
    typename Stock::ElementPairB, typename Stock::StridePairB,
    typename Stock::TiledMma,
    typename Stock::GmemTiledCopyPairA, typename Stock::SmemLayoutAtomsA, typename Stock::SmemCopyAtomsA, typename Stock::TransformA,
    typename Stock::GmemTiledCopyPairB, typename Stock::SmemLayoutAtomsB, typename Stock::SmemCopyAtomsB, typename Stock::TransformB,
    ClearAccum>;

template <int TM, int TN, int NVST, int MXST>
struct SplitKGemm {
  template <class ElementAB, int Alignment, int Stages>
  struct Stock {
    using ElementD = cutlass::bfloat16_t;
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
  using ForkNV = SkForkMma<typename NVbuild::CollectiveMainloop, /*ClearAccum=*/true>;
  using ForkMX = SkForkMma<typename MXbuild::CollectiveMainloop, /*ClearAccum=*/false>;
  using FusedK = codesign::CodesignSplitKKernel<
      Shape<int,int,int,int>, ForkNV, ForkMX, Epilogue, /*DualPhase=*/true>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<FusedK>;
  using Scheduler = typename FusedK::TileScheduler;
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
                           int Mi, int Ni, int Klo, int Khi, float alpha,
                           int splits, int decomp) {
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
    arguments.scheduler.splits = splits;
    arguments.scheduler.reduction_mode = Scheduler::ReductionMode::Deterministic;
    switch (decomp) {
      case 0: arguments.scheduler.decomposition_mode = Scheduler::DecompositionMode::Heuristic; break;
      case 1: arguments.scheduler.decomposition_mode = Scheduler::DecompositionMode::SplitK; break;
      case 2: arguments.scheduler.decomposition_mode = Scheduler::DecompositionMode::StreamK; break;
      case 3: arguments.scheduler.decomposition_mode = Scheduler::DecompositionMode::DataParallel; break;
      default: TORCH_CHECK(false, "decomp must be 0..3, got ", decomp);
    }

    static Gemm gemm;
    static torch::Tensor ws_cache;
    size_t ws = Gemm::get_workspace_size(arguments);
    if (!ws_cache.defined() || (size_t)ws_cache.numel() < std::max<size_t>(ws, 1)) {
      ws_cache = torch::empty({(int64_t)std::max<size_t>(ws, 1)},
                              torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    }
    // can_implement every call: scheduler legality depends on (shape, splits,
    // decomp), unlike the fixed-decomposition exts. Host-side only, cheap.
    auto st = gemm.can_implement(arguments);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "splitk can_implement: ", cutlassGetStatusString(st));
    st = gemm.initialize(arguments, ws_cache.data_ptr(), stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "splitk initialize: ", cutlassGetStatusString(st));
    st = gemm.run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "splitk run: ", cutlassGetStatusString(st));
    // capture-safe host barrier (same contract as run_with_cached_ws)
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap == cudaStreamCaptureStatusNone)
      sk_check_cuda(cudaStreamSynchronize(stream), "splitk run sync");
    return D;
  }

  static std::string report() {
    return "tile=" + std::to_string(TM) + "x" + std::to_string(TN) +
           " nv=" + std::to_string(NVST) + " mx=" + std::to_string(MXST) +
           " smem=" + std::to_string((int)FusedK::SharedStorageSize) + "B";
  }
};

using SkCfg0 = SplitKGemm<128, 128, 4, 2>;   // shipping tile
using SkCfg1 = SplitKGemm<128, 64, 4, 2>;    // r2 small-M tile

torch::Tensor fused_gemm_prepacked_sk(torch::Tensor A_lo, torch::Tensor physSFA_lo,
                                      torch::Tensor B_lo, torch::Tensor physSFB_lo,
                                      torch::Tensor A_hi, torch::Tensor physSFA_hi,
                                      torch::Tensor B_hi, torch::Tensor physSFB_hi,
                                      int64_t M, int64_t N, int64_t K_lo, int64_t K_hi,
                                      double alpha, int64_t cfg, int64_t splits, int64_t decomp) {
  TORCH_CHECK(K_lo % 16 == 0 && K_hi % 32 == 0, "K_lo%16, K_hi%32");
  TORCH_CHECK(K_hi > 0, "splitk ext is dual-phase only (K_hi > 0)");
  A_lo = A_lo.contiguous(); B_lo = B_lo.contiguous();
  A_hi = A_hi.contiguous(); B_hi = B_hi.contiguous();
  physSFA_lo = physSFA_lo.contiguous(); physSFB_lo = physSFB_lo.contiguous();
  physSFA_hi = physSFA_hi.contiguous(); physSFB_hi = physSFB_hi.contiguous();
  int Mi = (int)M, Ni = (int)N, Klo = (int)K_lo, Khi = (int)K_hi;
  float a = (float)alpha;
  switch (cfg) {
    case 0: return SkCfg0::run(A_lo, physSFA_lo, B_lo, physSFB_lo, A_hi, physSFA_hi, B_hi, physSFB_hi, Mi, Ni, Klo, Khi, a, (int)splits, (int)decomp);
    case 1: return SkCfg1::run(A_lo, physSFA_lo, B_lo, physSFB_lo, A_hi, physSFA_hi, B_hi, physSFB_hi, Mi, Ni, Klo, Khi, a, (int)splits, (int)decomp);
    default: TORCH_CHECK(false, "cfg must be 0..1, got ", cfg);
  }
  return {};
}

static std::string splitk_report() {
  return "sk0 " + SkCfg0::report() + "\nsk1 " + SkCfg1::report();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_gemm_prepacked_sk", &fused_gemm_prepacked_sk,
        "dual-format fused GEMM with stream-K/split-K K-partitioning over the union "
        "k-tile domain (prepacked SF, cached ws, capture-safe); cfg 0:128x128 1:128x64; "
        "decomp 0:heuristic 1:splitk(splits) 2:streamk 3:data-parallel(serial cross-check)");
  m.def("report", &splitk_report, "per-config tile/stages/smem report");
}
