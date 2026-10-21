// codesign_optc_kernel.hpp — round-23 Option-C register-feasibility fork.
//
// Fork of codesign_fused_kernel.hpp. Adds a TwoAccum template path: the two
// GEMM phases accumulate into SEPARATE FP32 fragments (accNV + accMX) that are
// summed just before the epilogue, instead of a single shared fragment. This is
// the register cost of "2x partial accumulators at 128x128" that round-20
// INFERRED (never built) would blow the 64K-reg/SM budget. Here we BUILD it and
// read the real ptxas register count / occupancy. Still SERIAL issue (256-thread
// cooperative mma, NV phase then MX phase) — it isolates the REGISTER question,
// not the concurrent-issue question (that needs per-WG 128-thread collectives;
// see codesign_optc_pp_* probe). Bit-exact: accNV + accMX == shared accumulate.
//
// PROTECTED files and the shipping kernel are untouched.
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/barrier.h"
#include "cute/tensor.hpp"
#include "cutlass/trace.h"
#include "cutlass/arch/grid_dependency_control.h"

namespace codesign {
using namespace cute;
using namespace cutlass;
using namespace cutlass::gemm;
using namespace cutlass::gemm::kernel;

template <
  class ProblemShape_,
  class CollectiveMainloopNV_,
  class CollectiveMainloopMX_,
  class CollectiveEpilogue_,
  class TileSchedulerTag_,
  bool  DualPhase_,
  bool  TwoAccum_ = false>
class CodesignOptCKernel {
public:
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  static constexpr bool DualPhase = DualPhase_;
  static constexpr bool TwoAccum  = TwoAccum_;

  // Phase-1 (NVFP4) collective drives all the shared derived types.
  using CollectiveMainloop   = CollectiveMainloopNV_;   // alias expected by GemmUniversalAdapter
  using CollectiveMainloopNV = CollectiveMainloopNV_;
  using CollectiveMainloopMX = CollectiveMainloopMX_;

  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArgumentsNV = typename CollectiveMainloopNV::Arguments;
  using MainloopParamsNV    = typename CollectiveMainloopNV::Params;
  using MainloopArgumentsMX = typename CollectiveMainloopMX::Arguments;
  using MainloopParamsMX    = typename CollectiveMainloopMX::Params;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(ArchTag::kMinComputeCapability >= 90);

  // Shared-accumulator precondition (the Path-A de-risk, now enforced in the glue):
  // NV and MX have DIFFERENT TiledMma (different SF vec size / block-scaled atom),
  // but their glue-owned accumulator FRAGMENT — partition_fragment_C(TiledMma,
  // take<0,2>(TileShape)) — must be IDENTICAL in element/size/shape so one shared
  // fragment can be written by both collectives' mma(). (STAGE-1 verified size 64,
  // float, 128x128.)
  using AccLayNV = decltype(partition_fragment_C(typename CollectiveMainloopNV::TiledMma{}, take<0,2>(TileShape{})).layout());
  using AccLayMX = decltype(partition_fragment_C(typename CollectiveMainloopMX::TiledMma{}, take<0,2>(TileShape{})).layout());
  static_assert(cute::is_same_v<typename CollectiveMainloopNV::ElementAccumulator,
                                typename CollectiveMainloopMX::ElementAccumulator>,
    "NV/MX accumulator element types differ — shared accumulator not type-safe.");
  static_assert(cute::size(AccLayNV{}) == cute::size(AccLayMX{}),
    "NV/MX accumulator fragment sizes differ — cannot share one fragment.");
  static_assert(cute::is_same_v<decltype(cute::shape(AccLayNV{})), decltype(cute::shape(AccLayMX{}))>,
    "NV/MX accumulator fragment shapes differ.");
  static_assert(cute::is_same_v<typename CollectiveMainloopNV::TileShape,
                                typename CollectiveMainloopMX::TileShape>,
    "NV/MX TileShape differ.");

  using TileSchedulerTag = TileSchedulerTag_;
  static constexpr uint32_t TileSchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  using TileScheduler = typename cutlass::gemm::kernel::detail::TileSchedulerSelector<
      TileSchedulerTag, ArchTag, TileShape, ClusterShape, TileSchedulerPipelineStageCount>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  static_assert(TileScheduler::IsDynamicPersistent == false,
    "codesign fork targets the STATIC persistent scheduler only.");

  static constexpr uint32_t NumMMAThreads          = size(TiledMma{});       // 256
  static constexpr uint32_t NumLoadWarpGroups      = 1;
  static constexpr uint32_t MaxThreadsPerBlock     = NumMMAThreads + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumMmaWarpGroups       = NumMMAThreads / NumThreadsPerWarpGroup;
  static constexpr uint32_t NumProducerThreadsNV   = CollectiveMainloopNV::NumProducerThreadEvents;
  static constexpr uint32_t NumProducerThreadsMX   = CollectiveMainloopMX::NumProducerThreadEvents;
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp;

  static constexpr int RegsPerThread =
    size<0>(TileShape{}) * size<1>(TileShape{}) / NumMMAThreads *
    sizeof(ElementAccumulator) / sizeof(uint32_t);
  static constexpr bool HeavyRegisterPressure = (RegsPerThread >= 128);
  static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
  static constexpr uint32_t MmaRegisterRequirement  = !HeavyRegisterPressure ? 232 : 240;

  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  using TileSchedulerStorage = typename TileScheduler::SharedStorage;

  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorageNV = typename CollectiveMainloopNV::PipelineStorage;
      using MainloopPipelineStorageMX = typename CollectiveMainloopMX::PipelineStorage;
      using EpiLoadPipelineStorage    = typename CollectiveEpilogue::PipelineStorage;
      alignas(16) MainloopPipelineStorageNV mainloop_nv;
      alignas(16) MainloopPipelineStorageMX mainloop_mx;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
    } pipelines;

    alignas(16) TileSchedulerStorage scheduler;

    struct TensorStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorStorageNV = typename CollectiveMainloopNV::TensorStorage;
      using MainloopTensorStorageMX = typename CollectiveMainloopMX::TensorStorage;
      using EpilogueTensorStorage   = typename CollectiveEpilogue::TensorStorage;
      EpilogueTensorStorage epilogue;
      // UNION: Phase 1 (NV) fully drains before Phase 2 (MX) overwrites — the two
      // never live simultaneously, so they reuse the SAME smem. SM120 only has
      // ~101 KB/block; this is what makes BLK_K=128 dual-format fit. Correctness
      // is protected by the two phase NamedBarriers in operator().
      union MainloopUnion {
        MainloopTensorStorageNV nv;
        MainloopTensorStorageMX mx;
        CUTLASS_DEVICE MainloopUnion() {}
      } mainloop;
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};             // {M, N, K_low, L}
    MainloopArgumentsNV mainloop_nv{};
    MainloopArgumentsMX mainloop_mx{};
    int K_high{0};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};             // {M, N, K_low, L}
    MainloopParamsNV mainloop_nv{};
    MainloopParamsMX mainloop_mx{};
    int mx_k_tile_count{0};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
  };

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    auto problem_shape = args.problem_shape;          // {M,N,Klow,L}
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    int max_active_clusters = args.hw_info.max_active_clusters;
    if (max_active_clusters <= 0) { max_active_clusters = 0; }
    KernelHardwareInfo hw_info = args.hw_info;
    hw_info.sm_count = sm_count;
    hw_info.max_active_clusters = max_active_clusters;

    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);

    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
      problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);

    // MX phase K tile count (K_high / BLK_K); 0 if single-phase.
    int blk_k = size<2>(TileShape{});
    int mx_k_tiles = DualPhase ? ((args.K_high + blk_k - 1) / blk_k) : 0;

    // MX mainloop params use the SAME M,N but K = K_high. Only built in DualPhase
    // (single-phase gate needs no MX data — default Params has no descriptor encode).
    MainloopParamsMX mx_params{};
    if constexpr (DualPhase) {
      auto mx_problem = args.problem_shape;
      get<2>(mx_problem) = args.K_high;
      mx_params = CollectiveMainloopMX::to_underlying_arguments(mx_problem, args.mainloop_mx, nullptr);
    }

    return {
      args.mode,
      problem_shape,
      CollectiveMainloopNV::to_underlying_arguments(args.problem_shape, args.mainloop_nv, nullptr),
      mx_params,
      mx_k_tiles,
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      hw_info,
      scheduler,
      workspace
    };
  }

  static bool can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && cute::rank(ProblemShape{}) == 4);
    if (!implementable) return false;
    implementable &= CollectiveMainloopNV::can_implement(args.problem_shape, args.mainloop_nv);
    if constexpr (DualPhase) {
      auto mx_problem = args.problem_shape;
      get<2>(mx_problem) = args.K_high;
      implementable &= CollectiveMainloopMX::can_implement(mx_problem, args.mainloop_mx);
    }
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler, args.hw_info);
    return implementable;
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);
    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups, 1);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);
    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
                       CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    static constexpr uint32_t NumAccumulatorMtxs = 1;
    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) return status;
    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles, NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    return status;
  }

  static dim3 get_grid_shape(Params const& params) {
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

#  if (defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) ||\
      CUDA_ARCH_CONDITIONAL_OR_FAMILY(1200) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1210))
#    define CD_ENABLE_SM90 1
#  endif
#if ! defined(CD_ENABLE_SM90)
    CUTE_INVALID_CONTROL_PATH("Arch conditional MMA used without targeting SM90+/SM120. Aborting.\n");
#else
    static_assert(NumMMAThreads == 256, "Cooperative kernel must have TiledMMA operating using 256 threads.");
    static_assert(size<0>(TileShape{}) >= 128, "Cooperative kernel requires Tile M >= 128.");

    enum class WarpGroupRole { Producer = 0, Consumer0 = 1, Consumer1 = 2 };
    enum class ProducerWarpRole { Mainloop = 0, Warp1 = 1, Epilogue = 2, MainloopAux = 3 };

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    int mma_thread_idx = thread_idx % NumMMAThreads;
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloopNV::prefetch_tma_descriptors(params.mainloop_nv);
      if constexpr (DualPhase) { CollectiveMainloopMX::prefetch_tma_descriptors(params.mainloop_mx); }
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);
    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();

    // Mainloop Load pipelines (NV + MX), same role assignment.
    using MainloopPipelineNV = typename CollectiveMainloopNV::MainloopPipeline;
    using MainloopPipelineMX = typename CollectiveMainloopMX::MainloopPipeline;
    typename MainloopPipelineNV::Params mlp_nv;
    typename MainloopPipelineMX::Params mlp_mx;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Mainloop) {
      mlp_nv.role = MainloopPipelineNV::ThreadCategory::Producer;
      mlp_mx.role = MainloopPipelineMX::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      mlp_nv.role = MainloopPipelineNV::ThreadCategory::Consumer;
      mlp_mx.role = MainloopPipelineMX::ThreadCategory::Consumer;
    }
    mlp_nv.is_leader = warp_group_thread_idx == 0;
    mlp_nv.num_consumers = NumMMAThreads;
    mlp_nv.num_producers = NumProducerThreadsNV;
    mlp_nv.transaction_bytes = params.mainloop_nv.tma_transaction_bytes;
    MainloopPipelineNV mainloop_pipeline_nv(shared_storage.pipelines.mainloop_nv, mlp_nv, ClusterShape{});
    mlp_mx.is_leader = warp_group_thread_idx == 0;
    mlp_mx.num_consumers = NumMMAThreads;
    mlp_mx.num_producers = NumProducerThreadsMX;
    mlp_mx.transaction_bytes = params.mainloop_mx.tma_transaction_bytes;
    MainloopPipelineMX mainloop_pipeline_mx(shared_storage.pipelines.mainloop_mx, mlp_mx, ClusterShape{});

    // Phase barriers (DualPhase only): protect the UNIONED mainloop smem so Phase-2
    // (MX) producer never overwrites union smem still being read by Phase-1 (NV)
    // consumers, and the next tile's Phase-1 producer never overwrites smem still
    // being read by this tile's Phase-2 consumers. Participants: the single mainloop
    // producer warp (32) + both consumer warpgroups (256) = 288 threads (a multiple
    // of the warp size). IDs 8/9 sit above the reserved Sm120MainloopBarrier (7).
    static constexpr uint32_t PhaseBarrierThreads = NumThreadsPerWarp + NumMMAThreads;  // 288
    // Use HIGH user ids (effective 14/15) to avoid colliding with the cooperative
    // epilogue's own user NamedBarriers (which start at user id 0 = effective 8).
    cutlass::arch::NamedBarrier phase_bar1(PhaseBarrierThreads, /*user id=*/6u);  // effective 14
    cutlass::arch::NamedBarrier phase_bar2(PhaseBarrierThreads, /*user id=*/7u);  // effective 15

    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Epilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumEpilogueLoadThreads;
    epi_load_pipeline_params.consumer_arv_count = NumMMAThreads;
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id = producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
    params_load_order_barrier.group_size = NumThreadsPerWarp;
    LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, params_load_order_barrier);

    typename CollectiveMainloopNV::PipelineState mainloop_pipe_consumer_state_nv;
    typename CollectiveMainloopMX::PipelineState mainloop_pipe_consumer_state_mx;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    PipelineState<CollectiveMainloopNV::DispatchPolicy::Stages> mainloop_pipe_producer_state_nv =
      cutlass::make_producer_start_state<MainloopPipelineNV>();
    PipelineState<CollectiveMainloopMX::DispatchPolicy::Stages> mainloop_pipe_producer_state_mx =
      cutlass::make_producer_start_state<MainloopPipelineMX>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    auto cluster_wait_fn = [] () {
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return [] () { cute::cluster_wait(); };
      } else {
        __syncthreads();
        return [] () {};
      }
    } ();

    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});

    TiledMma tiled_mma;
    auto blk_shape = TileShape{};

    TileScheduler scheduler{params.scheduler};
    typename TileScheduler::WorkTileInfo work_tile_info;

    CollectiveMainloopNV collective_mainloop_nv;
    CollectiveMainloopMX collective_mainloop_mx;

    auto load_inputs_nv = collective_mainloop_nv.load_init(problem_shape_MNKL, params.mainloop_nv);
    Tensor gA_mkl = get<0>(load_inputs_nv);
    Tensor gB_nkl = get<1>(load_inputs_nv);

    // MX load inputs built against the K_high problem shape (only in DualPhase).
    auto load_inputs_mx = [&]() {
      if constexpr (DualPhase) {
        auto mx_problem_MNKL = problem_shape_MNKL;
        get<2>(mx_problem_MNKL) = params.mx_k_tile_count * size<2>(TileShape{});
        return collective_mainloop_mx.load_init(mx_problem_MNKL, params.mainloop_mx);
      } else {
        return load_inputs_nv;  // placeholder, never used when !DualPhase
      }
    }();

    cluster_wait_fn();

    if (warp_group_role == WarpGroupRole::Producer) {
      work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        cutlass::arch::wait_on_dependent_grids();
        bool do_load_order_arrive = true;
        while (work_tile_info.is_valid()) {
          if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
            work_tile_info = get<0>(scheduler.fetch_next_work(work_tile_info));
            continue;
          }
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
          auto work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
          auto k_tile_iter = cute::make_coord_iterator(idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

          // Phase 1: NVFP4 load over K_low
          collective_mainloop_nv.load(
            params.mainloop_nv, mainloop_pipeline_nv, mainloop_pipe_producer_state_nv,
            load_inputs_nv, blk_coord, k_tile_iter, work_k_tile_count,
            lane_idx, block_rank_in_cluster, shared_storage.tensors.mainloop.nv);
          mainloop_pipe_producer_state_nv.advance(work_k_tile_count);
#ifdef CD_TRACE
          if (blockIdx.x==0 && warp_idx==0 && cute::elect_one_sync())
            printf("P nvload done kcount=%d mxcount=%d\n", (int)work_k_tile_count, (int)params.mx_k_tile_count);
#endif

          // Phase 2: MXFP8 load over K_high
          if constexpr (DualPhase) {
            phase_bar1.arrive_and_wait();   // NV reads done -> safe to overwrite union smem with MX
#ifdef CD_TRACE
            if (blockIdx.x==0 && warp_idx==0 && cute::elect_one_sync()) printf("P bar1 passed\n");
#endif
            Tensor gA_mkl_mx = get<0>(load_inputs_mx);
            auto k_tile_iter_mx = cute::make_coord_iterator(idx2crd(0, shape<3>(gA_mkl_mx)), shape<3>(gA_mkl_mx));
            collective_mainloop_mx.load(
              params.mainloop_mx, mainloop_pipeline_mx, mainloop_pipe_producer_state_mx,
              load_inputs_mx, blk_coord, k_tile_iter_mx, params.mx_k_tile_count,
              lane_idx, block_rank_in_cluster, shared_storage.tensors.mainloop.mx);
            mainloop_pipe_producer_state_mx.advance(params.mx_k_tile_count);
#ifdef CD_TRACE
            if (blockIdx.x==0 && warp_idx==0 && cute::elect_one_sync()) printf("P mxload done\n");
#endif
            phase_bar2.arrive_and_wait();   // MX reads (this tile) will complete before next tile's NV load
#ifdef CD_TRACE
            if (blockIdx.x==0 && warp_idx==0 && cute::elect_one_sync()) printf("P bar2 passed\n");
#endif
          }

          if (do_load_order_arrive) { load_order_barrier.arrive(); do_load_order_arrive = false; }
          work_tile_info = get<0>(scheduler.fetch_next_work(work_tile_info));
        }
        collective_mainloop_nv.load_tail(mainloop_pipeline_nv, mainloop_pipe_producer_state_nv);
        if constexpr (DualPhase) { collective_mainloop_mx.load_tail(mainloop_pipeline_mx, mainloop_pipe_producer_state_mx); }
      }
      else if (producer_warp_role == ProducerWarpRole::Epilogue && is_epi_load_needed) {
        cutlass::arch::wait_on_dependent_grids();
        if (!TileScheduler::requires_separate_reduction(params.scheduler) && work_tile_info.is_valid()) {
          load_order_barrier.wait();
        }
        CollectiveEpilogue collective_epilogue_prod(params.epilogue, shared_storage.tensors.epilogue);
        while (work_tile_info.is_valid()) {
          if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
            epi_load_pipe_producer_state = collective_epilogue_prod.load(
              epi_load_pipeline, epi_load_pipe_producer_state, problem_shape_MNKL, blk_shape, blk_coord,
              tiled_mma, lane_idx, shared_storage.tensors.epilogue, work_tile_info.reduction_subtile_idx());
          }
          work_tile_info = get<0>(scheduler.fetch_next_work(work_tile_info));
        }
        collective_epilogue_prod.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
      }
    }
    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      CollectiveEpilogue collective_epilogue_cons(params.epilogue, shared_storage.tensors.epilogue);
      bool do_store_tail = false;
      while (work_tile_info.is_valid()) {
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
        auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);

        auto accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));   // SHARED fragment
        // Option-C register probe: a SECOND accumulator for the MX phase, held live
        // across both phases -> the CTA pays 2x FP32 accumulator registers
        // (128x128/256 = 64 floats/thread EACH -> 128 floats/thread total).
        auto accMX2 = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));
        if constexpr (TwoAccum) { cute::clear(accMX2); }
        if (TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
          // Phase 1: NVFP4 mma (ClearAccum=true -> clears then accumulates over K_low)
          collective_mainloop_nv.mma(
            mainloop_pipeline_nv, mainloop_pipe_consumer_state_nv, accumulators, work_k_tile_count,
            mma_thread_idx, shared_storage.tensors.mainloop.nv, params.mainloop_nv, blk_coord);
          collective_mainloop_nv.mma_tail(mainloop_pipeline_nv, mainloop_pipe_consumer_state_nv, work_k_tile_count);
          mainloop_pipe_consumer_state_nv.advance(work_k_tile_count);
#ifdef CD_TRACE
          if (blockIdx.x==0 && thread_idx==128) printf("C nvmma done\n");
#endif

          // Phase 2: MXFP8 mma. TwoAccum -> into the SEPARATE accMX2 (its collective is
          // ClearAccum=true); else into the shared `accumulators` (ClearAccum=false).
          if constexpr (DualPhase) {
            phase_bar1.arrive_and_wait();   // signal NV reads done (pairs with producer phase_bar1)
#ifdef CD_TRACE
            if (blockIdx.x==0 && thread_idx==128) printf("C bar1 passed\n");
#endif
            if constexpr (TwoAccum) {
              collective_mainloop_mx.mma(
                mainloop_pipeline_mx, mainloop_pipe_consumer_state_mx, accMX2, params.mx_k_tile_count,
                mma_thread_idx, shared_storage.tensors.mainloop.mx, params.mainloop_mx, blk_coord);
            } else {
              collective_mainloop_mx.mma(
                mainloop_pipeline_mx, mainloop_pipe_consumer_state_mx, accumulators, params.mx_k_tile_count,
                mma_thread_idx, shared_storage.tensors.mainloop.mx, params.mainloop_mx, blk_coord);
            }
            collective_mainloop_mx.mma_tail(mainloop_pipeline_mx, mainloop_pipe_consumer_state_mx, params.mx_k_tile_count);
            mainloop_pipe_consumer_state_mx.advance(params.mx_k_tile_count);
#ifdef CD_TRACE
            if (blockIdx.x==0 && thread_idx==128) printf("C mxmma done\n");
#endif
            phase_bar2.arrive_and_wait();   // signal MX reads done (pairs with producer phase_bar2)
#ifdef CD_TRACE
            if (blockIdx.x==0 && thread_idx==128) printf("C bar2 passed\n");
#endif
            // Sum the two partials just before the epilogue (bit-exact vs shared accum).
            if constexpr (TwoAccum) {
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(accumulators); ++i) { accumulators(i) += accMX2(i); }
            }
          }
        }

        int consumer_warp_group_idx = canonical_warp_group_idx() - NumLoadWarpGroups;
        TileScheduler::fixup(params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, consumer_warp_group_idx);

        if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
          auto [els_next, esp_next] = collective_epilogue_cons.store(
            epi_load_pipeline, epi_load_pipe_consumer_state, epi_store_pipeline, epi_store_pipe_producer_state,
            problem_shape_MNKL, blk_shape, blk_coord, accumulators, tiled_mma, mma_thread_idx,
            shared_storage.tensors.epilogue, work_tile_info.reduction_subtile_idx());
          epi_load_pipe_consumer_state = els_next;
          epi_store_pipe_producer_state = esp_next;
          do_store_tail = true;
        }
        work_tile_info = get<0>(scheduler.fetch_next_work(work_tile_info));
      }
      if (do_store_tail) {
        collective_epilogue_cons.store_tail(epi_load_pipeline, epi_load_pipe_consumer_state, epi_store_pipeline, epi_store_pipe_producer_state);
      }
    }
#endif
  }
};

} // namespace codesign
