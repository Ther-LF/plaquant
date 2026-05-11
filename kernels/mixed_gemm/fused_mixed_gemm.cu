/***************************************************************************************************
 * Fused Mixed-Precision GEMM — TRUE single kernel (CuTe native)
 *
 * One __global__ that performs:
 *   Phase 1: TMA load A_main/B_main → SMEM → WGMMA U8S8 → acc_main (INT32 registers)
 *   Phase 2: TMA load A_high/B_high → SMEM → WGMMA U8S8 → acc_high (INT32 registers)
 *   Epilogue: in-register FP32 dequant + combine → store FP16 to global
 *
 * Written using CuTe primitives directly (no CUTLASS CollectiveMainloop wrapper).
 * This gives us full control over the two-phase mainloop.
 *
 * v1: Simple synchronous approach (no async pipeline overlap between phases).
 *     Each K-tile: TMA load → barrier wait → WGMMA → next tile.
 *
 * Target: NVIDIA H20 (SM90a)
 **************************************************************************************************/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/arch/cluster_sm90.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

#define CUTLASS_ARCH_MMA_SM90_SUPPORTED 1

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// =============================================================================
// Configuration
// =============================================================================

using ElementA = uint8_t;
using ElementB = int8_t;
using ElementAccum = int32_t;
using ElementOutput = cutlass::half_t;
using ElementCompute = float;

using TileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;
constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentD = 8;

// =============================================================================
// We still use CUTLASS to build a complete Gemm type — not to run it,
// but to get its SharedStorage layout, TiledMma type, and TMA setup.
// Then our custom kernel will mimic its operator() but with two phases.
// =============================================================================

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, AlignmentA,
    ElementB, cutlass::layout::ColumnMajor, AlignmentB,
    ElementAccum,
    TileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::KernelTmaWarpSpecialized
>::CollectiveOp;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    ElementOutput, cutlass::layout::RowMajor, AlignmentD,
    ElementOutput, cutlass::layout::RowMajor, AlignmentD,
    cutlass::epilogue::TmaWarpSpecialized
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

// Extract types we need
using TiledMma = typename CollectiveMainloop::TiledMma;
using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
using PipelineState_ = cutlass::PipelineState<CollectiveMainloop::DispatchPolicy::Stages>;
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;

// SharedStorage type from CUTLASS (proven correct layout)
using KernelSharedStorage = typename GemmKernel::SharedStorage;
using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;

static constexpr uint32_t NumThreadsPerWarpGroup_ = cutlass::NumThreadsPerWarpGroup;  // 128
static constexpr uint32_t MaxThreadsPerBlock_ = GemmKernel::MaxThreadsPerBlock;        // 256
static constexpr int Stages = CollectiveMainloop::DispatchPolicy::Stages;

// =============================================================================
// Fused kernel parameters
// =============================================================================

struct FusedKernelParams {
    int M, N, K_main, K_high;

    // Mainloop params for each phase (TMA descriptors + strides)
    typename CollectiveMainloop::Params mainloop_main;
    typename CollectiveMainloop::Params mainloop_high;

    // Dequant parameters
    ElementOutput const* s_x_m;
    ElementOutput const* s_w_m;
    ElementOutput const* neg_zero_m;
    float const* colsum_m;
    ElementOutput const* s_x_h;
    ElementOutput const* s_w_h;
    ElementOutput const* neg_zero_h;
    float const* colsum_h;

    // Output
    ElementOutput* D;
    int ldd;  // leading dimension of D (= N)
};

// =============================================================================
// The fused kernel — mirrors GemmKernel::operator() structure
// =============================================================================

__global__ void __launch_bounds__(MaxThreadsPerBlock_, 1)
fused_gemm_kernel(FusedKernelParams params) {
    using namespace cute;
    using namespace cutlass;

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)

    // ---- Warp role setup (same as CUTLASS GemmKernel) ----
    enum class WarpGroupRole { Producer = 0, Consumer = 1 };
    enum class ProducerWarpRole { Mainloop = 0, Warp1 = 1, Warp2 = 2, Warp3 = 3 };

    int thread_idx = int(threadIdx.x);
    int warp_idx_in_warp_group = (thread_idx / 32) % 4;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup_;
    auto warp_group_role = WarpGroupRole(thread_idx / NumThreadsPerWarpGroup_);
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();

    // ---- Shared memory ----
    extern __shared__ char smem_buf[];

    // We only use the mainloop portion of shared storage (reused between phases)
    // Layout: TensorStorage (smem_A + smem_B) + PipelineStorage (barriers)
    struct FusedSharedStorage {
        MainloopTensorStorage tensors;
        MainloopPipelineStorage pipeline;
    };
    FusedSharedStorage& shared_storage = *reinterpret_cast<FusedSharedStorage*>(smem_buf);

    // ---- Pipeline setup ----
    typename MainloopPipeline::Params pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Mainloop) {
        pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
        pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumThreadsPerWarpGroup_;
    pipeline_params.transaction_bytes = params.mainloop_main.tma_transaction_bytes;

    MainloopPipeline mainloop_pipeline(shared_storage.pipeline, pipeline_params, ClusterShape_MNK{});

    // ---- TMA descriptor prefetch ----
    if (thread_idx == 0) {
        CollectiveMainloop::prefetch_tma_descriptors(params.mainloop_main);
    }

    // ---- Block coordinates ----
    int m_coord = int(blockIdx.x);
    int n_coord = int(blockIdx.y);
    auto blk_coord = make_coord(m_coord, n_coord, _, 0);

    // ---- Mainloop collective (stateless helper) ----
    CollectiveMainloop collective_mainloop;
    TiledMma tiled_mma;

    // ---- Problem shapes ----
    auto problem_shape_main = make_shape(params.M, params.N, params.K_main, 1);
    auto problem_shape_high = make_shape(params.M, params.N, params.K_high, 1);

    // ---- Compute tile counts ----
    int k_tile_count_main = cute::ceil_div(params.K_main, get<2>(TileShape_MNK{}));
    int k_tile_count_high = cute::ceil_div(params.K_high, get<2>(TileShape_MNK{}));

    // Ensure cluster sync before mainloop
    __syncthreads();

    // ==================================================================
    // PHASE 1: Main GEMM
    // ==================================================================
    auto acc_main = partition_fragment_C(tiled_mma, take<0,2>(TileShape_MNK{}));
    clear(acc_main);

    {
        auto load_inputs = collective_mainloop.load_init(problem_shape_main, params.mainloop_main);
        auto k_tile_iter = cute::make_coord_iterator(shape<3>(get<0>(load_inputs)));

        auto pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
        PipelineState_ pipe_consumer_state;

        if (warp_group_role == WarpGroupRole::Producer) {
            if (producer_warp_role == ProducerWarpRole::Mainloop) {
                cutlass::arch::wait_on_dependent_grids();
                collective_mainloop.load(
                    params.mainloop_main,
                    mainloop_pipeline,
                    pipe_producer_state,
                    load_inputs,
                    blk_coord,
                    k_tile_iter, k_tile_count_main,
                    thread_idx % 32,
                    cute::block_rank_in_cluster(),
                    shared_storage.tensors);
                pipe_producer_state.advance(k_tile_count_main);
                collective_mainloop.load_tail(mainloop_pipeline, pipe_producer_state);
            }
        }
        else if (warp_group_role == WarpGroupRole::Consumer) {
            collective_mainloop.mma(
                mainloop_pipeline,
                pipe_consumer_state,
                acc_main,
                k_tile_count_main,
                warp_group_thread_idx,
                shared_storage.tensors,
                params.mainloop_main);
            collective_mainloop.mma_tail(mainloop_pipeline, pipe_consumer_state, k_tile_count_main);
        }
    }

    // Sync between phases: all threads must finish phase 1
    __syncthreads();

    // ==================================================================
    // PHASE 2: High GEMM
    // ==================================================================
    auto acc_high = partition_fragment_C(tiled_mma, take<0,2>(TileShape_MNK{}));
    clear(acc_high);

    if (k_tile_count_high > 0) {
        // Re-initialize pipeline for phase 2 (reset barriers)
        pipeline_params.transaction_bytes = params.mainloop_high.tma_transaction_bytes;
        MainloopPipeline mainloop_pipeline2(shared_storage.pipeline, pipeline_params, ClusterShape_MNK{});

        // Prefetch phase 2 TMA descriptors
        if (thread_idx == 0) {
            CollectiveMainloop::prefetch_tma_descriptors(params.mainloop_high);
        }
        __syncthreads();

        auto load_inputs_h = collective_mainloop.load_init(problem_shape_high, params.mainloop_high);
        auto k_tile_iter_h = cute::make_coord_iterator(shape<3>(get<0>(load_inputs_h)));

        auto pipe_producer_state2 = cutlass::make_producer_start_state<MainloopPipeline>();
        PipelineState_ pipe_consumer_state2;

        if (warp_group_role == WarpGroupRole::Producer) {
            if (producer_warp_role == ProducerWarpRole::Mainloop) {
                collective_mainloop.load(
                    params.mainloop_high,
                    mainloop_pipeline2,
                    pipe_producer_state2,
                    load_inputs_h,
                    blk_coord,
                    k_tile_iter_h, k_tile_count_high,
                    thread_idx % 32,
                    cute::block_rank_in_cluster(),
                    shared_storage.tensors);
                pipe_producer_state2.advance(k_tile_count_high);
                collective_mainloop.load_tail(mainloop_pipeline2, pipe_producer_state2);
            }
        }
        else if (warp_group_role == WarpGroupRole::Consumer) {
            collective_mainloop.mma(
                mainloop_pipeline2,
                pipe_consumer_state2,
                acc_high,
                k_tile_count_high,
                warp_group_thread_idx,
                shared_storage.tensors,
                params.mainloop_high);
            collective_mainloop.mma_tail(mainloop_pipeline2, pipe_consumer_state2, k_tile_count_high);
        }
    }

    // ==================================================================
    // EPILOGUE: In-register dequant + combine + store
    // Only consumer warp group has the accumulators
    // ==================================================================
    if (warp_group_role == WarpGroupRole::Consumer) {
        int tile_m_start = m_coord * get<0>(TileShape_MNK{});
        int tile_n_start = n_coord * get<1>(TileShape_MNK{});

        // Map accumulator elements to (M,N) coordinates using identity tensor
        auto cD = make_identity_tensor(make_shape(get<0>(TileShape_MNK{}), get<1>(TileShape_MNK{})));
        auto thread_mma = tiled_mma.get_slice(warp_group_thread_idx);
        auto tCcD = thread_mma.partition_C(cD);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc_main); ++i) {
            auto coord = tCcD(i);
            int m_local = get<0>(coord);
            int n_local = get<1>(coord);
            int m_global = tile_m_start + m_local;
            int n_global = tile_n_start + n_local;

            if (m_global >= params.M || n_global >= params.N) continue;

            // Dequant main
            float a_main_f = static_cast<float>(acc_main(i));
            float sx_m = float(params.s_x_m[m_global]);
            float sw_m = float(params.s_w_m[n_global]);
            float nz_m = float(params.neg_zero_m[m_global]);
            float cs_m = params.colsum_m[n_global];
            float out_main = sx_m * sw_m * (a_main_f + nz_m * cs_m);

            // Dequant high
            float a_high_f = static_cast<float>(acc_high(i));
            float sx_h = float(params.s_x_h[m_global]);
            float sw_h = float(params.s_w_h[n_global]);
            float nz_h = float(params.neg_zero_h[m_global]);
            float cs_h = params.colsum_h[n_global];
            float out_high = sx_h * sw_h * (a_high_f + nz_h * cs_h);

            // Store combined result
            params.D[m_global * params.ldd + n_global] = ElementOutput(out_main + out_high);
        }
    }

#else
    if (threadIdx.x == 0) {
        printf("ERROR: SM90a not supported\n");
    }
#endif
}

// =============================================================================
// Host-side launch
// =============================================================================

torch::Tensor fused_mixed_gemm(
    torch::Tensor A_main,
    torch::Tensor B_main,
    torch::Tensor A_high,
    torch::Tensor B_high,
    torch::Tensor s_x_m,
    torch::Tensor s_w_m,
    torch::Tensor neg_zero_m,
    torch::Tensor colsum_m,
    torch::Tensor s_x_h,
    torch::Tensor s_w_h,
    torch::Tensor neg_zero_h,
    torch::Tensor colsum_h)
{
    TORCH_CHECK(A_main.is_cuda() && B_main.is_cuda() && A_high.is_cuda() && B_high.is_cuda(),
                "All inputs must be on CUDA");
    TORCH_CHECK(A_main.dtype() == torch::kUInt8, "A_main must be uint8");
    TORCH_CHECK(B_main.dtype() == torch::kInt8, "B_main must be int8");
    TORCH_CHECK(A_high.dtype() == torch::kUInt8, "A_high must be uint8");
    TORCH_CHECK(B_high.dtype() == torch::kInt8, "B_high must be int8");
    TORCH_CHECK(A_main.is_contiguous() && B_main.is_contiguous(), "main must be contiguous");
    TORCH_CHECK(A_high.is_contiguous() && B_high.is_contiguous(), "high must be contiguous");

    auto A_main_shape = A_main.sizes().vec();
    A_main = A_main.reshape({-1, A_main.size(-1)});
    A_high = A_high.reshape({-1, A_high.size(-1)});

    int M = A_main.size(0);
    int K_main = A_main.size(1);
    int K_high = A_high.size(1);
    int N = B_main.size(0);

    TORCH_CHECK(K_main == B_main.size(1), "K_main mismatch");
    TORCH_CHECK(K_high == B_high.size(1), "K_high mismatch");
    TORCH_CHECK(A_high.size(0) == M, "M mismatch");
    TORCH_CHECK(B_high.size(0) == N, "N mismatch");

    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A_main.device()));
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Build mainloop params (creates TMA descriptors on host)
    auto problem_shape_main = cute::make_shape(M, N, K_main, 1);
    auto problem_shape_high = cute::make_shape(M, N, K_high, 1);

    typename CollectiveMainloop::Arguments mainloop_args_main{
        reinterpret_cast<ElementA const*>(A_main.data_ptr()),
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K_main, 1)),
        reinterpret_cast<ElementB const*>(B_main.data_ptr()),
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K_main, 1)),
    };
    typename CollectiveMainloop::Arguments mainloop_args_high{
        reinterpret_cast<ElementA const*>(A_high.data_ptr()),
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K_high, 1)),
        reinterpret_cast<ElementB const*>(B_high.data_ptr()),
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K_high, 1)),
    };

    auto mainloop_params_main = CollectiveMainloop::to_underlying_arguments(
        problem_shape_main, mainloop_args_main, nullptr);
    auto mainloop_params_high = CollectiveMainloop::to_underlying_arguments(
        problem_shape_high, mainloop_args_high, nullptr);

    // Build kernel params
    FusedKernelParams kernel_params;
    kernel_params.M = M;
    kernel_params.N = N;
    kernel_params.K_main = K_main;
    kernel_params.K_high = K_high;
    kernel_params.mainloop_main = mainloop_params_main;
    kernel_params.mainloop_high = mainloop_params_high;
    kernel_params.s_x_m = reinterpret_cast<ElementOutput const*>(s_x_m.data_ptr());
    kernel_params.s_w_m = reinterpret_cast<ElementOutput const*>(s_w_m.data_ptr());
    kernel_params.neg_zero_m = reinterpret_cast<ElementOutput const*>(neg_zero_m.data_ptr());
    kernel_params.colsum_m = reinterpret_cast<float const*>(colsum_m.data_ptr());
    kernel_params.s_x_h = reinterpret_cast<ElementOutput const*>(s_x_h.data_ptr());
    kernel_params.s_w_h = reinterpret_cast<ElementOutput const*>(s_w_h.data_ptr());
    kernel_params.neg_zero_h = reinterpret_cast<ElementOutput const*>(neg_zero_h.data_ptr());
    kernel_params.colsum_h = reinterpret_cast<float const*>(colsum_h.data_ptr());
    kernel_params.D = reinterpret_cast<ElementOutput*>(output.data_ptr());
    kernel_params.ldd = N;

    // Grid/block config (same as CUTLASS GemmKernel)
    int grid_m = cute::ceil_div(M, get<0>(TileShape_MNK{}));
    int grid_n = cute::ceil_div(N, get<1>(TileShape_MNK{}));
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(MaxThreadsPerBlock_, 1, 1);

    // Shared memory = MainloopTensorStorage + PipelineStorage
    int smem_size = static_cast<int>(sizeof(MainloopTensorStorage) + sizeof(MainloopPipelineStorage));

    // Set max dynamic shared memory for this kernel
    auto err = cudaFuncSetAttribute(fused_gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    TORCH_CHECK(err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(err));

    // Launch
    fused_gemm_kernel<<<grid, block, smem_size, stream>>>(kernel_params);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fused_gemm_kernel launch failed: ", cudaGetErrorString(err));

    if (A_main_shape.size() == 3) {
        output = output.reshape({A_main_shape[0], A_main_shape[1], N});
    }
    return output;
}

#else
torch::Tensor fused_mixed_gemm(
    torch::Tensor A_main, torch::Tensor B_main,
    torch::Tensor A_high, torch::Tensor B_high,
    torch::Tensor s_x_m, torch::Tensor s_w_m, torch::Tensor neg_zero_m, torch::Tensor colsum_m,
    torch::Tensor s_x_h, torch::Tensor s_w_h, torch::Tensor neg_zero_h, torch::Tensor colsum_h) {
    TORCH_CHECK(false, "SM90 not supported"); return {};
}
#endif

// Binding is in mixed_gemm.cu PYBIND11_MODULE via forward declaration
