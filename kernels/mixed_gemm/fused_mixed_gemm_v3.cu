/***************************************************************************************************
 * Fused Mixed-Precision GEMM v3 — Single accumulator with workspace
 *
 * Fixes v2's register pressure by using only ONE set of accumulator registers:
 *   Phase 1: WGMMA K_main tiles → acc → dequant main → store FP16 to workspace
 *   Phase 2: clear(acc), WGMMA K_high tiles → acc → dequant high → load workspace → combine → store
 *
 * This halves register pressure vs v2's dual-accumulator approach.
 * Trade-off: one extra M×N FP16 read+write to workspace (global memory).
 *
 * Target: NVIDIA H20 (SM90a)
 **************************************************************************************************/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

#define CUTLASS_ARCH_MMA_SM90_SUPPORTED 1

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// =============================================================================
// Configuration (same as v1/v2)
// =============================================================================

using TileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;
using ElementA = uint8_t;
using ElementB = int8_t;
using ElementAccum = int32_t;
using ElementOutput = cutlass::half_t;
using ElementCompute = float;
constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentD = 8;

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

using TiledMma = typename CollectiveMainloop::TiledMma;
using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;
using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;

static constexpr int K_PIPE_MMAS = 1;
static constexpr uint32_t MaxThreadsPerBlock_v3 = GemmKernel::MaxThreadsPerBlock;

// =============================================================================
// Params
// =============================================================================

struct FusedV3Params {
    int M, N, K_main, K_high;
    typename CollectiveMainloop::Params mainloop_main;
    typename CollectiveMainloop::Params mainloop_high;
    ElementOutput const* s_x_m;
    ElementOutput const* s_w_m;
    ElementOutput const* neg_zero_m;
    float const* colsum_m;
    ElementOutput const* s_x_h;
    ElementOutput const* s_w_h;
    ElementOutput const* neg_zero_h;
    float const* colsum_h;
    ElementOutput* D;           // final output (M, N) fp16
    ElementOutput* workspace;   // intermediate buffer (M, N) fp16
    int ldd;
};

// =============================================================================
// Helper: inlined mainloop (producer + consumer for one phase)
// Returns via the accumulator reference.
// =============================================================================

// Inlined producer load loop — only executed by producer warp 0's elected lane
#define INLINE_PRODUCER_LOAD(pipeline_ref, state_ref, tma_a, tma_b, tAgA_ref, tAsA_ref, tBgB_ref, tBsB_ref, k_count) \
    do { \
        CUTLASS_PRAGMA_NO_UNROLL \
        for (int _k = 0; _k < (k_count); ++_k) { \
            (pipeline_ref).producer_acquire(state_ref); \
            auto* _barrier = (pipeline_ref).producer_get_barrier(state_ref); \
            int _ws = (state_ref).index(); \
            copy((tma_a).with(*_barrier, 0), (tAgA_ref)(_,_,_,_k), (tAsA_ref)(_,_,_,_ws)); \
            copy((tma_b).with(*_barrier, 0), (tBgB_ref)(_,_,_,_k), (tBsB_ref)(_,_,_,_ws)); \
            ++(state_ref); \
        } \
        (pipeline_ref).producer_tail(state_ref); \
    } while(0)

// Inlined consumer MMA loop
#define INLINE_CONSUMER_MMA(pipeline_ref, read_state_ref, release_state_ref, tCrA_ref, tCrB_ref, acc_ref, k_count, tiled_mma_ref) \
    do { \
        /* Prologue: first tile */ \
        (tiled_mma_ref).accumulate_ = GMMA::ScaleOut::Zero; \
        warpgroup_fence_operand(acc_ref); \
        { \
            auto _bt = (pipeline_ref).consumer_try_wait(read_state_ref); \
            (pipeline_ref).consumer_wait(read_state_ref, _bt); \
            int _rs = (read_state_ref).index(); \
            warpgroup_arrive(); \
            CUTLASS_PRAGMA_UNROLL \
            for (int _kb = 0; _kb < size<2>(tCrA_ref); ++_kb) { \
                cute::gemm((tiled_mma_ref), (tCrA_ref)(_,_,_kb,_rs), (tCrB_ref)(_,_,_kb,_rs), acc_ref); \
                (tiled_mma_ref).accumulate_ = GMMA::ScaleOut::One; \
            } \
            warpgroup_commit_batch(); \
            ++(read_state_ref); \
        } \
        (tiled_mma_ref).accumulate_ = GMMA::ScaleOut::One; \
        warpgroup_fence_operand(acc_ref); \
        /* Mainloop: remaining tiles */ \
        { \
            int _rem = (k_count) - 1; \
            CUTLASS_PRAGMA_NO_UNROLL \
            for (; _rem > 0; --_rem) { \
                auto _bt = (pipeline_ref).consumer_try_wait(read_state_ref); \
                (pipeline_ref).consumer_wait(read_state_ref, _bt); \
                int _rs = (read_state_ref).index(); \
                warpgroup_fence_operand(acc_ref); \
                warpgroup_arrive(); \
                cute::gemm((tiled_mma_ref), (tCrA_ref)(_,_,_,_rs), (tCrB_ref)(_,_,_,_rs), acc_ref); \
                warpgroup_commit_batch(); \
                warpgroup_wait<K_PIPE_MMAS>(); \
                warpgroup_fence_operand(acc_ref); \
                (pipeline_ref).consumer_release(release_state_ref); \
                ++(read_state_ref); \
                ++(release_state_ref); \
            } \
        } \
        /* mma_tail */ \
        warpgroup_fence_operand(acc_ref); \
        warpgroup_wait<0>(); \
        (pipeline_ref).consumer_release(release_state_ref); \
        ++(release_state_ref); \
    } while(0)

// =============================================================================
// v3 Kernel
// =============================================================================

__global__ void __launch_bounds__(MaxThreadsPerBlock_v3, 1)
fused_gemm_kernel_v3(__grid_constant__ FusedV3Params const params) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using namespace cute;
    using namespace cutlass;

    constexpr int NumThreadsPerWarpGroup = 128;
    int thread_idx = int(threadIdx.x);
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    int warp_in_group = (thread_idx / 32) % 4;

    enum class WarpGroupRole { Producer = 0, Consumer = 1 };
    auto warp_group_role = WarpGroupRole(thread_idx / NumThreadsPerWarpGroup);
    int lane_predicate = cute::elect_one_sync();

    // Shared memory
    extern __shared__ char smem_buf[];
    using SharedStorage = typename GemmKernel::SharedStorage;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // Pipeline
    typename MainloopPipeline::Params pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer) {
        pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
        pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    pipeline_params.transaction_bytes = params.mainloop_main.tma_transaction_bytes;

    MainloopPipeline pipeline(shared_storage.pipelines.mainloop, pipeline_params, ClusterShape_MNK{});

    if ((thread_idx == 0) && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(params.mainloop_main);
        CollectiveMainloop::prefetch_tma_descriptors(params.mainloop_high);
    }

    __syncthreads();

    // SMEM tensors
    auto& tensor_storage = shared_storage.tensors.mainloop;
    Tensor sA = make_tensor(make_smem_ptr(tensor_storage.smem_A.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(tensor_storage.smem_B.data()), SmemLayoutB{});

    // TiledMma + partitioning
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_idx);
    Tensor tCsA = thread_mma.partition_A(sA);
    Tensor tCsB = thread_mma.partition_B(sB);
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);

    // Block coordinates
    int m_coord = int(blockIdx.x);
    int n_coord = int(blockIdx.y);
    int tile_m_start = m_coord * get<0>(TileShape_MNK{});
    int tile_n_start = n_coord * get<1>(TileShape_MNK{});

    int k_tile_count_main = cute::ceil_div(params.K_main, get<2>(TileShape_MNK{}));
    int k_tile_count_high = cute::ceil_div(params.K_high, get<2>(TileShape_MNK{}));

    // Coordinate mapping for epilogue (computed once, used twice)
    auto cD = make_identity_tensor(make_shape(get<0>(TileShape_MNK{}), get<1>(TileShape_MNK{})));
    auto tCcD = thread_mma.partition_C(cD);

    // Single accumulator (reused between phases!)
    auto acc = partition_fragment_C(tiled_mma, take<0,2>(TileShape_MNK{}));

    // ==================================================================
    // PHASE 1: Main GEMM → dequant → store to workspace
    // ==================================================================
    clear(acc);
    {
        CollectiveMainloop collective_mainloop;
        auto problem_shape_main = make_shape(params.M, params.N, params.K_main, 1);
        auto load_inputs = collective_mainloop.load_init(problem_shape_main, params.mainloop_main);
        Tensor gA = get<0>(load_inputs)(_,_,m_coord,_,0);
        Tensor gB = get<1>(load_inputs)(_,_,n_coord,_,0);

        auto block_tma_a = params.mainloop_main.tma_load_a.get_slice(0);
        auto block_tma_b = params.mainloop_main.tma_load_b.get_slice(0);
        Tensor tAgA = block_tma_a.partition_S(gA);
        Tensor tAsA = block_tma_a.partition_D(sA);
        Tensor tBgB = block_tma_b.partition_S(gB);
        Tensor tBsB = block_tma_b.partition_D(sB);

        auto pipe_prod = make_producer_start_state<MainloopPipeline>();
        typename MainloopPipeline::PipelineState pipe_read;
        auto pipe_release = pipe_read;

        if (warp_group_role == WarpGroupRole::Producer) {
            if (warp_in_group == 0 && lane_predicate) {
                INLINE_PRODUCER_LOAD(pipeline, pipe_prod,
                    params.mainloop_main.tma_load_a, params.mainloop_main.tma_load_b,
                    tAgA, tAsA, tBgB, tBsB, k_tile_count_main);
            }
        }
        else { // Consumer
            INLINE_CONSUMER_MMA(pipeline, pipe_read, pipe_release,
                tCrA, tCrB, acc, k_tile_count_main, tiled_mma);

            // Dequant main and store to workspace
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(acc); ++i) {
                auto coord = tCcD(i);
                int m_g = tile_m_start + get<0>(coord);
                int n_g = tile_n_start + get<1>(coord);
                if (m_g >= params.M || n_g >= params.N) continue;

                float a = static_cast<float>(acc(i));
                float sx = float(params.s_x_m[m_g]);
                float sw = float(params.s_w_m[n_g]);
                float nz = float(params.neg_zero_m[m_g]);
                float cs = params.colsum_m[n_g];
                float result = sx * sw * (a + nz * cs);

                params.workspace[m_g * params.ldd + n_g] = ElementOutput(result);
            }
        }
    }

    // ==================================================================
    // PHASE 2: High GEMM → dequant → load workspace → combine → store
    // ==================================================================
    clear(acc);  // Reuse same accumulator registers!

    if (k_tile_count_high > 0) {
        pipeline_params.transaction_bytes = params.mainloop_high.tma_transaction_bytes;
        MainloopPipeline pipeline2(shared_storage.pipelines.mainloop, pipeline_params, ClusterShape_MNK{});

        CollectiveMainloop collective_mainloop_h;
        auto problem_shape_high = make_shape(params.M, params.N, params.K_high, 1);
        auto load_inputs_h = collective_mainloop_h.load_init(problem_shape_high, params.mainloop_high);
        Tensor gA_h = get<0>(load_inputs_h)(_,_,m_coord,_,0);
        Tensor gB_h = get<1>(load_inputs_h)(_,_,n_coord,_,0);

        auto block_tma_a_h = params.mainloop_high.tma_load_a.get_slice(0);
        auto block_tma_b_h = params.mainloop_high.tma_load_b.get_slice(0);
        Tensor tAgA_h = block_tma_a_h.partition_S(gA_h);
        Tensor tAsA_h = block_tma_a_h.partition_D(sA);
        Tensor tBgB_h = block_tma_b_h.partition_S(gB_h);
        Tensor tBsB_h = block_tma_b_h.partition_D(sB);

        auto pipe_prod2 = make_producer_start_state<MainloopPipeline>();
        typename MainloopPipeline::PipelineState pipe_read2;
        auto pipe_release2 = pipe_read2;

        if (warp_group_role == WarpGroupRole::Producer) {
            if (warp_in_group == 0 && lane_predicate) {
                INLINE_PRODUCER_LOAD(pipeline2, pipe_prod2,
                    params.mainloop_high.tma_load_a, params.mainloop_high.tma_load_b,
                    tAgA_h, tAsA_h, tBgB_h, tBsB_h, k_tile_count_high);
            }
        }
        else { // Consumer
            INLINE_CONSUMER_MMA(pipeline2, pipe_read2, pipe_release2,
                tCrA, tCrB, acc, k_tile_count_high, tiled_mma);

            // Dequant high + load workspace (Phase 1 result) + combine + store final
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(acc); ++i) {
                auto coord = tCcD(i);
                int m_g = tile_m_start + get<0>(coord);
                int n_g = tile_n_start + get<1>(coord);
                if (m_g >= params.M || n_g >= params.N) continue;

                float a_h = static_cast<float>(acc(i));
                float sx_h = float(params.s_x_h[m_g]);
                float sw_h = float(params.s_w_h[n_g]);
                float nz_h = float(params.neg_zero_h[m_g]);
                float cs_h = params.colsum_h[n_g];
                float out_high = sx_h * sw_h * (a_h + nz_h * cs_h);

                // Load Phase 1 result from workspace
                float out_main = float(params.workspace[m_g * params.ldd + n_g]);

                params.D[m_g * params.ldd + n_g] = ElementOutput(out_main + out_high);
            }
        }
    }
    else {
        // No high phase — just copy workspace to output (consumer only)
        if (warp_group_role == WarpGroupRole::Consumer) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(acc); ++i) {
                auto coord = tCcD(i);
                int m_g = tile_m_start + get<0>(coord);
                int n_g = tile_n_start + get<1>(coord);
                if (m_g >= params.M || n_g >= params.N) continue;
                params.D[m_g * params.ldd + n_g] = params.workspace[m_g * params.ldd + n_g];
            }
        }
    }
#endif
}

// =============================================================================
// Host launch
// =============================================================================

torch::Tensor fused_mixed_gemm_v3(
    torch::Tensor A_main, torch::Tensor B_main,
    torch::Tensor A_high, torch::Tensor B_high,
    torch::Tensor s_x_m, torch::Tensor s_w_m, torch::Tensor neg_zero_m, torch::Tensor colsum_m,
    torch::Tensor s_x_h, torch::Tensor s_w_h, torch::Tensor neg_zero_h, torch::Tensor colsum_h)
{
    TORCH_CHECK(A_main.is_cuda() && B_main.is_cuda() && A_high.is_cuda() && B_high.is_cuda(),
                "All inputs must be on CUDA");
    TORCH_CHECK(A_main.dtype() == torch::kUInt8 && A_high.dtype() == torch::kUInt8, "A must be uint8");
    TORCH_CHECK(B_main.dtype() == torch::kInt8 && B_high.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A_main.is_contiguous() && B_main.is_contiguous(), "must be contiguous");
    TORCH_CHECK(A_high.is_contiguous() && B_high.is_contiguous(), "must be contiguous");

    auto A_main_shape = A_main.sizes().vec();
    A_main = A_main.reshape({-1, A_main.size(-1)});
    A_high = A_high.reshape({-1, A_high.size(-1)});

    int M = A_main.size(0);
    int K_main = A_main.size(1);
    int K_high = A_high.size(1);
    int N = B_main.size(0);

    TORCH_CHECK(K_main == B_main.size(1) && K_high == B_high.size(1), "K mismatch");
    TORCH_CHECK(A_high.size(0) == M && B_high.size(0) == N, "M/N mismatch");

    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A_main.device()));
    auto workspace = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A_main.device()));
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Build mainloop params
    auto problem_main = cute::make_shape(M, N, K_main, 1);
    auto problem_high = cute::make_shape(M, N, K_high, 1);

    typename CollectiveMainloop::Arguments args_main{
        reinterpret_cast<ElementA const*>(A_main.data_ptr()),
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K_main, 1)),
        reinterpret_cast<ElementB const*>(B_main.data_ptr()),
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K_main, 1)),
    };
    typename CollectiveMainloop::Arguments args_high{
        reinterpret_cast<ElementA const*>(A_high.data_ptr()),
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K_high, 1)),
        reinterpret_cast<ElementB const*>(B_high.data_ptr()),
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K_high, 1)),
    };

    FusedV3Params kernel_params;
    kernel_params.M = M;
    kernel_params.N = N;
    kernel_params.K_main = K_main;
    kernel_params.K_high = K_high;
    kernel_params.mainloop_main = CollectiveMainloop::to_underlying_arguments(problem_main, args_main, nullptr);
    kernel_params.mainloop_high = CollectiveMainloop::to_underlying_arguments(problem_high, args_high, nullptr);
    kernel_params.s_x_m = reinterpret_cast<ElementOutput const*>(s_x_m.data_ptr());
    kernel_params.s_w_m = reinterpret_cast<ElementOutput const*>(s_w_m.data_ptr());
    kernel_params.neg_zero_m = reinterpret_cast<ElementOutput const*>(neg_zero_m.data_ptr());
    kernel_params.colsum_m = reinterpret_cast<float const*>(colsum_m.data_ptr());
    kernel_params.s_x_h = reinterpret_cast<ElementOutput const*>(s_x_h.data_ptr());
    kernel_params.s_w_h = reinterpret_cast<ElementOutput const*>(s_w_h.data_ptr());
    kernel_params.neg_zero_h = reinterpret_cast<ElementOutput const*>(neg_zero_h.data_ptr());
    kernel_params.colsum_h = reinterpret_cast<float const*>(colsum_h.data_ptr());
    kernel_params.D = reinterpret_cast<ElementOutput*>(output.data_ptr());
    kernel_params.workspace = reinterpret_cast<ElementOutput*>(workspace.data_ptr());
    kernel_params.ldd = N;

    int grid_m = cute::ceil_div(M, get<0>(TileShape_MNK{}));
    int grid_n = cute::ceil_div(N, get<1>(TileShape_MNK{}));
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(MaxThreadsPerBlock_v3, 1, 1);

    int smem_size = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));
    auto err = cudaFuncSetAttribute(fused_gemm_kernel_v3,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    TORCH_CHECK(err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(err));

    fused_gemm_kernel_v3<<<grid, block, smem_size, stream>>>(kernel_params);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fused_gemm_kernel_v3 launch failed: ", cudaGetErrorString(err));

    if (A_main_shape.size() == 3) {
        output = output.reshape({A_main_shape[0], A_main_shape[1], N});
    }
    return output;
}

#else
torch::Tensor fused_mixed_gemm_v3(
    torch::Tensor A_main, torch::Tensor B_main,
    torch::Tensor A_high, torch::Tensor B_high,
    torch::Tensor s_x_m, torch::Tensor s_w_m, torch::Tensor neg_zero_m, torch::Tensor colsum_m,
    torch::Tensor s_x_h, torch::Tensor s_w_h, torch::Tensor neg_zero_h, torch::Tensor colsum_h) {
    TORCH_CHECK(false, "SM90 not supported"); return {};
}
#endif
