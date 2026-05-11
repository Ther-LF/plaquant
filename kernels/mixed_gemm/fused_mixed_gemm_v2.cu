/***************************************************************************************************
 * Fused Mixed-Precision GEMM v2 — Hand-written K-loop (no function call boundary)
 *
 * Fixes v1's WGMMA serialization by inlining the entire mainloop.
 * All TMA loads and WGMMA instructions are in a single flat function body,
 * allowing the WGMMA pipeline to operate without serialization.
 *
 * Architecture (same as v1):
 *   Phase 1: TMA + WGMMA over K_main tiles → acc_main (INT32)
 *   Phase 2: TMA + WGMMA over K_high tiles → acc_high (INT32)
 *   Epilogue: in-register FP32 dequant + combine → FP16 store
 *
 * Key difference from v1: no CollectiveMainloop::load()/mma() calls.
 * Everything is inlined using CuTe primitives directly.
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
// Configuration (identical to v1)
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

// Build types from CUTLASS (for TMA descriptors, SMEM layout, TiledMma)
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

// Extract types
using TiledMma = typename CollectiveMainloop::TiledMma;
using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;
using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;

static constexpr int Stages = DispatchPolicy::Stages;
static constexpr int K_PIPE_MMAS = 1;  // same as CUTLASS default
static constexpr uint32_t MaxThreadsPerBlock_v2 = GemmKernel::MaxThreadsPerBlock;

// =============================================================================
// Kernel params
// =============================================================================

struct FusedV2Params {
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
    ElementOutput* D;
    int ldd;
};

// =============================================================================
// v2 Kernel: fully inlined K-loop
// =============================================================================

__global__ void __launch_bounds__(MaxThreadsPerBlock_v2, 1)
fused_gemm_kernel_v2(__grid_constant__ FusedV2Params const params) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using namespace cute;
    using namespace cutlass;

    // ---- Thread/warp setup ----
    constexpr int NumThreadsPerWarpGroup = 128;
    int thread_idx = int(threadIdx.x);
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;

    enum class WarpGroupRole { Producer = 0, Consumer = 1 };
    auto warp_group_role = WarpGroupRole(thread_idx / NumThreadsPerWarpGroup);
    int lane_predicate = cute::elect_one_sync();

    // ---- Shared memory (CUTLASS exact layout) ----
    extern __shared__ char smem_buf[];
    using SharedStorage = typename GemmKernel::SharedStorage;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // ---- Pipeline init ----
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

    // TMA prefetch
    if ((thread_idx == 0) && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(params.mainloop_main);
        CollectiveMainloop::prefetch_tma_descriptors(params.mainloop_high);
    }

    __syncthreads();

    // ---- SMEM tensors ----
    auto& tensor_storage = shared_storage.tensors.mainloop;
    Tensor sA = make_tensor(make_smem_ptr(tensor_storage.smem_A.data()), SmemLayoutA{});  // (BLK_M, BLK_K, PIPE)
    Tensor sB = make_tensor(make_smem_ptr(tensor_storage.smem_B.data()), SmemLayoutB{});  // (BLK_N, BLK_K, PIPE)

    // ---- TiledMma + partitioning ----
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_idx);
    Tensor tCsA = thread_mma.partition_A(sA);  // (MMA, MMA_M, MMA_K, PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);  // (MMA, MMA_N, MMA_K, PIPE)
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);  // GMMA descriptors
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);

    // ---- Block coordinates ----
    int m_coord = int(blockIdx.x);
    int n_coord = int(blockIdx.y);

    // ---- Tile counts ----
    int k_tile_count_main = cute::ceil_div(params.K_main, get<2>(TileShape_MNK{}));
    int k_tile_count_high = cute::ceil_div(params.K_high, get<2>(TileShape_MNK{}));

    // ---- Accumulators ----
    auto acc_main = partition_fragment_C(tiled_mma, take<0,2>(TileShape_MNK{}));
    auto acc_high = partition_fragment_C(tiled_mma, take<0,2>(TileShape_MNK{}));
    clear(acc_main);
    clear(acc_high);

    // ==================================================================
    // PHASE 1: Main GEMM (K_main tiles) — fully inlined
    // ==================================================================
    {
        // Producer: set up TMA tensors for phase 1
        // Use CollectiveMainloop::load_init (lightweight device function, just creates CuTe tensors)
        CollectiveMainloop collective_mainloop;
        auto problem_shape_main = make_shape(params.M, params.N, params.K_main, 1);
        auto load_inputs = collective_mainloop.load_init(problem_shape_main, params.mainloop_main);
        Tensor gA_mkl = get<0>(load_inputs);  // (BLK_M, BLK_K, m, k, l)
        Tensor gB_nkl = get<1>(load_inputs);  // (BLK_N, BLK_K, n, k, l)

        // Slice for this block's m,n coordinates
        Tensor gA = gA_mkl(_,_,m_coord,_,0);  // (BLK_M, BLK_K, k)
        Tensor gB = gB_nkl(_,_,n_coord,_,0);  // (BLK_N, BLK_K, k)

        auto block_tma_a = params.mainloop_main.tma_load_a.get_slice(0);
        auto block_tma_b = params.mainloop_main.tma_load_b.get_slice(0);
        Tensor tAgA = block_tma_a.partition_S(gA);  // (TMA, TMA_M, TMA_K, k)
        Tensor tAsA = block_tma_a.partition_D(sA);  // (TMA, TMA_M, TMA_K, PIPE)
        Tensor tBgB = block_tma_b.partition_S(gB);  // (TMA, TMA_N, TMA_K, k)
        Tensor tBsB = block_tma_b.partition_D(sB);  // (TMA, TMA_N, TMA_K, PIPE)

        auto pipe_producer_state = make_producer_start_state<MainloopPipeline>();
        typename MainloopPipeline::PipelineState pipe_consumer_state;
        auto pipe_consumer_release = pipe_consumer_state;

        if (warp_group_role == WarpGroupRole::Producer) {
            // Producer: TMA load loop (single elected lane)
            if (lane_predicate) {
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < k_tile_count_main; ++k) {
                    pipeline.producer_acquire(pipe_producer_state);
                    auto* tma_barrier = pipeline.producer_get_barrier(pipe_producer_state);
                    int write_stage = pipe_producer_state.index();
                    copy(params.mainloop_main.tma_load_a.with(*tma_barrier, 0), tAgA(_,_,_,k), tAsA(_,_,_,write_stage));
                    copy(params.mainloop_main.tma_load_b.with(*tma_barrier, 0), tBgB(_,_,_,k), tBsB(_,_,_,write_stage));
                    ++pipe_producer_state;
                }
                // Producer tail: wait for all buffers to be consumed
                pipeline.producer_tail(pipe_producer_state);
            }
        }
        else { // Consumer
            // Prologue: first tile with ScaleOut::Zero
            tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
            warpgroup_fence_operand(acc_main);
            {
                auto barrier_token = pipeline.consumer_try_wait(pipe_consumer_state);
                pipeline.consumer_wait(pipe_consumer_state, barrier_token);
                int read_stage = pipe_consumer_state.index();
                warpgroup_arrive();
                CUTLASS_PRAGMA_UNROLL
                for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
                    cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), acc_main);
                    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
                }
                warpgroup_commit_batch();
                ++pipe_consumer_state;
            }

            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
            warpgroup_fence_operand(acc_main);

            // Mainloop: remaining tiles
            int remaining = k_tile_count_main - 1;
            CUTLASS_PRAGMA_NO_UNROLL
            for (; remaining > 0; --remaining) {
                auto barrier_token = pipeline.consumer_try_wait(pipe_consumer_state);
                pipeline.consumer_wait(pipe_consumer_state, barrier_token);

                int read_stage = pipe_consumer_state.index();
                warpgroup_fence_operand(acc_main);
                warpgroup_arrive();
                cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), acc_main);
                warpgroup_commit_batch();

                warpgroup_wait<K_PIPE_MMAS>();
                warpgroup_fence_operand(acc_main);

                pipeline.consumer_release(pipe_consumer_release);
                ++pipe_consumer_state;
                ++pipe_consumer_release;
            }

            // mma_tail: wait for all outstanding GMMAs and release prologue buffers
            warpgroup_fence_operand(acc_main);
            pipe_consumer_release.advance(0);  // no-op since we released in loop
            warpgroup_wait<0>();
            // Release the prologue buffer (K_PIPE_MMAS=1, so 1 buffer was held)
            pipeline.consumer_release(pipe_consumer_release);
            ++pipe_consumer_release;
        }
    }

    // ==================================================================
    // PHASE 2: High GEMM (K_high tiles) — same structure, new TMA tensors
    // ==================================================================
    if (k_tile_count_high > 0) {
        // Re-initialize pipeline for phase 2
        pipeline_params.transaction_bytes = params.mainloop_high.tma_transaction_bytes;
        MainloopPipeline pipeline2(shared_storage.pipelines.mainloop, pipeline_params, ClusterShape_MNK{});

        CollectiveMainloop collective_mainloop_h;
        auto problem_shape_high = make_shape(params.M, params.N, params.K_high, 1);
        auto load_inputs_h = collective_mainloop_h.load_init(problem_shape_high, params.mainloop_high);
        Tensor gA_h_mkl = get<0>(load_inputs_h);
        Tensor gB_h_nkl = get<1>(load_inputs_h);
        Tensor gA_h = gA_h_mkl(_,_,m_coord,_,0);
        Tensor gB_h = gB_h_nkl(_,_,n_coord,_,0);

        auto block_tma_a_h = params.mainloop_high.tma_load_a.get_slice(0);
        auto block_tma_b_h = params.mainloop_high.tma_load_b.get_slice(0);
        Tensor tAgA_h = block_tma_a_h.partition_S(gA_h);
        Tensor tAsA_h = block_tma_a_h.partition_D(sA);  // reuse same SMEM
        Tensor tBgB_h = block_tma_b_h.partition_S(gB_h);
        Tensor tBsB_h = block_tma_b_h.partition_D(sB);

        auto pipe_producer_state2 = make_producer_start_state<MainloopPipeline>();
        typename MainloopPipeline::PipelineState pipe_consumer_state2;
        auto pipe_consumer_release2 = pipe_consumer_state2;

        if (warp_group_role == WarpGroupRole::Producer) {
            if (lane_predicate) {
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < k_tile_count_high; ++k) {
                    pipeline2.producer_acquire(pipe_producer_state2);
                    auto* tma_barrier = pipeline2.producer_get_barrier(pipe_producer_state2);
                    int write_stage = pipe_producer_state2.index();
                    copy(params.mainloop_high.tma_load_a.with(*tma_barrier, 0), tAgA_h(_,_,_,k), tAsA_h(_,_,_,write_stage));
                    copy(params.mainloop_high.tma_load_b.with(*tma_barrier, 0), tBgB_h(_,_,_,k), tBsB_h(_,_,_,write_stage));
                    ++pipe_producer_state2;
                }
                pipeline2.producer_tail(pipe_producer_state2);
            }
        }
        else { // Consumer
            tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
            warpgroup_fence_operand(acc_high);
            {
                auto barrier_token = pipeline2.consumer_try_wait(pipe_consumer_state2);
                pipeline2.consumer_wait(pipe_consumer_state2, barrier_token);
                int read_stage = pipe_consumer_state2.index();
                warpgroup_arrive();
                CUTLASS_PRAGMA_UNROLL
                for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
                    cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), acc_high);
                    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
                }
                warpgroup_commit_batch();
                ++pipe_consumer_state2;
            }

            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
            warpgroup_fence_operand(acc_high);

            int remaining_h = k_tile_count_high - 1;
            CUTLASS_PRAGMA_NO_UNROLL
            for (; remaining_h > 0; --remaining_h) {
                auto barrier_token = pipeline2.consumer_try_wait(pipe_consumer_state2);
                pipeline2.consumer_wait(pipe_consumer_state2, barrier_token);

                int read_stage = pipe_consumer_state2.index();
                warpgroup_fence_operand(acc_high);
                warpgroup_arrive();
                cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), acc_high);
                warpgroup_commit_batch();

                warpgroup_wait<K_PIPE_MMAS>();
                warpgroup_fence_operand(acc_high);

                pipeline2.consumer_release(pipe_consumer_release2);
                ++pipe_consumer_state2;
                ++pipe_consumer_release2;
            }

            warpgroup_fence_operand(acc_high);
            warpgroup_wait<0>();
            pipeline2.consumer_release(pipe_consumer_release2);
            ++pipe_consumer_release2;
        }
    }

    // ==================================================================
    // EPILOGUE: In-register dequant + combine + store (consumer only)
    // ==================================================================
    if (warp_group_role == WarpGroupRole::Consumer) {
        int tile_m_start = m_coord * get<0>(TileShape_MNK{});
        int tile_n_start = n_coord * get<1>(TileShape_MNK{});

        auto cD = make_identity_tensor(make_shape(get<0>(TileShape_MNK{}), get<1>(TileShape_MNK{})));
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
            float a_m = static_cast<float>(acc_main(i));
            float sx_m = float(params.s_x_m[m_global]);
            float sw_m = float(params.s_w_m[n_global]);
            float nz_m = float(params.neg_zero_m[m_global]);
            float cs_m = params.colsum_m[n_global];
            float out_main = sx_m * sw_m * (a_m + nz_m * cs_m);

            // Dequant high
            float a_h = static_cast<float>(acc_high(i));
            float sx_h = float(params.s_x_h[m_global]);
            float sw_h = float(params.s_w_h[n_global]);
            float nz_h = float(params.neg_zero_h[m_global]);
            float cs_h = params.colsum_h[n_global];
            float out_high = sx_h * sw_h * (a_h + nz_h * cs_h);

            params.D[m_global * params.ldd + n_global] = ElementOutput(out_main + out_high);
        }
    }
#endif
}

// =============================================================================
// Host launch
// =============================================================================

torch::Tensor fused_mixed_gemm_v2(
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
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Build mainloop params (TMA descriptors)
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

    FusedV2Params kernel_params;
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
    kernel_params.ldd = N;

    int grid_m = cute::ceil_div(M, get<0>(TileShape_MNK{}));
    int grid_n = cute::ceil_div(N, get<1>(TileShape_MNK{}));
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(MaxThreadsPerBlock_v2, 1, 1);

    int smem_size = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));

    auto err = cudaFuncSetAttribute(fused_gemm_kernel_v2,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    TORCH_CHECK(err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(err));

    fused_gemm_kernel_v2<<<grid, block, smem_size, stream>>>(kernel_params);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fused_gemm_kernel_v2 launch failed: ", cudaGetErrorString(err));

    if (A_main_shape.size() == 3) {
        output = output.reshape({A_main_shape[0], A_main_shape[1], N});
    }
    return output;
}

#else
torch::Tensor fused_mixed_gemm_v2(
    torch::Tensor A_main, torch::Tensor B_main,
    torch::Tensor A_high, torch::Tensor B_high,
    torch::Tensor s_x_m, torch::Tensor s_w_m, torch::Tensor neg_zero_m, torch::Tensor colsum_m,
    torch::Tensor s_x_h, torch::Tensor s_w_h, torch::Tensor neg_zero_h, torch::Tensor colsum_h) {
    TORCH_CHECK(false, "SM90 not supported"); return {};
}
#endif
