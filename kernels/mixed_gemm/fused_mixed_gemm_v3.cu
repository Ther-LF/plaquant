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

// (Macros removed — producer/consumer loops are inlined directly in the kernel body)

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
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < k_tile_count_main; ++k) {
                    pipeline.producer_acquire(pipe_prod);
                    auto* barrier = pipeline.producer_get_barrier(pipe_prod);
                    int ws = pipe_prod.index();
                    copy(params.mainloop_main.tma_load_a.with(*barrier, 0), tAgA(_,_,_,k), tAsA(_,_,_,ws));
                    copy(params.mainloop_main.tma_load_b.with(*barrier, 0), tBgB(_,_,_,k), tBsB(_,_,_,ws));
                    ++pipe_prod;
                }
                pipeline.producer_tail(pipe_prod);
            }
        }
        else { // Consumer — Phase 1 MMA
            tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
            warpgroup_fence_operand(acc);
            {
                auto bt = pipeline.consumer_try_wait(pipe_read);
                pipeline.consumer_wait(pipe_read, bt);
                int rs = pipe_read.index();
                warpgroup_arrive();
                CUTLASS_PRAGMA_UNROLL
                for (int kb = 0; kb < size<2>(tCrA); ++kb) {
                    cute::gemm(tiled_mma, tCrA(_,_,kb,rs), tCrB(_,_,kb,rs), acc);
                    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
                }
                warpgroup_commit_batch();
                ++pipe_read;
            }
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
            warpgroup_fence_operand(acc);
            {
                int rem = k_tile_count_main - 1;
                CUTLASS_PRAGMA_NO_UNROLL
                for (; rem > 0; --rem) {
                    auto bt = pipeline.consumer_try_wait(pipe_read);
                    pipeline.consumer_wait(pipe_read, bt);
                    int rs = pipe_read.index();
                    warpgroup_fence_operand(acc);
                    warpgroup_arrive();
                    cute::gemm(tiled_mma, tCrA(_,_,_,rs), tCrB(_,_,_,rs), acc);
                    warpgroup_commit_batch();
                    warpgroup_wait<K_PIPE_MMAS>();
                    warpgroup_fence_operand(acc);
                    pipeline.consumer_release(pipe_release);
                    ++pipe_read;
                    ++pipe_release;
                }
            }
            warpgroup_fence_operand(acc);
            warpgroup_wait<0>();
            pipeline.consumer_release(pipe_release);
            ++pipe_release;

            // Dequant main into FP16 fragment (consumer threads only have acc)
            auto acc_fp16 = make_fragment_like<ElementOutput>(acc);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(acc); ++i) {
                auto coord = tCcD(i);
                int m_g = tile_m_start + get<0>(coord);
                int n_g = tile_n_start + get<1>(coord);
                float result = 0.0f;
                if (m_g < params.M && n_g < params.N) {
                    float a = static_cast<float>(acc(i));
                    float sx = float(params.s_x_m[m_g]);
                    float sw = float(params.s_w_m[n_g]);
                    float nz = float(params.neg_zero_m[m_g]);
                    float cs = params.colsum_m[n_g];
                    result = sx * sw * (a + nz * cs);
                }
                acc_fp16(i) = ElementOutput(result);
            }

            // Scatter dequanted results to SMEM (reuse mainloop SMEM as epilogue buffer)
            auto* smem_epi = reinterpret_cast<ElementOutput*>(smem_buf);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(acc_fp16); ++i) {
                auto coord = tCcD(i);
                int m_local = get<0>(coord);
                int n_local = get<1>(coord);
                smem_epi[m_local * 128 + n_local] = acc_fp16(i);
            }
        }
    }

    // Block-wide sync: all 256 threads (producer done with load_tail, consumer done with dequant→SMEM)
    __syncthreads();

    // Coalesced store from SMEM to workspace — ALL 256 threads participate
    {
        auto* smem_epi = reinterpret_cast<ElementOutput*>(smem_buf);
        constexpr int TILE_SIZE = 128 * 128;
        // 256 threads, each stores TILE_SIZE/256 = 64 elements
        constexpr int ELEMS_PER_THREAD = TILE_SIZE / 256;
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            int flat_idx = thread_idx + e * 256;
            int m_local = flat_idx / 128;
            int n_local = flat_idx % 128;
            int m_g = tile_m_start + m_local;
            int n_g = tile_n_start + n_local;
            if (m_g < params.M && n_g < params.N) {
                params.workspace[m_g * params.ldd + n_g] = smem_epi[flat_idx];
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
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < k_tile_count_high; ++k) {
                    pipeline2.producer_acquire(pipe_prod2);
                    auto* barrier = pipeline2.producer_get_barrier(pipe_prod2);
                    int ws = pipe_prod2.index();
                    copy(params.mainloop_high.tma_load_a.with(*barrier, 0), tAgA_h(_,_,_,k), tAsA_h(_,_,_,ws));
                    copy(params.mainloop_high.tma_load_b.with(*barrier, 0), tBgB_h(_,_,_,k), tBsB_h(_,_,_,ws));
                    ++pipe_prod2;
                }
                pipeline2.producer_tail(pipe_prod2);
            }
        }
        else { // Consumer — Phase 2 MMA
            tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
            warpgroup_fence_operand(acc);
            {
                auto bt = pipeline2.consumer_try_wait(pipe_read2);
                pipeline2.consumer_wait(pipe_read2, bt);
                int rs = pipe_read2.index();
                warpgroup_arrive();
                CUTLASS_PRAGMA_UNROLL
                for (int kb = 0; kb < size<2>(tCrA); ++kb) {
                    cute::gemm(tiled_mma, tCrA(_,_,kb,rs), tCrB(_,_,kb,rs), acc);
                    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
                }
                warpgroup_commit_batch();
                ++pipe_read2;
            }
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
            warpgroup_fence_operand(acc);
            {
                int rem = k_tile_count_high - 1;
                CUTLASS_PRAGMA_NO_UNROLL
                for (; rem > 0; --rem) {
                    auto bt = pipeline2.consumer_try_wait(pipe_read2);
                    pipeline2.consumer_wait(pipe_read2, bt);
                    int rs = pipe_read2.index();
                    warpgroup_fence_operand(acc);
                    warpgroup_arrive();
                    cute::gemm(tiled_mma, tCrA(_,_,_,rs), tCrB(_,_,_,rs), acc);
                    warpgroup_commit_batch();
                    warpgroup_wait<K_PIPE_MMAS>();
                    warpgroup_fence_operand(acc);
                    pipeline2.consumer_release(pipe_release2);
                    ++pipe_read2;
                    ++pipe_release2;
                }
            }
            warpgroup_fence_operand(acc);
            warpgroup_wait<0>();
            pipeline2.consumer_release(pipe_release2);
            ++pipe_release2;

            // Dequant high + combine → scatter to SMEM
            auto* smem_epi2 = reinterpret_cast<ElementOutput*>(smem_buf);
            auto acc_fp16_2 = make_fragment_like<ElementOutput>(acc);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(acc); ++i) {
                auto coord = tCcD(i);
                int m_g = tile_m_start + get<0>(coord);
                int n_g = tile_n_start + get<1>(coord);
                float combined = 0.0f;
                if (m_g < params.M && n_g < params.N) {
                    float a_h = static_cast<float>(acc(i));
                    float sx_h = float(params.s_x_h[m_g]);
                    float sw_h = float(params.s_w_h[n_g]);
                    float nz_h = float(params.neg_zero_h[m_g]);
                    float cs_h = params.colsum_h[n_g];
                    float out_high = sx_h * sw_h * (a_h + nz_h * cs_h);
                    float out_main = float(params.workspace[m_g * params.ldd + n_g]);
                    combined = out_main + out_high;
                }
                acc_fp16_2(i) = ElementOutput(combined);
            }
            // Scatter to SMEM
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(acc_fp16_2); ++i) {
                auto coord = tCcD(i);
                smem_epi2[get<0>(coord) * 128 + get<1>(coord)] = acc_fp16_2(i);
            }
        }
    }

    // Block-wide sync for Phase 2 epilogue
    __syncthreads();

    // Coalesced store from SMEM to D — all 256 threads
    if (k_tile_count_high > 0) {
        auto* smem_epi2 = reinterpret_cast<ElementOutput*>(smem_buf);
        constexpr int TILE_SIZE = 128 * 128;
        constexpr int ELEMS_PER_THREAD = TILE_SIZE / 256;
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            int flat_idx = thread_idx + e * 256;
            int m_g = tile_m_start + flat_idx / 128;
            int n_g = tile_n_start + flat_idx % 128;
            if (m_g < params.M && n_g < params.N) {
                params.D[m_g * params.ldd + n_g] = smem_epi2[flat_idx];
            }
        }
    }
    else {
        // No high phase — coalesced copy workspace to output
        if (warp_group_role == WarpGroupRole::Consumer) {
            constexpr int TILE_M = 128;
            constexpr int TILE_N = 128;
            constexpr int ELEMS_PER_THREAD = (TILE_M * TILE_N) / 128;
            int tid = warp_group_thread_idx;
            CUTLASS_PRAGMA_UNROLL
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                int flat_idx = tid + e * 128;
                int m_local = flat_idx / TILE_N;
                int n_local = flat_idx % TILE_N;
                int m_g = tile_m_start + m_local;
                int n_g = tile_n_start + n_local;
                if (m_g < params.M && n_g < params.N) {
                    params.D[m_g * params.ldd + n_g] = params.workspace[m_g * params.ldd + n_g];
                }
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
