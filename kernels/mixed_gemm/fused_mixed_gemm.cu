/***************************************************************************************************
 * Fused Mixed-Precision GEMM — TRUE single kernel
 *
 * One kernel launch that:
 *   Phase 1: U8S8 WGMMA mainloop (K_main tiles) → acc_main (INT32 registers)
 *   Phase 2: U8S8 WGMMA mainloop (K_high tiles) → acc_high (INT32 registers)
 *   Epilogue: in-register FP32 dequant + combine → store FP16 to HBM
 *
 * Uses CUTLASS 3.x warp-specialized architecture:
 *   - Producer warp group: TMA loads for phase 1, then phase 2
 *   - Consumer warp group: WGMMA compute for both phases, then epilogue
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
// Configuration
// =============================================================================

using TileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using ElementA = uint8_t;    // activation (unsigned)
using ElementB = int8_t;     // weight (signed)
using ElementAccum = int32_t;
using ElementOutput = cutlass::half_t;
using ElementCompute = float;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;

// Build mainloop collective (handles TMA descriptors, SMEM layout, pipeline, WGMMA)
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, AlignmentA,
    ElementB, cutlass::layout::ColumnMajor, AlignmentB,
    ElementAccum,
    TileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::KernelTmaWarpSpecialized
>::CollectiveOp;

// We also need a minimal epilogue type to get the full GemmUniversal kernel
// (for its shared storage layout and grid computation), but we override the
// epilogue behavior in our custom kernel.
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    ElementOutput, cutlass::layout::RowMajor, 8,
    ElementOutput, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized
>::CollectiveOp;

// Full kernel type — we use this to get proper types for arguments/params
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;

// =============================================================================
// Fused kernel: run two CUTLASS mainloops + custom epilogue in ONE launch
//
// Strategy: instantiate two GemmUniversal (one for main, one for high),
// but instead of launching them separately, we launch one and manually
// invoke the second's compute after the first completes.
//
// Actually, the cleanest approach for v1 is to use a SINGLE GemmUniversal
// for the larger phase (main), and overlap the smaller phase (high) using
// the same kernel grid with a modified approach.
//
// SIMPLEST TRUE FUSION: Use stream-ordered kernel launch with cudaGraph
// or just accept that for v1, we use the CUTLASS machinery properly.
//
// TRUE v1: Write custom __global__ that uses the mainloop collective's
// mma() method twice. This requires us to set up the kernel infrastructure
// (shared memory, pipeline, warp roles) ourselves.
// =============================================================================

// We build a helper that creates mainloop params for a given (M,N,K) problem
static typename CollectiveMainloop::Params make_mainloop_params(
    int M, int N, int K,
    void const* ptr_A, void const* ptr_B)
{
    auto problem_shape = cute::make_shape(M, N, K, 1);
    typename CollectiveMainloop::Arguments mainloop_args{
        reinterpret_cast<ElementA const*>(ptr_A),
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1)),
        reinterpret_cast<ElementB const*>(ptr_B),
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1)),
    };
    return CollectiveMainloop::to_underlying_arguments(problem_shape, mainloop_args, nullptr);
}

// =============================================================================
// Custom fused kernel
// =============================================================================

struct FusedKernelParams {
    // Problem shape
    int M, N, K_main, K_high;

    // Mainloop params for each phase (contains TMA descriptors)
    typename CollectiveMainloop::Params mainloop_main;
    typename CollectiveMainloop::Params mainloop_high;

    // Dequant parameters
    ElementOutput const* s_x_m;       // (M,) per-token scale main
    ElementOutput const* s_w_m;       // (N,) per-channel scale main
    ElementOutput const* neg_zero_m;  // (M,) negated zero main
    float const* colsum_m;            // (N,) weight colsum main
    ElementOutput const* s_x_h;       // (M,) per-token scale high
    ElementOutput const* s_w_h;       // (N,) per-channel scale high
    ElementOutput const* neg_zero_h;  // (M,) negated zero high
    float const* colsum_h;            // (N,) weight colsum high

    // Output
    ElementOutput* D;                 // (M, N) output
};

// The kernel uses CUTLASS mainloop internals directly
__global__ void __launch_bounds__(256)
fused_kernel(FusedKernelParams params) {

    using namespace cute;
    using namespace cutlass;

    // Warp specialization roles
    constexpr int NumThreadsPerWarpGroup = 128;
    constexpr int NumWarpsPerWarpGroup = 4;

    enum class WarpGroupRole { Producer = 0, Consumer = 1 };
    enum class ProducerWarpRole { Mainloop = 0, Warp1 = 1, Warp2 = 2, Warp3 = 3 };

    int thread_idx = int(threadIdx.x);
    int warp_idx = thread_idx / 32;
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(thread_idx / NumThreadsPerWarpGroup);
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();

    // Shared memory — we reuse the same buffer for both phases (sequentially)
    extern __shared__ char smem_buf[];
    using TensorStorage = typename CollectiveMainloop::TensorStorage;
    using PipelineStorage = typename CollectiveMainloop::MainloopPipeline::SharedStorage;

    struct SharedStorage {
        TensorStorage tensors;
        PipelineStorage pipeline;
    };
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // Pipeline setup
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Mainloop) {
        pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
        pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    pipeline_params.transaction_bytes = params.mainloop_main.tma_transaction_bytes;
    MainloopPipeline pipeline(shared_storage.pipeline, pipeline_params, ClusterShape_MNK{});

    // Compute tile coordinates
    auto blk_shape = TileShape_MNK{};
    auto problem_shape_main = cute::make_shape(params.M, params.N, params.K_main, 1);
    auto problem_shape_high = cute::make_shape(params.M, params.N, params.K_high, 1);

    // Block coordinates (same for both phases — same M,N tile)
    int m_coord = int(blockIdx.x);
    int n_coord = int(blockIdx.y);
    auto blk_coord = make_coord(m_coord, n_coord, _, 0);

    CollectiveMainloop collective_mainloop;

    // TiledMma for accumulator partitioning
    typename CollectiveMainloop::TiledMma tiled_mma;

    __syncthreads();

    // ==== PHASE 1: Main GEMM (K_main tiles) ====
    int k_tile_count_main = (params.K_main + get<2>(blk_shape) - 1) / get<2>(blk_shape);

    // Accumulators
    auto acc_main = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));
    clear(acc_main);

    {
        // Producer: load main data
        auto load_inputs_main = collective_mainloop.load_init(problem_shape_main, params.mainloop_main);
        auto mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
        typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;

        if (warp_group_role == WarpGroupRole::Producer) {
            if (producer_warp_role == ProducerWarpRole::Mainloop) {
                collective_mainloop.load(
                    params.mainloop_main,
                    pipeline,
                    mainloop_pipe_producer_state,
                    load_inputs_main,
                    blk_coord,
                    cute::make_coord_iterator(shape<3>(get<0>(load_inputs_main))),
                    k_tile_count_main,
                    thread_idx % 32,  // lane_idx
                    cute::block_rank_in_cluster(),
                    shared_storage.tensors);
                mainloop_pipe_producer_state.advance(k_tile_count_main);
                collective_mainloop.load_tail(pipeline, mainloop_pipe_producer_state);
            }
        }
        else if (warp_group_role == WarpGroupRole::Consumer) {
            collective_mainloop.mma(
                pipeline,
                mainloop_pipe_consumer_state,
                acc_main,
                k_tile_count_main,
                warp_group_thread_idx,
                shared_storage.tensors,
                params.mainloop_main);
            collective_mainloop.mma_tail(pipeline, mainloop_pipe_consumer_state, k_tile_count_main);
        }
    }

    // Sync between phases — all warps must complete phase 1 before starting phase 2
    __syncthreads();

    // ==== PHASE 2: High GEMM (K_high tiles) ====
    int k_tile_count_high = (params.K_high + get<2>(blk_shape) - 1) / get<2>(blk_shape);

    auto acc_high = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));
    clear(acc_high);

    {
        // Re-initialize pipeline for phase 2
        // Need to reset pipeline state
        pipeline_params.transaction_bytes = params.mainloop_high.tma_transaction_bytes;
        // Reconstruct pipeline (reuse same shared storage)
        MainloopPipeline pipeline2(shared_storage.pipeline, pipeline_params, ClusterShape_MNK{});

        auto load_inputs_high = collective_mainloop.load_init(problem_shape_high, params.mainloop_high);
        auto mainloop_pipe_producer_state2 = cutlass::make_producer_start_state<MainloopPipeline>();
        typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state2;

        if (warp_group_role == WarpGroupRole::Producer) {
            if (producer_warp_role == ProducerWarpRole::Mainloop) {
                collective_mainloop.load(
                    params.mainloop_high,
                    pipeline2,
                    mainloop_pipe_producer_state2,
                    load_inputs_high,
                    blk_coord,
                    cute::make_coord_iterator(shape<3>(get<0>(load_inputs_high))),
                    k_tile_count_high,
                    thread_idx % 32,
                    cute::block_rank_in_cluster(),
                    shared_storage.tensors);
                mainloop_pipe_producer_state2.advance(k_tile_count_high);
                collective_mainloop.load_tail(pipeline2, mainloop_pipe_producer_state2);
            }
        }
        else if (warp_group_role == WarpGroupRole::Consumer) {
            collective_mainloop.mma(
                pipeline2,
                mainloop_pipe_consumer_state2,
                acc_high,
                k_tile_count_high,
                warp_group_thread_idx,
                shared_storage.tensors,
                params.mainloop_high);
            collective_mainloop.mma_tail(pipeline2, mainloop_pipe_consumer_state2, k_tile_count_high);
        }
    }

    // ==== EPILOGUE: In-register dequant + combine + store ====
    // Only consumer warp group does the epilogue
    if (warp_group_role == WarpGroupRole::Consumer) {
        // Get M,N coordinates for this thread's accumulator elements
        int tile_m_start = m_coord * get<0>(blk_shape);
        int tile_n_start = n_coord * get<1>(blk_shape);

        // Iterate over accumulator elements owned by this thread
        // acc_main and acc_high have shape (MMA, MMA_M, MMA_N)
        // The thread-to-element mapping is determined by TiledMma
        auto thread_mma = tiled_mma.get_slice(warp_group_thread_idx);

        // Get the (M,N) coordinates for each accumulator element
        // For SM90 WGMMA with tile 128x128, each thread owns multiple elements
        auto cD = make_identity_tensor(make_shape(get<0>(blk_shape), get<1>(blk_shape)));
        auto tCcD = thread_mma.partition_C(cD);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc_main); ++i) {
            // Get (m_local, n_local) within tile
            auto coord = tCcD(i);
            int m_local = get<0>(coord);
            int n_local = get<1>(coord);
            int m_global = tile_m_start + m_local;
            int n_global = tile_n_start + n_local;

            // Bounds check
            if (m_global >= params.M || n_global >= params.N) continue;

            // Dequant main: s_x_m * s_w_m * (acc_main + neg_zero_m * colsum_m)
            float a_main = static_cast<float>(acc_main(i));
            float sx_m = static_cast<float>(params.s_x_m[m_global]);
            float sw_m = static_cast<float>(params.s_w_m[n_global]);
            float nz_m = static_cast<float>(params.neg_zero_m[m_global]);
            float cs_m = params.colsum_m[n_global];
            float out_main = sx_m * sw_m * (a_main + nz_m * cs_m);

            // Dequant high: s_x_h * s_w_h * (acc_high + neg_zero_h * colsum_h)
            float a_high = static_cast<float>(acc_high(i));
            float sx_h = static_cast<float>(params.s_x_h[m_global]);
            float sw_h = static_cast<float>(params.s_w_h[n_global]);
            float nz_h = static_cast<float>(params.neg_zero_h[m_global]);
            float cs_h = params.colsum_h[n_global];
            float out_high = sx_h * sw_h * (a_high + nz_h * cs_h);

            // Combine and store
            params.D[m_global * params.N + n_global] = ElementOutput(out_main + out_high);
        }
    }
}

// =============================================================================
// Host-side launch
// =============================================================================

torch::Tensor fused_mixed_gemm(
    torch::Tensor A_main,      // (M, K_main) uint8
    torch::Tensor B_main,      // (N, K_main) int8
    torch::Tensor A_high,      // (M, K_high) uint8
    torch::Tensor B_high,      // (N, K_high) int8
    torch::Tensor s_x_m,       // (M,) fp16
    torch::Tensor s_w_m,       // (N,) fp16
    torch::Tensor neg_zero_m,  // (M,) fp16
    torch::Tensor colsum_m,    // (N,) fp32
    torch::Tensor s_x_h,       // (M,) fp16
    torch::Tensor s_w_h,       // (N,) fp16
    torch::Tensor neg_zero_h,  // (M,) fp16
    torch::Tensor colsum_h)    // (N,) fp32
{
    // Input validation
    TORCH_CHECK(A_main.is_cuda() && B_main.is_cuda() && A_high.is_cuda() && B_high.is_cuda(),
                "All inputs must be on CUDA");
    TORCH_CHECK(A_main.dtype() == torch::kUInt8, "A_main must be uint8");
    TORCH_CHECK(B_main.dtype() == torch::kInt8, "B_main must be int8");
    TORCH_CHECK(A_high.dtype() == torch::kUInt8, "A_high must be uint8");
    TORCH_CHECK(B_high.dtype() == torch::kInt8, "B_high must be int8");
    TORCH_CHECK(A_main.is_contiguous() && B_main.is_contiguous(), "main must be contiguous");
    TORCH_CHECK(A_high.is_contiguous() && B_high.is_contiguous(), "high must be contiguous");

    // Handle 2D or 3D input
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

    // Build params
    FusedKernelParams kernel_params;
    kernel_params.M = M;
    kernel_params.N = N;
    kernel_params.K_main = K_main;
    kernel_params.K_high = K_high;
    kernel_params.mainloop_main = make_mainloop_params(M, N, K_main, A_main.data_ptr(), B_main.data_ptr());
    kernel_params.mainloop_high = make_mainloop_params(M, N, K_high, A_high.data_ptr(), B_high.data_ptr());
    kernel_params.s_x_m = reinterpret_cast<ElementOutput const*>(s_x_m.data_ptr());
    kernel_params.s_w_m = reinterpret_cast<ElementOutput const*>(s_w_m.data_ptr());
    kernel_params.neg_zero_m = reinterpret_cast<ElementOutput const*>(neg_zero_m.data_ptr());
    kernel_params.colsum_m = reinterpret_cast<float const*>(colsum_m.data_ptr());
    kernel_params.s_x_h = reinterpret_cast<ElementOutput const*>(s_x_h.data_ptr());
    kernel_params.s_w_h = reinterpret_cast<ElementOutput const*>(s_w_h.data_ptr());
    kernel_params.neg_zero_h = reinterpret_cast<ElementOutput const*>(neg_zero_h.data_ptr());
    kernel_params.colsum_h = reinterpret_cast<float const*>(colsum_h.data_ptr());
    kernel_params.D = reinterpret_cast<ElementOutput*>(output.data_ptr());

    // Grid: one CTA per (M_tile, N_tile)
    int grid_m = (M + get<0>(TileShape_MNK{}) - 1) / get<0>(TileShape_MNK{});
    int grid_n = (N + get<1>(TileShape_MNK{}) - 1) / get<1>(TileShape_MNK{});
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(256, 1, 1);  // 2 warp groups × 128 threads

    // Shared memory size (reused between phases)
    int smem_size = sizeof(typename CollectiveMainloop::TensorStorage)
                  + sizeof(typename CollectiveMainloop::MainloopPipeline::SharedStorage);

    // Set max dynamic shared memory
    cudaFuncSetAttribute(fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    fused_kernel<<<grid, block, smem_size, stream>>>(kernel_params);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fused_kernel launch failed: ", cudaGetErrorString(err));

    // Reshape output to match input batch dims
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
