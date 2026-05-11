/***************************************************************************************************
 * Fused Mixed-Precision GEMM — DEBUG VERSION
 * Minimal kernel to isolate the crash.
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
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;

static constexpr uint32_t MaxThreadsPerBlock_ = GemmKernel::MaxThreadsPerBlock;

// =============================================================================
// Minimal debug kernel: just writes zeros to output
// Tests: kernel launch + smem allocation + simple global store
// =============================================================================

struct FusedKernelParams {
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

__global__ void __launch_bounds__(256, 1)
fused_gemm_kernel(__grid_constant__ FusedKernelParams const params) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using namespace cute;
    using namespace cutlass;

    constexpr int NumThreadsPerWarpGroup__ = 128;

    int thread_idx = int(threadIdx.x);
    int warp_idx_in_warp_group = (thread_idx / 32) % 4;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup__;

    enum class WarpGroupRole { Producer = 0, Consumer = 1 };
    enum class ProducerWarpRole { Mainloop = 0, Warp1 = 1, Warp2 = 2, Warp3 = 3 };
    auto warp_group_role = WarpGroupRole(thread_idx / NumThreadsPerWarpGroup__);
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

    // Shared memory - use EXACT GemmKernel SharedStorage type
    extern __shared__ char smem_buf[];
    using KernelSharedStorage = typename GemmKernel::SharedStorage;
    KernelSharedStorage& shared_storage = *reinterpret_cast<KernelSharedStorage*>(smem_buf);

    // Pipeline construction
    using MainloopPipeline_ = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline_::Params pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Mainloop) {
        pipeline_params.role = MainloopPipeline_::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
        pipeline_params.role = MainloopPipeline_::ThreadCategory::Consumer;
    }
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumThreadsPerWarpGroup__;
    pipeline_params.transaction_bytes = params.mainloop_main.tma_transaction_bytes;

    MainloopPipeline_ mainloop_pipeline(shared_storage.pipelines.mainloop, pipeline_params, ClusterShape_MNK{});

    // TMA prefetch
    if (thread_idx == 0) {
        CollectiveMainloop::prefetch_tma_descriptors(params.mainloop_main);
    }

    __syncthreads();

    // Mainloop setup
    CollectiveMainloop collective_mainloop;
    TiledMma tiled_mma;

    auto problem_shape_main = make_shape(params.M, params.N, params.K_main, 1);
    int k_tile_count_main = cute::ceil_div(params.K_main, get<2>(TileShape_MNK{}));
    int k_tile_count_high = cute::ceil_div(params.K_high, get<2>(TileShape_MNK{}));

    int m_coord = int(blockIdx.x);
    int n_coord = int(blockIdx.y);
    auto blk_coord = make_coord(m_coord, n_coord, _, 0);

    // Phase 1 accumulator
    auto acc_main = partition_fragment_C(tiled_mma, take<0,2>(TileShape_MNK{}));
    clear(acc_main);

    // Phase 1: Load + MMA
    {
        auto load_inputs = collective_mainloop.load_init(problem_shape_main, params.mainloop_main);
        auto k_tile_iter = cute::make_coord_iterator(shape<3>(get<0>(load_inputs)));

        auto pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline_>();
        cutlass::PipelineState<CollectiveMainloop::DispatchPolicy::Stages> pipe_consumer_state;

        if (warp_group_role == WarpGroupRole::Producer) {
            if (producer_warp_role == ProducerWarpRole::Mainloop) {
                collective_mainloop.load(
                    params.mainloop_main,
                    mainloop_pipeline,
                    pipe_producer_state,
                    load_inputs,
                    blk_coord,
                    k_tile_iter, k_tile_count_main,
                    thread_idx % 32,
                    0,
                    shared_storage.tensors.mainloop);
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
                shared_storage.tensors.mainloop,
                params.mainloop_main);
            collective_mainloop.mma_tail(mainloop_pipeline, pipe_consumer_state, k_tile_count_main);
        }
    }

    // ==================================================================
    // PHASE 2: High GEMM (reuse same smem and pipeline)
    // ==================================================================
    auto acc_high = partition_fragment_C(tiled_mma, take<0,2>(TileShape_MNK{}));
    clear(acc_high);

    if (k_tile_count_high > 0) {
        auto problem_shape_high = make_shape(params.M, params.N, params.K_high, 1);

        // Re-init pipeline for phase 2
        pipeline_params.transaction_bytes = params.mainloop_high.tma_transaction_bytes;
        MainloopPipeline_ mainloop_pipeline2(shared_storage.pipelines.mainloop, pipeline_params, ClusterShape_MNK{});

        if (thread_idx == 0) {
            CollectiveMainloop::prefetch_tma_descriptors(params.mainloop_high);
        }

        auto load_inputs_h = collective_mainloop.load_init(problem_shape_high, params.mainloop_high);
        auto k_tile_iter_h = cute::make_coord_iterator(shape<3>(get<0>(load_inputs_h)));
        auto pipe_producer_state2 = cutlass::make_producer_start_state<MainloopPipeline_>();
        cutlass::PipelineState<CollectiveMainloop::DispatchPolicy::Stages> pipe_consumer_state2;

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
                    0,
                    shared_storage.tensors.mainloop);
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
                shared_storage.tensors.mainloop,
                params.mainloop_high);
            collective_mainloop.mma_tail(mainloop_pipeline2, pipe_consumer_state2, k_tile_count_high);
        }
    }

    // ==================================================================
    // EPILOGUE: Dequant both phases + combine + store
    // ==================================================================
    if (warp_group_role == WarpGroupRole::Consumer) {
        int tile_m_start = m_coord * 128;
        int tile_n_start = n_coord * 128;

        // Map accumulator elements to (M,N) coordinates
        auto cD = make_identity_tensor(make_shape(get<0>(TileShape_MNK{}), get<1>(TileShape_MNK{})));
        auto thread_mma_slice = tiled_mma.get_slice(warp_group_thread_idx);
        auto tCcD = thread_mma_slice.partition_C(cD);

        for (int i = 0; i < size(acc_main); ++i) {
            auto coord = tCcD(i);
            int m_local = get<0>(coord);
            int n_local = get<1>(coord);
            int m_global = tile_m_start + m_local;
            int n_global = tile_n_start + n_local;

            if (m_global >= params.M || n_global >= params.N) continue;

            // Dequant: D = s_x * s_w * (acc + neg_zero * colsum)
            float acc_f = static_cast<float>(acc_main(i));
            float sx = float(params.s_x_m[m_global]);
            float sw = float(params.s_w_m[n_global]);
            float nz = float(params.neg_zero_m[m_global]);
            float cs = params.colsum_m[n_global];
            float out_main = sx * sw * (acc_f + nz * cs);

            // Dequant high
            float acc_h_f = static_cast<float>(acc_high(i));
            float sx_h = float(params.s_x_h[m_global]);
            float sw_h = float(params.s_w_h[n_global]);
            float nz_h = float(params.neg_zero_h[m_global]);
            float cs_h = params.colsum_h[n_global];
            float out_high = sx_h * sw_h * (acc_h_f + nz_h * cs_h);

            params.D[m_global * params.ldd + n_global] = ElementOutput(out_main + out_high);
        }
    }
#endif
}

// =============================================================================
// Host launch
// =============================================================================

torch::Tensor fused_mixed_gemm(
    torch::Tensor A_main, torch::Tensor B_main,
    torch::Tensor A_high, torch::Tensor B_high,
    torch::Tensor s_x_m, torch::Tensor s_w_m, torch::Tensor neg_zero_m, torch::Tensor colsum_m,
    torch::Tensor s_x_h, torch::Tensor s_w_h, torch::Tensor neg_zero_h, torch::Tensor colsum_h)
{
    TORCH_CHECK(A_main.is_cuda() && B_main.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A_main.dtype() == torch::kUInt8, "A_main must be uint8");
    TORCH_CHECK(B_main.dtype() == torch::kInt8, "B_main must be int8");
    TORCH_CHECK(A_high.dtype() == torch::kUInt8, "A_high must be uint8");
    TORCH_CHECK(B_high.dtype() == torch::kInt8, "B_high must be int8");
    TORCH_CHECK(A_main.is_contiguous() && B_main.is_contiguous(), "must be contiguous");
    TORCH_CHECK(A_high.is_contiguous() && B_high.is_contiguous(), "must be contiguous");

    auto A_main_shape = A_main.sizes().vec();
    A_main = A_main.reshape({-1, A_main.size(-1)});
    A_high = A_high.reshape({-1, A_high.size(-1)});

    int M = A_main.size(0);
    int K_main = A_main.size(1);
    int K_high = A_high.size(1);
    int N = B_main.size(0);

    TORCH_CHECK(K_main == B_main.size(1), "K_main mismatch");
    TORCH_CHECK(K_high == B_high.size(1), "K_high mismatch");

    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A_main.device()));
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Build mainloop params (TMA descriptors)
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

    auto mainloop_params_main = CollectiveMainloop::to_underlying_arguments(problem_shape_main, mainloop_args_main, nullptr);
    auto mainloop_params_high = CollectiveMainloop::to_underlying_arguments(problem_shape_high, mainloop_args_high, nullptr);

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

    // Simple grid for debug kernel
    int grid_m = cute::ceil_div(M, get<0>(TileShape_MNK{}));
    int grid_n = cute::ceil_div(N, get<1>(TileShape_MNK{}));
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(MaxThreadsPerBlock_, 1, 1);

    // Dynamic smem test: request full FusedSmemSize
    int smem_size = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));
    printf("[host] smem_size = %d bytes (%.1f KB)\n", smem_size, smem_size / 1024.0);

    auto err = cudaFuncSetAttribute(fused_gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) {
        printf("[host] cudaFuncSetAttribute FAILED: %s\n", cudaGetErrorString(err));
        // Try with 0 smem
        smem_size = 0;
    }

    fused_gemm_kernel<<<grid, block, smem_size, stream>>>(kernel_params);

    auto launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess, "fused_gemm_kernel launch failed: ", cudaGetErrorString(launch_err));

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
