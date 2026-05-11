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
fused_gemm_kernel(FusedKernelParams params) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    // Simple store: each thread writes one or more elements
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = params.M * params.N;
    // Each thread handles multiple elements with stride
    for (int idx = tid; idx < total; idx += gridDim.x * blockDim.x) {
        params.D[idx] = ElementOutput(0.0f);
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

    // No dynamic smem needed for this debug version
    fused_gemm_kernel<<<grid, block, 0, stream>>>(kernel_params);

    auto err = cudaGetLastError();
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
