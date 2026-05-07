/***************************************************************************************************
 * Baseline Mixed-Precision GEMM for ResQ
 *
 * Two separate GEMM kernels:
 *   1. UINT8 × INT8 → INT32 → FP16 (high-precision 256-dim portion)
 *   2. UINT4 × INT4 → INT32 → FP16 (main 1792-dim portion)
 *
 * Both use CUTLASS 3.x CollectiveBuilder for Hopper SM90a WGMMA.
 * Epilogue: convert INT32 accumulator to FP16 output.
 *
 * Dequantization (scale, zero, colsum_w bias) is done OUTSIDE the kernel
 * in Python for the baseline. The fused kernel will do it in epilogue.
 *
 * Target: NVIDIA H20 (SM90a)
 **************************************************************************************************/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// =============================================================================
// GEMM Type Definitions
// =============================================================================

// --- UINT8 × INT8 → INT32 GEMM (high-precision portion) ---
// A = activation (uint8, row-major), B = weight (int8, column-major)
// Both must be K-major for SM90 GMMA S32U8S8 ops
using GemmU8S8_CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    uint8_t, cutlass::layout::RowMajor, 16,          // A: uint8, row-major (K-major), alignment=16
    int8_t, cutlass::layout::ColumnMajor, 16,         // B: int8, col-major (K-major), alignment=16
    int32_t,                                           // Accumulator: int32
    Shape<_128, _128, _128>, Shape<_1, _1, _1>,       // Tile shape, cluster shape
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmU8S8_Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _128>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    int32_t, int32_t,
    int32_t, cutlass::layout::RowMajor, 4,
    int32_t, cutlass::layout::RowMajor, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using GemmU8S8_Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    GemmU8S8_CollectiveOp,
    GemmU8S8_Epilogue
>;

using GemmU8S8 = cutlass::gemm::device::GemmUniversalAdapter<GemmU8S8_Kernel>;

// --- INT8 × INT8 → INT32 GEMM (main portion: int4 sign-extended to int8) ---
// Temporary: until we implement native INT4 path, main uses int8 with expanded data
using GemmS8S8_CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    int8_t, cutlass::layout::RowMajor, 16,
    int8_t, cutlass::layout::ColumnMajor, 16,
    int32_t,
    Shape<_128, _128, _128>, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmS8S8_Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _128>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    int32_t, int32_t,
    int32_t, cutlass::layout::RowMajor, 4,
    int32_t, cutlass::layout::RowMajor, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using GemmS8S8_Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    GemmS8S8_CollectiveOp,
    GemmS8S8_Epilogue
>;

using GemmS8S8 = cutlass::gemm::device::GemmUniversalAdapter<GemmS8S8_Kernel>;

// =============================================================================
// Launch helpers
// =============================================================================

template <typename Gemm>
cutlass::Status run_gemm(
    int M, int N, int K,
    void const* ptr_A,
    void const* ptr_B,
    void* ptr_D,
    cudaStream_t stream)
{
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},           // problem size (M, N, K, batch)
        {                        // mainloop args
            reinterpret_cast<typename Gemm::GemmKernel::CollectiveMainloop::ElementA const*>(ptr_A),
            stride_A,
            reinterpret_cast<typename Gemm::GemmKernel::CollectiveMainloop::ElementB const*>(ptr_B),
            stride_B
        },
        {                        // epilogue args
            {1, 0},              // alpha=1, beta=0 (C = A*B, no accumulate)
            nullptr,             // C (not used since beta=0)
            stride_D,
            reinterpret_cast<typename Gemm::GemmKernel::CollectiveEpilogue::ElementD*>(ptr_D),
            stride_D
        }
    };

    Gemm gemm_op;
    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    // Get workspace
    size_t workspace_size = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                   torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    status = gemm_op.initialize(args, workspace.data_ptr(), stream);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    status = gemm_op.run(stream);
    return status;
}

// =============================================================================
// PyTorch-facing functions
// =============================================================================

/// UINT8 × INT8 GEMM (high portion): (M, K) × (N, K)^T → (M, N) in INT32
/// A = unsigned activation [0, 255], B = signed weight [-128, 127]
torch::Tensor gemm_u8s8(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kUInt8, "A must be uint8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "K dimensions must match");

    A = A.contiguous();
    B = B.contiguous();

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kInt32).device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto status = run_gemm<GemmU8S8>(M, N, K, A.data_ptr(), B.data_ptr(), output.data_ptr(), stream);

    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS U8S8 GEMM failed: ", cutlass::cutlassGetStatusString(status));
    return output;
}

/// INT8 × INT8 GEMM (main portion): (M, K) × (N, K)^T → (M, N) in INT32
/// For INT4 data sign-extended to INT8. Temporary until native INT4 path.
torch::Tensor gemm_s8s8(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "K dimensions must match");

    A = A.contiguous();
    B = B.contiguous();

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kInt32).device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto status = run_gemm<GemmS8S8>(M, N, K, A.data_ptr(), B.data_ptr(), output.data_ptr(), stream);

    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS S8S8 GEMM failed: ", cutlass::cutlassGetStatusString(status));
    return output;
}

#else
// Fallback for non-SM90 compilation
torch::Tensor gemm_u8s8(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(false, "SM90 not supported in this build");
    return torch::Tensor();
}
torch::Tensor gemm_s8s8(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(false, "SM90 not supported in this build");
    return torch::Tensor();
}
#endif

// =============================================================================
// Python binding
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_u8s8", &gemm_u8s8,
          "UINT8 x INT8 GEMM → INT32 (SM90 WGMMA, for high-precision portion)",
          py::arg("A"), py::arg("B"));
    m.def("gemm_s8s8", &gemm_s8s8,
          "INT8 x INT8 GEMM → INT32 (SM90 WGMMA, for main portion with expanded int4)",
          py::arg("A"), py::arg("B"));
}
