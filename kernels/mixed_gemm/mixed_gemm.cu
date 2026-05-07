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

// --- INT8 × INT8 → INT32 GEMM ---
// A = activation (int8, row-major), B = weight (int8, column-major)
// C/D = int32 output
// Note: For unsigned activation, Python shifts q_x_unsigned - 128 to signed range before calling.
using GemmS8S8_CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    int8_t, cutlass::layout::RowMajor, 16,          // A: int8, row-major, alignment=16
    int8_t, cutlass::layout::ColumnMajor, 16,        // B: int8, col-major, alignment=16
    int32_t,                                          // Accumulator: int32
    Shape<_128, _128, _128>, Shape<_1, _1, _1>,      // Tile shape, cluster shape
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmS8S8_Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _128>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    int32_t, int32_t,                                 // Accumulator type, compute type
    int32_t, cutlass::layout::RowMajor, 4,            // C: int32 output, row-major
    int32_t, cutlass::layout::RowMajor, 4,            // D: int32 output, row-major
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using GemmS8S8_Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    GemmS8S8_CollectiveOp,
    GemmS8S8_Epilogue
>;

using GemmS8S8 = cutlass::gemm::device::GemmUniversalAdapter<GemmS8S8_Kernel>;

// --- INT4 × INT4 → INT32 GEMM ---
// A = activation (int4, row-major), B = weight (int4, column-major)
// Note: For unsigned activation, Python shifts q_x_unsigned - 8 to signed range before calling.
using GemmS4S4_CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::int4b_t, cutlass::layout::RowMajor, 32,   // A: int4, alignment=32 (128 bits / 4 bits)
    cutlass::int4b_t, cutlass::layout::ColumnMajor, 32, // B: int4, alignment=32
    int32_t,
    Shape<_128, _128, _256>, Shape<_1, _1, _1>,         // Larger K tile for int4
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmS4S4_Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _256>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    int32_t, int32_t,
    int32_t, cutlass::layout::RowMajor, 4,
    int32_t, cutlass::layout::RowMajor, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using GemmS4S4_Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    GemmS4S4_CollectiveOp,
    GemmS4S4_Epilogue
>;

using GemmS4S4 = cutlass::gemm::device::GemmUniversalAdapter<GemmS4S4_Kernel>;

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

/// INT8 × INT8 GEMM: (M, K) × (N, K)^T → (M, N) in INT32
/// A is row-major int8 (shifted from unsigned), B is int8 weight (stored row-major, used as col-major)
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

    // B is (N, K) row-major in memory = (K, N) column-major → matches ColumnMajor layout
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto status = run_gemm<GemmS8S8>(
        M, N, K,
        A.data_ptr(),
        B.data_ptr(),
        output.data_ptr(),
        stream
    );

    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS S8S8 GEMM failed: ", cutlass::cutlassGetStatusString(status));

    return output;
}

/// INT4 × INT4 GEMM: (M, K) × (N, K)^T → (M, N) in INT32
/// A is packed int4 (2 elements per byte), B is packed int4
torch::Tensor gemm_s4s4(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    // A and B are stored as int8 with 2 int4 values packed per byte
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8 (packed int4)");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8 (packed int4)");

    A = A.contiguous();
    B = B.contiguous();

    // Actual dimensions: K is 2x the stored K (since 2 elements per byte)
    int M = A.size(0);
    int K = A.size(1) * 2;  // Each byte stores 2 int4 values
    int N = B.size(0);

    TORCH_CHECK(A.size(1) == B.size(1), "Packed K dimensions must match");

    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kInt32).device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto status = run_gemm<GemmS4S4>(
        M, N, K,
        A.data_ptr(),
        B.data_ptr(),
        output.data_ptr(),
        stream
    );

    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS S4S4 GEMM failed: ", cutlass::cutlassGetStatusString(status));

    return output;
}

#else
// Fallback for non-SM90 compilation
torch::Tensor gemm_s8s8(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(false, "SM90 not supported in this build");
    return torch::Tensor();
}
torch::Tensor gemm_s4s4(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(false, "SM90 not supported in this build");
    return torch::Tensor();
}
#endif

// =============================================================================
// Python binding
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_s8s8", &gemm_s8s8,
          "INT8 x INT8 GEMM → INT32 (CUTLASS SM90)",
          py::arg("A"), py::arg("B"));
    m.def("gemm_s4s4", &gemm_s4s4,
          "INT4 x INT4 GEMM → INT32 (CUTLASS SM90)",
          py::arg("A"), py::arg("B"));
}
