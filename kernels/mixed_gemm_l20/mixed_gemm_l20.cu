#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/util/host_tensor.h"

#include <cuda_fp16.h>
#include <torch/extension.h>

namespace mixed_gemm_l20 {

// ============================================================================
// Type definitions
// ============================================================================

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = int32_t;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using ArchTag = cutlass::arch::Sm80;  // SM89 is compatible with SM80 tensor ops
using OpClass = cutlass::arch::OpClassTensorOp;

// Tile configuration
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;  // INT8 tensor op

constexpr int kStages = 4;
constexpr int kAlignmentA = 16;  // 128 bits / 8 bits = 16 elements
constexpr int kAlignmentB = 16;

// ============================================================================
// GEMM type for baseline (single precision GEMM)
// ============================================================================

using GemmBaseline = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OpClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator,
        ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    kAlignmentA,
    kAlignmentB>;

// ============================================================================
// Fused Mixed GEMM Kernel
// ============================================================================

// We use the same MMA type for both low and high precision (both are INT8×INT8)
// The difference is just the K dimension (number of iterations)

// Extract Mma and Epilogue types from the device::Gemm type
using GemmKernel = typename GemmBaseline::GemmKernel;
using Mma = typename GemmKernel::Mma;
using Epilogue = typename GemmKernel::Epilogue;
using OutputOp = typename Epilogue::OutputOp;
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

static constexpr int kThreadCount = GemmKernel::kThreadCount;

// ============================================================================
// Custom Fused Kernel
// ============================================================================

struct FusedMixedGemmKernel {

    struct Params {
        cutlass::gemm::GemmCoord problem_size;  // (M, N, K_total) for grid computation
        cutlass::gemm::GemmCoord grid_tiled_shape;
        int swizzle_log_tile;

        // High precision operands
        typename Mma::IteratorA::Params params_A_high;
        typename Mma::IteratorA::TensorRef ref_A_high;
        typename Mma::IteratorB::Params params_B_high;
        typename Mma::IteratorB::TensorRef ref_B_high;
        int gemm_k_iterations_high;
        int k_high;

        // Low precision operands (INT4→INT8, same MMA type)
        typename Mma::IteratorA::Params params_A_low;
        typename Mma::IteratorA::TensorRef ref_A_low;
        typename Mma::IteratorB::Params params_B_low;
        typename Mma::IteratorB::TensorRef ref_B_low;
        int gemm_k_iterations_low;
        int k_low;

        // Output
        typename Epilogue::OutputTileIterator::Params params_D;
        typename Epilogue::OutputTileIterator::TensorRef ref_D;
        typename OutputOp::Params output_op;

        CUTLASS_HOST_DEVICE
        Params() : swizzle_log_tile(0) {}

        CUTLASS_HOST_DEVICE
        Params(
            cutlass::gemm::GemmCoord problem_size_,
            cutlass::gemm::GemmCoord grid_tiled_shape_,
            typename Mma::IteratorA::TensorRef ref_A_high_,
            typename Mma::IteratorB::TensorRef ref_B_high_,
            int k_high,
            typename Mma::IteratorA::TensorRef ref_A_low_,
            typename Mma::IteratorB::TensorRef ref_B_low_,
            int k_low,
            typename Epilogue::OutputTileIterator::TensorRef ref_D_,
            typename OutputOp::Params output_op_ = typename OutputOp::Params()
        ) :
            problem_size(problem_size_),
            grid_tiled_shape(grid_tiled_shape_),
            swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape_)),
            params_A_high(ref_A_high_.layout()),
            ref_A_high(ref_A_high_),
            params_B_high(ref_B_high_.layout()),
            ref_B_high(ref_B_high_),
            gemm_k_iterations_high((k_high + ThreadblockShape::kK - 1) / ThreadblockShape::kK),
            k_high(k_high),
            params_A_low(ref_A_low_.layout()),
            ref_A_low(ref_A_low_),
            params_B_low(ref_B_low_.layout()),
            ref_B_low(ref_B_low_),
            gemm_k_iterations_low((k_low + ThreadblockShape::kK - 1) / ThreadblockShape::kK),
            k_low(k_low),
            params_D(ref_D_.layout()),
            ref_D(ref_D_),
            output_op(output_op_) {}
    };

    using SharedStorage = union {
        typename Mma::SharedStorage mma;
        typename Epilogue::SharedStorage epilogue;
    };

    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage) {
        ThreadblockSwizzle threadblock_swizzle;
        cutlass::gemm::GemmCoord threadblock_tile_offset =
            threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        // Early exit
        if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
            params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
            return;
        }

        int thread_idx = threadIdx.x;
        int warp_idx = __shfl_sync(0xFFFFFFFF, threadIdx.x / 32, 0);
        int lane_idx = threadIdx.x % 32;

        // Compute tile offsets
        cutlass::MatrixCoord tb_offset_A{
            threadblock_tile_offset.m() * Mma::Shape::kM, 0};
        cutlass::MatrixCoord tb_offset_B{
            0, threadblock_tile_offset.n() * Mma::Shape::kN};

        // Initialize accumulator
        typename Mma::FragmentC accumulators;
        accumulators.clear();

        // ====================================================================
        // Phase 1: Low precision GEMM (K_low tiles)
        // ====================================================================
        if (params.gemm_k_iterations_low > 0) {
            typename Mma::IteratorA iterator_A_low(
                params.params_A_low,
                params.ref_A_low.data(),
                {params.problem_size.m(), params.k_low},
                thread_idx,
                tb_offset_A);

            typename Mma::IteratorB iterator_B_low(
                params.params_B_low,
                params.ref_B_low.data(),
                {params.k_low, params.problem_size.n()},
                thread_idx,
                tb_offset_B);

            Mma mma_low(shared_storage.mma, thread_idx, warp_idx, lane_idx);
            mma_low(params.gemm_k_iterations_low, accumulators, iterator_A_low, iterator_B_low, accumulators);
        }

        // ====================================================================
        // Phase 2: High precision GEMM (K_high tiles)
        // ====================================================================
        if (params.gemm_k_iterations_high > 0) {
            typename Mma::IteratorA iterator_A_high(
                params.params_A_high,
                params.ref_A_high.data(),
                {params.problem_size.m(), params.k_high},
                thread_idx,
                tb_offset_A);

            typename Mma::IteratorB iterator_B_high(
                params.params_B_high,
                params.ref_B_high.data(),
                {params.k_high, params.problem_size.n()},
                thread_idx,
                tb_offset_B);

            Mma mma_high(shared_storage.mma, thread_idx, warp_idx, lane_idx);
            mma_high(params.gemm_k_iterations_high, accumulators, iterator_A_high, iterator_B_high, accumulators);
        }

        // ====================================================================
        // Epilogue: write output
        // ====================================================================
        OutputOp output_op(params.output_op);

        threadblock_tile_offset =
            threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        cutlass::MatrixCoord threadblock_offset(
            threadblock_tile_offset.m() * Mma::Shape::kM,
            threadblock_tile_offset.n() * Mma::Shape::kN);

        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

        // No source (C) tensor - just write acc to D
        typename Epilogue::OutputTileIterator iterator_D(
            params.params_D,
            params.ref_D.data(),
            params.problem_size.mn(),
            thread_idx,
            threadblock_offset);

        epilogue(output_op, iterator_D, accumulators, iterator_D);
    }
};

// ============================================================================
// Global kernel entry point
// ============================================================================

__global__ void __launch_bounds__(kThreadCount)
fused_mixed_gemm_kernel(FusedMixedGemmKernel::Params params) {
    extern __shared__ char smem_buf[];
    auto& shared_storage = *reinterpret_cast<FusedMixedGemmKernel::SharedStorage*>(smem_buf);
    FusedMixedGemmKernel kernel;
    kernel(params, shared_storage);
}

// ============================================================================
// Host-side launch function
// ============================================================================

torch::Tensor fused_mixed_gemm(
    torch::Tensor A_low,   // (M, K_low) INT8, RowMajor
    torch::Tensor B_low,   // (N, K_low) INT8, contiguous — treated as (K_low, N) ColumnMajor
    torch::Tensor A_high,  // (M, K_high) INT8, RowMajor
    torch::Tensor B_high   // (N, K_high) INT8, contiguous — treated as (K_high, N) ColumnMajor
) {
    int M = A_low.size(0);
    int N = B_low.size(0);
    int K_low = A_low.size(1);
    int K_high = A_high.size(1);

    TORCH_CHECK(A_low.dtype() == torch::kInt8);
    TORCH_CHECK(B_low.dtype() == torch::kInt8);
    TORCH_CHECK(A_high.dtype() == torch::kInt8);
    TORCH_CHECK(B_high.dtype() == torch::kInt8);
    TORCH_CHECK(A_low.is_contiguous());
    TORCH_CHECK(B_low.is_contiguous());
    TORCH_CHECK(A_high.is_contiguous());
    TORCH_CHECK(B_high.is_contiguous());

    auto D = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(A_low.device()));

    // Construct CUTLASS tensor refs
    using TensorRefA = typename Mma::IteratorA::TensorRef;
    using TensorRefB = typename Mma::IteratorB::TensorRef;
    using TensorRefD = typename Epilogue::OutputTileIterator::TensorRef;

    TensorRefA ref_A_low(
        reinterpret_cast<ElementA*>(A_low.data_ptr()),
        LayoutA::packed({M, K_low}));
    TensorRefB ref_B_low(
        reinterpret_cast<ElementB*>(B_low.data_ptr()),
        LayoutB(N));  // ColumnMajor leading dim = N (stored as (N,K) contiguous)
    TensorRefA ref_A_high(
        reinterpret_cast<ElementA*>(A_high.data_ptr()),
        LayoutA::packed({M, K_high}));
    TensorRefB ref_B_high(
        reinterpret_cast<ElementB*>(B_high.data_ptr()),
        LayoutB(N));  // ColumnMajor leading dim = N
    TensorRefD ref_D(
        reinterpret_cast<ElementC*>(D.data_ptr()),
        LayoutC::packed({M, N}));

    // Grid shape
    cutlass::gemm::GemmCoord problem_size(M, N, K_low + K_high);
    cutlass::gemm::GemmCoord grid_tiled_shape(
        (M + ThreadblockShape::kM - 1) / ThreadblockShape::kM,
        (N + ThreadblockShape::kN - 1) / ThreadblockShape::kN,
        1);

    // Epilogue params: alpha=1, beta=0
    typename OutputOp::Params output_op_params(
        ElementCompute(1.0f), ElementCompute(0.0f));

    FusedMixedGemmKernel::Params params(
        problem_size,
        grid_tiled_shape,
        ref_A_high, ref_B_high, K_high,
        ref_A_low, ref_B_low, K_low,
        ref_D,
        output_op_params);

    // Launch
    dim3 grid(grid_tiled_shape.m(), grid_tiled_shape.n());
    dim3 block(kThreadCount);
    int smem_size = sizeof(FusedMixedGemmKernel::SharedStorage);

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            fused_mixed_gemm_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
    }

    fused_mixed_gemm_kernel<<<grid, block, smem_size>>>(params);

    return D;
}

// ============================================================================
// Baseline: two separate GEMMs + add
// ============================================================================

torch::Tensor baseline_mixed_gemm(
    torch::Tensor A_low,   // (M, K_low) INT8
    torch::Tensor B_low,   // (N, K_low) INT8
    torch::Tensor A_high,  // (M, K_high) INT8
    torch::Tensor B_high   // (N, K_high) INT8
) {
    // Use PyTorch's matmul for baseline (will use cuBLAS internally)
    auto A_low_f = A_low.to(torch::kFloat16);
    auto B_low_f = B_low.to(torch::kFloat16);  // (N, K_low)
    auto A_high_f = A_high.to(torch::kFloat16);
    auto B_high_f = B_high.to(torch::kFloat16);  // (N, K_high)

    auto out_low = torch::matmul(A_low_f, B_low_f.t());
    auto out_high = torch::matmul(A_high_f, B_high_f.t());

    return out_low + out_high;
}

}  // namespace mixed_gemm_l20

// ============================================================================
// Python binding
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mixed_gemm", &mixed_gemm_l20::fused_mixed_gemm,
          "Fused mixed-precision GEMM (single launch)");
    m.def("baseline_mixed_gemm", &mixed_gemm_l20::baseline_mixed_gemm,
          "Baseline mixed-precision GEMM (separate launches + add)");
}
