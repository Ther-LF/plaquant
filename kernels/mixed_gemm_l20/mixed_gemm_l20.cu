#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/integer_subbyte.h"
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

// High precision path: INT8 x INT8 → INT32
using ElementA_High = int8_t;
using ElementB_High = int8_t;

// Low precision path: INT4 x INT4 → INT32
using ElementA_Low = cutlass::int4b_t;
using ElementB_Low = cutlass::int4b_t;

// Output and accumulator
using ElementC = cutlass::half_t;
using ElementAccumulator = int32_t;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using ArchTag = cutlass::arch::Sm80;
using OpClass = cutlass::arch::OpClassTensorOp;

// ============================================================================
// High precision GEMM config: INT8, InstructionShape<16,8,32>
// ============================================================================
using ThreadblockShape_High = cutlass::gemm::GemmShape<64, 64, 64>;
using WarpShape_High = cutlass::gemm::GemmShape<32, 32, 64>;
using InstructionShape_High = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int kStages_High = 4;
constexpr int kAlignmentA_High = 16;   // 128 bits / 8 bits = 16
constexpr int kAlignmentB_High = 16;

using GemmHigh = cutlass::gemm::device::Gemm<
    ElementA_High, LayoutA,
    ElementB_High, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OpClass, ArchTag,
    ThreadblockShape_High, WarpShape_High, InstructionShape_High,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages_High, kAlignmentA_High, kAlignmentB_High>;

// ============================================================================
// Low precision GEMM config: INT4, InstructionShape<16,8,64>
// ============================================================================
using ThreadblockShape_Low = cutlass::gemm::GemmShape<64, 64, 128>;
using WarpShape_Low = cutlass::gemm::GemmShape<32, 32, 128>;
using InstructionShape_Low = cutlass::gemm::GemmShape<16, 8, 64>;
constexpr int kStages_Low = 5;   // More stages for low (14 K-iterations benefit from deeper pipeline)
constexpr int kAlignmentA_Low = 32;    // 128 bits / 4 bits = 32
constexpr int kAlignmentB_Low = 32;

using GemmLow = cutlass::gemm::device::Gemm<
    ElementA_Low, LayoutA,
    ElementB_Low, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OpClass, ArchTag,
    ThreadblockShape_Low, WarpShape_Low, InstructionShape_Low,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages_Low, kAlignmentA_Low, kAlignmentB_Low>;

// ============================================================================
// Extract kernel types
// ============================================================================
using GemmKernelHigh = typename GemmHigh::GemmKernel;
using MmaHigh = typename GemmKernelHigh::Mma;
using EpilogueHigh = typename GemmKernelHigh::Epilogue;
using OutputOp = typename EpilogueHigh::OutputOp;

using GemmKernelLow = typename GemmLow::GemmKernel;
using MmaLow = typename GemmKernelLow::Mma;

using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Both MMA types must have the same thread count and accumulator fragment type
static_assert(GemmKernelHigh::kThreadCount == GemmKernelLow::kThreadCount,
    "High and Low precision kernels must have the same thread count");
static constexpr int kThreadCount = GemmKernelHigh::kThreadCount;

// ============================================================================
// Custom Fused Mixed-Precision Kernel
// ============================================================================

struct FusedMixedGemmKernel {

    struct Params {
        cutlass::gemm::GemmCoord problem_size;  // (M, N, max(K_low, K_high))
        cutlass::gemm::GemmCoord grid_tiled_shape;
        int swizzle_log_tile;

        // High precision (INT8)
        typename MmaHigh::IteratorA::Params params_A_high;
        typename MmaHigh::IteratorA::TensorRef ref_A_high;
        typename MmaHigh::IteratorB::Params params_B_high;
        typename MmaHigh::IteratorB::TensorRef ref_B_high;
        int gemm_k_iterations_high;
        int k_high;

        // Low precision (INT4)
        typename MmaLow::IteratorA::Params params_A_low;
        typename MmaLow::IteratorA::TensorRef ref_A_low;
        typename MmaLow::IteratorB::Params params_B_low;
        typename MmaLow::IteratorB::TensorRef ref_B_low;
        int gemm_k_iterations_low;
        int k_low;

        // Output (shared epilogue)
        typename EpilogueHigh::OutputTileIterator::Params params_D;
        typename EpilogueHigh::OutputTileIterator::TensorRef ref_D;
        typename OutputOp::Params output_op;

        CUTLASS_HOST_DEVICE
        Params() : swizzle_log_tile(0) {}

        CUTLASS_HOST_DEVICE
        Params(
            cutlass::gemm::GemmCoord problem_size_,
            cutlass::gemm::GemmCoord grid_tiled_shape_,
            typename MmaHigh::IteratorA::TensorRef ref_A_high_,
            typename MmaHigh::IteratorB::TensorRef ref_B_high_,
            int k_high_,
            typename MmaLow::IteratorA::TensorRef ref_A_low_,
            typename MmaLow::IteratorB::TensorRef ref_B_low_,
            int k_low_,
            typename EpilogueHigh::OutputTileIterator::TensorRef ref_D_,
            typename OutputOp::Params output_op_ = typename OutputOp::Params()
        ) :
            problem_size(problem_size_),
            grid_tiled_shape(grid_tiled_shape_),
            swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape_)),
            params_A_high(ref_A_high_.layout()),
            ref_A_high(ref_A_high_),
            params_B_high(ref_B_high_.layout()),
            ref_B_high(ref_B_high_),
            gemm_k_iterations_high((k_high_ + ThreadblockShape_High::kK - 1) / ThreadblockShape_High::kK),
            k_high(k_high_),
            params_A_low(ref_A_low_.layout()),
            ref_A_low(ref_A_low_),
            params_B_low(ref_B_low_.layout()),
            ref_B_low(ref_B_low_),
            gemm_k_iterations_low((k_low_ + ThreadblockShape_Low::kK - 1) / ThreadblockShape_Low::kK),
            k_low(k_low_),
            params_D(ref_D_.layout()),
            ref_D(ref_D_),
            output_op(output_op_) {}
    };

    // SharedStorage: union of both MMA types + epilogue
    union SharedStorage {
        typename MmaLow::SharedStorage mma_low;
        typename MmaHigh::SharedStorage mma_high;
        typename EpilogueHigh::SharedStorage epilogue;
    };

    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage) {
        ThreadblockSwizzle threadblock_swizzle;
        cutlass::gemm::GemmCoord threadblock_tile_offset =
            threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
            params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
            return;
        }

        int thread_idx = threadIdx.x;
        int warp_idx = __shfl_sync(0xFFFFFFFF, threadIdx.x / 32, 0);
        int lane_idx = threadIdx.x % 32;

        // Tile offsets for A (M dim) and B (N dim)
        cutlass::MatrixCoord tb_offset_A{
            threadblock_tile_offset.m() * MmaHigh::Shape::kM, 0};
        cutlass::MatrixCoord tb_offset_B{
            0, threadblock_tile_offset.n() * MmaHigh::Shape::kN};

        // Initialize accumulator (shared between both phases)
        typename MmaHigh::FragmentC accumulators;
        accumulators.clear();

        // ====================================================================
        // Phase 1: Low precision GEMM (INT4 × INT4, larger K)
        // ====================================================================
        if (params.gemm_k_iterations_low > 0) {
            typename MmaLow::IteratorA iterator_A_low(
                params.params_A_low,
                params.ref_A_low.data(),
                {params.problem_size.m(), params.k_low},
                thread_idx,
                tb_offset_A);

            typename MmaLow::IteratorB iterator_B_low(
                params.params_B_low,
                params.ref_B_low.data(),
                {params.k_low, params.problem_size.n()},
                thread_idx,
                tb_offset_B);

            MmaLow mma_low(shared_storage.mma_low, thread_idx, warp_idx, lane_idx);

            // Accumulate directly into shared accumulator — no separate accum_low
            mma_low(params.gemm_k_iterations_low, accumulators, iterator_A_low, iterator_B_low, accumulators);
        }

        // ====================================================================
        // Phase 2: High precision GEMM (INT8 × INT8, smaller K)
        // ====================================================================
        if (params.gemm_k_iterations_high > 0) {
            typename MmaHigh::IteratorA iterator_A_high(
                params.params_A_high,
                params.ref_A_high.data(),
                {params.problem_size.m(), params.k_high},
                thread_idx,
                tb_offset_A);

            typename MmaHigh::IteratorB iterator_B_high(
                params.params_B_high,
                params.ref_B_high.data(),
                {params.k_high, params.problem_size.n()},
                thread_idx,
                tb_offset_B);

            MmaHigh mma_high(shared_storage.mma_high, thread_idx, warp_idx, lane_idx);
            mma_high(params.gemm_k_iterations_high, accumulators, iterator_A_high, iterator_B_high, accumulators);
        }

        // ====================================================================
        // Epilogue
        // ====================================================================
        OutputOp output_op(params.output_op);

        threadblock_tile_offset =
            threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        cutlass::MatrixCoord threadblock_offset(
            threadblock_tile_offset.m() * MmaHigh::Shape::kM,
            threadblock_tile_offset.n() * MmaHigh::Shape::kN);

        EpilogueHigh epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

        typename EpilogueHigh::OutputTileIterator iterator_D(
            params.params_D,
            params.ref_D.data(),
            params.problem_size.mn(),
            thread_idx,
            threadblock_offset);

        epilogue(output_op, iterator_D, accumulators, iterator_D);
    }
};

// ============================================================================
// Global kernel entry
// ============================================================================

__global__ void __launch_bounds__(kThreadCount)
fused_mixed_gemm_kernel(FusedMixedGemmKernel::Params params) {
    extern __shared__ char smem_buf[];
    auto& shared_storage = *reinterpret_cast<FusedMixedGemmKernel::SharedStorage*>(smem_buf);
    FusedMixedGemmKernel kernel;
    kernel(params, shared_storage);
}

// ============================================================================
// Host launch
// ============================================================================

torch::Tensor fused_mixed_gemm(
    torch::Tensor A_low,   // (M, K_low) INT4 packed as INT8 (each byte = 2 elements)
    torch::Tensor B_low,   // (N, K_low) INT4 packed as INT8
    torch::Tensor A_high,  // (M, K_high) INT8
    torch::Tensor B_high   // (N, K_high) INT8
) {
    int M = A_high.size(0);
    int N = B_high.size(0);
    int K_high = A_high.size(1);
    // For INT4: each byte stores 2 elements, so actual K_low = A_low.size(1) * 2
    int K_low = A_low.size(1) * 2;

    TORCH_CHECK(A_low.dtype() == torch::kInt8, "A_low must be INT8 (packed INT4)");
    TORCH_CHECK(B_low.dtype() == torch::kInt8, "B_low must be INT8 (packed INT4)");
    TORCH_CHECK(A_high.dtype() == torch::kInt8);
    TORCH_CHECK(B_high.dtype() == torch::kInt8);
    TORCH_CHECK(A_low.is_contiguous());
    TORCH_CHECK(B_low.is_contiguous());
    TORCH_CHECK(A_high.is_contiguous());
    TORCH_CHECK(B_high.is_contiguous());

    auto D = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(A_high.device()));

    // TensorRef for high precision (INT8)
    using TensorRefA_High = typename MmaHigh::IteratorA::TensorRef;
    using TensorRefB_High = typename MmaHigh::IteratorB::TensorRef;
    using TensorRefD = typename EpilogueHigh::OutputTileIterator::TensorRef;

    TensorRefA_High ref_A_high(
        reinterpret_cast<ElementA_High*>(A_high.data_ptr()),
        LayoutA::packed({M, K_high}));
    TensorRefB_High ref_B_high(
        reinterpret_cast<ElementB_High*>(B_high.data_ptr()),
        LayoutB(K_high));

    // TensorRef for low precision (INT4)
    // INT4 stored packed: physical bytes = K_low / 2
    using TensorRefA_Low = typename MmaLow::IteratorA::TensorRef;
    using TensorRefB_Low = typename MmaLow::IteratorB::TensorRef;

    TensorRefA_Low ref_A_low(
        reinterpret_cast<ElementA_Low*>(A_low.data_ptr()),
        LayoutA::packed({M, K_low}));
    TensorRefB_Low ref_B_low(
        reinterpret_cast<ElementB_Low*>(B_low.data_ptr()),
        LayoutB(K_low));

    TensorRefD ref_D(
        reinterpret_cast<ElementC*>(D.data_ptr()),
        LayoutC::packed({M, N}));

    // Grid
    cutlass::gemm::GemmCoord problem_size(M, N, K_low + K_high);
    cutlass::gemm::GemmCoord grid_tiled_shape(
        (M + ThreadblockShape_High::kM - 1) / ThreadblockShape_High::kM,
        (N + ThreadblockShape_High::kN - 1) / ThreadblockShape_High::kN,
        1);

    typename OutputOp::Params output_op_params(
        ElementCompute(1.0f), ElementCompute(0.0f));

    FusedMixedGemmKernel::Params params(
        problem_size, grid_tiled_shape,
        ref_A_high, ref_B_high, K_high,
        ref_A_low, ref_B_low, K_low,
        ref_D, output_op_params);

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

}  // namespace mixed_gemm_l20

// ============================================================================
// Baseline: 2x CUTLASS GEMM (same SM80 tensor core) + add
// ============================================================================

torch::Tensor baseline_cutlass_mixed_gemm(
    torch::Tensor A_low,   // (M, K_low/2) packed INT4
    torch::Tensor B_low,   // (N, K_low/2) packed INT4
    torch::Tensor A_high,  // (M, K_high) INT8
    torch::Tensor B_high   // (N, K_high) INT8
) {
    using namespace mixed_gemm_l20;

    int M = A_high.size(0);
    int N = B_high.size(0);
    int K_high = A_high.size(1);
    int K_low = A_low.size(1) * 2;

    // Output buffers
    auto D_high = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(A_high.device()));
    auto D_low = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(A_high.device()));

    // === Launch 1: INT8 GEMM (high precision path) ===
    {
        using TensorRefA = typename MmaHigh::IteratorA::TensorRef;
        using TensorRefB = typename MmaHigh::IteratorB::TensorRef;
        using TensorRefD = typename EpilogueHigh::OutputTileIterator::TensorRef;

        TensorRefA ref_A(reinterpret_cast<ElementA_High*>(A_high.data_ptr()), LayoutA::packed({M, K_high}));
        TensorRefB ref_B(reinterpret_cast<ElementB_High*>(B_high.data_ptr()), LayoutB(K_high));
        TensorRefD ref_D(reinterpret_cast<ElementC*>(D_high.data_ptr()), LayoutC::packed({M, N}));

        typename GemmHigh::Arguments args(
            {M, N, K_high},
            ref_A, ref_B, ref_D, ref_D,
            {ElementCompute(1.0f), ElementCompute(0.0f)});

        GemmHigh gemm_high;
        gemm_high.initialize(args);
        gemm_high();
    }

    // === Launch 2: INT4 GEMM (low precision path) ===
    {
        using TensorRefA = typename MmaLow::IteratorA::TensorRef;
        using TensorRefB = typename MmaLow::IteratorB::TensorRef;
        using TensorRefD = typename EpilogueHigh::OutputTileIterator::TensorRef;

        TensorRefA ref_A(reinterpret_cast<ElementA_Low*>(A_low.data_ptr()), LayoutA::packed({M, K_low}));
        TensorRefB ref_B(reinterpret_cast<ElementB_Low*>(B_low.data_ptr()), LayoutB(K_low));
        TensorRefD ref_D(reinterpret_cast<ElementC*>(D_low.data_ptr()), LayoutC::packed({M, N}));

        typename GemmLow::Arguments args(
            {M, N, K_low},
            ref_A, ref_B, ref_D, ref_D,
            {ElementCompute(1.0f), ElementCompute(0.0f)});

        GemmLow gemm_low;
        gemm_low.initialize(args);
        gemm_low();
    }

    // === Launch 3: elementwise add ===
    return D_high.add_(D_low);
}

// ============================================================================
// Python binding
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mixed_gemm", &mixed_gemm_l20::fused_mixed_gemm,
          "Fused mixed-precision GEMM: INT4(low) + INT8(high) single launch");
    m.def("baseline_cutlass_mixed_gemm", &baseline_cutlass_mixed_gemm,
          "Baseline: 2x CUTLASS GEMM (INT4 + INT8) + add");
}
