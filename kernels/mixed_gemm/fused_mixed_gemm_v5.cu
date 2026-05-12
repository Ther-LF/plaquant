/***************************************************************************************************
 * Fused Mixed-Precision GEMM v5 — Fork of CUTLASS GemmKernel::operator()
 *
 * Uses CUTLASS's full TMA mainloop + TMA epilogue for both phases.
 * The key insight: previous versions were slow because of hand-written epilogue
 * (scatter store). This version uses CUTLASS CollectiveEpilogue::store() which
 * does acc → SMEM → TMA store (fully pipelined, coalesced).
 *
 * EVT epilogue: D = DequantEVT(acc) + beta * C
 *   Phase 1: beta=0, D=workspace → workspace = dequant_main(acc)
 *   Phase 2: beta=1, C=workspace, D=output → output = dequant_high(acc) + workspace
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
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/device_kernel.h"

using namespace cute;

#define CUTLASS_ARCH_MMA_SM90_SUPPORTED 1

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// =============================================================================
// Configuration
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
constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

// =============================================================================
// Mainloop (same as baseline)
// =============================================================================

// =============================================================================
// Mainloop — uses StageCountAutoCarveout with the epilogue smem size
// (defined after epilogue is built, see below)
// =============================================================================

// Forward declaration pattern: build epilogue first, then mainloop with carveout

using namespace cutlass::epilogue::fusion;

// === Leaf nodes (same as baseline) ===
using Leaf_sx       = Sm90ColBroadcast<0, TileShape_MNK, ElementOutput, ElementCompute>;
using Leaf_sw       = Sm90RowBroadcast<0, TileShape_MNK, ElementOutput, ElementCompute>;
using Leaf_neg_zero = Sm90ColBroadcast<0, TileShape_MNK, ElementOutput, ElementCompute>;
using Leaf_colsum   = Sm90RowBroadcast<0, TileShape_MNK, ElementCompute, ElementCompute>;

// bias_term = neg_zero[m] * colsum[n]
using BiasCompute = Sm90EVT<
    Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
    Leaf_neg_zero, Leaf_colsum>;

// corrected_acc = acc + bias_term
using CorrectedAcc = Sm90EVT<
    Sm90Compute<cutlass::plus, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
    Sm90AccFetch, BiasCompute>;

// scale_product = s_x[m] * s_w[n]
using ScaleProduct = Sm90EVT<
    Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
    Leaf_sx, Leaf_sw>;

// dequant: D = scale_product * corrected_acc (same as baseline DequantEVT)
using PlainDequantEVT = Sm90EVT<
    Sm90Compute<cutlass::multiplies, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
    ScaleProduct, CorrectedAcc>;

// =============================================================================
// Build GemmKernel with PlainDequantEVT (same as baseline, for 2-launch+add)
// =============================================================================

using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    ElementOutput, cutlass::layout::RowMajor, AlignmentOutput,
    ElementOutput, cutlass::layout::RowMajor, AlignmentOutput,
    cutlass::epilogue::TmaWarpSpecialized,
    PlainDequantEVT
>::CollectiveOp;

// Shared memory carveout for epilogue
constexpr int EpilogueSmemBytes = static_cast<int>(sizeof(typename EpilogueOp::SharedStorage));

// Build mainloop with epilogue smem carveout
using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, AlignmentA,
    ElementB, cutlass::layout::ColumnMajor, AlignmentB,
    ElementAccum,
    TileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<EpilogueSmemBytes>,
    cutlass::gemm::KernelTmaWarpSpecialized
>::CollectiveOp;

// Full GemmKernel type
using FusedGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopOp, EpilogueOp>;
using FusedGemm = cutlass::gemm::device::GemmUniversalAdapter<FusedGemmKernel>;

// =============================================================================
// Helper: build FusedEVT arguments for one phase
// =============================================================================

static typename FusedGemmKernel::CollectiveEpilogue::FusionCallbacks::Arguments
make_dequant_evt_args(
    ElementOutput const* s_x, ElementOutput const* s_w,
    ElementOutput const* neg_zero, float const* colsum)
{
    // PlainDequantEVT = Sm90EVT<Compute<mul>, ScaleProduct, CorrectedAcc>
    // Same as baseline DequantEVT args

    using FusionArgs = typename FusedGemmKernel::CollectiveEpilogue::FusionCallbacks::Arguments;

    return FusionArgs{
        // ScaleProduct = {Leaf_sx, Leaf_sw, mul_op}
        {
            {s_x},
            {s_w},
            {}
        },
        // CorrectedAcc = {AccFetch, BiasCompute, plus_op}
        {
            {},
            {
                {neg_zero},
                {colsum},
                {}
            },
            {}
        },
        {}  // mul op (top)
    };
}

// =============================================================================
// Host-side: run two CUTLASS GEMMs with FusedEVT epilogue
// Phase 1: beta=0, D=workspace
// Phase 2: beta=1, C=workspace, D=output
// =============================================================================

torch::Tensor fused_mixed_gemm_v5(
    torch::Tensor A_main, torch::Tensor B_main,
    torch::Tensor A_high, torch::Tensor B_high,
    torch::Tensor s_x_m, torch::Tensor s_w_m, torch::Tensor neg_zero_m, torch::Tensor colsum_m,
    torch::Tensor s_x_h, torch::Tensor s_w_h, torch::Tensor neg_zero_h, torch::Tensor colsum_h)
{
    TORCH_CHECK(A_main.is_cuda() && B_main.is_cuda() && A_high.is_cuda() && B_high.is_cuda(),
                "All inputs must be on CUDA");
    TORCH_CHECK(A_main.dtype() == torch::kUInt8, "A_main must be uint8");
    TORCH_CHECK(B_main.dtype() == torch::kInt8, "B_main must be int8");
    TORCH_CHECK(A_high.dtype() == torch::kUInt8, "A_high must be uint8");
    TORCH_CHECK(B_high.dtype() == torch::kInt8, "B_high must be int8");
    TORCH_CHECK(A_main.is_contiguous() && B_main.is_contiguous(), "main must be contiguous");
    TORCH_CHECK(A_high.is_contiguous() && B_high.is_contiguous(), "high must be contiguous");

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
    auto workspace = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A_main.device()));
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    using StrideA = typename FusedGemmKernel::StrideA;
    using StrideB = typename FusedGemmKernel::StrideB;
    using StrideC = typename FusedGemmKernel::StrideC;
    using StrideD = typename FusedGemmKernel::StrideD;

    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));

    auto run_phase = [&](
        void const* ptr_A, void const* ptr_B, int K,
        ElementOutput const* s_x, ElementOutput const* s_w,
        ElementOutput const* neg_zero, float const* colsum,
        ElementOutput* D_ptr)
    {
        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));

        auto fusion_args = make_dequant_evt_args(s_x, s_w, neg_zero, colsum);

        typename FusedGemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {reinterpret_cast<ElementA const*>(ptr_A), stride_A,
             reinterpret_cast<ElementB const*>(ptr_B), stride_B},
            {fusion_args, nullptr, stride_C, D_ptr, stride_D}
        };

        FusedGemm gemm_op;
        auto status = gemm_op.can_implement(args);
        TORCH_CHECK(status == cutlass::Status::kSuccess,
                    "can_implement failed: ", cutlass::cutlassGetStatusString(status));

        size_t ws_size = FusedGemm::get_workspace_size(args);
        auto ws = torch::empty({static_cast<int64_t>(ws_size)},
                               torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

        status = gemm_op.initialize(args, ws.data_ptr(), stream);
        TORCH_CHECK(status == cutlass::Status::kSuccess,
                    "initialize failed: ", cutlass::cutlassGetStatusString(status));

        status = gemm_op.run(stream);
        TORCH_CHECK(status == cutlass::Status::kSuccess,
                    "run failed: ", cutlass::cutlassGetStatusString(status));
    };

    // Phase 1: GEMM main → workspace (with dequant EVT)
    run_phase(A_main.data_ptr(), B_main.data_ptr(), K_main,
              reinterpret_cast<ElementOutput const*>(s_x_m.data_ptr()),
              reinterpret_cast<ElementOutput const*>(s_w_m.data_ptr()),
              reinterpret_cast<ElementOutput const*>(neg_zero_m.data_ptr()),
              reinterpret_cast<float const*>(colsum_m.data_ptr()),
              reinterpret_cast<ElementOutput*>(workspace.data_ptr()));

    // Phase 2: GEMM high → output (with dequant EVT)
    run_phase(A_high.data_ptr(), B_high.data_ptr(), K_high,
              reinterpret_cast<ElementOutput const*>(s_x_h.data_ptr()),
              reinterpret_cast<ElementOutput const*>(s_w_h.data_ptr()),
              reinterpret_cast<ElementOutput const*>(neg_zero_h.data_ptr()),
              reinterpret_cast<float const*>(colsum_h.data_ptr()),
              reinterpret_cast<ElementOutput*>(output.data_ptr()));

    // Phase 3: combine — output += workspace
    output.add_(workspace);

    if (A_main_shape.size() == 3) {
        output = output.reshape({A_main_shape[0], A_main_shape[1], N});
    }
    return output;
}

#else
torch::Tensor fused_mixed_gemm_v5(
    torch::Tensor A_main, torch::Tensor B_main,
    torch::Tensor A_high, torch::Tensor B_high,
    torch::Tensor s_x_m, torch::Tensor s_w_m, torch::Tensor neg_zero_m, torch::Tensor colsum_m,
    torch::Tensor s_x_h, torch::Tensor s_w_h, torch::Tensor neg_zero_h, torch::Tensor colsum_h) {
    TORCH_CHECK(false, "SM90 not supported"); return {};
}
#endif
