/***************************************************************************************************
 * Baseline Mixed-Precision GEMM for ResQ
 *
 * Two separate GEMM kernels with fused dequant epilogue (CUTLASS EVT):
 *   1. UINT8 × INT8 → FP16 (high-precision 256-dim portion)
 *   2. INT8 × INT8 → FP16 (main 1792-dim portion, int4 expanded to int8)
 *
 * Dequant formula (fused in epilogue via EVT):
 *   D[m,n] = s_x[m] * s_w[n] * (float(acc[m,n]) - zero_x[m] * colsum_w[n])
 *
 * EVT = Epilogue Visitor Tree: CUTLASS 3.x composable epilogue framework.
 * We build a tree of: ColBroadcast(per-row), RowBroadcast(per-col), Compute(op).
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
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#define CUTLASS_ARCH_MMA_SM90_SUPPORTED 1

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// =============================================================================
// Common type definitions
// =============================================================================

using TileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using ElementOutput = cutlass::half_t;
using ElementCompute = float;
constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;  // 8

// =============================================================================
// EVT Epilogue: D[m,n] = s_x[m] * s_w[n] * (acc[m,n] + (-zero_x[m]) * colsum_w[n])
//
// Tree structure (read bottom-up):
//   Store(D) = Compute<multiplies>(
//       Compute<multiplies>(ColBcast(s_x), RowBcast(s_w)),         // per-element scale
//       Compute<plus>(AccFetch, Compute<multiplies>(               // bias-corrected acc
//           ColBcast(neg_zero_x), RowBcast(colsum_w_float)))
//   )
// =============================================================================

using namespace cutlass::epilogue::fusion;

// Leaf nodes: per-row and per-col vectors loaded via TMA
//   First template arg is smem pipeline stage (must be 0, pipelining not supported)
using Leaf_sx       = Sm90ColBroadcast<0, TileShape_MNK, ElementOutput, ElementCompute, Stride<_1, _0, int64_t>>;
using Leaf_sw       = Sm90RowBroadcast<0, TileShape_MNK, ElementOutput, ElementCompute, Stride<_0, _1, int64_t>>;
using Leaf_neg_zero = Sm90ColBroadcast<0, TileShape_MNK, ElementOutput, ElementCompute, Stride<_1, _0, int64_t>>;
using Leaf_colsum   = Sm90RowBroadcast<0, TileShape_MNK, ElementCompute, ElementCompute, Stride<_0, _1, int64_t>>;

// bias_term = neg_zero_x[m] * colsum_w[n]
using BiasCompute = Sm90EVT<
    Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
    Leaf_neg_zero, Leaf_colsum
>;

// corrected_acc = acc + bias_term
using CorrectedAcc = Sm90EVT<
    Sm90Compute<cutlass::plus, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
    Sm90AccFetch, BiasCompute
>;

// scale_product = s_x[m] * s_w[n]
using ScaleProduct = Sm90EVT<
    Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
    Leaf_sx, Leaf_sw
>;

// Final: D = scale_product * corrected_acc
using DequantEVT = Sm90EVT<
    Sm90Compute<cutlass::multiplies, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
    ScaleProduct, CorrectedAcc
>;

// =============================================================================
// GEMM Type Definitions
// =============================================================================

// Helper: build epilogue with EVT
using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    int32_t, ElementCompute,
    ElementOutput, cutlass::layout::RowMajor, AlignmentOutput,
    ElementOutput, cutlass::layout::RowMajor, AlignmentOutput,
    cutlass::epilogue::TmaWarpSpecialized,
    DequantEVT
>::CollectiveOp;

// Shared memory carveout for epilogue
constexpr int EpilogueSmemBytes = static_cast<int>(sizeof(typename EpilogueOp::SharedStorage));

// --- UINT8 × INT8 → dequant → FP16 (high portion) ---
using GemmU8S8_Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    uint8_t, cutlass::layout::RowMajor, 16,
    int8_t, cutlass::layout::ColumnMajor, 16,
    int32_t,
    TileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<EpilogueSmemBytes>,
    cutlass::gemm::KernelTmaWarpSpecialized
>::CollectiveOp;

using GemmU8S8_Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, GemmU8S8_Mainloop, EpilogueOp>;
using GemmU8S8 = cutlass::gemm::device::GemmUniversalAdapter<GemmU8S8_Kernel>;

// --- INT8 × INT8 → dequant → FP16 (main portion) ---
using GemmS8S8_Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    int8_t, cutlass::layout::RowMajor, 16,
    int8_t, cutlass::layout::ColumnMajor, 16,
    int32_t,
    TileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<EpilogueSmemBytes>,
    cutlass::gemm::KernelTmaWarpSpecialized
>::CollectiveOp;

using GemmS8S8_Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, GemmS8S8_Mainloop, EpilogueOp>;
using GemmS8S8 = cutlass::gemm::device::GemmUniversalAdapter<GemmS8S8_Kernel>;

// =============================================================================
// Launch helper
// =============================================================================

template <typename Gemm>
cutlass::Status run_gemm_dequant(
    int M, int N, int K,
    void const* ptr_A, void const* ptr_B, void* ptr_D,
    void const* ptr_s_x,         // (M,) fp16
    void const* ptr_s_w,         // (N,) fp16
    void const* ptr_neg_zero_x,  // (M,) fp16
    void const* ptr_colsum_w,    // (N,) float32
    cudaStream_t stream)
{
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    // Build EVT fusion arguments following the tree structure:
    // DequantEVT = EVT<Compute<mul>, ScaleProduct, CorrectedAcc>
    //   ScaleProduct = EVT<Compute<mul>, Leaf_sx, Leaf_sw>
    //   CorrectedAcc = EVT<Compute<plus>, AccFetch, BiasCompute>
    //     BiasCompute = EVT<Compute<mul>, Leaf_neg_zero, Leaf_colsum>
    using FusionArgs = typename Gemm::GemmKernel::CollectiveEpilogue::FusionCallbacks::Arguments;

    FusionArgs fusion_args{
        {},  // DequantEVT: Compute<mul> args (empty)
        // ScaleProduct args:
        {
            {},  // Compute<mul> args (empty)
            // Leaf_sx (ColBroadcast): {ptr, null_default, stride}
            {reinterpret_cast<ElementOutput const*>(ptr_s_x), ElementOutput(0), Stride<_1, _0, int64_t>{_1{}, _0{}, 0}},
            // Leaf_sw (RowBroadcast): {ptr, null_default, stride}
            {reinterpret_cast<ElementOutput const*>(ptr_s_w), ElementOutput(0), Stride<_0, _1, int64_t>{_0{}, _1{}, 0}}
        },
        // CorrectedAcc args:
        {
            {},  // Compute<plus> args (empty)
            {},  // AccFetch args (empty)
            // BiasCompute args:
            {
                {},  // Compute<mul> args (empty)
                // Leaf_neg_zero (ColBroadcast): {ptr, null_default, stride}
                {reinterpret_cast<ElementOutput const*>(ptr_neg_zero_x), ElementOutput(0), Stride<_1, _0, int64_t>{_1{}, _0{}, 0}},
                // Leaf_colsum (RowBroadcast): {ptr, null_default, stride}
                {reinterpret_cast<ElementCompute const*>(ptr_colsum_w), ElementCompute(0), Stride<_0, _1, int64_t>{_0{}, _1{}, 0}}
            }
        }
    };

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        // Mainloop args
        {
            reinterpret_cast<typename Gemm::GemmKernel::CollectiveMainloop::ElementA const*>(ptr_A),
            stride_A,
            reinterpret_cast<typename Gemm::GemmKernel::CollectiveMainloop::ElementB const*>(ptr_B),
            stride_B
        },
        // Epilogue args
        {
            fusion_args,
            nullptr,  // C ptr (unused)
            stride_D,
            reinterpret_cast<ElementOutput*>(ptr_D),
            stride_D
        }
    };

    Gemm gemm_op;
    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return status;

    size_t workspace_size = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                   torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    status = gemm_op.initialize(args, workspace.data_ptr(), stream);
    if (status != cutlass::Status::kSuccess) return status;

    return gemm_op.run(stream);
}

// =============================================================================
// PyTorch-facing functions
// =============================================================================

/// UINT8×INT8 GEMM + fused dequant → FP16 (high portion)
torch::Tensor gemm_u8s8_dequant(
    torch::Tensor A,           // (M, K) uint8
    torch::Tensor B,           // (N, K) int8
    torch::Tensor s_x,         // (M,) fp16 per-token scale
    torch::Tensor s_w,         // (N,) fp16 per-channel scale
    torch::Tensor neg_zero_x,  // (M,) fp16 negated per-token zero point
    torch::Tensor colsum_w)    // (N,) float32 precomputed weight column sum
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kUInt8, "A must be uint8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "K dimensions must match");

    A = A.contiguous(); B = B.contiguous();
    s_x = s_x.contiguous(); s_w = s_w.contiguous();
    neg_zero_x = neg_zero_x.contiguous(); colsum_w = colsum_w.contiguous();

    int M = A.size(0), K = A.size(1), N = B.size(0);
    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto status = run_gemm_dequant<GemmU8S8>(
        M, N, K, A.data_ptr(), B.data_ptr(), output.data_ptr(),
        s_x.data_ptr(), s_w.data_ptr(), neg_zero_x.data_ptr(), colsum_w.data_ptr(),
        stream);

    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS U8S8 dequant GEMM failed: ", cutlass::cutlassGetStatusString(status));
    return output;
}

/// INT8×INT8 GEMM + fused dequant → FP16 (main portion, int4 expanded to int8)
torch::Tensor gemm_s8s8_dequant(
    torch::Tensor A,           // (M, K) int8 (shifted activation)
    torch::Tensor B,           // (N, K) int8 weight
    torch::Tensor s_x,         // (M,) fp16 per-token scale
    torch::Tensor s_w,         // (N,) fp16 per-channel scale
    torch::Tensor neg_zero_x,  // (M,) fp16 negated per-token zero (adjusted for shift)
    torch::Tensor colsum_w)    // (N,) float32 precomputed weight column sum
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "K dimensions must match");

    A = A.contiguous(); B = B.contiguous();
    s_x = s_x.contiguous(); s_w = s_w.contiguous();
    neg_zero_x = neg_zero_x.contiguous(); colsum_w = colsum_w.contiguous();

    int M = A.size(0), K = A.size(1), N = B.size(0);
    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto status = run_gemm_dequant<GemmS8S8>(
        M, N, K, A.data_ptr(), B.data_ptr(), output.data_ptr(),
        s_x.data_ptr(), s_w.data_ptr(), neg_zero_x.data_ptr(), colsum_w.data_ptr(),
        stream);

    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS S8S8 dequant GEMM failed: ", cutlass::cutlassGetStatusString(status));
    return output;
}

#else
// Fallback for non-SM90 builds
torch::Tensor gemm_u8s8_dequant(torch::Tensor A, torch::Tensor B,
    torch::Tensor s_x, torch::Tensor s_w, torch::Tensor neg_zero_x, torch::Tensor colsum_w) {
    TORCH_CHECK(false, "SM90 not supported"); return {};
}
torch::Tensor gemm_s8s8_dequant(torch::Tensor A, torch::Tensor B,
    torch::Tensor s_x, torch::Tensor s_w, torch::Tensor neg_zero_x, torch::Tensor colsum_w) {
    TORCH_CHECK(false, "SM90 not supported"); return {};
}
#endif

// =============================================================================
// Python binding
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_u8s8_dequant", &gemm_u8s8_dequant,
          "UINT8×INT8 GEMM + fused dequant → FP16 (high portion, single kernel via EVT)",
          py::arg("A"), py::arg("B"), py::arg("s_x"), py::arg("s_w"),
          py::arg("neg_zero_x"), py::arg("colsum_w"));
    m.def("gemm_s8s8_dequant", &gemm_s8s8_dequant,
          "INT8×INT8 GEMM + fused dequant → FP16 (main portion, single kernel via EVT)",
          py::arg("A"), py::arg("B"), py::arg("s_x"), py::arg("s_w"),
          py::arg("neg_zero_x"), py::arg("colsum_w"));
}
