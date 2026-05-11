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

// Leaf nodes: per-row and per-col vectors
//   Sm90ColBroadcast: broadcasts a column vector (M,) across N dimension → per-row
//   Sm90RowBroadcast: broadcasts a row vector (N,) across M dimension → per-col
//   Stage=0 (no smem pipelining), no alignment specified
using Leaf_sx       = Sm90ColBroadcast<0, TileShape_MNK, ElementOutput, ElementCompute>;
using Leaf_sw       = Sm90RowBroadcast<0, TileShape_MNK, ElementOutput, ElementCompute>;
using Leaf_neg_zero = Sm90ColBroadcast<0, TileShape_MNK, ElementOutput, ElementCompute>;
using Leaf_colsum   = Sm90RowBroadcast<0, TileShape_MNK, ElementCompute, ElementCompute>;

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
    // IMPORTANT: EVT Arguments order is {child0_args, child1_args, ..., node_op_args}
    //   (children first, op last — opposite of template params!)
    //
    // DequantEVT = EVT<Compute<mul>, ScaleProduct, CorrectedAcc>
    //   Args: {ScaleProduct_args, CorrectedAcc_args, mul_op_args}
    //
    //   ScaleProduct = EVT<Compute<mul>, Leaf_sx, Leaf_sw>
    //     Args: {Leaf_sx_args, Leaf_sw_args, mul_op_args}
    //
    //   CorrectedAcc = EVT<Compute<plus>, AccFetch, BiasCompute>
    //     Args: {AccFetch_args, BiasCompute_args, plus_op_args}
    //
    //   BiasCompute = EVT<Compute<mul>, Leaf_neg_zero, Leaf_colsum>
    //     Args: {Leaf_neg_zero_args, Leaf_colsum_args, mul_op_args}

    using FusionArgs = typename Gemm::GemmKernel::CollectiveEpilogue::FusionCallbacks::Arguments;

    FusionArgs fusion_args{
        // ScaleProduct args = {Leaf_sx_args, Leaf_sw_args, mul_op_args}
        {
            {reinterpret_cast<ElementOutput const*>(ptr_s_x)},    // Leaf_sx (ColBroadcast): just ptr
            {reinterpret_cast<ElementOutput const*>(ptr_s_w)},    // Leaf_sw (RowBroadcast): just ptr
            {}                                                      // Compute<mul> op args (empty)
        },
        // CorrectedAcc args = {AccFetch_args, BiasCompute_args, plus_op_args}
        {
            {},  // AccFetch args (empty)
            // BiasCompute args = {Leaf_neg_zero_args, Leaf_colsum_args, mul_op_args}
            {
                {reinterpret_cast<ElementOutput const*>(ptr_neg_zero_x)},   // Leaf_neg_zero (ColBroadcast)
                {reinterpret_cast<ElementCompute const*>(ptr_colsum_w)},    // Leaf_colsum (RowBroadcast)
                {}                                                           // Compute<mul> op args (empty)
            },
            {}   // Compute<plus> op args (empty)
        },
        {}  // DequantEVT top-level Compute<mul> op args (empty)
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
    torch::Tensor A,           // (M, K) or (batch, seq, K) uint8
    torch::Tensor B,           // (N, K) int8
    torch::Tensor s_x,         // (M,) fp16 per-token scale
    torch::Tensor s_w,         // (N,) fp16 per-channel scale
    torch::Tensor neg_zero_x,  // (M,) fp16 negated per-token zero point
    torch::Tensor colsum_w)    // (N,) float32 precomputed weight column sum
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kUInt8, "A must be uint8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be 2D or 3D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");

    auto A_shape = A.sizes().vec();
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(s_x.is_contiguous(), "s_x must be contiguous");
    TORCH_CHECK(s_w.is_contiguous(), "s_w must be contiguous");
    TORCH_CHECK(neg_zero_x.is_contiguous(), "neg_zero_x must be contiguous");
    TORCH_CHECK(colsum_w.is_contiguous(), "colsum_w must be contiguous");

    A = A.reshape({-1, A.size(-1)});  // flatten to (M, K), no copy since already contiguous

    int M = A.size(0), K = A.size(1), N = B.size(0);
    TORCH_CHECK(K == B.size(1), "K dimensions must match");

    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto status = run_gemm_dequant<GemmU8S8>(
        M, N, K, A.data_ptr(), B.data_ptr(), output.data_ptr(),
        s_x.data_ptr(), s_w.data_ptr(), neg_zero_x.data_ptr(), colsum_w.data_ptr(),
        stream);

    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS U8S8 dequant GEMM failed: ", cutlass::cutlassGetStatusString(status));

    // Reshape output to match input batch dims: (batch, seq, N)
    if (A_shape.size() == 3) {
        output = output.reshape({A_shape[0], A_shape[1], N});
    }
    return output;
}

/// INT8×INT8 GEMM + fused dequant → FP16 (main portion, int4 expanded to int8)
torch::Tensor gemm_s8s8_dequant(
    torch::Tensor A,           // (M, K) or (batch, seq, K) int8
    torch::Tensor B,           // (N, K) int8 weight
    torch::Tensor s_x,         // (M,) fp16 per-token scale
    torch::Tensor s_w,         // (N,) fp16 per-channel scale
    torch::Tensor neg_zero_x,  // (M,) fp16 negated per-token zero (adjusted for shift)
    torch::Tensor colsum_w)    // (N,) float32 precomputed weight column sum
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be 2D or 3D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");

    auto A_shape = A.sizes().vec();
    A = A.contiguous().reshape({-1, A.size(-1)});
    B = B.contiguous();
    s_x = s_x.contiguous(); s_w = s_w.contiguous();
    neg_zero_x = neg_zero_x.contiguous(); colsum_w = colsum_w.contiguous();

    int M = A.size(0), K = A.size(1), N = B.size(0);
    TORCH_CHECK(K == B.size(1), "K dimensions must match");

    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto status = run_gemm_dequant<GemmS8S8>(
        M, N, K, A.data_ptr(), B.data_ptr(), output.data_ptr(),
        s_x.data_ptr(), s_w.data_ptr(), neg_zero_x.data_ptr(), colsum_w.data_ptr(),
        stream);

    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS S8S8 dequant GEMM failed: ", cutlass::cutlassGetStatusString(status));

    if (A_shape.size() == 3) {
        output = output.reshape({A_shape[0], A_shape[1], N});
    }
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

// Forward declaration of fused kernels (defined in fused_mixed_gemm.cu / fused_mixed_gemm_v2.cu)
torch::Tensor fused_mixed_gemm(
    torch::Tensor A_main, torch::Tensor B_main,
    torch::Tensor A_high, torch::Tensor B_high,
    torch::Tensor s_x_m, torch::Tensor s_w_m, torch::Tensor neg_zero_m, torch::Tensor colsum_m,
    torch::Tensor s_x_h, torch::Tensor s_w_h, torch::Tensor neg_zero_h, torch::Tensor colsum_h);

torch::Tensor fused_mixed_gemm_v2(
    torch::Tensor A_main, torch::Tensor B_main,
    torch::Tensor A_high, torch::Tensor B_high,
    torch::Tensor s_x_m, torch::Tensor s_w_m, torch::Tensor neg_zero_m, torch::Tensor colsum_m,
    torch::Tensor s_x_h, torch::Tensor s_w_h, torch::Tensor neg_zero_h, torch::Tensor colsum_h);

torch::Tensor fused_mixed_gemm_v3(
    torch::Tensor A_main, torch::Tensor B_main,
    torch::Tensor A_high, torch::Tensor B_high,
    torch::Tensor s_x_m, torch::Tensor s_w_m, torch::Tensor neg_zero_m, torch::Tensor colsum_m,
    torch::Tensor s_x_h, torch::Tensor s_w_h, torch::Tensor neg_zero_h, torch::Tensor colsum_h);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_u8s8_dequant", &gemm_u8s8_dequant,
          "UINT8×INT8 GEMM + fused dequant → FP16 (high portion, single kernel via EVT)",
          py::arg("A"), py::arg("B"), py::arg("s_x"), py::arg("s_w"),
          py::arg("neg_zero_x"), py::arg("colsum_w"));
    m.def("gemm_s8s8_dequant", &gemm_s8s8_dequant,
          "INT8×INT8 GEMM + fused dequant → FP16 (main portion, single kernel via EVT)",
          py::arg("A"), py::arg("B"), py::arg("s_x"), py::arg("s_w"),
          py::arg("neg_zero_x"), py::arg("colsum_w"));
    m.def("fused_mixed_gemm", &fused_mixed_gemm,
          "Fused dual-phase GEMM v1 (uses CollectiveMainloop calls)",
          py::arg("A_main"), py::arg("B_main"), py::arg("A_high"), py::arg("B_high"),
          py::arg("s_x_m"), py::arg("s_w_m"), py::arg("neg_zero_m"), py::arg("colsum_m"),
          py::arg("s_x_h"), py::arg("s_w_h"), py::arg("neg_zero_h"), py::arg("colsum_h"));
    m.def("fused_mixed_gemm_v2", &fused_mixed_gemm_v2,
          "Fused dual-phase GEMM v2 (hand-written inline K-loop, no serialization)",
          py::arg("A_main"), py::arg("B_main"), py::arg("A_high"), py::arg("B_high"),
          py::arg("s_x_m"), py::arg("s_w_m"), py::arg("neg_zero_m"), py::arg("colsum_m"),
          py::arg("s_x_h"), py::arg("s_w_h"), py::arg("neg_zero_h"), py::arg("colsum_h"));
    m.def("fused_mixed_gemm_v3", &fused_mixed_gemm_v3,
          "Fused dual-phase GEMM v3 (single accumulator + workspace, lowest register pressure)",
          py::arg("A_main"), py::arg("B_main"), py::arg("A_high"), py::arg("B_high"),
          py::arg("s_x_m"), py::arg("s_w_m"), py::arg("neg_zero_m"), py::arg("colsum_m"),
          py::arg("s_x_h"), py::arg("s_w_h"), py::arg("neg_zero_h"), py::arg("colsum_h"));
}
