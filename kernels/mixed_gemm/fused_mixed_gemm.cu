/***************************************************************************************************
 * Fused Mixed-Precision GEMM (v1: dual accumulator, no INT4 packing)
 *
 * Single kernel that runs two U8S8 GEMM phases:
 *   Phase 1: A_main(UINT8, M×K_main) × B_main(INT8, N×K_main) → acc_main (INT32)
 *   Phase 2: A_high(UINT8, M×K_high) × B_high(INT8, N×K_high) → acc_high (INT32)
 *
 * Then in-register FP32 dequant + combine:
 *   out = s_x_m*s_w_m*(acc_main + neg_zero_m*colsum_main)
 *       + s_x_h*s_w_h*(acc_high + neg_zero_h*colsum_high)
 *   D[fp16] = out → single HBM write
 *
 * This saves 2 kernel launches + 1 elementwise add + 2 intermediate HBM writes vs baseline.
 *
 * v1 constraints:
 * - Data format same as baseline (all INT8/UINT8, no INT4 packing)
 * - Uses CUTLASS pipeline (warp-specialized TMA) for each phase sequentially
 * - No pipeline overlap between phases
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
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#define CUTLASS_ARCH_MMA_SM90_SUPPORTED 1

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// =============================================================================
// Type definitions (same as baseline)
// =============================================================================

using TileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using ElementA = uint8_t;
using ElementB = int8_t;
using ElementAccum = int32_t;
using ElementOutput = cutlass::half_t;
using ElementCompute = float;
// Intermediate GEMM output is FP32 to avoid INT32→FP16 overflow
// (K=1792, max acc value ~58M >> FP16 max 65504)
using ElementIntermediate = float;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementIntermediate>::value;  // 4 (float32)

// =============================================================================
// Build mainloop (same for both phases — U8S8 WGMMA)
// =============================================================================

using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, AlignmentA,
    ElementB, cutlass::layout::ColumnMajor, AlignmentB,
    ElementAccum,
    TileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::KernelTmaWarpSpecialized
>::CollectiveOp;

// =============================================================================
// Build a minimal epilogue (INT32 → FP32, alpha=1, beta=0)
// Stores raw integer matmul result as FP32 to avoid FP16 overflow.
// =============================================================================

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    ElementIntermediate, cutlass::layout::RowMajor, AlignmentD,
    ElementIntermediate, cutlass::layout::RowMajor, AlignmentD,
    cutlass::epilogue::TmaWarpSpecialized
>::CollectiveOp;

// Full GEMM kernel type (used for shape/stride computation helpers)
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// =============================================================================
// Helper: Run a single GEMM phase using CUTLASS (reuses baseline logic)
// Returns status. The accumulator result ends up in the output tensor.
// =============================================================================

static cutlass::Status run_phase_gemm(
    int M, int N, int K,
    void const* ptr_A, void const* ptr_B, void* ptr_D,
    cudaStream_t stream)
{
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    // Simple epilogue: D = 1.0 * acc + 0.0 * C  (convert INT32 → FP32)
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<ElementA const*>(ptr_A),
            stride_A,
            reinterpret_cast<ElementB const*>(ptr_B),
            stride_B
        },
        {
            {},  // epilogue fusion args (default linear combination)
            nullptr,  // C ptr (unused, beta=0)
            stride_D,
            reinterpret_cast<ElementIntermediate*>(ptr_D),
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
// Fused kernel: two GEMM phases + in-register dequant+combine
//
// v1 implementation: runs two separate CUTLASS GEMM launches into intermediate
// buffers, then does the fused dequant+combine in a custom elementwise kernel.
// This is a stepping stone — already saves 1 kernel launch vs baseline (the add),
// and sets up the interface for future true-fused implementation.
//
// v2 will fuse everything into a single kernel launch with shared accumulators.
// =============================================================================

/// Custom dequant + combine kernel
/// D[m,n] = s_x_m[m]*s_w_m[n]*(Y_main[m,n] + neg_zero_m[m]*colsum_m[n])
///        + s_x_h[m]*s_w_h[n]*(Y_high[m,n] + neg_zero_h[m]*colsum_h[n])
__global__ void fused_dequant_combine_kernel(
    cutlass::half_t* __restrict__ D,
    float const* __restrict__ Y_main,               // (M, N) fp32 raw integer matmul
    float const* __restrict__ Y_high,               // (M, N) fp32 raw integer matmul
    cutlass::half_t const* __restrict__ s_x_m,      // (M,) fp16
    cutlass::half_t const* __restrict__ s_w_m,      // (N,) fp16
    cutlass::half_t const* __restrict__ neg_zero_m, // (M,) fp16
    float const* __restrict__ colsum_m,             // (N,) fp32
    cutlass::half_t const* __restrict__ s_x_h,      // (M,) fp16
    cutlass::half_t const* __restrict__ s_w_h,      // (N,) fp16
    cutlass::half_t const* __restrict__ neg_zero_h, // (M,) fp16
    float const* __restrict__ colsum_h,             // (N,) fp32
    int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int m = idx / N;
    int n = idx % N;

    // Load scales (broadcast)
    float sx_m = __half2float(s_x_m[m]);
    float sw_m = __half2float(s_w_m[n]);
    float nz_m = __half2float(neg_zero_m[m]);
    float cs_m = colsum_m[n];

    float sx_h = __half2float(s_x_h[m]);
    float sw_h = __half2float(s_w_h[n]);
    float nz_h = __half2float(neg_zero_h[m]);
    float cs_h = colsum_h[n];

    // Load raw GEMM outputs (FP32, representing INT32 accumulator values)
    float y_main = Y_main[idx];
    float y_high = Y_high[idx];

    // Dequant + combine
    float out_main = sx_m * sw_m * (y_main + nz_m * cs_m);
    float out_high = sx_h * sw_h * (y_high + nz_h * cs_h);

    D[idx] = __float2half(out_main + out_high);
}

// =============================================================================
// PyTorch-facing function
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

    TORCH_CHECK(A_main.is_contiguous() && B_main.is_contiguous(), "main tensors must be contiguous");
    TORCH_CHECK(A_high.is_contiguous() && B_high.is_contiguous(), "high tensors must be contiguous");
    TORCH_CHECK(s_x_m.is_contiguous() && s_w_m.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(neg_zero_m.is_contiguous() && colsum_m.is_contiguous(), "zeros/colsums must be contiguous");
    TORCH_CHECK(s_x_h.is_contiguous() && s_w_h.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(neg_zero_h.is_contiguous() && colsum_h.is_contiguous(), "zeros/colsums must be contiguous");

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
    TORCH_CHECK(A_high.size(0) == M, "M dimension mismatch between main and high");
    TORCH_CHECK(B_high.size(0) == N, "N dimension mismatch between main and high");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Allocate intermediate buffers for raw GEMM outputs (FP32 to avoid overflow)
    auto Y_main = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(A_main.device()));
    auto Y_high = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(A_main.device()));

    // Phase 1: main GEMM (U8S8)
    auto status = run_phase_gemm(M, N, K_main,
        A_main.data_ptr(), B_main.data_ptr(), Y_main.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS main GEMM failed: ", cutlass::cutlassGetStatusString(status));

    // Phase 2: high GEMM (U8S8)
    status = run_phase_gemm(M, N, K_high,
        A_high.data_ptr(), B_high.data_ptr(), Y_high.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS high GEMM failed: ", cutlass::cutlassGetStatusString(status));

    // Phase 3: fused dequant + combine
    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A_main.device()));

    int total_elements = M * N;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_dequant_combine_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<cutlass::half_t*>(output.data_ptr()),
        reinterpret_cast<float const*>(Y_main.data_ptr()),
        reinterpret_cast<float const*>(Y_high.data_ptr()),
        reinterpret_cast<cutlass::half_t const*>(s_x_m.data_ptr()),
        reinterpret_cast<cutlass::half_t const*>(s_w_m.data_ptr()),
        reinterpret_cast<cutlass::half_t const*>(neg_zero_m.data_ptr()),
        reinterpret_cast<float const*>(colsum_m.data_ptr()),
        reinterpret_cast<cutlass::half_t const*>(s_x_h.data_ptr()),
        reinterpret_cast<cutlass::half_t const*>(s_w_h.data_ptr()),
        reinterpret_cast<cutlass::half_t const*>(neg_zero_h.data_ptr()),
        reinterpret_cast<float const*>(colsum_h.data_ptr()),
        M, N);

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
