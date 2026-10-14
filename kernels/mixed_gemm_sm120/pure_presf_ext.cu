// T0-presf bounds: pure-format full-K GEMMs with PRE-INTERLEAVED weight SF.
//
// The round-3 T0 bounds (nvfp4_gemm/mxfp8_gemm) scatter BOTH the activation and
// the weight SF inside every timed call; the weight scatter is model-load-time
// work in a real pipeline, so those bounds are pessimistic at small M (band
// artifacts of -130%..-290%). These entries accept the weight SF already
// interleaved (prepack_sf_*_b, run once) and fill only the activation SF per
// call — the same methodology as rungB_presf and b1_fused. GEMM configs are
// copied VERBATIM from nvfp4_gemm_ext.cu (ex79a) / mxfp8_gemm_ext.cu (ex79c);
// those files stay untouched.
//
// Entries:
//   prepack_sf_fp4_b(SFB u8[N,K/16], N, K)  -> phys u8 buffer
//   nvfp4_gemm_presf(A u8[M,K/2], SFA u8[M,K/16], B u8[N,K/2], physSFB, M,N,K) -> bf16
//   prepack_sf_fp8_b(SFB u8[N,K/32], N, K)  -> phys u8 buffer
//   mxfp8_gemm_presf(A u8[M,K],  SFA u8[M,K/32], B u8[N,K],  physSFB, M,N,K) -> bf16

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

static inline void check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// Scatter logical [rows, K/VEC] SF bytes into the CUTLASS interleaved buffer.
template <class LayoutSF>
__global__ void presf_fill_sf_kernel(uint8_t* __restrict__ phys,
                                     const uint8_t* __restrict__ logical,
                                     LayoutSF layout, int rows, int K, int VEC) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * K;
  if (idx >= total) return;
  int r = idx / K;
  int k = idx % K;
  int nblk = K / VEC;
  int off = layout(make_coord(r, k, 0));
  phys[off] = logical[r * nblk + (k / VEC)];
}

// Shared epilogue/GEMM scaffolding per format (configs verbatim from the
// round-1 extensions).
template <class ElementAB, int Alignment, int VEC>
struct PresfGemm {
  using ElementA   = ElementAB;
  using LayoutATag = cutlass::layout::RowMajor;
  using ElementB   = ElementAB;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using ElementD   = cutlass::bfloat16_t;
  using ElementC   = cutlass::bfloat16_t;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  using ElementAccumulator = float;
  using ArchTag       = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using ThreadBlockShape = Shape<_128,_128,_128>;
  using ClusterShape     = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ThreadBlockShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, Alignment,
      ElementB, LayoutBTag, Alignment,
      ElementAccumulator,
      ThreadBlockShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using BSConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  // Prepack the (static) weight-side SF. tile_atom_to_shape_SFB depends only on
  // (N,K,L), not M (verified round-2), so M=1 here is valid for every M later.
  static torch::Tensor prepack_b(torch::Tensor SFB, int64_t N, int64_t K) {
    TORCH_CHECK(SFB.is_cuda() && SFB.scalar_type() == torch::kUInt8, "SFB must be CUDA uint8");
    SFB = SFB.contiguous();
    auto dev = SFB.device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev.index());
    int Ni = (int)N, Ki = (int)K;
    auto layout_SFB = BSConfig::tile_atom_to_shape_SFB(make_shape(1, Ni, Ki, 1));
    int64_t sz = (int64_t)size(filter_zeros(layout_SFB));
    auto phys = torch::empty({sz}, torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    int total = Ni * Ki, tpb = 256;
    presf_fill_sf_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(), layout_SFB, Ni, Ki, VEC);
    check_cuda(cudaGetLastError(), "presf prepack_b launch");
    check_cuda(cudaStreamSynchronize(stream), "presf prepack_b sync");
    return phys;
  }

  // Per-call: fill ONLY the activation SF, then GEMM.
  static torch::Tensor run(torch::Tensor A, torch::Tensor SFA,
                           torch::Tensor B, torch::Tensor physSFB,
                           int64_t M, int64_t N, int64_t K) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && SFA.is_cuda() && physSFB.is_cuda(),
                "all inputs must be CUDA tensors");
    A = A.contiguous(); B = B.contiguous();
    SFA = SFA.contiguous(); physSFB = physSFB.contiguous();
    auto dev = A.device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev.index());
    int Mi = (int)M, Ni = (int)N, Ki = (int)K;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {Mi, Ki, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {Ni, Ki, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {Mi, Ni, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {Mi, Ni, 1});
    auto layout_SFA = BSConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Ki, 1));
    auto layout_SFB = BSConfig::tile_atom_to_shape_SFB(make_shape(Mi, Ni, Ki, 1));
    TORCH_CHECK((int64_t)size(filter_zeros(layout_SFB)) == physSFB.numel(),
                "physSFB size mismatch — prepacked with different (N,K)?");

    auto u8opts = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    int64_t sfa_size = (int64_t)size(filter_zeros(layout_SFA));
    auto phys_SFA = torch::empty({sfa_size}, u8opts);
    {
      int total = Mi * Ki, tpb = 256;
      presf_fill_sf_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
          phys_SFA.data_ptr<uint8_t>(), SFA.data_ptr<uint8_t>(), layout_SFA, Mi, Ki, VEC);
      check_cuda(cudaGetLastError(), "presf fill SFA launch");
    }

    auto D = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {Mi, Ni, Ki, 1},
      {
        reinterpret_cast<typename ElementAB::DataType const*>(A.data_ptr<uint8_t>()), stride_A,
        reinterpret_cast<typename ElementAB::DataType const*>(B.data_ptr<uint8_t>()), stride_B,
        reinterpret_cast<typename ElementAB::ScaleFactorType const*>(phys_SFA.data_ptr<uint8_t>()), layout_SFA,
        reinterpret_cast<typename ElementAB::ScaleFactorType const*>(physSFB.data_ptr<uint8_t>()), layout_SFB
      },
      {
        {1.0f, 0.0f},
        reinterpret_cast<ElementC const*>(D.data_ptr()), stride_C,
        reinterpret_cast<ElementD*>(D.data_ptr()), stride_D
      }
    };
    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(arguments);
    auto workspace = torch::empty({(int64_t)ws}, u8opts);
    cutlass::Status st = gemm.can_implement(arguments);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "can_implement failed: ", cutlassGetStatusString(st));
    st = gemm.initialize(arguments, workspace.data_ptr(), stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "initialize failed: ", cutlassGetStatusString(st));
    st = gemm.run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "run failed: ", cutlassGetStatusString(st));
    check_cuda(cudaStreamSynchronize(stream), "presf gemm sync");
    return D;
  }
};

using PresfNV = PresfGemm<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 32, 16>;
using PresfMX = PresfGemm<cutlass::mx_float8_t<cutlass::float_e4m3_t>, 16, 32>;

torch::Tensor prepack_sf_fp4_b(torch::Tensor SFB, int64_t N, int64_t K) {
  TORCH_CHECK(K % 16 == 0, "K must be divisible by 16");
  return PresfNV::prepack_b(SFB, N, K);
}
torch::Tensor nvfp4_gemm_presf(torch::Tensor A, torch::Tensor SFA,
                               torch::Tensor B, torch::Tensor physSFB,
                               int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(K % 16 == 0, "K must be divisible by 16");
  return PresfNV::run(A, SFA, B, physSFB, M, N, K);
}
torch::Tensor prepack_sf_fp8_b(torch::Tensor SFB, int64_t N, int64_t K) {
  TORCH_CHECK(K % 32 == 0, "K must be divisible by 32");
  return PresfMX::prepack_b(SFB, N, K);
}
torch::Tensor mxfp8_gemm_presf(torch::Tensor A, torch::Tensor SFA,
                               torch::Tensor B, torch::Tensor physSFB,
                               int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(K % 32 == 0, "K must be divisible by 32");
  return PresfMX::run(A, SFA, B, physSFB, M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepack_sf_fp4_b", &prepack_sf_fp4_b, "prepack NVFP4 weight SF (once)");
  m.def("nvfp4_gemm_presf", &nvfp4_gemm_presf, "pure NVFP4 full-K bound, presf weight SF");
  m.def("prepack_sf_fp8_b", &prepack_sf_fp8_b, "prepack MXFP8 weight SF (once)");
  m.def("mxfp8_gemm_presf", &mxfp8_gemm_presf, "pure MXFP8 full-K bound, presf weight SF");
}
