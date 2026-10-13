// MXFP8 x MXFP8 block-scaled GEMM -> bf16, PyTorch extension for SM120 (Blackwell).
// Config taken from CUTLASS example 79c (MXFP8 mainloop), with B switched from mxfp6
// to mxfp8 (symmetric MXFP8xMXFP8). Accepts external device pointers for A/B (FP8
// E4M3, 1 byte/element) and logical block scale factors (ue8m0 exponent bytes),
// builds the interleaved SF buffers via CUTLASS's own tile_atom_to_shape_SFA/SFB.
//
// Signature (see mxfp8_gemm.py):
//   mxfp8_gemm(A uint8[M,K], SFA uint8[M,K/32],
//              B uint8[N,K], SFB uint8[N,K/32], M, N, K) -> bf16[M,N]

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

// ----------------------------- GEMM configuration (ex79c, MXFP8xMXFP8) -----------------------------
using ElementA   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;   // MXFP8 (E4M3)
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;

using ElementB   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;   // MXFP8 (E4M3)
using LayoutBTag = cutlass::layout::ColumnMajor;                  // weights stored (N,K)
constexpr int AlignmentB = 16;

using ElementD   = cutlass::bfloat16_t;
using ElementC   = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag       = cutlass::arch::Sm120;
using OperatorClass  = cutlass::arch::OpClassBlockScaledTensorOp;

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
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// ----------------------------- SF fill kernel -----------------------------
// Scatter logical [rows, K/VEC] ue8m0 bytes into the CUTLASS interleaved SF buffer.
// VEC = 32 for MXFP8. Writing every k (redundant within a block) is layout-agnostic.
template <class LayoutSF>
__global__ void fill_sf_kernel(uint8_t* __restrict__ phys,
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

static inline void check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// ----------------------------- entry point -----------------------------
torch::Tensor mxfp8_gemm(torch::Tensor A, torch::Tensor SFA,
                         torch::Tensor B, torch::Tensor SFB,
                         int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && SFA.is_cuda() && SFB.is_cuda(),
              "all inputs must be CUDA tensors");
  TORCH_CHECK(A.scalar_type() == torch::kUInt8 && B.scalar_type() == torch::kUInt8,
              "A/B must be uint8 (e4m3 bytes)");
  TORCH_CHECK(SFA.scalar_type() == torch::kUInt8 && SFB.scalar_type() == torch::kUInt8,
              "SFA/SFB must be uint8 (ue8m0 exponent bytes)");
  TORCH_CHECK(K % 32 == 0, "K must be divisible by 32 (MXFP8 block size)");
  A = A.contiguous();
  B = B.contiguous();
  SFA = SFA.contiguous();
  SFB = SFB.contiguous();

  auto dev = A.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev.index());

  int Mi = (int)M, Ni = (int)N, Ki = (int)K;
  const int VEC = 32;
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {Mi, Ki, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {Ni, Ki, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {Mi, Ni, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {Mi, Ni, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Ki, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(Mi, Ni, Ki, 1));

  int64_t sfa_size = (int64_t)size(filter_zeros(layout_SFA));
  int64_t sfb_size = (int64_t)size(filter_zeros(layout_SFB));

  auto u8opts = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto phys_SFA = torch::empty({sfa_size}, u8opts);
  auto phys_SFB = torch::empty({sfb_size}, u8opts);

  {
    int totalA = Mi * Ki;
    int totalB = Ni * Ki;
    int tpb = 256;
    fill_sf_kernel<<<(totalA + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys_SFA.data_ptr<uint8_t>(), SFA.data_ptr<uint8_t>(), layout_SFA, Mi, Ki, VEC);
    fill_sf_kernel<<<(totalB + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys_SFB.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(), layout_SFB, Ni, Ki, VEC);
    check_cuda(cudaGetLastError(), "fill_sf_kernel launch");
  }

  auto D = torch::empty({M, N},
      torch::TensorOptions().dtype(torch::kBFloat16).device(dev));

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {Mi, Ni, Ki, 1},
    { // mainloop
      reinterpret_cast<ElementA::DataType const*>(A.data_ptr<uint8_t>()), stride_A,
      reinterpret_cast<ElementB::DataType const*>(B.data_ptr<uint8_t>()), stride_B,
      reinterpret_cast<ElementA::ScaleFactorType const*>(phys_SFA.data_ptr<uint8_t>()), layout_SFA,
      reinterpret_cast<ElementB::ScaleFactorType const*>(phys_SFB.data_ptr<uint8_t>()), layout_SFB
    },
    { // epilogue  (C unused, beta=0; reuse D ptr for C)
      {1.0f, 0.0f},
      reinterpret_cast<ElementC const*>(D.data_ptr()), stride_C,
      reinterpret_cast<ElementD*>(D.data_ptr()), stride_D
    }
  };

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty({(int64_t)ws}, u8opts);

  cutlass::Status st = gemm.can_implement(arguments);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "can_implement failed: ", cutlassGetStatusString(st));
  st = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "initialize failed: ", cutlassGetStatusString(st));
  st = gemm.run(stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "run failed: ", cutlassGetStatusString(st));
  check_cuda(cudaStreamSynchronize(stream), "gemm sync");

  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mxfp8_gemm", &mxfp8_gemm, "MXFP8xMXFP8 block-scaled GEMM -> bf16 (SM120)");
}
