// Fused mixed-precision GEMM -> fp16, PyTorch extension for SM120 (Blackwell).
//
// ONE host launch that computes the full mixed-precision result the ProMix model
// needs, removing the Python-level 2-launch overhead (double sync, double .float(),
// Python add + scale, bf16 round-trip):
//
//   D_low  = nvfp4_gemm(A_low, B_low)      [M,N]  (NVFP4 block-scaled, K_low)
//   D_high = mxfp8_gemm(A_high, B_high)    [M,N]  (MXFP8 block-scaled, K_high)
//   out[m,n] = (D_low[m,n] + D_high[m,n]) * s_x[m] * s_w[n]   -> fp16
//
// Both GEMMs reuse the exact validated CUTLASS configs from nvfp4_gemm_ext.cu /
// mxfp8_gemm_ext.cu (cosine 0.999999). All SF fills + both GEMMs + the fused
// combine run on a single CUDA stream with a SINGLE cudaStreamSynchronize.
//
// This is "Rung A": one host call, one sync, fused combine epilogue. It is correct
// by construction (same MMA configs) and removes the dominant 2-launch overhead.
//
// Signature (see mixed_gemm.py mixed_gemm_fused):
//   mixed_gemm_fused(A_low uint8[M,Klow/2], SFA_low uint8[M,Klow/16],
//                    B_low uint8[N,Klow/2], SFB_low uint8[N,Klow/16],
//                    A_high uint8[M,Khigh], SFA_high uint8[M,Khigh/32],
//                    B_high uint8[N,Khigh], SFB_high uint8[N,Khigh/32],
//                    s_x f32[M], s_w f32[N], M, N, Klow, Khigh) -> fp16[M,N]

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

// ============================= NVFP4 config (ex79a) =============================
namespace nvcfg {
using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;
using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

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
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
} // namespace nvcfg

// ============================= MXFP8 config (ex79c) =============================
namespace mxcfg {
using ElementA   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;
using ElementB   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
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
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
} // namespace mxcfg

// ----------------------------- SF fill kernel (generic) -----------------------------
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

// ----------------------------- fused combine epilogue -----------------------------
// out[m,n] = (float(D_low) + float(D_high)) * s_x[m] * s_w[n]  -> fp16
__global__ void combine_scale_kernel(const __nv_bfloat16* __restrict__ D_low,
                                     const __nv_bfloat16* __restrict__ D_high,
                                     const float* __restrict__ s_x,
                                     const float* __restrict__ s_w,
                                     __half* __restrict__ out,
                                     int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  if (idx >= total) return;
  int m = idx / N;
  int n = idx % N;
  float v = __bfloat162float(D_low[idx]) + __bfloat162float(D_high[idx]);
  v *= s_x[m] * s_w[n];
  out[idx] = __float2half(v);
}

static inline void check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// ----------------------------- entry point -----------------------------
torch::Tensor mixed_gemm_fused(torch::Tensor A_low, torch::Tensor SFA_low,
                               torch::Tensor B_low, torch::Tensor SFB_low,
                               torch::Tensor A_high, torch::Tensor SFA_high,
                               torch::Tensor B_high, torch::Tensor SFB_high,
                               torch::Tensor s_x, torch::Tensor s_w,
                               int64_t M, int64_t N, int64_t Klow, int64_t Khigh) {
  TORCH_CHECK(A_low.is_cuda() && A_high.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(Klow % 16 == 0, "Klow must be divisible by 16");
  TORCH_CHECK(Khigh % 32 == 0, "Khigh must be divisible by 32");
  A_low = A_low.contiguous();   SFA_low = SFA_low.contiguous();
  B_low = B_low.contiguous();   SFB_low = SFB_low.contiguous();
  A_high = A_high.contiguous(); SFA_high = SFA_high.contiguous();
  B_high = B_high.contiguous(); SFB_high = SFB_high.contiguous();
  s_x = s_x.contiguous();       s_w = s_w.contiguous();

  auto dev = A_low.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto u8opts = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto bf16opts = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);

  int Mi = (int)M, Ni = (int)N, Kl = (int)Klow, Kh = (int)Khigh;

  // ---- intermediate fp(bf16) products, one buffer each ----
  auto D_low  = torch::empty({M, N}, bf16opts);
  auto D_high = torch::empty({M, N}, bf16opts);

  // ================= NVFP4 low-segment GEMM =================
  {
    using namespace nvcfg;
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {Mi, Kl, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {Ni, Kl, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {Mi, Ni, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {Mi, Ni, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Kl, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(Mi, Ni, Kl, 1));
    int64_t sfa_size = (int64_t)size(filter_zeros(layout_SFA));
    int64_t sfb_size = (int64_t)size(filter_zeros(layout_SFB));
    auto phys_SFA = torch::empty({sfa_size}, u8opts);
    auto phys_SFB = torch::empty({sfb_size}, u8opts);

    int tpb = 256;
    fill_sf_kernel<<<(Mi * Kl + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys_SFA.data_ptr<uint8_t>(), SFA_low.data_ptr<uint8_t>(), layout_SFA, Mi, Kl, 16);
    fill_sf_kernel<<<(Ni * Kl + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys_SFB.data_ptr<uint8_t>(), SFB_low.data_ptr<uint8_t>(), layout_SFB, Ni, Kl, 16);
    check_cuda(cudaGetLastError(), "nvfp4 fill_sf launch");

    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {Mi, Ni, Kl, 1},
      {
        reinterpret_cast<ElementA::DataType const*>(A_low.data_ptr<uint8_t>()), stride_A,
        reinterpret_cast<ElementB::DataType const*>(B_low.data_ptr<uint8_t>()), stride_B,
        reinterpret_cast<ElementA::ScaleFactorType const*>(phys_SFA.data_ptr<uint8_t>()), layout_SFA,
        reinterpret_cast<ElementB::ScaleFactorType const*>(phys_SFB.data_ptr<uint8_t>()), layout_SFB
      },
      {
        {1.0f, 0.0f},
        reinterpret_cast<ElementC const*>(D_low.data_ptr()), stride_C,
        reinterpret_cast<ElementD*>(D_low.data_ptr()), stride_D
      }
    };
    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)ws}, u8opts);
    cutlass::Status st = gemm.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "nvfp4 can_implement: ", cutlassGetStatusString(st));
    st = gemm.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "nvfp4 initialize: ", cutlassGetStatusString(st));
    st = gemm.run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "nvfp4 run: ", cutlassGetStatusString(st));
  }

  // ================= MXFP8 high-segment GEMM =================
  {
    using namespace mxcfg;
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {Mi, Kh, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {Ni, Kh, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {Mi, Ni, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {Mi, Ni, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Kh, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(Mi, Ni, Kh, 1));
    int64_t sfa_size = (int64_t)size(filter_zeros(layout_SFA));
    int64_t sfb_size = (int64_t)size(filter_zeros(layout_SFB));
    auto phys_SFA = torch::empty({sfa_size}, u8opts);
    auto phys_SFB = torch::empty({sfb_size}, u8opts);

    int tpb = 256;
    fill_sf_kernel<<<(Mi * Kh + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys_SFA.data_ptr<uint8_t>(), SFA_high.data_ptr<uint8_t>(), layout_SFA, Mi, Kh, 32);
    fill_sf_kernel<<<(Ni * Kh + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys_SFB.data_ptr<uint8_t>(), SFB_high.data_ptr<uint8_t>(), layout_SFB, Ni, Kh, 32);
    check_cuda(cudaGetLastError(), "mxfp8 fill_sf launch");

    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {Mi, Ni, Kh, 1},
      {
        reinterpret_cast<ElementA::DataType const*>(A_high.data_ptr<uint8_t>()), stride_A,
        reinterpret_cast<ElementB::DataType const*>(B_high.data_ptr<uint8_t>()), stride_B,
        reinterpret_cast<ElementA::ScaleFactorType const*>(phys_SFA.data_ptr<uint8_t>()), layout_SFA,
        reinterpret_cast<ElementB::ScaleFactorType const*>(phys_SFB.data_ptr<uint8_t>()), layout_SFB
      },
      {
        {1.0f, 0.0f},
        reinterpret_cast<ElementC const*>(D_high.data_ptr()), stride_C,
        reinterpret_cast<ElementD*>(D_high.data_ptr()), stride_D
      }
    };
    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)ws}, u8opts);
    cutlass::Status st = gemm.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "mxfp8 can_implement: ", cutlassGetStatusString(st));
    st = gemm.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "mxfp8 initialize: ", cutlassGetStatusString(st));
    st = gemm.run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "mxfp8 run: ", cutlassGetStatusString(st));
  }

  // ================= fused combine + outer scale -> fp16 =================
  auto out = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(dev));
  {
    int total = Mi * Ni;
    int tpb = 256;
    combine_scale_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(D_low.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(D_high.data_ptr()),
        s_x.data_ptr<float>(), s_w.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr()), Mi, Ni);
    check_cuda(cudaGetLastError(), "combine_scale launch");
  }

  check_cuda(cudaStreamSynchronize(stream), "fused sync");
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mixed_gemm_fused", &mixed_gemm_fused,
        "Fused mixed-precision NVFP4+MXFP8 block-scaled GEMM -> fp16 (SM120)");
}
