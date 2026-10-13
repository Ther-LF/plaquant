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
#include <algorithm>

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
using ThreadBlockShape = Shape<_128,_64,_128>;
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
using ThreadBlockShape = Shape<_128,_64,_128>;
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

// ----------------------------- Rung-B-fallback kernels -----------------------------
// Merged SF-fill: one launch fills all 4 interleaved SF buffers (NVFP4 A/B + MXFP8
// A/B) via a flat cumulative index space, so exactly (w0+w1+w2+w3) threads run with
// no idle oversizing. Removes 3 of the 4 fill launches of Rung A.
template <class L0, class L1, class L2, class L3>
__global__ void fill_sf_merged(
    uint8_t* p0, const uint8_t* l0, L0 lay0, int K0, int V0, int w0,
    uint8_t* p1, const uint8_t* l1, L1 lay1, int K1, int V1, int w1,
    uint8_t* p2, const uint8_t* l2, L2 lay2, int K2, int V2, int w2,
    uint8_t* p3, const uint8_t* l3, L3 lay3, int K3, int V3, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  if (idx < w0) { int r = idx / K0, k = idx % K0;
    p0[lay0(make_coord(r, k, 0))] = l0[r * (K0 / V0) + (k / V0)]; return; }
  idx -= w0;
  if (idx < w1) { int r = idx / K1, k = idx % K1;
    p1[lay1(make_coord(r, k, 0))] = l1[r * (K1 / V1) + (k / V1)]; return; }
  idx -= w1;
  if (idx < w2) { int r = idx / K2, k = idx % K2;
    p2[lay2(make_coord(r, k, 0))] = l2[r * (K2 / V2) + (k / V2)]; return; }
  idx -= w2;
  { int r = idx / K3, k = idx % K3;
    p3[lay3(make_coord(r, k, 0))] = l3[r * (K3 / V3) + (k / V3)]; }
}

// 2-region merged SF-fill (activation side only, for the pre-SF presf paths).
// Fills just the two A-side interleaved SF buffers (NVFP4 A + MXFP8 A) in one launch;
// the weight-side SFB is pre-interleaved once at model load (prepack_sf_b) and reused.
template <class L0, class L1>
__global__ void fill_sf_merged2(
    uint8_t* p0, const uint8_t* l0, L0 lay0, int K0, int V0, int w0,
    uint8_t* p1, const uint8_t* l1, L1 lay1, int K1, int V1, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  if (idx < w0) { int r = idx / K0, k = idx % K0;
    p0[lay0(make_coord(r, k, 0))] = l0[r * (K0 / V0) + (k / V0)]; return; }
  idx -= w0;
  { int r = idx / K1, k = idx % K1;
    p1[lay1(make_coord(r, k, 0))] = l1[r * (K1 / V1) + (k / V1)]; }
}

// out[m,n] = D[m,n] * s_x[m] * s_w[n]  -> fp16  (combine-add already done in GEMM2)
__global__ void scale_only_kernel(const __nv_bfloat16* __restrict__ D,
                                  const float* __restrict__ s_x,
                                  const float* __restrict__ s_w,
                                  __half* __restrict__ out, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int m = idx / N, n = idx % N;
  float v = __bfloat162float(D[idx]) * s_x[m] * s_w[n];
  out[idx] = __float2half(v);
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

// ----------------------------- Rung-B-fallback entry point -----------------------------
// Reuses the proven nvcfg/mxcfg CUTLASS configs. Reduces Rung A's 7 device launches
// (4 SF-fill + 2 GEMM + 1 combine) to 4 (1 merged SF-fill + 2 GEMM + 1 scale):
//   D_low  = nvfp4_gemm(A_low,B_low)                 -> bf16 D_low
//   D_comb = mxfp8_gemm(A_high,B_high) + 1.0*D_low   -> bf16 (beta=1 accumulate in epilogue)
//   out[m,n] = D_comb[m,n] * s_x[m] * s_w[n]         -> fp16
// The combine-add is folded into GEMM2's epilogue (beta=1), so no separate combine pass.
torch::Tensor mixed_gemm_fused_b(torch::Tensor A_low, torch::Tensor SFA_low,
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

  auto D_low  = torch::empty({M, N}, bf16opts);   // GEMM1 output
  auto D_comb = torch::empty({M, N}, bf16opts);   // GEMM2 output (= acc_high + D_low)

  // ---- strides + SF layouts for both GEMMs (needed before the merged fill) ----
  nvcfg::StrideA nv_sA = cutlass::make_cute_packed_stride(nvcfg::StrideA{}, {Mi, Kl, 1});
  nvcfg::StrideB nv_sB = cutlass::make_cute_packed_stride(nvcfg::StrideB{}, {Ni, Kl, 1});
  nvcfg::StrideC nv_sC = cutlass::make_cute_packed_stride(nvcfg::StrideC{}, {Mi, Ni, 1});
  nvcfg::StrideD nv_sD = cutlass::make_cute_packed_stride(nvcfg::StrideD{}, {Mi, Ni, 1});
  auto nv_lSFA = nvcfg::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Kl, 1));
  auto nv_lSFB = nvcfg::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(Mi, Ni, Kl, 1));

  mxcfg::StrideA mx_sA = cutlass::make_cute_packed_stride(mxcfg::StrideA{}, {Mi, Kh, 1});
  mxcfg::StrideB mx_sB = cutlass::make_cute_packed_stride(mxcfg::StrideB{}, {Ni, Kh, 1});
  mxcfg::StrideC mx_sC = cutlass::make_cute_packed_stride(mxcfg::StrideC{}, {Mi, Ni, 1});
  mxcfg::StrideD mx_sD = cutlass::make_cute_packed_stride(mxcfg::StrideD{}, {Mi, Ni, 1});
  auto mx_lSFA = mxcfg::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Kh, 1));
  auto mx_lSFB = mxcfg::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(Mi, Ni, Kh, 1));

  auto nv_phys_SFA = torch::empty({(int64_t)size(filter_zeros(nv_lSFA))}, u8opts);
  auto nv_phys_SFB = torch::empty({(int64_t)size(filter_zeros(nv_lSFB))}, u8opts);
  auto mx_phys_SFA = torch::empty({(int64_t)size(filter_zeros(mx_lSFA))}, u8opts);
  auto mx_phys_SFB = torch::empty({(int64_t)size(filter_zeros(mx_lSFB))}, u8opts);

  // ================= ONE merged SF-fill launch (4 regions) =================
  {
    int w0 = Mi * Kl, w1 = Ni * Kl, w2 = Mi * Kh, w3 = Ni * Kh;
    int total = w0 + w1 + w2 + w3;
    int tpb = 256;
    dim3 grid((total + tpb - 1) / tpb, 1, 1);
    fill_sf_merged<<<grid, tpb, 0, stream>>>(
        nv_phys_SFA.data_ptr<uint8_t>(), SFA_low.data_ptr<uint8_t>(), nv_lSFA, Kl, 16, w0,
        nv_phys_SFB.data_ptr<uint8_t>(), SFB_low.data_ptr<uint8_t>(), nv_lSFB, Kl, 16, w1,
        mx_phys_SFA.data_ptr<uint8_t>(), SFA_high.data_ptr<uint8_t>(), mx_lSFA, Kh, 32, w2,
        mx_phys_SFB.data_ptr<uint8_t>(), SFB_high.data_ptr<uint8_t>(), mx_lSFB, Kh, 32, total);
    check_cuda(cudaGetLastError(), "fill_sf_merged launch");
  }

  // ================= GEMM1: NVFP4 low segment -> bf16 D_low =================
  {
    using namespace nvcfg;
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {Mi, Ni, Kl, 1},
      {
        reinterpret_cast<ElementA::DataType const*>(A_low.data_ptr<uint8_t>()), nv_sA,
        reinterpret_cast<ElementB::DataType const*>(B_low.data_ptr<uint8_t>()), nv_sB,
        reinterpret_cast<ElementA::ScaleFactorType const*>(nv_phys_SFA.data_ptr<uint8_t>()), nv_lSFA,
        reinterpret_cast<ElementB::ScaleFactorType const*>(nv_phys_SFB.data_ptr<uint8_t>()), nv_lSFB
      },
      {
        {1.0f, 0.0f},
        reinterpret_cast<ElementC const*>(D_low.data_ptr()), nv_sC,
        reinterpret_cast<ElementD*>(D_low.data_ptr()), nv_sD
      }
    };
    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)ws}, u8opts);
    cutlass::Status st = gemm.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "nvfp4(b) can_implement: ", cutlassGetStatusString(st));
    st = gemm.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "nvfp4(b) initialize: ", cutlassGetStatusString(st));
    st = gemm.run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "nvfp4(b) run: ", cutlassGetStatusString(st));
  }

  // ========= GEMM2: MXFP8 high segment + beta*D_low -> bf16 D_comb =========
  {
    using namespace mxcfg;
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {Mi, Ni, Kh, 1},
      {
        reinterpret_cast<ElementA::DataType const*>(A_high.data_ptr<uint8_t>()), mx_sA,
        reinterpret_cast<ElementB::DataType const*>(B_high.data_ptr<uint8_t>()), mx_sB,
        reinterpret_cast<ElementA::ScaleFactorType const*>(mx_phys_SFA.data_ptr<uint8_t>()), mx_lSFA,
        reinterpret_cast<ElementB::ScaleFactorType const*>(mx_phys_SFB.data_ptr<uint8_t>()), mx_lSFB
      },
      {
        {1.0f, 1.0f},  // alpha=1, beta=1: D_comb = acc_high + 1.0 * D_low
        reinterpret_cast<ElementC const*>(D_low.data_ptr()), mx_sC,
        reinterpret_cast<ElementD*>(D_comb.data_ptr()), mx_sD
      }
    };
    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)ws}, u8opts);
    cutlass::Status st = gemm.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "mxfp8(b) can_implement: ", cutlassGetStatusString(st));
    st = gemm.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "mxfp8(b) initialize: ", cutlassGetStatusString(st));
    st = gemm.run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "mxfp8(b) run: ", cutlassGetStatusString(st));
  }

  // ================= scale -> fp16 =================
  auto out = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(dev));
  {
    int total = Mi * Ni, tpb = 256;
    scale_only_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(D_comb.data_ptr()),
        s_x.data_ptr<float>(), s_w.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr()), Mi, Ni);
    check_cuda(cudaGetLastError(), "scale_only launch");
  }

  check_cuda(cudaStreamSynchronize(stream), "fused_b sync");
  return out;
}

// ----------------------------- T1: weight-SF pre-interleave -----------------------------
// Run the SF-fill ONCE for a weight-side logical SF buffer and return the INTERLEAVED
// physical buffer that the GEMM reads. tile_atom_to_shape_SFB uses only (N,K,L) — it is
// independent of M — so this physical buffer is reusable across every call / every M.
// vec selects the config: 16 -> NVFP4 (nvcfg), 32 -> MXFP8 (mxcfg).
//   SFB_logical: uint8[N, K/vec]  ->  physical interleaved uint8[size(filter_zeros(layout))]
torch::Tensor prepack_sf_b(torch::Tensor SFB_logical, int64_t N, int64_t K, int64_t vec) {
  TORCH_CHECK(SFB_logical.is_cuda(), "SFB_logical must be a CUDA tensor");
  TORCH_CHECK(SFB_logical.scalar_type() == torch::kUInt8, "SFB_logical must be uint8");
  TORCH_CHECK(vec == 16 || vec == 32, "vec must be 16 (NVFP4) or 32 (MXFP8)");
  TORCH_CHECK(K % vec == 0, "K must be divisible by vec");
  SFB_logical = SFB_logical.contiguous();
  auto dev = SFB_logical.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto u8opts = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  int Ni = (int)N, Ki = (int)K;
  int total = Ni * Ki, tpb = 256;
  torch::Tensor phys;
  if (vec == 16) {
    auto layout = nvcfg::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, Ni, Ki, 1));
    phys = torch::empty({(int64_t)size(filter_zeros(layout))}, u8opts);
    fill_sf_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys.data_ptr<uint8_t>(), SFB_logical.data_ptr<uint8_t>(), layout, Ni, Ki, 16);
  } else {
    auto layout = mxcfg::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, Ni, Ki, 1));
    phys = torch::empty({(int64_t)size(filter_zeros(layout))}, u8opts);
    fill_sf_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys.data_ptr<uint8_t>(), SFB_logical.data_ptr<uint8_t>(), layout, Ni, Ki, 32);
  }
  check_cuda(cudaGetLastError(), "prepack_sf_b launch");
  check_cuda(cudaStreamSynchronize(stream), "prepack_sf_b sync");
  return phys;
}

// ----------------------------- T1: Rung-B presf entry point -----------------------------
// Same as mixed_gemm_fused_b but the weight-side SFB buffers are ALREADY interleaved
// (physSFB_low / physSFB_high from prepack_sf_b). Only the dynamic activation-side SF is
// filled per call (one merged 2-region launch), killing the dominant weight-SF scatter.
torch::Tensor mixed_gemm_fused_b_presf(
    torch::Tensor A_low, torch::Tensor SFA_low, torch::Tensor B_low, torch::Tensor physSFB_low,
    torch::Tensor A_high, torch::Tensor SFA_high, torch::Tensor B_high, torch::Tensor physSFB_high,
    torch::Tensor s_x, torch::Tensor s_w,
    int64_t M, int64_t N, int64_t Klow, int64_t Khigh) {
  TORCH_CHECK(A_low.is_cuda() && A_high.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(Klow % 16 == 0, "Klow must be divisible by 16");
  TORCH_CHECK(Khigh % 32 == 0, "Khigh must be divisible by 32");
  A_low = A_low.contiguous();     SFA_low = SFA_low.contiguous();
  B_low = B_low.contiguous();     physSFB_low = physSFB_low.contiguous();
  A_high = A_high.contiguous();   SFA_high = SFA_high.contiguous();
  B_high = B_high.contiguous();   physSFB_high = physSFB_high.contiguous();
  s_x = s_x.contiguous();         s_w = s_w.contiguous();

  auto dev = A_low.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto u8opts = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto bf16opts = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);

  int Mi = (int)M, Ni = (int)N, Kl = (int)Klow, Kh = (int)Khigh;

  auto D_low  = torch::empty({M, N}, bf16opts);
  auto D_comb = torch::empty({M, N}, bf16opts);

  nvcfg::StrideA nv_sA = cutlass::make_cute_packed_stride(nvcfg::StrideA{}, {Mi, Kl, 1});
  nvcfg::StrideB nv_sB = cutlass::make_cute_packed_stride(nvcfg::StrideB{}, {Ni, Kl, 1});
  nvcfg::StrideC nv_sC = cutlass::make_cute_packed_stride(nvcfg::StrideC{}, {Mi, Ni, 1});
  nvcfg::StrideD nv_sD = cutlass::make_cute_packed_stride(nvcfg::StrideD{}, {Mi, Ni, 1});
  auto nv_lSFA = nvcfg::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Kl, 1));
  auto nv_lSFB = nvcfg::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(Mi, Ni, Kl, 1));

  mxcfg::StrideA mx_sA = cutlass::make_cute_packed_stride(mxcfg::StrideA{}, {Mi, Kh, 1});
  mxcfg::StrideB mx_sB = cutlass::make_cute_packed_stride(mxcfg::StrideB{}, {Ni, Kh, 1});
  mxcfg::StrideC mx_sC = cutlass::make_cute_packed_stride(mxcfg::StrideC{}, {Mi, Ni, 1});
  mxcfg::StrideD mx_sD = cutlass::make_cute_packed_stride(mxcfg::StrideD{}, {Mi, Ni, 1});
  auto mx_lSFA = mxcfg::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Kh, 1));
  auto mx_lSFB = mxcfg::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(Mi, Ni, Kh, 1));

  // weight SF already interleaved: sizes must match the GEMM's expected layout image.
  TORCH_CHECK(physSFB_low.numel()  == (int64_t)size(filter_zeros(nv_lSFB)),
              "physSFB_low size mismatch: got ", physSFB_low.numel(),
              " expected ", (int64_t)size(filter_zeros(nv_lSFB)));
  TORCH_CHECK(physSFB_high.numel() == (int64_t)size(filter_zeros(mx_lSFB)),
              "physSFB_high size mismatch: got ", physSFB_high.numel(),
              " expected ", (int64_t)size(filter_zeros(mx_lSFB)));

  auto nv_phys_SFA = torch::empty({(int64_t)size(filter_zeros(nv_lSFA))}, u8opts);
  auto mx_phys_SFA = torch::empty({(int64_t)size(filter_zeros(mx_lSFA))}, u8opts);

  // ===== one merged fill of ONLY the activation-side SF (2 regions) =====
  {
    int w0 = Mi * Kl, w1 = Mi * Kh, total = w0 + w1, tpb = 256;
    fill_sf_merged2<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        nv_phys_SFA.data_ptr<uint8_t>(), SFA_low.data_ptr<uint8_t>(), nv_lSFA, Kl, 16, w0,
        mx_phys_SFA.data_ptr<uint8_t>(), SFA_high.data_ptr<uint8_t>(), mx_lSFA, Kh, 32, total);
    check_cuda(cudaGetLastError(), "fill_sf_merged2 launch");
  }

  // ===== GEMM1: NVFP4 low segment -> bf16 D_low =====
  {
    using namespace nvcfg;
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {Mi, Ni, Kl, 1},
      {
        reinterpret_cast<ElementA::DataType const*>(A_low.data_ptr<uint8_t>()), nv_sA,
        reinterpret_cast<ElementB::DataType const*>(B_low.data_ptr<uint8_t>()), nv_sB,
        reinterpret_cast<ElementA::ScaleFactorType const*>(nv_phys_SFA.data_ptr<uint8_t>()), nv_lSFA,
        reinterpret_cast<ElementB::ScaleFactorType const*>(physSFB_low.data_ptr<uint8_t>()), nv_lSFB
      },
      {
        {1.0f, 0.0f},
        reinterpret_cast<ElementC const*>(D_low.data_ptr()), nv_sC,
        reinterpret_cast<ElementD*>(D_low.data_ptr()), nv_sD
      }
    };
    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)ws}, u8opts);
    cutlass::Status st = gemm.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "nvfp4(b-presf) can_implement: ", cutlassGetStatusString(st));
    st = gemm.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "nvfp4(b-presf) initialize: ", cutlassGetStatusString(st));
    st = gemm.run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "nvfp4(b-presf) run: ", cutlassGetStatusString(st));
  }

  // ===== GEMM2: MXFP8 high segment + beta*D_low -> bf16 D_comb =====
  {
    using namespace mxcfg;
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {Mi, Ni, Kh, 1},
      {
        reinterpret_cast<ElementA::DataType const*>(A_high.data_ptr<uint8_t>()), mx_sA,
        reinterpret_cast<ElementB::DataType const*>(B_high.data_ptr<uint8_t>()), mx_sB,
        reinterpret_cast<ElementA::ScaleFactorType const*>(mx_phys_SFA.data_ptr<uint8_t>()), mx_lSFA,
        reinterpret_cast<ElementB::ScaleFactorType const*>(physSFB_high.data_ptr<uint8_t>()), mx_lSFB
      },
      {
        {1.0f, 1.0f},
        reinterpret_cast<ElementC const*>(D_low.data_ptr()), mx_sC,
        reinterpret_cast<ElementD*>(D_comb.data_ptr()), mx_sD
      }
    };
    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)ws}, u8opts);
    cutlass::Status st = gemm.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "mxfp8(b-presf) can_implement: ", cutlassGetStatusString(st));
    st = gemm.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "mxfp8(b-presf) initialize: ", cutlassGetStatusString(st));
    st = gemm.run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "mxfp8(b-presf) run: ", cutlassGetStatusString(st));
  }

  // ===== scale -> fp16 =====
  auto out = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(dev));
  {
    int total = Mi * Ni, tpb = 256;
    scale_only_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(D_comb.data_ptr()),
        s_x.data_ptr<float>(), s_w.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr()), Mi, Ni);
    check_cuda(cudaGetLastError(), "scale_only(presf) launch");
  }

  check_cuda(cudaStreamSynchronize(stream), "fused_b_presf sync");
  return out;
}

// ----------------------------- T1: Rung-A presf entry point -----------------------------
// Rung A (separate D_low/D_high + combine_scale) with pre-interleaved weight SF.
torch::Tensor mixed_gemm_fused_presf(
    torch::Tensor A_low, torch::Tensor SFA_low, torch::Tensor B_low, torch::Tensor physSFB_low,
    torch::Tensor A_high, torch::Tensor SFA_high, torch::Tensor B_high, torch::Tensor physSFB_high,
    torch::Tensor s_x, torch::Tensor s_w,
    int64_t M, int64_t N, int64_t Klow, int64_t Khigh) {
  TORCH_CHECK(A_low.is_cuda() && A_high.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(Klow % 16 == 0 && Khigh % 32 == 0, "K divisibility");
  A_low = A_low.contiguous();     SFA_low = SFA_low.contiguous();
  B_low = B_low.contiguous();     physSFB_low = physSFB_low.contiguous();
  A_high = A_high.contiguous();   SFA_high = SFA_high.contiguous();
  B_high = B_high.contiguous();   physSFB_high = physSFB_high.contiguous();
  s_x = s_x.contiguous();         s_w = s_w.contiguous();

  auto dev = A_low.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto u8opts = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto bf16opts = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);

  int Mi = (int)M, Ni = (int)N, Kl = (int)Klow, Kh = (int)Khigh;
  auto D_low  = torch::empty({M, N}, bf16opts);
  auto D_high = torch::empty({M, N}, bf16opts);

  // NVFP4 low segment
  {
    using namespace nvcfg;
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {Mi, Kl, 1});
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {Ni, Kl, 1});
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {Mi, Ni, 1});
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {Mi, Ni, 1});
    auto lSFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Kl, 1));
    auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(Mi, Ni, Kl, 1));
    TORCH_CHECK(physSFB_low.numel() == (int64_t)size(filter_zeros(lSFB)), "physSFB_low size mismatch");
    auto phys_SFA = torch::empty({(int64_t)size(filter_zeros(lSFA))}, u8opts);
    int tpb = 256;
    fill_sf_kernel<<<(Mi * Kl + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys_SFA.data_ptr<uint8_t>(), SFA_low.data_ptr<uint8_t>(), lSFA, Mi, Kl, 16);
    check_cuda(cudaGetLastError(), "nvfp4(a-presf) fill");
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {Mi, Ni, Kl, 1},
      {
        reinterpret_cast<ElementA::DataType const*>(A_low.data_ptr<uint8_t>()), sA,
        reinterpret_cast<ElementB::DataType const*>(B_low.data_ptr<uint8_t>()), sB,
        reinterpret_cast<ElementA::ScaleFactorType const*>(phys_SFA.data_ptr<uint8_t>()), lSFA,
        reinterpret_cast<ElementB::ScaleFactorType const*>(physSFB_low.data_ptr<uint8_t>()), lSFB
      },
      { {1.0f, 0.0f},
        reinterpret_cast<ElementC const*>(D_low.data_ptr()), sC,
        reinterpret_cast<ElementD*>(D_low.data_ptr()), sD }
    };
    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)ws}, u8opts);
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "nvfp4(a-presf) can_implement");
    TORCH_CHECK(gemm.initialize(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess, "nvfp4(a-presf) init");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "nvfp4(a-presf) run");
  }
  // MXFP8 high segment
  {
    using namespace mxcfg;
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {Mi, Kh, 1});
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {Ni, Kh, 1});
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {Mi, Ni, 1});
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {Mi, Ni, 1});
    auto lSFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Kh, 1));
    auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(Mi, Ni, Kh, 1));
    TORCH_CHECK(physSFB_high.numel() == (int64_t)size(filter_zeros(lSFB)), "physSFB_high size mismatch");
    auto phys_SFA = torch::empty({(int64_t)size(filter_zeros(lSFA))}, u8opts);
    int tpb = 256;
    fill_sf_kernel<<<(Mi * Kh + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys_SFA.data_ptr<uint8_t>(), SFA_high.data_ptr<uint8_t>(), lSFA, Mi, Kh, 32);
    check_cuda(cudaGetLastError(), "mxfp8(a-presf) fill");
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {Mi, Ni, Kh, 1},
      {
        reinterpret_cast<ElementA::DataType const*>(A_high.data_ptr<uint8_t>()), sA,
        reinterpret_cast<ElementB::DataType const*>(B_high.data_ptr<uint8_t>()), sB,
        reinterpret_cast<ElementA::ScaleFactorType const*>(phys_SFA.data_ptr<uint8_t>()), lSFA,
        reinterpret_cast<ElementB::ScaleFactorType const*>(physSFB_high.data_ptr<uint8_t>()), lSFB
      },
      { {1.0f, 0.0f},
        reinterpret_cast<ElementC const*>(D_high.data_ptr()), sC,
        reinterpret_cast<ElementD*>(D_high.data_ptr()), sD }
    };
    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)ws}, u8opts);
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "mxfp8(a-presf) can_implement");
    TORCH_CHECK(gemm.initialize(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess, "mxfp8(a-presf) init");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "mxfp8(a-presf) run");
  }
  // combine + scale -> fp16
  auto out = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(dev));
  {
    int total = Mi * Ni, tpb = 256;
    combine_scale_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(D_low.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(D_high.data_ptr()),
        s_x.data_ptr<float>(), s_w.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr()), Mi, Ni);
    check_cuda(cudaGetLastError(), "combine_scale(presf) launch");
  }
  check_cuda(cudaStreamSynchronize(stream), "fused_presf sync");
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mixed_gemm_fused", &mixed_gemm_fused,
        "Fused mixed-precision NVFP4+MXFP8 block-scaled GEMM -> fp16 (SM120)");
  m.def("mixed_gemm_fused_b", &mixed_gemm_fused_b,
        "Rung-B-fallback: merged SF-fill + GEMM2 beta=1 accumulate + scale (SM120)");
  m.def("prepack_sf_b", &prepack_sf_b,
        "Pre-interleave a weight-side logical SF buffer once (vec=16 NVFP4 / 32 MXFP8) -> physical SF");
  m.def("mixed_gemm_fused_b_presf", &mixed_gemm_fused_b_presf,
        "Rung-B with pre-interleaved weight SF (only activation SF filled per call) (SM120)");
  m.def("mixed_gemm_fused_presf", &mixed_gemm_fused_presf,
        "Rung-A with pre-interleaved weight SF (SM120)");
}
