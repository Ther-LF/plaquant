// presf_cfg_ext.cu — round-24 PURE-FORMAT (NVFP4 / MXFP8) config grid-search harness.
//
// A MACRO-PARAMETERIZED copy of the config layer of pure_presf_ext.cu (PROTECTED,
// untouched). Same methodology: pure single-format full-K block-scaled GEMM with
// PRE-INTERLEAVED weight SF (prepack_sf_*_b run once), only the activation SF filled
// per call. The ONLY thing that varies vs pure_presf_ext.cu is the CollectiveBuilder
// CONFIG (TileShape / ClusterShape / mainloop stage count) swapped via -D macros, so
// each config is a separate compile / separate named extension. The GEMM math and the
// SF-scatter path are byte-identical to pure_presf_ext.cu.
//
// WHY: pure_presf_ext.cu hardcodes 128x128x128 + StageCountAutoCarveout. For a SINGLE
// format only ONE format's operand smem is resident (not the de-union dual of the
// co-design), so bigger tiles (256x128, 128x256, 256x256) and/or different stage
// counts that the co-design CANNOT fit may be legal and faster here. Round-24 fixes the
// "under-tuned FP4/FP8 endpoints" by grid-searching the best per shape x M.
//
// Macros (defaults = the shipping pure_presf 128x128x128 1x1x1, StageCountAutoCarveout):
//   PSCFG_TM/TN/TK   CTA TileShape
//   PSCFG_CM/CN/CK   ClusterShape (SM120 only accepts 1x1x1)
//   PSCFG_STAGES     if defined (>0): explicit StageCount<N>; else StageCountAutoCarveout
//   PSCFG_SCHED      0=KernelScheduleAuto(default) 1=Cooperative 2=Pingpong
//
// One TU exposes BOTH formats (nvfp4_gemm_presf + mxfp8_gemm_presf) + a compat_report
// giving each format's mainloop smem + the auto-chosen stage count + build/fit status.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <string>

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

#ifndef PSCFG_TM
#define PSCFG_TM 128
#endif
#ifndef PSCFG_TN
#define PSCFG_TN 128
#endif
#ifndef PSCFG_TK
#define PSCFG_TK 128
#endif
#ifndef PSCFG_CM
#define PSCFG_CM 1
#endif
#ifndef PSCFG_CN
#define PSCFG_CN 1
#endif
#ifndef PSCFG_CK
#define PSCFG_CK 1
#endif
// PSCFG_STAGES: 0 (or undefined) => StageCountAutoCarveout; >0 => explicit StageCount<N>
#ifndef PSCFG_STAGES
#define PSCFG_STAGES 0
#endif
// PSCFG_SCHED: 0 auto, 1 cooperative, 2 pingpong
#ifndef PSCFG_SCHED
#define PSCFG_SCHED 0
#endif

static inline void check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

template <class LayoutSF>
__global__ void pscfg_fill_sf_kernel(uint8_t* __restrict__ phys,
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

#if PSCFG_SCHED == 1
using KSchedTag = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using ESchedTag = cutlass::epilogue::TmaWarpSpecializedCooperative;
#elif PSCFG_SCHED == 2
using KSchedTag = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESchedTag = cutlass::epilogue::TmaWarpSpecialized;
#else
using KSchedTag = cutlass::gemm::collective::KernelScheduleAuto;
using ESchedTag = cutlass::epilogue::collective::EpilogueScheduleAuto;
#endif

template <class ElementAB, int Alignment, int VEC>
struct PresfCfgGemm {
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
  using ThreadBlockShape = Shape<Int<PSCFG_TM>, Int<PSCFG_TN>, Int<PSCFG_TK>>;
  using ClusterShape     = Shape<Int<PSCFG_CM>, Int<PSCFG_CN>, Int<PSCFG_CK>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ThreadBlockShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      ESchedTag
    >::CollectiveOp;

#if PSCFG_STAGES > 0
  using StageArg = cutlass::gemm::collective::StageCount<PSCFG_STAGES>;
#else
  using StageArg = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
#endif

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, Alignment,
      ElementB, LayoutBTag, Alignment,
      ElementAccumulator,
      ThreadBlockShape, ClusterShape,
      StageArg,
      KSchedTag
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using BSConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  static constexpr int chosen_stages = CollectiveMainloop::DispatchPolicy::Stages;
  static constexpr int mainloop_smem = (int)sizeof(typename CollectiveMainloop::SharedStorage);
  static constexpr int epilogue_smem = (int)sizeof(typename CollectiveEpilogue::SharedStorage);

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
    pscfg_fill_sf_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(), layout_SFB, Ni, Ki, VEC);
    check_cuda(cudaGetLastError(), "pscfg prepack_b launch");
    check_cuda(cudaStreamSynchronize(stream), "pscfg prepack_b sync");
    return phys;
  }

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
      pscfg_fill_sf_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
          phys_SFA.data_ptr<uint8_t>(), SFA.data_ptr<uint8_t>(), layout_SFA, Mi, Ki, VEC);
      check_cuda(cudaGetLastError(), "pscfg fill SFA launch");
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
    check_cuda(cudaStreamSynchronize(stream), "pscfg gemm sync");
    return D;
  }
};

using PresfNV = PresfCfgGemm<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 32, 16>;
using PresfMX = PresfCfgGemm<cutlass::mx_float8_t<cutlass::float_e4m3_t>, 16, 32>;

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

static std::string pscfg_compat_report() {
  std::string s;
  s += "PSCFG_BUILD=OK\n";
  s += "  tile=" + std::to_string(PSCFG_TM) + "x" + std::to_string(PSCFG_TN) + "x" + std::to_string(PSCFG_TK) + "\n";
  s += "  cluster=" + std::to_string(PSCFG_CM) + "x" + std::to_string(PSCFG_CN) + "x" + std::to_string(PSCFG_CK) + "\n";
  s += "  sched=" + std::to_string(PSCFG_SCHED) + " stages_macro=" + std::to_string(PSCFG_STAGES) + "\n";
  s += "  NV: stages=" + std::to_string(PresfNV::chosen_stages)
     + " mainloop_smem=" + std::to_string(PresfNV::mainloop_smem)
     + " epi_smem=" + std::to_string(PresfNV::epilogue_smem) + "\n";
  s += "  MX: stages=" + std::to_string(PresfMX::chosen_stages)
     + " mainloop_smem=" + std::to_string(PresfMX::mainloop_smem)
     + " epi_smem=" + std::to_string(PresfMX::epilogue_smem) + "\n";
  return s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepack_sf_fp4_b", &prepack_sf_fp4_b);
  m.def("nvfp4_gemm_presf", &nvfp4_gemm_presf);
  m.def("prepack_sf_fp8_b", &prepack_sf_fp8_b);
  m.def("mxfp8_gemm_presf", &mxfp8_gemm_presf);
  m.def("compat_report", &pscfg_compat_report);
}
