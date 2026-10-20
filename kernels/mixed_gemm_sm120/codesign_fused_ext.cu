// codesign_fused_ext.cu — STAGE 2 of the co-design rewrite (b1-opt round-14).
//
// Dual-format shared-accumulator fused GEMM: one kernel launch computes
//   D = (A_low  @ B_low^T   scaled by NVFP4 block-16 SFs)      [Phase 1, K_low]
//     + (A_high @ B_high^T  scaled by MXFP8 block-32 SFs)      [Phase 2, K_high]
// into ONE glue-owned FP32 accumulator fragment (no clear between phases, no gmem
// intermediate), then one shared epilogue -> bf16.
//
// Built from:
//   * codesign_sm120_bs_mma.hpp   — fork of the stock SM120 block-scaled collective,
//                                    adds ClearAccum (Phase 2 accumulates, no reset).
//   * codesign_fused_kernel.hpp   — fork of the SM90-cooperative GemmUniversal glue,
//                                    owns TWO collectives + the shared accumulator.
//
// Data contract identical to mixed_gemm.py oracle:
//   NVFP4: E2M1 codes, 2xFP4/byte, block-16 UE4M3 SF, A=(M,Klo)/B=(N,Klo).
//   MXFP8: E4M3, 1 byte/elem, block-32 UE8M0 SF, A=(M,Khi)/B=(N,Khi).
//   SF logical (rows, K/VEC) scattered into CUTLASS interleaved atom buffer.
//   high_fraction = 1/8 : K_high = K/8, K_low = 7K/8 (reorder [low|main|high]).
//
// Shipping kernel b1_tma_ext.cu and PROTECTED files are UNTOUCHED.

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

#include "codesign_sm120_bs_mma.hpp"
#include "codesign_fused_kernel.hpp"

using namespace cute;

#ifndef CD_NV_STAGES
#define CD_NV_STAGES 4
#endif
#ifndef CD_MX_STAGES
#define CD_MX_STAGES 2
#endif

static inline void check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

template <class LayoutSF>
__global__ void cd_fill_sf_kernel(uint8_t* __restrict__ phys,
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

// Decompose a MainloopSm120TmaWarpSpecializedBlockScaled policy into its parts so
// the forked collective can be re-instantiated with the builder's exact config.
namespace {
template <class Policy> struct PolicyParts;
template <int S, int Sch, class CS, class KS>
struct PolicyParts<cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled<S, Sch, CS, KS>> {
  static constexpr int Stages = S;
  static constexpr int SchedStages = Sch;
  using ClusterShape = CS;
  using KernelSchedule = KS;
};
}

// Fork a stock SM120 block-scaled CollectiveMma into our CodesignBlockScaledMma
// (same template config, adds ClearAccum).
template <class Stock, bool ClearAccum>
using ForkMma = cutlass::gemm::collective::CodesignBlockScaledMma<
    PolicyParts<typename Stock::DispatchPolicy>::Stages,
    PolicyParts<typename Stock::DispatchPolicy>::SchedStages,
    typename PolicyParts<typename Stock::DispatchPolicy>::ClusterShape,
    typename PolicyParts<typename Stock::DispatchPolicy>::KernelSchedule,
    typename Stock::TileShape,
    typename Stock::ElementPairA, typename Stock::StridePairA,
    typename Stock::ElementPairB, typename Stock::StridePairB,
    typename Stock::TiledMma,
    typename Stock::GmemTiledCopyPairA, typename Stock::SmemLayoutAtomsA, typename Stock::SmemCopyAtomsA, typename Stock::TransformA,
    typename Stock::GmemTiledCopyPairB, typename Stock::SmemLayoutAtomsB, typename Stock::SmemCopyAtomsB, typename Stock::TransformB,
    ClearAccum>;

// Per-format stock CollectiveMma via CollectiveBuilder (configs verbatim from
// pure_presf_ext.cu / ex79a,c). Only the mainloop differs per format.
template <class ElementAB, int Alignment, int Stages>
struct StockBuilder {
  using ElementD = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  // SM120 (RTX Pro 6000 Blackwell) has only ~101 KB opt-in smem/block. The two
  // block-scaled collectives (NV+MX) are UNIONED in the forked glue (sequential
  // phases), so BLK_K=128 fits: union size = max(NV stages, MX stages) tensor smem.
  using TileShape = Shape<_128,_128,_128>;
  using ClusterShape = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementAB, cutlass::layout::RowMajor, Alignment,
      ElementAB, cutlass::layout::ColumnMajor, Alignment,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<Stages>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
};

using NVbuild = StockBuilder<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 32, CD_NV_STAGES>;
using MXbuild = StockBuilder<cutlass::mx_float8_t<cutlass::float_e4m3_t>, 16, CD_MX_STAGES>;

using StockNV = NVbuild::CollectiveMainloop;
using StockMX = MXbuild::CollectiveMainloop;
using Epilogue = NVbuild::CollectiveEpilogue;   // shared epilogue (both formats identical)

using ForkNV = ForkMma<StockNV, /*ClearAccum=*/true>;    // Phase 1 clears
using ForkMX = ForkMma<StockMX, /*ClearAccum=*/false>;   // Phase 2 accumulates

template <bool DualPhase>
using FusedKernel = codesign::CodesignFusedKernel<
    Shape<int,int,int,int>, ForkNV, ForkMX, Epilogue,
    cutlass::gemm::StaticPersistentScheduler, DualPhase>;

using GemmDual   = cutlass::gemm::device::GemmUniversalAdapter<FusedKernel<true>>;
using GemmSingle = cutlass::gemm::device::GemmUniversalAdapter<FusedKernel<false>>;

using BSConfigNV = typename StockNV::Sm1xxBlkScaledConfig;
using BSConfigMX = typename StockMX::Sm1xxBlkScaledConfig;
using StrideA = typename FusedKernel<true>::StrideA;
using StrideB = typename FusedKernel<true>::StrideB;
using StrideC = typename FusedKernel<true>::StrideC;
using StrideD = typename FusedKernel<true>::StrideD;
using ElementDataNV_A = typename cutlass::nv_float4_t<cutlass::float_e2m1_t>::DataType;
using ElementDataNV_SF = typename cutlass::nv_float4_t<cutlass::float_e2m1_t>::ScaleFactorType;
using ElementDataMX_A = typename cutlass::mx_float8_t<cutlass::float_e4m3_t>::DataType;
using ElementDataMX_SF = typename cutlass::mx_float8_t<cutlass::float_e4m3_t>::ScaleFactorType;

// ---- SF scatter helpers (per format) ----
template <class BSConfig, int VEC>
static torch::Tensor prepack_b_impl(torch::Tensor SFB, int64_t N, int64_t K) {
  TORCH_CHECK(SFB.is_cuda() && SFB.scalar_type() == torch::kUInt8, "SFB must be CUDA uint8");
  SFB = SFB.contiguous();
  auto dev = SFB.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev.index());
  int Ni = (int)N, Ki = (int)K;
  auto layout_SFB = BSConfig::tile_atom_to_shape_SFB(make_shape(1, Ni, Ki, 1));
  int64_t sz = (int64_t)size(filter_zeros(layout_SFB));
  auto phys = torch::empty({sz}, torch::TensorOptions().dtype(torch::kUInt8).device(dev));
  int total = Ni * Ki, tpb = 256;
  cd_fill_sf_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
      phys.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(), layout_SFB, Ni, Ki, VEC);
  check_cuda(cudaGetLastError(), "prepack_b launch");
  check_cuda(cudaStreamSynchronize(stream), "prepack_b sync");
  return phys;
}

torch::Tensor cd_prepack_sf_fp4_b(torch::Tensor SFB, int64_t N, int64_t K) {
  TORCH_CHECK(K % 16 == 0, "K%16");
  return prepack_b_impl<BSConfigNV, 16>(SFB, N, K);
}
torch::Tensor cd_prepack_sf_fp8_b(torch::Tensor SFB, int64_t N, int64_t K) {
  TORCH_CHECK(K % 32 == 0, "K%32");
  return prepack_b_impl<BSConfigMX, 32>(SFB, N, K);
}

// Build a collective's mainloop Arguments (A/SFA/B/physSFB already on device).
template <class Coll, class BSConfig, int VEC>
static typename Coll::Arguments make_mainloop_args(torch::Tensor A, torch::Tensor SFA,
                                torch::Tensor B, torch::Tensor physSFB,
                                torch::Tensor& phys_SFA_out,
                                int Mi, int Ni, int Ki, cudaStream_t stream) {
  auto dev = A.device();
  auto layout_SFA = BSConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Ki, 1));
  auto layout_SFB = BSConfig::tile_atom_to_shape_SFB(make_shape(Mi, Ni, Ki, 1));
  auto u8opts = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  int64_t sfa_size = (int64_t)size(filter_zeros(layout_SFA));
  phys_SFA_out = torch::empty({sfa_size}, u8opts);
  int total = Mi * Ki, tpb = 256;
  cd_fill_sf_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
      phys_SFA_out.data_ptr<uint8_t>(), SFA.data_ptr<uint8_t>(), layout_SFA, Mi, Ki, VEC);
  check_cuda(cudaGetLastError(), "fill SFA launch");

  using ElemA = typename Coll::ElementA;      // packed data type (fp4/fp8)
  using ElemSF = typename Coll::ElementSF;
  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {Mi, Ki, 1});
  StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {Ni, Ki, 1});
  typename Coll::Arguments m;
  m.ptr_A = reinterpret_cast<ElemA const*>(A.data_ptr<uint8_t>());
  m.dA = sA;
  m.ptr_B = reinterpret_cast<ElemA const*>(B.data_ptr<uint8_t>());
  m.dB = sB;
  m.ptr_SFA = reinterpret_cast<ElemSF const*>(phys_SFA_out.data_ptr<uint8_t>());
  m.layout_SFA = layout_SFA;
  m.ptr_SFB = reinterpret_cast<ElemSF const*>(physSFB.data_ptr<uint8_t>());
  m.layout_SFB = layout_SFB;
  return m;
}

static typename Epilogue::Arguments make_epi_args(torch::Tensor D, int Mi, int Ni) {
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {Mi, Ni, 1});
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {Mi, Ni, 1});
  using ElementC = typename Epilogue::ElementC;
  using ElementD = typename Epilogue::ElementD;
  return {
    {1.0f, 0.0f},
    reinterpret_cast<ElementC const*>(D.data_ptr()), sC,
    reinterpret_cast<ElementD*>(D.data_ptr()), sD
  };
}

// ---- Single-phase NVFP4 through the FORKED glue (Milestone-A gate) ----
torch::Tensor cd_single_nvfp4_gemm(torch::Tensor A, torch::Tensor SFA,
                                   torch::Tensor B, torch::Tensor physSFB,
                                   int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(K % 16 == 0, "K%16");
  A = A.contiguous(); B = B.contiguous(); SFA = SFA.contiguous(); physSFB = physSFB.contiguous();
  auto dev = A.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev.index());
  int Mi=(int)M, Ni=(int)N, Ki=(int)K;
  auto D = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));

  torch::Tensor phys_SFA;
  auto nv_args = make_mainloop_args<ForkNV, BSConfigNV, 16>(
      A, SFA, B, physSFB, phys_SFA, Mi, Ni, Ki, stream);

  typename GemmSingle::Arguments arguments;
  arguments.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  arguments.problem_shape = {Mi, Ni, Ki, 1};
  arguments.mainloop_nv = nv_args;
  arguments.K_high = 0;
  arguments.epilogue = make_epi_args(D, Mi, Ni);

  GemmSingle gemm;
  size_t ws = GemmSingle::get_workspace_size(arguments);
  auto workspace = torch::empty({(int64_t)ws}, torch::TensorOptions().dtype(torch::kUInt8).device(dev));
  auto st = gemm.can_implement(arguments);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "single can_implement: ", cutlassGetStatusString(st));
  st = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "single initialize: ", cutlassGetStatusString(st));
  st = gemm.run(stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "single run: ", cutlassGetStatusString(st));
  check_cuda(cudaStreamSynchronize(stream), "single sync");
  return D;
}

// ---- Dual-phase fused NVFP4(K_low) + MXFP8(K_high) ----
torch::Tensor cd_fused_gemm(torch::Tensor A_lo, torch::Tensor SFA_lo, torch::Tensor B_lo, torch::Tensor physSFB_lo,
                            torch::Tensor A_hi, torch::Tensor SFA_hi, torch::Tensor B_hi, torch::Tensor physSFB_hi,
                            int64_t M, int64_t N, int64_t K_lo, int64_t K_hi) {
  TORCH_CHECK(K_lo % 16 == 0 && K_hi % 32 == 0, "K_lo%16, K_hi%32");
  A_lo=A_lo.contiguous(); B_lo=B_lo.contiguous(); SFA_lo=SFA_lo.contiguous(); physSFB_lo=physSFB_lo.contiguous();
  A_hi=A_hi.contiguous(); B_hi=B_hi.contiguous(); SFA_hi=SFA_hi.contiguous(); physSFB_hi=physSFB_hi.contiguous();
  auto dev = A_lo.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev.index());
  int Mi=(int)M, Ni=(int)N, Klo=(int)K_lo, Khi=(int)K_hi;
  auto D = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));

  torch::Tensor phys_SFA_lo, phys_SFA_hi;
  auto nv_args = make_mainloop_args<ForkNV, BSConfigNV, 16>(
      A_lo, SFA_lo, B_lo, physSFB_lo, phys_SFA_lo, Mi, Ni, Klo, stream);
  auto mx_args = make_mainloop_args<ForkMX, BSConfigMX, 32>(
      A_hi, SFA_hi, B_hi, physSFB_hi, phys_SFA_hi, Mi, Ni, Khi, stream);

  typename GemmDual::Arguments arguments;
  arguments.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  arguments.problem_shape = {Mi, Ni, Klo, 1};
  arguments.mainloop_nv = nv_args;
  arguments.mainloop_mx = mx_args;
  arguments.K_high = Khi;
  arguments.epilogue = make_epi_args(D, Mi, Ni);

  GemmDual gemm;
  size_t ws = GemmDual::get_workspace_size(arguments);
  auto workspace = torch::empty({(int64_t)ws}, torch::TensorOptions().dtype(torch::kUInt8).device(dev));
  auto st = gemm.can_implement(arguments);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "fused can_implement: ", cutlassGetStatusString(st));
  st = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "fused initialize: ", cutlassGetStatusString(st));
  st = gemm.run(stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "fused run: ", cutlassGetStatusString(st));
  check_cuda(cudaStreamSynchronize(stream), "fused sync");
  return D;
}

static std::string cd_compat_report() {
  using AccNV = typename ForkNV::ElementAccumulator;
  using AccMX = typename ForkMX::ElementAccumulator;
  std::string s;
  s += "CODESIGN_FUSED_BUILD=OK\n";
  s += "  NV stages=" + std::to_string(CD_NV_STAGES) + " MX stages=" + std::to_string(CD_MX_STAGES) + "\n";
  s += "  SharedStorageSize(dual) = " + std::to_string((int)FusedKernel<true>::SharedStorageSize) + " bytes\n";
  s += "  SharedStorageSize(single) = " + std::to_string((int)FusedKernel<false>::SharedStorageSize) + " bytes\n";
  s += "  MaxThreadsPerBlock = " + std::to_string((int)FusedKernel<true>::MaxThreadsPerBlock) + "\n";
  s += "  accum NV=" + std::string(std::is_same<AccNV,float>::value ? "float" : "?") +
       " MX=" + std::string(std::is_same<AccMX,float>::value ? "float" : "?") + "\n";
  return s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepack_sf_fp4_b", &cd_prepack_sf_fp4_b, "prepack NVFP4 weight SF");
  m.def("prepack_sf_fp8_b", &cd_prepack_sf_fp8_b, "prepack MXFP8 weight SF");
  m.def("single_nvfp4_gemm", &cd_single_nvfp4_gemm, "single-format NVFP4 through forked glue (gate)");
  m.def("fused_gemm", &cd_fused_gemm, "dual-format NVFP4(Klo)+MXFP8(Khi) shared-accumulator fused GEMM");
  m.def("compat_report", &cd_compat_report, "build/smem report");
}
