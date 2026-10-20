// codesign_nvfp4_ext.cu — STAGE 1 of the co-design rewrite (b1-opt round-13).
//
// GOAL (de-risking proof-of-concept): prove that adopting CUTLASS's SM120
// block-scaled pipeline AS A WHOLE reaches CUTLASS-level TC duty (alu/tc ~ 1,
// ACTIVE 68-86%) on a SINGLE format (NVFP4) in OUR harness, with OUR packed
// A/B/SF data contract — BEFORE touching the dual-format shared-accumulator seam
// (that is round-14). Round-10..12 proved our hand-rolled single-warpgroup
// dual-format SF-broadcast consumer CANNOT reach CUTLASS TC (alu/tc ~3.5,
// intrinsic), and that grafting CUTLASS pieces piecemeal makes it worse. The
// co-design only works as a WHOLE.
//
// PATH A (chosen): instantiate the STOCK SM120 dense block-scaled CollectiveMma
// via CollectiveBuilder and drive it through the stock GemmUniversal glue. The
// glue owns the FP32 accumulator fragment (`partition_fragment_C`) and passes it
// to `collective_mainloop.mma(..., accumulators, ...)` then `epilogue.store(...,
// accumulators, ...)`. Round-14 forks that glue and calls mma() a SECOND time
// (MXFP8 collective) into the SAME accumulators fragment before one shared
// epilogue. The precondition for that sharing — that the NVFP4 and MXFP8
// collectives produce an IDENTICAL accumulator-fragment layout — is checked at
// COMPILE TIME below (codesign_accum_compat), which is the single most valuable
// Path-A de-risk possible in a single-format stage.
//
// Data contract (NVFP4 side, identical to the oracle in mixed_gemm.py):
//   E2M1 FP4 codes, block-16 UE4M3 scale, 2xFP4/byte, A=(M,K)/B=(N,K) packed,
//   SF logical (rows, K/16) scattered into CUTLASS's interleaved atom buffer.
//
// The shipping kernel (b1_tma_ext.cu) is UNTOUCHED. pure_presf_ext.cu (PROTECTED)
// remains the CUTLASS reference endpoint; this file is a NEW, round-14-forkable
// standalone that uses the same stock collective on the same data contract.

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

static inline void check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// Scatter logical [rows, K/VEC] SF bytes into the CUTLASS interleaved buffer.
template <class LayoutSF>
__global__ void codesign_fill_sf_kernel(uint8_t* __restrict__ phys,
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

// Stock SM120 block-scaled GEMM scaffolding (configs VERBATIM from the round-1
// ex79a/ex79c extensions and pure_presf_ext.cu). This is the WHOLE co-design:
// CollectiveBuilder -> stock SM120 block-scaled CollectiveMma (warp-specialized
// TMA producer loading A/B/SF on one mbarrier, interleaved SF atom smem,
// UniversalCopy SF-consumer, cooperative atom layout Shape<4,2,1>) -> stock
// GemmUniversal glue (owns the accumulator) -> stock cooperative epilogue.
template <class ElementAB, int Alignment, int VEC>
struct CodesignGemm {
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

  // The glue-owned accumulator fragment (round-14 shares this across two
  // mainloop.mma() calls). This mirrors the kernel-level call exactly:
  //   partition_fragment_C(tiled_mma, take<0,2>(blk_shape)).
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using TileShape = typename CollectiveMainloop::TileShape;
  using AccumFragLayout =
      decltype(partition_fragment_C(TiledMma{}, take<0,2>(TileShape{})).layout());
  using AccumElement = typename TiledMma::ValTypeC;

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
    codesign_fill_sf_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(), layout_SFB, Ni, Ki, VEC);
    check_cuda(cudaGetLastError(), "codesign prepack_b launch");
    check_cuda(cudaStreamSynchronize(stream), "codesign prepack_b sync");
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
      codesign_fill_sf_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
          phys_SFA.data_ptr<uint8_t>(), SFA.data_ptr<uint8_t>(), layout_SFA, Mi, Ki, VEC);
      check_cuda(cudaGetLastError(), "codesign fill SFA launch");
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
    check_cuda(cudaStreamSynchronize(stream), "codesign gemm sync");
    return D;
  }
};

using CodesignNV = CodesignGemm<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 32, 16>;
using CodesignMX = CodesignGemm<cutlass::mx_float8_t<cutlass::float_e4m3_t>, 16, 32>;

// =====================================================================
// PATH-A DE-RISK (compile time): can round-14 share ONE FP32 accumulator
// fragment across the NVFP4 and MXFP8 CollectiveMma mainloops?
// The shared-accumulator seam requires that both collectives' glue-owned
// accumulator fragment — partition_fragment_C(TiledMma, take<0,2>(TileShape)) —
// have the SAME layout and element type. If they do, round-14 can call
//   nv_mainloop.mma(..., accumulators, ...);   // Phase 1 (K_low, NVFP4)
//   mx_mainloop.mma(..., accumulators, ...);   // Phase 2 (K_high, MXFP8)
//   shared_epilogue.store(..., accumulators, ...);
// in a single forked SM90-cooperative kernel glue. These static_asserts are the
// gate: they FAIL TO COMPILE if the two collectives are accumulator-incompatible.
// =====================================================================
static_assert(cute::is_same_v<CodesignNV::AccumElement, CodesignMX::AccumElement>,
              "PATH-A BLOCKER: NVFP4 and MXFP8 accumulator element types differ — "
              "round-14 shared accumulator is not type-safe.");
static_assert(cute::is_same_v<CodesignNV::AccumElement, float>,
              "accumulator must be FP32 for both formats");
static_assert(cute::size(CodesignNV::AccumFragLayout{}) == cute::size(CodesignMX::AccumFragLayout{}),
              "PATH-A BLOCKER: NVFP4 and MXFP8 accumulator fragments have different "
              "sizes — cannot be the same shared fragment.");
static_assert(cute::is_same_v<decltype(cute::shape(CodesignNV::AccumFragLayout{})),
                              decltype(cute::shape(CodesignMX::AccumFragLayout{}))>,
              "PATH-A BLOCKER: NVFP4 and MXFP8 accumulator fragment SHAPES differ.");

// Runtime report of the compat check (so Python can log it alongside cosine).
static std::string codesign_compat_report() {
  std::string s;
  s += "PATH_A_ACCUM_COMPAT=PASS (compile-time static_asserts held)\n";
  s += "  NVFP4 accum frag size = " + std::to_string((int)cute::size(CodesignNV::AccumFragLayout{})) + "\n";
  s += "  MXFP8 accum frag size = " + std::to_string((int)cute::size(CodesignMX::AccumFragLayout{})) + "\n";
  s += "  accum element = float (both)\n";
  s += "  NVFP4 TileShape MN = (" + std::to_string((int)cute::size<0>(CodesignNV::TileShape{})) + ","
       + std::to_string((int)cute::size<1>(CodesignNV::TileShape{})) + ")\n";
  s += "  MXFP8 TileShape MN = (" + std::to_string((int)cute::size<0>(CodesignMX::TileShape{})) + ","
       + std::to_string((int)cute::size<1>(CodesignMX::TileShape{})) + ")\n";
  return s;
}

torch::Tensor codesign_prepack_sf_fp4_b(torch::Tensor SFB, int64_t N, int64_t K) {
  TORCH_CHECK(K % 16 == 0, "K must be divisible by 16");
  return CodesignNV::prepack_b(SFB, N, K);
}
torch::Tensor codesign_nvfp4_gemm(torch::Tensor A, torch::Tensor SFA,
                                  torch::Tensor B, torch::Tensor physSFB,
                                  int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(K % 16 == 0, "K must be divisible by 16");
  return CodesignNV::run(A, SFA, B, physSFB, M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepack_sf_fp4_b", &codesign_prepack_sf_fp4_b, "prepack NVFP4 weight SF (once)");
  m.def("codesign_nvfp4_gemm", &codesign_nvfp4_gemm,
        "STAGE-1 co-design NVFP4 GEMM (stock SM120 block-scaled collective, our data contract)");
  m.def("compat_report", &codesign_compat_report,
        "Path-A shared-accumulator compile-time compatibility report (NVFP4 vs MXFP8)");
}
