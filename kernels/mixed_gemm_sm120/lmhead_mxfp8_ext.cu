// lmhead_mxfp8_ext.cu — LEVER 2 (lm_head on the kernel): pure MXFP8 (W8A8) GEMM with
// PRE-INTERLEAVED activation + weight SF, for quantizing the shared ~10 ms FP16 lm_head
// (vocab 128256 x hidden 2048). W4A4-mixed on the un-rotated lm_head input regresses PPL
// +0.295 (fails the gate); W8 MXFP8-only holds at +0.048 (14.898 <= 14.95), so lm_head is
// shipped as MXFP8.
//
// Mirrors presf_cfg_ext.cu's PresfMX (128x128x128 StageCountAutoCarveout, the measured
// pure-MXFP8 config) EXCEPT the SFA is passed ALREADY interleaved (produced by
// quant_pack_float_fp8, one pack launch) so there is NO per-call SF re-interleave kernel —
// the analogue of codesign_fused_ext.cu's fused_gemm_prepacked for the single-format path.
// The SFA/SFB interleave layout is BSConfig::tile_atom_to_shape_SF* at 128x128x128 MXFP8,
// byte-identical to what quant_pack_float_fp8 writes. GEMM math byte-identical to presf.
//
// NEW standalone extension; PROTECTED mainloop files untouched.

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

static inline void lmh_check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

template <class LayoutSF>
__global__ void lmh_fill_sf_kernel(uint8_t* __restrict__ phys, const uint8_t* __restrict__ logical,
                                   LayoutSF layout, int rows, int K, int VEC) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * K;
  if (idx >= total) return;
  int r = idx / K, k = idx % K, nblk = K / VEC;
  phys[layout(make_coord(r, k, 0))] = logical[r * nblk + (k / VEC)];
}

struct LmhMXGemm {
  using ElementAB  = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  static constexpr int Alignment = 16;
  static constexpr int VEC = 32;
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
      ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using StageArg = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutATag, Alignment, ElementB, LayoutBTag, Alignment,
      ElementAccumulator, ThreadBlockShape, ClusterShape, StageArg,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using BSConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  static torch::Tensor prepack_b(torch::Tensor SFB, int64_t N, int64_t K) {
    TORCH_CHECK(SFB.is_cuda() && SFB.scalar_type() == torch::kUInt8, "SFB CUDA uint8");
    SFB = SFB.contiguous(); auto dev = SFB.device();
    auto stream = at::cuda::getCurrentCUDAStream(dev.index());
    int Ni=(int)N, Ki=(int)K;
    auto layout = BSConfig::tile_atom_to_shape_SFB(make_shape(1, Ni, Ki, 1));
    int64_t sz = (int64_t)size(filter_zeros(layout));
    auto phys = torch::empty({sz}, torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    int total=Ni*Ki, tpb=256;
    lmh_fill_sf_kernel<<<(total+tpb-1)/tpb, tpb, 0, stream>>>(
        phys.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(), layout, Ni, Ki, VEC);
    lmh_check_cuda(cudaGetLastError(), "lmh prepack_b");
    lmh_check_cuda(cudaStreamSynchronize(stream), "lmh prepack_b sync");
    return phys;
  }

  // A bytes + PRE-INTERLEAVED physSFA (from quant_pack_float_fp8), weight bytes +
  // PRE-INTERLEAVED physSFB (from prepack_b). No per-call SF fill.
  static torch::Tensor run_prepacked(torch::Tensor A, torch::Tensor physSFA,
                                     torch::Tensor B, torch::Tensor physSFB,
                                     int64_t M, int64_t N, int64_t K) {
    A=A.contiguous(); B=B.contiguous(); physSFA=physSFA.contiguous(); physSFB=physSFB.contiguous();
    auto dev=A.device(); auto stream=at::cuda::getCurrentCUDAStream(dev.index());
    int Mi=(int)M, Ni=(int)N, Ki=(int)K;
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {Mi,Ki,1});
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {Ni,Ki,1});
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {Mi,Ni,1});
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {Mi,Ni,1});
    auto layout_SFA = BSConfig::tile_atom_to_shape_SFA(make_shape(Mi,Ni,Ki,1));
    auto layout_SFB = BSConfig::tile_atom_to_shape_SFB(make_shape(Mi,Ni,Ki,1));
    auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    auto D = torch::empty({M,N}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {Mi,Ni,Ki,1},
      { reinterpret_cast<typename ElementAB::DataType const*>(A.data_ptr<uint8_t>()), sA,
        reinterpret_cast<typename ElementAB::DataType const*>(B.data_ptr<uint8_t>()), sB,
        reinterpret_cast<typename ElementAB::ScaleFactorType const*>(physSFA.data_ptr<uint8_t>()), layout_SFA,
        reinterpret_cast<typename ElementAB::ScaleFactorType const*>(physSFB.data_ptr<uint8_t>()), layout_SFB },
      { {1.0f,0.0f}, reinterpret_cast<ElementC const*>(D.data_ptr()), sC,
        reinterpret_cast<ElementD*>(D.data_ptr()), sD } };
    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    auto wsb = torch::empty({(int64_t)std::max<size_t>(ws,1)}, u8);
    auto st = gemm.can_implement(args); TORCH_CHECK(st==cutlass::Status::kSuccess, "lmh can_implement: ", cutlassGetStatusString(st));
    st = gemm.initialize(args, wsb.data_ptr(), stream); TORCH_CHECK(st==cutlass::Status::kSuccess, "lmh initialize: ", cutlassGetStatusString(st));
    st = gemm.run(stream); TORCH_CHECK(st==cutlass::Status::kSuccess, "lmh run: ", cutlassGetStatusString(st));
    // Host-side sync only when NOT capturing a CUDA graph (the sync would abort
    // capture). Skipped during capture; replay carries no host sync. Mirrors
    // codesign_fused_ext.cu's run_with_cached_ws guard.
    cudaStreamCaptureStatus _cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &_cap) != cudaSuccess || _cap == cudaStreamCaptureStatusNone)
      lmh_check_cuda(cudaStreamSynchronize(stream), "lmh run sync");
    return D;
  }
};

torch::Tensor prepack_sf_fp8_b(torch::Tensor SFB, int64_t N, int64_t K) {
  TORCH_CHECK(K % 32 == 0, "K%32"); return LmhMXGemm::prepack_b(SFB, N, K);
}
torch::Tensor mxfp8_gemm_prepacked(torch::Tensor A, torch::Tensor physSFA,
                                   torch::Tensor B, torch::Tensor physSFB,
                                   int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(K % 32 == 0, "K%32"); return LmhMXGemm::run_prepacked(A, physSFA, B, physSFB, M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepack_sf_fp8_b", &prepack_sf_fp8_b, "prepack MXFP8 weight SF (load-time)");
  m.def("mxfp8_gemm_prepacked", &mxfp8_gemm_prepacked,
        "pure MXFP8 GEMM, activation + weight SF pre-interleaved (single-format prepacked)");
}
