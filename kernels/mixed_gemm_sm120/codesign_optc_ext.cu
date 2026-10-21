// codesign_fused_ext.cu — STAGE 2 of the co-design rewrite (b1-opt round-14/17).
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
// Shipping kernel b1_tma_ext.cu and PROTECTED mainloop headers are UNTOUCHED.
// (codesign_sm120_bs_mma.hpp / codesign_fused_kernel.hpp mainloop logic unchanged.)
//
// ===================================================================================
// round-17 wrapper-level fairness work (NO mainloop logic changes). Three overheads
// removed so the end-to-end A/B vs the shipping kernel is fair:
//
//  (1) WEIGHT-SF interleave -> load/pack time (already hoisted: prepack_sf_*_b is
//      called once by the caller; made efficient here too, free).
//  (2) ACTIVATION-SF interleave -> the per-call scatter is now ONE thread per logical
//      SF byte (rows*K/VEC) instead of rows*K. The stock kernel wrote the SAME physical
//      byte VEC(=16/32)x redundantly because the within-block (VEC) coordinate is a
//      stride-0 broadcast mode of the CUTLASS atom layout (filter_zeros collapses it,
//      hence the phys buffer is only rows*K/VEC bytes). The redundant kernel was the
//      16-44%-of-device-time cost round-16 identified. Bit-identical output (a layout
//      permutation, not a value change); the old redundant kernel is kept as
//      fused_gemm_slow / prepack_sf_*_a_slow purely to MEASURE the before->after cost.
//  (3) WORKSPACE -> cached/preallocated across calls; can_implement run once. initialize
//      still runs per call (TMA descriptors depend on the per-call pointers) but no
//      per-call device alloc/free of the workspace and no per-call can_implement.
//
// Entry points for the fair A/B:
//   fused_gemm(...)            : realistic per-call = efficient activation-SF interleave
//                                + cached-workspace dual GEMM (co-design's TRUE per-call cost)
//   fused_gemm_prepacked(...)  : GEMM-only ceiling (activation SF pre-interleaved by caller)
//   fused_gemm_slow(...)       : old redundant interleave (before number only)
//   prepack_sf_fp4_a / _fp8_a  : interleave activation SF out-of-line (efficient)
//   prepack_sf_fp4_a_slow / _fp8_a_slow : old redundant interleave (before number only)
// ===================================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <string>
#include <vector>

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
#include "codesign_optc_kernel.hpp"

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

// OLD (stock round-14) scatter: one thread per (row,k) element -> VEC-fold redundant
// writes to the same physical byte (the within-block coord is a stride-0 broadcast of
// the atom layout). Kept ONLY to measure the before->after interleave cost.
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

// NEW (round-17) scatter: one thread per LOGICAL SF byte (row, block). Each (row,block)
// maps to a unique physical offset (buffer size == rows*nblk), so this writes every
// physical byte exactly once -> bit-identical to the redundant kernel, VEC(=16/32)x
// fewer threads, coalesced logical reads. This is the co-design's minimal honest
// per-call activation-SF interleave cost.
template <class LayoutSF>
__global__ void cd_fill_sf_block_kernel(uint8_t* __restrict__ phys,
                                        const uint8_t* __restrict__ logical,
                                        LayoutSF layout, int rows, int nblk, int VEC) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * nblk;
  if (idx >= total) return;
  int r = idx / nblk;
  int blk = idx % nblk;
  int off = layout(make_coord(r, blk * VEC, 0));
  phys[off] = logical[r * nblk + blk];
}

// ===================================================================================
// round-18: FUSED activation quant/pack step. The REAL inference pipeline quantizes the
// activation per call -- the quantize/pack step already touches every element and emits
// the block SF. This kernel models that step: ONE pass over (row, block) that (a) packs
// the data (fp4: 2 codes/byte; fp8: byte copy) AND (b) stores the block SF to EITHER the
// natural contiguous layout (ILV=false) OR the CUTLASS interleaved atom layout (ILV=true).
// The data-pack path is byte-identical for both SF targets; the ONLY difference between
// the two variants is the SF store address (one contiguous write vs one scattered write).
// Measuring NAT vs ILV therefore isolates the true marginal cost of emitting interleaved
// SF inside the step that already runs per call -> answers "is folding the interleave
// into quant/pack free?". When ILV output feeds fused_gemm_prepacked there is NO separate
// per-call interleave launch: the co-design reaches its GEMM-only ceiling.
// Bit-exact by construction: the SF VALUES are unchanged (a layout permutation); the ILV
// output is torch.equal to the standalone interleave of the same natural SF (r17-proven).
template <class LayoutSF, int VEC, bool PACK, bool ILV>
__global__ void cd_quant_pack_kernel(uint8_t* __restrict__ data_out,
                                     uint8_t* __restrict__ sf_out,
                                     const uint8_t* __restrict__ data_in,
                                     const uint8_t* __restrict__ sf_in,
                                     LayoutSF layout, int rows, int nblk, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * nblk;
  if (idx >= total) return;
  int r = idx / nblk;
  int b = idx % nblk;
  // --- SF store: the ONLY NAT/ILV difference (same value, different address) ---
  uint8_t sfv = sf_in[r * nblk + b];
  if (ILV) {
    int off = layout(make_coord(r, b * VEC, 0));
    sf_out[off] = sfv;
  } else {
    sf_out[r * nblk + b] = sfv;
  }
  // --- data pack/copy: byte-identical for both variants ---
  int base = r * K + b * VEC;
  if (PACK) {                      // fp4: VEC codes -> VEC/2 packed bytes (even k = low nibble)
    int obase = r * (K / 2) + b * (VEC / 2);
#pragma unroll
    for (int j = 0; j < VEC / 2; ++j) {
      uint8_t lo = data_in[base + 2 * j] & 0x0F;
      uint8_t hi = data_in[base + 2 * j + 1] & 0x0F;
      data_out[obase + j] = lo | (hi << 4);
    }
  } else {                         // fp8: copy VEC bytes
#pragma unroll
    for (int j = 0; j < VEC; ++j) data_out[base + j] = data_in[base + j];
  }
}

// round-21: VECTORIZED quant/pack. Same (row,block) grid, same SF store, but the data
// path uses 128-bit aligned loads/stores instead of scalar byte traffic. One thread owns
// one block (VEC contiguous input bytes starting at a VEC-aligned offset), so within a
// warp the 32 threads touch 32*VEC contiguous bytes -> fully coalesced 128-bit sectors.
//   fp4 (VEC=16, PACK): ONE 128-bit load -> register nibble-pack -> ONE 64-bit store.
//   fp8 (VEC=32, copy) : TWO 128-bit loads -> TWO 128-bit stores.
// Register-only packing (no local-memory byte array) keeps the pack bit-identical to the
// scalar kernel above (see cd_pack4nib below). SF store path is unchanged.
__device__ __forceinline__ uint32_t cd_pack4nib(uint32_t w) {
  // 4 input code bytes in w -> 2 packed bytes (16 bits); even k = low nibble.
  // out[0] = (in0 & 0xF) | ((in1 & 0xF) << 4); out[1] = (in2 & 0xF) | ((in3 & 0xF) << 4)
  uint32_t p0 = (w & 0x0Fu) | ((w >> 4) & 0xF0u);            // bytes 0,1
  uint32_t p1 = ((w >> 16) & 0x0Fu) | ((w >> 20) & 0xF0u);   // bytes 2,3
  return p0 | (p1 << 8);
}

template <class LayoutSF, int VEC, bool PACK, bool ILV>
__global__ void cd_quant_pack_kernel_vec(uint8_t* __restrict__ data_out,
                                         uint8_t* __restrict__ sf_out,
                                         const uint8_t* __restrict__ data_in,
                                         const uint8_t* __restrict__ sf_in,
                                         LayoutSF layout, int rows, int nblk, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * nblk;
  if (idx >= total) return;
  int r = idx / nblk;
  int b = idx % nblk;
  // --- SF store: the ONLY NAT/ILV difference (same value, different address) ---
  uint8_t sfv = sf_in[r * nblk + b];
  if (ILV) {
    int off = layout(make_coord(r, b * VEC, 0));
    sf_out[off] = sfv;
  } else {
    sf_out[r * nblk + b] = sfv;
  }
  // --- data pack/copy: vectorized 128-bit, bit-identical to scalar kernel ---
  int base = r * K + b * VEC;
  if (PACK) {                      // fp4 VEC=16: one 128-bit load -> one 64-bit store
    const uint4 v = *reinterpret_cast<const uint4*>(data_in + base);
    uint64_t out = (uint64_t)cd_pack4nib(v.x)
                 | ((uint64_t)cd_pack4nib(v.y) << 16)
                 | ((uint64_t)cd_pack4nib(v.z) << 32)
                 | ((uint64_t)cd_pack4nib(v.w) << 48);
    int obase = r * (K / 2) + b * (VEC / 2);
    *reinterpret_cast<uint64_t*>(data_out + obase) = out;
  } else {                         // fp8 VEC=32: two 128-bit copies
#pragma unroll
    for (int j = 0; j < VEC; j += 16) {
      uint4 v = *reinterpret_cast<const uint4*>(data_in + base + j);
      *reinterpret_cast<uint4*>(data_out + base + j) = v;
    }
  }
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
using ForkMX = ForkMma<StockMX, /*ClearAccum=*/false>;   // Phase 2 accumulates (baseline, shared accum)
using ForkMXclear = ForkMma<StockMX, /*ClearAccum=*/true>;  // Option-C: clears its OWN accMX2

template <bool DualPhase>
using FusedKernel = codesign::CodesignOptCKernel<
    Shape<int,int,int,int>, ForkNV, ForkMX, Epilogue,
    cutlass::gemm::StaticPersistentScheduler, DualPhase, /*TwoAccum=*/false>;

// Option-C kernel: two separate FP32 accumulators (NV shared-frag + MX accMX2),
// summed before the epilogue. MX collective clears its own accumulator.
using OptCKernel = codesign::CodesignOptCKernel<
    Shape<int,int,int,int>, ForkNV, ForkMXclear, Epilogue,
    cutlass::gemm::StaticPersistentScheduler, /*DualPhase=*/true, /*TwoAccum=*/true>;

using GemmDual   = cutlass::gemm::device::GemmUniversalAdapter<FusedKernel<true>>;
using GemmSingle = cutlass::gemm::device::GemmUniversalAdapter<FusedKernel<false>>;
using GemmOptC   = cutlass::gemm::device::GemmUniversalAdapter<OptCKernel>;

using BSConfigNV = typename StockNV::Sm1xxBlkScaledConfig;
using BSConfigMX = typename StockMX::Sm1xxBlkScaledConfig;
using StrideA = typename FusedKernel<true>::StrideA;
using StrideB = typename FusedKernel<true>::StrideB;
using StrideC = typename FusedKernel<true>::StrideC;
using StrideD = typename FusedKernel<true>::StrideD;

static inline torch::TensorOptions u8opts(torch::Device dev) {
  return torch::TensorOptions().dtype(torch::kUInt8).device(dev);
}

// ---- generic SF interleave (efficient by default; EFFICIENT=false -> old redundant) ----
// SFB path: tile_atom_to_shape_SFB(make_shape(1,N,K,1)), rows=N.
template <class BSConfig, int VEC, bool EFFICIENT>
static torch::Tensor interleave_sfb(torch::Tensor SFB, int Ni, int Ki, cudaStream_t stream) {
  TORCH_CHECK(SFB.is_cuda() && SFB.scalar_type() == torch::kUInt8, "SFB must be CUDA uint8");
  SFB = SFB.contiguous();
  auto dev = SFB.device();
  auto layout_SFB = BSConfig::tile_atom_to_shape_SFB(make_shape(1, Ni, Ki, 1));
  int64_t sz = (int64_t)size(filter_zeros(layout_SFB));
  auto phys = torch::empty({sz}, u8opts(dev));
  int nblk = Ki / VEC, tpb = 256;
  if (EFFICIENT) {
    int total = Ni * nblk;
    cd_fill_sf_block_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(), layout_SFB, Ni, nblk, VEC);
  } else {
    int total = Ni * Ki;
    cd_fill_sf_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(), layout_SFB, Ni, Ki, VEC);
  }
  check_cuda(cudaGetLastError(), "interleave_sfb launch");
  return phys;
}

// SFA path: tile_atom_to_shape_SFA(make_shape(M,N,K,1)), rows=M.
template <class BSConfig, int VEC, bool EFFICIENT>
static torch::Tensor interleave_sfa(torch::Tensor SFA, int Mi, int Ni, int Ki, cudaStream_t stream) {
  SFA = SFA.contiguous();
  auto dev = SFA.device();
  auto layout_SFA = BSConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Ki, 1));
  int64_t sz = (int64_t)size(filter_zeros(layout_SFA));
  auto phys = torch::empty({sz}, u8opts(dev));
  int nblk = Ki / VEC, tpb = 256;
  if (EFFICIENT) {
    int total = Mi * nblk;
    cd_fill_sf_block_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys.data_ptr<uint8_t>(), SFA.data_ptr<uint8_t>(), layout_SFA, Mi, nblk, VEC);
  } else {
    int total = Mi * Ki;
    cd_fill_sf_kernel<<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys.data_ptr<uint8_t>(), SFA.data_ptr<uint8_t>(), layout_SFA, Mi, Ki, VEC);
  }
  check_cuda(cudaGetLastError(), "interleave_sfa launch");
  return phys;
}

// ---- load-time weight-SF interleave (hoisted; efficient) ----
torch::Tensor cd_prepack_sf_fp4_b(torch::Tensor SFB, int64_t N, int64_t K) {
  TORCH_CHECK(K % 16 == 0, "K%16");
  auto stream = at::cuda::getCurrentCUDAStream(SFB.device().index());
  auto phys = interleave_sfb<BSConfigNV, 16, true>(SFB, (int)N, (int)K, stream);
  check_cuda(cudaStreamSynchronize(stream), "prepack_b sync");
  return phys;
}
torch::Tensor cd_prepack_sf_fp8_b(torch::Tensor SFB, int64_t N, int64_t K) {
  TORCH_CHECK(K % 32 == 0, "K%32");
  auto stream = at::cuda::getCurrentCUDAStream(SFB.device().index());
  auto phys = interleave_sfb<BSConfigMX, 32, true>(SFB, (int)N, (int)K, stream);
  check_cuda(cudaStreamSynchronize(stream), "prepack_b sync");
  return phys;
}

// ---- out-of-line activation-SF interleave (efficient + slow variants) ----
torch::Tensor cd_prepack_sf_fp4_a(torch::Tensor SFA, int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(K % 16 == 0, "K%16");
  auto stream = at::cuda::getCurrentCUDAStream(SFA.device().index());
  auto phys = interleave_sfa<BSConfigNV, 16, true>(SFA, (int)M, (int)N, (int)K, stream);
  check_cuda(cudaStreamSynchronize(stream), "prepack_a sync");
  return phys;
}
torch::Tensor cd_prepack_sf_fp8_a(torch::Tensor SFA, int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(K % 32 == 0, "K%32");
  auto stream = at::cuda::getCurrentCUDAStream(SFA.device().index());
  auto phys = interleave_sfa<BSConfigMX, 32, true>(SFA, (int)M, (int)N, (int)K, stream);
  check_cuda(cudaStreamSynchronize(stream), "prepack_a sync");
  return phys;
}
torch::Tensor cd_prepack_sf_fp4_a_slow(torch::Tensor SFA, int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(K % 16 == 0, "K%16");
  auto stream = at::cuda::getCurrentCUDAStream(SFA.device().index());
  auto phys = interleave_sfa<BSConfigNV, 16, false>(SFA, (int)M, (int)N, (int)K, stream);
  check_cuda(cudaStreamSynchronize(stream), "prepack_a_slow sync");
  return phys;
}
torch::Tensor cd_prepack_sf_fp8_a_slow(torch::Tensor SFA, int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(K % 32 == 0, "K%32");
  auto stream = at::cuda::getCurrentCUDAStream(SFA.device().index());
  auto phys = interleave_sfa<BSConfigMX, 32, false>(SFA, (int)M, (int)N, (int)K, stream);
  check_cuda(cudaStreamSynchronize(stream), "prepack_a_slow sync");
  return phys;
}

// ---- round-18: fused quant/pack step (data pack + SF layout in one pass) ----
// Returns {packed_data, SF}. ilv=true emits interleaved atom-layout SF (folds the r17
// standalone interleave into the pack step -> GEMM-only ceiling reachable per call with
// NO separate interleave launch). ilv=false emits natural contiguous SF (what the
// shipping kernel consumes natively). Data pack is identical for both.
template <class BSConfig, int VEC, bool PACK>
static std::vector<torch::Tensor> quant_pack_impl(torch::Tensor data_in, torch::Tensor sf_in,
                                                  int Mi, int Ni, int Ki, bool ilv, int impl,
                                                  cudaStream_t stream) {
  TORCH_CHECK(data_in.is_cuda() && data_in.scalar_type() == torch::kUInt8, "data_in CUDA uint8");
  TORCH_CHECK(sf_in.is_cuda() && sf_in.scalar_type() == torch::kUInt8, "sf_in CUDA uint8");
  data_in = data_in.contiguous(); sf_in = sf_in.contiguous();
  auto dev = data_in.device();
  int nblk = Ki / VEC;
  int64_t dsz = PACK ? (int64_t)Mi * (Ki / 2) : (int64_t)Mi * Ki;
  auto data_out = torch::empty({dsz}, u8opts(dev));
  auto layout_SFA = BSConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Ki, 1));
  int64_t sfsz = ilv ? (int64_t)size(filter_zeros(layout_SFA)) : (int64_t)Mi * nblk;
  auto sf_out = torch::empty({sfsz}, u8opts(dev));
  int total = Mi * nblk, tpb = 256;
  dim3 grid((total + tpb - 1) / tpb);
  using L = decltype(layout_SFA);
  uint8_t* dout = data_out.data_ptr<uint8_t>();
  uint8_t* sout = sf_out.data_ptr<uint8_t>();
  const uint8_t* din = data_in.data_ptr<uint8_t>();
  const uint8_t* sin = sf_in.data_ptr<uint8_t>();
  if (impl == 0) {                 // scalar (round-18 reference; before-number only)
    if (ilv)
      cd_quant_pack_kernel<L, VEC, PACK, true><<<grid, tpb, 0, stream>>>(dout, sout, din, sin, layout_SFA, Mi, nblk, Ki);
    else
      cd_quant_pack_kernel<L, VEC, PACK, false><<<grid, tpb, 0, stream>>>(dout, sout, din, sin, layout_SFA, Mi, nblk, Ki);
  } else {                         // vectorized (round-21 default)
    if (ilv)
      cd_quant_pack_kernel_vec<L, VEC, PACK, true><<<grid, tpb, 0, stream>>>(dout, sout, din, sin, layout_SFA, Mi, nblk, Ki);
    else
      cd_quant_pack_kernel_vec<L, VEC, PACK, false><<<grid, tpb, 0, stream>>>(dout, sout, din, sin, layout_SFA, Mi, nblk, Ki);
  }
  check_cuda(cudaGetLastError(), "quant_pack launch");
  data_out = data_out.view({Mi, PACK ? Ki / 2 : Ki});
  return {data_out, sf_out};
}

std::vector<torch::Tensor> cd_quant_pack_fp4(torch::Tensor codes, torch::Tensor SFA,
                                             int64_t M, int64_t N, int64_t K, bool ilv, int64_t impl) {
  TORCH_CHECK(K % 16 == 0, "K%16");
  auto stream = at::cuda::getCurrentCUDAStream(codes.device().index());
  return quant_pack_impl<BSConfigNV, 16, true>(codes, SFA, (int)M, (int)N, (int)K, ilv, (int)impl, stream);
}
std::vector<torch::Tensor> cd_quant_pack_fp8(torch::Tensor bytes, torch::Tensor SFA,
                                             int64_t M, int64_t N, int64_t K, bool ilv, int64_t impl) {
  TORCH_CHECK(K % 32 == 0, "K%32");
  auto stream = at::cuda::getCurrentCUDAStream(bytes.device().index());
  return quant_pack_impl<BSConfigMX, 32, false>(bytes, SFA, (int)M, (int)N, (int)K, ilv, (int)impl, stream);
}

// Build a collective's mainloop Arguments from pre-interleaved SF (phys_SFA/phys_SFB).
template <class Coll, class BSConfig>
static typename Coll::Arguments make_mainloop_args_pre(torch::Tensor A, torch::Tensor physSFA,
                                torch::Tensor B, torch::Tensor physSFB,
                                int Mi, int Ni, int Ki) {
  auto layout_SFA = BSConfig::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Ki, 1));
  auto layout_SFB = BSConfig::tile_atom_to_shape_SFB(make_shape(Mi, Ni, Ki, 1));
  using ElemA = typename Coll::ElementA;      // packed data type (fp4/fp8)
  using ElemSF = typename Coll::ElementSF;
  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {Mi, Ki, 1});
  StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {Ni, Ki, 1});
  typename Coll::Arguments m;
  m.ptr_A = reinterpret_cast<ElemA const*>(A.data_ptr<uint8_t>());
  m.dA = sA;
  m.ptr_B = reinterpret_cast<ElemA const*>(B.data_ptr<uint8_t>());
  m.dB = sB;
  m.ptr_SFA = reinterpret_cast<ElemSF const*>(physSFA.data_ptr<uint8_t>());
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

// ---- cached workspace helpers (overhead #3) ----
// Persistent per-adapter workspace; grows only. can_implement run once per adapter.
template <class Gemm>
static void run_with_cached_ws(Gemm& gemm, typename Gemm::Arguments& arguments,
                               torch::Device dev, cudaStream_t stream,
                               torch::Tensor& ws_cache, bool& checked,
                               const char* tag) {
  size_t ws = Gemm::get_workspace_size(arguments);
  if (!ws_cache.defined() || (size_t)ws_cache.numel() < std::max<size_t>(ws, 1)) {
    ws_cache = torch::empty({(int64_t)std::max<size_t>(ws, 1)}, u8opts(dev));
  }
  if (!checked) {
    auto st = gemm.can_implement(arguments);
    TORCH_CHECK(st == cutlass::Status::kSuccess, tag, " can_implement: ", cutlassGetStatusString(st));
    checked = true;
  }
  auto st = gemm.initialize(arguments, ws_cache.data_ptr(), stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, tag, " initialize: ", cutlassGetStatusString(st));
  st = gemm.run(stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, tag, " run: ", cutlassGetStatusString(st));
  check_cuda(cudaStreamSynchronize(stream), "run sync");
}

// ---- dual GEMM core: takes PRE-INTERLEAVED activation+weight SF, GEMM only ----
template <class Gemm, class MXFork>
static torch::Tensor run_dual_pre_t(torch::Tensor A_lo, torch::Tensor physSFA_lo, torch::Tensor B_lo, torch::Tensor physSFB_lo,
                                  torch::Tensor A_hi, torch::Tensor physSFA_hi, torch::Tensor B_hi, torch::Tensor physSFB_hi,
                                  int Mi, int Ni, int Klo, int Khi, cudaStream_t stream, torch::Device dev,
                                  const char* tag) {
  auto D = torch::empty({Mi, Ni}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
  auto nv_args = make_mainloop_args_pre<ForkNV, BSConfigNV>(A_lo, physSFA_lo, B_lo, physSFB_lo, Mi, Ni, Klo);
  auto mx_args = make_mainloop_args_pre<MXFork, BSConfigMX>(A_hi, physSFA_hi, B_hi, physSFB_hi, Mi, Ni, Khi);

  typename Gemm::Arguments arguments;
  arguments.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  arguments.problem_shape = {Mi, Ni, Klo, 1};
  arguments.mainloop_nv = nv_args;
  arguments.mainloop_mx = mx_args;
  arguments.K_high = Khi;
  arguments.epilogue = make_epi_args(D, Mi, Ni);

  static Gemm gemm;
  static torch::Tensor ws_cache;
  static bool checked = false;
  run_with_cached_ws(gemm, arguments, dev, stream, ws_cache, checked, tag);
  return D;
}

static torch::Tensor run_dual_pre(torch::Tensor A_lo, torch::Tensor physSFA_lo, torch::Tensor B_lo, torch::Tensor physSFB_lo,
                                  torch::Tensor A_hi, torch::Tensor physSFA_hi, torch::Tensor B_hi, torch::Tensor physSFB_hi,
                                  int Mi, int Ni, int Klo, int Khi, cudaStream_t stream, torch::Device dev) {
  return run_dual_pre_t<GemmDual, ForkMX>(A_lo, physSFA_lo, B_lo, physSFB_lo, A_hi, physSFA_hi, B_hi, physSFB_hi,
                                          Mi, Ni, Klo, Khi, stream, dev, "fused");
}

// ---- Option-C: two-accumulator dual GEMM (GEMM-only, pre-interleaved SF) ----
torch::Tensor cd_fused_gemm_prepacked_optc(torch::Tensor A_lo, torch::Tensor physSFA_lo, torch::Tensor B_lo, torch::Tensor physSFB_lo,
                                           torch::Tensor A_hi, torch::Tensor physSFA_hi, torch::Tensor B_hi, torch::Tensor physSFB_hi,
                                           int64_t M, int64_t N, int64_t K_lo, int64_t K_hi) {
  TORCH_CHECK(K_lo % 16 == 0 && K_hi % 32 == 0, "K_lo%16, K_hi%32");
  A_lo=A_lo.contiguous(); B_lo=B_lo.contiguous(); A_hi=A_hi.contiguous(); B_hi=B_hi.contiguous();
  physSFA_lo=physSFA_lo.contiguous(); physSFB_lo=physSFB_lo.contiguous();
  physSFA_hi=physSFA_hi.contiguous(); physSFB_hi=physSFB_hi.contiguous();
  auto dev = A_lo.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  return run_dual_pre_t<GemmOptC, ForkMXclear>(A_lo, physSFA_lo, B_lo, physSFB_lo, A_hi, physSFA_hi, B_hi, physSFB_hi,
                                               (int)M, (int)N, (int)K_lo, (int)K_hi, stream, dev, "optc");
}

// ---- GEMM-only ceiling: caller supplies already-interleaved activation SF ----
torch::Tensor cd_fused_gemm_prepacked(torch::Tensor A_lo, torch::Tensor physSFA_lo, torch::Tensor B_lo, torch::Tensor physSFB_lo,
                                      torch::Tensor A_hi, torch::Tensor physSFA_hi, torch::Tensor B_hi, torch::Tensor physSFB_hi,
                                      int64_t M, int64_t N, int64_t K_lo, int64_t K_hi) {
  TORCH_CHECK(K_lo % 16 == 0 && K_hi % 32 == 0, "K_lo%16, K_hi%32");
  A_lo=A_lo.contiguous(); B_lo=B_lo.contiguous(); A_hi=A_hi.contiguous(); B_hi=B_hi.contiguous();
  physSFA_lo=physSFA_lo.contiguous(); physSFB_lo=physSFB_lo.contiguous();
  physSFA_hi=physSFA_hi.contiguous(); physSFB_hi=physSFB_hi.contiguous();
  auto dev = A_lo.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  return run_dual_pre(A_lo, physSFA_lo, B_lo, physSFB_lo, A_hi, physSFA_hi, B_hi, physSFB_hi,
                      (int)M, (int)N, (int)K_lo, (int)K_hi, stream, dev);
}

// ---- realistic per-call: efficient internal activation-SF interleave + cached dual GEMM ----
template <bool EFFICIENT>
static torch::Tensor fused_gemm_impl(torch::Tensor A_lo, torch::Tensor SFA_lo, torch::Tensor B_lo, torch::Tensor physSFB_lo,
                                     torch::Tensor A_hi, torch::Tensor SFA_hi, torch::Tensor B_hi, torch::Tensor physSFB_hi,
                                     int64_t M, int64_t N, int64_t K_lo, int64_t K_hi) {
  TORCH_CHECK(K_lo % 16 == 0 && K_hi % 32 == 0, "K_lo%16, K_hi%32");
  A_lo=A_lo.contiguous(); B_lo=B_lo.contiguous(); SFA_lo=SFA_lo.contiguous(); physSFB_lo=physSFB_lo.contiguous();
  A_hi=A_hi.contiguous(); B_hi=B_hi.contiguous(); SFA_hi=SFA_hi.contiguous(); physSFB_hi=physSFB_hi.contiguous();
  auto dev = A_lo.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  int Mi=(int)M, Ni=(int)N, Klo=(int)K_lo, Khi=(int)K_hi;
  // per-call activation-SF interleave (no intermediate sync; same stream as GEMM)
  auto physSFA_lo = interleave_sfa<BSConfigNV, 16, EFFICIENT>(SFA_lo, Mi, Ni, Klo, stream);
  auto physSFA_hi = interleave_sfa<BSConfigMX, 32, EFFICIENT>(SFA_hi, Mi, Ni, Khi, stream);
  return run_dual_pre(A_lo, physSFA_lo, B_lo, physSFB_lo, A_hi, physSFA_hi, B_hi, physSFB_hi,
                      Mi, Ni, Klo, Khi, stream, dev);
}

torch::Tensor cd_fused_gemm(torch::Tensor A_lo, torch::Tensor SFA_lo, torch::Tensor B_lo, torch::Tensor physSFB_lo,
                            torch::Tensor A_hi, torch::Tensor SFA_hi, torch::Tensor B_hi, torch::Tensor physSFB_hi,
                            int64_t M, int64_t N, int64_t K_lo, int64_t K_hi) {
  return fused_gemm_impl<true>(A_lo, SFA_lo, B_lo, physSFB_lo, A_hi, SFA_hi, B_hi, physSFB_hi, M, N, K_lo, K_hi);
}
torch::Tensor cd_fused_gemm_slow(torch::Tensor A_lo, torch::Tensor SFA_lo, torch::Tensor B_lo, torch::Tensor physSFB_lo,
                                 torch::Tensor A_hi, torch::Tensor SFA_hi, torch::Tensor B_hi, torch::Tensor physSFB_hi,
                                 int64_t M, int64_t N, int64_t K_lo, int64_t K_hi) {
  return fused_gemm_impl<false>(A_lo, SFA_lo, B_lo, physSFB_lo, A_hi, SFA_hi, B_hi, physSFB_hi, M, N, K_lo, K_hi);
}

// ---- Single-phase NVFP4 through the FORKED glue (Milestone-A gate) ----
torch::Tensor cd_single_nvfp4_gemm(torch::Tensor A, torch::Tensor SFA,
                                   torch::Tensor B, torch::Tensor physSFB,
                                   int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(K % 16 == 0, "K%16");
  A = A.contiguous(); B = B.contiguous(); SFA = SFA.contiguous(); physSFB = physSFB.contiguous();
  auto dev = A.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  int Mi=(int)M, Ni=(int)N, Ki=(int)K;
  auto D = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
  auto physSFA = interleave_sfa<BSConfigNV, 16, true>(SFA, Mi, Ni, Ki, stream);

  auto nv_args = make_mainloop_args_pre<ForkNV, BSConfigNV>(A, physSFA, B, physSFB, Mi, Ni, Ki);

  typename GemmSingle::Arguments arguments;
  arguments.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  arguments.problem_shape = {Mi, Ni, Ki, 1};
  arguments.mainloop_nv = nv_args;
  arguments.K_high = 0;
  arguments.epilogue = make_epi_args(D, Mi, Ni);

  static GemmSingle gemm;
  static torch::Tensor ws_cache;
  static bool checked = false;
  run_with_cached_ws(gemm, arguments, dev, stream, ws_cache, checked, "single");
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
  s += "  SharedStorageSize(optc2acc) = " + std::to_string((int)OptCKernel::SharedStorageSize) + " bytes\n";
  // Concurrent Option-C needs BOTH formats' tensor smem resident (de-unioned):
  {
    int tnv = (int)sizeof(typename ForkNV::TensorStorage);
    int tmx = (int)sizeof(typename ForkMXclear::TensorStorage);
    s += "  tensor_smem NV=" + std::to_string(tnv) + " MX=" + std::to_string(tmx) +
         "  union(serial)=" + std::to_string(std::max(tnv, tmx)) +
         "  deunion(concurrent)=" + std::to_string(tnv + tmx) + "  SM120_cap=101376\n";
  }
  s += "  MaxThreadsPerBlock = " + std::to_string((int)FusedKernel<true>::MaxThreadsPerBlock) + "\n";
  s += "  accum NV=" + std::string(std::is_same<AccNV,float>::value ? "float" : "?") +
       " MX=" + std::string(std::is_same<AccMX,float>::value ? "float" : "?") + "\n";
  return s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepack_sf_fp4_b", &cd_prepack_sf_fp4_b, "prepack NVFP4 weight SF (load-time, efficient)");
  m.def("prepack_sf_fp8_b", &cd_prepack_sf_fp8_b, "prepack MXFP8 weight SF (load-time, efficient)");
  m.def("prepack_sf_fp4_a", &cd_prepack_sf_fp4_a, "interleave NVFP4 activation SF (efficient)");
  m.def("prepack_sf_fp8_a", &cd_prepack_sf_fp8_a, "interleave MXFP8 activation SF (efficient)");
  m.def("prepack_sf_fp4_a_slow", &cd_prepack_sf_fp4_a_slow, "interleave NVFP4 activation SF (old redundant; before-number only)");
  m.def("prepack_sf_fp8_a_slow", &cd_prepack_sf_fp8_a_slow, "interleave MXFP8 activation SF (old redundant; before-number only)");
  m.def("quant_pack_fp4", &cd_quant_pack_fp4, "round-18/21 fused pack: NVFP4 data pack + SF layout (ilv=interleaved atom / natural; impl=1 vectorized default, 0 scalar-ref); returns {packed, SF}",
        pybind11::arg("codes"), pybind11::arg("SFA"), pybind11::arg("M"), pybind11::arg("N"), pybind11::arg("K"), pybind11::arg("ilv"), pybind11::arg("impl") = 1);
  m.def("quant_pack_fp8", &cd_quant_pack_fp8, "round-18/21 fused pack: MXFP8 byte copy + SF layout (ilv=interleaved atom / natural; impl=1 vectorized default, 0 scalar-ref); returns {bytes, SF}",
        pybind11::arg("bytes"), pybind11::arg("SFA"), pybind11::arg("M"), pybind11::arg("N"), pybind11::arg("K"), pybind11::arg("ilv"), pybind11::arg("impl") = 1);
  m.def("single_nvfp4_gemm", &cd_single_nvfp4_gemm, "single-format NVFP4 through forked glue (gate)");
  m.def("fused_gemm", &cd_fused_gemm, "dual-format fused GEMM; efficient per-call activation-SF interleave + cached workspace");
  m.def("fused_gemm_slow", &cd_fused_gemm_slow, "dual-format fused GEMM with OLD redundant interleave (before-number only)");
  m.def("fused_gemm_prepacked", &cd_fused_gemm_prepacked, "dual-format fused GEMM, activation SF pre-interleaved (GEMM-only ceiling)");
  m.def("fused_gemm_prepacked_optc", &cd_fused_gemm_prepacked_optc, "Option-C two-accumulator dual-format fused GEMM (register probe), activation SF pre-interleaved");
  m.def("compat_report", &cd_compat_report, "build/smem report");
}
