// B1 — TRUE single-CUDA-kernel fused mixed-precision GEMM for SM120 (RTX Pro 6000 Blackwell).
//
//   D[m,n] = ( sum_{k<K_low}  A_fp4[m,k]*sfa4*B_fp4[n,k]*sfb4
//            + sum_{k<K_high} A_fp8[m,k]*sfa8*B_fp8[n,k]*sfb8 ) * s_x[m] * s_w[n]
//
// Phase 1: NVFP4 block-scaled MMA (mxf4nvf4, m16n8k64, SFVecSize=16) over K_low.
// Phase 2: MXFP8 block-scaled MMA (f8f6f4,  m16n8k32, SFVecSize=32) over K_high.
// Both phases accumulate into the SAME FP32 register accumulator (CLayout identical).
//
// Optimization round 1 (b1_mw -> b1_mw_pipe -> b1_opt): multi-warp CTA tiles via
// make_tiled_mma(Atom, Layout<Shape<WM,WN,1>>), 16B-vectorized gmem->smem staging
// with cp.async (ZFILL for OOB rows), a STAGES-deep smem pipeline (STAGES=1
// degenerates to synchronous staging), and an optional KMUL=2 (two MMA-K atoms
// per staged chunk). Config menu (cfg):
//   1: 16x64, 4 warps   (decode, large N)      2: 16x32, 2 warps (decode, N<=4096)
//   3: 32x64, 4 warps   (mid-M / small-N prefill grid filler)
//   4: 128x128, 8 warps (prefill)              5: cfg4 + KMUL=2  (big-N prefill)
//   6: 64x128, 8 warps + KMUL=2 (occupancy probe)
//   10: warp-specialized 128x128 + KMUL=2 (8 MMA warps + 2 cp.async producer
//       warps, mbarrier stage hand-off, 3/4-deep; prefill best). Probed and
//       rejected (see candidates.jsonl): KMUL=4/3 chunks, 128x256 fat tile,
//       16-consumer-warp WS — all slower than cfg5/cfg10.
// Auto dispatch (cfg=0, stages=0) picks the measured-best per (M, N); see host fn.
// SMEM buffers are reused across the two phases (barrier separated). Epilogue is
// partition_C + identity-tensor predication.
//
// CORRECTNESS INVARIANT (root cause of the original cosine~0 bug, do not regress):
// MMA_Traits ValTypeA/B = uint4_t (NVFP4) / uint8_t (MXFP8), so smem views MUST use
// those UINT types — cute::copy between float_e2m1_t and uint4_t does a VALUE
// conversion (float->uint truncation, negatives clamped to 0). See docs/b1_report.md.
//
// Exposes b1_nvfp4_only (Step-1 isolation entry, naive 1-warp kernel, fp32 out) and
// b1_fused(..., stages) (full dual-phase kernel, s_x*s_w scaling, fp16 out).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cutlass/float8.h"

using namespace cute;

// ============================================================================
// SF-partition helpers — copied verbatim from CUTLASS
// gemm/collective/sm120_blockscaled_mma_tma.hpp (methods made free functions).
// ============================================================================
template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto
b1_thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});
  using AtomShape_MNK  = typename Atom::Shape_MNK;
  using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;
  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto t_tile = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfatensor, t_tile);
  auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                          make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);
  auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{},_);
  auto thr_tile = make_tile(_,
                            make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                      make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
  return thr_tensor;
}

template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto
b1_thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});
  using AtomShape_MNK  = typename Atom::Shape_MNK;
  using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;
  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto t_tile = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfbtensor, t_tile);
  auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                          make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);
  auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{},_);
  auto thr_tile = make_tile(_,
                            make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                      make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
  return thr_tensor;
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
b1_partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(),
                                b1_thrfrg_SFA(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  auto partition_SFA = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(partition_SFA);
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
b1_partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(),
                                b1_thrfrg_SFB(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  auto partition_SFB = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(partition_SFB);
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto
b1_get_layoutSFA_TV(TiledMma& mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto atile = make_tile(_,
                make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                      make_stride(               Int<1>{} ,                Int<0>{} )),
                          _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return b1_thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto
b1_get_layoutSFB_TV(TiledMma& mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto btile = make_tile(_,
                make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                      make_stride(               Int<0>{} ,                Int<1>{} )),
                          _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return b1_thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

// ============================================================================
// Atom types
// ============================================================================
using AtomNV = SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t, 16>;
using AtomMX = SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, cutlass::float_ue8m0_t, 32>;

// Shared accumulator across phases is only valid if both atoms partition C
// identically (FP32[4]/thread, SM80_16x8_Row).
static_assert(cute::is_same_v<typename cute::MMA_Traits<AtomNV>::CLayout,
                              typename cute::MMA_Traits<AtomMX>::CLayout>,
              "NVFP4 and MXFP8 atoms must share the same C layout");

using ElemSF_NV = cutlass::float_ue4m3_t;      // 8-bit
using ElemSF_MX = cutlass::float_ue8m0_t;      // 8-bit (exponent byte)
// UINT value types for bit-exact smem->reg copies (see invariant at top of file).
using ElemFP4   = cute::uint4_t;                // raw 4-bit code (bit-copied)
using ElemFP8   = uint8_t;                      // raw e4m3 byte (bit-copied)

static constexpr int K_NV    = 64;              // NVFP4 atom K
static constexpr int VEC_NV  = 16;              // NVFP4 SFVecSize
static constexpr int K_MX    = 32;              // MXFP8 atom K
static constexpr int VEC_MX  = 32;              // MXFP8 SFVecSize

static inline void check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// ============================================================================
// cp.async helpers (ZFILL: src-size 0 reads nothing, fills zeros)
// ============================================================================
__device__ __forceinline__ void b1_cp_async16(void* smem, const void* gmem, bool pred) {
  uint32_t saddr = cute::cast_smem_ptr_to_uint(smem);
  int src_sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(saddr), "l"(gmem), "r"(src_sz));
}
__device__ __forceinline__ void b1_cp_async4(void* smem, const void* gmem, bool pred) {
  uint32_t saddr = cute::cast_smem_ptr_to_uint(smem);
  int src_sz = pred ? 4 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
               :: "r"(saddr), "l"(gmem), "r"(src_sz));
}

// ============================================================================
// Multi-warp pipelined phase runner.
//   - CTA tile (BM, BN); TiledMma = make_tiled_mma(Atom, (WM,WN,1)) over NT threads.
//   - K is consumed in (KATOM*KMUL)-wide chunks (MMA_K == KMUL per chunk; KMUL>1
//     halves barrier/loop overhead per MMA at 2x smem per stage).
//   - gmem->smem: 16B cp.async for A/B, 4B-granular cp.async for SF when the
//     per-row SF chunk is a multiple of 4 bytes, plain byte staging otherwise
//     (MXFP8 KMUL<4 — visible after the per-iteration barrier).
//   - STAGES-deep circular smem pipeline; STAGES==1 is synchronous staging.
// ============================================================================
template <class ElemData, class ElemSF, int KATOM, int VEC, int KMUL, int BM, int BN,
          int STAGES, int NT, class TiledMma, class AccumT>
__device__ inline void b1_phase(TiledMma tiled_mma,
                                const uint8_t* __restrict__ Ag,
                                const uint8_t* __restrict__ SFAg,
                                const uint8_t* __restrict__ Bg,
                                const uint8_t* __restrict__ SFBg,
                                uint8_t* sA_b, uint8_t* sB_b,
                                uint8_t* sSFA_b, uint8_t* sSFB_b,
                                int m0, int n0, int M, int N, int K,
                                int tid, AccumT& accum) {
  constexpr int EB      = cute::sizeof_bits_v<ElemData>;  // 4 or 8
  constexpr int KCH     = KATOM * KMUL;                   // K elems per chunk
  constexpr int BYTES_K = KCH * EB / 8;                   // bytes/row/chunk
  constexpr int SEG     = BYTES_K / 16;                   // 16B transfers per row
  constexpr int ROW_B   = BYTES_K + 16;                   // +16B pad: ldmatrix row
                                                          // addresses hit distinct
                                                          // banks (row-major smem)
  constexpr int ROW_E   = ROW_B * 8 / EB;                 // padded row in elements
  constexpr int NBLK    = KCH / VEC;                      // SF blocks per chunk
  constexpr int A_ST  = BM * ROW_B;                       // per-stage strides
  constexpr int B_ST  = BN * ROW_B;
  constexpr int SFA_ST = BM * NBLK;
  constexpr int SFB_ST = BN * NBLK;
  const int KpRow = K * EB / 8;                           // gmem bytes per row
  const int Knb   = K / VEC;                              // gmem SF blocks per row
  const int nchunks = K / KCH;

  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto tile_shape_mnk = tile_shape(tiled_mma);
  auto tcopy_SFA = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF>{},
                                        b1_get_layoutSFA_TV(tiled_mma),
                                        make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto tcopy_SFB = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF>{},
                                        b1_get_layoutSFB_TV(tiled_mma),
                                        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_copy_SFA = tcopy_SFA.get_thread_slice(tid);
  auto thr_copy_SFB = tcopy_SFB.get_thread_slice(tid);

  auto tcopy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, ElemData>{}, tiled_mma);
  auto tcopy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, ElemData>{}, tiled_mma);  // B frag is 8B/thr
  auto thr_copy_A = tcopy_A.get_thread_slice(tid);
  auto thr_copy_B = tcopy_B.get_thread_slice(tid);

  // ---- issue async loads of chunk c into stage s ----
  auto issue = [&](int c, int s) {
    const int kbyte0 = c * BYTES_K;
    const int kb0 = c * NBLK;
    uint8_t* dA = sA_b + s * A_ST;
    uint8_t* dB = sB_b + s * B_ST;
    uint8_t* dSFA = sSFA_b + s * SFA_ST;
    uint8_t* dSFB = sSFB_b + s * SFB_ST;
    CUTE_UNROLL
    for (int i = tid; i < BM * SEG; i += NT) {
      int row = i / SEG, seg = i % SEG;
      int mg = m0 + row;
      const uint8_t* src = Ag + (size_t)min(mg, M - 1) * KpRow + kbyte0 + seg * 16;
      b1_cp_async16(dA + row * ROW_B + seg * 16, src, mg < M);
    }
    CUTE_UNROLL
    for (int i = tid; i < BN * SEG; i += NT) {
      int row = i / SEG, seg = i % SEG;
      int ng = n0 + row;
      const uint8_t* src = Bg + (size_t)min(ng, N - 1) * KpRow + kbyte0 + seg * 16;
      b1_cp_async16(dB + row * ROW_B + seg * 16, src, ng < N);
    }
    if constexpr (NBLK % 4 == 0) {   // NVFP4: 4*KMUL SF bytes/row/chunk, Knb % 4 == 0
      constexpr int SFW = NBLK / 4;  // 4B words per row
      for (int i = tid; i < BM * SFW; i += NT) {
        int row = i / SFW, w = i % SFW;
        int mg = m0 + row;
        const uint8_t* src = SFAg + (size_t)min(mg, M - 1) * Knb + kb0 + w * 4;
        b1_cp_async4(dSFA + row * NBLK + w * 4, src, mg < M);
      }
      for (int i = tid; i < BN * SFW; i += NT) {
        int row = i / SFW, w = i % SFW;
        int ng = n0 + row;
        const uint8_t* src = SFBg + (size_t)min(ng, N - 1) * Knb + kb0 + w * 4;
        b1_cp_async4(dSFB + row * NBLK + w * 4, src, ng < N);
      }
    } else {                          // MXFP8: KMUL SF byte(s)/row/chunk — plain
      static_assert(NBLK < 4);        // staging, visible after the per-iter barrier.
      for (int i = tid; i < BM * NBLK; i += NT) {
        int row = i / NBLK, b = i % NBLK;
        int mg = m0 + row;
        dSFA[i] = (mg < M) ? SFAg[(size_t)mg * Knb + kb0 + b] : 0;
      }
      for (int i = tid; i < BN * NBLK; i += NT) {
        int row = i / NBLK, b = i % NBLK;
        int ng = n0 + row;
        dSFB[i] = (ng < N) ? SFBg[(size_t)ng * Knb + kb0 + b] : 0;
      }
    }
  };

  // ---- consume stage s: partition, bit-copy smem->reg, block-scaled MMA ----
  auto consume = [&](int s) {
    Tensor sA = make_tensor(recast_ptr<ElemData>(make_smem_ptr(sA_b + s * A_ST)),
                            make_layout(make_shape(Int<BM>{}, Int<KCH>{}),
                                        make_stride(Int<ROW_E>{}, Int<1>{})));
    Tensor sB = make_tensor(recast_ptr<ElemData>(make_smem_ptr(sB_b + s * B_ST)),
                            make_layout(make_shape(Int<BN>{}, Int<KCH>{}),
                                        make_stride(Int<ROW_E>{}, Int<1>{})));
    // SF: (rows,(VEC,NBLK)) = (rows,K) with stride-0 broadcast within each block.
    Tensor sSFA = make_tensor(recast_ptr<ElemSF>(make_smem_ptr(sSFA_b + s * SFA_ST)),
                              make_layout(make_shape(Int<BM>{}, make_shape(Int<VEC>{}, Int<NBLK>{})),
                                          make_stride(Int<NBLK>{}, make_stride(Int<0>{}, Int<1>{}))));
    Tensor sSFB = make_tensor(recast_ptr<ElemSF>(make_smem_ptr(sSFB_b + s * SFB_ST)),
                              make_layout(make_shape(Int<BN>{}, make_shape(Int<VEC>{}, Int<NBLK>{})),
                                          make_stride(Int<NBLK>{}, make_stride(Int<0>{}, Int<1>{}))));

    Tensor tCrA = thr_mma.partition_fragment_A(sA);     // (MMA,MMA_M,KMUL)
    Tensor tCrB = thr_mma.partition_fragment_B(sB);     // (MMA,MMA_N,KMUL)
    // ldmatrix smem->reg (SM75_U32x4_LDSM_N == what the CUTLASS SM120 builder
    // selects for both operands here); raw bit copy, same as the generic copy.
    copy(tcopy_A, thr_copy_A.partition_S(sA), thr_copy_A.retile_D(tCrA));
    copy(tcopy_B, thr_copy_B.partition_S(sB), thr_copy_B.retile_D(tCrB));
    // fp4_shift is a no-op for the block-scaled VS atoms; call for parity.
    fp4_shift_A(typename TiledMma::MMA_Op{}, tCrA);
    fp4_shift_B(typename TiledMma::MMA_Op{}, tCrB);

    Tensor tCrSFA = b1_partition_fragment_SFA(sSFA, thr_mma);
    Tensor tCrSFB = b1_partition_fragment_SFB(sSFB, thr_mma);
    copy(thr_copy_SFA.partition_S(sSFA), thr_copy_SFA.retile_D(tCrSFA));
    copy(thr_copy_SFB.partition_S(sSFB), thr_copy_SFB.retile_D(tCrSFB));

    CUTE_UNROLL
    for (int kb = 0; kb < KMUL; ++kb) {
      cute::gemm(tiled_mma,
                 make_zip_tensor(tCrA(_,_,kb), tCrSFA(_,_,kb)),
                 make_zip_tensor(tCrB(_,_,kb), tCrSFB(_,_,kb)),
                 accum);
    }
  };

  // ---- pipeline ----
  if constexpr (STAGES == 1) {
    for (int c = 0; c < nchunks; ++c) {
      issue(c, 0);
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();
      consume(0);
      __syncthreads();              // WAR guard before next overwrite
    }
  } else {
    // Classic sm80-style multistage: ONE barrier per chunk. Ordering per iter:
    //   barrier -> issue chunk c+S-1 -> wait(chunk c ready) -> consume chunk c.
    // The issue targets stage (c-1)%S, consumed at iter c-1; the top barrier
    // orders every thread past that consume before the overwrite (WAR safe).
    // Issuing BEFORE the wait lets chunk c+S-1's gmem latency overlap the
    // wait for chunk c (the old wait-then-issue order serialized them; ncu
    // showed long_scoreboard 4.7/issue as a co-dominant stall). Plain SF
    // byte stores (MXFP8) are issued >=1 iteration before their consume, so a
    // top barrier always separates store and load (visibility safe).
    const int pre = (nchunks < STAGES - 1) ? nchunks : STAGES - 1;
    for (int c = 0; c < pre; ++c) {
      issue(c, c);
      cp_async_fence();
    }
    for (int c = 0; c < nchunks; ++c) {
      __syncthreads();
      int cnext = c + STAGES - 1;
      if (cnext < nchunks) {
        issue(cnext, cnext % STAGES);
      }
      cp_async_fence();              // keep group accounting monotone (may be empty)
      // Pending data groups: chunks c .. c+S-1 (S groups); waiting for <= S-1
      // retires chunk c's group (FIFO) while chunks c+1..c+S-1 stay in flight.
      cp_async_wait<STAGES - 1>();
      __syncthreads();
      consume(c % STAGES);
    }
    __syncthreads();                 // phase boundary: smem reused by the caller
  }
}

// ============================================================================
// mbarrier helpers (sm90+ PTX, valid on sm120; pattern from cutlass/arch/barrier.h)
// ============================================================================
__device__ __forceinline__ void b1_mbar_init(uint64_t* mbar, uint32_t expected) {
  uint32_t addr = cute::cast_smem_ptr_to_uint(mbar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(expected));
}
__device__ __forceinline__ void b1_mbar_arrive(uint64_t* mbar) {
  uint32_t addr = cute::cast_smem_ptr_to_uint(mbar);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(addr));
}
// Make all of this thread's prior cp.async ops arrive on mbar when they complete
// (default inc form: pending count bumps at issue, balances at completion, so the
// phase cannot complete before the async copies land; expected counts only the
// plain arrives).
__device__ __forceinline__ void b1_cp_async_mbar_arrive(uint64_t* mbar) {
  uint32_t addr = cute::cast_smem_ptr_to_uint(mbar);
  asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n" :: "r"(addr));
}
__device__ __forceinline__ void b1_mbar_wait(uint64_t* mbar, uint32_t parity) {
  uint32_t addr = cute::cast_smem_ptr_to_uint(mbar);
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\n"
      "bra     LAB_WAIT;\n"
      "DONE:\n"
      "}\n"
      :: "r"(addr), "r"(parity));
}

// ============================================================================
// Warp-specialized B1 kernel (cfg 10): WM*WN consumer warps run ONLY
// ldmatrix+MMA; 2 producer warps run ONLY cp.async staging. Stage hand-off via
// smem mbarriers (full[s]: producers' plain arrive (releases the MXFP8 plain SF
// byte stores) + cp.async completion; free[s]: consumer threads arrive after
// the stage's MMAs). No CTA-wide __syncthreads in the K loop, and MMA warps
// never issue global loads — the two co-dominant stalls (mio_throttle,
// long_scoreboard) of the homogeneous kernel. 1 CTA/SM (smem-bound), STAGES=3/4.
// Both phases run one global chunk sequence g=0..n1+n2-1 over shared buffers.
// ============================================================================
template <int BM, int BN, int WM, int WN, int STAGES, int KMUL>
__global__ void __launch_bounds__(WM * WN * 32 + 64)
b1_ws_kernel(const uint8_t* __restrict__ Al,   const uint8_t* __restrict__ SFAl,
             const uint8_t* __restrict__ Bl,   const uint8_t* __restrict__ SFBl,
             const uint8_t* __restrict__ Ah,   const uint8_t* __restrict__ SFAh,
             const uint8_t* __restrict__ Bh,   const uint8_t* __restrict__ SFBh,
             const float* __restrict__ s_x,    const float* __restrict__ s_w,
             __half* __restrict__ D,
             int M, int N, int K_low, int K_high) {
  constexpr int NT_C   = WM * WN * 32;      // consumer threads (tid < NT_C)
  constexpr int NT_P   = 64;                // producer threads
  constexpr int BYTES_K = 32 * KMUL;        // bytes/row/chunk (both phases)
  constexpr int SEG    = BYTES_K / 16;
  constexpr int ROW_B  = BYTES_K + 16;      // ldmatrix bank pad
  constexpr int NBLK4  = 4 * KMUL;          // fp4 SF blocks/row/chunk
  constexpr int NBLK8  = KMUL;              // fp8 SF blocks/row/chunk
  constexpr int A_ST   = BM * ROW_B;
  constexpr int B_ST   = BN * ROW_B;
  constexpr int SFA_ST = BM * NBLK4;        // sized for the fp4 phase (larger)
  constexpr int SFB_ST = BN * NBLK4;
  static_assert(NBLK4 % 4 == 0, "fp4 SF rows must be 4B-multiples for cp.async");

  __shared__ alignas(16) uint8_t sA_b[STAGES * A_ST];
  __shared__ alignas(16) uint8_t sB_b[STAGES * B_ST];
  __shared__ alignas(4) uint8_t sSFA_b[STAGES * SFA_ST];
  __shared__ alignas(4) uint8_t sSFB_b[STAGES * SFB_ST];
  __shared__ alignas(8) uint64_t mbar[2 * STAGES];  // [0,S): full, [S,2S): free

  const int tid = threadIdx.x;
  const int m0 = blockIdx.x * BM;
  const int n0 = blockIdx.y * BN;
  const int n1 = K_low / (K_NV * KMUL);
  const int n2 = K_high / (K_MX * KMUL);
  const int ntot = n1 + n2;

  if (tid == 0) {
    CUTE_UNROLL
    for (int s = 0; s < STAGES; ++s) {
      b1_mbar_init(&mbar[s], NT_P);           // full: producers' plain arrives
      b1_mbar_init(&mbar[STAGES + s], NT_C);  // free: consumer thread arrives
    }
  }
  __syncthreads();

  if (tid >= NT_C) {
    // ---------------- producer warps ----------------
    const int ptid = tid - NT_C;
    for (int g = 0; g < ntot; ++g) {
      const int s = g % STAGES;
      if (g >= STAGES) b1_mbar_wait(&mbar[STAGES + s], ((g / STAGES) - 1) & 1);
      uint8_t* dA = sA_b + s * A_ST;
      uint8_t* dB = sB_b + s * B_ST;
      uint8_t* dSFA = sSFA_b + s * SFA_ST;
      uint8_t* dSFB = sSFB_b + s * SFB_ST;
      const bool f4 = (g < n1);
      const int c = f4 ? g : g - n1;
      const uint8_t* Ag = f4 ? Al : Ah;
      const uint8_t* Bg = f4 ? Bl : Bh;
      const int KpRow = f4 ? (K_low / 2) : K_high;   // gmem bytes/row
      const int kbyte0 = c * BYTES_K;
      for (int i = ptid; i < BM * SEG; i += NT_P) {
        int row = i / SEG, seg = i % SEG;
        int mg = m0 + row;
        b1_cp_async16(dA + row * ROW_B + seg * 16,
                      Ag + (size_t)min(mg, M - 1) * KpRow + kbyte0 + seg * 16, mg < M);
      }
      for (int i = ptid; i < BN * SEG; i += NT_P) {
        int row = i / SEG, seg = i % SEG;
        int ng = n0 + row;
        b1_cp_async16(dB + row * ROW_B + seg * 16,
                      Bg + (size_t)min(ng, N - 1) * KpRow + kbyte0 + seg * 16, ng < N);
      }
      if (f4) {   // 4B-granular cp.async SF staging
        constexpr int SFW = NBLK4 / 4;
        const int Knb = K_low / VEC_NV;
        const int kb0 = c * NBLK4;
        for (int i = ptid; i < BM * SFW; i += NT_P) {
          int row = i / SFW, w = i % SFW;
          int mg = m0 + row;
          b1_cp_async4(dSFA + row * NBLK4 + w * 4,
                       SFAl + (size_t)min(mg, M - 1) * Knb + kb0 + w * 4, mg < M);
        }
        for (int i = ptid; i < BN * SFW; i += NT_P) {
          int row = i / SFW, w = i % SFW;
          int ng = n0 + row;
          b1_cp_async4(dSFB + row * NBLK4 + w * 4,
                       SFBl + (size_t)min(ng, N - 1) * Knb + kb0 + w * 4, ng < N);
        }
      } else {    // plain byte staging (released by the plain mbarrier arrive)
        const int Knb = K_high / VEC_MX;
        const int kb0 = c * NBLK8;
        for (int i = ptid; i < BM * NBLK8; i += NT_P) {
          int row = i / NBLK8, b = i % NBLK8;
          int mg = m0 + row;
          dSFA[i] = (mg < M) ? SFAh[(size_t)mg * Knb + kb0 + b] : 0;
        }
        for (int i = ptid; i < BN * NBLK8; i += NT_P) {
          int row = i / NBLK8, b = i % NBLK8;
          int ng = n0 + row;
          dSFB[i] = (ng < N) ? SFBh[(size_t)ng * Knb + kb0 + b] : 0;
        }
      }
      b1_cp_async_mbar_arrive(&mbar[s]);  // inc now, arrive at completion
      b1_mbar_arrive(&mbar[s]);           // plain arrive (counts; releases st.shared)
    }
    return;  // producers hold no accumulator
  }

  // ---------------- consumer warps ----------------
  using TiledNV = decltype(make_tiled_mma(AtomNV{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{}));
  using TiledMX = decltype(make_tiled_mma(AtomMX{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{}));
  TiledNV mma_nv;
  TiledMX mma_mx;
  Tensor accum = partition_fragment_C(mma_nv, Shape<Int<BM>, Int<BN>>{});
  static_assert(cute::is_same_v<
      decltype(partition_fragment_C(TiledNV{}, Shape<Int<BM>, Int<BN>>{}).layout()),
      decltype(partition_fragment_C(TiledMX{}, Shape<Int<BM>, Int<BN>>{}).layout())>,
      "NV/MX tiled MMAs must partition C identically to share the accumulator");
  clear(accum);

  auto thr_nv = mma_nv.get_thread_slice(tid);
  auto thr_mx = mma_mx.get_thread_slice(tid);
  auto tile_nv = tile_shape(mma_nv);
  auto tile_mx = tile_shape(mma_mx);
  auto tcopy_SFA4 = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF_NV>{},
                                         b1_get_layoutSFA_TV(mma_nv),
                                         make_shape(size<0>(tile_nv), size<2>(tile_nv)));
  auto tcopy_SFB4 = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF_NV>{},
                                         b1_get_layoutSFB_TV(mma_nv),
                                         make_shape(size<1>(tile_nv), size<2>(tile_nv)));
  auto tcopy_SFA8 = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF_MX>{},
                                         b1_get_layoutSFA_TV(mma_mx),
                                         make_shape(size<0>(tile_mx), size<2>(tile_mx)));
  auto tcopy_SFB8 = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF_MX>{},
                                         b1_get_layoutSFB_TV(mma_mx),
                                         make_shape(size<1>(tile_mx), size<2>(tile_mx)));
  auto tcopy_A4 = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, ElemFP4>{}, mma_nv);
  auto tcopy_B4 = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, ElemFP4>{}, mma_nv);
  auto tcopy_A8 = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, ElemFP8>{}, mma_mx);
  auto tcopy_B8 = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, ElemFP8>{}, mma_mx);

  auto consume4 = [&](int s) {
    constexpr int KCH = K_NV * KMUL;
    Tensor sA = make_tensor(recast_ptr<ElemFP4>(make_smem_ptr(sA_b + s * A_ST)),
                            make_layout(make_shape(Int<BM>{}, Int<KCH>{}),
                                        make_stride(Int<ROW_B * 2>{}, Int<1>{})));
    Tensor sB = make_tensor(recast_ptr<ElemFP4>(make_smem_ptr(sB_b + s * B_ST)),
                            make_layout(make_shape(Int<BN>{}, Int<KCH>{}),
                                        make_stride(Int<ROW_B * 2>{}, Int<1>{})));
    Tensor sSFA = make_tensor(recast_ptr<ElemSF_NV>(make_smem_ptr(sSFA_b + s * SFA_ST)),
                              make_layout(make_shape(Int<BM>{}, make_shape(Int<VEC_NV>{}, Int<NBLK4>{})),
                                          make_stride(Int<NBLK4>{}, make_stride(Int<0>{}, Int<1>{}))));
    Tensor sSFB = make_tensor(recast_ptr<ElemSF_NV>(make_smem_ptr(sSFB_b + s * SFB_ST)),
                              make_layout(make_shape(Int<BN>{}, make_shape(Int<VEC_NV>{}, Int<NBLK4>{})),
                                          make_stride(Int<NBLK4>{}, make_stride(Int<0>{}, Int<1>{}))));
    Tensor tCrA = thr_nv.partition_fragment_A(sA);
    Tensor tCrB = thr_nv.partition_fragment_B(sB);
    copy(tcopy_A4, tcopy_A4.get_thread_slice(tid).partition_S(sA),
         tcopy_A4.get_thread_slice(tid).retile_D(tCrA));
    copy(tcopy_B4, tcopy_B4.get_thread_slice(tid).partition_S(sB),
         tcopy_B4.get_thread_slice(tid).retile_D(tCrB));
    fp4_shift_A(typename TiledNV::MMA_Op{}, tCrA);
    fp4_shift_B(typename TiledNV::MMA_Op{}, tCrB);
    Tensor tCrSFA = b1_partition_fragment_SFA(sSFA, thr_nv);
    Tensor tCrSFB = b1_partition_fragment_SFB(sSFB, thr_nv);
    copy(tcopy_SFA4.get_thread_slice(tid).partition_S(sSFA),
         tcopy_SFA4.get_thread_slice(tid).retile_D(tCrSFA));
    copy(tcopy_SFB4.get_thread_slice(tid).partition_S(sSFB),
         tcopy_SFB4.get_thread_slice(tid).retile_D(tCrSFB));
    CUTE_UNROLL
    for (int kb = 0; kb < KMUL; ++kb)
      cute::gemm(mma_nv, make_zip_tensor(tCrA(_,_,kb), tCrSFA(_,_,kb)),
                 make_zip_tensor(tCrB(_,_,kb), tCrSFB(_,_,kb)), accum);
  };
  auto consume8 = [&](int s) {
    constexpr int KCH = K_MX * KMUL;
    Tensor sA = make_tensor(recast_ptr<ElemFP8>(make_smem_ptr(sA_b + s * A_ST)),
                            make_layout(make_shape(Int<BM>{}, Int<KCH>{}),
                                        make_stride(Int<ROW_B>{}, Int<1>{})));
    Tensor sB = make_tensor(recast_ptr<ElemFP8>(make_smem_ptr(sB_b + s * B_ST)),
                            make_layout(make_shape(Int<BN>{}, Int<KCH>{}),
                                        make_stride(Int<ROW_B>{}, Int<1>{})));
    Tensor sSFA = make_tensor(recast_ptr<ElemSF_MX>(make_smem_ptr(sSFA_b + s * SFA_ST)),
                              make_layout(make_shape(Int<BM>{}, make_shape(Int<VEC_MX>{}, Int<NBLK8>{})),
                                          make_stride(Int<NBLK8>{}, make_stride(Int<0>{}, Int<1>{}))));
    Tensor sSFB = make_tensor(recast_ptr<ElemSF_MX>(make_smem_ptr(sSFB_b + s * SFB_ST)),
                              make_layout(make_shape(Int<BN>{}, make_shape(Int<VEC_MX>{}, Int<NBLK8>{})),
                                          make_stride(Int<NBLK8>{}, make_stride(Int<0>{}, Int<1>{}))));
    Tensor tCrA = thr_mx.partition_fragment_A(sA);
    Tensor tCrB = thr_mx.partition_fragment_B(sB);
    copy(tcopy_A8, tcopy_A8.get_thread_slice(tid).partition_S(sA),
         tcopy_A8.get_thread_slice(tid).retile_D(tCrA));
    copy(tcopy_B8, tcopy_B8.get_thread_slice(tid).partition_S(sB),
         tcopy_B8.get_thread_slice(tid).retile_D(tCrB));
    fp4_shift_A(typename TiledMX::MMA_Op{}, tCrA);
    fp4_shift_B(typename TiledMX::MMA_Op{}, tCrB);
    Tensor tCrSFA = b1_partition_fragment_SFA(sSFA, thr_mx);
    Tensor tCrSFB = b1_partition_fragment_SFB(sSFB, thr_mx);
    copy(tcopy_SFA8.get_thread_slice(tid).partition_S(sSFA),
         tcopy_SFA8.get_thread_slice(tid).retile_D(tCrSFA));
    copy(tcopy_SFB8.get_thread_slice(tid).partition_S(sSFB),
         tcopy_SFB8.get_thread_slice(tid).retile_D(tCrSFB));
    CUTE_UNROLL
    for (int kb = 0; kb < KMUL; ++kb)
      cute::gemm(mma_mx, make_zip_tensor(tCrA(_,_,kb), tCrSFA(_,_,kb)),
                 make_zip_tensor(tCrB(_,_,kb), tCrSFB(_,_,kb)), accum);
  };

  for (int g = 0; g < ntot; ++g) {
    const int s = g % STAGES;
    b1_mbar_wait(&mbar[s], (g / STAGES) & 1);
    if (g < n1) consume4(s); else consume8(s);
    b1_mbar_arrive(&mbar[STAGES + s]);
  }

  // Epilogue (consumers only) — same vectorized form as b1_mw_kernel.
  Tensor cD = thr_nv.partition_C(make_identity_tensor(Shape<Int<BM>, Int<BN>>{}));
  CUTE_UNROLL
  for (int i = 0; i < size(accum); i += 2) {
    int mg = m0 + get<0>(cD(i));
    int ng = n0 + get<1>(cD(i));
    if (mg >= M) continue;
    float sx = s_x[mg];
    if (ng + 1 < N) {
      __half2 h = __floats2half2_rn(accum(i) * sx * s_w[ng],
                                    accum(i + 1) * sx * s_w[ng + 1]);
      *reinterpret_cast<__half2*>(&D[(size_t)mg * N + ng]) = h;
    } else if (ng < N) {
      D[(size_t)mg * N + ng] = __float2half(accum(i) * sx * s_w[ng]);
    }
  }
}

// ============================================================================
// STEP 1 kernel — NVFP4-only isolation entry (naive 1-warp, kept as the
// correctness yardstick for the MMA/SF plumbing). Output fp32 [M,N].
// grid = (ceil(M/16), ceil(N/8)); block = 32 threads (one warp).
// ============================================================================
__global__ void b1_nvfp4_kernel(const uint8_t* __restrict__ A_packed,  // [M, K/2]
                                const uint8_t* __restrict__ SFA,        // [M, K/16]
                                const uint8_t* __restrict__ B_packed,  // [N, K/2]
                                const uint8_t* __restrict__ SFB,        // [N, K/16]
                                float* __restrict__ D,                 // [M, N]
                                int M, int N, int K) {
  const int tid = threadIdx.x;              // 0..31
  const int m0 = blockIdx.x * 16;
  const int n0 = blockIdx.y * 8;

  __shared__ alignas(16) uint8_t sA_b[16 * 48];   // 32B/row + 16B ldmatrix pad
  __shared__ alignas(16) uint8_t sB_b[8 * 48];
  __shared__ alignas(4) uint8_t sSFA_b[16 * 4];
  __shared__ alignas(4) uint8_t sSFB_b[8 * 4];

  auto tiled_mma = make_tiled_mma(AtomNV{});
  Tensor accum = partition_fragment_C(tiled_mma, Shape<Int<16>, Int<8>>{});
  clear(accum);

  b1_phase<ElemFP4, ElemSF_NV, K_NV, VEC_NV, 1, 16, 8, 1, 32>(
      tiled_mma, A_packed, SFA, B_packed, SFB,
      sA_b, sB_b, sSFA_b, sSFB_b, m0, n0, M, N, K, tid, accum);

  // store accum -> D (SM80_16x8_Row: thread t -> rows {t/4,t/4+8}, cols {2(t%4),+1})
  int r0 = tid / 4;
  int cc = (tid % 4) * 2;
  int rows[2] = {r0, r0 + 8};
  int cols[2] = {cc, cc + 1};
  float vals[4] = {accum(0), accum(1), accum(2), accum(3)};
  int idx = 0;
  for (int rr = 0; rr < 2; ++rr) {
    for (int ccx = 0; ccx < 2; ++ccx) {
      int mg = m0 + rows[rr];
      int ng = n0 + cols[ccx];
      if (mg < M && ng < N) D[mg * N + ng] = vals[idx];
      ++idx;
    }
  }
}

// ============================================================================
// Full B1 kernel, multi-warp — Phase 1 NVFP4 over K_low, Phase 2 MXFP8 over
// K_high, ONE register accumulator; epilogue scales by s_x[m]*s_w[n] -> fp16.
// ============================================================================
template <int BM, int BN, int WM, int WN, int STAGES, int KMUL>
__global__ void __launch_bounds__(WM * WN * 32)
b1_mw_kernel(const uint8_t* __restrict__ Al,   // [M, K_low/2]
             const uint8_t* __restrict__ SFAl, // [M, K_low/16]
             const uint8_t* __restrict__ Bl,   // [N, K_low/2]
             const uint8_t* __restrict__ SFBl, // [N, K_low/16]
             const uint8_t* __restrict__ Ah,   // [M, K_high]
             const uint8_t* __restrict__ SFAh, // [M, K_high/32]
             const uint8_t* __restrict__ Bh,   // [N, K_high]
             const uint8_t* __restrict__ SFBh, // [N, K_high/32]
             const float* __restrict__ s_x,    // [M]
             const float* __restrict__ s_w,    // [N]
             __half* __restrict__ D,           // [M, N]
             int M, int N, int K_low, int K_high) {
  constexpr int NT = WM * WN * 32;
  const int tid = threadIdx.x;
  const int m0 = blockIdx.x * BM;
  const int n0 = blockIdx.y * BN;

  // Both phases stage 32 bytes/row/chunk; buffers reused across phases.
  // SF sized for the NVFP4 phase (4 blk/row); MXFP8 needs 1.
  __shared__ alignas(16) uint8_t sA_b[STAGES * BM * (32 * KMUL + 16)];  // +16B/row
  __shared__ alignas(16) uint8_t sB_b[STAGES * BN * (32 * KMUL + 16)];  // ldmatrix pad
  __shared__ alignas(4) uint8_t sSFA_b[STAGES * BM * 4 * KMUL];
  __shared__ alignas(4) uint8_t sSFB_b[STAGES * BN * 4 * KMUL];

  using TiledNV = decltype(make_tiled_mma(AtomNV{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{}));
  using TiledMX = decltype(make_tiled_mma(AtomMX{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{}));
  TiledNV mma_nv;
  TiledMX mma_mx;
  Tensor accum = partition_fragment_C(mma_nv, Shape<Int<BM>, Int<BN>>{});
  static_assert(cute::is_same_v<
      decltype(partition_fragment_C(TiledNV{}, Shape<Int<BM>, Int<BN>>{}).layout()),
      decltype(partition_fragment_C(TiledMX{}, Shape<Int<BM>, Int<BN>>{}).layout())>,
      "NV/MX tiled MMAs must partition C identically to share the accumulator");
  clear(accum);

  // Phase 1: NVFP4 over K_low
  b1_phase<ElemFP4, ElemSF_NV, K_NV, VEC_NV, KMUL, BM, BN, STAGES, NT>(
      mma_nv, Al, SFAl, Bl, SFBl,
      sA_b, sB_b, sSFA_b, sSFB_b, m0, n0, M, N, K_low, tid, accum);

  // Phase 2: MXFP8 over K_high into the SAME accumulator (buffers reused;
  // phase-1 loop ends with a barrier, so all reads of the old data are done).
  b1_phase<ElemFP8, ElemSF_MX, K_MX, VEC_MX, KMUL, BM, BN, STAGES, NT>(
      mma_mx, Ah, SFAh, Bh, SFBh,
      sA_b, sB_b, sSFA_b, sSFB_b, m0, n0, M, N, K_high, tid, accum);

  // Epilogue: out[m,n] = accum * s_x[m] * s_w[n] -> fp16. CLayout SM80_16x8_Row
  // puts accum pairs (i, i+1), i even, at the same row and adjacent (even) cols
  // -> __half2 vector stores (4B-aligned since c is even and N is even), one
  // s_x load per pair instead of per element (s_w/s_x CSE'd across MMA_M/N).
  auto thr_mma = mma_nv.get_thread_slice(tid);
  Tensor cD = thr_mma.partition_C(make_identity_tensor(Shape<Int<BM>, Int<BN>>{}));
  CUTE_UNROLL
  for (int i = 0; i < size(accum); i += 2) {
    int mg = m0 + get<0>(cD(i));
    int ng = n0 + get<1>(cD(i));
    if (mg >= M) continue;
    float sx = s_x[mg];
    if (ng + 1 < N) {
      __half2 h = __floats2half2_rn(accum(i) * sx * s_w[ng],
                                    accum(i + 1) * sx * s_w[ng + 1]);
      *reinterpret_cast<__half2*>(&D[(size_t)mg * N + ng]) = h;
    } else if (ng < N) {
      D[(size_t)mg * N + ng] = __float2half(accum(i) * sx * s_w[ng]);
    }
  }
}

// ============================================================================
// entry points
// ============================================================================
torch::Tensor b1_nvfp4_only(torch::Tensor A_packed, torch::Tensor SFA,
                            torch::Tensor B_packed, torch::Tensor SFB,
                            int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(A_packed.is_cuda() && B_packed.is_cuda() && SFA.is_cuda() && SFB.is_cuda(),
              "inputs must be CUDA");
  TORCH_CHECK(K % K_NV == 0, "K must be divisible by 64");
  A_packed = A_packed.contiguous(); B_packed = B_packed.contiguous();
  SFA = SFA.contiguous(); SFB = SFB.contiguous();
  auto dev = A_packed.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev.index());

  auto D = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));

  dim3 grid((M + 15) / 16, (N + 7) / 8);
  b1_nvfp4_kernel<<<grid, 32, 0, stream>>>(
      A_packed.data_ptr<uint8_t>(), SFA.data_ptr<uint8_t>(),
      B_packed.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(),
      D.data_ptr<float>(), (int)M, (int)N, (int)K);
  check_cuda(cudaGetLastError(), "b1_nvfp4_kernel launch");
  check_cuda(cudaStreamSynchronize(stream), "b1 sync");
  return D;
}

template <int BM, int BN, int WM, int WN, int STAGES, int KMUL = 1>
static void launch_b1_mw(const torch::Tensor& Al, const torch::Tensor& SFAl,
                         const torch::Tensor& Bl, const torch::Tensor& SFBl,
                         const torch::Tensor& Ah, const torch::Tensor& SFAh,
                         const torch::Tensor& Bh, const torch::Tensor& SFBh,
                         const torch::Tensor& s_x, const torch::Tensor& s_w,
                         torch::Tensor& D, int M, int N, int K_low, int K_high,
                         cudaStream_t stream) {
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  b1_mw_kernel<BM, BN, WM, WN, STAGES, KMUL><<<grid, WM * WN * 32, 0, stream>>>(
      Al.data_ptr<uint8_t>(), SFAl.data_ptr<uint8_t>(),
      Bl.data_ptr<uint8_t>(), SFBl.data_ptr<uint8_t>(),
      Ah.data_ptr<uint8_t>(), SFAh.data_ptr<uint8_t>(),
      Bh.data_ptr<uint8_t>(), SFBh.data_ptr<uint8_t>(),
      s_x.data_ptr<float>(), s_w.data_ptr<float>(),
      reinterpret_cast<__half*>(D.data_ptr<at::Half>()),
      M, N, K_low, K_high);
}

template <int BM, int BN, int WM, int WN, int STAGES, int KMUL = 1>
static void launch_b1_ws(const torch::Tensor& Al, const torch::Tensor& SFAl,
                         const torch::Tensor& Bl, const torch::Tensor& SFBl,
                         const torch::Tensor& Ah, const torch::Tensor& SFAh,
                         const torch::Tensor& Bh, const torch::Tensor& SFBh,
                         const torch::Tensor& s_x, const torch::Tensor& s_w,
                         torch::Tensor& D, int M, int N, int K_low, int K_high,
                         cudaStream_t stream) {
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  b1_ws_kernel<BM, BN, WM, WN, STAGES, KMUL><<<grid, WM * WN * 32 + 64, 0, stream>>>(
      Al.data_ptr<uint8_t>(), SFAl.data_ptr<uint8_t>(),
      Bl.data_ptr<uint8_t>(), SFBl.data_ptr<uint8_t>(),
      Ah.data_ptr<uint8_t>(), SFAh.data_ptr<uint8_t>(),
      Bh.data_ptr<uint8_t>(), SFBh.data_ptr<uint8_t>(),
      s_x.data_ptr<float>(), s_w.data_ptr<float>(),
      reinterpret_cast<__half*>(D.data_ptr<at::Half>()),
      M, N, K_low, K_high);
}

// stages: 0 = auto (best measured), 1 = synchronous, 2/3 = pipeline depth.
// cfg:    0 = auto (by M), 1 = 16x64x4w, 2 = 16x32x2w, 3 = 32x64x4w, 4 = 128x128x8w.
torch::Tensor b1_fused(torch::Tensor Al, torch::Tensor SFAl,
                       torch::Tensor Bl, torch::Tensor SFBl,
                       torch::Tensor Ah, torch::Tensor SFAh,
                       torch::Tensor Bh, torch::Tensor SFBh,
                       torch::Tensor s_x, torch::Tensor s_w,
                       int64_t M, int64_t N, int64_t K_low, int64_t K_high,
                       int64_t stages, int64_t cfg) {
  TORCH_CHECK(Al.is_cuda() && Bl.is_cuda() && Ah.is_cuda() && Bh.is_cuda(),
              "inputs must be CUDA");
  TORCH_CHECK(K_low % K_NV == 0, "K_low must be divisible by 64");
  TORCH_CHECK(K_high % K_MX == 0, "K_high must be divisible by 32");
  TORCH_CHECK(s_x.scalar_type() == torch::kFloat32 && s_w.scalar_type() == torch::kFloat32,
              "s_x/s_w must be fp32");
  Al = Al.contiguous(); Bl = Bl.contiguous();
  SFAl = SFAl.contiguous(); SFBl = SFBl.contiguous();
  Ah = Ah.contiguous(); Bh = Bh.contiguous();
  SFAh = SFAh.contiguous(); SFBh = SFBh.contiguous();
  s_x = s_x.contiguous(); s_w = s_w.contiguous();
  auto dev = Al.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev.index());

  auto D = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(dev));

  int cf = (int)cfg;
  // Auto cfg (measured, GPU6 quiet): decode -> 16x64. Mid-M: the 32x64 tile fills
  // the 188 SMs when the 128x128 grid would be tiny (M=256,N=2048 -> 32 CTAs);
  // when N alone yields enough 128x128 CTAs (N>=4096) the bigger tile's B-reuse
  // wins (gate/up M=256: cfg4+pipe 0.043ms vs cfg3 0.071ms). Prefill: with
  // ldmatrix smem->reg (b1_stepB2 sweep) KMUL=2 (cfg5, stages=2) wins on ALL
  // CTA-rich M=2048 shapes incl. N=2048 (down1B 0.162 vs cfg4s1 0.180, q1B
  // 0.063 vs 0.068, gate 0.159 vs 0.191, down3B 0.240 vs 0.330) -> drop the
  // old pre-ldmatrix N>=3072 gate.
  const bool kmul2_ok = (K_low % (2 * K_NV) == 0) && (K_high % (2 * K_MX) == 0);
  if (cf == 0) {
    const int ctas128 = (((int)M + 127) / 128) * (((int)N + 127) / 128);
    if (M <= 16)       cf = (N <= 4096) ? 2 : 1;  // 16x32 doubles decode grid; at
                                                  // N=8192 the 16x64 grid is enough
                                                  // (q 0.0199 vs 0.0214, down 0.0338
                                                  // vs 0.0393; gate prefers cfg1)
    else if (M <= 256) cf = (N >= 4096) ? 4 : 3;
    else if (ctas128 < 160) cf = kmul2_ok ? 6 : 3; // small-N prefill (k/v: 64 CTAs
                                                   // at 128x128 underfill 188 SMs;
                                                   // 64x128+KMUL2 0.032 vs 0.048)
    else               cf = kmul2_ok ? 10 : 4;     // CTA-rich prefill: WS kernel
                                                   // (gate 0.135 vs cfg5 0.137,
                                                   // down3B 0.215 vs 0.231)
  }
  TORCH_CHECK((cf >= 1 && cf <= 6) || cf == 10, "cfg must be 0..6 or 10");
  if (cf == 5 || cf == 6 || cf == 10)
    TORCH_CHECK(kmul2_ok, "cfg 5/6/10 (KMUL=2) needs K_low%128==0 and K_high%64==0");

  int st = (int)stages;
  // Auto stages (measured): pipeline (3) wins when the grid underfills the SMs
  // (M<=256); at M=2048 a 2-stage pipeline wins for the homogeneous KMUL=2
  // kernels and the 4-deep variant (st=3) wins for the WS kernel (cfg 10).
  if (st == 0) st = (M <= 256) ? 3 : ((cf == 10) ? 3 : ((cf >= 5) ? 2 : 1));
  TORCH_CHECK(st >= 1 && st <= 3, "stages must be 0..3");

  const int Mi = (int)M, Ni = (int)N, Kl = (int)K_low, Kh = (int)K_high;
#define B1_LAUNCH(BM, BN, WM, WN, ST, KM) \
    launch_b1_mw<BM, BN, WM, WN, ST, KM>(Al, SFAl, Bl, SFBl, Ah, SFAh, Bh, SFBh, \
                                         s_x, s_w, D, Mi, Ni, Kl, Kh, stream)
#define B1_CFG(BM, BN, WM, WN, KM) \
    { if (st == 1) B1_LAUNCH(BM, BN, WM, WN, 1, KM); \
      else if (st == 2) B1_LAUNCH(BM, BN, WM, WN, 2, KM); \
      else B1_LAUNCH(BM, BN, WM, WN, 3, KM); }
  switch (cf) {
    case 1: B1_CFG(16, 64, 1, 4, 1); break;    // decode
    case 2: B1_CFG(16, 32, 1, 2, 1); break;    // decode probe: 2x the CTAs
    case 3: B1_CFG(32, 64, 2, 2, 1); break;    // mid-M: grid-count first
    case 5: B1_CFG(128, 128, 2, 4, 2); break;  // prefill, 2 MMA-K per chunk
    case 6: B1_CFG(64, 128, 2, 4, 2); break;   // prefill probe: halve accum regs
                                               // (occupancy 2->3 CTA/SM target)
    case 10:  // warp-specialized: stages arg 3 -> 4-deep (90KB), else 3-deep (68KB)
      { if (st == 3) launch_b1_ws<128, 128, 2, 4, 4, 2>(Al, SFAl, Bl, SFBl, Ah, SFAh,
                                                        Bh, SFBh, s_x, s_w, D, Mi, Ni,
                                                        Kl, Kh, stream);
        else         launch_b1_ws<128, 128, 2, 4, 3, 2>(Al, SFAl, Bl, SFBl, Ah, SFAh,
                                                        Bh, SFBh, s_x, s_w, D, Mi, Ni,
                                                        Kl, Kh, stream); }
      break;
    default: B1_CFG(128, 128, 2, 4, 1); break; // prefill
  }
#undef B1_CFG
#undef B1_LAUNCH
  check_cuda(cudaGetLastError(), "b1_mw_kernel launch");
  check_cuda(cudaStreamSynchronize(stream), "b1_fused sync");
  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("b1_nvfp4_only", &b1_nvfp4_only, "B1 step1: NVFP4-only hand kernel -> fp32 (SM120)");
  m.def("b1_fused", &b1_fused,
        "B1: TRUE single-kernel fused NVFP4(K_low)+MXFP8(K_high) GEMM * s_x * s_w -> fp16 (SM120)",
        py::arg("Al"), py::arg("SFAl"), py::arg("Bl"), py::arg("SFBl"),
        py::arg("Ah"), py::arg("SFAh"), py::arg("Bh"), py::arg("SFBh"),
        py::arg("s_x"), py::arg("s_w"),
        py::arg("M"), py::arg("N"), py::arg("K_low"), py::arg("K_high"),
        py::arg("stages") = 0, py::arg("cfg") = 0);
}
