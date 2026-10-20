// B1 TMA — Phase-2 (MXFP8) single-kernel GEMM for SM120 with TMA data loads.
//
// Targets the ncu-identified worst case: down-class r=1 (pure MXFP8, K_high only,
// M=2048 N=2048 K=8192) where the cp.async WS kernel sits +43.6% over the pure
// MXFP8 bound, stall mix dominated by wait/long_scoreboard at 1 CTA/SM (smem-bound).
//
// Design: warp-specialized producer/consumer like b1_ws_kernel, but the A_high/
// B_high DATA tiles are staged gmem->smem via cp.async.bulk.tensor (TMA) into a
// swizzled smem layout (UMMA::Layout_K_SW*_Atom<uint8>, the layout the SM120
// block-scaled collective itself uses), consumed by SM75_U32x4_LDSM_N ldmatrix
// (the atom the SM120 collective pairs with that swizzle for 8-bit). mbarrier
// expect_tx gates TMA completion; SF stays on the proven plain-store path (tiny:
// KMUL bytes/row/chunk). One elected producer thread issues both TMAs per stage.
//
// Validated plumbing: tma_probe_ext.cu round-trips the swizzle bit-exactly.
// CORRECTNESS INVARIANT (same as b1_fused_ext.cu): smem viewed as uint8 (bit-copy).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cutlass/gemm/collective/builders/sm120_common.inl"
#include "cutlass/float8.h"

using namespace cute;

// ---- SF-partition helpers (verbatim from b1_fused_ext.cu / CUTLASS) ----
template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto
tt_thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
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
  return zipped_divide(tv_tensor, thr_tile);
}
template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto
tt_thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
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
  return zipped_divide(tv_tensor, thr_tile);
}
template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
tt_partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(),
                                tt_thrfrg_SFA(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  auto partition_SFA = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(partition_SFA);
}
template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
tt_partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(),
                                tt_thrfrg_SFB(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  auto partition_SFB = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(partition_SFB);
}
template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto tt_get_layoutSFA_TV(TiledMma& mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto atile = make_tile(_,
                make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                      make_stride(               Int<1>{} ,                Int<0>{} )),
                          _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return tt_thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}
template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto tt_get_layoutSFB_TV(TiledMma& mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto btile = make_tile(_,
                make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                      make_stride(               Int<0>{} ,                Int<1>{} )),
                          _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return tt_thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

using AtomMX = SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, cutlass::float_ue8m0_t, 32>;
using ElemSF_MX = cutlass::float_ue8m0_t;
using ElemFP8   = uint8_t;
static constexpr int K_MX  = 32;
static constexpr int VEC_MX = 32;

// NVFP4 (Phase-1) atom + constants — RECIPE B dual-phase TMA (round-5 T1).
using AtomNV = SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t, 16>;
using ElemSF_NV = cutlass::float_ue4m3_t;    // 8-bit
using ElemFP4   = cute::uint4_t;              // raw 4-bit code (bit-copied)
static constexpr int K_NV  = 64;
static constexpr int VEC_NV = 16;

// Compact, UNSWIZZLED (SW_NONE) ring stage: (ROWS, CHB bytes, STAGES), row-major.
// TMA writes this box directly (no swizzle -> no SU4 vectorization constraint);
// the proven consume4/consume8 ldmatrix paths read it with a plain compact stride.
template <int ROWS, int CHB, int STAGES>
using SmemLayoutFlat = decltype(make_layout(
    make_shape(Int<ROWS>{}, Int<CHB>{}, Int<STAGES>{}),
    make_stride(Int<CHB>{}, Int<1>{}, Int<ROWS * CHB>{})));

static inline void check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// ---- mbarrier helpers ----
__device__ __forceinline__ void mbar_init(uint64_t* m, uint32_t c) {
  uint32_t a = cast_smem_ptr_to_uint(m);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(c));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* m) {
  uint32_t a = cast_smem_ptr_to_uint(m);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(a));
}
__device__ __forceinline__ void mbar_expect(uint64_t* m, uint32_t bytes) {
  uint32_t a = cast_smem_ptr_to_uint(m);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* m, uint32_t parity) {
  uint32_t a = cast_smem_ptr_to_uint(m);
  asm volatile("{\n .reg .pred P;\n LAB:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
               "@P bra DONE;\n bra LAB;\n DONE:\n}\n" :: "r"(a), "r"(parity));
}

// Width-generic coalesced SF row copy (round-6: makes BK/KMUL a real axis).
// NB = SF bytes/row/chunk; picks the widest aligned vector type. For KMUL=2 this
// is byte-identical to the round-5 hardcoded uint64(fp4)/uint16(fp8) path.
template <int NB>
__device__ __forceinline__ void sf_copy_row(uint8_t* dst, const uint8_t* src, bool valid) {
  if constexpr (NB == 1) {
    *dst = valid ? *src : (uint8_t)0;
  } else if constexpr (NB == 2) {
    *reinterpret_cast<uint16_t*>(dst) = valid ? *reinterpret_cast<const uint16_t*>(src) : (uint16_t)0;
  } else if constexpr (NB == 4) {
    *reinterpret_cast<uint32_t*>(dst) = valid ? *reinterpret_cast<const uint32_t*>(src) : 0u;
  } else if constexpr (NB == 8) {
    *reinterpret_cast<uint64_t*>(dst) = valid ? *reinterpret_cast<const uint64_t*>(src) : 0ull;
  } else {  // NB == 16
    uint64_t a = 0ull, b = 0ull;
    if (valid) { a = reinterpret_cast<const uint64_t*>(src)[0]; b = reinterpret_cast<const uint64_t*>(src)[1]; }
    reinterpret_cast<uint64_t*>(dst)[0] = a;
    reinterpret_cast<uint64_t*>(dst)[1] = b;
  }
}

// MXFP8 chunk K and swizzle: KCH = K_MX*KMUL elements (= bytes, uint8).
template <int KMUL> using SmemAtomMX =
    decltype(cutlass::gemm::collective::detail::
             sm120_rr_smem_selector<uint8_t, Int<K_MX * KMUL>>());

template <int BM, int BN, int STAGES, int KMUL>
using SmemLayoutA_t = decltype(tile_to_shape(
    SmemAtomMX<KMUL>{}, make_shape(Int<BM>{}, Int<K_MX * KMUL>{}, Int<STAGES>{})));
template <int BM, int BN, int STAGES, int KMUL>
using SmemLayoutB_t = decltype(tile_to_shape(
    SmemAtomMX<KMUL>{}, make_shape(Int<BN>{}, Int<K_MX * KMUL>{}, Int<STAGES>{})));

// ============================================================================
// Phase-2 (MXFP8) warp-specialized kernel with TMA data loads.
// ============================================================================
template <int BM, int BN, int WM, int WN, int STAGES, int KMUL, int MINB, int SWZ, class TmaA, class TmaB>
__global__ void __launch_bounds__(WM * WN * 32 + 32, MINB)
b1_tma_mx_kernel(CUTE_GRID_CONSTANT TmaA const tma_a,
                 CUTE_GRID_CONSTANT TmaB const tma_b,
                 const uint8_t* __restrict__ SFAh, const uint8_t* __restrict__ SFBh,
                 const float* __restrict__ s_x, const float* __restrict__ s_w,
                 __half* __restrict__ D, int M, int N, int K_high) {
  constexpr int NT_C  = WM * WN * 32;
  constexpr int NT_P  = 32;                // one producer warp
  constexpr int KCH   = K_MX * KMUL;       // elems/bytes per chunk
  constexpr int NBLK8 = KMUL;              // SF blocks/row/chunk
  constexpr int SFA_ST = BM * NBLK8;
  constexpr int SFB_ST = BN * NBLK8;

  using LA = SmemLayoutA_t<BM, BN, STAGES, KMUL>;
  using LB = SmemLayoutB_t<BM, BN, STAGES, KMUL>;
  __shared__ alignas(1024) uint8_t sA_b[cosize_v<LA>];
  __shared__ alignas(1024) uint8_t sB_b[cosize_v<LB>];
  __shared__ alignas(4) uint8_t sSFA_b[STAGES * SFA_ST];
  __shared__ alignas(4) uint8_t sSFB_b[STAGES * SFB_ST];
  __shared__ alignas(8) uint64_t mbar[2 * STAGES];

  const int tid = threadIdx.x;
  // CTA rasterization for L2 reuse (T3). swz_group<=0: native 2D grid.
  // swz_group=G>0: 1D grid, Triton-style group-M ordering — within a panel of G
  // consecutive M-tiles, sweep all N-tiles (keeps G A-tiles hot across N; shrinks
  // the per-wave working set vs the native all-M sweep on large N).
  int pm, pn;
  if constexpr (SWZ <= 0) {
    pm = blockIdx.x; pn = blockIdx.y;
  } else {
    const int num_m = (M + BM - 1) / BM;
    const int num_n = (N + BN - 1) / BN;
    const int pid = blockIdx.x;
    const int in_group = SWZ * num_n;
    const int gid = pid / in_group;
    const int first_m = gid * SWZ;
    const int gsize = min(num_m - first_m, SWZ);
    pm = first_m + (pid % gsize);
    pn = (pid % in_group) / gsize;
  }
  const int m0 = pm * BM;
  const int n0 = pn * BN;
  const int n2 = K_high / KCH;             // chunks
  const int Knb = K_high / VEC_MX;         // gmem SF blocks/row

  Tensor sA = make_tensor(make_smem_ptr(sA_b), LA{});   // (BM,KCH,STAGES) swizzled
  Tensor sB = make_tensor(make_smem_ptr(sB_b), LB{});

  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K_high));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K_high));
  auto cta_a = tma_a.get_slice(Int<0>{});
  auto cta_b = tma_b.get_slice(Int<0>{});
  constexpr uint32_t TX = (uint32_t)(BM * KCH + BN * KCH);  // bytes/stage (uint8)

  if (tid == 0) {
    CUTE_UNROLL
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&mbar[s], NT_P);            // full: producer-warp arrives + tx bytes
      mbar_init(&mbar[STAGES + s], NT_C);   // free
    }
  }
  __syncthreads();

  if (tid >= NT_C) {
    const int ptid = tid - NT_C;            // 0..31
    for (int g = 0; g < n2; ++g) {
      const int s = g % STAGES;
      if (g >= STAGES) mbar_wait(&mbar[STAGES + s], ((g / STAGES) - 1) & 1);
      // SF plain stores (cooperative over the producer warp)
      uint8_t* dSFA = sSFA_b + s * SFA_ST;
      uint8_t* dSFB = sSFB_b + s * SFB_ST;
      const int kb0 = g * NBLK8;
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
      if (ptid == 0) {
        Tensor gA_g = local_tile(mA, make_shape(Int<BM>{}, Int<KCH>{}), make_coord(pm, g)); // (BM,KCH)
        Tensor gB_g = local_tile(mB, make_shape(Int<BN>{}, Int<KCH>{}), make_coord(pn, g));
        Tensor tAgA = cta_a.partition_S(gA_g);
        Tensor tAsA = cta_a.partition_D(sA(_, _, s));
        Tensor tBgB = cta_b.partition_S(gB_g);
        Tensor tBsB = cta_b.partition_D(sB(_, _, s));
        mbar_expect(&mbar[s], TX);
        copy(tma_a.with(mbar[s]), tAgA, tAsA);
        copy(tma_b.with(mbar[s]), tBgB, tBsB);
      } else {
        mbar_arrive(&mbar[s]);
      }
    }
    return;
  }

  // ---------------- consumers ----------------
  using TiledMX = decltype(make_tiled_mma(AtomMX{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{}));
  TiledMX mma_mx;
  Tensor accum = partition_fragment_C(mma_mx, Shape<Int<BM>, Int<BN>>{});
  clear(accum);
  auto thr_mx = mma_mx.get_thread_slice(tid);
  auto tile_mx = tile_shape(mma_mx);
  auto tcopy_SFA8 = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF_MX>{},
                                         tt_get_layoutSFA_TV(mma_mx),
                                         make_shape(size<0>(tile_mx), size<2>(tile_mx)));
  auto tcopy_SFB8 = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF_MX>{},
                                         tt_get_layoutSFB_TV(mma_mx),
                                         make_shape(size<1>(tile_mx), size<2>(tile_mx)));
  auto tcopy_A8 = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, ElemFP8>{}, mma_mx);
  auto tcopy_B8 = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, ElemFP8>{}, mma_mx);

  for (int g = 0; g < n2; ++g) {
    const int s = g % STAGES;
    mbar_wait(&mbar[s], (g / STAGES) & 1);
    Tensor sAs = sA(_, _, s);    // (BM,KCH) swizzled
    Tensor sBs = sB(_, _, s);
    Tensor sSFA = make_tensor(recast_ptr<ElemSF_MX>(make_smem_ptr(sSFA_b + s * SFA_ST)),
                              make_layout(make_shape(Int<BM>{}, make_shape(Int<VEC_MX>{}, Int<NBLK8>{})),
                                          make_stride(Int<NBLK8>{}, make_stride(Int<0>{}, Int<1>{}))));
    Tensor sSFB = make_tensor(recast_ptr<ElemSF_MX>(make_smem_ptr(sSFB_b + s * SFB_ST)),
                              make_layout(make_shape(Int<BN>{}, make_shape(Int<VEC_MX>{}, Int<NBLK8>{})),
                                          make_stride(Int<NBLK8>{}, make_stride(Int<0>{}, Int<1>{}))));
    Tensor tCrA = thr_mx.partition_fragment_A(sAs);
    Tensor tCrB = thr_mx.partition_fragment_B(sBs);
    copy(tcopy_A8, tcopy_A8.get_thread_slice(tid).partition_S(sAs),
         tcopy_A8.get_thread_slice(tid).retile_D(tCrA));
    copy(tcopy_B8, tcopy_B8.get_thread_slice(tid).partition_S(sBs),
         tcopy_B8.get_thread_slice(tid).retile_D(tCrB));
    fp4_shift_A(typename TiledMX::MMA_Op{}, tCrA);
    fp4_shift_B(typename TiledMX::MMA_Op{}, tCrB);
    Tensor tCrSFA = tt_partition_fragment_SFA(sSFA, thr_mx);
    Tensor tCrSFB = tt_partition_fragment_SFB(sSFB, thr_mx);
    copy(tcopy_SFA8.get_thread_slice(tid).partition_S(sSFA),
         tcopy_SFA8.get_thread_slice(tid).retile_D(tCrSFA));
    copy(tcopy_SFB8.get_thread_slice(tid).partition_S(sSFB),
         tcopy_SFB8.get_thread_slice(tid).retile_D(tCrSFB));
    mbar_arrive(&mbar[STAGES + s]);
    CUTE_UNROLL
    for (int kb = 0; kb < KMUL; ++kb)
      cute::gemm(mma_mx, make_zip_tensor(tCrA(_,_,kb), tCrSFA(_,_,kb)),
                 make_zip_tensor(tCrB(_,_,kb), tCrSFB(_,_,kb)), accum);
  }

  // Epilogue: accum * s_x[m] * s_w[n] -> fp16 (direct half2 store).
  Tensor cD = thr_mx.partition_C(make_identity_tensor(Shape<Int<BM>, Int<BN>>{}));
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
// Dual-phase (NVFP4 + MXFP8) TMA kernel — RECIPE B (round-5 T1).
// Both phases TMA-load A/B DATA into a COMPACT (SW_NONE, unswizzled) smem ring
// (CHB = 32*KMUL bytes/row/chunk, identical for both phases), consumed by the
// PROVEN non-TMA ldmatrix paths: consume4 (uint4_t view + SM75_U32x4_LDSM_N) for
// NVFP4, consume8 (uint8 view + SM75) for MXFP8. SW_NONE sidesteps the
// SM100_SU4_DU8x16 swizzle-vectorization blocker that killed round-4's swizzled
// fp4 TMA. One elected producer thread issues both TMAs/stage; the mbarrier ring
// serializes the phase boundary (stage s consumed by the same g that loaded it,
// phase selected by g<n1). SF stays on the plain-store path (union-sized for fp4).
// ============================================================================
template <int BM, int BN, int WM, int WN, int STAGES, int KMUL, int MINB, int SWZ,
          int PERSIST, int KPIPE, class TmaAl, class TmaBl, class TmaAh, class TmaBh>
__global__ void __launch_bounds__(WM * WN * 32 + 32, MINB)
b1_tma_dual_kernel(CUTE_GRID_CONSTANT TmaAl const tma_al,
                   CUTE_GRID_CONSTANT TmaBl const tma_bl,
                   CUTE_GRID_CONSTANT TmaAh const tma_ah,
                   CUTE_GRID_CONSTANT TmaBh const tma_bh,
                   const uint8_t* __restrict__ SFAl, const uint8_t* __restrict__ SFBl,
                   const uint8_t* __restrict__ SFAh, const uint8_t* __restrict__ SFBh,
                   const float* __restrict__ s_x, const float* __restrict__ s_w,
                   __half* __restrict__ D, int M, int N, int K_low, int K_high) {
  constexpr int NT_C  = WM * WN * 32;
  constexpr int NT_P  = 32;                 // one producer warp
  constexpr int CHB   = K_MX * KMUL;        // bytes/row/chunk (both phases)
  constexpr int KCH4  = K_NV * KMUL;        // fp4 codes/chunk  (= 2*CHB)
  constexpr int KCH8  = K_MX * KMUL;        // fp8 bytes/chunk  (= CHB)
  constexpr int NBLK4 = 4 * KMUL;           // fp4 SF bytes/row/chunk
  constexpr int NBLK8 = KMUL;               // fp8 SF bytes/row/chunk
  constexpr int A_ST   = BM * CHB;          // ring stage stride (bytes)
  constexpr int B_ST   = BN * CHB;
  constexpr int SFA_ST = BM * NBLK4;        // union SF buffer sized for fp4
  constexpr int SFB_ST = BN * NBLK4;

  // SW64-swizzled ring (same layout the zero-bank-conflict Phase-2 TMA kernel
  // uses). CHB = K_MX*KMUL = 64 bytes/row/chunk is identical for fp4 and fp8, so
  // ONE SW64 atom serves both phases; consumers read it via recast (fp4) / direct
  // (fp8). Round-5 T1 fix: SW_NONE flat ring caused 18-38M ldmatrix bank conflicts.
  using LA = SmemLayoutA_t<BM, BN, STAGES, KMUL>;
  using LB = SmemLayoutB_t<BM, BN, STAGES, KMUL>;
  __shared__ alignas(1024) uint8_t sA_b[cosize_v<LA>];
  __shared__ alignas(1024) uint8_t sB_b[cosize_v<LB>];
  __shared__ alignas(16) uint8_t sSFA_b[STAGES * SFA_ST];
  __shared__ alignas(16) uint8_t sSFB_b[STAGES * SFB_ST];
  __shared__ alignas(8) uint64_t mbar[2 * STAGES];

  const int tid = threadIdx.x;
  // CTA rasterization for L2 reuse (round-6 gate attack). SWZ<=0: native 2D grid.
  // SWZ=G>0: 1D grid, Triton-style group-M ordering (G consecutive M-tiles sweep
  // all N-tiles, keeping G A-tiles hot across the 64 N-tiles of the gate row).
  int pm = 0, pn = 0;
  if constexpr (!PERSIST) {           // PERSIST=1 decodes (pm,pn) per-tile inside the loop
  if constexpr (SWZ <= 0) {
    pm = blockIdx.x; pn = blockIdx.y;
  } else {
    const int num_m = (M + BM - 1) / BM;
    const int num_n = (N + BN - 1) / BN;
    const int pid = blockIdx.x;
    const int in_group = SWZ * num_n;
    const int gid = pid / in_group;
    const int first_m = gid * SWZ;
    const int gsize = min(num_m - first_m, SWZ);
    pm = first_m + (pid % gsize);
    pn = (pid % in_group) / gsize;
  }
  }
  const int m0 = pm * BM, n0 = pn * BN;
  const int n1 = K_low / KCH4;              // fp4 chunks
  const int n2 = K_high / KCH8;             // fp8 chunks
  const int ntot = n1 + n2;

  Tensor sA = make_tensor(make_smem_ptr(sA_b), LA{});   // (BM,CHB,STAGES) compact
  Tensor sB = make_tensor(make_smem_ptr(sB_b), LB{});
  Tensor mAl = tma_al.get_tma_tensor(make_shape(M, K_low / 2));
  Tensor mBl = tma_bl.get_tma_tensor(make_shape(N, K_low / 2));
  Tensor mAh = tma_ah.get_tma_tensor(make_shape(M, K_high));
  Tensor mBh = tma_bh.get_tma_tensor(make_shape(N, K_high));
  auto cta_al = tma_al.get_slice(Int<0>{});
  auto cta_bl = tma_bl.get_slice(Int<0>{});
  auto cta_ah = tma_ah.get_slice(Int<0>{});
  auto cta_bh = tma_bh.get_slice(Int<0>{});
  constexpr uint32_t TX = (uint32_t)(BM * CHB + BN * CHB);

  if (tid == 0) {
    CUTE_UNROLL
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&mbar[s], NT_P);
      mbar_init(&mbar[STAGES + s], NT_C);
    }
  }
  __syncthreads();

  // Persistent-scheduler tile iteration (round-7, maps StaticPersistentTileSchedulerSm90):
  // grid = #CTAs, each CTA strides over output tiles; ring parity uses a CONTINUOUS
  // giter so the pipeline fill is paid ONCE per CTA (not per tile) and the producer runs
  // ahead across tile boundaries (hides the next tile's fill behind the prev epilogue).
  const int num_m = (M + BM - 1) / BM;
  const int num_n = (N + BN - 1) / BN;
  const int num_tiles = num_m * num_n;
#define DECODE_TILE(TIX, PPM, PPN) do {                                          \
    if constexpr (SWZ <= 0) { (PPM) = (TIX) / num_n; (PPN) = (TIX) % num_n; }    \
    else { int ig_ = SWZ * num_n; int gid_ = (TIX) / ig_; int fm_ = gid_ * SWZ; \
           int gs_ = min(num_m - fm_, SWZ);                                      \
           (PPM) = fm_ + (((TIX) % ig_) % gs_); (PPN) = ((TIX) % ig_) / gs_; }   \
  } while (0)

  if (tid >= NT_C) {
    const int ptid = tid - NT_C;
    // Load one K-chunk g of tile (ppm,ppn) into ring stage s. pm0/pn0 = ppm*BM/ppn*BN.
    auto prod = [&](int g, int s, int ppm, int ppn, int pm0, int pn0) {
      const bool f4 = (g < n1);
      const int c = f4 ? g : (g - n1);
      uint8_t* dSFA = sSFA_b + s * SFA_ST;
      uint8_t* dSFB = sSFB_b + s * SFB_ST;
      if (f4) {
        // Vectorized SF load (round-5 T1): NBLK4=8 contiguous SF bytes/row ->
        // one coalesced uint64/row (vs 8x 1-byte scalar). Separate gmem->reg and
        // reg->smem across the unrolled rows gives memory-level parallelism,
        // cutting the long-scoreboard stall that dominated the fp4 phase.
        const int Knb = K_low / VEC_NV;
        const size_t off = (size_t)c * NBLK4;
        for (int row = ptid; row < BM; row += NT_P) {
          int mg = pm0 + row;
          sf_copy_row<NBLK4>(dSFA + (size_t)row * NBLK4, SFAl + (size_t)mg * Knb + off, mg < M);
        }
        for (int row = ptid; row < BN; row += NT_P) {
          int ng = pn0 + row;
          sf_copy_row<NBLK4>(dSFB + (size_t)row * NBLK4, SFBl + (size_t)ng * Knb + off, ng < N);
        }
      } else {
        const int Knb = K_high / VEC_MX;
        const size_t off = (size_t)c * NBLK8;
        for (int row = ptid; row < BM; row += NT_P) {
          int mg = pm0 + row;
          sf_copy_row<NBLK8>(dSFA + (size_t)row * NBLK8, SFAh + (size_t)mg * Knb + off, mg < M);
        }
        for (int row = ptid; row < BN; row += NT_P) {
          int ng = pn0 + row;
          sf_copy_row<NBLK8>(dSFB + (size_t)row * NBLK8, SFBh + (size_t)ng * Knb + off, ng < N);
        }
      }
      if (ptid == 0) {
        mbar_expect(&mbar[s], TX);
        if (f4) {
          Tensor gA = local_tile(mAl, make_shape(Int<BM>{}, Int<CHB>{}), make_coord(ppm, c));
          Tensor gB = local_tile(mBl, make_shape(Int<BN>{}, Int<CHB>{}), make_coord(ppn, c));
          copy(tma_al.with(mbar[s]), cta_al.partition_S(gA), cta_al.partition_D(sA(_, _, s)));
          copy(tma_bl.with(mbar[s]), cta_bl.partition_S(gB), cta_bl.partition_D(sB(_, _, s)));
        } else {
          Tensor gA = local_tile(mAh, make_shape(Int<BM>{}, Int<CHB>{}), make_coord(ppm, c));
          Tensor gB = local_tile(mBh, make_shape(Int<BN>{}, Int<CHB>{}), make_coord(ppn, c));
          copy(tma_ah.with(mbar[s]), cta_ah.partition_S(gA), cta_ah.partition_D(sA(_, _, s)));
          copy(tma_bh.with(mbar[s]), cta_bh.partition_S(gB), cta_bh.partition_D(sB(_, _, s)));
        }
      } else {
        mbar_arrive(&mbar[s]);
      }
    };
    if constexpr (PERSIST) {
      int giter = 0;
      for (int tix = blockIdx.x; tix < num_tiles; tix += gridDim.x) {
        int ppm, ppn; DECODE_TILE(tix, ppm, ppn);
        const int pm0 = ppm * BM, pn0 = ppn * BN;
        for (int g = 0; g < ntot; ++g, ++giter) {
          const int s = giter % STAGES;
          if (giter >= STAGES) mbar_wait(&mbar[STAGES + s], ((giter / STAGES) - 1) & 1);
          prod(g, s, ppm, ppn, pm0, pn0);
        }
      }
    } else {
      for (int g = 0; g < ntot; ++g) {
        const int s = g % STAGES;
        if (g >= STAGES) mbar_wait(&mbar[STAGES + s], ((g / STAGES) - 1) & 1);
        prod(g, s, pm, pn, m0, n0);
      }
    }
    return;
  }

  // ---------------- consumers ----------------
  using TiledNV = decltype(make_tiled_mma(AtomNV{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{}));
  using TiledMX = decltype(make_tiled_mma(AtomMX{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{}));
  TiledNV mma_nv; TiledMX mma_mx;
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
      tt_get_layoutSFA_TV(mma_nv), make_shape(size<0>(tile_nv), size<2>(tile_nv)));
  auto tcopy_SFB4 = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF_NV>{},
      tt_get_layoutSFB_TV(mma_nv), make_shape(size<1>(tile_nv), size<2>(tile_nv)));
  auto tcopy_SFA8 = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF_MX>{},
      tt_get_layoutSFA_TV(mma_mx), make_shape(size<0>(tile_mx), size<2>(tile_mx)));
  auto tcopy_SFB8 = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF_MX>{},
      tt_get_layoutSFB_TV(mma_mx), make_shape(size<1>(tile_mx), size<2>(tile_mx)));
  auto tcopy_A4 = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, ElemFP4>{}, mma_nv);
  auto tcopy_B4 = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, ElemFP4>{}, mma_nv);
  auto tcopy_A8 = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, ElemFP8>{}, mma_mx);
  auto tcopy_B8 = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, ElemFP8>{}, mma_mx);

  auto consume4 = [&](int s) {
    Tensor sAv = recast<ElemFP4>(sA(_, _, s));   // SW64-swizzled uint8 ring -> fp4 view
    Tensor sBv = recast<ElemFP4>(sB(_, _, s));
    Tensor sSFA = make_tensor(recast_ptr<ElemSF_NV>(make_smem_ptr(sSFA_b + s * SFA_ST)),
        make_layout(make_shape(Int<BM>{}, make_shape(Int<VEC_NV>{}, Int<NBLK4>{})),
                    make_stride(Int<NBLK4>{}, make_stride(Int<0>{}, Int<1>{}))));
    Tensor sSFB = make_tensor(recast_ptr<ElemSF_NV>(make_smem_ptr(sSFB_b + s * SFB_ST)),
        make_layout(make_shape(Int<BN>{}, make_shape(Int<VEC_NV>{}, Int<NBLK4>{})),
                    make_stride(Int<NBLK4>{}, make_stride(Int<0>{}, Int<1>{}))));
    Tensor tCrA = thr_nv.partition_fragment_A(sAv);
    Tensor tCrB = thr_nv.partition_fragment_B(sBv);
    copy(tcopy_A4, tcopy_A4.get_thread_slice(tid).partition_S(sAv),
         tcopy_A4.get_thread_slice(tid).retile_D(tCrA));
    copy(tcopy_B4, tcopy_B4.get_thread_slice(tid).partition_S(sBv),
         tcopy_B4.get_thread_slice(tid).retile_D(tCrB));
    fp4_shift_A(typename TiledNV::MMA_Op{}, tCrA);
    fp4_shift_B(typename TiledNV::MMA_Op{}, tCrB);
    Tensor tCrSFA = tt_partition_fragment_SFA(sSFA, thr_nv);
    Tensor tCrSFB = tt_partition_fragment_SFB(sSFB, thr_nv);
    copy(tcopy_SFA4.get_thread_slice(tid).partition_S(sSFA),
         tcopy_SFA4.get_thread_slice(tid).retile_D(tCrSFA));
    copy(tcopy_SFB4.get_thread_slice(tid).partition_S(sSFB),
         tcopy_SFB4.get_thread_slice(tid).retile_D(tCrSFB));
    mbar_arrive(&mbar[STAGES + s]);     // T2a early stage release (frags in regs)
    CUTE_UNROLL
    for (int kb = 0; kb < KMUL; ++kb)
      cute::gemm(mma_nv, make_zip_tensor(tCrA(_,_,kb), tCrSFA(_,_,kb)),
                 make_zip_tensor(tCrB(_,_,kb), tCrSFB(_,_,kb)), accum);
  };
  auto consume8 = [&](int s) {
    Tensor sAv = sA(_, _, s);   // SW64-swizzled uint8 ring (fp8 = uint8, direct)
    Tensor sBv = sB(_, _, s);
    Tensor sSFA = make_tensor(recast_ptr<ElemSF_MX>(make_smem_ptr(sSFA_b + s * SFA_ST)),
        make_layout(make_shape(Int<BM>{}, make_shape(Int<VEC_MX>{}, Int<NBLK8>{})),
                    make_stride(Int<NBLK8>{}, make_stride(Int<0>{}, Int<1>{}))));
    Tensor sSFB = make_tensor(recast_ptr<ElemSF_MX>(make_smem_ptr(sSFB_b + s * SFB_ST)),
        make_layout(make_shape(Int<BN>{}, make_shape(Int<VEC_MX>{}, Int<NBLK8>{})),
                    make_stride(Int<NBLK8>{}, make_stride(Int<0>{}, Int<1>{}))));
    Tensor tCrA = thr_mx.partition_fragment_A(sAv);
    Tensor tCrB = thr_mx.partition_fragment_B(sBv);
    copy(tcopy_A8, tcopy_A8.get_thread_slice(tid).partition_S(sAv),
         tcopy_A8.get_thread_slice(tid).retile_D(tCrA));
    copy(tcopy_B8, tcopy_B8.get_thread_slice(tid).partition_S(sBv),
         tcopy_B8.get_thread_slice(tid).retile_D(tCrB));
    fp4_shift_A(typename TiledMX::MMA_Op{}, tCrA);
    fp4_shift_B(typename TiledMX::MMA_Op{}, tCrB);
    Tensor tCrSFA = tt_partition_fragment_SFA(sSFA, thr_mx);
    Tensor tCrSFB = tt_partition_fragment_SFB(sSFB, thr_mx);
    copy(tcopy_SFA8.get_thread_slice(tid).partition_S(sSFA),
         tcopy_SFA8.get_thread_slice(tid).retile_D(tCrSFA));
    copy(tcopy_SFB8.get_thread_slice(tid).partition_S(sSFB),
         tcopy_SFB8.get_thread_slice(tid).retile_D(tCrSFB));
    mbar_arrive(&mbar[STAGES + s]);
    CUTE_UNROLL
    for (int kb = 0; kb < KMUL; ++kb)
      cute::gemm(mma_mx, make_zip_tensor(tCrA(_,_,kb), tCrSFA(_,_,kb)),
                 make_zip_tensor(tCrB(_,_,kb), tCrSFB(_,_,kb)), accum);
  };

  // Epilogue: accum * s_x[m] * s_w[n] -> fp16 (direct half2 store).
  Tensor cD = thr_nv.partition_C(make_identity_tensor(Shape<Int<BM>, Int<BN>>{}));
  auto epi = [&](int em0, int en0) {
    CUTE_UNROLL
    for (int i = 0; i < size(accum); i += 2) {
      int mg = em0 + get<0>(cD(i));
      int ng = en0 + get<1>(cD(i));
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
  };

  // ---- T1 (round-8): register K-block pipelined consumer (KPIPE=1). ----
  // Faithful port of CUTLASS sm120_blockscaled_mma_tma.hpp mma(): a persistent
  // register fragment with K_BLOCK_MAX(=KMUL) slots; prologue loads kb0, then the
  // steady loop issues copy_kblock(kb+1) (ldmatrix smem->rmem) BEFORE gemm_kblock(kb)
  // (MMA on the current slot), and at the chunk boundary (kb==KMUL-1) releases the
  // smem stage and prefetches kb0 of the NEXT chunk (which lives in the next ring
  // stage, already TMA-filled) — hiding smem->reg latency inside the math. Phases are
  // pipelined separately (uniform atom within a phase); the one phase-seam ldmatrix is
  // non-overlapped (round-7: seam<=0, negligible). Continuous giter keeps the mbar
  // ring parity identical to the non-pipelined path (producer unchanged).
  auto run_tile_kpipe = [&](int giter0) {
    // FP4 phase: local chunks c in [0, n1).
    if (n1 > 0) {
      Tensor sAv0 = recast<ElemFP4>(sA(_, _, 0));
      Tensor sBv0 = recast<ElemFP4>(sB(_, _, 0));
      Tensor tCrA = thr_nv.partition_fragment_A(sAv0);
      Tensor tCrB = thr_nv.partition_fragment_B(sBv0);
      auto rA = tcopy_A4.get_thread_slice(tid).retile_D(tCrA);
      auto rB = tcopy_B4.get_thread_slice(tid).retile_D(tCrB);
      auto sfA = [&](int s) { return make_tensor(recast_ptr<ElemSF_NV>(make_smem_ptr(sSFA_b + s * SFA_ST)),
          make_layout(make_shape(Int<BM>{}, make_shape(Int<VEC_NV>{}, Int<NBLK4>{})),
                      make_stride(Int<NBLK4>{}, make_stride(Int<0>{}, Int<1>{})))); };
      auto sfB = [&](int s) { return make_tensor(recast_ptr<ElemSF_NV>(make_smem_ptr(sSFB_b + s * SFB_ST)),
          make_layout(make_shape(Int<BN>{}, make_shape(Int<VEC_NV>{}, Int<NBLK4>{})),
                      make_stride(Int<NBLK4>{}, make_stride(Int<0>{}, Int<1>{})))); };
      Tensor tCrSFA = tt_partition_fragment_SFA(sfA(0), thr_nv);
      Tensor tCrSFB = tt_partition_fragment_SFB(sfB(0), thr_nv);
      auto rSFA = tcopy_SFA4.get_thread_slice(tid).retile_D(tCrSFA);
      auto rSFB = tcopy_SFB4.get_thread_slice(tid).retile_D(tCrSFB);
      auto load = [&](int s, int kb) {
        Tensor sAv = recast<ElemFP4>(sA(_, _, s));
        Tensor sBv = recast<ElemFP4>(sB(_, _, s));
        copy(tcopy_A4, tcopy_A4.get_thread_slice(tid).partition_S(sAv)(_, _, kb), rA(_, _, kb));
        copy(tcopy_B4, tcopy_B4.get_thread_slice(tid).partition_S(sBv)(_, _, kb), rB(_, _, kb));
        fp4_shift_A(typename TiledNV::MMA_Op{}, rA(_, _, kb));
        fp4_shift_B(typename TiledNV::MMA_Op{}, rB(_, _, kb));
        copy(tcopy_SFA4.get_thread_slice(tid).partition_S(sfA(s))(_, _, kb), rSFA(_, _, kb));
        copy(tcopy_SFB4.get_thread_slice(tid).partition_S(sfB(s))(_, _, kb), rSFB(_, _, kb));
      };
      auto mma = [&](int kb) {
        cute::gemm(mma_nv, make_zip_tensor(tCrA(_, _, kb), tCrSFA(_, _, kb)),
                   make_zip_tensor(tCrB(_, _, kb), tCrSFB(_, _, kb)), accum);
      };
      int s0 = giter0 % STAGES;
      mbar_wait(&mbar[s0], (giter0 / STAGES) & 1);
      load(s0, 0);
      for (int c = 0; c < n1; ++c) {
        int g = giter0 + c, sc = g % STAGES;
        CUTE_UNROLL
        for (int kb = 0; kb < KMUL; ++kb) {
          int kbn = (kb + 1 == KMUL) ? 0 : kb + 1;
          if (kb == KMUL - 1) {
            mbar_arrive(&mbar[STAGES + sc]);
            if (c + 1 < n1) { int gn = g + 1, sn = gn % STAGES;
              mbar_wait(&mbar[sn], (gn / STAGES) & 1); load(sn, 0); }
            mma(kb);
          } else { load(sc, kbn); mma(kb); }
        }
      }
    }
    // FP8 phase: local chunks c in [n1, ntot).
    if (n2 > 0) {
      Tensor sAv0 = sA(_, _, 0);
      Tensor sBv0 = sB(_, _, 0);
      Tensor tCrA = thr_mx.partition_fragment_A(sAv0);
      Tensor tCrB = thr_mx.partition_fragment_B(sBv0);
      auto rA = tcopy_A8.get_thread_slice(tid).retile_D(tCrA);
      auto rB = tcopy_B8.get_thread_slice(tid).retile_D(tCrB);
      auto sfA = [&](int s) { return make_tensor(recast_ptr<ElemSF_MX>(make_smem_ptr(sSFA_b + s * SFA_ST)),
          make_layout(make_shape(Int<BM>{}, make_shape(Int<VEC_MX>{}, Int<NBLK8>{})),
                      make_stride(Int<NBLK8>{}, make_stride(Int<0>{}, Int<1>{})))); };
      auto sfB = [&](int s) { return make_tensor(recast_ptr<ElemSF_MX>(make_smem_ptr(sSFB_b + s * SFB_ST)),
          make_layout(make_shape(Int<BN>{}, make_shape(Int<VEC_MX>{}, Int<NBLK8>{})),
                      make_stride(Int<NBLK8>{}, make_stride(Int<0>{}, Int<1>{})))); };
      Tensor tCrSFA = tt_partition_fragment_SFA(sfA(0), thr_mx);
      Tensor tCrSFB = tt_partition_fragment_SFB(sfB(0), thr_mx);
      auto rSFA = tcopy_SFA8.get_thread_slice(tid).retile_D(tCrSFA);
      auto rSFB = tcopy_SFB8.get_thread_slice(tid).retile_D(tCrSFB);
      auto load = [&](int s, int kb) {
        Tensor sAv = sA(_, _, s);
        Tensor sBv = sB(_, _, s);
        copy(tcopy_A8, tcopy_A8.get_thread_slice(tid).partition_S(sAv)(_, _, kb), rA(_, _, kb));
        copy(tcopy_B8, tcopy_B8.get_thread_slice(tid).partition_S(sBv)(_, _, kb), rB(_, _, kb));
        fp4_shift_A(typename TiledMX::MMA_Op{}, rA(_, _, kb));
        fp4_shift_B(typename TiledMX::MMA_Op{}, rB(_, _, kb));
        copy(tcopy_SFA8.get_thread_slice(tid).partition_S(sfA(s))(_, _, kb), rSFA(_, _, kb));
        copy(tcopy_SFB8.get_thread_slice(tid).partition_S(sfB(s))(_, _, kb), rSFB(_, _, kb));
      };
      auto mma = [&](int kb) {
        cute::gemm(mma_mx, make_zip_tensor(tCrA(_, _, kb), tCrSFA(_, _, kb)),
                   make_zip_tensor(tCrB(_, _, kb), tCrSFB(_, _, kb)), accum);
      };
      int g0 = giter0 + n1, s0 = g0 % STAGES;
      mbar_wait(&mbar[s0], (g0 / STAGES) & 1);
      load(s0, 0);
      for (int c = n1; c < ntot; ++c) {
        int g = giter0 + c, sc = g % STAGES;
        CUTE_UNROLL
        for (int kb = 0; kb < KMUL; ++kb) {
          int kbn = (kb + 1 == KMUL) ? 0 : kb + 1;
          if (kb == KMUL - 1) {
            mbar_arrive(&mbar[STAGES + sc]);
            if (c + 1 < ntot) { int gn = g + 1, sn = gn % STAGES;
              mbar_wait(&mbar[sn], (gn / STAGES) & 1); load(sn, 0); }
            mma(kb);
          } else { load(sc, kbn); mma(kb); }
        }
      }
    }
  };

  if constexpr (KPIPE) {
    if constexpr (PERSIST) {
      int giter = 0;
      for (int tix = blockIdx.x; tix < num_tiles; tix += gridDim.x) {
        int ppm, ppn; DECODE_TILE(tix, ppm, ppn);
        clear(accum);
        run_tile_kpipe(giter);
        giter += ntot;
        epi(ppm * BM, ppn * BN);
      }
    } else {
      run_tile_kpipe(0);
      epi(m0, n0);
    }
  } else if constexpr (PERSIST) {
    int giter = 0;
    for (int tix = blockIdx.x; tix < num_tiles; tix += gridDim.x) {
      int ppm, ppn; DECODE_TILE(tix, ppm, ppn);
      clear(accum);                                   // fresh accumulator per output tile
      for (int g = 0; g < ntot; ++g, ++giter) {
        const int s = giter % STAGES;
        mbar_wait(&mbar[s], (giter / STAGES) & 1);
        if (g < n1) consume4(s); else consume8(s);
      }
      epi(ppm * BM, ppn * BN);
    }
  } else {
    for (int g = 0; g < ntot; ++g) {
      const int s = g % STAGES;
      mbar_wait(&mbar[s], (g / STAGES) & 1);
      if (g < n1) consume4(s); else consume8(s);
    }
    epi(m0, n0);
  }
#undef DECODE_TILE
}

// ============================================================================
// Host entry — dual-phase (NVFP4 + MXFP8) TMA kernel (RECIPE B).
// ============================================================================
template <int BM, int BN, int WM, int WN, int STAGES, int KMUL, int MINB = 1, int SWZ = 0, int PERSIST = 0, int KPIPE = 0>
static void launch_tma_dual(const torch::Tensor& Al, const torch::Tensor& SFAl,
                            const torch::Tensor& Bl, const torch::Tensor& SFBl,
                            const torch::Tensor& Ah, const torch::Tensor& SFAh,
                            const torch::Tensor& Bh, const torch::Tensor& SFBh,
                            const torch::Tensor& s_x, const torch::Tensor& s_w,
                            torch::Tensor& D, int M, int N, int K_low, int K_high,
                            cudaStream_t stream) {
  constexpr int CHB = K_MX * KMUL;
  using LA = SmemLayoutA_t<BM, BN, STAGES, KMUL>;   // SW64-swizzled (round-5 T1 fix)
  using LB = SmemLayoutB_t<BM, BN, STAGES, KMUL>;
  // fp4 gmem is packed [M, K_low/2] bytes; fp8 gmem is [M, K_high] bytes.
  // When a phase is empty (K_low==0 or K_high==0) build a never-read placeholder
  // descriptor off the other phase's pointer so make_tma_copy stays valid.
  const int Kl2 = (K_low > 0) ? K_low / 2 : CHB;
  const int Khb = (K_high > 0) ? K_high : CHB;
  const uint8_t* pAl = (K_low  > 0) ? Al.data_ptr<uint8_t>() : Ah.data_ptr<uint8_t>();
  const uint8_t* pBl = (K_low  > 0) ? Bl.data_ptr<uint8_t>() : Bh.data_ptr<uint8_t>();
  const uint8_t* pAh = (K_high > 0) ? Ah.data_ptr<uint8_t>() : Al.data_ptr<uint8_t>();
  const uint8_t* pBh = (K_high > 0) ? Bh.data_ptr<uint8_t>() : Bl.data_ptr<uint8_t>();
  Tensor mAl = make_tensor(make_gmem_ptr(pAl), make_shape(M, Kl2), make_stride(Kl2, _1{}));
  Tensor mBl = make_tensor(make_gmem_ptr(pBl), make_shape(N, Kl2), make_stride(Kl2, _1{}));
  Tensor mAh = make_tensor(make_gmem_ptr(pAh), make_shape(M, Khb), make_stride(Khb, _1{}));
  Tensor mBh = make_tensor(make_gmem_ptr(pBh), make_shape(N, Khb), make_stride(Khb, _1{}));
  auto tma_al = make_tma_copy(SM90_TMA_LOAD{}, mAl, LA{}(_, _, Int<0>{}),
                              make_shape(Int<BM>{}, Int<CHB>{}), _1{});
  auto tma_bl = make_tma_copy(SM90_TMA_LOAD{}, mBl, LB{}(_, _, Int<0>{}),
                              make_shape(Int<BN>{}, Int<CHB>{}), _1{});
  auto tma_ah = make_tma_copy(SM90_TMA_LOAD{}, mAh, LA{}(_, _, Int<0>{}),
                              make_shape(Int<BM>{}, Int<CHB>{}), _1{});
  auto tma_bh = make_tma_copy(SM90_TMA_LOAD{}, mBh, LB{}(_, _, Int<0>{}),
                              make_shape(Int<BN>{}, Int<CHB>{}), _1{});
  const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN;
  auto kptr = &b1_tma_dual_kernel<BM, BN, WM, WN, STAGES, KMUL, MINB, SWZ, PERSIST, KPIPE,
                                  decltype(tma_al), decltype(tma_bl), decltype(tma_ah), decltype(tma_bh)>;
  dim3 grid;
  if constexpr (PERSIST) {
    // Persistent grid = min(#tiles, #SM * max_active_blocks). Each CTA strides over
    // tiles (StaticPersistentTileSchedulerSm90 semantics). Occupancy is typically 1
    // (smem-bound ~99KB), so grid ~= #SM.
    int dev = 0; cudaGetDevice(&dev);
    int nsm = 1; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    int occ = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kptr, WM * WN * 32 + 32, 0);
    if (occ < 1) occ = 1;
    long cap = (long)nsm * occ;
    long nt = (long)num_m * num_n;
    int g = (int)(nt < cap ? nt : cap);
    if (g < 1) g = 1;
    grid = dim3(g);
  } else {
    grid = (SWZ > 0) ? dim3(num_m * num_n) : dim3(num_m, num_n);
  }
  b1_tma_dual_kernel<BM, BN, WM, WN, STAGES, KMUL, MINB, SWZ, PERSIST, KPIPE,
                     decltype(tma_al), decltype(tma_bl), decltype(tma_ah), decltype(tma_bh)>
      <<<grid, WM * WN * 32 + 32, 0, stream>>>(
          tma_al, tma_bl, tma_ah, tma_bh,
          SFAl.data_ptr<uint8_t>(), SFBl.data_ptr<uint8_t>(),
          SFAh.data_ptr<uint8_t>(), SFBh.data_ptr<uint8_t>(),
          s_x.data_ptr<float>(), s_w.data_ptr<float>(),
          reinterpret_cast<__half*>(D.data_ptr<at::Half>()), M, N, K_low, K_high);
}

torch::Tensor b1_fused_tma_dual(torch::Tensor Al, torch::Tensor SFAl,
                                torch::Tensor Bl, torch::Tensor SFBl,
                                torch::Tensor Ah, torch::Tensor SFAh,
                                torch::Tensor Bh, torch::Tensor SFBh,
                                torch::Tensor s_x, torch::Tensor s_w,
                                int64_t M, int64_t N, int64_t K_low, int64_t K_high,
                                int64_t stages, int64_t tile,
                                int64_t swz, int64_t kmul, int64_t minb,
                                int64_t persist, int64_t kpipe) {
  TORCH_CHECK(Al.is_cuda() && Bl.is_cuda() && Ah.is_cuda() && Bh.is_cuda(), "cuda");
  const int km = (int)kmul;
  TORCH_CHECK(km == 1 || km == 2 || km == 4, "kmul must be 1, 2 or 4");
  TORCH_CHECK(K_low  % (km * K_NV) == 0, "K_low must be divisible by kmul*64");
  TORCH_CHECK(K_high % (km * K_MX) == 0, "K_high must be divisible by kmul*32");
  Al = Al.contiguous(); Bl = Bl.contiguous(); SFAl = SFAl.contiguous(); SFBl = SFBl.contiguous();
  Ah = Ah.contiguous(); Bh = Bh.contiguous(); SFAh = SFAh.contiguous(); SFBh = SFBh.contiguous();
  s_x = s_x.contiguous(); s_w = s_w.contiguous();
  auto dev = Al.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto D = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(dev));
  int st = (int)stages; if (st == 0) st = 4;
  const int sw = (int)swz, mb = (int)minb, pst = (int)persist, kp = (int)kpipe;
  const int Mi = M, Ni = N, Kl = K_low, Kh = K_high;

  // ----- round-8 KPIPE=1 (register K-block pipelined consumer). Curated instantiations
  // (km2 mb1): tile {0,6} x st3 x sw {0,8} x persist {0,1}. Covers per-shape dispatch
  // (qkv/down non-persist t0/t6 sw0; gate persist sw8) + compare cells for the sweep.
#define LDK(BM,BN,WM,WN,ST,KM,MB,SW,PS) do { \
    launch_tma_dual<BM,BN,WM,WN,ST,KM,MB,SW,PS,1>(Al,SFAl,Bl,SFBl,Ah,SFAh,Bh,SFBh,s_x,s_w,D,Mi,Ni,Kl,Kh,stream); \
    check_cuda(cudaGetLastError(), "b1_tma_dual(kpipe) launch"); \
    check_cuda(cudaStreamSynchronize(stream), "b1_tma_dual(kpipe) sync"); \
    return D; } while(0)
  if (kp == 1) {
    TORCH_CHECK((int)kmul == 2 && st == 3, "kpipe curated to km2/st3");
    if (tile == 0) {
      if (sw == 0) { if (pst == 0) LDK(128,128,2,4,3,2,1,0,0); else LDK(128,128,2,4,3,2,1,0,1); }
      if (sw == 8) { if (pst == 0) LDK(128,128,2,4,3,2,1,8,0); else LDK(128,128,2,4,3,2,1,8,1); }
    } else if (tile == 6) {
      if (sw == 0) { if (pst == 0) LDK(128,128,4,2,3,2,1,0,0); else LDK(128,128,4,2,3,2,1,0,1); }
      if (sw == 8) { if (pst == 0) LDK(128,128,4,2,3,2,1,8,0); else LDK(128,128,4,2,3,2,1,8,1); }
    }
    TORCH_CHECK(false, "unsupported kpipe config: tile=", tile, " st=", st, " swz=", sw, " persist=", pst);
  }
#undef LDK

  // ----- round-7 persistent-scheduler variants (PERSIST=1): grid=#SM, CTA strides over
  // tiles with a continuous ring (StaticPersistentTileSchedulerSm90 semantics). Mirrors
  // the 128x128 / 128x128-w42 km2 family (the only tiles that ever win per round-6).
#define LDP(BM,BN,WM,WN,ST,KM,MB,SW) do { \
    launch_tma_dual<BM,BN,WM,WN,ST,KM,MB,SW,1>(Al,SFAl,Bl,SFBl,Ah,SFAh,Bh,SFBh,s_x,s_w,D,Mi,Ni,Kl,Kh,stream); \
    check_cuda(cudaGetLastError(), "b1_tma_dual(persist) launch"); \
    check_cuda(cudaStreamSynchronize(stream), "b1_tma_dual(persist) sync"); \
    return D; } while(0)
  if (pst == 1) {
    if (km == 2) {
      if (tile == 0) {
        if (sw == 0) { if (st == 3) LDP(128,128,2,4,3,2,1,0); if (st == 4) LDP(128,128,2,4,4,2,1,0); }
        if (sw == 4) { if (st == 3) LDP(128,128,2,4,3,2,1,4); if (st == 4) LDP(128,128,2,4,4,2,1,4); }
        if (sw == 8) { if (st == 3) LDP(128,128,2,4,3,2,1,8); if (st == 4) LDP(128,128,2,4,4,2,1,8); }
      } else if (tile == 6) {
        if (sw == 0) { if (st == 3) LDP(128,128,4,2,3,2,1,0); if (st == 4) LDP(128,128,4,2,4,2,1,0); }
        if (sw == 4) { if (st == 3) LDP(128,128,4,2,3,2,1,4); if (st == 4) LDP(128,128,4,2,4,2,1,4); }
        if (sw == 8) { if (st == 3) LDP(128,128,4,2,3,2,1,8); if (st == 4) LDP(128,128,4,2,4,2,1,8); }
      }
    }
    TORCH_CHECK(false, "unsupported persistent dual config: tile=", tile, " st=", st,
                " swz=", sw, " kmul=", km);
  }
#undef LDP

  // ----- round-6 multi-axis grid: tile(BM,BN,WM,WN) x stages x swz x kmul x minb.
  // Each instantiated (tile,st,sw,km,mb) tuple below is one kernel; unlisted tuples
  // throw so the Python harness records them as unsupported and moves on.
#define LD(BM,BN,WM,WN,ST,KM,MB,SW) do { \
    launch_tma_dual<BM,BN,WM,WN,ST,KM,MB,SW>(Al,SFAl,Bl,SFBl,Ah,SFAh,Bh,SFBh,s_x,s_w,D,Mi,Ni,Kl,Kh,stream); \
    check_cuda(cudaGetLastError(), "b1_tma_dual launch"); \
    check_cuda(cudaStreamSynchronize(stream), "b1_tma_dual sync"); \
    return D; } while(0)

  if (tile == 0) {               // 128x128, 8 consumer warps (WM2 WN4) — baseline tile
    if (km == 2) {
      if (sw == 0 && mb == 1) {
        if (st == 2) LD(128,128,2,4,2,2,1,0);
        if (st == 3) LD(128,128,2,4,3,2,1,0);
        if (st == 4) LD(128,128,2,4,4,2,1,0);
        if (st == 5) LD(128,128,2,4,5,2,1,0);
        // st6 @128x128/km2 = 110688 B > 99KB smem max (ptxas); pruned (illegal).
      }
      if (sw == 0 && mb == 2 && st == 3) LD(128,128,2,4,3,2,2,0);
      if (sw == 2) { if (st == 3) LD(128,128,2,4,3,2,1,2); if (st == 4) LD(128,128,2,4,4,2,1,2); }
      if (sw == 4) { if (st == 3) LD(128,128,2,4,3,2,1,4); if (st == 4) LD(128,128,2,4,4,2,1,4); }
      if (sw == 8) { if (st == 3) LD(128,128,2,4,3,2,1,8); if (st == 4) LD(128,128,2,4,4,2,1,8); }
    } else if (km == 1 && sw == 0 && mb == 1) {   // BK=32: longer K-pipe for short-K gate
      if (st == 4) LD(128,128,2,4,4,1,1,0);
      if (st == 6) LD(128,128,2,4,6,1,1,0);
      if (st == 8) LD(128,128,2,4,8,1,1,0);
    } else if (km == 4 && sw == 0 && mb == 1) {   // BK=128: fewer, fatter K-iters
      if (st == 2) LD(128,128,2,4,2,4,1,0);
    }
  } else if (tile == 6) {        // 128x128, warp-layout variant WM4 WN2 (consumer-warp axis)
    if (km == 2) {
      if (sw == 0) { if (st == 3) LD(128,128,4,2,3,2,1,0); if (st == 4) LD(128,128,4,2,4,2,1,0); }
      if (sw == 4) { if (st == 3) LD(128,128,4,2,3,2,1,4); if (st == 4) LD(128,128,4,2,4,2,1,4); }
    }
  } else if (tile == 1) {        // 128x256, 16 warps (WM2 WN8) — large-N (gate)
    TORCH_CHECK(N % 256 == 0, "tile=1 (BN=256) needs N%256==0");
    if (km == 2) {               // st<=3 only (st4+ > 99KB smem)
      if (sw == 0) { if (st == 2) LD(128,256,2,8,2,2,1,0); if (st == 3) LD(128,256,2,8,3,2,1,0); }
      if (sw == 4) { if (st == 2) LD(128,256,2,8,2,2,1,4); if (st == 3) LD(128,256,2,8,3,2,1,4); }
    }
  } else if (tile == 2) {        // 256x128, 16 warps (WM4 WN4) — large-M
    TORCH_CHECK(M % 256 == 0, "tile=2 (BM=256) needs M%256==0");
    if (km == 2 && sw == 0) { if (st == 2) LD(256,128,4,4,2,2,1,0); if (st == 3) LD(256,128,4,4,3,2,1,0); }
  } else if (tile == 3) {        // 64x128, 4 warps (WM1 WN4) — small-M / more CTAs
    if (km == 2 && sw == 0) {
      if (st == 3) LD(64,128,1,4,3,2,1,0);
      if (st == 4) LD(64,128,1,4,4,2,1,0);
      if (st == 5) LD(64,128,1,4,5,2,1,0);
    }
  } else if (tile == 4) {        // 128x64, 4 warps (WM2 WN2) — small-N
    if (km == 2 && sw == 0) {
      if (st == 3) LD(128,64,2,2,3,2,1,0);
      if (st == 4) LD(128,64,2,2,4,2,1,0);
      if (st == 5) LD(128,64,2,2,5,2,1,0);
    }
  }
#undef LD
  TORCH_CHECK(false, "unsupported dual config: tile=", tile, " st=", st,
              " swz=", sw, " kmul=", km, " minb=", mb);
  return D;
}


// ============================================================================
// Host entry — pure MXFP8 (Phase-2 only) TMA kernel.
// ============================================================================
template <int BM, int BN, int WM, int WN, int STAGES, int KMUL, int MINB = 1, int SWZ = 0>
static void launch_tma_mx(const torch::Tensor& Ah, const torch::Tensor& SFAh,
                          const torch::Tensor& Bh, const torch::Tensor& SFBh,
                          const torch::Tensor& s_x, const torch::Tensor& s_w,
                          torch::Tensor& D, int M, int N, int K_high,
                          cudaStream_t stream) {
  using LA = SmemLayoutA_t<BM, BN, STAGES, KMUL>;
  using LB = SmemLayoutB_t<BM, BN, STAGES, KMUL>;
  Tensor mA = make_tensor(make_gmem_ptr(Ah.data_ptr<uint8_t>()),
                          make_shape(M, K_high), make_stride(K_high, _1{}));
  Tensor mB = make_tensor(make_gmem_ptr(Bh.data_ptr<uint8_t>()),
                          make_shape(N, K_high), make_stride(K_high, _1{}));
  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, LA{}(_, _, Int<0>{}),
                             make_shape(Int<BM>{}, Int<K_MX * KMUL>{}), _1{});
  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, LB{}(_, _, Int<0>{}),
                             make_shape(Int<BN>{}, Int<K_MX * KMUL>{}), _1{});
  const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN;
  // SWZ>0 -> flat 1D grid (grouped decode inside the kernel).
  dim3 grid = (SWZ > 0) ? dim3(num_m * num_n) : dim3(num_m, num_n);
  auto kptr = &b1_tma_mx_kernel<BM, BN, WM, WN, STAGES, KMUL, MINB, SWZ, decltype(tma_a), decltype(tma_b)>;
  if (getenv("TMA_ATTRS")) {
    static bool once = false;
    if (!once) {
      int dev = 0; cudaGetDevice(&dev);
      int smem_sm = 0, regs_sm = 0;
      cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
      cudaDeviceGetAttribute(&regs_sm, cudaDevAttrMaxRegistersPerMultiprocessor, dev);
      printf("[dev] smemPerSM=%d regsPerSM=%d\n", smem_sm, regs_sm);
      once = true;
    }
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, kptr);
    int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kptr, WM * WN * 32 + 32, 0);
    printf("[attrs] BM=%d BN=%d st=%d KMUL=%d MINB=%d SWZ=%d  smem=%zu regs=%d localB=%zu maxBlocks/SM=%d\n",
           BM, BN, STAGES, KMUL, MINB, SWZ, fa.sharedSizeBytes, fa.numRegs, fa.localSizeBytes, nb);
    fflush(stdout);
  }
  b1_tma_mx_kernel<BM, BN, WM, WN, STAGES, KMUL, MINB, SWZ, decltype(tma_a), decltype(tma_b)>
      <<<grid, WM * WN * 32 + 32, 0, stream>>>(
          tma_a, tma_b, SFAh.data_ptr<uint8_t>(), SFBh.data_ptr<uint8_t>(),
          s_x.data_ptr<float>(), s_w.data_ptr<float>(),
          reinterpret_cast<__half*>(D.data_ptr<at::Half>()), M, N, K_high);
}

torch::Tensor b1_fused_tma_mx(torch::Tensor Ah, torch::Tensor SFAh,
                              torch::Tensor Bh, torch::Tensor SFBh,
                              torch::Tensor s_x, torch::Tensor s_w,
                              int64_t M, int64_t N, int64_t K_high,
                              int64_t stages, int64_t minb, int64_t tile, int64_t swz) {
  TORCH_CHECK(Ah.is_cuda() && Bh.is_cuda(), "cuda");
  TORCH_CHECK(K_high % (2 * K_MX) == 0, "K_high must be divisible by 64 (KMUL=2)");
  Ah = Ah.contiguous(); Bh = Bh.contiguous();
  SFAh = SFAh.contiguous(); SFBh = SFBh.contiguous();
  s_x = s_x.contiguous(); s_w = s_w.contiguous();
  auto dev = Ah.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto D = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(dev));
  int st = (int)stages; if (st == 0) st = 4;
  const int Mi = M, Ni = N, Kh = K_high;
  const int sw = (int)swz;
  // tile: 0 = 128x128 (default), 1 = 128x64 (half-B / half-accum -> 2 CTA/SM).
  if (tile == 1) {
    // 128x64, 8 warps (WM=2,WN=4): BN=64 halves B-smem + accumulator regs.
    if (st == 3)      launch_tma_mx<128, 64, 2, 4, 3, 2>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
    else if (st == 5) launch_tma_mx<128, 64, 2, 4, 5, 2>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
    else              launch_tma_mx<128, 64, 2, 4, 4, 2>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
  } else if (st == 3 && minb == 2) {
    launch_tma_mx<128, 128, 2, 4, 3, 2, 2>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
  } else if (sw == 4) {
    if (st == 3)      launch_tma_mx<128, 128, 2, 4, 3, 2, 1, 4>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
    else if (st == 5) launch_tma_mx<128, 128, 2, 4, 5, 2, 1, 4>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
    else              launch_tma_mx<128, 128, 2, 4, 4, 2, 1, 4>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
  } else if (sw == 8) {
    if (st == 3)      launch_tma_mx<128, 128, 2, 4, 3, 2, 1, 8>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
    else if (st == 5) launch_tma_mx<128, 128, 2, 4, 5, 2, 1, 8>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
    else              launch_tma_mx<128, 128, 2, 4, 4, 2, 1, 8>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
  } else if (sw == 2) {
    if (st == 3)      launch_tma_mx<128, 128, 2, 4, 3, 2, 1, 2>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
    else if (st == 5) launch_tma_mx<128, 128, 2, 4, 5, 2, 1, 2>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
    else              launch_tma_mx<128, 128, 2, 4, 4, 2, 1, 2>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
  } else if (st == 3) {
    launch_tma_mx<128, 128, 2, 4, 3, 2>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
  } else if (st == 5) {
    launch_tma_mx<128, 128, 2, 4, 5, 2>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
  } else {
    launch_tma_mx<128, 128, 2, 4, 4, 2>(Ah, SFAh, Bh, SFBh, s_x, s_w, D, Mi, Ni, Kh, stream);
  }
  check_cuda(cudaGetLastError(), "b1_tma_mx launch");
  check_cuda(cudaStreamSynchronize(stream), "b1_tma_mx sync");
  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("b1_fused_tma_mx", &b1_fused_tma_mx,
        "B1 Phase-2 (MXFP8) TMA fused GEMM * s_x * s_w -> fp16 (SM120)",
        py::arg("Ah"), py::arg("SFAh"), py::arg("Bh"), py::arg("SFBh"),
        py::arg("s_x"), py::arg("s_w"),
        py::arg("M"), py::arg("N"), py::arg("K_high"), py::arg("stages") = 0,
        py::arg("minb") = 1, py::arg("tile") = 0, py::arg("swz") = 0);
  m.def("b1_fused_tma_dual", &b1_fused_tma_dual,
        "B1 dual-phase (NVFP4+MXFP8) TMA fused GEMM * s_x * s_w -> fp16 (SM120, RECIPE B)",
        py::arg("Al"), py::arg("SFAl"), py::arg("Bl"), py::arg("SFBl"),
        py::arg("Ah"), py::arg("SFAh"), py::arg("Bh"), py::arg("SFBh"),
        py::arg("s_x"), py::arg("s_w"),
        py::arg("M"), py::arg("N"), py::arg("K_low"), py::arg("K_high"),
        py::arg("stages") = 0, py::arg("tile") = 0,
        py::arg("swz") = 0, py::arg("kmul") = 2, py::arg("minb") = 1,
        py::arg("persist") = 0, py::arg("kpipe") = 0);
}
