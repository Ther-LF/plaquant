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
}
