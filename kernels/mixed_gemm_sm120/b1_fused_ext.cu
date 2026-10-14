// B1 — TRUE single-CUDA-kernel fused mixed-precision GEMM for SM120 (RTX Pro 6000 Blackwell).
//
//   D[m,n] = ( sum_{k<K_low}  A_fp4[m,k]*sfa4*B_fp4[n,k]*sfb4
//            + sum_{k<K_high} A_fp8[m,k]*sfa8*B_fp8[n,k]*sfb8 ) * s_x[m] * s_w[n]
//
// Phase 1: NVFP4 block-scaled MMA (mxf4nvf4, m16n8k64, SFVecSize=16) over K_low.
// Phase 2: MXFP8 block-scaled MMA (f8f6f4,  m16n8k32, SFVecSize=32) over K_high.
// Both phases accumulate into the SAME FP32 register accumulator (CLayout identical).
//
// Design (correctness-first, naive): one warp (32 threads) per 16x8 output tile.
// Stage A/B/SF per K-chunk through NON-swizzled smem (handles M<16 zero-padding),
// partition with the raw atom TiledMMA, plain smem->reg copies, cute::gemm with
// zip(data,SF). No TMA / no ldmatrix / no warp-specialization.
//
// Exposes b1_nvfp4_only (Step-1 isolation entry, fp32 out) and b1_fused (full
// dual-phase kernel, s_x*s_w scaling, fp16 out). Build/validate via b1_fused.py.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm120.hpp"
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
// Atom + TiledMMA types
// ============================================================================
using AtomNV = SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t, 16>;
using TiledMmaNV = decltype(make_tiled_mma(AtomNV{}));

using AtomMX = SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, cutlass::float_ue8m0_t, 32>;
using TiledMmaMX = decltype(make_tiled_mma(AtomMX{}));

// Shared accumulator across phases is only valid if both atoms partition C
// identically (FP32[4]/thread, SM80_16x8_Row).
static_assert(cute::is_same_v<typename cute::MMA_Traits<AtomNV>::CLayout,
                              typename cute::MMA_Traits<AtomMX>::CLayout>,
              "NVFP4 and MXFP8 atoms must share the same C layout");

using ElemSF_NV = cutlass::float_ue4m3_t;      // 8-bit
using ElemSF_MX = cutlass::float_ue8m0_t;      // 8-bit (exponent byte)
// NOTE: the MMA traits declare ValTypeA/B = uint4_t (NVFP4) / uint8_t (MXFP8),
// so partition_fragment_A/B produce integer-typed fragments. The smem views
// MUST use the SAME types: cute::copy between float_e2m1_t and uint4_t does a
// VALUE conversion (float->uint truncation, negatives clamped to 0), silently
// corrupting every nibble. (Root cause of the original cosine~0 bug; see
// docs/b1_report.md.)
using ElemFP4   = cute::uint4_t;                // raw 4-bit code (bit-copied)
using ElemFP8   = uint8_t;                      // raw e4m3 byte (bit-copied)

static constexpr int M_TILE = 16;
static constexpr int N_TILE = 8;
static constexpr int K_NV    = 64;              // NVFP4 atom K
static constexpr int VEC_NV  = 16;              // NVFP4 SFVecSize
static constexpr int K_MX    = 32;              // MXFP8 atom K
static constexpr int VEC_MX  = 32;              // MXFP8 SFVecSize

static inline void check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// ============================================================================
// Templated phase runner: stage one K-chunk at a time through smem, partition
// with the raw atom TiledMMA, plain bit-copies smem->reg, block-scaled MMA
// into the shared accumulator. KATOM == atom K so MMA_K == 1 per chunk.
// ============================================================================
template <class ElemData, class ElemSF, int KATOM, int VEC,
          class TiledMma, class AccumT>
__device__ inline void b1_run_phase(TiledMma tiled_mma,
                                    const uint8_t* __restrict__ Ag,
                                    const uint8_t* __restrict__ SFAg,
                                    const uint8_t* __restrict__ Bg,
                                    const uint8_t* __restrict__ SFBg,
                                    uint8_t* sA_b, uint8_t* sB_b,
                                    uint8_t* sSFA_b, uint8_t* sSFB_b,
                                    int m0, int n0, int M, int N, int K,
                                    int tid, AccumT& accum) {
  constexpr int EB      = cute::sizeof_bits_v<ElemData>;  // 4 or 8
  constexpr int BYTES_K = KATOM * EB / 8;                 // bytes per row per chunk (32)
  constexpr int NBLK    = KATOM / VEC;                    // SF blocks per chunk
  const int KpRow = K * EB / 8;                           // gmem bytes per row
  const int Knb   = K / VEC;                              // gmem SF blocks per row

  // --- cute views over smem ---
  Tensor sA = make_tensor(recast_ptr<ElemData>(make_smem_ptr(sA_b)),
                          make_layout(make_shape(Int<M_TILE>{}, Int<KATOM>{}),
                                      make_stride(Int<KATOM>{}, Int<1>{})));
  Tensor sB = make_tensor(recast_ptr<ElemData>(make_smem_ptr(sB_b)),
                          make_layout(make_shape(Int<N_TILE>{}, Int<KATOM>{}),
                                      make_stride(Int<KATOM>{}, Int<1>{})));
  // SF: (rows,(VEC,NBLK)) = (rows,K) with stride-0 broadcast within each block.
  Tensor sSFA = make_tensor(recast_ptr<ElemSF>(make_smem_ptr(sSFA_b)),
                            make_layout(make_shape(Int<M_TILE>{}, make_shape(Int<VEC>{}, Int<NBLK>{})),
                                        make_stride(Int<NBLK>{}, make_stride(Int<0>{}, Int<1>{}))));
  Tensor sSFB = make_tensor(recast_ptr<ElemSF>(make_smem_ptr(sSFB_b)),
                            make_layout(make_shape(Int<N_TILE>{}, make_shape(Int<VEC>{}, Int<NBLK>{})),
                                        make_stride(Int<NBLK>{}, make_stride(Int<0>{}, Int<1>{}))));

  auto thr_mma = tiled_mma.get_thread_slice(tid);

  // SF smem->reg tiled copies
  auto tile_shape_mnk = tile_shape(tiled_mma);
  auto tcopy_SFA = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF>{},
                                        b1_get_layoutSFA_TV(tiled_mma),
                                        make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto tcopy_SFB = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF>{},
                                        b1_get_layoutSFB_TV(tiled_mma),
                                        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_copy_SFA = tcopy_SFA.get_thread_slice(tid);
  auto thr_copy_SFB = tcopy_SFB.get_thread_slice(tid);

  const int nchunks = K / KATOM;
  for (int c = 0; c < nchunks; ++c) {
    const int kbyte0 = c * BYTES_K;          // first gmem byte of this chunk
    const int kb0 = c * NBLK;                // first SF block of this chunk

    // --- cooperative load gmem -> smem (zero-pad OOB rows) ---
    for (int i = tid; i < M_TILE * BYTES_K; i += 32) {
      int row = i / BYTES_K, byte = i % BYTES_K;
      int mg = m0 + row;
      sA_b[i] = (mg < M) ? Ag[mg * KpRow + kbyte0 + byte] : 0;
    }
    for (int i = tid; i < N_TILE * BYTES_K; i += 32) {
      int row = i / BYTES_K, byte = i % BYTES_K;
      int ng = n0 + row;
      sB_b[i] = (ng < N) ? Bg[ng * KpRow + kbyte0 + byte] : 0;
    }
    for (int i = tid; i < M_TILE * NBLK; i += 32) {
      int row = i / NBLK, blk = i % NBLK;
      int mg = m0 + row;
      sSFA_b[i] = (mg < M) ? SFAg[mg * Knb + kb0 + blk] : 0;
    }
    for (int i = tid; i < N_TILE * NBLK; i += 32) {
      int row = i / NBLK, blk = i % NBLK;
      int ng = n0 + row;
      sSFB_b[i] = (ng < N) ? SFBg[ng * Knb + kb0 + blk] : 0;
    }
    __syncthreads();

    // --- partition + bit-copy smem -> reg ---
    Tensor tCrA = thr_mma.partition_fragment_A(sA);     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB);     // (MMA,MMA_N,MMA_K)
    copy(thr_mma.partition_A(sA), tCrA);
    copy(thr_mma.partition_B(sB), tCrB);
    // fp4_shift is a no-op for the block-scaled VS atoms; call for parity.
    fp4_shift_A(typename TiledMma::MMA_Op{}, tCrA);
    fp4_shift_B(typename TiledMma::MMA_Op{}, tCrB);

    Tensor tCrSFA = b1_partition_fragment_SFA(sSFA, thr_mma);
    Tensor tCrSFB = b1_partition_fragment_SFB(sSFB, thr_mma);
    copy(thr_copy_SFA.partition_S(sSFA), thr_copy_SFA.retile_D(tCrSFA));
    copy(thr_copy_SFB.partition_S(sSFB), thr_copy_SFB.retile_D(tCrSFB));

    // --- single-atom block-scaled MMA into the shared accumulator ---
    cute::gemm(tiled_mma,
               make_zip_tensor(tCrA(_,_,_0{}), tCrSFA(_,_,_0{})),
               make_zip_tensor(tCrB(_,_,_0{}), tCrSFB(_,_,_0{})),
               accum);
    __syncthreads();
  }
}

// ============================================================================
// STEP 1 kernel — NVFP4-only isolation entry. Output fp32 [M,N].
// grid = (ceil(M/16), ceil(N/8)); block = 32 threads (one warp).
// ============================================================================
__global__ void b1_nvfp4_kernel(const uint8_t* __restrict__ A_packed,  // [M, K/2]
                                const uint8_t* __restrict__ SFA,        // [M, K/16]
                                const uint8_t* __restrict__ B_packed,  // [N, K/2]
                                const uint8_t* __restrict__ SFB,        // [N, K/16]
                                float* __restrict__ D,                 // [M, N]
                                int M, int N, int K) {
  const int tid = threadIdx.x;              // 0..31
  const int m0 = blockIdx.x * M_TILE;
  const int n0 = blockIdx.y * N_TILE;

  __shared__ uint8_t sA_b[M_TILE * 32];     // 512
  __shared__ uint8_t sB_b[N_TILE * 32];     // 256
  __shared__ uint8_t sSFA_b[M_TILE * 4];    // 64
  __shared__ uint8_t sSFB_b[N_TILE * 4];    // 32

  TiledMmaNV tiled_mma;
  Tensor accum = partition_fragment_C(tiled_mma, Shape<Int<M_TILE>, Int<N_TILE>>{});
  clear(accum);

  b1_run_phase<ElemFP4, ElemSF_NV, K_NV, VEC_NV>(
      tiled_mma, A_packed, SFA, B_packed, SFB,
      sA_b, sB_b, sSFA_b, sSFB_b, m0, n0, M, N, K, tid, accum);

  // --- store accum -> D with SM80_16x8_Row mapping + bounds ---
  // thread t: rows {t/4, t/4+8}, cols {(t%4)*2, (t%4)*2+1}; vals c0..c3.
  int r0 = tid / 4;
  int cc = (tid % 4) * 2;
  int rows[2] = {r0, r0 + 8};
  int cols[2] = {cc, cc + 1};
  // accum layout: (MMA=(2,2),1,1). flatten 4 values in order (c0,c1,c2,c3)
  //   c0=(row0,col0) c1=(row0,col1) c2=(row1,col0) c3=(row1,col1)
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
// Full B1 kernel — Phase 1 NVFP4 over K_low, Phase 2 MXFP8 over K_high, both
// into ONE register accumulator; epilogue scales by s_x[m]*s_w[n] -> fp16.
// ============================================================================
__global__ void b1_fused_kernel(const uint8_t* __restrict__ Al,   // [M, K_low/2]
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
  const int tid = threadIdx.x;              // 0..31
  const int m0 = blockIdx.x * M_TILE;
  const int n0 = blockIdx.y * N_TILE;

  // Both phases stage 32 bytes/row per chunk; buffers are reused sequentially.
  __shared__ uint8_t sA_b[M_TILE * 32];     // 512
  __shared__ uint8_t sB_b[N_TILE * 32];     // 256
  __shared__ uint8_t sSFA_b[M_TILE * 4];    // 64 (NV needs 4 blk/row, MX needs 1)
  __shared__ uint8_t sSFB_b[N_TILE * 4];    // 32

  TiledMmaNV tiled_mma_nv;
  TiledMmaMX tiled_mma_mx;
  Tensor accum = partition_fragment_C(tiled_mma_nv, Shape<Int<M_TILE>, Int<N_TILE>>{});
  clear(accum);

  // Phase 1: NVFP4 over K_low
  b1_run_phase<ElemFP4, ElemSF_NV, K_NV, VEC_NV>(
      tiled_mma_nv, Al, SFAl, Bl, SFBl,
      sA_b, sB_b, sSFA_b, sSFB_b, m0, n0, M, N, K_low, tid, accum);

  // Phase 2: MXFP8 over K_high into the SAME accumulator
  b1_run_phase<ElemFP8, ElemSF_MX, K_MX, VEC_MX>(
      tiled_mma_mx, Ah, SFAh, Bh, SFBh,
      sA_b, sB_b, sSFA_b, sSFB_b, m0, n0, M, N, K_high, tid, accum);

  // Epilogue: out[m,n] = accum * s_x[m] * s_w[n] -> fp16 (SM80_16x8_Row map)
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
      if (mg < M && ng < N)
        D[mg * N + ng] = __float2half(vals[idx] * s_x[mg] * s_w[ng]);
      ++idx;
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

  dim3 grid((M + M_TILE - 1) / M_TILE, (N + N_TILE - 1) / N_TILE);
  b1_nvfp4_kernel<<<grid, 32, 0, stream>>>(
      A_packed.data_ptr<uint8_t>(), SFA.data_ptr<uint8_t>(),
      B_packed.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(),
      D.data_ptr<float>(), (int)M, (int)N, (int)K);
  check_cuda(cudaGetLastError(), "b1_nvfp4_kernel launch");
  check_cuda(cudaStreamSynchronize(stream), "b1 sync");
  return D;
}

torch::Tensor b1_fused(torch::Tensor Al, torch::Tensor SFAl,
                       torch::Tensor Bl, torch::Tensor SFBl,
                       torch::Tensor Ah, torch::Tensor SFAh,
                       torch::Tensor Bh, torch::Tensor SFBh,
                       torch::Tensor s_x, torch::Tensor s_w,
                       int64_t M, int64_t N, int64_t K_low, int64_t K_high) {
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

  dim3 grid((M + M_TILE - 1) / M_TILE, (N + N_TILE - 1) / N_TILE);
  b1_fused_kernel<<<grid, 32, 0, stream>>>(
      Al.data_ptr<uint8_t>(), SFAl.data_ptr<uint8_t>(),
      Bl.data_ptr<uint8_t>(), SFBl.data_ptr<uint8_t>(),
      Ah.data_ptr<uint8_t>(), SFAh.data_ptr<uint8_t>(),
      Bh.data_ptr<uint8_t>(), SFBh.data_ptr<uint8_t>(),
      s_x.data_ptr<float>(), s_w.data_ptr<float>(),
      reinterpret_cast<__half*>(D.data_ptr<at::Half>()),
      (int)M, (int)N, (int)K_low, (int)K_high);
  check_cuda(cudaGetLastError(), "b1_fused_kernel launch");
  check_cuda(cudaStreamSynchronize(stream), "b1_fused sync");
  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("b1_nvfp4_only", &b1_nvfp4_only, "B1 step1: NVFP4-only hand kernel -> fp32 (SM120)");
  m.def("b1_fused", &b1_fused,
        "B1: TRUE single-kernel fused NVFP4(K_low)+MXFP8(K_high) GEMM * s_x * s_w -> fp16 (SM120)");
}
