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
// This file currently implements STEP 1 (NVFP4-only) exposed as b1_nvfp4_only, plus
// the full STEP 3 b1_fused once phases 2/3 land. Build/validate via b1_fused.py.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

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

using ElemSF_NV = cutlass::float_ue4m3_t;      // 8-bit
using ElemFP4   = cutlass::float_e2m1_t;        // uint4-like subbyte

static constexpr int M_TILE = 16;
static constexpr int N_TILE = 8;
static constexpr int K_NV    = 64;              // NVFP4 atom K
static constexpr int VEC_NV  = 16;              // NVFP4 SFVecSize

static inline void check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// ============================================================================
// STEP 1 kernel — NVFP4-only. Output fp32 [M,N].
// grid = (ceil(M/16), ceil(N/8)); block = 32 threads (one warp).
// ============================================================================
__global__ void b1_nvfp4_kernel(const uint8_t* __restrict__ A_packed,  // [M, K/2]
                                const uint8_t* __restrict__ SFA,        // [M, K/16]
                                const uint8_t* __restrict__ B_packed,  // [N, K/2]
                                const uint8_t* __restrict__ SFB,        // [N, K/16]
                                float* __restrict__ D,                 // [M, N]
                                int M, int N, int K) {
  const int tile_m = blockIdx.x;
  const int tile_n = blockIdx.y;
  const int tid    = threadIdx.x;             // 0..31
  const int m0 = tile_m * M_TILE;
  const int n0 = tile_n * N_TILE;

  const int Kp   = K / 2;    // packed bytes per row of A/B
  const int Knb  = K / VEC_NV; // SF blocks per row

  // Shared staging (per 64-wide K chunk).
  __shared__ uint8_t sA_b[M_TILE * (K_NV / 2)];   // 16*32 = 512
  __shared__ uint8_t sB_b[N_TILE * (K_NV / 2)];   // 8*32  = 256
  __shared__ uint8_t sSFA_b[M_TILE * (K_NV / VEC_NV)]; // 16*4 = 64
  __shared__ uint8_t sSFB_b[N_TILE * (K_NV / VEC_NV)]; // 8*4  = 32

  // --- build cute views over smem ---
  // A: (16,64) fp4, row-major -> packed 2/byte. Element type ElemFP4 (subbyte).
  Tensor sA = make_tensor(recast_ptr<ElemFP4>(make_smem_ptr(sA_b)),
                          make_layout(make_shape(Int<M_TILE>{}, Int<K_NV>{}),
                                      make_stride(Int<K_NV>{}, Int<1>{})));
  Tensor sB = make_tensor(recast_ptr<ElemFP4>(make_smem_ptr(sB_b)),
                          make_layout(make_shape(Int<N_TILE>{}, Int<K_NV>{}),
                                      make_stride(Int<K_NV>{}, Int<1>{})));
  // SFA: (16,(16,4)) = (M,K) with stride-0 broadcast across the 16-wide block.
  Tensor sSFA = make_tensor(recast_ptr<ElemSF_NV>(make_smem_ptr(sSFA_b)),
                            make_layout(make_shape(Int<M_TILE>{}, make_shape(Int<VEC_NV>{}, Int<K_NV/VEC_NV>{})),
                                        make_stride(Int<K_NV/VEC_NV>{}, make_stride(Int<0>{}, Int<1>{}))));
  Tensor sSFB = make_tensor(recast_ptr<ElemSF_NV>(make_smem_ptr(sSFB_b)),
                            make_layout(make_shape(Int<N_TILE>{}, make_shape(Int<VEC_NV>{}, Int<K_NV/VEC_NV>{})),
                                        make_stride(Int<K_NV/VEC_NV>{}, make_stride(Int<0>{}, Int<1>{}))));

  TiledMmaNV tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);

  // Accumulator fragment (MMA, MMA_M, MMA_N) == float[4]
  Tensor accum = partition_fragment_C(tiled_mma, Shape<Int<M_TILE>, Int<N_TILE>>{});
  clear(accum);

  // SF smem->reg tiled copies
  auto tile_shape_mnk = tile_shape(tiled_mma);
  auto tcopy_SFA = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF_NV>{},
                                        b1_get_layoutSFA_TV(tiled_mma),
                                        make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto tcopy_SFB = make_tiled_copy_impl(Copy_Atom<DefaultCopy, ElemSF_NV>{},
                                        b1_get_layoutSFB_TV(tiled_mma),
                                        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_copy_SFA = tcopy_SFA.get_thread_slice(tid);
  auto thr_copy_SFB = tcopy_SFB.get_thread_slice(tid);

  const int nchunks = K / K_NV;
  for (int c = 0; c < nchunks; ++c) {
    const int k0 = c * K_NV;
    const int kb0 = k0 / VEC_NV;           // first SF block index for this chunk

    // --- cooperative load gmem -> smem (zero-pad OOB rows) ---
    // A packed: 16 rows x 32 bytes
    for (int i = tid; i < M_TILE * (K_NV / 2); i += 32) {
      int row = i / (K_NV / 2);
      int byte = i % (K_NV / 2);
      int mg = m0 + row;
      sA_b[i] = (mg < M) ? A_packed[mg * Kp + k0 / 2 + byte] : 0;
    }
    // B packed: 8 rows x 32 bytes
    for (int i = tid; i < N_TILE * (K_NV / 2); i += 32) {
      int row = i / (K_NV / 2);
      int byte = i % (K_NV / 2);
      int ng = n0 + row;
      sB_b[i] = (ng < N) ? B_packed[ng * Kp + k0 / 2 + byte] : 0;
    }
    // SFA: 16 rows x 4 blocks
    for (int i = tid; i < M_TILE * (K_NV / VEC_NV); i += 32) {
      int row = i / (K_NV / VEC_NV);
      int blk = i % (K_NV / VEC_NV);
      int mg = m0 + row;
      sSFA_b[i] = (mg < M) ? SFA[mg * Knb + kb0 + blk] : 0;
    }
    // SFB: 8 rows x 4 blocks
    for (int i = tid; i < N_TILE * (K_NV / VEC_NV); i += 32) {
      int row = i / (K_NV / VEC_NV);
      int blk = i % (K_NV / VEC_NV);
      int ng = n0 + row;
      sSFB_b[i] = (ng < N) ? SFB[ng * Knb + kb0 + blk] : 0;
    }
    __syncthreads();

    // --- partition + copy smem -> reg ---
    Tensor tCrA = thr_mma.partition_fragment_A(sA);     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB);     // (MMA,MMA_N,MMA_K)
    copy(thr_mma.partition_A(sA), tCrA);
    copy(thr_mma.partition_B(sB), tCrB);
    // fp4_shift is a no-op for mxf4nvf4 atom; call for parity w/ collective.
    fp4_shift_A(typename TiledMmaNV::MMA_Op{}, tCrA);
    fp4_shift_B(typename TiledMmaNV::MMA_Op{}, tCrB);

    Tensor tCrSFA = b1_partition_fragment_SFA(sSFA, thr_mma);
    Tensor tCrSFB = b1_partition_fragment_SFB(sSFB, thr_mma);
    copy(thr_copy_SFA.partition_S(sSFA), thr_copy_SFA.retile_D(tCrSFA));
    copy(thr_copy_SFB.partition_S(sSFB), thr_copy_SFB.retile_D(tCrSFB));

    // --- single-atom MMA (k_block 0) into shared accumulator ---
    cute::gemm(tiled_mma,
               make_zip_tensor(tCrA(_,_,_0{}), tCrSFA(_,_,_0{})),
               make_zip_tensor(tCrB(_,_,_0{}), tCrSFB(_,_,_0{})),
               accum);
    __syncthreads();
  }

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
// entry point (STEP 1)
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("b1_nvfp4_only", &b1_nvfp4_only, "B1 step1: NVFP4-only hand kernel -> fp32 (SM120)");
}
