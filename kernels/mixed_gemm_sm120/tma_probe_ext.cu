// TMA plumbing probe for SM120 (RTX Pro 6000 Blackwell).
// Goal: confirm cp.async.bulk.tensor (TMA) + mbarrier expect-tx + a swizzled
// smem layout (UMMA::Layout_K_SW*_Atom<uint8_t>) round-trip bit-exactly in our
// JIT extension setup, as the de-risking step before wiring TMA into the B1 WS
// kernel's Phase-2 (MXFP8) data loads. SF path is NOT touched here.
//
// Kernel: one CTA loads a (BN, KCH) uint8 tile of B[n0:n0+BN, k0:k0+KCH] into
// swizzled smem via a single-thread TMA issue, waits on an mbarrier, then every
// thread copies smem->gmem (through the SAME swizzled smem tensor view, so the
// swizzle must invert correctly). Host compares against the gmem source tile.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cutlass/gemm/collective/builders/sm120_common.inl"

using namespace cute;

static constexpr int BN_P   = 128;
static constexpr int KCH_P  = 64;    // MXFP8 KMUL=2 chunk

// Swizzle atom the SM120 collective would pick for 8-bit K-major.
using SmemAtomP = decltype(cutlass::gemm::collective::detail::
                           sm120_rr_smem_selector<uint8_t, Int<KCH_P>>());
using SmemLayoutP = decltype(tile_to_shape(
    SmemAtomP{}, make_shape(Int<BN_P>{}, Int<KCH_P>{})));

__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t cnt) {
  uint32_t a = cast_smem_ptr_to_uint(mbar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect(uint64_t* mbar, uint32_t bytes) {
  uint32_t a = cast_smem_ptr_to_uint(mbar);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t parity) {
  uint32_t a = cast_smem_ptr_to_uint(mbar);
  asm volatile(
      "{\n .reg .pred P;\n LAB:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
      "@P bra DONE;\n bra LAB;\n DONE:\n}\n" :: "r"(a), "r"(parity));
}

template <class TmaB>
__global__ void tma_probe_kernel(CUTE_GRID_CONSTANT TmaB const tma_b,
                                 const uint8_t* __restrict__ Bsrc,  // for shape only
                                 uint8_t* __restrict__ Dout,
                                 int N, int K, int n0, int k0) {
  __shared__ alignas(1024) uint8_t smem[cosize_v<SmemLayoutP>];
  __shared__ alignas(8) uint64_t mbar;

  Tensor gB = tma_b.get_tma_tensor(make_shape(N, K));          // (N,K)
  Tensor sB = make_tensor(make_smem_ptr(smem), SmemLayoutP{}); // (BN,KCH) swizzled

  // tile the gmem tensor; our CTA handles tile (n0/BN, k0/KCH)
  Tensor gBt = local_tile(gB, make_shape(Int<BN_P>{}, Int<KCH_P>{}),
                          make_coord(n0 / BN_P, k0 / KCH_P));   // (BN,KCH)

  auto cta_tma = tma_b.get_slice(Int<0>{});
  Tensor tAgB = cta_tma.partition_S(gBt);   // (TMA,...)
  Tensor tAsB = cta_tma.partition_D(sB);     // (TMA,...)

  const int tid = threadIdx.x;
  if (tid == 0) {
    mbar_init(&mbar, 1);
    mbar_expect(&mbar, (uint32_t)(BN_P * KCH_P));  // bytes transferred
    copy(tma_b.with(mbar), tAgB, tAsB);
  }
  __syncthreads();
  if (tid == 0) mbar_wait(&mbar, 0);
  __syncthreads();

  // smem -> gmem through the swizzled view (row-major Dout tile)
  for (int i = tid; i < BN_P * KCH_P; i += blockDim.x) {
    int r = i / KCH_P, c = i % KCH_P;
    Dout[(size_t)r * KCH_P + c] = sB(r, c);
  }
}

torch::Tensor tma_probe(torch::Tensor B, int64_t N, int64_t K,
                        int64_t n0, int64_t k0) {
  TORCH_CHECK(B.is_cuda() && B.scalar_type() == torch::kUInt8, "B uint8 cuda");
  B = B.contiguous();
  auto dev = B.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto D = torch::zeros({BN_P, KCH_P}, torch::TensorOptions().dtype(torch::kUInt8).device(dev));

  Tensor gB = make_tensor(make_gmem_ptr(B.data_ptr<uint8_t>()),
                          make_shape((int)N, (int)K), make_stride((int)K, _1{}));
  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, gB, SmemLayoutP{},
                             make_shape(Int<BN_P>{}, Int<KCH_P>{}), _1{});

  tma_probe_kernel<<<1, 256, 0, stream>>>(tma_b, B.data_ptr<uint8_t>(),
                                          D.data_ptr<uint8_t>(),
                                          (int)N, (int)K, (int)n0, (int)k0);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch");
  TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "sync");
  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tma_probe", &tma_probe, "SM120 TMA swizzle round-trip probe");
}
