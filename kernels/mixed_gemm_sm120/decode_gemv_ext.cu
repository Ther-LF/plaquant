// decode_gemv_ext.cu — memory-bound W4A4 DECODE GEMV (SM120 RTX PRO 6000 Blackwell).
//
// The fused NVFP4+MXFP8 GEMM (codesign_fused_ext.cu) is a 128x128 tensor-core tile
// tuned for prefill (large M). At DECODE (M=1) its TC units are ~1/128 utilized and
// it reaches only 1-14% of HBM bandwidth -> it is 7-100x SLOWER than the 4-bit
// weight-streaming floor and loses even to FP16 cuBLAS GEMV. The competitive W4A4
// decode win is purely MEMORY-BANDWIDTH: 4-bit weights = ~3.5x fewer weight bytes
// per token, so a memory-bound GEMV that streams the packed weights at HBM BW is
// ~3x faster than FP16 decode.
//
// This is a NEW, non-protected standalone extension (no CUTLASS, no protected
// kernel/mainloop touched). One threadblock per output row n streams that row's
// packed weight (NVFP4 low segment: 2 codes/byte + per-16 E4M3 scale; MXFP8 high
// segment: 1 FP8 byte + per-32 E8M0 power-of-2 scale), dequantizes on the fly, and
// dots with the (tiny, pre-quantized) activation. Output D[n] = sum_k xq[k]*W_deq[n,k]
// in FP32 — bit-faithful to FakeMixedLinear / the fused kernel (same dequantized
// weight values, FP32 accumulate).
//
// Activation is pre-quantized in Python (fake_quantize_nvfp4/mxfp8, the spec path)
// and passed as a dense FP32 (K,) vector: at M=1 the activation is 1xK and its
// read/quant cost is negligible, so the kernel only needs to stream the weight.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>

#define CUDA_CHECK(e) do { cudaError_t e_ = (e); if (e_ != cudaSuccess) { \
  throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_)); } } while (0)

// NVFP4 E2M1 absolute grid value by 3-bit index; bit 3 = sign.
__device__ __forceinline__ float fp4_decode(uint8_t code) {
  const float L[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float a = L[code & 7];
  return (code & 8) ? -a : a;
}

// ---- activation quantize (dequantized fp32), bit-exact to quant_utils
// fake_quantize_nvfp4 / fake_quantize_mxfp8 (== the fused path's qpf). ----
__constant__ float cFP4[8];
static bool g_fp4_ready = false;
static void init_fp4_once() {
  if (g_fp4_ready) return;
  float fp4[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  CUDA_CHECK(cudaMemcpyToSymbol(cFP4, fp4, sizeof(fp4)));
  g_fp4_ready = true;
}
__device__ __forceinline__ float e4m3_to_float(unsigned char b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}
__device__ __forceinline__ float e4m3_ceil_val(float ideal) {
  unsigned char b = (unsigned char)__nv_cvt_float_to_fp8(ideal, __NV_SATFINITE, __NV_E4M3);
  float v = e4m3_to_float(b);
  if (v < ideal && b < 0x7E) { ++b; v = e4m3_to_float(b); }
  return v;
}
__device__ __forceinline__ int rtn_idx(float absx, const float* v, int n) {
  int lo = 0, hi = n;
  while (lo < hi) { int mid = (lo + hi) >> 1; if (v[mid] < absx) lo = mid + 1; else hi = mid; }
  int insert = lo; if (insert > n - 1) insert = n - 1; if (insert < 0) insert = 0;
  int idx_hi = insert < 1 ? 1 : (insert > n - 1 ? n - 1 : insert);
  int idx_lo = idx_hi - 1; if (idx_lo < 0) idx_lo = 0;
  float vlo = v[idx_lo], vhi = v[idx_hi];
  float dlo = fabsf(absx - vlo), dhi = fabsf(vhi - absx);
  bool pick_hi = dhi < dlo;
  if (fabsf(dhi - dlo) < 1e-12f) pick_hi = !(((idx_lo & 1) == 0));
  return pick_hi ? idx_hi : idx_lo;
}

// One warp per scale-block. warps [0, Klow/16) = NVFP4 (16 elems, 16 lanes active);
// warps [Klow/16, Klow/16 + Khigh/32) = MXFP8 (32 elems). Writes dequantized fp32
// xq (K,) that the GEMV dots against. M=1 only (xq length = K).
__global__ void quant_act_kernel(const float* __restrict__ x, float* __restrict__ xq,
                                 int Klow, int Khigh) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  int nlo = Klow / 16;
  int nhi = Khigh / 32;
  if (warp >= nlo + nhi) return;
  if (warp < nlo) {                                   // NVFP4 block of 16
    int k = 16 * warp + lane;                         // lanes 16..31 idle
    float f = (lane < 16) ? x[k] : 0.f;
    float a = fabsf(f);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float ideal = fmaxf(a / 6.0f, 0.001953125f);      // /FP4_E2M1_MAX, clamp 2^-9
    float scale = e4m3_ceil_val(ideal);
    if (lane < 16) {
      float xs = fminf(fmaxf(f / scale, -6.0f), 6.0f);
      int idx = rtn_idx(fabsf(xs), cFP4, 8);
      float q = (xs < 0.f) ? -cFP4[idx] : cFP4[idx];
      xq[k] = q * scale;
    }
  } else {                                            // MXFP8 block of 32
    int b = warp - nlo;
    int k = Klow + 32 * b + lane;
    float f = x[k];
    float a = fabsf(f);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float ideal = a / 448.0f;
    float y = fmaxf(ideal, 1.1754944e-38f);
    int ex; float m = frexpf(y, &ex);
    int e = (m == 0.5f) ? ex - 1 : ex;
    float scale = ldexpf(1.0f, e);
    float xs = fminf(fmaxf(f / scale, -448.0f), 448.0f);
    unsigned char byte = (unsigned char)__nv_cvt_float_to_fp8(xs, __NV_SATFINITE, __NV_E4M3);
    xq[k] = e4m3_to_float(byte) * scale;
  }
}

torch::Tensor quant_act(torch::Tensor x, int64_t Klow, int64_t Khigh) {
  TORCH_CHECK(x.is_cuda(), "x must be cuda");
  init_fp4_once();
  auto xf = x.reshape({-1}).to(torch::kFloat32).contiguous();
  int K = (int)(Klow + Khigh);
  auto xq = torch::empty({K}, torch::dtype(torch::kFloat32).device(x.device()));
  int nwarps = (int)(Klow / 16 + Khigh / 32);
  int tpb = 256;
  int blocks = (nwarps * 32 + tpb - 1) / tpb;
  auto stream = at::cuda::getCurrentCUDAStream();
  quant_act_kernel<<<blocks, tpb, 0, stream>>>(xf.data_ptr<float>(), xq.data_ptr<float>(),
                                               (int)Klow, (int)Khigh);
  CUDA_CHECK(cudaGetLastError());
  return xq;
}


// One block per output row n. Threads stride over K, block-reduce the dot product.
template <int T>
__global__ void gemv_w4a4(
    const float* __restrict__ xq,      // (K) = [xq_low(K_low) | xq_high(K_high)]
    const uint8_t* __restrict__ Wlo,   // (N, K_low/2)  2 NVFP4 codes/byte
    const float*  __restrict__ Slo,    // (N, K_low/16) FP32 NVFP4 block scale
    const uint8_t* __restrict__ Whi,   // (N, K_high)   FP8 E4M3 bytes
    const float*  __restrict__ Shi,    // (N, K_high/32) FP32 MXFP8 block scale
    float* __restrict__ D, int N, int Klow, int Khigh) {
  int n = blockIdx.x;
  if (n >= N) return;
  int tid = threadIdx.x;
  const uint8_t* wl = Wlo + (size_t)n * (Klow / 2);
  const float*   sl = Slo + (size_t)n * (Klow / 16);
  const uint8_t* wh = Whi + (size_t)n * Khigh;
  const float*   sh = Shi + (size_t)n * (Khigh / 32);
  float acc = 0.f;

  // NVFP4 low/main segment: each byte holds codes for k=2b (low nibble) and 2b+1.
  // A block of 16 elements = 8 bytes; byte b is in scale-block b/8.
  for (int b = tid; b < Klow / 2; b += T) {
    uint8_t by = wl[b];
    float sc = sl[b >> 3];
    int k0 = 2 * b;
    acc += xq[k0]     * (fp4_decode(by & 0xF) * sc);
    acc += xq[k0 + 1] * (fp4_decode(by >> 4)  * sc);
  }
  // MXFP8 high segment: 1 FP8 E4M3 byte/elem, 32 elems/scale-block.
  for (int k = tid; k < Khigh; k += T) {
    __nv_fp8_e4m3 f; f.__x = wh[k];
    acc += xq[Klow + k] * (float(f) * sh[k >> 5]);
  }

  // block reduction
  __shared__ float red[T];
  red[tid] = acc;
  __syncthreads();
  for (int s = T >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) D[n] = red[0];
}

// v2: warp-per-row, vectorized uint32 loads (coalesced 128B across 32 lanes),
// activation staged in shared, warp-shuffle reduction. Requires Klow%8==0,
// Khigh%4==0 (always true: Klow%16==0, Khigh%32==0). WARPS rows per block.
template <int WARPS>
__global__ void gemv_w4a4_v2(
    const float* __restrict__ xq,
    const uint8_t* __restrict__ Wlo, const float* __restrict__ Slo,
    const uint8_t* __restrict__ Whi, const float* __restrict__ Shi,
    float* __restrict__ D, int N, int Klow, int Khigh) {
  extern __shared__ float sx[];           // Klow+Khigh activation
  int K = Klow + Khigh;
  for (int i = threadIdx.x; i < K; i += blockDim.x) sx[i] = xq[i];
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n = blockIdx.x * WARPS + warp;
  if (n >= N) return;

  const uint32_t* wl = reinterpret_cast<const uint32_t*>(Wlo + (size_t)n * (Klow / 2));
  const float*    sl = Slo + (size_t)n * (Klow / 16);
  const uint32_t* wh = reinterpret_cast<const uint32_t*>(Whi + (size_t)n * Khigh);
  const float*    sh = Shi + (size_t)n * (Khigh / 32);
  float acc = 0.f;

  // NVFP4: each uint32 = 4 bytes = 8 codes (k0..k0+7), all in one 16-elem scale
  // block -> scale sl[w>>1]. Nibble c of the word is code k0+c.
  int nwl = Klow / 8;
  for (int w = lane; w < nwl; w += 32) {
    uint32_t v = wl[w];
    float sc = sl[w >> 1];
    int k0 = 8 * w;
    #pragma unroll
    for (int c = 0; c < 8; c++)
      acc += sx[k0 + c] * (fp4_decode((v >> (4 * c)) & 0xF) * sc);
  }
  // MXFP8: each uint32 = 4 FP8 bytes (k0..k0+3), all in one 32-elem scale block
  // -> scale sh[w>>3].
  int nwh = Khigh / 4;
  for (int w = lane; w < nwh; w += 32) {
    uint32_t v = wh[w];
    float sc = sh[w >> 3];
    int k0 = Klow + 4 * w;
    #pragma unroll
    for (int c = 0; c < 4; c++) {
      __nv_fp8_e4m3 f; f.__x = (uint8_t)((v >> (8 * c)) & 0xFF);
      acc += sx[k0 + c] * (float(f) * sc);
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) D[n] = acc;
}

// Device helper: quantize the whole activation into shared `sxq` (dequantized
// fp32), bit-exact to quant_act_kernel. Called cooperatively by a CTA so a CTA
// does quant + GEMV in ONE launch (no separate quant_act launch, pure latency at M=1).
__device__ __forceinline__ void quant_act_to_shared(const float* __restrict__ xf,
                                                     float* sxq, int Klow, int Khigh) {
  int nlo = Klow / 16, nhi = Khigh / 32, nblk = nlo + nhi;
  for (int blk = threadIdx.x; blk < nblk; blk += blockDim.x) {
    if (blk < nlo) {                                   // NVFP4 16-elem block
      int k0 = 16 * blk;
      float amax = 0.f;
      #pragma unroll
      for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(xf[k0 + i]));
      float scale = e4m3_ceil_val(fmaxf(amax / 6.0f, 0.001953125f));
      #pragma unroll
      for (int i = 0; i < 16; i++) {
        float xs = fminf(fmaxf(xf[k0 + i] / scale, -6.0f), 6.0f);
        int idx = rtn_idx(fabsf(xs), cFP4, 8);
        sxq[k0 + i] = ((xs < 0.f) ? -cFP4[idx] : cFP4[idx]) * scale;
      }
    } else {                                           // MXFP8 32-elem block
      int k0 = Klow + 32 * (blk - nlo);
      float amax = 0.f;
      for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(xf[k0 + i]));
      float y = fmaxf(amax / 448.0f, 1.1754944e-38f);
      int ex; float m = frexpf(y, &ex);
      float scale = ldexpf(1.0f, (m == 0.5f) ? ex - 1 : ex);
      for (int i = 0; i < 32; i++) {
        float xs = fminf(fmaxf(xf[k0 + i] / scale, -448.0f), 448.0f);
        unsigned char byte = (unsigned char)__nv_cvt_float_to_fp8(xs, __NV_SATFINITE, __NV_E4M3);
        sxq[k0 + i] = e4m3_to_float(byte) * scale;
      }
    }
  }
}

// v3: SINGLE-LAUNCH W4A4 decode GEMV = quant activation into shared + warp-per-row
// weight-streaming GEMV. Removes the separate quant_act launch (pure latency at M=1).
template <int WARPS>
__global__ void gemv_w4a4_v3(
    const float* __restrict__ xf,
    const uint8_t* __restrict__ Wlo, const float* __restrict__ Slo,
    const uint8_t* __restrict__ Whi, const float* __restrict__ Shi,
    float* __restrict__ D, int N, int Klow, int Khigh) {
  extern __shared__ float sxq[];
  quant_act_to_shared(xf, sxq, Klow, Khigh);
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n = blockIdx.x * WARPS + warp;
  if (n >= N) return;
  const uint32_t* wl = reinterpret_cast<const uint32_t*>(Wlo + (size_t)n * (Klow / 2));
  const float*    sl = Slo + (size_t)n * (Klow / 16);
  const uint32_t* wh = reinterpret_cast<const uint32_t*>(Whi + (size_t)n * Khigh);
  const float*    sh = Shi + (size_t)n * (Khigh / 32);
  float acc = 0.f;
  int nwl = Klow / 8;
  for (int w = lane; w < nwl; w += 32) {
    uint32_t v = wl[w]; float sc = sl[w >> 1]; int k0 = 8 * w;
    #pragma unroll
    for (int c = 0; c < 8; c++) acc += sxq[k0 + c] * (fp4_decode((v >> (4 * c)) & 0xF) * sc);
  }
  int nwh = Khigh / 4;
  for (int w = lane; w < nwh; w += 32) {
    uint32_t v = wh[w]; float sc = sh[w >> 3]; int k0 = Klow + 4 * w;
    #pragma unroll
    for (int c = 0; c < 4; c++) {
      __nv_fp8_e4m3 f; f.__x = (uint8_t)((v >> (8 * c)) & 0xFF);
      acc += sxq[k0 + c] * (float(f) * sc);
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) D[n] = acc;
}

// ========================================================================================
// v4 (PERF round-2): warp-per-row, 128-bit (uint4) weight loads, NO shared-memory
// activation staging.  v2/v3 stage the whole K-vector in shared (K floats); on
// large-K shapes (8B down_proj K=14336 -> 57 KB) that caps occupancy at ~1 block/SM
// and starves HBM (the 28-44% ceiling).  The activation is tiny and read-only, so it
// stays resident in L2 across all rows -> read it straight from global.  Freeing the
// shared budget lets many more warps co-reside, raising in-flight memory parallelism.
// uint4 loads = 16 B/lane/instr (vs uint32's 4 B) -> 4x fewer load instructions and
// full 512 B/warp coalesced transactions.  Bit-identical dequant to v2 (same grid).
// WARPS rows per block (tunable); requires Klow%32==0 (8 nibbles/uint4 * 4 =32... here
// each uint4 = 8 bytes*2 = 16 codes-pairs; we require Klow%32==0 and Khigh%16==0,
// always true since Klow%16==0 with high_fraction splits giving multiples of 32/16).
template <int WARPS>
__global__ void gemv_w4a4_v4(
    const float* __restrict__ xq,
    const uint8_t* __restrict__ Wlo, const float* __restrict__ Slo,
    const uint8_t* __restrict__ Whi, const float* __restrict__ Shi,
    float* __restrict__ D, int N, int Klow, int Khigh) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n = blockIdx.x * WARPS + warp;
  if (n >= N) return;
  const uint4* wl = reinterpret_cast<const uint4*>(Wlo + (size_t)n * (Klow / 2));
  const float* sl = Slo + (size_t)n * (Klow / 16);
  const uint4* wh = reinterpret_cast<const uint4*>(Whi + (size_t)n * Khigh);
  const float* sh = Shi + (size_t)n * (Khigh / 32);
  float acc = 0.f;

  // NVFP4 low: each uint4 = 16 bytes = 32 codes (k0..k0+31), spanning two 16-elem
  // scale blocks (first 16 codes -> sl[(k0)/16], next 16 -> sl[(k0+16)/16]).
  int nwl = Klow / 32;                                  // uint4 words in the low segment
  for (int w = lane; w < nwl; w += 32) {
    uint4 v = wl[w];
    int k0 = 32 * w;
    const uint32_t* vv = reinterpret_cast<const uint32_t*>(&v);
    float sc0 = sl[(k0) >> 4];                          // scale for first 16 codes
    float sc1 = sl[(k0 + 16) >> 4];                     // scale for next 16 codes
    #pragma unroll
    for (int q = 0; q < 4; q++) {                       // 4 uint32 -> 8 codes each
      uint32_t word = vv[q];
      float sc = (q < 2) ? sc0 : sc1;                   // codes 0..15 vs 16..31
      int kb = k0 + 8 * q;
      #pragma unroll
      for (int c = 0; c < 8; c++)
        acc += xq[kb + c] * (fp4_decode((word >> (4 * c)) & 0xF) * sc);
    }
  }
  // MXFP8 high: each uint4 = 16 FP8 bytes (k0..k0+15), spanning scale blocks of 32.
  int nwh = Khigh / 16;
  for (int w = lane; w < nwh; w += 32) {
    uint4 v = wh[w];
    int k0 = Klow + 16 * w;
    const uint8_t* vb = reinterpret_cast<const uint8_t*>(&v);
    float sc = sh[(16 * w) >> 5];                       // 16 bytes within one 32-block
    #pragma unroll
    for (int c = 0; c < 16; c++) {
      __nv_fp8_e4m3 f; f.__x = vb[c];
      acc += xq[k0 + c] * (float(f) * sc);
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) D[n] = acc;
}

template <int WARPS>
static torch::Tensor launch_v4(torch::Tensor xq, torch::Tensor Wlo, torch::Tensor Slo,
                               torch::Tensor Whi, torch::Tensor Shi,
                               int64_t N, int64_t Klow, int64_t Khigh) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto D = torch::empty({N}, torch::dtype(torch::kFloat32).device(xq.device()));
  int blocks = ((int)N + WARPS - 1) / WARPS;
  gemv_w4a4_v4<WARPS><<<blocks, WARPS * 32, 0, stream>>>(
      xq.data_ptr<float>(), Wlo.data_ptr<uint8_t>(), Slo.data_ptr<float>(),
      Whi.data_ptr<uint8_t>(), Shi.data_ptr<float>(),
      D.data_ptr<float>(), (int)N, (int)Klow, (int)Khigh);
  CUDA_CHECK(cudaGetLastError());
  return D;
}

torch::Tensor decode_gemv_v4(torch::Tensor xq, torch::Tensor Wlo, torch::Tensor Slo,
                             torch::Tensor Whi, torch::Tensor Shi,
                             int64_t N, int64_t Klow, int64_t Khigh, int64_t warps) {
  TORCH_CHECK(xq.is_cuda() && xq.scalar_type() == torch::kFloat32, "xq must be cuda fp32");
  TORCH_CHECK(Klow % 32 == 0 && Khigh % 16 == 0, "v4 needs Klow%32==0, Khigh%16==0");
  switch (warps) {
    case 4:  return launch_v4<4>(xq, Wlo, Slo, Whi, Shi, N, Klow, Khigh);
    case 8:  return launch_v4<8>(xq, Wlo, Slo, Whi, Shi, N, Klow, Khigh);
    case 16: return launch_v4<16>(xq, Wlo, Slo, Whi, Shi, N, Klow, Khigh);
    case 32: return launch_v4<32>(xq, Wlo, Slo, Whi, Shi, N, Klow, Khigh);
    default: return launch_v4<8>(xq, Wlo, Slo, Whi, Shi, N, Klow, Khigh);
  }
}

torch::Tensor decode_gemv(torch::Tensor xq, torch::Tensor Wlo, torch::Tensor Slo,
                          torch::Tensor Whi, torch::Tensor Shi,
                          int64_t N, int64_t Klow, int64_t Khigh) {
  TORCH_CHECK(xq.is_cuda() && xq.scalar_type() == torch::kFloat32, "xq must be cuda fp32");
  TORCH_CHECK(xq.numel() == Klow + Khigh, "xq length must be Klow+Khigh");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto D = torch::empty({N}, torch::dtype(torch::kFloat32).device(xq.device()));
  constexpr int T = 256;
  gemv_w4a4<T><<<(int)N, T, 0, stream>>>(
      xq.data_ptr<float>(), Wlo.data_ptr<uint8_t>(), Slo.data_ptr<float>(),
      Whi.data_ptr<uint8_t>(), Shi.data_ptr<float>(),
      D.data_ptr<float>(), (int)N, (int)Klow, (int)Khigh);
  CUDA_CHECK(cudaGetLastError());
  return D;
}

torch::Tensor decode_gemv_v2(torch::Tensor xq, torch::Tensor Wlo, torch::Tensor Slo,
                             torch::Tensor Whi, torch::Tensor Shi,
                             int64_t N, int64_t Klow, int64_t Khigh) {
  TORCH_CHECK(xq.is_cuda() && xq.scalar_type() == torch::kFloat32, "xq must be cuda fp32");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto D = torch::empty({N}, torch::dtype(torch::kFloat32).device(xq.device()));
  constexpr int WARPS = 8;
  int blocks = ((int)N + WARPS - 1) / WARPS;
  size_t smem = (size_t)(Klow + Khigh) * sizeof(float);
  gemv_w4a4_v2<WARPS><<<blocks, WARPS * 32, smem, stream>>>(
      xq.data_ptr<float>(), Wlo.data_ptr<uint8_t>(), Slo.data_ptr<float>(),
      Whi.data_ptr<uint8_t>(), Shi.data_ptr<float>(),
      D.data_ptr<float>(), (int)N, (int)Klow, (int)Khigh);
  CUDA_CHECK(cudaGetLastError());
  return D;
}

// v5 (PERF round-2): the WINNING shared-staged design (same body as v2) but with
// (a) a smem opt-in so it runs when Klow+Khigh floats exceed 48 KB (8B down_proj,
// K=14336 -> 57 KB — v2 crashed with "invalid argument" there), and (b) a tunable
// rows-per-block (WARPS) so the HBM roof can be swept per shape.  v4's no-shared /
// uint4 variant was tried and LOST on every shape (shared staging of the reused
// activation beats re-reading it from L2), so this keeps shared staging.
template <int WARPS>
static torch::Tensor launch_v5(torch::Tensor xq, torch::Tensor Wlo, torch::Tensor Slo,
                               torch::Tensor Whi, torch::Tensor Shi,
                               int64_t N, int64_t Klow, int64_t Khigh) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto D = torch::empty({N}, torch::dtype(torch::kFloat32).device(xq.device()));
  int blocks = ((int)N + WARPS - 1) / WARPS;
  size_t smem = (size_t)(Klow + Khigh) * sizeof(float);
  static size_t attr_smem = 0;
  if (smem > 48 * 1024 && smem > attr_smem) {
    CUDA_CHECK(cudaSetDevice(xq.get_device()));
    CUDA_CHECK(cudaFuncSetAttribute(gemv_w4a4_v2<WARPS>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    attr_smem = smem;
  }
  gemv_w4a4_v2<WARPS><<<blocks, WARPS * 32, smem, stream>>>(
      xq.data_ptr<float>(), Wlo.data_ptr<uint8_t>(), Slo.data_ptr<float>(),
      Whi.data_ptr<uint8_t>(), Shi.data_ptr<float>(),
      D.data_ptr<float>(), (int)N, (int)Klow, (int)Khigh);
  CUDA_CHECK(cudaGetLastError());
  return D;
}

torch::Tensor decode_gemv_v5(torch::Tensor xq, torch::Tensor Wlo, torch::Tensor Slo,
                             torch::Tensor Whi, torch::Tensor Shi,
                             int64_t N, int64_t Klow, int64_t Khigh, int64_t warps) {
  TORCH_CHECK(xq.is_cuda() && xq.scalar_type() == torch::kFloat32, "xq must be cuda fp32");
  switch (warps) {
    case 4:  return launch_v5<4>(xq, Wlo, Slo, Whi, Shi, N, Klow, Khigh);
    case 8:  return launch_v5<8>(xq, Wlo, Slo, Whi, Shi, N, Klow, Khigh);
    case 16: return launch_v5<16>(xq, Wlo, Slo, Whi, Shi, N, Klow, Khigh);
    case 24: return launch_v5<24>(xq, Wlo, Slo, Whi, Shi, N, Klow, Khigh);
    case 32: return launch_v5<32>(xq, Wlo, Slo, Whi, Shi, N, Klow, Khigh);
    default: return launch_v5<8>(xq, Wlo, Slo, Whi, Shi, N, Klow, Khigh);
  }
}

template <int WARPS>
static torch::Tensor launch_v3(torch::Tensor xf, torch::Tensor Wlo, torch::Tensor Slo,
                               torch::Tensor Whi, torch::Tensor Shi, torch::Tensor D,
                               int N, int Klow, int Khigh, int devid, cudaStream_t stream) {
  int blocks = (N + WARPS - 1) / WARPS;
  size_t smem = (size_t)(Klow + Khigh) * sizeof(float);
  // SMEM opt-in (keyed on largest K seen): required for 8B down_proj K=14336
  // whose staging buffer (57,344 B) exceeds the 48 KB default dynamic ceiling.
  static size_t attr_smem[33] = {0};
  if (smem > attr_smem[WARPS]) {
    CUDA_CHECK(cudaSetDevice(devid));
    CUDA_CHECK(cudaFuncSetAttribute(gemv_w4a4_v3<WARPS>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    int maxs = 0;
    cudaDeviceGetAttribute(&maxs, cudaDevAttrMaxSharedMemoryPerBlockOptin, devid);
    TORCH_CHECK((size_t)maxs >= smem,
                "decode_gemv_fused: smem ", smem, " B exceeds device opt-in max ", maxs, " B");
    attr_smem[WARPS] = smem;
  }
  gemv_w4a4_v3<WARPS><<<blocks, WARPS * 32, smem, stream>>>(
      xf.data_ptr<float>(), Wlo.data_ptr<uint8_t>(), Slo.data_ptr<float>(),
      Whi.data_ptr<uint8_t>(), Shi.data_ptr<float>(),
      D.data_ptr<float>(), N, Klow, Khigh);
  CUDA_CHECK(cudaGetLastError());
  return D;
}

// ========================================================================================
// W4A16 DECODE GEMV (round-12): weight-only path for M=1.  At decode the GEMV is
// bandwidth/latency-bound, NOT compute-bound, so quantizing the activation (W4A4)
// buys no tensor-core throughput and only costs an extra quantize pass + loses
// accuracy.  This kernel keeps the activation in its native fp16/bf16/fp32 and
// dequantizes ONLY the packed 4-bit(NVFP4)/8-bit(MXFP8) weights in-register
// (Marlin-style), dotting against the EXACT activation.  Weight HBM traffic is
// identical to the W4A4 decode GEMV (same packed bytes), but there is no activation
// quantization — a single launch, exact activations.
//
// Body is gemv_w4a4_v2 verbatim EXCEPT the shared-staging loop reads the raw
// activation (converted to fp32) instead of a pre-quantized xq vector.  Templated
// on the activation C-type AT so fp16 / bf16 / fp32 activations all work with no
// extra torch convert launch.
__device__ __forceinline__ float a_to_f(const __half v)         { return __half2float(v); }
__device__ __forceinline__ float a_to_f(const __nv_bfloat16 v)  { return __bfloat162float(v); }
__device__ __forceinline__ float a_to_f(const float v)          { return v; }

template <int WARPS, typename AT>
__global__ void gemv_w4a16(
    const AT* __restrict__ x,                // (K) raw activation, NOT quantized
    const uint8_t* __restrict__ Wlo, const float* __restrict__ Slo,
    const uint8_t* __restrict__ Whi, const float* __restrict__ Shi,
    float* __restrict__ D, int N, int Klow, int Khigh) {
  extern __shared__ float sx[];            // Klow+Khigh activation (fp32, exact)
  int K = Klow + Khigh;
  for (int i = threadIdx.x; i < K; i += blockDim.x) sx[i] = a_to_f(x[i]);
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n = blockIdx.x * WARPS + warp;
  if (n >= N) return;

  const uint32_t* wl = reinterpret_cast<const uint32_t*>(Wlo + (size_t)n * (Klow / 2));
  const float*    sl = Slo + (size_t)n * (Klow / 16);
  const uint32_t* wh = reinterpret_cast<const uint32_t*>(Whi + (size_t)n * Khigh);
  const float*    sh = Shi + (size_t)n * (Khigh / 32);
  float acc = 0.f;

  int nwl = Klow / 8;
  for (int w = lane; w < nwl; w += 32) {
    uint32_t v = wl[w];
    float sc = sl[w >> 1];
    int k0 = 8 * w;
    #pragma unroll
    for (int c = 0; c < 8; c++)
      acc += sx[k0 + c] * (fp4_decode((v >> (4 * c)) & 0xF) * sc);
  }
  int nwh = Khigh / 4;
  for (int w = lane; w < nwh; w += 32) {
    uint32_t v = wh[w];
    float sc = sh[w >> 3];
    int k0 = Klow + 4 * w;
    #pragma unroll
    for (int c = 0; c < 4; c++) {
      __nv_fp8_e4m3 f; f.__x = (uint8_t)((v >> (8 * c)) & 0xFF);
      acc += sx[k0 + c] * (float(f) * sc);
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) D[n] = acc;
}

template <int WARPS, typename AT>
static void launch_w4a16(const AT* xptr, torch::Tensor Wlo, torch::Tensor Slo,
                         torch::Tensor Whi, torch::Tensor Shi, torch::Tensor D,
                         int N, int Klow, int Khigh, int devid, cudaStream_t stream) {
  int blocks = (N + WARPS - 1) / WARPS;
  size_t smem = (size_t)(Klow + Khigh) * sizeof(float);
  static size_t attr_smem[33] = {0};
  if (smem > 48 * 1024 && smem > attr_smem[WARPS]) {
    CUDA_CHECK(cudaSetDevice(devid));
    CUDA_CHECK(cudaFuncSetAttribute(gemv_w4a16<WARPS, AT>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    int maxs = 0;
    cudaDeviceGetAttribute(&maxs, cudaDevAttrMaxSharedMemoryPerBlockOptin, devid);
    TORCH_CHECK((size_t)maxs >= smem,
                "decode_gemv_w4a16: smem ", smem, " B exceeds device opt-in max ", maxs, " B");
    attr_smem[WARPS] = smem;
  }
  gemv_w4a16<WARPS, AT><<<blocks, WARPS * 32, smem, stream>>>(
      xptr, Wlo.data_ptr<uint8_t>(), Slo.data_ptr<float>(),
      Whi.data_ptr<uint8_t>(), Shi.data_ptr<float>(),
      D.data_ptr<float>(), N, Klow, Khigh);
  CUDA_CHECK(cudaGetLastError());
}

template <typename AT>
static torch::Tensor dispatch_w4a16(const AT* xptr, torch::Tensor Wlo, torch::Tensor Slo,
                                    torch::Tensor Whi, torch::Tensor Shi, torch::Tensor D,
                                    int N, int Kl, int Kh, int dev, cudaStream_t stream,
                                    int64_t warps) {
  switch (warps) {
    case 4:  launch_w4a16<4>(xptr, Wlo, Slo, Whi, Shi, D, N, Kl, Kh, dev, stream); break;
    case 8:  launch_w4a16<8>(xptr, Wlo, Slo, Whi, Shi, D, N, Kl, Kh, dev, stream); break;
    case 16: launch_w4a16<16>(xptr, Wlo, Slo, Whi, Shi, D, N, Kl, Kh, dev, stream); break;
    case 24: launch_w4a16<24>(xptr, Wlo, Slo, Whi, Shi, D, N, Kl, Kh, dev, stream); break;
    case 32: launch_w4a16<32>(xptr, Wlo, Slo, Whi, Shi, D, N, Kl, Kh, dev, stream); break;
    default: launch_w4a16<8>(xptr, Wlo, Slo, Whi, Shi, D, N, Kl, Kh, dev, stream); break;
  }
  return D;
}

// ========================================================================================
// BATCHED W4A16 DECODE GEMV (vllm-perf round-1): small-M decode steps (vLLM CUDA-graph
// capture pads decode batches to fixed sizes 1,2,4,8,16,...).  At M=2..32 the 128x128
// fused tensor-core GEMM is still ~1/8..1/64 tile-utilized, while the weight-streaming
// GEMV amortizes the SAME packed weight bytes over all M rows (decode stays weight-
// bandwidth-bound).  Design: warp-per-output-row n with MB fp32 accumulators per lane;
// the (M,K) activation is staged K-TILED in shared memory (KT elems per row, smem =
// M*KT*4 <= 48 KB) so every weight byte is dotted against all M rows from smem, and the
// activation gmem traffic stays one L2-resident pass per block.  Per-lane accumulation
// order over k is IDENTICAL to gemv_w4a16 (same lane-strided uint32 walk, tiles visited
// in ascending k), so each row's result is bit-identical to the M=1 kernel.
template <int WARPS, int MB, typename AT>
__global__ void gemv_w4a16_mb(
    const AT* __restrict__ x,                // (M, K) raw activation, row-major
    const uint8_t* __restrict__ Wlo, const float* __restrict__ Slo,
    const uint8_t* __restrict__ Whi, const float* __restrict__ Shi,
    float* __restrict__ D,                   // (M, N) fp32
    int M, int N, int Klow, int Khigh, int KT) {
  extern __shared__ float sx[];              // M * KT staged activation tile
  const int K = Klow + Khigh;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * WARPS + warp;
  const bool rowok = (n < N);                // NO early return: all threads must reach
                                             // the __syncthreads() in the tile loops.
  const uint32_t* wl = rowok ? reinterpret_cast<const uint32_t*>(Wlo + (size_t)n * (Klow / 2)) : nullptr;
  const float*    sl = rowok ? Slo + (size_t)n * (Klow / 16) : nullptr;
  const uint32_t* wh = rowok ? reinterpret_cast<const uint32_t*>(Whi + (size_t)n * Khigh) : nullptr;
  const float*    sh = rowok ? Shi + (size_t)n * (Khigh / 32) : nullptr;
  float acc[MB];
  #pragma unroll
  for (int m = 0; m < MB; ++m) acc[m] = 0.f;

  // ---- NVFP4 low segment, K-tiled (each uint32 = 8 codes; scale sl[w>>1]) ----
  for (int k0 = 0; k0 < Klow; k0 += KT) {
    int kt = min(KT, Klow - k0);
    __syncthreads();                         // previous tile fully consumed
    for (int idx = threadIdx.x; idx < M * kt; idx += blockDim.x) {
      int m = idx / kt, i = idx - m * kt;
      sx[m * KT + i] = a_to_f(x[(size_t)m * K + k0 + i]);
    }
    __syncthreads();
    if (rowok) {
      int w0 = k0 >> 3, w1 = (k0 + kt) >> 3;  // global uint32 index range of tile
      for (int w = w0 + lane; w < w1; w += 32) {
        uint32_t v = wl[w];
        float sc = sl[w >> 1];
        int i0 = 8 * w - k0;                  // tile-local element index (mult of 8)
        float wv[8];
        #pragma unroll
        for (int c = 0; c < 8; c++) wv[c] = fp4_decode((v >> (4 * c)) & 0xF) * sc;
        #pragma unroll
        for (int m = 0; m < MB; ++m) {
          if (m < M) {
            const float4* s4 = reinterpret_cast<const float4*>(sx + m * KT + i0);
            float4 a0 = s4[0], a1 = s4[1];    // 2x LDS.128 instead of 8 scalar LDS
            acc[m] += a0.x * wv[0]; acc[m] += a0.y * wv[1];
            acc[m] += a0.z * wv[2]; acc[m] += a0.w * wv[3];
            acc[m] += a1.x * wv[4]; acc[m] += a1.y * wv[5];
            acc[m] += a1.z * wv[6]; acc[m] += a1.w * wv[7];
          }
        }
      }
    }
  }
  // ---- MXFP8 high segment, K-tiled (each uint32 = 4 FP8 bytes; scale sh[w>>3]) ----
  for (int k0 = 0; k0 < Khigh; k0 += KT) {
    int kt = min(KT, Khigh - k0);
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * kt; idx += blockDim.x) {
      int m = idx / kt, i = idx - m * kt;
      sx[m * KT + i] = a_to_f(x[(size_t)m * K + Klow + k0 + i]);
    }
    __syncthreads();
    if (rowok) {
      int w0 = k0 >> 2, w1 = (k0 + kt) >> 2;
      for (int w = w0 + lane; w < w1; w += 32) {
        uint32_t v = wh[w];
        float sc = sh[w >> 3];
        int i0 = 4 * w - k0;                  // tile-local element index (mult of 4)
        float wv[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
          __nv_fp8_e4m3 f; f.__x = (uint8_t)((v >> (8 * c)) & 0xFF);
          wv[c] = float(f) * sc;
        }
        #pragma unroll
        for (int m = 0; m < MB; ++m) {
          if (m < M) {
            const float4 a0 = *reinterpret_cast<const float4*>(sx + m * KT + i0);
            acc[m] += a0.x * wv[0]; acc[m] += a0.y * wv[1];
            acc[m] += a0.z * wv[2]; acc[m] += a0.w * wv[3];
          }
        }
      }
    }
  }
  // per-m warp reduction + store (D is (M, N))
  #pragma unroll
  for (int m = 0; m < MB; ++m) {
    float a = acc[m];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
    if (lane == 0 && rowok && m < M) D[(size_t)m * N + n] = a;
  }
}

template <int WARPS, int MB, typename AT>
static void launch_w4a16_mb(const AT* xptr, torch::Tensor Wlo, torch::Tensor Slo,
                            torch::Tensor Whi, torch::Tensor Shi, torch::Tensor D,
                            int M, int N, int Klow, int Khigh, cudaStream_t stream) {
  // KT: largest multiple of 256 with M*KT*4 <= 48 KB (no smem opt-in), cap 4096.
  // The 256 alignment keeps each tile's uint32 count a multiple of 32 in BOTH
  // segments, so every lane's k-walk is the exact strided sequence of the M=1
  // kernel -> per-row results stay bit-identical to decode_gemv_w4a16.
  int KT = ((48 * 1024 / 4) / M) & ~255;
  if (KT > 4096) KT = 4096;
  TORCH_CHECK(KT >= 256, "decode_gemv_w4a16_mb: M too large for 48KB staging");
  size_t smem = (size_t)M * KT * sizeof(float);
  int blocks = (N + WARPS - 1) / WARPS;
  gemv_w4a16_mb<WARPS, MB, AT><<<blocks, WARPS * 32, smem, stream>>>(
      xptr, Wlo.data_ptr<uint8_t>(), Slo.data_ptr<float>(),
      Whi.data_ptr<uint8_t>(), Shi.data_ptr<float>(),
      D.data_ptr<float>(), M, N, Klow, Khigh, KT);
  CUDA_CHECK(cudaGetLastError());
}

template <int MB, typename AT>
static void dispatch_w4a16_mb_w(const AT* xptr, torch::Tensor Wlo, torch::Tensor Slo,
                                torch::Tensor Whi, torch::Tensor Shi, torch::Tensor D,
                                int M, int N, int Kl, int Kh, cudaStream_t s, int64_t warps) {
  switch (warps) {
    case 8:  launch_w4a16_mb<8,  MB>(xptr, Wlo, Slo, Whi, Shi, D, M, N, Kl, Kh, s); break;
    case 32: launch_w4a16_mb<32, MB>(xptr, Wlo, Slo, Whi, Shi, D, M, N, Kl, Kh, s); break;
    default: launch_w4a16_mb<16, MB>(xptr, Wlo, Slo, Whi, Shi, D, M, N, Kl, Kh, s); break;
  }
}

template <typename AT>
static void dispatch_w4a16_mb(const AT* xptr, torch::Tensor Wlo, torch::Tensor Slo,
                              torch::Tensor Whi, torch::Tensor Shi, torch::Tensor D,
                              int M, int N, int Kl, int Kh, cudaStream_t s, int64_t warps) {
  if      (M <= 2)  dispatch_w4a16_mb_w<2>(xptr, Wlo, Slo, Whi, Shi, D, M, N, Kl, Kh, s, warps);
  else if (M <= 4)  dispatch_w4a16_mb_w<4>(xptr, Wlo, Slo, Whi, Shi, D, M, N, Kl, Kh, s, warps);
  else if (M <= 8)  dispatch_w4a16_mb_w<8>(xptr, Wlo, Slo, Whi, Shi, D, M, N, Kl, Kh, s, warps);
  else              dispatch_w4a16_mb_w<16>(xptr, Wlo, Slo, Whi, Shi, D, M, N, Kl, Kh, s, warps);
}

torch::Tensor decode_gemv_w4a16_mb(torch::Tensor x, torch::Tensor Wlo, torch::Tensor Slo,
                                   torch::Tensor Whi, torch::Tensor Shi,
                                   int64_t N, int64_t Klow, int64_t Khigh, int64_t warps) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "x must be cuda (M, K)");
  auto xc = x.contiguous();
  int M = (int)xc.size(0);
  TORCH_CHECK(xc.size(1) == Klow + Khigh, "x K dim must be Klow+Khigh");
  TORCH_CHECK(M >= 1 && M <= 16, "decode_gemv_w4a16_mb supports M in [1, 16]");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto D = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(x.device()));
  int Ni = (int)N, Kl = (int)Klow, Kh = (int)Khigh;
  if (xc.scalar_type() == torch::kFloat16)
    dispatch_w4a16_mb(reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()),
                      Wlo, Slo, Whi, Shi, D, M, Ni, Kl, Kh, stream, warps);
  else if (xc.scalar_type() == torch::kBFloat16)
    dispatch_w4a16_mb(reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>()),
                      Wlo, Slo, Whi, Shi, D, M, Ni, Kl, Kh, stream, warps);
  else {
    TORCH_CHECK(xc.scalar_type() == torch::kFloat32, "x must be fp16/bf16/fp32");
    dispatch_w4a16_mb(xc.data_ptr<float>(), Wlo, Slo, Whi, Shi, D, M, Ni, Kl, Kh, stream, warps);
  }
  return D;
}

torch::Tensor decode_gemv_w4a16(torch::Tensor x, torch::Tensor Wlo, torch::Tensor Slo,
                                torch::Tensor Whi, torch::Tensor Shi,
                                int64_t N, int64_t Klow, int64_t Khigh, int64_t warps) {
  TORCH_CHECK(x.is_cuda(), "x must be cuda");
  auto xc = x.reshape({-1}).contiguous();
  TORCH_CHECK(xc.numel() == Klow + Khigh, "x length must be Klow+Khigh");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto D = torch::empty({N}, torch::dtype(torch::kFloat32).device(x.device()));
  int Ni = (int)N, Kl = (int)Klow, Kh = (int)Khigh, dev = x.get_device();
  if (xc.scalar_type() == torch::kFloat16)
    return dispatch_w4a16(reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()),
                          Wlo, Slo, Whi, Shi, D, Ni, Kl, Kh, dev, stream, warps);
  if (xc.scalar_type() == torch::kBFloat16)
    return dispatch_w4a16(reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>()),
                          Wlo, Slo, Whi, Shi, D, Ni, Kl, Kh, dev, stream, warps);
  TORCH_CHECK(xc.scalar_type() == torch::kFloat32, "x must be fp16/bf16/fp32");
  return dispatch_w4a16(xc.data_ptr<float>(), Wlo, Slo, Whi, Shi, D, Ni, Kl, Kh, dev, stream, warps);
}

torch::Tensor decode_gemv_fused(torch::Tensor x, torch::Tensor Wlo, torch::Tensor Slo,
                                torch::Tensor Whi, torch::Tensor Shi,
                                int64_t N, int64_t Klow, int64_t Khigh, int64_t warps) {
  TORCH_CHECK(x.is_cuda(), "x must be cuda");
  init_fp4_once();
  auto xf = x.reshape({-1}).to(torch::kFloat32).contiguous();
  auto stream = at::cuda::getCurrentCUDAStream();
  auto D = torch::empty({N}, torch::dtype(torch::kFloat32).device(x.device()));
  int Ni = (int)N, Kl = (int)Klow, Kh = (int)Khigh, dev = x.get_device();
  // Win 4 (round-3): tunable rows/block. Large N saturates HBM with more warps;
  // bit-identical to the fixed-8-warp v3 (same kernel, same FP32 accumulate).
  switch (warps) {
    case 4:  return launch_v3<4>(xf, Wlo, Slo, Whi, Shi, D, Ni, Kl, Kh, dev, stream);
    case 8:  return launch_v3<8>(xf, Wlo, Slo, Whi, Shi, D, Ni, Kl, Kh, dev, stream);
    case 16: return launch_v3<16>(xf, Wlo, Slo, Whi, Shi, D, Ni, Kl, Kh, dev, stream);
    case 24: return launch_v3<24>(xf, Wlo, Slo, Whi, Shi, D, Ni, Kl, Kh, dev, stream);
    case 32: return launch_v3<32>(xf, Wlo, Slo, Whi, Shi, D, Ni, Kl, Kh, dev, stream);
    default: return launch_v3<8>(xf, Wlo, Slo, Whi, Shi, D, Ni, Kl, Kh, dev, stream);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_gemv", &decode_gemv,
        "memory-bound W4A4 decode GEMV (v1: block-per-row): D(N)=sum_k xq[k]*deq[n,k]",
        pybind11::arg("xq"), pybind11::arg("Wlo"), pybind11::arg("Slo"),
        pybind11::arg("Whi"), pybind11::arg("Shi"),
        pybind11::arg("N"), pybind11::arg("Klow"), pybind11::arg("Khigh"));
  m.def("decode_gemv_v2", &decode_gemv_v2,
        "memory-bound W4A4 decode GEMV (v2: warp-per-row, vectorized, shared x)",
        pybind11::arg("xq"), pybind11::arg("Wlo"), pybind11::arg("Slo"),
        pybind11::arg("Whi"), pybind11::arg("Shi"),
        pybind11::arg("N"), pybind11::arg("Klow"), pybind11::arg("Khigh"));
  m.def("decode_gemv_v4", &decode_gemv_v4,
        "PERF-r2 W4A4 decode GEMV (v4: warp-per-row, uint4/128-bit loads, NO shared "
        "activation staging -> high occupancy on large-K). warps = rows/block (4/8/16/32)",
        pybind11::arg("xq"), pybind11::arg("Wlo"), pybind11::arg("Slo"),
        pybind11::arg("Whi"), pybind11::arg("Shi"),
        pybind11::arg("N"), pybind11::arg("Klow"), pybind11::arg("Khigh"),
        pybind11::arg("warps") = 8);
  m.def("decode_gemv_v5", &decode_gemv_v5,
        "PERF-r2 W4A4 decode GEMV (v5: shared-staged, uint32, smem opt-in for large K, "
        "tunable rows/block). warps in {4,8,16,24,32}",
        pybind11::arg("xq"), pybind11::arg("Wlo"), pybind11::arg("Slo"),
        pybind11::arg("Whi"), pybind11::arg("Shi"),
        pybind11::arg("N"), pybind11::arg("Klow"), pybind11::arg("Khigh"),
        pybind11::arg("warps") = 8);
  m.def("decode_gemv_fused", &decode_gemv_fused,
        "SINGLE-LAUNCH W4A4 decode GEMV: quant activation (raw x) + weight-stream GEMV "
        "(round-3: tunable rows/block via warps in {4,8,16,24,32})",
        pybind11::arg("x"), pybind11::arg("Wlo"), pybind11::arg("Slo"),
        pybind11::arg("Whi"), pybind11::arg("Shi"),
        pybind11::arg("N"), pybind11::arg("Klow"), pybind11::arg("Khigh"),
        pybind11::arg("warps") = 8);
  m.def("decode_gemv_w4a16", &decode_gemv_w4a16,
        "WEIGHT-ONLY W4A16 decode GEMV (round-12): dequant packed 4/8-bit weights "
        "in-register, dot against the EXACT fp16/bf16/fp32 activation (no activation "
        "quantize). Single launch, tunable rows/block via warps in {4,8,16,24,32}",
        pybind11::arg("x"), pybind11::arg("Wlo"), pybind11::arg("Slo"),
        pybind11::arg("Whi"), pybind11::arg("Shi"),
        pybind11::arg("N"), pybind11::arg("Klow"), pybind11::arg("Khigh"),
        pybind11::arg("warps") = 8);
  m.def("decode_gemv_w4a16_mb", &decode_gemv_w4a16_mb,
        "BATCHED weight-only W4A16 decode GEMV (vllm-perf r1): x (M,K) raw activation, "
        "M in [1,16] (vLLM captured decode batch sizes). Warp-per-row, K-tiled smem "
        "activation staging; per-row results bit-identical to decode_gemv_w4a16. "
        "Returns (M,N) fp32. warps = rows/block in {8,16,32}",
        pybind11::arg("x"), pybind11::arg("Wlo"), pybind11::arg("Slo"),
        pybind11::arg("Whi"), pybind11::arg("Shi"),
        pybind11::arg("N"), pybind11::arg("Klow"), pybind11::arg("Khigh"),
        pybind11::arg("warps") = 16);
  m.def("quant_act", &quant_act,
        "fast in-kernel activation quantize -> dequantized fp32 (K,), bit-exact to "
        "fake_quantize_nvfp4/mxfp8 (M=1)",
        pybind11::arg("x"), pybind11::arg("Klow"), pybind11::arg("Khigh"));
}
