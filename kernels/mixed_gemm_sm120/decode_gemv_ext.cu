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

torch::Tensor decode_gemv_fused(torch::Tensor x, torch::Tensor Wlo, torch::Tensor Slo,
                                torch::Tensor Whi, torch::Tensor Shi,
                                int64_t N, int64_t Klow, int64_t Khigh) {
  TORCH_CHECK(x.is_cuda(), "x must be cuda");
  init_fp4_once();
  auto xf = x.reshape({-1}).to(torch::kFloat32).contiguous();
  auto stream = at::cuda::getCurrentCUDAStream();
  auto D = torch::empty({N}, torch::dtype(torch::kFloat32).device(x.device()));
  constexpr int WARPS = 8;
  int blocks = ((int)N + WARPS - 1) / WARPS;
  size_t smem = (size_t)(Klow + Khigh) * sizeof(float);
  gemv_w4a4_v3<WARPS><<<blocks, WARPS * 32, smem, stream>>>(
      xf.data_ptr<float>(), Wlo.data_ptr<uint8_t>(), Slo.data_ptr<float>(),
      Whi.data_ptr<uint8_t>(), Shi.data_ptr<float>(),
      D.data_ptr<float>(), (int)N, (int)Klow, (int)Khigh);
  CUDA_CHECK(cudaGetLastError());
  return D;
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
  m.def("decode_gemv_fused", &decode_gemv_fused,
        "SINGLE-LAUNCH W4A4 decode GEMV: quant activation (raw x) + weight-stream GEMV",
        pybind11::arg("x"), pybind11::arg("Wlo"), pybind11::arg("Slo"),
        pybind11::arg("Whi"), pybind11::arg("Shi"),
        pybind11::arg("N"), pybind11::arg("Klow"), pybind11::arg("Khigh"));
  m.def("quant_act", &quant_act,
        "fast in-kernel activation quantize -> dequantized fp32 (K,), bit-exact to "
        "fake_quantize_nvfp4/mxfp8 (M=1)",
        pybind11::arg("x"), pybind11::arg("Klow"), pybind11::arg("Khigh"));
}
