// add_rmsnorm_pack_ext.cu — round-26 lever B: fuse residual-add + RMSNorm into
// the activation quantize+pack PRODUCER (SM120).
//
// Problem. Per decoder layer the prefill path ran, for each of the two norms:
//
//   r     = residual + x                       # separate elementwise kernel (fp16)
//   xs    = r.to(fp32)                         # separate cast
//   var   = xs.pow(2).mean(-1)                 # separate reduce kernel
//   xn    = (xs * rsqrt(var + eps)).to(fp16)   # 3 more elementwise kernels
//   out   = weight * xn
//   then FOR EACH consumer Linear (q,k,v / gate,up):
//     pack(xn)                                 # NVFP4+MXFP8 quantize+pack launch
//
// i.e. ~11 launches plus a full [M,N] norm-output tensor per norm, and the SAME
// norm output packed 3x (q,k,v) / 2x (gate,up). This kernel does
//
//   residual_out = to_add + x                  (written ONCE, reference values)
//   xn           = round_to_fp16((x + to_add) * rsqrt(mean((x+to_add)^2) + eps))
//   (codes_lo, sfa_lo, codes_hi, sfa_hi) = quant_pack(xn)   [verbatim qpf device code]
//
// in ONE launch, never materialising the fp32 norm input nor a separate fp16
// norm-output tensor (the latter lives in SMEM and is consumed by the pack stage)
// — the same pattern that made hadamard_pack_ext.cu work.
//
// NUMERICS. Reference is transformers' LlamaRMSNorm (weight == 1 after
// fuse_layer_norms):
//   t = x.to(fp32); var = t.pow(2).mean(-1); t *= torch.rsqrt(var + eps);
//   out = weight * t.to(input_dtype)
// The kernel reproduces: the residual add in the input dtype (one rounding), the
// fp32 square / row mean / rsqrt(var+eps), and the single round back to fp16
// before the pack. The pack is the SAME device code as quant_pack_float_ext.cu
// (per-block absmax -> round-up E4M3 / UE8M0 scale -> grid RTN / E4M3 cast ->
// interleaved CUTLASS SFA atom layout).
//
// CONSUMER CONTRACT. The returned packed tensors are byte-identical in layout and
// meaning to `quant_pack_float_mixed(xn, M, N_gemm, K_low, K_high)`, i.e. the
// fused GEMM's A operand + interleaved SFA. N_gemm (the consumer's output width)
// is required because the SFA atom layout depends on it; the DATA pack does not.
// Consumers that share an input (k/v, gate/up) share ONE pack: pass the same
// N_gemm and reuse the returned tensors (see MixedQuantLinear.forward_norm_pack).
//
// PROTECTED files untouched: standalone extension.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

using namespace cute;

static inline void arn_check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// SM120 block-scaled SFA atom layouts (config-only, same as qpf/hpk).
using BSConfigNV = cutlass::detail::Sm1xxBlockScaledConfig<16>;
using BSConfigMX = cutlass::detail::Sm1xxBlockScaledConfig<32>;

static inline torch::TensorOptions u8opts(torch::Device dev) {
  return torch::TensorOptions().dtype(torch::kUInt8).device(dev);
}

// SF scatter: one thread per LOGICAL SF byte (row, block) -> its physical
// offset in the CUTLASS interleaved atom layout (same kernel as the GEMM ext).
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

// ====================== device helpers (verbatim from quant_pack_float_ext.cu) ==
__constant__ float cFP4_val_a[8];
static bool g_grids_ready_a = false;
static void init_grids_once_a() {
  if (g_grids_ready_a) return;
  float fp4[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  arn_check_cuda(cudaMemcpyToSymbol(cFP4_val_a, fp4, sizeof(fp4)), "cpy fp4");
  g_grids_ready_a = true;
}

__device__ __forceinline__ float to_float_a(const half v)          { return __half2float(v); }
__device__ __forceinline__ float to_float_a(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ half   add_dtype_a(half a, half b)      { return __hadd(a, b); }
__device__ __forceinline__ __nv_bfloat16 add_dtype_a(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }
// the reference's `t.to(input_dtype)` (one round), kept as a float for the pack
__device__ __forceinline__ float round_in_dtype_a(float v, const half)          { return __half2float(__float2half_rn(v)); }
__device__ __forceinline__ float round_in_dtype_a(float v, const __nv_bfloat16) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float e4m3_to_float_a(unsigned char b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}
__device__ __forceinline__ void e4m3_ceil_a(float ideal, unsigned char& byte, float& val) {
  unsigned char b = (unsigned char)__nv_cvt_float_to_fp8(ideal, __NV_SATFINITE, __NV_E4M3);
  float v = e4m3_to_float_a(b);
  if (v < ideal && b < 0x7E) { ++b; v = e4m3_to_float_a(b); }
  byte = b; val = v;
}
__device__ __forceinline__ int rtn_idx_a(float absx, const float* v, int n) {
  int lo = 0, hi = n;
  while (lo < hi) { int mid = (lo + hi) >> 1; if (v[mid] < absx) lo = mid + 1; else hi = mid; }
  int insert = lo; if (insert > n - 1) insert = n - 1; if (insert < 0) insert = 0;
  int idx_hi = insert < 1 ? 1 : (insert > n - 1 ? n - 1 : insert);
  int idx_lo = idx_hi - 1; if (idx_lo < 0) idx_lo = 0;
  float vlo = v[idx_lo], vhi = v[idx_hi];
  float dlo = fabsf(absx - vlo), dhi = fabsf(vhi - absx);
  bool pick_hi = dhi < dlo;
  bool is_tie = fabsf(dhi - dlo) < 1e-12f;
  bool even_lo = (idx_lo & 1) == 0;
  if (is_tie) pick_hi = !even_lo;
  return pick_hi ? idx_hi : idx_lo;
}

// NVFP4 pack of 16 fp32 values (contiguous block) -> codes + SFA byte.
template <class LayoutSF>
__device__ __forceinline__ void pack_fp4_block_a(const float* xf, uint8_t* code_out,
                                                 uint8_t* sf_out, LayoutSF layout,
                                                 int r, int b, int K_low) {
  float bmax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) bmax = fmaxf(bmax, fabsf(xf[j]));
  float ideal = fmaxf(bmax / 6.0f, 0.001953125f);
  unsigned char sfb; float scale;
  e4m3_ceil_a(ideal, sfb, scale);
#if defined(PROMIX_ACT_MSE) && (PROMIX_ACT_MSE)
  // Win 1 (round-3): per-block argmin-MSE block scale (twin of fake_quantize_nvfp4).
  {
    float best_err = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float xs = fminf(fmaxf(xf[j] / scale, -6.0f), 6.0f);
      int ii = rtn_idx_a(fabsf(xs), cFP4_val_a, 8);
      float dq = (xs < 0.0f ? -1.0f : 1.0f) * cFP4_val_a[ii] * scale;
      float e = dq - xf[j]; best_err += e * e;
    }
    unsigned char cb = sfb;
    for (int s = 0; s < 3 && cb > 1; ++s) {
      cb = (unsigned char)(cb - 1);
      float csc = e4m3_to_float_a(cb); if (!(csc > 0.0f)) break;
      float err = 0.0f;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        float xs = fminf(fmaxf(xf[j] / csc, -6.0f), 6.0f);
        int ii = rtn_idx_a(fabsf(xs), cFP4_val_a, 8);
        float dq = (xs < 0.0f ? -1.0f : 1.0f) * cFP4_val_a[ii] * csc;
        float e = dq - xf[j]; err += e * e;
      }
      if (err < best_err) { best_err = err; scale = csc; sfb = cb; }
    }
  }
#endif
  sf_out[layout(make_coord(r, b * 16, 0))] = sfb;
  int obase = r * (K_low / 2) + b * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float xs0 = fminf(fmaxf(xf[2 * j]     / scale, -6.0f), 6.0f);
    float xs1 = fminf(fmaxf(xf[2 * j + 1] / scale, -6.0f), 6.0f);
    int i0 = rtn_idx_a(fabsf(xs0), cFP4_val_a, 8);
    int i1 = rtn_idx_a(fabsf(xs1), cFP4_val_a, 8);
    unsigned char c0 = (unsigned char)i0; if (xs0 < 0.0f && i0 > 0) c0 |= 0x8;
    unsigned char c1 = (unsigned char)i1; if (xs1 < 0.0f && i1 > 0) c1 |= 0x8;
    code_out[obase + j] = (c0 & 0x0F) | (c1 << 4);
  }
}

// MXFP8 block-32 pack from 32 values held by one warp (lane j holds q[j]).
template <class LayoutSF>
__device__ __forceinline__ void pack_fp8_block_warp_a(const float q,
                                                      uint8_t* __restrict__ code,
                                                      uint8_t* __restrict__ sf_out,
                                                      LayoutSF layout, int r, int b,
                                                      int K_high, uint32_t full_mask) {
  const int lane = threadIdx.x & 31;
  float a = fabsf(q);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(full_mask, a, o));
  float ideal = a / 448.0f;
  float yv = fmaxf(ideal, 1.1754944e-38f);
  int ex; float m = frexpf(yv, &ex);
  int e = (m == 0.5f) ? ex - 1 : ex;
  float sc = ldexpf(1.0f, e);
  if (lane == 0) {
    int sb = e + 127; sb = sb < 0 ? 0 : (sb > 254 ? 254 : sb);
    sf_out[layout(make_coord(r, b * 32, 0))] = (unsigned char)sb;
  }
  sc = __shfl_sync(full_mask, sc, 0);
  float xs = fminf(fmaxf(q / sc, -448.0f), 448.0f);
  code[(long)r * K_high + b * 32 + lane] =
      (unsigned char)__nv_cvt_float_to_fp8(xs, __NV_SATFINITE, __NV_E4M3);
}

// ===================================================================================
// Kernel: one CTA (THREADS) per token row, N = row width (2048 or 8192).
//   phase 0  r = to_add + x  (input dtype, one rounding), written ONCE
//   phase 1  row mean of r^2 in fp32 (block reduction) + rsqrt(var + eps)
//   phase 2  xn = round_to_fp16(r * scale) into SMEM (N floats = 8/32 KB)
//   phase 3  qpf pack of the low NVFP4 + high MXFP8 segments (verbatim device code)
// ===================================================================================
template <int N, int THREADS, class TIn, class LayoutSF4, class LayoutSF8>
__global__ __launch_bounds__(THREADS)
void add_rmsnorm_pack_kernel(uint8_t* __restrict__ code_lo, uint8_t* __restrict__ sf_lo,
                             LayoutSF4 layout4,
                             uint8_t* __restrict__ code_hi, uint8_t* __restrict__ sf_hi,
                             LayoutSF8 layout8,
                             const TIn* __restrict__ x,
                             const TIn* __restrict__ to_add,
                             TIn* __restrict__ residual_out,
                             float* __restrict__ norm_out_dbg,
                             int K_low, int K_high, float eps) {
  constexpr int kVec = 8;                     // 16 B of half per thread per chunk
  constexpr int kNPerChunk = THREADS * kVec;
  constexpr int kChunks = N / kNPerChunk;
  static_assert(N % kNPerChunk == 0, "row width must tile evenly");

  extern __shared__ __align__(16) float smem[];   // N floats (8 KB @ 2048 / 32 KB @ 8192)

  const int r = blockIdx.x;
  const int tid = threadIdx.x;
  const long rowoff = (long)r * N;
  const TIn* xr = x + rowoff;
  const TIn* ar = to_add ? to_add + rowoff : nullptr;   // nullptr -> r == x
  TIn* rr = residual_out + rowoff;

  float ssq = 0.0f;
  // ---- phase 0: residual add (input dtype) + sum of squares (fp32) -----------
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    const int base = c * kNPerChunk + tid * kVec;
    float4 xa = *reinterpret_cast<const float4*>(xr + base);
    float4 xb = *reinterpret_cast<const float4*>(xr + base + 4);
    TIn hx[8], ha[8], hr[8];
    *reinterpret_cast<float4*>(hx) = xa;
    *reinterpret_cast<float4*>(hx + 4) = xb;
    if (ar) {                                   // residual add (pre-norm: absent)
      float4 aa = *reinterpret_cast<const float4*>(ar + base);
      float4 ab = *reinterpret_cast<const float4*>(ar + base + 4);
      *reinterpret_cast<float4*>(ha) = aa;
      *reinterpret_cast<float4*>(ha + 4) = ab;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      hr[i] = ar ? add_dtype_a(hx[i], ha[i]) : hx[i];   // +residual (1 rounding)
      const float rf = to_float_a(hr[i]);
      ssq += rf * rf;                           // fp32 square + accumulate
    }
    if (rr) {                                   // residual stream written ONCE
      *reinterpret_cast<float4*>(rr + base) = *reinterpret_cast<float4*>(hr);
      *reinterpret_cast<float4*>(rr + base + 4) = *reinterpret_cast<float4*>(hr + 4);
    }
  }

  // ---- phase 1: row mean (block reduction) -----------------------------------
  float v = ssq;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  constexpr int kWarps = THREADS / 32;
  __shared__ float warp_sums[kWarps];
  const int warp = tid >> 5, lane = tid & 31;
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) total += warp_sums[w];
  const float mean = total / (float)N;
  const float scale = rsqrtf(mean + eps);       // matches ATen rsqrt in fp32

  // ---- phase 2: xn = round_to_fp16(r * scale) into SMEM (row-linear) ---------
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    const int base = c * kNPerChunk + tid * kVec;
    // The residual stream is still in registers (phase 0); a global reload is
    // only needed when it was just written by another... it was written by this
    // same thread, so registers are authoritative and no reload is needed.
    TIn hr[8];
    if (ar) {
      float4 sa = *reinterpret_cast<const float4*>(rr + base);
      float4 sb = *reinterpret_cast<const float4*>(rr + base + 4);
      *reinterpret_cast<float4*>(hr) = sa;
      *reinterpret_cast<float4*>(hr + 4) = sb;
    } else {
      const TIn* xr_ = xr + base;
      *reinterpret_cast<float4*>(hr) = *reinterpret_cast<const float4*>(xr_);
      *reinterpret_cast<float4*>(hr + 4) = *reinterpret_cast<const float4*>(xr_ + 4);
    }
    float* dst = smem + base;
    float rnd[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      // the reference's `t.to(input_dtype)` — a REAL fp16 round, kept as float
      rnd[i] = round_in_dtype_a(to_float_a(hr[i]) * scale, TIn{});
      dst[i] = rnd[i];
    }
    if (norm_out_dbg) {                          // probe-only: the rounded norm output
      float* d24 = norm_out_dbg + rowoff + base;
#pragma unroll
      for (int i = 0; i < 8; ++i) d24[i] = rnd[i];
    }
  }
  __syncthreads();

  // ---- phase 3: pack (identical decomposition to hadamard_pack_ext.cu) -------
  // Warp-clean partition: the first n_lo_warps warps own the NVFP4 blocks (each
  // thread takes blocks b = tid, tid + n_lo_warps*32), the remaining warps each
  // own whole MXFP8 block-32s (one lane per column). No warp runs both loops.
  const int n_lo = K_low / 16;
  const int n_hi = (K_high > 0) ? K_high / 32 : 0;
  const int n_lo_warps = (n_lo / 2 + 31) / 32 < kWarps ? (n_lo / 2 + 31) / 32 : kWarps;
  if (warp < n_lo_warps) {
    for (int b = tid; b < n_lo; b += n_lo_warps * 32) {
      const int c0 = b * 16;
      float xf[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) xf[j] = smem[c0 + j];
      pack_fp4_block_a<LayoutSF4>(xf, code_lo, sf_lo, layout4, r, b, K_low);
    }
  }
  const int hi_warps = kWarps - n_lo_warps;
  if (n_hi > 0 && warp >= n_lo_warps) {
    for (int b = warp - n_lo_warps; b < n_hi; b += hi_warps) {
      const int col = K_low + b * 32 + lane;     // high segment starts at K_low
      pack_fp8_block_warp_a<LayoutSF8>(smem[col], code_hi, sf_hi, layout8, r, b, K_high, 0xffffffffu);
    }
  }
}

// ===================================================================================
// Host entry point.
//   add_rmsnorm_pack_mixed(x, to_add, N_gemm, K_low, K_high, eps)
//     -> {residual_out, code_lo(M,K_low/2), sfa_lo, code_hi(M,K_high), sfa_hi}
// x / to_add: [M, N] fp16 (contiguous). N = x.shape[1] (2048 or 8192).
// `to_add` may be None/empty (pre-norm decoders: the norm input IS the residual
// stream), in which case residual_out is omitted from the result.
// ===================================================================================
template <int N, class TIn, class LayoutSF4, class LayoutSF8>
static void launch_arn(const TIn* x, const TIn* to_add, TIn* residual_out, float* norm_out_dbg,
                       uint8_t* code_lo, uint8_t* sf_lo, LayoutSF4 layout4,
                       uint8_t* code_hi, uint8_t* sf_hi, LayoutSF8 layout8,
                       int rows, int K_low, int K_high, float eps, cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int smem_bytes = N * (int)sizeof(float);
  auto kernel = &add_rmsnorm_pack_kernel<N, kThreads, TIn, LayoutSF4, LayoutSF8>;
  static bool attr_set = false;
  if (!attr_set) {
    arn_check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_bytes), "arn smem attr");
    attr_set = true;
  }
  kernel<<<dim3((unsigned)rows, 1, 1), kThreads, smem_bytes, stream>>>(
      code_lo, sf_lo, layout4, code_hi, sf_hi, layout8, x, to_add, residual_out, norm_out_dbg,
      K_low, K_high, eps);
}

std::vector<torch::Tensor> add_rmsnorm_pack_mixed(torch::Tensor x, c10::optional<torch::Tensor> to_add_opt,
                                                  int64_t N_gemm, int64_t K_low, int64_t K_high,
                                                  double eps, bool dbg) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16,
              "x must be fp16/bf16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  torch::Tensor to_add = to_add_opt.has_value() ? *to_add_opt : torch::Tensor();
  const bool has_add = to_add.defined() && to_add.numel() > 0;
  if (has_add) {
    TORCH_CHECK(to_add.is_cuda(), "to_add must be CUDA");
    TORCH_CHECK(x.scalar_type() == to_add.scalar_type(), "x/to_add dtype mismatch");
    TORCH_CHECK(to_add.is_contiguous(), "to_add must be contiguous");
    TORCH_CHECK(x.sizes() == to_add.sizes(), "x/to_add shape mismatch");
  }
  TORCH_CHECK(K_low % 16 == 0, "K_low % 16");
  TORCH_CHECK(K_high % 32 == 0 || K_high == 0, "K_high % 32");
  const int64_t M = x.size(0), N = x.size(1);
  TORCH_CHECK(K_low + K_high == N, "K_low + K_high != N");
  TORCH_CHECK(N == 2048 || N == 8192, "add_rmsnorm_pack_mixed is instantiated for N=2048/8192");
  init_grids_once_a();

  const int Mi = (int)M, Ni = (int)N_gemm, Klo = (int)K_low, Khi = (int)K_high;
  auto dev = x.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto layout4 = BSConfigNV::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Klo, 1));
  auto layout8 = BSConfigMX::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Khi > 0 ? Khi : 32, 1));

  auto resid_t   = has_add ? torch::empty({M, N}, x.options()) : torch::Tensor();
  auto code_lo_t = torch::empty({M, Klo / 2}, u8opts(dev));
  auto sf_lo_t   = torch::empty({(int64_t)size(filter_zeros(layout4))}, u8opts(dev));
  auto code_hi_t = torch::empty({M, Khi}, u8opts(dev));
  auto sf_hi_t   = torch::empty({(int64_t)size(filter_zeros(layout8))}, u8opts(dev));
  // probe-only norm output (empty unless requested); never allocated on the
  // production path.
  auto dbg_t = dbg ? torch::zeros({M, N}, x.options().dtype(torch::kFloat32))
                   : torch::Tensor();

#define LAUNCH_ARN(N_, TY)                                                                     \
  launch_arn<N_, TY>(reinterpret_cast<const TY*>(x.data_ptr()),                                \
                     has_add ? reinterpret_cast<const TY*>(to_add.data_ptr()) : nullptr,       \
                     has_add ? reinterpret_cast<TY*>(resid_t.data_ptr()) : nullptr,            \
                     dbg ? dbg_t.data_ptr<float>() : nullptr,                                  \
                     code_lo_t.data_ptr<uint8_t>(), sf_lo_t.data_ptr<uint8_t>(), layout4,      \
                     code_hi_t.data_ptr<uint8_t>(), sf_hi_t.data_ptr<uint8_t>(), layout8,      \
                     Mi, Klo, Khi, (float)eps, stream)
  if (x.scalar_type() == torch::kFloat16) {
    if (N == 2048)      LAUNCH_ARN(2048, half);
    else                LAUNCH_ARN(8192, half);
  } else {
    if (N == 2048)      LAUNCH_ARN(2048, __nv_bfloat16);
    else                LAUNCH_ARN(8192, __nv_bfloat16);
  }
#undef LAUNCH_ARN
  arn_check_cuda(cudaGetLastError(), "add_rmsnorm_pack launch");
  // FIXED layout: {residual (empty when no add), code_lo, sfa_lo, code_hi, sfa_hi[, norm_dbg]}
  auto resid_out = resid_t.defined() ? resid_t : torch::empty({0}, x.options());
  if (dbg) return {resid_out, code_lo_t, sf_lo_t, code_hi_t, sf_hi_t, dbg_t};
  return {resid_out, code_lo_t, sf_lo_t, code_hi_t, sf_hi_t};
}

// ---------------------------------------------------------------------------
// Host helper: derive the packed A operand for ONE consumer N from a
// `add_rmsnorm_pack_mixed` result. Consumers that share an activation differ
// only in their GEMM's N, which changes the SFA ATOM LAYOUT, not the values:
// the packed CODES are shared, and only the SFA bytes are re-scattered into the
// consumer's layout. Cheap (rows * K/VEC bytes per segment) and byte-identical
// to packing that consumer separately.
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> norm_pack_for_n(std::vector<torch::Tensor> const& packed,
                                           int64_t N_gemm, int64_t K_low, int64_t K_high) {
  // Accepts either the full add_rmsnorm_pack_mixed vector ({residual, lo, sfa_lo,
  // hi, sfa_hi, ...}) or just the four packed tensors ({lo, sfa_lo, hi, sfa_hi}).
  TORCH_CHECK(packed.size() >= 4, "norm_pack_for_n expects the packed tensors");
  const int off = (packed.size() >= 5) ? 1 : 0;
  auto code_lo_t = packed[off + 0];   // (M, K_low/2)
  auto sf_lo_t   = packed[off + 1];   // (M, K_low/16)
  auto code_hi_t = packed[off + 2];   // (M, K_high)
  auto sf_hi_t   = packed[off + 3];   // (M, K_high/32)
  const int64_t M = code_hi_t.size(0);
  const int Mi = (int)M, Ni = (int)N_gemm, Klo = (int)K_low, Khi = (int)K_high;
  auto dev = code_lo_t.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto layout4 = BSConfigNV::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Klo, 1));
  auto layout8 = BSConfigMX::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Khi > 0 ? Khi : 32, 1));
  int64_t sz4 = (int64_t)size(filter_zeros(layout4));
  int64_t sz8 = (int64_t)size(filter_zeros(layout8));
  // If the source layout already matches (sf buffer size equals the target),
  // reuse the SFA tensor as-is.
  if ((int64_t)sf_lo_t.numel() == sz4 && (int64_t)sf_hi_t.numel() == sz8) {
    return {code_lo_t, sf_lo_t, code_hi_t, sf_hi_t};
  }
  auto phys4 = torch::empty({sz4}, u8opts(dev));
  auto phys8 = torch::empty({sz8}, u8opts(dev));
  {
    const int nblk = Klo / 16, tpb = 256;
    int total = Mi * nblk;
    using L = decltype(layout4);
    cd_fill_sf_block_kernel<L><<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys4.data_ptr<uint8_t>(), sf_lo_t.data_ptr<uint8_t>(), layout4, Mi, nblk, 16);
    arn_check_cuda(cudaGetLastError(), "norm_pack_for_n nv sfa");
  }
  {
    const int nblk = Khi / 32, tpb = 256;
    int total = Mi * nblk;
    using L = decltype(layout8);
    cd_fill_sf_block_kernel<L><<<(total + tpb - 1) / tpb, tpb, 0, stream>>>(
        phys8.data_ptr<uint8_t>(), sf_hi_t.data_ptr<uint8_t>(), layout8, Mi, nblk, 32);
    arn_check_cuda(cudaGetLastError(), "norm_pack_for_n mx sfa");
  }
  return {code_lo_t, phys4, code_hi_t, phys8};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_rmsnorm_pack_mixed", &add_rmsnorm_pack_mixed,
        "FUSED residual-add + RMSNorm -> fp16 -> NVFP4(low)+MXFP8(high) codes + "
        "interleaved SFA in one launch; returns "
        "{residual_out, code_lo(M,K_low/2), sfa_lo, code_hi(M,K_high), sfa_hi}. "
        "N_gemm is the consumer GEMM's N (SFA layout); row width is x.shape[1] (2048/8192)",
        pybind11::arg("x"), pybind11::arg("to_add"), pybind11::arg("N_gemm"),
        pybind11::arg("K_low"), pybind11::arg("K_high"), pybind11::arg("eps"),
        pybind11::arg("dbg") = false);
  m.def("norm_pack_for_n", &norm_pack_for_n,
        "re-derive the packed A operand (codes shared, SFA atom layout for this "
        "consumer N) from an add_rmsnorm_pack_mixed result; returns "
        "{code_lo, sfa_lo, code_hi, sfa_hi}",
        pybind11::arg("packed"), pybind11::arg("N_gemm"),
        pybind11::arg("K_low"), pybind11::arg("K_high"));
}
