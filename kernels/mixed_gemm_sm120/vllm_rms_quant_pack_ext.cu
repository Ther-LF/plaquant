// vllm_rms_quant_pack_ext.cu — vllm-perf round-3 lever 2: producer-side quant
// fusion. ONE kernel = vLLM's fused_add_rms_norm (bit-exact replica) + the qpf
// NVFP4/MXFP8 quantize+pack (verbatim device code) for the qkv_proj /
// gate_up_proj input sites in the vLLM serving path.
//
// WHY: vLLM-NVFP4 never launches a standalone act-quant kernel — the fp4 quant
// rides the producing RMSNorm via inductor fusion passes (rms_quant_fusion.py).
// Our promix path pays, per norm site: vLLM's fused_add_rms_norm kernel + a
// separate quant_pack_float_mixed launch + an fp16 [M,K] HBM round-trip between
// them. This kernel does all of it in one launch and additionally emits the
// fp16 norm output xn (so the decode GEMV routes, which consume fp16, still
// work) and the functional residual_out.
//
// BIT-EXACTNESS (the whole design): round-26's add_rmsnorm_pack_ext replicated
// the ATen reduction order and FAILED the promix gate (cos 0.9889) because a
// 1-ULP rsqrt(var) shift flips NVFP4 absmax blocks. In the vLLM path the
// reference is NOT ATen — it is vLLM csrc/layernorm_kernels.cu
// fused_add_rms_norm_kernel<scalar_t, width=8, HasWeight=true>. This file
// replicates THAT kernel exactly:
//   * grid = num_tokens, block = min(hidden, num_tokens < 256 ? 1024 : 256)
//     (the same num_tokens-dependent rule — REQUIRED, the cub reduction order
//     depends on blockDim);
//   * the _f16Vec<scalar_t, 8> vectorized residual add (half2 adds, 1 rounding)
//     + sum_squares() per-thread strided accumulation (verbatim vendored);
//   * cub::BlockReduce<float, 1024>(storage).Reduce(var, CubAddOp{}, blockDim.x);
//   * s_variance = rsqrtf(variance / hidden + eps);
//   * out = convert(convert(res) * s_variance * convert(w)) — one fp16 round.
// The pack stage is the verbatim qpf device code reading the SAME rounded fp16
// values, so packed codes == quant_pack_float_mixed(xn) byte-for-byte. Gated by
// promix/vllm_integ/norm_pack_probe.py: torch.equal on xn/residual vs
// torch.ops._C.fused_add_rms_norm AND on packs vs qpfm.
//
// _typeConvert/_f16Vec/CubAddOp are vendored verbatim from vLLM v0.32.0
// csrc/type_convert.cuh + csrc/cub_helpers.h (Apache-2.0).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "act_mse.cuh"

using namespace cute;

static inline void vrq_check_cuda(cudaError_t e, const char* what) {
  TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// ===================== vendored from vLLM v0.32.0 (verbatim semantics) =======
#if CUB_VERSION >= 200800
  #include <cuda/std/functional>
using CubAddOp = cuda::std::plus<>;
#else
using CubAddOp = cub::Sum;
#endif

namespace vrq {

template <typename T>
struct _typeConvert { static constexpr bool exists = false; };

template <>
struct _typeConvert<half> {
  static constexpr bool exists = true;
  using hip_type = __half;
  using packed_hip_type = __half2;
  __device__ static __forceinline__ float convert(hip_type x) { return __half2float(x); }
  __device__ static __forceinline__ float2 convert(packed_hip_type x) { return __half22float2(x); }
  __device__ static __forceinline__ hip_type convert(float x) { return __float2half_rn(x); }
  __device__ static __forceinline__ packed_hip_type convert(float2 x) { return __float22half2_rn(x); }
};

template <>
struct _typeConvert<__nv_bfloat16> {
  static constexpr bool exists = true;
  using hip_type = __nv_bfloat16;
  using packed_hip_type = __nv_bfloat162;
  __device__ static __forceinline__ float convert(hip_type x) { return __bfloat162float(x); }
  __device__ static __forceinline__ float2 convert(packed_hip_type x) { return __bfloat1622float2(x); }
  __device__ static __forceinline__ hip_type convert(float x) { return __float2bfloat16(x); }
  __device__ static __forceinline__ packed_hip_type convert(float2 x) { return __float22bfloat162_rn(x); }
};

template <typename scalar_t, int width>
struct alignas(16) _f16Vec {
  static_assert(width > 0 && (width & (width - 1)) == 0);
  using Converter = _typeConvert<scalar_t>;
  using T1 = typename Converter::hip_type;
  using T2 = typename Converter::packed_hip_type;
  T1 data[width];

  __device__ _f16Vec& operator+=(const _f16Vec<scalar_t, width>& other) {
    // torch cpp_extension builds define __CUDA_NO_HALF2_OPERATORS__, so the
    // packed adds are spelled __hadd2 explicitly — the SAME instruction vLLM's
    // `temp += T2{...}` lowers to (half2/bfloat162 add, one rounding).
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      T2 temp = __hadd2(T2{data[i], data[i + 1]},
                        T2{other.data[i], other.data[i + 1]});
      data[i] = temp.x;
      data[i + 1] = temp.y;
    }
    return *this;
  }

  __device__ float sum_squares() const {
    float result = 0.0f;
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      float2 z = Converter::convert(T2{data[i], data[i + 1]});
      result += z.x * z.x + z.y * z.y;
    }
    return result;
  }
};

}  // namespace vrq
// ===================== end vendored ==========================================

// SM120 block-scaled SFA atom layouts (config-only, same as qpf/arn).
using BSConfigNV = cutlass::detail::Sm1xxBlockScaledConfig<16>;
using BSConfigMX = cutlass::detail::Sm1xxBlockScaledConfig<32>;

static inline torch::TensorOptions u8opts(torch::Device dev) {
  return torch::TensorOptions().dtype(torch::kUInt8).device(dev);
}

// ====================== pack device helpers (verbatim from qpf/arn exts) =====
__constant__ float cFP4_val_v[8];
static bool g_grids_ready_v = false;
static void init_grids_once_v() {
  if (g_grids_ready_v) return;
  float fp4[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  vrq_check_cuda(cudaMemcpyToSymbol(cFP4_val_v, fp4, sizeof(fp4)), "cpy fp4");
  g_grids_ready_v = true;
}

__device__ __forceinline__ float e4m3_to_float_v(unsigned char b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}
__device__ __forceinline__ void e4m3_ceil_v(float ideal, unsigned char& byte, float& val) {
  unsigned char b = (unsigned char)__nv_cvt_float_to_fp8(ideal, __NV_SATFINITE, __NV_E4M3);
  float v = e4m3_to_float_v(b);
  if (v < ideal && b < 0x7E) { ++b; v = e4m3_to_float_v(b); }
  byte = b; val = v;
}
__device__ __forceinline__ int rtn_idx_v(float absx, const float* v, int n) {
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

template <class LayoutSF>
__device__ __forceinline__ void pack_fp4_block_v(const float* xf, uint8_t* code_out,
                                                 uint8_t* sf_out, LayoutSF layout,
                                                 int r, int b, int K_low) {
  float bmax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) bmax = fmaxf(bmax, fabsf(xf[j]));
  float ideal = fmaxf(bmax / 6.0f, 0.001953125f);
  unsigned char sfb; float scale;
  e4m3_ceil_v(ideal, sfb, scale);
#if defined(PROMIX_ACT_MSE) && (PROMIX_ACT_MSE)
  promix_act_mse_select(xf, sfb, scale);   // round-4 cheap search (act_mse.cuh)
#endif
  sf_out[layout(make_coord(r, b * 16, 0))] = sfb;
  int obase = r * (K_low / 2) + b * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float xs0 = fminf(fmaxf(xf[2 * j]     / scale, -6.0f), 6.0f);
    float xs1 = fminf(fmaxf(xf[2 * j + 1] / scale, -6.0f), 6.0f);
    int i0 = rtn_idx_v(fabsf(xs0), cFP4_val_v, 8);
    int i1 = rtn_idx_v(fabsf(xs1), cFP4_val_v, 8);
    unsigned char c0 = (unsigned char)i0; if (xs0 < 0.0f && i0 > 0) c0 |= 0x8;
    unsigned char c1 = (unsigned char)i1; if (xs1 < 0.0f && i1 > 0) c1 |= 0x8;
    code_out[obase + j] = (c0 & 0x0F) | (c1 << 4);
  }
}

template <class LayoutSF>
__device__ __forceinline__ void pack_fp8_block_warp_v(const float q,
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
// Kernel: one CTA per token row; runtime N (vLLM-style strided vec8 loops).
//   phase A  vLLM fused_add_rms_norm EXACT replica: residual add (half2, 1
//            rounding) + per-thread strided sum_squares + cub BlockReduce +
//            rsqrtf(var/N + eps). residual_out written (functional).
//   phase B  xn = fp16(convert(res) * s_variance * convert(w)) — vLLM's exact
//            output arithmetic — written to gmem (for the decode GEMV route)
//            AND held in SMEM as float for the pack.
//   phase C  verbatim qpf pack of the low NVFP4 + high MXFP8 segments.
// ===================================================================================
template <int BLOCK, bool PACK, class TIn, class LayoutSF4, class LayoutSF8>
__global__ __launch_bounds__(BLOCK)
void vllm_arn_pack_kernel(uint8_t* __restrict__ code_lo, uint8_t* __restrict__ sf_lo,
                          LayoutSF4 layout4,
                          uint8_t* __restrict__ code_hi, uint8_t* __restrict__ sf_hi,
                          LayoutSF8 layout8,
                          const TIn* __restrict__ x,
                          const TIn* __restrict__ residual,
                          const TIn* __restrict__ weight,
                          TIn* __restrict__ residual_out,
                          TIn* __restrict__ xn_out,
                          int N, int K_low, int K_high, float eps) {
  constexpr int width = 8;
  using V = vrq::_f16Vec<TIn, width>;
  using Converter = vrq::_typeConvert<TIn>;

  const int r = blockIdx.x;
  const int tid = threadIdx.x;
  const int vec_hidden = N / width;
  const long vrow = (long)r * vec_hidden;

  auto* __restrict__ x_v    = reinterpret_cast<const V*>(x);
  auto* __restrict__ res_v  = reinterpret_cast<const V*>(residual);
  auto* __restrict__ w_v    = reinterpret_cast<const V*>(weight);
  auto* __restrict__ rout_v = reinterpret_cast<V*>(residual_out);
  auto* __restrict__ xn_v   = reinterpret_cast<V*>(xn_out);

  extern __shared__ __align__(16) char smem_raw[];
  TIn* smem = reinterpret_cast<TIn*>(smem_raw);   // N halves (norm output row)
  __shared__ float s_variance;
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;

  // ---- phase A: vLLM fused_add_rms_norm loop 1 (EXACT order). The summed
  // vectors are additionally kept in registers (<=2 per thread for
  // hidden<=4096 at block>=256) so phase B avoids vLLM's residual re-read —
  // same VALUES (this thread wrote them), bit-exact.
  constexpr int kKeep = 2;
  V keep[kKeep];
  float variance = 0.0f;
  int nvec = 0;
  for (int idx = tid; idx < vec_hidden; idx += blockDim.x) {
    V temp = x_v[vrow + idx];
    temp += res_v[vrow + idx];
    variance += temp.sum_squares();
    rout_v[vrow + idx] = temp;
    if (nvec < kKeep) keep[nvec] = temp;
    ++nvec;
  }
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);
  if (tid == 0) { s_variance = rsqrtf(variance / N + eps); }
  __syncthreads();

  // ---- phase B: vLLM loop 2 (EXACT arithmetic) + SMEM fp16 copy ------------
  {
    int i = 0;
    for (int idx = tid; idx < vec_hidden; idx += blockDim.x, ++i) {
      V res = (i < kKeep) ? keep[i] : rout_v[vrow + idx];
      V w = w_v[idx];
      V out;
#pragma unroll
      for (int j = 0; j < width; ++j) {
        float xv = Converter::convert(res.data[j]);
        float wf = Converter::convert(w.data[j]);
        out.data[j] = Converter::convert(xv * s_variance * wf);
      }
      xn_v[vrow + idx] = out;
      if constexpr (PACK) *reinterpret_cast<V*>(smem + idx * width) = out;
    }
  }
  if constexpr (!PACK) return;   // GEMV-routed M: norm-only (consumer uses xn)
  __syncthreads();

  // ---- phase C: pack — qpfm's parallel decomposition inside the row CTA:
  // one THREAD per NVFP4 16-block (all threads), then one WARP per MXFP8
  // 32-block. Reads the rounded fp16 norm values from SMEM (same values qpfm
  // reads from gmem -> bit-identical codes).
  const int kWarps = blockDim.x >> 5;
  const int warp = tid >> 5, lane = tid & 31;
  const int n_lo = K_low / 16;
  const int n_hi = (K_high > 0) ? K_high / 32 : 0;
  for (int b = tid; b < n_lo; b += blockDim.x) {
    const int c0 = b * 16;
    float xf[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) xf[j] = vrq::_typeConvert<TIn>::convert(smem[c0 + j]);
    pack_fp4_block_v<LayoutSF4>(xf, code_lo, sf_lo, layout4, r, b, K_low);
  }
  for (int b = warp; b < n_hi; b += kWarps) {
    const int col = K_low + b * 32 + lane;     // high segment starts at K_low
    float q = vrq::_typeConvert<TIn>::convert(smem[col]);
    pack_fp8_block_warp_v<LayoutSF8>(q, code_hi, sf_hi, layout8, r, b, K_high, 0xffffffffu);
  }
}

// ===================================================================================
// Host entry.
//   vllm_add_rms_norm_quant_pack(x, residual, weight, N_gemm, K_low, K_high, eps,
//                                do_pack=true)
//     -> {code_lo(M,K_low/2), sfa_lo, code_hi(M,K_high), sfa_hi, xn(M,N), residual_out(M,N)}
// Block size follows vLLM's fused_add_rms_norm rule EXACTLY (the cub reduction
// order depends on it): block = min(hidden, num_tokens < 256 ? 1024 : 256).
// do_pack=false (GEMV-routed decode M, chosen per captured graph size by the
// caller) skips the pack phase entirely — the consumer uses xn — but STILL
// allocates the pack outputs at their full (uninitialized) sizes so runtime
// shapes match the op's fake impl (inductor asserts them).
// ===================================================================================
std::vector<torch::Tensor> vllm_add_rms_norm_quant_pack(
    torch::Tensor x, torch::Tensor residual, torch::Tensor weight,
    int64_t N_gemm, int64_t K_low, int64_t K_high, double eps, bool do_pack) {
  TORCH_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda());
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16,
              "x must be fp16/bf16");
  TORCH_CHECK(x.scalar_type() == residual.scalar_type() &&
              x.scalar_type() == weight.scalar_type(), "dtype mismatch");
  TORCH_CHECK(x.is_contiguous() && residual.is_contiguous() && weight.is_contiguous());
  TORCH_CHECK(x.dim() == 2 && x.sizes() == residual.sizes(), "x/residual must be (M, N)");
  const int64_t M = x.size(0), N = x.size(1);
  TORCH_CHECK(weight.numel() == N, "weight must be (N,)");
  TORCH_CHECK(N % 8 == 0, "hidden % 8 (vec8 path)");
  TORCH_CHECK(K_low % 16 == 0 && K_low > 0, "K_low % 16");
  TORCH_CHECK(K_high % 32 == 0 && K_high > 0, "K_high % 32");
  TORCH_CHECK(K_low + K_high == N, "K_low + K_high != N");
  init_grids_once_v();

  const int Mi = (int)M, Ni = (int)N_gemm, Klo = (int)K_low, Khi = (int)K_high;
  auto dev = x.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  auto layout4 = BSConfigNV::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Klo, 1));
  auto layout8 = BSConfigMX::tile_atom_to_shape_SFA(make_shape(Mi, Ni, Khi, 1));

  auto code_lo_t = torch::empty({M, Klo / 2}, u8opts(dev));
  auto sf_lo_t   = torch::empty({(int64_t)size(filter_zeros(layout4))}, u8opts(dev));
  auto code_hi_t = torch::empty({M, Khi}, u8opts(dev));
  auto sf_hi_t   = torch::empty({(int64_t)size(filter_zeros(layout8))}, u8opts(dev));
  auto xn_t      = torch::empty({M, N}, x.options());
  auto resid_t   = torch::empty({M, N}, x.options());

  // vLLM's launch rule (layernorm_kernels.cu fused_add_rms_norm): REQUIRED for
  // the bit-exact cub reduction order.
  const int max_block_size = (Mi < 256) ? 1024 : 256;
  const int block = std::min((int)N, max_block_size);
  TORCH_CHECK(block % 32 == 0, "block must be warp-multiple");
  const int smem_bytes = do_pack ? (int)N * (int)x.element_size() : 0;

#define LAUNCH_VRQ_ONE(BLK, PACKF, TY)                                                   \
  do {                                                                                   \
    auto kernel = &vllm_arn_pack_kernel<BLK, PACKF, TY, decltype(layout4), decltype(layout8)>; \
    static bool attr_set = false;                                                        \
    if (!attr_set) {                                                                     \
      vrq_check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                          64 * 1024), "vrq smem attr");                  \
      attr_set = true;                                                                   \
    }                                                                                    \
    kernel<<<dim3((unsigned)Mi, 1, 1), block, smem_bytes, stream>>>(                     \
        code_lo_t.data_ptr<uint8_t>(), sf_lo_t.data_ptr<uint8_t>(), layout4,             \
        code_hi_t.data_ptr<uint8_t>(), sf_hi_t.data_ptr<uint8_t>(), layout8,             \
        reinterpret_cast<const TY*>(x.data_ptr()),                                       \
        reinterpret_cast<const TY*>(residual.data_ptr()),                                \
        reinterpret_cast<const TY*>(weight.data_ptr()),                                  \
        reinterpret_cast<TY*>(resid_t.data_ptr()),                                       \
        reinterpret_cast<TY*>(xn_t.data_ptr()),                                          \
        (int)N, Klo, Khi, (float)eps);                                                   \
  } while (0)
#define LAUNCH_VRQ(TY)                                                                   \
  do {                                                                                   \
    if (block == 1024) {                                                                 \
      if (do_pack) LAUNCH_VRQ_ONE(1024, true, TY); else LAUNCH_VRQ_ONE(1024, false, TY); \
    } else {                                                                             \
      TORCH_CHECK(block == 256, "unexpected block size ", block);                        \
      if (do_pack) LAUNCH_VRQ_ONE(256, true, TY); else LAUNCH_VRQ_ONE(256, false, TY);   \
    }                                                                                    \
  } while (0)
  if (x.scalar_type() == torch::kFloat16) LAUNCH_VRQ(half);
  else                                    LAUNCH_VRQ(__nv_bfloat16);
#undef LAUNCH_VRQ
#undef LAUNCH_VRQ_ONE
  vrq_check_cuda(cudaGetLastError(), "vllm_arn_pack launch");
  return {code_lo_t, sf_lo_t, code_hi_t, sf_hi_t, xn_t, resid_t};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vllm_add_rms_norm_quant_pack", &vllm_add_rms_norm_quant_pack,
        "ONE launch: vLLM fused_add_rms_norm (bit-exact replica, vec8 + cub "
        "BlockReduce + vLLM block-size rule) + qpf NVFP4/MXFP8 quant+pack. "
        "Returns {code_lo, sfa_lo, code_hi, sfa_hi, xn fp16, residual_out}.",
        pybind11::arg("x"), pybind11::arg("residual"), pybind11::arg("weight"),
        pybind11::arg("N_gemm"), pybind11::arg("K_low"), pybind11::arg("K_high"),
        pybind11::arg("eps"), pybind11::arg("do_pack") = true);
}
