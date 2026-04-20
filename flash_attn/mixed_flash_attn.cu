/*
 * INT8 FlashAttention — v2: Single launch, scalar Q·K^T (CosSim=0.9999)
 *
 * WGMMA investigation summary:
 * - MMA_Atom<SM90_64x64x32_S32S8S8_SS_TN> compiles ✓
 * - atom.fma(desc_a, desc_b, acc_regs, scale) compiles ✓
 * - make_gmma_desc<Major::K>(tensor) compiles ✓
 * - Kernel runs without crash ✓
 * - BLOCKED: SMEM layout (tile_to_shape of Layout_K_INTER_Atom) creates
 *   hierarchical M-major layout incompatible with GMMA K-major descriptor.
 *   CUTE's INT8 gemm() doesn't work. ss_smem_selector doesn't support int8_t.
 *   Fix requires proper K-tile tensor slicing + descriptor per K iteration.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

constexpr int kBr = 64;
constexpr int kBc = 64;

extern "C" __global__ void
int8_fa_v2_kernel(
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ K,
    const half*    __restrict__ V,
    half*          __restrict__ O,
    int Lq, int Lkv, int D,
    float scale_qk, bool causal)
{
    int head_idx = blockIdx.y, q_tile = blockIdx.x;
    int q_start = q_tile * kBr, q_end = min(q_start + kBr, Lq);
    int q_rows = q_end - q_start;

    const int8_t* Q_head = Q + head_idx * Lq * D;
    const int8_t* K_head = K + head_idx * Lkv * D;
    const half*   V_head = V + head_idx * Lkv * D;
    half*         O_head = O + head_idx * Lq * D;

    extern __shared__ char smem_buf[];
    int8_t* Q_smem = reinterpret_cast<int8_t*>(smem_buf);
    int8_t* K_smem = Q_smem + kBr * D;
    half*   V_smem = reinterpret_cast<half*>(K_smem + kBc * D);
    float* O_acc   = reinterpret_cast<float*>(V_smem + kBc * D);
    float* m_i     = O_acc + kBr * D;
    float* l_i     = m_i + kBr;

    // Load Q
    const int8_t* Q_ptr = Q_head + q_start * D;
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) Q_smem[i] = Q_ptr[i];
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) O_acc[i] = 0.0f;
    for (int i = threadIdx.x; i < q_rows; i += blockDim.x) m_i[i] = -INFINITY, l_i[i] = 0.0f;
    __syncthreads();

    int num_kv = (Lkv + kBc - 1) / kBc, wid = threadIdx.x / 32, nw = blockDim.x / 32;
    for (int kv = 0; kv < num_kv; kv++) {
        int ks = kv * kBc, ke = min(ks + kBc, Lkv), kr = ke - ks;
        if (causal && ks > q_end - 1) break;
        const int8_t* Kp = K_head + ks * D;
        const half*   Vp = V_head + ks * D;
        for (int i = threadIdx.x; i < kr * D; i += blockDim.x) K_smem[i] = Kp[i];
        for (int i = threadIdx.x; i < kr * D; i += blockDim.x) V_smem[i] = Vp[i];
        __syncthreads();

        for (int qi = wid; qi < q_rows; qi += nw) {
            float mo = m_i[qi], lo = l_i[qi], mn = mo;
            float sr[64];
            for (int kj = 0; kj < kr; kj++) sr[kj] = 0.0f;
            for (int d = 0; d < D; d++) {
                int32_t qv = Q_smem[qi * D + d];
                for (int kj = 0; kj < kr; kj++)
                    sr[kj] += float(qv * K_smem[kj * D + d]);
            }
            for (int kj = 0; kj < kr; kj++) {
                sr[kj] *= scale_qk;
                if (causal && ks + kj > q_start + qi) sr[kj] = -INFINITY;
                mn = fmaxf(mn, sr[kj]);
            }
            float re = expf(mo - mn), ln = lo * re;
            float pr[64];
            for (int kj = 0; kj < kr; kj++) { pr[kj] = expf(sr[kj] - mn); ln += pr[kj]; }
            for (int d = 0; d < D; d++) {
                float ov = O_acc[qi * D + d] * re;
                for (int kj = 0; kj < kr; kj++) ov += pr[kj] * __half2float(V_smem[kj * D + d]);
                O_acc[qi * D + d] = ov;
            }
            m_i[qi] = mn; l_i[qi] = ln;
        }
        __syncthreads();
    }

    for (int qi = threadIdx.x; qi < q_rows; qi += blockDim.x) {
        float il = l_i[qi] > 0 ? 1.0f / l_i[qi] : 0;
        for (int d = 0; d < D; d++) O_acc[qi * D + d] *= il;
    }
    __syncthreads();
    half* Op = O_head + q_start * D;
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) Op[i] = __float2half(O_acc[i]);
}

torch::Tensor int8_flash_attn(
    torch::Tensor Q_int8, torch::Tensor K_int8, torch::Tensor V_fp16,
    float scale_q, float scale_k, float scale_s, bool causal)
{
    TORCH_CHECK(Q_int8.dtype() == torch::kInt8 && K_int8.dtype() == torch::kInt8);
    const int B=Q_int8.size(0), H=Q_int8.size(1), Lq=Q_int8.size(2), D=Q_int8.size(3), Lkv=K_int8.size(2);
    TORCH_CHECK(D==256);
    auto Qf=Q_int8.reshape({B*H,Lq,D}).contiguous(), Kf=K_int8.reshape({B*H,Lkv,D}).contiguous();
    auto Vf=V_fp16.reshape({B*H,Lkv,D}).contiguous();
    auto Of=torch::empty({B*H,Lq,D}, torch::TensorOptions().device(Q_int8.device()).dtype(torch::kFloat16));
    auto s=at::cuda::getCurrentCUDAStream().stream();
    constexpr int th=256;
    int nt=(Lq+kBr-1)/kBr;
    float sqk=scale_q*scale_k*scale_s;
    size_t sm=kBr*D*1 + kBc*D*1 + kBc*D*2 + kBr*D*4 + kBr*4*2;
    cudaFuncSetAttribute(int8_fa_v2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    int8_fa_v2_kernel<<<dim3(nt,B*H), th, sm, s>>>(
        (const int8_t*)Qf.data_ptr(), (const int8_t*)Kf.data_ptr(),
        (const half*)Vf.data_ptr(), (half*)Of.data_ptr(), Lq, Lkv, D, sqk, causal);
    return Of.reshape({B,H,Lq,D});
}

#else
torch::Tensor int8_flash_attn(torch::Tensor, torch::Tensor, torch::Tensor,
                               float, float, float, bool) {
    throw std::runtime_error("SM90 not supported");
}
#endif
