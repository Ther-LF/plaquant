/*
 * INT8 FlashAttention v3 — WGMMA via atom.fma() with SW32 SMEM layout
 *
 * Key insight: Layout_K_SW32_Atom<int8_t> creates a canonical GMMA K-major
 * layout (B32 type) that make_gmma_desc<Major::K> accepts.
 * Native shape (8,32), tiled to (64,32). Stride: r*32 + c.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
using namespace cute;

constexpr int kBr = 64, kBc = 64;

// INT8 WGMMA atom
using MmaAtomQK = decltype(
    GMMA::ss_op_selector<int8_t, int8_t, int32_t,
        Shape<Int<64>, Int<64>, Int<64>>>());

// SMEM layout: SW32 (B32) → canonical GMMA K-major, accepted by make_gmma_desc
using SmemLayoutTile = decltype(tile_to_shape(
    GMMA::Layout_K_SW32_Atom<int8_t>{},
    make_shape(Int<kBr>{}, Int<32>{})));  // (64, 32) for one WGMMA K tile

extern "C" __global__ void
int8_fa_v3_kernel(
    const int8_t* __restrict__ Q, const int8_t* __restrict__ K,
    const half* __restrict__ V, half* __restrict__ O,
    int Lq, int Lkv, int D, float scale_qk, bool causal)
{
    int head_idx = blockIdx.y, q_tile = blockIdx.x;
    int q_start = q_tile * kBr, q_end = min(q_start + kBr, Lq), q_rows = q_end - q_start;

    const int8_t* Qh = Q + head_idx * Lq * D;
    const int8_t* Kh = K + head_idx * Lkv * D;
    const half*   Vh = V + head_idx * Lkv * D;
    half*         Oh = O + head_idx * Lq * D;

    // SMEM: Q_tile (64,256) + K_tile (64,256) + V + S_smem + O_acc + softmax
    extern __shared__ char smem[];
    int8_t* Q_smem = reinterpret_cast<int8_t*>(smem);
    int8_t* K_smem = Q_smem + kBr * D;
    half*   V_smem = reinterpret_cast<half*>(K_smem + kBc * D);
    float*  S_smem = reinterpret_cast<float*>(V_smem + kBc * D);
    float*  O_acc  = S_smem + kBr * kBc;
    float*  m_i    = O_acc + kBr * D;
    float*  l_i    = m_i + kBr;

    // Load Q (row-major into K-major SMEM: r*D + c → (r in tile) * 32 + (c in tile))
    const int8_t* Qp = Qh + q_start * D;
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) {
        int r = i / D, c = i % D;
        int kt = c / 32, ci = c % 32;  // K tile index, column within tile
        Q_smem[kt * kBr * 32 + r * 32 + ci] = Qp[i];
    }

    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) O_acc[i] = 0.0f;
    for (int i = threadIdx.x; i < q_rows; i += blockDim.x) m_i[i] = -INFINITY, l_i[i] = 0.0f;
    __syncthreads();

    int num_kv = (Lkv + kBc - 1) / kBc, wid = threadIdx.x / 32, nw = blockDim.x / 32;

    for (int kv = 0; kv < num_kv; kv++) {
        int ks = kv * kBc, ke = min(ks + kBc, Lkv), kr = ke - ks;
        if (causal && ks > q_end - 1) break;

        // Load K (same K-major pattern)
        const int8_t* Kp = Kh + ks * D;
        for (int i = threadIdx.x; i < kr * D; i += blockDim.x) {
            int r = i / D, c = i % D;
            int kt = c / 32, ci = c % 32;
            K_smem[kt * kBc * 32 + r * 32 + ci] = Kp[i];
        }
        // Load V (row-major)
        const half* Vp = Vh + ks * D;
        for (int i = threadIdx.x; i < kr * D; i += blockDim.x) V_smem[i] = Vp[i];
        __syncthreads();

        // ---- INT8 WGMMA for Q·K^T ----
        uint32_t S_acc[32]; for (int i = 0; i < 32; i++) S_acc[i] = 0;
        MmaAtomQK mma;

        // 8 K iterations (D=256, 32 per WGMMA)
        for (int k = 0; k < 8; k++) {
            // SMEM pointer for k-th K tile
            int8_t* qk = Q_smem + k * kBr * 32;
            int8_t* kk = K_smem + k * kBc * 32;

            Tensor sQ = make_tensor(make_smem_ptr(qk), SmemLayoutTile{});
            Tensor sK = make_tensor(make_smem_ptr(kk), SmemLayoutTile{});
            uint64_t dq = cute::SM90::GMMA::make_gmma_desc<GMMA::Major::K>(sQ);
            uint64_t dk = cute::SM90::GMMA::make_gmma_desc<GMMA::Major::K>(sK);

            auto sc = (k == 0) ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
            mma.fma(dq, dk,
                S_acc[0],  S_acc[1],  S_acc[2],  S_acc[3],
                S_acc[4],  S_acc[5],  S_acc[6],  S_acc[7],
                S_acc[8],  S_acc[9],  S_acc[10], S_acc[11],
                S_acc[12], S_acc[13], S_acc[14], S_acc[15],
                S_acc[16], S_acc[17], S_acc[18], S_acc[19],
                S_acc[20], S_acc[21], S_acc[22], S_acc[23],
                S_acc[24], S_acc[25], S_acc[26], S_acc[27],
                S_acc[28], S_acc[29], S_acc[30], S_acc[31], sc);
        }

        // Extract using partition_fragment_C (canonical CUTE WGMMA fragment mapping)
        int tid = threadIdx.x;
        if (tid < 128) {
            auto tiled_mma = make_tiled_mma(MmaAtomQK{}, Layout<Shape<_1, _1, _1>>{});
            // Get accumulator fragment with correct CLayout_64x64 mapping
            auto acc_frag = partition_fragment_C(tiled_mma,
                make_shape(Int<kBr>{}, Int<kBc>{}));
            // Partition SMEM destination using the same TiledMma
            Tensor sS = make_tensor(make_smem_ptr(S_smem),
                make_layout(make_shape(Int<kBr>{}, Int<kBc>{}), LayoutRight{}));
            auto thr_mma = tiled_mma.get_slice(tid);
            auto tCsC = thr_mma.partition_C(sS);
            // Both acc_frag and tCsC have the same layout → direct element copy
            for (int i = 0; i < size(acc_frag); i++)
                tCsC(i) = float(S_acc[i]) * scale_qk;
        }
        __syncthreads();

        // Softmax + P·V
        for (int qi = wid; qi < q_rows; qi += nw) {
            float mo = m_i[qi], lo = l_i[qi], mn = mo;
            float sr[64];
            for (int kj = 0; kj < kr; kj++) {
                sr[kj] = S_smem[qi * kBc + kj];
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
    half* Op = Oh + q_start * D;
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) Op[i] = __float2half(O_acc[i]);
}

torch::Tensor int8_flash_attn(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    float sq, float sk, float ss, bool causal)
{
    const int B=Q.size(0), H=Q.size(1), Lq=Q.size(2), D=Q.size(3), Lkv=K.size(2);
    TORCH_CHECK(D==256);
    auto Qf=Q.reshape({B*H,Lq,D}).contiguous(), Kf=K.reshape({B*H,Lkv,D}).contiguous();
    auto Vf=V.reshape({B*H,Lkv,D}).contiguous();
    auto Of=torch::empty({B*H,Lq,D}, torch::TensorOptions().device(Q.device()).dtype(torch::kFloat16));
    constexpr int th=256;
    int nt=(Lq+kBr-1)/kBr;
    float sqk=sq*sk*ss;
    size_t sm=kBr*D + kBc*D + kBc*D*2 + kBr*kBc*4 + kBr*D*4 + kBr*4*2;
    cudaFuncSetAttribute(int8_fa_v3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    int8_fa_v3_kernel<<<dim3(nt,B*H), th, sm, at::cuda::getCurrentCUDAStream().stream()>>>(
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
