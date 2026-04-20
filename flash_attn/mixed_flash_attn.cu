/*
 * INT8 FlashAttention — CUTLASS 3.x Hopper (SM90)
 *
 * Q·K^T uses INT8 WGMMA → INT32 accumulator.
 * P·V   uses FP16 WGMMA → FP32 accumulator.
 *
 * v1: Single-CTA per (head, Q_tile), no warp specialization.
 *     TMA loads + online softmax + KV tiling loop.
 *
 * Semantics:
 *   O[Lq, D] = softmax(scale * dequant(Q @ K^T)) @ V
 *   where Q, K are INT8, V is FP16, O is FP16.
 */

#include <cuda_runtime.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// ============================================================
// Main kernel (v1: scalar correctness, WGMMA to be added in v2)
//
// D is a runtime parameter for flexibility across head dims.
// ============================================================

__global__ void int8_flash_attn_kernel(
    const int8_t* __restrict__ Q,      // (Lq, D) for one head, row-major
    const int8_t* __restrict__ K,      // (Lkv, D) for one head, row-major
    const half*    __restrict__ V,      // (Lkv, D) for one head, row-major
    half*          __restrict__ O,      // (Lq, D) for one head, row-major
    int Lq,
    int Lkv,
    int D,             // head dimension (runtime)
    float scale_q,     // Q dequant scale
    float scale_k,     // K dequant scale
    float scale_s,     // softmax scale = 1/sqrt(D)
    bool causal
) {
    // Tile config (can be tuned)
    constexpr int kBr = 64;
    constexpr int kBc = 64;

    // Thread block = one Q tile (kBr rows)
    int q_tile_idx = blockIdx.x;  // which Q tile (0..ceil(Lq/kBr)-1)
    int q_start = q_tile_idx * kBr;
    int q_end = min(q_start + kBr, Lq);
    int q_rows = q_end - q_start;

    extern __shared__ char smem_buf[];
    int8_t* Q_smem = reinterpret_cast<int8_t*>(smem_buf);
    int8_t* K_smem = Q_smem + kBr * D;
    half*   V_smem = reinterpret_cast<half*>(K_smem + kBc * D);
    // O accumulator and softmax state after V_smem
    float* O_acc = reinterpret_cast<float*>(reinterpret_cast<char*>(V_smem) + kBc * D * sizeof(half));
    float* m_i   = O_acc + kBr * D;
    float* l_i   = m_i + kBr;

    // Load Q tile (all threads cooperate)
    const int8_t* Q_ptr = Q + q_start * D;
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) {
        int r = i / D;
        int c = i % D;
        Q_smem[r * D + c] = Q_ptr[r * D + c];
    }

    // Initialize O_acc to 0, m_i to -inf, l_i to 0
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) {
        int r = i / D;
        int c = i % D;
        O_acc[r * D + c] = 0.0f;
    }
    for (int i = threadIdx.x; i < q_rows; i += blockDim.x) {
        m_i[i] = -INFINITY;
        l_i[i] = 0.0f;
    }
    __syncthreads();

    // Dequant scale for Q·K^T
    float dq_scale = scale_q * scale_k * scale_s;

    // Loop over KV tiles
    int num_kv_tiles = (Lkv + kBc - 1) / kBc;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * kBc;
        int kv_end = min(kv_start + kBc, Lkv);
        int kv_rows = kv_end - kv_start;

        // Causal mask: skip tiles where all Q positions come after all KV
        if (causal) {
            int last_q = q_end - 1;
            if (kv_start > last_q) break;  // all KV after Q → masked out
        }

        // Load K tile (1D layout: K_smem[r*D + c])
        const int8_t* K_ptr = K + kv_start * D;
        for (int i = threadIdx.x; i < kv_rows * D; i += blockDim.x) {
            int r = i / D;
            int c = i % D;
            K_smem[r * D + c] = K_ptr[r * D + c];
        }

        // Load V tile
        const half* V_ptr = V + kv_start * D;
        for (int i = threadIdx.x; i < kv_rows * D; i += blockDim.x) {
            int r = i / D;
            int c = i % D;
            V_smem[r * D + c] = V_ptr[r * D + c];
        }
        __syncthreads();

        // ---- S = Q @ K^T (INT8 GEMM, v1: scalar for correctness) ----
        int warp_id = threadIdx.x / 32;
        int num_warps = blockDim.x / 32;

        // Each warp computes assigned Q rows
        for (int qi = warp_id; qi < q_rows; qi += num_warps) {
            float m_old = m_i[qi];
            float l_old = l_i[qi];
            float m_new = m_old;

            // Compute S[qi, :] = Q_smem[qi, :] @ K_smem[:, :]^T
            float s_row[64];  // max kBc = 64
            for (int kj = 0; kj < kv_rows; kj++) s_row[kj] = 0.0f;

            for (int d = 0; d < D; d++) {
                int32_t q_val = static_cast<int32_t>(Q_smem[qi * D + d]);
                for (int kj = 0; kj < kv_rows; kj++) {
                    int32_t k_val = static_cast<int32_t>(K_smem[kj * D + d]);
                    s_row[kj] += static_cast<float>(q_val * k_val);
                }
            }

            // Dequant
            for (int kj = 0; kj < kv_rows; kj++) s_row[kj] *= dq_scale;

            // Causal mask
            if (causal) {
                int qi_global = q_start + qi;
                for (int kj = 0; kj < kv_rows; kj++) {
                    int kj_global = kv_start + kj;
                    if (kj_global > qi_global) s_row[kj] = -INFINITY;
                }
            }

            // Row max
            for (int kj = 0; kj < kv_rows; kj++) {
                m_new = fmaxf(m_new, s_row[kj]);
            }

            // Softmax + update O
            float rescale = expf(m_old - m_new);
            float l_new = l_old * rescale;
            float p_row[64];
            for (int kj = 0; kj < kv_rows; kj++) {
                p_row[kj] = expf(s_row[kj] - m_new);
                l_new += p_row[kj];
            }

            // Update O accumulator: O[qi, :] = O[qi, :]*rescale + p_row @ V
            for (int d = 0; d < D; d++) {
                float o_val = O_acc[qi * D + d] * rescale;
                for (int kj = 0; kj < kv_rows; kj++) {
                    o_val += p_row[kj] * __half2float(V_smem[kj * D + d]);
                }
                O_acc[qi * D + d] = o_val;
            }

            m_i[qi] = m_new;
            l_i[qi] = l_new;
        }
        __syncthreads();
    }

    // Final rescale: O /= l
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    for (int qi = warp_id; qi < q_rows; qi += num_warps) {
        float inv_l = (l_i[qi] > 0.0f) ? (1.0f / l_i[qi]) : 0.0f;
        for (int d = 0; d < D; d++) {
            O_acc[qi * D + d] *= inv_l;
        }
    }
    __syncthreads();

    // Write output
    half* O_ptr = O + q_start * D;
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) {
        int r = i / D;
        int c = i % D;
        O_ptr[r * D + c] = __float2half(O_acc[r * D + c]);
    }
}

// ============================================================
// Host entry point
// ============================================================

torch::Tensor int8_flash_attn(
    torch::Tensor Q_int8,    // (B, H, Lq, D) INT8
    torch::Tensor K_int8,    // (B, H, Lkv, D) INT8
    torch::Tensor V_fp16,    // (B, H, Lkv, D) FP16
    float scale_q,
    float scale_k,
    float scale_s,
    bool causal)
{
    TORCH_CHECK(Q_int8.dtype() == torch::kInt8, "Q must be int8");
    TORCH_CHECK(K_int8.dtype() == torch::kInt8, "K must be int8");
    TORCH_CHECK(V_fp16.dtype() == torch::kFloat16, "V must be float16");
    TORCH_CHECK(Q_int8.is_cuda(), "tensors must be on CUDA");

    const int B  = Q_int8.size(0);
    const int H  = Q_int8.size(1);
    const int Lq = Q_int8.size(2);
    const int D  = Q_int8.size(3);
    const int Lkv = K_int8.size(2);

    TORCH_CHECK(K_int8.size(3) == D, "K head dim mismatch");
    TORCH_CHECK(V_fp16.size(3) == D, "V head dim mismatch");

    auto O = torch::empty({B, H, Lq, D},
                          torch::TensorOptions().device(Q_int8.device()).dtype(torch::kFloat16));
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int threads = 256;  // 8 warps
    constexpr int kBr = 64;
    dim3 grid((Lq + kBr - 1) / kBr);  // one block per Q tile

    // Compute dynamic shared memory size
    // Q_smem: kBr*D int8, K_smem: Bc*D int8, V_smem: Bc*D half
    // O_acc:  kBr*D float, m_i: kBr float, l_i: kBr float
    constexpr int kBc = 64;
    size_t smem_bytes = kBr * D * sizeof(int8_t)    // Q_smem
                      + kBc * D * sizeof(int8_t)     // K_smem
                      + kBc * D * sizeof(half)       // V_smem
                      + kBr * D * sizeof(float)      // O_acc
                      + kBr * sizeof(float)          // m_i
                      + kBr * sizeof(float);         // l_i

    // Process each head separately
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            auto Q_head = Q_int8.select(0, b).select(0, h).contiguous();
            auto K_head = K_int8.select(0, b).select(0, h).contiguous();
            auto V_head = V_fp16.select(0, b).select(0, h).contiguous();
            auto O_head = O.select(0, b).select(0, h);

            const int8_t* Q_ptr = reinterpret_cast<const int8_t*>(Q_head.data_ptr());
            const int8_t* K_ptr = reinterpret_cast<const int8_t*>(K_head.data_ptr());
            const half* V_ptr = reinterpret_cast<const half*>(V_head.data_ptr());
            half* O_ptr = reinterpret_cast<half*>(O_head.data_ptr());

            int8_flash_attn_kernel
                <<<grid, threads, smem_bytes, stream>>>(
                    Q_ptr, K_ptr, V_ptr, O_ptr,
                    Lq, Lkv, D,
                    scale_q, scale_k, scale_s, causal);
        }
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("Kernel launch failed: ", cudaGetErrorString(err));
    }

    return O;
}

#else

torch::Tensor int8_flash_attn(
    torch::Tensor, torch::Tensor, torch::Tensor,
    float, float, float, bool)
{
    throw std::runtime_error("SM90 not supported");
}

#endif
