/*
 * INT8 FlashAttention v3 — CUTLASS 3.x Hopper (SM90) with WGMMA
 *
 * v3: INT8 WGMMA for Q·K^T using SM90_64x64x32_S32S8S8_SS_TN atom.
 *     GMMA interleaved SMEM layouts + cute::gemm().
 * v2: Single launch for all heads, scalar Q·K^T.
 *
 * Semantics:
 *   O[Lq, D] = softmax(dequant(Q_int8 @ K_int8^T) * scale) @ V_fp16
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

// ============================================================
// Tile config
// ============================================================
constexpr int kBr = 64;   // Q tile along Lq
constexpr int kBc = 64;   // KV tile along Lkv
// D = head dim, runtime parameter (must be multiple of 32 for WGMMA)

// INT8 WGMMA: SS mode needs K=64 (sparse), D=256 → 4 iterations
// ss_op_selector is a function → need decltype(...()) to get return type
using TiledMmaQK = decltype(make_tiled_mma(
    decltype(GMMA::ss_op_selector<int8_t, int8_t, int32_t,
        Shape<Int<64>, Int<64>, Int<64>>>()){},
    Layout<Shape<_1, _1, _1>>{}));

// ============================================================
// GMMA interleaved SMEM layouts for INT8
// Both operands use GMMA::Major::K (SS mode with TN layout)
// ============================================================
// Q: (M=64, K=256), K-major interleaved
using SmemLayoutQ = decltype(tile_to_shape(
    GMMA::Layout_K_INTER_Atom<int8_t>{},
    make_shape(Int<kBr>{}, Int<256>{})));

// K: (N=64, K=256), K-major interleaved (= col-major for K^T access)
using SmemLayoutK = decltype(tile_to_shape(
    GMMA::Layout_K_INTER_Atom<int8_t>{},
    make_shape(Int<kBc>{}, Int<256>{})));

// ============================================================
// Main kernel
// ============================================================

extern "C" __global__ void
int8_fa_v3_kernel(
    const int8_t* __restrict__ Q,   // (B*H, Lq, D) row-major
    const int8_t* __restrict__ K,   // (B*H, Lkv, D) row-major
    const half*    __restrict__ V,   // (B*H, Lkv, D) row-major
    half*          __restrict__ O,   // (B*H, Lq, D) row-major
    int Lq, int Lkv, int D,
    float scale_qk,    // scale_q * scale_k * scale_s
    bool causal
) {
    // Restrict to 128 threads (1 warpgroup) for WGMMA
    // blockIdx.x = Q_tile, blockIdx.y = head index
    int head_idx  = blockIdx.y;
    int q_tile    = blockIdx.x;
    int q_start   = q_tile * kBr;
    int q_end     = min(q_start + kBr, Lq);
    int q_rows    = q_end - q_start;

    const int8_t* Q_head = Q + head_idx * Lq * D;
    const int8_t* K_head = K + head_idx * Lkv * D;
    const half*   V_head = V + head_idx * Lkv * D;
    half*         O_head = O + head_idx * Lq * D;

    // Shared memory: Q (interleaved) + K (interleaved) + V + O_acc + softmax
    extern __shared__ char smem_buf[];

    int8_t* Q_smem = reinterpret_cast<int8_t*>(smem_buf);
    size_t q_sz = cosize(SmemLayoutQ{});
    int8_t* K_smem = Q_smem + q_sz;
    size_t k_sz = cosize(SmemLayoutK{});
    half*   V_smem = reinterpret_cast<half*>(K_smem + k_sz);
    float* O_acc   = reinterpret_cast<float*>(V_smem + kBc * D);
    float* m_i     = O_acc + kBr * D;
    float* l_i     = m_i + kBr;

    // Create SMEM tensors with interleaved layouts
    Tensor sQ_t = make_tensor(make_smem_ptr(Q_smem), SmemLayoutQ{});
    Tensor sK_t = make_tensor(make_smem_ptr(K_smem), SmemLayoutK{});

    // ---- Load Q into SMEM (row-major for now, interleaved later) ----
    const int8_t* Q_ptr = Q_head + q_start * D;
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) {
        Q_smem[i] = Q_ptr[i];  // FIXME: should use interleaved layout
    }

    // Init accumulators
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) {
        O_acc[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < q_rows; i += blockDim.x) {
        m_i[i] = -INFINITY;
        l_i[i] = 0.0f;
    }
    __syncthreads();

    // ---- KV loop ----
    int num_kv_tiles = (Lkv + kBc - 1) / kBc;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * kBc;
        int kv_end   = min(kv_start + kBc, Lkv);
        int kv_rows  = kv_end - kv_start;

        if (causal && kv_start > q_end - 1) break;

        // Load K into interleaved col-major SMEM (= K^T row-major)
        const int8_t* K_ptr = K_head + kv_start * D;
        for (int i = threadIdx.x; i < kv_rows * D; i += blockDim.x) {
            int r = i / D;
            int c = i % D;
            // Store as col-major: K_smem[c * kBc + r] = K_ptr[r * D + c]
            // For now, store row-major (same as scalar)
            K_smem[r * D + c] = K_ptr[r * D + c];
        }
        // Load V
        const half* V_ptr = V_head + kv_start * D;
        for (int i = threadIdx.x; i < kv_rows * D; i += blockDim.x) {
            V_smem[i] = V_ptr[i];
        }
        __syncthreads();

        // ---- Q·K^T via INT8 WGMMA (warpgroup 0 only) ----
        if (threadIdx.x < 128) {
            TiledMmaQK tiled_mma_qk;
            auto wg_slice = tiled_mma_qk.get_slice(threadIdx.x);

            // Partition SMEM tensors for WGMMA
            Tensor tSrQ = wg_slice.partition_A(sQ_t);
            Tensor tSrK = wg_slice.partition_B(sK_t);
            Tensor tSrS = partition_fragment_C(tiled_mma_qk,
                make_shape(Int<kBr>{}, Int<kBc>{}));

            // Clear INT32 accumulator
            for (int i = 0; i < size(tSrS); i++) tSrS(i) = int32_t(0);

            // INT8 WGMMA loop over K (256/64 = 4 sparse K tiles)
            constexpr int k_iters = 256 / 64;
            #pragma unroll
            for (int k = 0; k < k_iters; k++) {
                cute::gemm(tiled_mma_qk,
                    tSrQ(_, _, k), tSrK(_, _, k), tSrS);
                tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
            }
            // tSrS now contains INT32 S = Q @ K^T
            // TODO: dequant + write to SMEM for softmax
        }
        __syncthreads();

        // ---- Scalar Q·K^T (fallback) ----
        int num_warps = blockDim.x / 32;
        int warp_id  = threadIdx.x / 32;

        for (int qi = warp_id; qi < q_rows; qi += num_warps) {
            float m_old = m_i[qi];
            float l_old = l_i[qi];
            float m_new = m_old;

            float s_row[64];
            for (int kj = 0; kj < kv_rows; kj++) s_row[kj] = 0.0f;

            for (int d = 0; d < D; d++) {
                int32_t qv = static_cast<int32_t>(Q_smem[qi * D + d]);
                for (int kj = 0; kj < kv_rows; kj++) {
                    s_row[kj] += static_cast<float>(
                        qv * static_cast<int32_t>(K_smem[kj * D + d]));
                }
            }
            for (int kj = 0; kj < kv_rows; kj++) s_row[kj] *= scale_qk;

            if (causal) {
                int qi_g = q_start + qi;
                for (int kj = 0; kj < kv_rows; kj++)
                    if (kv_start + kj > qi_g) s_row[kj] = -INFINITY;
            }

            for (int kj = 0; kj < kv_rows; kj++)
                m_new = fmaxf(m_new, s_row[kj]);

            float rescale = expf(m_old - m_new);
            float l_new = l_old * rescale;
            float p_row[64];
            for (int kj = 0; kj < kv_rows; kj++) {
                p_row[kj] = expf(s_row[kj] - m_new);
                l_new += p_row[kj];
            }

            for (int d = 0; d < D; d++) {
                float ov = O_acc[qi * D + d] * rescale;
                for (int kj = 0; kj < kv_rows; kj++)
                    ov += p_row[kj] * __half2float(V_smem[kj * D + d]);
                O_acc[qi * D + d] = ov;
            }
            m_i[qi] = m_new;
            l_i[qi] = l_new;
        }
        __syncthreads();
    }

    // Final rescale
    for (int qi = threadIdx.x; qi < q_rows; qi += blockDim.x) {
        float inv_l = (l_i[qi] > 0.0f) ? (1.0f / l_i[qi]) : 0.0f;
        for (int d = 0; d < D; d++)
            O_acc[qi * D + d] *= inv_l;
    }
    __syncthreads();

    // Write output
    half* O_ptr = O_head + q_start * D;
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x)
        O_ptr[i] = __float2half(O_acc[i]);
}

// ============================================================
// Host entry point
// ============================================================

torch::Tensor int8_flash_attn(
    torch::Tensor Q_int8, torch::Tensor K_int8, torch::Tensor V_fp16,
    float scale_q, float scale_k, float scale_s, bool causal)
{
    TORCH_CHECK(Q_int8.dtype() == torch::kInt8, "Q must be int8");
    TORCH_CHECK(K_int8.dtype() == torch::kInt8, "K must be int8");
    TORCH_CHECK(V_fp16.dtype() == torch::kFloat16, "V must be float16");

    const int B = Q_int8.size(0), H = Q_int8.size(1);
    const int Lq = Q_int8.size(2), D = Q_int8.size(3);
    const int Lkv = K_int8.size(2);
    TORCH_CHECK(D == 256, "D must be 256 for v3 (hardcoded SMEM layout)");

    auto Q_flat = Q_int8.reshape({B * H, Lq, D}).contiguous();
    auto K_flat = K_int8.reshape({B * H, Lkv, D}).contiguous();
    auto V_flat = V_fp16.reshape({B * H, Lkv, D}).contiguous();
    auto O_flat = torch::empty({B * H, Lq, D},
        torch::TensorOptions().device(Q_int8.device()).dtype(torch::kFloat16));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int threads = 256;  // 8 warps
    int num_q_tiles = (Lq + kBr - 1) / kBr;
    dim3 grid(num_q_tiles, B * H);
    float scale_qk = scale_q * scale_k * scale_s;

    // SMEM: Q (interleaved) + K (interleaved) + V + O_acc + softmax
    size_t smem = cosize(SmemLayoutQ{}) + cosize(SmemLayoutK{})
                + kBc * D * sizeof(half)          // V_smem
                + kBr * D * sizeof(float)         // O_acc
                + kBr * sizeof(float) * 2;        // m_i, l_i

    cudaFuncSetAttribute(int8_fa_v3_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    int8_fa_v3_kernel<<<grid, threads, smem, stream>>>(
        reinterpret_cast<const int8_t*>(Q_flat.data_ptr()),
        reinterpret_cast<const int8_t*>(K_flat.data_ptr()),
        reinterpret_cast<const half*>(V_flat.data_ptr()),
        reinterpret_cast<half*>(O_flat.data_ptr()),
        Lq, Lkv, D, scale_qk, causal);

    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        AT_ERROR("Kernel launch failed: ", cudaGetErrorString(err));

    return O_flat.reshape({B, H, Lq, D});
}

#else
torch::Tensor int8_flash_attn(torch::Tensor, torch::Tensor, torch::Tensor,
                               float, float, float, bool) {
    throw std::runtime_error("SM90 not supported");
}
#endif
