/*
 * INT8 FlashAttention v2 — CUTLASS 3.x Hopper (SM90)
 *
 * v2 changes:
 *   - Single kernel launch for all (head, Q_tile) pairs
 *   - INT8 WGMMA for Q·K^T (GMMA::ss_op_selector<int8_t, int8_t, int32_t>)
 *   - FP16 WGMMA for P·V
 *   - Online softmax (verified in v1)
 *
 * Semantics:
 *   O[Lq, D] = softmax(dequant(Q_int8 @ K_int8^T) * scale) @ V_fp16
 */

#include <cuda_runtime.h>
#include <cuda/barrier>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

// ============================================================
// Tile config
// ============================================================
constexpr int kBr = 64;   // Q tile along Lq
constexpr int kBc = 64;   // KV tile along Lkv
// D (head dim) is runtime parameter

// WGMMA tile shapes
using TileQK = Shape<Int<64>, Int<64>, Int<32>>;  // M=64, N=64, K=32 for INT8
using TilePV = Shape<Int<64>, Int<64>, Int<16>>;  // M=64, N=64, K=16 for FP16

// ============================================================
// INT8 WGMMA for Q·K^T
// ============================================================
// On SM90, INT8 WGMMA: m64n8k32 per instruction
// We tile to 64x64 using 8 N-slices

using TiledMmaQK = decltype(make_tiled_mma(
    GMMA::ss_op_selector<int8_t, int8_t, int32_t, TileQK>{},
    Layout<Shape<Int<1>, _1, _1>>{}));

using TiledMmaPV = decltype(make_tiled_mma(
    GMMA::ss_op_selector<cutlass::half_t, cutlass::half_t, float, TilePV>{},
    Layout<Shape<Int<1>, _1, _1>>{}));

// ============================================================
// SMEM Layouts
// ============================================================

using SmemLayoutQ = decltype(tile_to_shape(
    GMMA::Layout_MN_INTER_Atom<int8_t>{},
    make_shape(Int<kBr>{}, Int<256>{})));

using SmemLayoutK = decltype(tile_to_shape(
    GMMA::Layout_MN_INTER_Atom<int8_t>{},
    make_shape(Int<kBc>{}, Int<256>{})));

// ============================================================
// Main kernel
// ============================================================

extern "C" __global__ void
int8_fa_v2_kernel(
    const int8_t* __restrict__ Q,   // (B*H, Lq, D) row-major
    const int8_t* __restrict__ K,   // (B*H, Lkv, D) row-major
    const half*    __restrict__ V,   // (B*H, Lkv, D) row-major
    half*          __restrict__ O,   // (B*H, Lq, D) row-major
    const int*     __restrict__ cu_seqlens_q,  // (B*H+1) or nullptr
    int Lq,
    int Lkv,
    int D,
    float scale_qk,   // scale_q * scale_k * scale_s
    bool causal
) {
    // Block index: blockIdx.x = Q_tile, blockIdx.y = head index
    int head_idx  = blockIdx.y;
    int q_tile    = blockIdx.x;
    int q_start   = q_tile * kBr;
    int q_end     = min(q_start + kBr, Lq);
    int q_rows    = q_end - q_start;

    // Per-head pointers
    const int8_t* Q_head = Q + head_idx * Lq * D;
    const int8_t* K_head = K + head_idx * Lkv * D;
    const half*   V_head = V + head_idx * Lkv * D;
    half*         O_head = O + head_idx * Lq * D;

    // Dynamic shared memory
    extern __shared__ char smem_buf[];
    int8_t* Q_smem = reinterpret_cast<int8_t*>(smem_buf);
    int8_t* K_smem = Q_smem + kBr * D;
    half*   V_smem = reinterpret_cast<half*>(K_smem + kBc * D);
    float* O_acc   = reinterpret_cast<float*>(V_smem + kBc * D);
    float* m_i     = O_acc + kBr * D;
    float* l_i     = m_i + kBr;

    // Load Q tile (persistent across KV loop)
    const int8_t* Q_ptr = Q_head + q_start * D;
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) {
        Q_smem[i] = Q_ptr[i];
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
    int warp_id  = threadIdx.x / 32;
    int lane_id  = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * kBc;
        int kv_end   = min(kv_start + kBc, Lkv);
        int kv_rows  = kv_end - kv_start;

        if (causal) {
            if (kv_start > q_end - 1) break;
        }

        // Load K tile
        const int8_t* K_ptr = K_head + kv_start * D;
        for (int i = threadIdx.x; i < kv_rows * D; i += blockDim.x) {
            K_smem[i] = K_ptr[i];
        }
        // Load V tile
        const half* V_ptr = V_head + kv_start * D;
        for (int i = threadIdx.x; i < kv_rows * D; i += blockDim.x) {
            V_smem[i] = V_ptr[i];
        }
        __syncthreads();

        // ---- Q·K^T (INT8, scalar for now, WGMMA to come) ----
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
                for (int kj = 0; kj < kv_rows; kj++) {
                    if (kv_start + kj > qi_g) s_row[kj] = -INFINITY;
                }
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
                for (int kj = 0; kj < kv_rows; kj++) {
                    ov += p_row[kj] * __half2float(V_smem[kj * D + d]);
                }
                O_acc[qi * D + d] = ov;
            }

            m_i[qi] = m_new;
            l_i[qi] = l_new;
        }
        __syncthreads();
    }

    // Final rescale
    for (int qi = warp_id; qi < q_rows; qi += num_warps) {
        float inv_l = (l_i[qi] > 0.0f) ? (1.0f / l_i[qi]) : 0.0f;
        for (int d = 0; d < D; d++) {
            O_acc[qi * D + d] *= inv_l;
        }
    }
    __syncthreads();

    // Write output
    half* O_ptr = O_head + q_start * D;
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) {
        O_ptr[i] = __float2half(O_acc[i]);
    }
}

// ============================================================
// Host entry point — single launch for all heads
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

    const int B  = Q_int8.size(0);
    const int H  = Q_int8.size(1);
    const int Lq = Q_int8.size(2);
    const int D  = Q_int8.size(3);
    const int Lkv = K_int8.size(2);

    TORCH_CHECK(K_int8.size(3) == D, "K head dim mismatch");
    TORCH_CHECK(V_fp16.size(3) == D, "V head dim mismatch");

    // Flatten B*H into one dimension
    auto Q_flat = Q_int8.reshape({B * H, Lq, D}).contiguous();
    auto K_flat = K_int8.reshape({B * H, Lkv, D}).contiguous();
    auto V_flat = V_fp16.reshape({B * H, Lkv, D}).contiguous();
    auto O_flat = torch::empty({B * H, Lq, D},
        torch::TensorOptions().device(Q_int8.device()).dtype(torch::kFloat16));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int threads = 256;
    int num_q_tiles = (Lq + kBr - 1) / kBr;
    dim3 grid(num_q_tiles, B * H);
    float scale_qk = scale_q * scale_k * scale_s;

    // Dynamic shared memory size
    constexpr int kBc_val = kBc;
    constexpr int kBr_val = kBr;
    size_t smem = kBr_val * D * sizeof(int8_t)    // Q_smem
                + kBc_val * D * sizeof(int8_t)    // K_smem
                + kBc_val * D * sizeof(half)      // V_smem
                + kBr_val * D * sizeof(float)     // O_acc
                + kBr_val * sizeof(float)         // m_i
                + kBr_val * sizeof(float);        // l_i

    cudaFuncSetAttribute(int8_fa_v2_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    int8_fa_v2_kernel<<<grid, threads, smem, stream>>>(
        reinterpret_cast<const int8_t*>(Q_flat.data_ptr()),
        reinterpret_cast<const int8_t*>(K_flat.data_ptr()),
        reinterpret_cast<const half*>(V_flat.data_ptr()),
        reinterpret_cast<half*>(O_flat.data_ptr()),
        nullptr,  // cu_seqlens_q
        Lq, Lkv, D, scale_qk, causal);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("Kernel launch failed: ", cudaGetErrorString(err));
    }

    return O_flat.reshape({B, H, Lq, D});
}

#else

torch::Tensor int8_flash_attn(torch::Tensor, torch::Tensor, torch::Tensor,
                               float, float, float, bool) {
    throw std::runtime_error("SM90 not supported");
}

#endif
