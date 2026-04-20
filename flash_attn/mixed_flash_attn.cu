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
#include <cuda/barrier>
#include <cuda/ptx_instructions>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

// ============================================================
// Tile config
// ============================================================
constexpr int Br = 64;   // Q tile size (rows)
constexpr int Bc = 64;   // KV tile size (rows)
constexpr int D  = 256;  // head dimension (must match k_high + k_low)

// WGMMA shapes
// INT8:  m64 x n8 x k32
// FP16:  m64 x n8 x k16

// ============================================================
// Online softmax helpers
// ============================================================

__device__ __forceinline__
float warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__
float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================
// INT8 WGMMA wrapper (Q·K^T): D_frag[m,n] = A[m,k] @ B[n,k]^T
// ============================================================

// Accumulator: 64×64 INT32 distributed across 32 threads in a warp
// Each thread in the warpgroup holds a slice

struct Int8WgmmaAccum {
    // For M=64, N=64, INT8 wgmma produces INT32 fragments
    // Distributed across 4 warps (warpgroup of 128 threads)
    // Each thread holds multiple INT32 values
    int32_t data[32];  // enough space for the fragment
    int count;

    __device__ void clear() {
        #pragma unroll
        for (int i = 0; i < 32; i++) data[i] = 0;
        count = 0;
    }
};

// Launch INT8 WGMMA: accumulator += A_smem @ B_smem^T
// A_smem: (M, K) INT8 in shared memory, row-major
// B_smem: (N, K) INT8 in shared memory, row-major (B is K,V stored row-major)
// We compute: S[M,N] += A[M,K] @ B[N,K]^T
template<int M, int N, int K>
__device__ void int8_wgmma_s8s8(
    Int8WgmmaAccum& accum,
    const int8_t* A_smem,  // (M, K) row-major
    const int8_t* B_smem   // (N, K) row-major (we need col-major access = B^T)
) {
    // WGMMA for INT8 on SM90: m64×n8×k32
    // Operand A from SMEM (row-major), Operand B from SMEM (col-major for B^T)
    // Actually B is stored row-major (N, K), we need it as (K, N) = B^T
    // WGMMA descriptor needs to describe the transpose

    constexpr int k_tile = 32;
    static_assert(K % k_tile == 0, "K must be multiple of 32");
    static_assert(M == 64, "M must be 64 for WGMMA");
    static_assert(N % 8 == 0, "N must be multiple of 8");

    // Use inline PTX for wgmma.mma_async.sync.aligned.m64n8k32.s8.s8.s32
    // A desc: smem, row-major (major=row)
    // B desc: smem, col-major (major=col) ← because we need B^T

    // For now, use a simpler path: load to registers and use mma.sync
    // In production, this should use proper WGMMA with descriptors

    // This is a placeholder — will be replaced with proper WGMMA PTX
    asm volatile(
        "// wgmma.mma_async.sync.aligned.m64n8k32.s8.s8.s32 placeholder\n"
        :: "r"(A_smem), "r"(B_smem)
    );
}

// ============================================================
// FP16 WGMMA wrapper (P·V): O_frag[m,d] = P[m,n] @ V[n,d]
// ============================================================

struct Fp16WgmmaAccum {
    float data[64];  // enough for 64×256 FP32 accumulator per thread
    int count;

    __device__ void clear() {
        #pragma unroll
        for (int i = 0; i < 64; i++) data[i] = 0.0f;
    }
};

// ============================================================
// Main kernel
// ============================================================

template<int kBr, int kBc, int kD>
__global__ void int8_flash_attn_kernel(
    const int8_t* __restrict__ Q,      // (Lq, D) for one head, row-major
    const int8_t* __restrict__ K,      // (Lkv, D) for one head, row-major
    const half*    __restrict__ V,      // (Lkv, D) for one head, row-major
    half*          __restrict__ O,      // (Lq, D) for one head, row-major
    int Lq,
    int Lkv,
    float scale_q,     // Q dequant scale
    float scale_k,     // K dequant scale
    float scale_s,     // softmax scale = 1/sqrt(D)
    bool causal
) {
    // Thread block = one Q tile (Br rows)
    int q_tile_idx = blockIdx.x;  // which Q tile (0..ceil(Lq/Br)-1)
    int q_start = q_tile_idx * kBr;
    int q_end = min(q_start + kBr, Lq);
    int q_rows = q_end - q_start;

    // Shared memory
    __shared__ int8_t Q_smem[kBr][kD];  // Q tile
    __shared__ int8_t K_smem[kBc][kD];  // K tile
    __shared__ half   V_smem[kBc][kD];  // V tile

    // Load Q tile (all threads cooperate)
    const int8_t* Q_ptr = Q + q_start * kD;
    for (int i = threadIdx.x; i < q_rows * kD; i += blockDim.x) {
        int r = i / kD;
        int c = i % kD;
        Q_smem[r][c] = Q_ptr[r * kD + c];
    }
    __syncthreads();

    // Output accumulator (in registers)
    // O_acc: (Br, D) FP32, distributed across threads
    // For simplicity in v1: use shared memory for O accumulator
    __shared__ float O_acc[kBr][kD];

    // Initialize O_acc to 0
    for (int i = threadIdx.x; i < q_rows * kD; i += blockDim.x) {
        int r = i / kD;
        int c = i % kD;
        O_acc[r][c] = 0.0f;
    }

    // Online softmax state
    __shared__ float m_i[kBr];   // row max
    __shared__ float l_i[kBr];   // row sum
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

        // Load K tile
        const int8_t* K_ptr = K + kv_start * kD;
        for (int i = threadIdx.x; i < kv_rows * kD; i += blockDim.x) {
            int r = i / kD;
            int c = i % kD;
            K_smem[r][c] = K_ptr[r * kD + c];
        }
        __syncthreads();

        // ---- S = Q @ K^T (INT8 GEMM, computed in FP32 for now) ----
        // v1: Use warp-level cooperative computation
        // Each warp computes a (32, kv_rows) tile of S
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        int num_warps = blockDim.x / 32;

        // Warp computes rows assigned to it
        for (int qi = warp_id; qi < q_rows; qi += num_warps) {
            float m_old = m_i[qi];
            float l_old = l_i[qi];
            float m_new = m_old;

            // Compute S[qi, :] = Q_smem[qi, :] @ K_smem[:, :]^T
            // Do this in chunks to use registers
            float s_row[kBc];
            #pragma unroll
            for (int kj = 0; kj < kBc; kj++) s_row[kj] = 0.0f;

            // Dot product over D
            for (int d = 0; d < kD; d++) {
                int8_t q_val = Q_smem[qi][d];
                int32_t q_i32 = static_cast<int32_t>(q_val);
                #pragma unroll
                for (int kj = 0; kj < kv_rows; kj++) {
                    int32_t k_val = static_cast<int32_t>(K_smem[kj][d]);
                    s_row[kj] += static_cast<float>(q_i32 * k_val);
                }
            }

            // Dequant and scale
            #pragma unroll
            for (int kj = 0; kj < kv_rows; kj++) {
                s_row[kj] *= dq_scale;
            }

            // Causal mask
            if (causal) {
                int qi_global = q_start + qi;
                #pragma unroll
                for (int kj = 0; kj < kv_rows; kj++) {
                    int kj_global = kv_start + kj;
                    if (kj_global > qi_global) s_row[kj] = -INFINITY;
                }
            }

            // Find row max
            #pragma unroll
            for (int kj = 0; kj < kv_rows; kj++) {
                m_new = fmaxf(m_new, s_row[kj]);
            }

            // Rescale and compute exp
            float l_new = 0.0f;
            float p_row[kBc];
            float rescale = expf(m_old - m_new);

            #pragma unroll
            for (int kj = 0; kj < kv_rows; kj++) {
                p_row[kj] = expf(s_row[kj] - m_new);
                l_new += p_row[kj];
            }
            l_new += l_old * rescale;

            // Update O accumulator
            for (int d = 0; d < kD; d++) {
                float o_val = O_acc[qi][d] * rescale;
                #pragma unroll
                for (int kj = 0; kj < kv_rows; kj++) {
                    o_val += p_row[kj] * __half2float(V_smem[kj][d]);
                }
                O_acc[qi][d] = o_val;
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
        for (int d = 0; d < kD; d++) {
            O_acc[qi][d] *= inv_l;
        }
    }
    __syncthreads();

    // Write output
    half* O_ptr = O + q_start * kD;
    for (int i = threadIdx.x; i < q_rows * kD; i += blockDim.x) {
        int r = i / kD;
        int c = i % kD;
        O_ptr[r * kD + c] = __float2half(O_acc[r][c]);
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

    TORCH_CHECK(D == ::D, "head dim must match D template param (", ::D, ")");
    TORCH_CHECK(K_int8.size(3) == D, "K head dim mismatch");
    TORCH_CHECK(V_fp16.size(3) == D, "V head dim mismatch");

    auto O = torch::empty({B, H, Lq, D},
                          torch::TensorOptions().device(Q_int8.device()).dtype(torch::kFloat16));
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int threads = 256;  // 8 warps
    dim3 grid(Lq / Br + (Lq % Br ? 1 : 0));  // one block per Q tile

    // Process each head separately
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            const int8_t* Q_ptr = reinterpret_cast<const int8_t*>(
                Q_int8.index({b, h}).data_ptr());
            const int8_t* K_ptr = reinterpret_cast<const int8_t*>(
                K_int8.index({b, h}).data_ptr());
            const half* V_ptr = reinterpret_cast<const half*>(
                V_fp16.index({b, h}).data_ptr());
            half* O_ptr = reinterpret_cast<half*>(
                O.index({b, h}).data_ptr());

            int8_flash_attn_kernel<Br, Bc, D>
                <<<grid, threads, 0, stream>>>(
                    Q_ptr, K_ptr, V_ptr, O_ptr,
                    Lq, Lkv, scale_q, scale_k, scale_s, causal);
        }
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
