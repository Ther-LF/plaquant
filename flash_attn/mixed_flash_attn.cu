/*
 * INT8 FlashAttention — CUTLASS 3.x Hopper (SM90)
 *
 * v2: Single launch for all heads, scalar Q·K^T (correct, CosSim=0.9999)
 * v3 WIP: INT8 WGMMA via CUTE partition + inline PTX
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
// D = 256 hardcoded

// INT8 WGMMA atom
using MmaAtomQK = decltype(GMMA::ss_op_selector<int8_t, int8_t, int32_t,
    Shape<Int<64>, Int<64>, Int<64>>>());
using TiledMmaQK = decltype(make_tiled_mma(MmaAtomQK{}, Layout<Shape<_1, _1, _1>>{}));

// SMEM layouts for GMMA (K-major interleaved)
using SmemLayoutQ = decltype(tile_to_shape(
    GMMA::Layout_K_INTER_Atom<int8_t>{},
    make_shape(Int<kBr>{}, Int<256>{})));
using SmemLayoutK = decltype(tile_to_shape(
    GMMA::Layout_K_INTER_Atom<int8_t>{},
    make_shape(Int<kBc>{}, Int<256>{})));

// ============================================================
// SMEM index for K-major interleaved layout
// ============================================================
__device__ inline int smem_idx_kmaj(int r, int c, int M) {
    // K-major with 32-element interleave
    int group = c / 32;
    int c_in  = c % 32;
    return group * 32 * M + c_in * M + r;
}

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
    float scale_qk,
    bool causal
) {
    int head_idx = blockIdx.y;
    int q_tile   = blockIdx.x;
    int q_start  = q_tile * kBr;
    int q_end    = min(q_start + kBr, Lq);
    int q_rows   = q_end - q_start;

    const int8_t* Q_head = Q + head_idx * Lq * D;
    const int8_t* K_head = K + head_idx * Lkv * D;
    const half*   V_head = V + head_idx * Lkv * D;
    half*         O_head = O + head_idx * Lq * D;

    // Shared memory
    extern __shared__ char smem_buf[];
    int8_t* Q_smem = reinterpret_cast<int8_t*>(smem_buf);
    size_t q_sz = cosize(SmemLayoutQ{});
    int8_t* K_smem = Q_smem + q_sz;
    size_t k_sz = cosize(SmemLayoutK{});
    half*   V_smem = reinterpret_cast<half*>(K_smem + k_sz);
    float*  S_smem = reinterpret_cast<float*>(V_smem + kBc * D); // (kBr, kBc) FP32 S
    float*  O_acc  = S_smem + kBr * kBc;
    float*  m_i    = O_acc + kBr * D;
    float*  l_i    = m_i + kBr;

    // Load Q into interleaved SMEM
    const int8_t* Q_ptr = Q_head + q_start * D;
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) {
        int r = i / D, c = i % D;
        Q_smem[smem_idx_kmaj(r, c, kBr)] = Q_ptr[i];
    }

    // Init accumulators
    for (int i = threadIdx.x; i < q_rows * D; i += blockDim.x) O_acc[i] = 0.0f;
    for (int i = threadIdx.x; i < q_rows; i += blockDim.x) {
        m_i[i] = -INFINITY;
        l_i[i] = 0.0f;
    }
    __syncthreads();

    // KV loop
    int num_kv_tiles = (Lkv + kBc - 1) / kBc;
    int warp_id  = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * kBc;
        int kv_end   = min(kv_start + kBc, Lkv);
        int kv_rows  = kv_end - kv_start;
        if (causal && kv_start > q_end - 1) break;

        // Load K into interleaved SMEM
        const int8_t* K_ptr = K_head + kv_start * D;
        for (int i = threadIdx.x; i < kv_rows * D; i += blockDim.x) {
            int r = i / D, c = i % D;
            K_smem[smem_idx_kmaj(r, c, kBc)] = K_ptr[i];
        }
        // Load V (plain row-major)
        const half* V_ptr = V_head + kv_start * D;
        for (int i = threadIdx.x; i < kv_rows * D; i += blockDim.x)
            V_smem[i] = V_ptr[i];
        __syncthreads();

        // ---- INT8 WGMMA for Q·K^T via atom.fma() ----
        uint32_t S_acc[32];
        for (int i = 0; i < 32; i++) S_acc[i] = 0;

        MmaAtomQK mma_atom;

        // 8 K iterations (D=256, K_tile=32)
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            // SMEM pointer for this K tile (K-major: stride = M bytes)
            int8_t* q_k = Q_smem + k * 32 * kBr;
            int8_t* k_k = K_smem + k * 32 * kBc;

            // Create single-tile GMMA tensor + descriptor
            // Layout_K_INTER_Atom<int8_t> has native shape (8, 16)
            // tile_to_shape to get (64, 32) for WGMMA m64n64k32
            auto k_tile_layout = tile_to_shape(
                GMMA::Layout_K_INTER_Atom<int8_t>{},
                make_shape(Int<kBr>{}, Int<32>{}));
            Tensor sQ_tile = make_tensor(make_smem_ptr(q_k), k_tile_layout);
            Tensor sK_tile = make_tensor(make_smem_ptr(k_k), k_tile_layout);

            uint64_t desc_q = cute::SM90::GMMA::make_gmma_desc<GMMA::Major::K>(sQ_tile);
            uint64_t desc_k = cute::SM90::GMMA::make_gmma_desc<GMMA::Major::K>(sK_tile);

            auto scale = (k == 0) ? GMMA::ScaleOut::Zero
                                  : GMMA::ScaleOut::One;
            mma_atom.fma(desc_q, desc_k,
                S_acc[0],  S_acc[1],  S_acc[2],  S_acc[3],
                S_acc[4],  S_acc[5],  S_acc[6],  S_acc[7],
                S_acc[8],  S_acc[9],  S_acc[10], S_acc[11],
                S_acc[12], S_acc[13], S_acc[14], S_acc[15],
                S_acc[16], S_acc[17], S_acc[18], S_acc[19],
                S_acc[20], S_acc[21], S_acc[22], S_acc[23],
                S_acc[24], S_acc[25], S_acc[26], S_acc[27],
                S_acc[28], S_acc[29], S_acc[30], S_acc[31],
                scale);
        }

        // Extract WGMMA accumulator to S_smem via CUTE copy
        TiledMmaQK tiled_mma;
        auto wg_slice = tiled_mma.get_slice(threadIdx.x);

        // Create fragment and fill with WGMMA results
        auto tSrS = partition_fragment_C(tiled_mma,
            make_shape(Int<kBr>{}, Int<kBc>{}));
        for (int i = 0; i < size(tSrS); i++)
            tSrS(i) = int32_t(S_acc[i]);

        // Create SMEM tensor for output
        Tensor sS = make_tensor(make_smem_ptr(S_smem),
            make_layout(make_shape(Int<kBr>{}, Int<kBc>{}),
                        LayoutRight{}));

        // Copy fragment to SMEM (CUTE handles thread-to-element mapping)
        // Use the warpgroup slice to write each thread's fragment
        auto tSrS_smem = wg_slice.retile_C(sS);
        for (int i = 0; i < size(tSrS_smem); i++) {
            float val = float(int32_t(tSrS(i))) * scale_qk;
            // tSrS and tSrS_smem have the same thread-level layout
            // so element i in both corresponds to the same (row, col)
            tSrS_smem(i) = val;
        }

        // Write S_acc to SMEM (simplified: each thread writes to its row)
        // The accumulator mapping: for M64×N64, 128 threads,
        // each thread holds 32 values covering specific (row, col) pairs.
        // For now, store linearly: thread t writes to S_smem[t * 32 .. t*32+31]
        int tid = threadIdx.x;
        if (tid < 128) {
            for (int i = 0; i < 32 && (tid * 32 + i) < kBr * kBc; i++) {
                S_smem[tid * 32 + i] = (float)S_acc[i] * scale_qk;
            }
        }
        __syncthreads();

        // ---- Softmax + P·V using S_smem ----
        for (int qi = warp_id; qi < q_rows; qi += num_warps) {
            float m_old = m_i[qi], l_old = l_i[qi], m_new = m_old;
            float s_row[64], p_row[64];

            for (int kj = 0; kj < kv_rows; kj++) {
                s_row[kj] = S_smem[qi * kBc + kj];  // read WGMMA output
                if (causal && (kv_start + kj > q_start + qi))
                    s_row[kj] = -INFINITY;
                m_new = fmaxf(m_new, s_row[kj]);
            }

            float rescale = expf(m_old - m_new);
            float l_new = l_old * rescale;
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

    // Final rescale + write
    for (int qi = threadIdx.x; qi < q_rows; qi += blockDim.x) {
        float inv_l = (l_i[qi] > 0) ? 1.0f / l_i[qi] : 0.0f;
        for (int d = 0; d < D; d++)
            O_acc[qi * D + d] *= inv_l;
    }
    __syncthreads();

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
    TORCH_CHECK(D == 256, "D must be 256 for v3");

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

    size_t smem = cosize(SmemLayoutQ{}) + cosize(SmemLayoutK{})
                + kBc * D * sizeof(half)              // V_smem
                + kBr * kBc * sizeof(float)           // S_smem (WGMMA output)
                + kBr * D * sizeof(float)             // O_acc
                + kBr * sizeof(float) * 2;            // m_i, l_i

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
