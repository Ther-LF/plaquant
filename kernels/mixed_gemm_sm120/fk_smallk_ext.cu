// fk-cuda-cflags: -arch=sm_121a
// fk_smallk_ext -- KDA round 10: ForgeKernel-evolved dual-phase (NVFP4+MXFP8) fused GEMM
// for the GB10 small-K / output-bound window (run promix_dualphase_gemm_261125-202319-9a91,
// confirmed 1.146x the shipping codesign reference on its graded shapes; merged under the
// measured _FK_TABLE_GB10 windows in promix/quantize/mixed_linear.py).
//
// vs the shipping CUTLASS codesign kernel, the structural difference is the epilogue
// (D0.10): the whole 128x128 fp16 tile is staged in smem and ONE bulk TMA store group is
// left in flight across the tile boundary (cp.async.bulk.wait_group.read 0 only at the
// NEXT tile's epilogue entry), so the output-write stream overlaps the next tile's
// mainloop -- the recon-measured failure mode of the small-K shapes was a SERIALIZED
// write + mainloop sum. NV pipeline 3 stages (vs 4), MX 2; tile walk sw1/AlongM.
//
// Added for the merge (not in the evolved artifact): epilogue alpha (the two-level
// weight global g_w rides it, like every codesign entry) and run_prepacked(), which
// consumes the pipeline's already-interleaved physical SF buffers directly (the evolved
// kernel's blocked SF layout was verified torch.equal to prepack_sf_*): steady state is
// ONE kernel launch with zero prep work.
//
// Value-returning: return the output tensor.
//
// Shapes (one kernel serves all):
//     gu1b_r8: {"m": 8192, "n": 8192, "k_lo": 1792, "k_hi": 256, "k_lo_b": 896, "k_lo_sf": 112, "k_hi_sf": 8}
//     gu1b_r2: {"m": 8192, "n": 8192, "k_lo": 1024, "k_hi": 1024, "k_lo_b": 512, "k_lo_sf": 64, "k_hi_sf": 32}
//     gu3b_r8: {"m": 8192, "n": 8192, "k_lo": 2688, "k_hi": 384, "k_lo_b": 1344, "k_lo_sf": 168, "k_hi_sf": 12}
// Inputs:
//     a_lo: int8 [m, k_lo_b]      sfa_lo: int8 [m, k_lo_sf]
//     a_hi: int8 [m, k_hi]        sfa_hi: int8 [m, k_hi_sf]
//     b_lo: int8 [n, k_lo_b]      sfb_lo: int8 [n, k_lo_sf]
//     b_hi: int8 [n, k_hi]        sfb_hi: int8 [n, k_hi_sf]
//     bias: float16 [n]
// Outputs:
//     y: float16 [m, n]
//
// Structure (milestone M0):
//   - SF prep kernel (cold call / SF change only): writes a blocked copy of each SF tensor,
//     512-B atoms per (128-row panel, 4 scale blocks), cached across calls (D0.3).
//   - one persistent GEMM kernel: grid 48, block 384 = producer WG + 2 cooperative MMA WGs,
//     tile 128x128x128, parameterized tile walk (D0.4/D0.5).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <array>
#include <cstdint>
#include <vector>

// Block-scaled mma.sync exists only in the arch-specific target (sm_120a / sm_121a). A build
// that also lowers a generic compute_12x target compiles a trap there instead of failing ptxas.
#if defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || \
    (defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ >= 1200) || \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200)
#define FK_BLOCK_SCALE_MMA 1
#else
#define FK_BLOCK_SCALE_MMA 0
#endif

// ------------------------------------------------------------------------------------
// Tile / smem constants (D0.4, D0.6)
// ------------------------------------------------------------------------------------
constexpr int BM = 128, BN = 128, BK = 128;
constexpr int NUM_THREADS = 384;
constexpr int NUM_CTAS = 48;

constexpr int NV_STAGES = 3;
constexpr int MX_STAGES = 2;

constexpr uint32_t STAGING_OFF = 0;
constexpr uint32_t STAGING_BYTES = BM * BN * 2;                    // 32768
constexpr uint32_t UNION_OFF = STAGING_OFF + STAGING_BYTES;        // 0x8000

constexpr uint32_t NV_A_BYTES = BM * BK / 2;                       // 8192
constexpr uint32_t NV_B_BYTES = BN * BK / 2;                       // 8192
constexpr uint32_t NV_SF_BYTES = 1024;                             // 2 atoms
constexpr uint32_t NV_STAGE_BYTES = NV_A_BYTES + NV_B_BYTES + 2 * NV_SF_BYTES;   // 18432
constexpr uint32_t MX_A_BYTES = BM * BK;                           // 16384
constexpr uint32_t MX_B_BYTES = BN * BK;                           // 16384
constexpr uint32_t MX_SF_BYTES = 512;                              // 1 atom
constexpr uint32_t MX_STAGE_BYTES = MX_A_BYTES + MX_B_BYTES + 2 * MX_SF_BYTES;   // 33792

constexpr uint32_t NV_UNION_BYTES = NV_STAGES * NV_STAGE_BYTES;    // 55296
constexpr uint32_t MX_UNION_BYTES = MX_STAGES * MX_STAGE_BYTES;    // 67584
constexpr uint32_t UNION_BYTES = NV_UNION_BYTES > MX_UNION_BYTES ? NV_UNION_BYTES : MX_UNION_BYTES;
constexpr uint32_t BAR_OFF = UNION_OFF + UNION_BYTES;              // 0x18800
constexpr uint32_t BAR_BYTES = 128;
constexpr uint32_t SMEM_BYTES = BAR_OFF + BAR_BYTES;               // 100480

static_assert(NV_STAGE_BYTES % 1024 == 0 && MX_STAGE_BYTES % 1024 == 0, "stage alignment");
static_assert(SMEM_BYTES <= 101376, "smem budget");

// ------------------------------------------------------------------------------------
// GEMM kernel parameters
// ------------------------------------------------------------------------------------
struct GemmParams {
    const uint8_t* sfa_lo;   // blocked copies (D0.3)
    const uint8_t* sfb_lo;
    const uint8_t* sfa_hi;
    const uint8_t* sfb_hi;
    const __half* bias;
    __half* y;
    int m, n;
    int kt_lo, kt_hi;        // k-tiles per segment
    int nq_lo, nq_hi;        // SF atoms per 128-row panel
    int tiles_m, tiles_n;
    int swz;                 // tile-walk swizzle s in {1,2,4}
    int along_m;             // raster: 1 = AlongM (m fast), 0 = AlongN
    float alpha;             // epilogue scale (two-level weight global g_w)
};

// Linear tile id -> (m_tile, n_tile) (D0.4).
__device__ __forceinline__ void tile_coords(int t, const GemmParams& p, int& mt, int& nt) {
    const int s = p.swz;
    const int o = t % s, e = t / s;
    const int tmaj = p.along_m ? p.tiles_m : p.tiles_n;
    const int major = e % tmaj;
    const int minor = (e / tmaj) * s + o;
    if (p.along_m) { mt = major; nt = minor; } else { nt = major; mt = minor; }
}

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Build-probe mount level (construction order of the milestone, §2): 2 = mbarrier rings
// only, 3 = + TMA loads, 4 = + MMA path and staging write, >= 5 = + bulk store (full kernel).
#ifndef FK_STAGE
#define FK_STAGE 9
#endif
#ifndef FK_BIAS_LATE
#define FK_BIAS_LATE 0
#endif
#ifndef FK_SINGLE_B
#define FK_SINGLE_B 0
#endif

// ------------------------------------------------------------------------------------
// mbarrier / named barrier / TMA primitives (D0.7, D0.10)
// ------------------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    uint32_t done;
    do {
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     " selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(bar), "r"(parity) : "memory");
    } while (!done);
}
__device__ __forceinline__ void named_sync(uint32_t id, uint32_t count) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void named_arrive(uint32_t id, uint32_t count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(reinterpret_cast<uint64_t>(src)), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, uint32_t src, int c0, int c1) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
                 :: "l"(reinterpret_cast<uint64_t>(map)), "r"(src), "r"(c0), "r"(c1) : "memory");
}

// Named barrier ids (0 is __syncthreads).
constexpr uint32_t BAR_NV_TO_MX = 1;      // 288: MMA threads arrive, producer warp syncs (D0.7)
constexpr uint32_t BAR_MX_TO_NV = 2;      // 288
constexpr uint32_t BAR_STAGE_FREE = 3;    // 256 MMA threads (D0.10)
constexpr uint32_t BAR_STAGE_FULL = 4;    // 256
constexpr uint32_t HANDOFF_THREADS = 256 + 32;

// ------------------------------------------------------------------------------------
// Operand fragments and block-scaled MMA atoms (D0.8)
// One k-step consumes 32 bytes of every operand row in both segments: NV K64 (E2M1),
// MX K32 (E4M3). Warp tile 32x64 = 2 m16 blocks x 8 n8 blocks.
// ------------------------------------------------------------------------------------
struct Frag {               // A operand + scales of one k-step
    uint32_t a[2][4];
    uint32_t sfa[2];
    uint32_t sfb[8];
};
struct BFrag {              // B operand of one k-step
    uint32_t b[8][2];
};

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
}

// Per-thread addressing of a stage, fixed for the whole kernel. A row offset already carries
// the lane's 16-byte half and the row's swizzle XOR; k-step ks then flips chunk bits by
// XOR-ing ks << 5 into the (1024-aligned) stage address, since (2ks + hi) ^ x == (ks << 1) ^ (hi ^ x).
struct SegAddr {
    uint32_t a[2];          // per m16 block
    uint32_t b[4];          // per n-block pair
};
struct SfAddr {             // same in both segments: byte offset of the scale word in a 512-B atom
    uint32_t a[2];
    uint32_t b[4];          // n-blocks j and j+4 share one 8-byte pair
};

template <bool MX>
__device__ __forceinline__ void make_addr(SegAddr& sa, int lane, int wm, int wn) {
    const int mi = lane >> 3, rr = lane & 7;
    const uint32_t row_bytes = MX ? 128 : 64;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int row = 32 * wm + 16 * i + rr + 8 * (mi & 1);
        const uint32_t x = MX ? (row & 7) : ((row >> 1) & 3);
        sa.a[i] = row * row_bytes + (((mi >> 1) ^ x) << 4);
    }
#pragma unroll
    for (int jp = 0; jp < 4; ++jp) {
        const int row = 64 * wn + 16 * jp + 8 * (mi >> 1) + rr;
        const uint32_t x = MX ? (row & 7) : ((row >> 1) & 3);
        sa.b[jp] = row * row_bytes + (((mi & 1) ^ x) << 4);
    }
}

__device__ __forceinline__ void make_sf_addr(SfAddr& sf, int lane, int wm, int wn) {
    const int g = lane >> 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) sf.a[i] = (16 * i + g + 8 * (lane & 1)) * 16 + wm * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) sf.b[j] = (8 * j + g) * 16 + (2 * wn) * 4;
}

// Loads the fragments of k-step ks from one stage. NV: scale word = 4 UE4M3 blocks of the
// k-step's K64 (atom ks of the stage); MX: one UE8M0 byte (byte ks of the row's word).
__device__ __forceinline__ void load_bfrag(BFrag& f, const SegAddr& sa, uint32_t b_base, int ks) {
    const uint32_t kx = (uint32_t)ks << 5;
#pragma unroll
    for (int jp = 0; jp < 4; ++jp)
        ldsm_x4((b_base + sa.b[jp]) ^ kx, f.b[2 * jp][0], f.b[2 * jp][1], f.b[2 * jp + 1][0],
                f.b[2 * jp + 1][1]);
}

template <bool MX>
__device__ __forceinline__ void load_frag(Frag& f, const SegAddr& sa, const SfAddr& sf, uint32_t a_base,
                                          uint32_t sfa_base, uint32_t sfb_base, int ks) {
    const uint32_t kx = (uint32_t)ks << 5;
#pragma unroll
    for (int i = 0; i < 2; ++i)
        ldsm_x4((a_base + sa.a[i]) ^ kx, f.a[i][0], f.a[i][1], f.a[i][2], f.a[i][3]);
    if (MX) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
            asm volatile("ld.shared.u8 %0, [%1];" : "=r"(f.sfa[i]) : "r"(sfa_base + sf.a[i] + ks) : "memory");
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile("ld.shared.u8 %0, [%1];" : "=r"(f.sfb[j]) : "r"(sfb_base + sf.b[j] + ks) : "memory");
            asm volatile("ld.shared.u8 %0, [%1];" : "=r"(f.sfb[j + 4]) : "r"(sfb_base + sf.b[j] + 4 + ks) : "memory");
        }
    } else {
        const uint32_t atom = ks * 512;
#pragma unroll
        for (int i = 0; i < 2; ++i)
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(f.sfa[i]) : "r"(sfa_base + atom + sf.a[i]) : "memory");
#pragma unroll
        for (int j = 0; j < 4; ++j)
            asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];" : "=r"(f.sfb[j]), "=r"(f.sfb[j + 4])
                         : "r"(sfb_base + atom + sf.b[j]) : "memory");
    }
}

template <bool MX>
__device__ __forceinline__ void mma_frag(float (&acc)[2][8][4], const Frag& f, const BFrag& g) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float* c = acc[i][j];
#if FK_BLOCK_SCALE_MMA
            if (MX) {
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};"
                    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                    : "r"(f.a[i][0]), "r"(f.a[i][1]), "r"(f.a[i][2]), "r"(f.a[i][3]), "r"(g.b[j][0]), "r"(g.b[j][1]),
                      "r"(f.sfa[i]), "h"((uint16_t)0), "h"((uint16_t)0), "r"(f.sfb[j]), "h"((uint16_t)0), "h"((uint16_t)0));
            } else {
                asm volatile(
                    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};"
                    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                    : "r"(f.a[i][0]), "r"(f.a[i][1]), "r"(f.a[i][2]), "r"(f.a[i][3]), "r"(g.b[j][0]), "r"(g.b[j][1]),
                      "r"(f.sfa[i]), "h"((uint16_t)0), "h"((uint16_t)0), "r"(f.sfb[j]), "h"((uint16_t)0), "h"((uint16_t)0));
            }
#else
            __trap();
#endif
        }
    }
}

// ------------------------------------------------------------------------------------
// Persistent GEMM kernel
// ------------------------------------------------------------------------------------
__global__ void __launch_bounds__(NUM_THREADS, 1)
promix_gemm_kernel(const __grid_constant__ CUtensorMap tm_a_lo,
                   const __grid_constant__ CUtensorMap tm_b_lo,
                   const __grid_constant__ CUtensorMap tm_a_hi,
                   const __grid_constant__ CUtensorMap tm_b_hi,
                   const __grid_constant__ CUtensorMap tm_y,
                   const GemmParams p) {
    extern __shared__ __align__(1024) uint8_t smem[];
    const uint32_t sbase = smem_u32(smem);
    // 128B-swizzled TMA boxes need a 1024-aligned base (D0.6).
    if (threadIdx.x == 0 && (sbase & 1023u) != 0) __trap();

    const uint32_t staging = sbase + STAGING_OFF;
    const uint32_t uni = sbase + UNION_OFF;
    const uint32_t bars = sbase + BAR_OFF;
    auto nv_full = [&](int s) { return bars + 8u * s; };
    auto nv_empty = [&](int s) { return bars + 8u * (NV_STAGES + s); };
    auto mx_full = [&](int s) { return bars + 8u * (2 * NV_STAGES + s); };
    auto mx_empty = [&](int s) { return bars + 8u * (2 * NV_STAGES + MX_STAGES + s); };

    if (threadIdx.x == 0) {
        for (int s = 0; s < NV_STAGES; ++s) { mbar_init(nv_full(s), 1); mbar_init(nv_empty(s), 8); }
        for (int s = 0; s < MX_STAGES; ++s) { mbar_init(mx_full(s), 1); mbar_init(mx_empty(s), 8); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    const int num_tiles = p.tiles_m * p.tiles_n;
    const int lane = threadIdx.x & 31;

    if (threadIdx.x < 32) {
        // Producer warp of WG0; lane 0 issues every load (D0.5, D0.7).
        int ns = 0, xs = 0;
        uint32_t nph = 0, xph = 0;
        bool first = true;
        for (int t = blockIdx.x; t < num_tiles; t += gridDim.x) {
            int mt, nt;
            tile_coords(t, p, mt, nt);
            if (!first) named_sync(BAR_MX_TO_NV, HANDOFF_THREADS);   // previous tile's MX view released
            first = false;
            const uint8_t* sfa_lo = p.sfa_lo + (size_t)mt * p.nq_lo * 512;
            const uint8_t* sfb_lo = p.sfb_lo + (size_t)nt * p.nq_lo * 512;
            for (int kt = 0; kt < p.kt_lo; ++kt) {
                mbar_wait(nv_empty(ns), nph ^ 1);
                if (lane == 0) {
                    const uint32_t st = uni + ns * NV_STAGE_BYTES;
                    const uint32_t fb = nv_full(ns);
#if FK_STAGE >= 3
                    mbar_expect_tx(fb, NV_STAGE_BYTES);
                    tma_load_2d(st, &tm_a_lo, kt * (BK / 2), mt * BM, fb);
                    tma_load_2d(st + NV_A_BYTES, &tm_b_lo, kt * (BK / 2), nt * BN, fb);
                    bulk_load(st + NV_A_BYTES + NV_B_BYTES, sfa_lo + kt * 2 * 512, NV_SF_BYTES, fb);
                    bulk_load(st + NV_A_BYTES + NV_B_BYTES + NV_SF_BYTES, sfb_lo + kt * 2 * 512, NV_SF_BYTES, fb);
#else
                    (void)st; mbar_arrive(fb);
#endif
                }
                __syncwarp();
                if (++ns == NV_STAGES) { ns = 0; nph ^= 1; }
            }
            named_sync(BAR_NV_TO_MX, HANDOFF_THREADS);               // NV view released
            const uint8_t* sfa_hi = p.sfa_hi + (size_t)mt * p.nq_hi * 512;
            const uint8_t* sfb_hi = p.sfb_hi + (size_t)nt * p.nq_hi * 512;
            for (int kt = 0; kt < p.kt_hi; ++kt) {
                mbar_wait(mx_empty(xs), xph ^ 1);
                if (lane == 0) {
                    const uint32_t st = uni + xs * MX_STAGE_BYTES;
                    const uint32_t fb = mx_full(xs);
#if FK_STAGE >= 3
                    mbar_expect_tx(fb, MX_STAGE_BYTES);
                    tma_load_2d(st, &tm_a_hi, kt * BK, mt * BM, fb);
                    tma_load_2d(st + MX_A_BYTES, &tm_b_hi, kt * BK, nt * BN, fb);
                    bulk_load(st + MX_A_BYTES + MX_B_BYTES, sfa_hi + kt * 512, MX_SF_BYTES, fb);
                    bulk_load(st + MX_A_BYTES + MX_B_BYTES + MX_SF_BYTES, sfb_hi + kt * 512, MX_SF_BYTES, fb);
#else
                    (void)st; mbar_arrive(fb);
#endif
                }
                __syncwarp();
                if (++xs == MX_STAGES) { xs = 0; xph ^= 1; }
            }
        }
        if (!first) named_sync(BAR_MX_TO_NV, HANDOFF_THREADS);       // last tile's arrivals
    } else if (threadIdx.x >= 128) {
        // MMA warpgroups 1+2: 256 cooperative threads share one 128x128 tile, warps 4(M) x 2(N),
        // warp tile 32x64, 64 FP32 accumulators per thread (D0.5).
        const int ct = threadIdx.x - 128;
        const int cw = ct >> 5;
        const int wm = cw & 3, wn = cw >> 2;
        const int q = lane & 3;
        SegAddr fn, fx;
        SfAddr fs;
        make_addr<false>(fn, lane, wm, wn);
        make_addr<true>(fx, lane, wm, wn);
        make_sf_addr(fs, lane, wm, wn);
        int ns = 0, xs = 0;
        uint32_t nph = 0, xph = 0;
        const __half2* bias2 = reinterpret_cast<const __half2*>(p.bias);

        auto nv_a = [&](int s) { return uni + s * NV_STAGE_BYTES; };
        auto mx_a = [&](int s) { return uni + s * MX_STAGE_BYTES; };
#define NV_ARGS(s) nv_a(s), nv_a(s) + NV_A_BYTES + NV_B_BYTES, nv_a(s) + NV_A_BYTES + NV_B_BYTES + NV_SF_BYTES
#define MX_ARGS(s) mx_a(s), mx_a(s) + MX_A_BYTES + MX_B_BYTES, mx_a(s) + MX_A_BYTES + MX_B_BYTES + MX_SF_BYTES
#if FK_STAGE >= 4
#define LDA_NV(f, s, ks) load_frag<false>(f, fn, fs, NV_ARGS(s), ks)
#define LDA_MX(f, s, ks) load_frag<true>(f, fx, fs, MX_ARGS(s), ks)
#define LDB_NV(g, s, ks) load_bfrag(g, fn, nv_a(s) + NV_A_BYTES, ks)
#define LDB_MX(g, s, ks) load_bfrag(g, fx, mx_a(s) + MX_A_BYTES, ks)
#define MMA_NV(f, g) mma_frag<false>(acc, f, g)
#define MMA_MX(f, g) mma_frag<true>(acc, f, g)
#else
#define LDA_NV(f, s, ks) ((void)0)
#define LDA_MX(f, s, ks) ((void)0)
#define LDB_NV(g, s, ks) ((void)0)
#define LDB_MX(g, s, ks) ((void)0)
#define MMA_NV(f, g) ((void)0)
#define MMA_MX(f, g) ((void)0)
#endif
        auto release = [&](uint32_t bar) {
            __syncwarp();
            if (lane == 0) mbar_arrive(bar);
        };

        for (int t = blockIdx.x; t < num_tiles; t += gridDim.x) {
            int mt, nt;
            tile_coords(t, p, mt, nt);
            // Bias of this thread's 16 output columns, in flight during the mainloop (D0.6);
            // FK_BIAS_LATE (D0.9 yield rung r1) loads it at epilogue entry instead.
            __half2 bz[8];
#define LOAD_BIAS()                                                                          \
    _Pragma("unroll") for (int j = 0; j < 8; ++j)                                            \
        bz[j] = __ldg(bias2 + ((nt * BN + 64 * wn + 8 * j + 2 * q) >> 1))
#if !FK_BIAS_LATE
            LOAD_BIAS();
#endif

            float acc[2][8][4];
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

            // Mainloop (D0.9): fragments of k-step j+1 are loaded before the MMAs of k-step j,
            // across k-tile and segment boundaries; a stage is released right after its last
            // fragment load.
            Frag f0, f1;
#if !FK_SINGLE_B
            BFrag g0, g1;
            mbar_wait(nv_full(ns), nph);
            LDA_NV(f0, ns, 0); LDB_NV(g0, ns, 0);
            for (int kt = 0; kt < p.kt_lo; ++kt) {
                LDA_NV(f1, ns, 1); LDB_NV(g1, ns, 1);
                release(nv_empty(ns));
                if (kt == p.kt_lo - 1) named_arrive(BAR_NV_TO_MX, HANDOFF_THREADS);
                MMA_NV(f0, g0);
                if (++ns == NV_STAGES) { ns = 0; nph ^= 1; }
                if (kt + 1 < p.kt_lo) {
                    mbar_wait(nv_full(ns), nph);
                    LDA_NV(f0, ns, 0); LDB_NV(g0, ns, 0);
                } else {
                    mbar_wait(mx_full(xs), xph);
                    LDA_MX(f0, xs, 0); LDB_MX(g0, xs, 0);
                }
                MMA_NV(f1, g1);
            }
            for (int kt = 0; kt < p.kt_hi; ++kt) {
                LDA_MX(f1, xs, 1); LDB_MX(g1, xs, 1);
                MMA_MX(f0, g0);
                LDA_MX(f0, xs, 2); LDB_MX(g0, xs, 2);
                MMA_MX(f1, g1);
                LDA_MX(f1, xs, 3); LDB_MX(g1, xs, 3);
                release(mx_empty(xs));
                if (kt == p.kt_hi - 1) named_arrive(BAR_MX_TO_NV, HANDOFF_THREADS);
                MMA_MX(f0, g0);
                if (++xs == MX_STAGES) { xs = 0; xph ^= 1; }
                if (kt + 1 < p.kt_hi) {
                    mbar_wait(mx_full(xs), xph);
                    LDA_MX(f0, xs, 0); LDB_MX(g0, xs, 0);
                }
                MMA_MX(f1, g1);
            }
#else
            // D0.9 yield rung r2: one B fragment; B of k-step j+1 is loaded right after the
            // MMAs of k-step j, A + scales stay double-buffered. A stage is released after
            // its last B load.
            BFrag g;
            mbar_wait(nv_full(ns), nph);
            LDA_NV(f0, ns, 0); LDB_NV(g, ns, 0);
            for (int kt = 0; kt < p.kt_lo; ++kt) {
                LDA_NV(f1, ns, 1);
                MMA_NV(f0, g);
                LDB_NV(g, ns, 1);
                release(nv_empty(ns));
                if (kt == p.kt_lo - 1) named_arrive(BAR_NV_TO_MX, HANDOFF_THREADS);
                if (++ns == NV_STAGES) { ns = 0; nph ^= 1; }
                if (kt + 1 < p.kt_lo) {
                    mbar_wait(nv_full(ns), nph);
                    LDA_NV(f0, ns, 0);
                    MMA_NV(f1, g);
                    LDB_NV(g, ns, 0);
                } else {
                    mbar_wait(mx_full(xs), xph);
                    LDA_MX(f0, xs, 0);
                    MMA_NV(f1, g);
                    LDB_MX(g, xs, 0);
                }
            }
            for (int kt = 0; kt < p.kt_hi; ++kt) {
                LDA_MX(f1, xs, 1);
                MMA_MX(f0, g);
                LDB_MX(g, xs, 1);
                LDA_MX(f0, xs, 2);
                MMA_MX(f1, g);
                LDB_MX(g, xs, 2);
                LDA_MX(f1, xs, 3);
                MMA_MX(f0, g);
                LDB_MX(g, xs, 3);
                release(mx_empty(xs));
                if (kt == p.kt_hi - 1) named_arrive(BAR_MX_TO_NV, HANDOFF_THREADS);
                if (++xs == MX_STAGES) { xs = 0; xph ^= 1; }
                if (kt + 1 < p.kt_hi) {
                    mbar_wait(mx_full(xs), xph);
                    LDA_MX(f0, xs, 0);
                    MMA_MX(f1, g);
                    LDB_MX(g, xs, 0);
                } else {
                    MMA_MX(f1, g);
                }
            }
#endif

#if FK_BIAS_LATE
            LOAD_BIAS();
#endif
#undef LOAD_BIAS
            // Epilogue (D0.10): whole-tile staging, one bulk store group in flight across the
            // next mainloop.
#if FK_STAGE >= 5
            if (ct == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
#endif
            named_sync(BAR_STAGE_FREE, 256);
#if FK_STAGE >= 4
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int rr = 16 * i + 8 * ((lane >> 3) & 1) + (lane & 7);   // row inside the 32-row box
                const uint32_t row_addr = staging + (wm * 2 + wn) * 4096 + rr * 128;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float2 b = __half22float2(bz[j]);
                    __half2 h0 = __floats2half2_rn(fmaf(acc[i][j][0], p.alpha, b.x),
                                                   fmaf(acc[i][j][1], p.alpha, b.y));
                    __half2 h1 = __floats2half2_rn(fmaf(acc[i][j][2], p.alpha, b.x),
                                                   fmaf(acc[i][j][3], p.alpha, b.y));
                    const uint32_t addr = row_addr + ((j ^ (rr & 7)) << 4);
                    asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};"
                                 :: "r"(addr), "r"(*reinterpret_cast<uint32_t*>(&h0)),
                                    "r"(*reinterpret_cast<uint32_t*>(&h1)) : "memory");
                }
            }
#else
            (void)bz; (void)acc; (void)staging;
#endif
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            named_sync(BAR_STAGE_FULL, 256);
#if FK_STAGE >= 5
            if (ct == 0) {
#pragma unroll
                for (int b = 0; b < 8; ++b)
                    tma_store_2d(&tm_y, staging + b * 4096, nt * BN + (b & 1) * 64, mt * BM + (b >> 1) * 32);
                asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            }
#endif
        }
#if FK_STAGE >= 5
        if (ct == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
#endif
#undef NV_ARGS
#undef MX_ARGS
#undef LDA_NV
#undef LDA_MX
#undef LDB_NV
#undef LDB_MX
#undef MMA_NV
#undef MMA_MX
    }
}

// ------------------------------------------------------------------------------------
// SF prep kernel (D0.3): blocked copy, 512-B atom per (panel p = r/128, q = b/4) at
// (p*NQ + q)*512; inside the atom byte (r%32)*16 + ((r/32)%4)*4 + (b%4); pad = 0x00.
// ------------------------------------------------------------------------------------
struct PrepJob {
    const uint8_t* src;
    uint8_t* dst;
    int rows, bpr;           // source rows, scale blocks per row (row stride == bpr)
    int nq;                  // atoms per panel
    int out_bytes;
};
struct PrepParams {
    PrepJob job[4];
    int njobs;
};

__global__ void sf_prep_kernel(const PrepParams pp) {
    if ((int)blockIdx.y >= pp.njobs) return;
    const PrepJob& j = pp.job[blockIdx.y];
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < j.out_bytes; o += gridDim.x * blockDim.x) {
        const int atom = o >> 9, in = o & 511;
        const int panel = atom / j.nq, q = atom - panel * j.nq;
        const int r = panel * 128 + ((in >> 2) & 3) * 32 + (in >> 4);
        const int b = q * 4 + (in & 3);
        uint8_t v = 0;
        if (r < j.rows && b < j.bpr) v = j.src[(size_t)r * j.bpr + b];
        j.dst[o] = v;
    }
}

// ------------------------------------------------------------------------------------
// Host side
// ------------------------------------------------------------------------------------
namespace {

// Cache of one blocked SF copy (D0.3). Key: storage identity (weak ref, so the identity
// cannot be reused while cached) + storage offset + sizes + strides + version counter.
struct SfCache {
    c10::weak_intrusive_ptr<c10::StorageImpl> storage{c10::intrusive_ptr<c10::StorageImpl>()};
    int64_t offset = -1;
    std::vector<int64_t> sizes, strides;
    int64_t version = -1;
    int device = -1;
    torch::Tensor blocked;

    bool hit(const torch::Tensor& t) const {
        if (!blocked.defined() || storage.expired()) return false;
        return storage._unsafe_get_target() == t.storage().unsafeGetStorageImpl()
            && offset == t.storage_offset()
            && sizes == t.sizes().vec() && strides == t.strides().vec()
            && version == static_cast<int64_t>(t._version())
            && device == t.get_device();
    }
    void store(const torch::Tensor& t, torch::Tensor b) {
        storage = c10::weak_intrusive_ptr<c10::StorageImpl>(
            c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(t.storage().unsafeGetStorageImpl()));
        offset = t.storage_offset();
        sizes = t.sizes().vec();
        strides = t.strides().vec();
        version = static_cast<int64_t>(t._version());
        device = t.get_device();
        blocked = std::move(b);
    }
};

SfCache* const g_sf_cache = new SfCache[4];   // never freed: outlives the CUDA context at exit
int64_t g_prep_launches = 0;
int g_swz = 1, g_along_m = 1;     // M0 starting order sw1/AlongM (D0.4)

inline int nq_of(int bpr) { return (bpr + 3) / 4; }

// Fills one prep job for SF tensor t; returns the (uninitialised) blocked destination.
torch::Tensor make_prep_job(const torch::Tensor& t, PrepJob& j, torch::Tensor& src_keep) {
    torch::Tensor src = t.view(torch::kUInt8).contiguous();
    src_keep = src;
    const int rows = (int)src.size(0), bpr = (int)src.size(1);
    const int nq = nq_of(bpr);
    const int panels = (rows + 127) / 128;
    const int out_bytes = panels * nq * 512;
    torch::Tensor dst = torch::empty({out_bytes}, src.options());
    j.src = src.data_ptr<uint8_t>();
    j.dst = dst.data_ptr<uint8_t>();
    j.rows = rows; j.bpr = bpr; j.nq = nq; j.out_bytes = out_bytes;
    return dst;
}

void launch_prep(const PrepParams& pp, int max_bytes, cudaStream_t stream) {
    const int threads = 256;
    int bx = (max_bytes + threads - 1) / threads;
    if (bx > 4096) bx = 4096;
    sf_prep_kernel<<<dim3(bx, pp.njobs), threads, 0, stream>>>(pp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    ++g_prep_launches;
}

// Returns the 4 blocked SF tensors, re-deriving only those whose key missed.
std::array<torch::Tensor, 4> blocked_sfs(const std::array<torch::Tensor, 4>& sf, cudaStream_t stream) {
    PrepParams pp{};
    pp.njobs = 0;
    std::array<torch::Tensor, 4> src_keep;
    int max_bytes = 0;
    for (int i = 0; i < 4; ++i) {
        if (g_sf_cache[i].hit(sf[i])) continue;
        PrepJob& j = pp.job[pp.njobs++];
        torch::Tensor dst = make_prep_job(sf[i], j, src_keep[i]);
        if (j.out_bytes > max_bytes) max_bytes = j.out_bytes;
        g_sf_cache[i].store(sf[i], dst);
    }
    if (pp.njobs > 0) launch_prep(pp, max_bytes, stream);
    return {g_sf_cache[0].blocked, g_sf_cache[1].blocked, g_sf_cache[2].blocked, g_sf_cache[3].blocked};
}

// 2D tensor map over a row-major [rows, cols] tensor; box [box_rows, box_cols] elements.
CUtensorMap make_map_2d(const void* base, CUtensorMapDataType dt, int elem_bytes,
                        uint64_t rows, uint64_t cols, uint32_t box_rows, uint32_t box_cols,
                        CUtensorMapSwizzle swz) {
    CUtensorMap map;
    cuuint64_t dims[2] = {cols, rows};
    cuuint64_t strides[1] = {cols * (uint64_t)elem_bytes};
    cuuint32_t box[2] = {box_cols, box_rows};
    cuuint32_t estr[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(&map, dt, 2, const_cast<void*>(base), dims, strides, box, estr,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return map;
}

bool g_attr_set = false;

}  // namespace

static torch::Tensor launch_fk_gemm(const torch::Tensor& A_lo, const torch::Tensor& B_lo,
                                    const torch::Tensor& A_hi, const torch::Tensor& B_hi,
                                    const uint8_t* p_sfa_lo, const uint8_t* p_sfb_lo,
                                    const uint8_t* p_sfa_hi, const uint8_t* p_sfb_hi,
                                    const torch::Tensor& Bias, double alpha, cudaStream_t stream) {
    const int m = (int)A_lo.size(0), n = (int)B_lo.size(0);
    const int k_lo = (int)A_lo.size(1) * 2, k_hi = (int)A_hi.size(1);
    TORCH_CHECK(k_lo % 128 == 0 && k_hi % 32 == 0, "k_lo % 128 / k_hi % 32");
    TORCH_CHECK(m % BM == 0 && n % BN == 0, "m, n must be multiples of 128");
    TORCH_CHECK(k_hi % BK == 0, "k_hi must be a multiple of 128 in this build");
    TORCH_CHECK(k_lo >= BK && k_hi >= BK, "both K segments must be non-empty in this build");

    torch::Tensor y = torch::empty({m, n}, Bias.options().dtype(torch::kFloat16));

    const CUtensorMap tm_a_lo = make_map_2d(A_lo.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                                            m, k_lo / 2, BM, BK / 2, CU_TENSOR_MAP_SWIZZLE_64B);
    const CUtensorMap tm_b_lo = make_map_2d(B_lo.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                                            n, k_lo / 2, BN, BK / 2, CU_TENSOR_MAP_SWIZZLE_64B);
    const CUtensorMap tm_a_hi = make_map_2d(A_hi.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                                            m, k_hi, BM, BK, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tm_b_hi = make_map_2d(B_hi.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                                            n, k_hi, BN, BK, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tm_y = make_map_2d(y.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
                                         m, n, 32, 64, CU_TENSOR_MAP_SWIZZLE_128B);

    GemmParams p;
    p.sfa_lo = p_sfa_lo;
    p.sfb_lo = p_sfb_lo;
    p.sfa_hi = p_sfa_hi;
    p.sfb_hi = p_sfb_hi;
    p.alpha = (float)alpha;
    p.bias = reinterpret_cast<const __half*>(Bias.data_ptr<at::Half>());
    p.y = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
    p.m = m; p.n = n;
    p.kt_lo = k_lo / BK; p.kt_hi = k_hi / BK;
    p.nq_lo = nq_of(k_lo / 16); p.nq_hi = nq_of(k_hi / 32);
    p.tiles_m = m / BM; p.tiles_n = n / BN;
    p.swz = g_swz; p.along_m = g_along_m;
    const int minor_tiles = p.along_m ? p.tiles_n : p.tiles_m;
    TORCH_CHECK(minor_tiles % p.swz == 0, "tile walk: minor tile count % swizzle");

    if (!g_attr_set) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(promix_gemm_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
        g_attr_set = true;
    }
    promix_gemm_kernel<<<NUM_CTAS, NUM_THREADS, SMEM_BYTES, stream>>>(tm_a_lo, tm_b_lo, tm_a_hi, tm_b_hi,
                                                                     tm_y, p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor run(torch::Tensor a_lo, torch::Tensor sfa_lo, torch::Tensor a_hi, torch::Tensor sfa_hi,
                  torch::Tensor b_lo, torch::Tensor sfb_lo, torch::Tensor b_hi, torch::Tensor sfb_hi,
                  torch::Tensor bias, double alpha = 1.0) {
    c10::cuda::CUDAGuard guard(a_lo.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // D0.2(a): byte tensors are reinterpreted as uint8, never value-converted.
    torch::Tensor A_lo = a_lo.view(torch::kUInt8).contiguous();
    torch::Tensor B_lo = b_lo.view(torch::kUInt8).contiguous();
    torch::Tensor A_hi = a_hi.view(torch::kUInt8).contiguous();
    torch::Tensor B_hi = b_hi.view(torch::kUInt8).contiguous();
    torch::Tensor Bias = bias.contiguous();
    auto sfs = blocked_sfs({sfa_lo, sfb_lo, sfa_hi, sfb_hi}, stream);
    return launch_fk_gemm(A_lo, B_lo, A_hi, B_hi,
                          sfs[0].data_ptr<uint8_t>(), sfs[1].data_ptr<uint8_t>(),
                          sfs[2].data_ptr<uint8_t>(), sfs[3].data_ptr<uint8_t>(),
                          Bias, alpha, stream);
}

// KDA round 10 merge entry: the pipeline's already-interleaved physical SF buffers
// (prepack_sf_fp4/8_* layout == the evolved kernel's blocked layout, torch.equal-verified)
// are consumed directly -- no prep kernel, no cache: ONE launch per call, capture-safe.
torch::Tensor run_prepacked(torch::Tensor a_lo, torch::Tensor psfa_lo,
                            torch::Tensor b_lo, torch::Tensor psfb_lo,
                            torch::Tensor a_hi, torch::Tensor psfa_hi,
                            torch::Tensor b_hi, torch::Tensor psfb_hi,
                            torch::Tensor bias, double alpha = 1.0) {
    c10::cuda::CUDAGuard guard(a_lo.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor A_lo = a_lo.view(torch::kUInt8).contiguous();
    torch::Tensor B_lo = b_lo.view(torch::kUInt8).contiguous();
    torch::Tensor A_hi = a_hi.view(torch::kUInt8).contiguous();
    torch::Tensor B_hi = b_hi.view(torch::kUInt8).contiguous();
    torch::Tensor pal = psfa_lo.view(torch::kUInt8).contiguous();
    torch::Tensor pbl = psfb_lo.view(torch::kUInt8).contiguous();
    torch::Tensor pah = psfa_hi.view(torch::kUInt8).contiguous();
    torch::Tensor pbh = psfb_hi.view(torch::kUInt8).contiguous();
    torch::Tensor Bias = bias.contiguous();
    return launch_fk_gemm(A_lo, B_lo, A_hi, B_hi,
                          pal.data_ptr<uint8_t>(), pbl.data_ptr<uint8_t>(),
                          pah.data_ptr<uint8_t>(), pbh.data_ptr<uint8_t>(),
                          Bias, alpha, stream);
}

// Diagnostics used by the build probes (not part of the timed path).
torch::Tensor debug_blocked_sf(torch::Tensor sf) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    PrepParams pp{};
    torch::Tensor keep;
    torch::Tensor dst = make_prep_job(sf, pp.job[0], keep);
    pp.njobs = 1;
    launch_prep(pp, pp.job[0].out_bytes, stream);
    return dst;
}
int64_t debug_prep_launches() { return g_prep_launches; }
void debug_set_order(int64_t swz, int64_t along_m) { g_swz = (int)swz; g_along_m = (int)along_m; }
std::vector<int64_t> debug_gemm_attrs() {
    cudaFuncAttributes a;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, promix_gemm_kernel));
    int optin = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)SMEM_BYTES, optin, a.maxThreadsPerBlock};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, pybind11::arg("a_lo"), pybind11::arg("sfa_lo"), pybind11::arg("a_hi"),
          pybind11::arg("sfa_hi"), pybind11::arg("b_lo"), pybind11::arg("sfb_lo"),
          pybind11::arg("b_hi"), pybind11::arg("sfb_hi"), pybind11::arg("bias"),
          pybind11::arg("alpha") = 1.0);
    m.def("run_prepacked", &run_prepacked, pybind11::arg("a_lo"), pybind11::arg("psfa_lo"),
          pybind11::arg("b_lo"), pybind11::arg("psfb_lo"), pybind11::arg("a_hi"),
          pybind11::arg("psfa_hi"), pybind11::arg("b_hi"), pybind11::arg("psfb_hi"),
          pybind11::arg("bias"), pybind11::arg("alpha") = 1.0);
    m.def("debug_blocked_sf", &debug_blocked_sf);
    m.def("debug_prep_launches", &debug_prep_launches);
    m.def("debug_set_order", &debug_set_order);
    m.def("debug_gemm_attrs", &debug_gemm_attrs);
}
