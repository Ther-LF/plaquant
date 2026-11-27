// fk_spg_ext.cu — KDA-4 r17: ForgeKernel-evolved SINGLE-phase NVFP4 block-scaled
// GEMM for the GB10 r=0 (pure-NVFP4) column. Evolved by the promix_spg_nvfp4
// ForgeKernel run (gen 1, best 22efdd3883, 2026-11-27) against the shipping
// castfold single-phase CUTLASS kernel; confirmed e2e 1.13x/1.07x/1.19x on the
// gu1b/gu3b/qk3b shapes. Same macro-structure as fk_smallk_ext (persistent
// 48-CTA, producer WG + 8 consumer warps, whole-tile smem output staging with
// ONE bulk TMA store group left in flight across tile boundaries) but a
// from-scratch single-phase mainloop: 4-stage NVFP4 ring, no phase switch, no
// dual-kernel rendezvous.
//
// Merge additions (this file vs the evolved best/kernel.cu):
//   * run_prepacked(): consumes the pipeline's PHYSICAL 32_4_4 SF layout
//     (prepack_sf_* / qpf outputs) directly — the evolved kernel's per-k-tile
//     1024B SF block is the same 1024 bytes, only the within-block byte order
//     differs, so the consumer smem index changes from (row*8 + ks*4) to
//     (ks*512 + (row%32)*16 + (row/32)*4)  [mapping verified byte-exact in
//     /tmp/kda_sfmap_probe.py]. Zero extra launches; steady state is ONE kernel.
//   * epilogue alpha (the per-tensor weight global g_w): y = fp16(alpha*acc + bias).
//   * per-call tile-walk override (swz_w, serpentine), like fk_smallk r13.
// Constraints: M%128==0, N%128==0, K%128==0 (K = K_low_pad on GB10).
//
// Inputs:
//     a_lo: uint8 [m, k/2]      sfa: row-major [m, k/16] (run) or physSF (run_prepacked)
//     b_lo: uint8 [n, k/2]      sfb: ditto
//     bias: float16 [n]
// Output: y float16 [m, n]

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------------------
// Tiling constants (D0.3): CTA tile 128 (m) x 128 (n) x 128 (k elements = 64 B per row).
// ---------------------------------------------------------------------------------------
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 128;                 // fp4 elements per k-tile
constexpr int BK_BYTES = BK / 2;        // 64 B of packed codes per row per k-tile
constexpr int SF_COLS = BK / 16;        // 8 ue4m3 scale bytes per row per k-tile
constexpr int SF_BLOCK_BYTES = 128 * SF_COLS;  // 1024 B: one 128-row x 8-column scale block
constexpr int NUM_THREADS = 384;        // 3 warpgroups
constexpr int STAGES = 4;               // D0.5 starting pipeline depth (lever of D1.2)

// smem inventory (D0.5). Every block below is endorsed by a consumption-path ledger row.
constexpr int SM_A    = 0;                          // A ring    4 x 8192
constexpr int SM_B    = SM_A + STAGES * 8192;       // B ring    4 x 8192
constexpr int SM_SFA  = SM_B + STAGES * 8192;       // SFA ring  4 x 1024
constexpr int SM_SFB  = SM_SFA + STAGES * 1024;     // SFB ring  4 x 1024
constexpr int SM_Y    = SM_SFB + STAGES * 1024;     // y staging 2 x 4096
constexpr int SM_BIAS = SM_Y + 2 * 4096;            // bias      256
constexpr int SM_BAR  = SM_BIAS + 256;              // mbarriers 2 x STAGES x 8
constexpr int SMEM_BYTES = SM_BAR + 2 * STAGES * 8; // 82240 B <= 101376 B

// ---------------------------------------------------------------------------------------
// D0.2: block-scaled MMA atom m16n8k64, kind::mxf4nvf4, scale_vec::4X, e2m1 x e2m1, ue4m3
// scales, fp32 accumulate (inline PTX; lowers to OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X).
// Fragment mapping (lane = 4g + q), verified by notes/mma_probe.py:
//   a[0] = A[g][4q..4q+3 B]  a[1] = A[g+8][4q..]  a[2] = A[g][16+4q..]  a[3] = A[g+8][16+4q..]
//   b[0] = B[g][4q..]        b[1] = B[g][16+4q..]
//   sfa  = the 4 scale bytes of row g + 8*(q&1); sfb = the 4 scale bytes of row g
//   d[0..1] = C[g][2q..2q+1], d[2..3] = C[g+8][2q..2q+1]
// ---------------------------------------------------------------------------------------
__device__ __forceinline__ void mma_fp4(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
                                        uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// ---------------------------------------------------------------------------------------
// smem / mbarrier / bulk-copy helpers (shapes certified by notes/supply_probe.py).
// ---------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t sm_addr(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void bar_init(uint32_t bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}

__device__ __forceinline__ void bar_expect(uint32_t bar, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes));
}

__device__ __forceinline__ void bar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar));
}

__device__ __forceinline__ void bar_wait(uint32_t bar, int phase) {
    uint32_t ok;
    do {
        asm volatile(
            "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
            "  selp.b32 %0, 1, 0, p; }"
            : "=r"(ok)
            : "r"(bar), "r"(phase));
    } while (!ok);
}

// 2-D bulk tensor copy, global -> smem, completion signalled on an mbarrier.
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1,
                                       uint32_t bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
        "l"((uint64_t)map), "r"(bar), "r"(c0), "r"(c1)
        : "memory");
}

// 1-D bulk copy of a contiguous scale block, global -> smem.
__device__ __forceinline__ void bulk_1d(uint32_t dst, const uint8_t* src, int bytes, uint32_t bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];" ::"r"(dst),
        "l"(src), "r"(bytes), "r"(bar)
        : "memory");
}

__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// smem swizzle of a 64 B-pitch code tile: chunk (16 B) c of row r lives at chunk c ^ ((r>>1)&3).
// [certified: notes/supply_probe.py rung A, candidate 1, 0/1024 mismatched bytes]
__device__ __forceinline__ int swz64(int row, int chunk) { return chunk ^ ((row >> 1) & 3); }

// ---------------------------------------------------------------------------------------
// D0.1: scale-factor re-layout. Row-major [rows, kc] -> blocks [rows/128][kc/8][128][8]:
// the scales of one 128-row panel for one k-tile are one contiguous 1024 B block, so a
// k-tile's SFA / SFB slot is fed by a single bulk copy. Rows are zero-padded to a multiple
// of 128 and columns to a multiple of 8.
// ---------------------------------------------------------------------------------------
__global__ void sf_relayout_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                   int rows, int kc, int row_blocks, int k_tiles) {
    // one thread per (row block, k-tile, row): 8 output bytes
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)row_blocks * k_tiles * 128;
    if (idx >= total) return;
    int r = (int)(idx % 128);
    int kt = (int)((idx / 128) % k_tiles);
    int rb = (int)(idx / (128 * (int64_t)k_tiles));
    int row = rb * 128 + r;
    uint8_t v[SF_COLS];
#pragma unroll
    for (int c = 0; c < SF_COLS; ++c) {
        int col = kt * SF_COLS + c;
        v[c] = (row < rows && col < kc) ? src[(int64_t)row * kc + col] : (uint8_t)0;
    }
    uint8_t* o = dst + idx * SF_COLS;
#pragma unroll
    for (int c = 0; c < SF_COLS; ++c) o[c] = v[c];
}

struct SfCacheEntry {
    void* ptr;
    std::vector<int64_t> sizes;
    at::ScalarType dtype;
    at::Device device;
    torch::Tensor src;   // keeps the input storage alive so its address cannot be recycled
    torch::Tensor out;   // the re-laid-out copy
};

static std::vector<SfCacheEntry> g_sf_cache;

static torch::Tensor sf_relayout_cached(const torch::Tensor& sf, int k_tiles, cudaStream_t stream) {
    std::vector<int64_t> sizes(sf.sizes().begin(), sf.sizes().end());
    for (const auto& e : g_sf_cache)
        if (e.ptr == sf.data_ptr() && e.sizes == sizes && e.dtype == sf.scalar_type() && e.device == sf.device())
            return e.out;
    int rows = (int)sf.size(0), kc = (int)sf.size(1);
    int row_blocks = (rows + 127) / 128;
    auto out = torch::empty({(int64_t)row_blocks * k_tiles * SF_BLOCK_BYTES}, sf.options().dtype(torch::kUInt8));
    int64_t total = (int64_t)row_blocks * k_tiles * 128;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    sf_relayout_kernel<<<blocks, threads, 0, stream>>>((const uint8_t*)sf.data_ptr(), out.data_ptr<uint8_t>(),
                                                       rows, kc, row_blocks, k_tiles);
    if (g_sf_cache.size() >= 64) g_sf_cache.erase(g_sf_cache.begin());
    g_sf_cache.push_back({sf.data_ptr(), sizes, sf.scalar_type(), sf.device(), sf, out});
    return out;
}

// ---------------------------------------------------------------------------------------
// D0.3/D0.4: persistent main kernel. One CTA per SM, 384 threads = 3 warpgroups:
// warpgroup 0 is the producer (one elected lane issues the four bulk copies of a k-tile),
// warpgroups 1-2 are the eight consumer warps, cooperating on the *same* output tile with
// one fp32 accumulator set (16 m16n8 chains x 4 registers per thread).
// Consumer warp cw: warp_m = cw>>2 (rows 64*warp_m..+63), warp_n = cw&3 (cols 32*warp_n..+31);
// 4 m-subtiles of 16 rows x 4 n-subtiles of 8 cols = the 16 chains.
// ---------------------------------------------------------------------------------------
struct GemmParams {
    const uint8_t* sfa;    // re-laid-out [m/128][k/128][128][8], or physSF 32_4_4 (PHYS_SF)
    const uint8_t* sfb;    // ditto for [n/128]...
    const __half* bias;    // [n]
    float alpha;           // per-tensor weight global g_w: y = fp16(alpha*acc + bias)
    int m, n, k;
    int tiles_m, tiles_n, k_tiles;
    int swz_w;        // D0.8 dispatch-order block width (lever of D1.1; starting value 4)
    int serpentine;   // D0.8 traversal direction (0 = forward, 1 = serpentine; starting value 0)
};

// ---------------------------------------------------------------------------------------
// D0.8: the persistent CTA's tile sequence is a parameterised map from the linear work
// index to (tile_m, tile_n) -- an m-block swizzle of width w, optionally serpentine in n.
// w == 1 && !serpentine reproduces the linear (identity) row-major order, which is the
// anchor of the residency line. The width and the direction are the levers of D1.1.
// ---------------------------------------------------------------------------------------
__device__ __forceinline__ void tile_coords(int idx, const GemmParams& p, int& tile_m, int& tile_n) {
    const int w = p.swz_w;
    const int gsz = w * p.tiles_n;          // tiles in one full m-block group
    const int group = idx / gsz;
    const int first_m = group * w;
    const int cur_w = min(w, p.tiles_m - first_m);   // the last group may be short
    const int r = idx - group * gsz;
    tile_m = first_m + r % cur_w;
    int tn = r / cur_w;
    if (p.serpentine && (group & 1)) tn = p.tiles_n - 1 - tn;
    tile_n = tn;
}

// KDA-4 r17 merge: PHYS_SF reads the pipeline's 32_4_4 physical SF layout —
// within the same 1024B per-(panel, k-tile) block, byte (row r, sf-col c) sits
// at (c/4)*512 + (r%32)*16 + (r/32)*4 + c%4 (verified byte-exact vs to_blocked).
template <bool PHYS_SF>
__device__ __forceinline__ uint32_t sf_word(const uint8_t* sf_s, int row, int ks) {
    if (PHYS_SF)
        return *(const uint32_t*)(sf_s + ks * 512 + (row & 31) * 16 + (row >> 5) * 4);
    return *(const uint32_t*)(sf_s + row * SF_COLS + ks * 4);
}

template <bool PHYS_SF>
__global__ void __launch_bounds__(NUM_THREADS, 1)
gemm_kernel(const __grid_constant__ CUtensorMap amap, const __grid_constant__ CUtensorMap bmap,
            const __grid_constant__ CUtensorMap ymap, const GemmParams p) {
    extern __shared__ __align__(1024) uint8_t smem[];
    const uint32_t sbase = sm_addr(smem);
    const uint32_t bar_full = sbase + SM_BAR;              // STAGES x 8 B
    const uint32_t bar_empty = bar_full + STAGES * 8;      // STAGES x 8 B
    __half* const s_bias = (__half*)(smem + SM_BIAS);

    const int tid = threadIdx.x;
    if (tid < STAGES) {
        bar_init(bar_full + tid * 8, 1);     // the elected producer lane publishes a k-tile
        bar_init(bar_empty + tid * 8, 8);    // one arrival per consumer warp releases a slot
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    const int num_tiles = p.tiles_m * p.tiles_n;
    const int k_tiles = p.k_tiles;

    // ------------------------------------------------------------------ producer
    if (tid < 128) {
        if (tid == 0) {
            int it = 0;
            for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
                int tile_m, tile_n;
                tile_coords(tile, p, tile_m, tile_n);   // D0.8
                const uint8_t* sfa_t = p.sfa + (int64_t)(tile_m * k_tiles) * SF_BLOCK_BYTES;
                const uint8_t* sfb_t = p.sfb + (int64_t)(tile_n * k_tiles) * SF_BLOCK_BYTES;
                for (int kt = 0; kt < k_tiles; ++kt, ++it) {
                    const int s = it & (STAGES - 1);
                    const int lap = it / STAGES;
                    if (it >= STAGES) bar_wait(bar_empty + s * 8, (lap - 1) & 1);   // P1
                    const uint32_t bf = bar_full + s * 8;
                    bar_expect(bf, 2 * 8192 + 2 * 1024);                            // P3 (tx count)
                    // P2: four copies in flight together, every address hoisted
                    tma_2d(sbase + SM_A + s * 8192, &amap, kt * BK_BYTES, tile_m * BM, bf);
                    tma_2d(sbase + SM_B + s * 8192, &bmap, kt * BK_BYTES, tile_n * BN, bf);
                    bulk_1d(sbase + SM_SFA + s * 1024, sfa_t + kt * SF_BLOCK_BYTES, 1024, bf);
                    bulk_1d(sbase + SM_SFB + s * 1024, sfb_t + kt * SF_BLOCK_BYTES, 1024, bf);
                }
            }
        }
        return;
    }

    // ------------------------------------------------------------------ consumers
    const int ctid = tid - 128;            // 0..255
    const int cw = ctid >> 5;              // consumer warp 0..7
    const int lane = ctid & 31;
    const int warp_m = cw >> 2;            // 0..1
    const int warp_n = cw & 3;             // 0..3
    const int g = lane >> 2, q = lane & 3;
    const int rho = (lane & 7) + 8 * ((lane >> 3) & 1);   // ldmatrix row within an 16-row group
    const int chq = lane >> 4;                            // ldmatrix 16 B-chunk selector

    float acc[16][4];
    int it = 0;

    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        int tile_m, tile_n;
        tile_coords(tile, p, tile_m, tile_n);   // D0.8 (same map as the producer)

        if (ctid < BN) s_bias[ctid] = p.bias[tile_n * BN + ctid];
#pragma unroll
        for (int c = 0; c < 16; ++c)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[c][e] = 0.0f;

        for (int kt = 0; kt < k_tiles; ++kt, ++it) {
            const int s = it & (STAGES - 1);
            bar_wait(bar_full + s * 8, (it / STAGES) & 1);

            const uint32_t pa = sbase + SM_A + s * 8192;
            const uint32_t pb = sbase + SM_B + s * 8192;
            const uint8_t* sfa_s = smem + SM_SFA + s * 1024;
            const uint8_t* sfb_s = smem + SM_SFB + s * 1024;

            // C1: the two k_block fragment sets are issued inside one unrolled body, so the
            // loads of k_block j+1 schedule under the MMA chain of k_block j.
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                uint32_t af[4][4], bf[4][2], sa[4], sb[4];
                const int c = 2 * ks + chq;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = warp_m * 64 + i * 16 + rho;
                    ldm_x4(af[i], pa + r * BK_BYTES + swz64(r, c) * 16);
                    sa[i] = sf_word<PHYS_SF>(sfa_s, warp_m * 64 + i * 16 + g + 8 * (q & 1), ks);
                }
#pragma unroll
                for (int jj = 0; jj < 2; ++jj) {
                    const int r = warp_n * 32 + jj * 16 + rho;
                    uint32_t t[4];
                    ldm_x4(t, pb + r * BK_BYTES + swz64(r, c) * 16);
                    bf[2 * jj][0] = t[0];     bf[2 * jj][1] = t[2];
                    bf[2 * jj + 1][0] = t[1]; bf[2 * jj + 1][1] = t[3];
                }
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    sb[j] = sf_word<PHYS_SF>(sfb_s, warp_n * 32 + j * 8 + g, ks);
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        mma_fp4(acc[i * 4 + j], af[i], bf[j], sa[i], sb[j]);
            }

            asm volatile("barrier.sync 1, 256;");            // C2: both warpgroups done with slot s
            if (lane == 0) bar_arrive(bar_empty + s * 8);    // C3
        }

        // -------------------------------------------------- epilogue (D0.7)
        // Each m-half owns one 4096 B staging buffer and four 16 x 128 subtiles; the four
        // warps of the half cover the 128 columns (32 each) with stmatrix, then one elected
        // lane moves the subtile out with two 64-column bulk tensor stores.
        const uint32_t ystage = sbase + SM_Y + warp_m * 4096;
        const int panel = warp_n >> 1;
        const int ch0 = (warp_n & 1) * 4 + (lane >> 3);
        const bool elect = (warp_n == 0) && (lane == 0);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            uint32_t regs[2][4];
#pragma unroll
            for (int half = 0; half < 2; ++half)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = warp_n * 32 + j * 8 + 2 * q;
                    float v0 = p.alpha * acc[t * 4 + j][half * 2 + 0] + __half2float(s_bias[col]);
                    float v1 = p.alpha * acc[t * 4 + j][half * 2 + 1] + __half2float(s_bias[col + 1]);
                    __half2 h = __halves2half2(__float2half(v0), __float2half(v1));
                    regs[half][j] = *(uint32_t*)&h;
                }
            if (t > 0) {   // the previous subtile's store must have drained the buffer
                if (elect) asm volatile("cp.async.bulk.wait_group.read 0;");
                if (warp_m == 0) asm volatile("barrier.sync 2, 128;");
                else asm volatile("barrier.sync 3, 128;");
            }
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int rr = (lane & 7) + 8 * half;
                const uint32_t ps = ystage + panel * 2048 + rr * 128 + (ch0 ^ (rr & 7)) * 16;
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};" ::"r"(ps),
                             "r"(regs[half][0]), "r"(regs[half][1]), "r"(regs[half][2]),
                             "r"(regs[half][3]));
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (warp_m == 0) asm volatile("barrier.sync 2, 128;");
            else asm volatile("barrier.sync 3, 128;");
            if (elect) {
                const int row0 = tile_m * BM + warp_m * 64 + t * 16;
                const int col0 = tile_n * BN;
#pragma unroll
                for (int pnl = 0; pnl < 2; ++pnl)
                    asm volatile(
                        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" ::
                        "l"((uint64_t)(&ymap)), "r"(ystage + pnl * 2048), "r"(col0 + pnl * 64),
                        "r"(row0)
                        : "memory");
                asm volatile("cp.async.bulk.commit_group;");
            }
        }
        if (elect) asm volatile("cp.async.bulk.wait_group.read 0;");
        asm volatile("barrier.sync 1, 256;");   // the staging buffers are free for the next tile
    }
}

// ---------------------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------------------
static CUtensorMap make_map2d(void* ptr, CUtensorMapDataType dt, uint64_t d0, uint64_t d1,
                              uint64_t stride1_bytes, uint32_t b0, uint32_t b1,
                              CUtensorMapSwizzle swz) {
    CUtensorMap map{};
    uint64_t gdim[2] = {d0, d1};
    uint64_t gstride[1] = {stride1_bytes};
    uint32_t bdim[2] = {b0, b1};
    uint32_t estride[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(&map, dt, 2, ptr, gdim, gstride, bdim, estride,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed, code ", (int)r);
    return map;
}

static int num_sms_of(int dev) {
    static int cached_dev = -1, cached_sms = 0;
    if (dev != cached_dev) {
        cudaDeviceGetAttribute(&cached_sms, cudaDevAttrMultiProcessorCount, dev);
        cached_dev = dev;
    }
    return cached_sms;
}

template <bool PHYS_SF>
static torch::Tensor launch_spg(torch::Tensor& a_lo, const uint8_t* sfa, torch::Tensor& b_lo,
                                const uint8_t* sfb, torch::Tensor& bias, double alpha,
                                int swz_w, int serpentine, cudaStream_t stream) {
    const int m = (int)a_lo.size(0), n = (int)b_lo.size(0), k = (int)a_lo.size(1) * 2;
    TORCH_CHECK(k % BK == 0 && m % BM == 0 && n % BN == 0, "shape must be a multiple of the 128x128x128 tile");
    auto y = torch::empty({m, n}, bias.options());

    CUtensorMap amap = make_map2d(a_lo.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, (uint64_t)k / 2,
                                  (uint64_t)m, (uint64_t)k / 2, BK_BYTES, BM,
                                  CU_TENSOR_MAP_SWIZZLE_64B);
    CUtensorMap bmap = make_map2d(b_lo.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, (uint64_t)k / 2,
                                  (uint64_t)n, (uint64_t)k / 2, BK_BYTES, BN,
                                  CU_TENSOR_MAP_SWIZZLE_64B);
    CUtensorMap ymap = make_map2d(y.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT16, (uint64_t)n,
                                  (uint64_t)m, (uint64_t)n * 2, 64, 16,
                                  CU_TENSOR_MAP_SWIZZLE_128B);

    GemmParams p;
    p.sfa = sfa;
    p.sfb = sfb;
    p.bias = (const __half*)bias.data_ptr();
    p.alpha = (float)alpha;
    p.m = m; p.n = n; p.k = k;
    p.tiles_m = m / BM; p.tiles_n = n / BN; p.k_tiles = k / BK;
    p.swz_w = swz_w > 0 ? swz_w : 4;       // evolved starting value 4
    p.serpentine = serpentine;

    static bool smem_opted_in = false;
    if (!smem_opted_in) {
        cudaFuncSetAttribute(gemm_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        cudaFuncSetAttribute(gemm_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        smem_opted_in = true;
    }
    const int sms = num_sms_of(a_lo.get_device());
    gemm_kernel<PHYS_SF><<<sms, NUM_THREADS, SMEM_BYTES, stream>>>(amap, bmap, ymap, p);
    return y;
}

torch::Tensor run(torch::Tensor a_lo, torch::Tensor sfa_lo, torch::Tensor b_lo, torch::Tensor sfb_lo,
                  torch::Tensor bias, double alpha = 1.0, int64_t swz_w = 0, int64_t serpentine = 0) {
    TORCH_CHECK(a_lo.is_cuda() && b_lo.is_cuda() && sfa_lo.is_cuda() && sfb_lo.is_cuda() && bias.is_cuda());
    TORCH_CHECK(bias.scalar_type() == torch::kFloat16);
    torch::Tensor A = a_lo.view(torch::kUInt8).contiguous();
    torch::Tensor B = b_lo.view(torch::kUInt8).contiguous();
    torch::Tensor sfa = sfa_lo.view(torch::kUInt8).contiguous();
    torch::Tensor sfb = sfb_lo.view(torch::kUInt8).contiguous();
    torch::Tensor Bias = bias.contiguous();
    const int m = (int)A.size(0), n = (int)B.size(0), k = (int)A.size(1) * 2;
    TORCH_CHECK(B.size(1) * 2 == k && sfa.size(1) * 16 == k && sfb.size(1) * 16 == k);
    TORCH_CHECK(sfa.size(0) == m && sfb.size(0) == n && Bias.numel() == n);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int k_tiles = k / BK;
    torch::Tensor sfa_r = sf_relayout_cached(sfa, k_tiles, stream);
    torch::Tensor sfb_r = sf_relayout_cached(sfb, k_tiles, stream);
    return launch_spg<false>(A, sfa_r.data_ptr<uint8_t>(), B, sfb_r.data_ptr<uint8_t>(),
                             Bias, alpha, (int)swz_w, (int)serpentine, stream);
}

// KDA-4 r17: steady-state entry — consumes the pipeline's physical 32_4_4 SF
// buffers (physSFB_lo from pack, psfA_lo from the qpf activation pack) directly.
// ONE kernel launch, CUDA-graph capturable, no prep work.
torch::Tensor run_prepacked(torch::Tensor a_lo, torch::Tensor psfa_lo,
                            torch::Tensor b_lo, torch::Tensor psfb_lo,
                            torch::Tensor bias, double alpha = 1.0,
                            int64_t swz_w = 0, int64_t serpentine = 0) {
    c10::cuda::CUDAGuard guard(a_lo.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor A = a_lo.view(torch::kUInt8).contiguous();
    torch::Tensor B = b_lo.view(torch::kUInt8).contiguous();
    torch::Tensor pa = psfa_lo.view(torch::kUInt8).contiguous();
    torch::Tensor pb = psfb_lo.view(torch::kUInt8).contiguous();
    torch::Tensor Bias = bias.contiguous();
    const int m = (int)A.size(0), n = (int)B.size(0), k = (int)A.size(1) * 2;
    TORCH_CHECK(B.size(1) * 2 == k && Bias.numel() == n);
    TORCH_CHECK((int64_t)pa.numel() >= (int64_t)((m + 127) / 128) * (k / 128) * 1024,
                "psfa_lo too small for m x k");
    TORCH_CHECK((int64_t)pb.numel() >= (int64_t)((n + 127) / 128) * (k / 128) * 1024,
                "psfb_lo too small for n x k");
    return launch_spg<true>(A, pa.data_ptr<uint8_t>(), B, pb.data_ptr<uint8_t>(),
                            Bias, alpha, (int)swz_w, (int)serpentine, stream);
}

std::vector<int64_t> debug_gemm_attrs() {
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, gemm_kernel<true>);
    return {a.numRegs, (int64_t)a.sharedSizeBytes, (int64_t)SMEM_BYTES};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, pybind11::arg("a_lo"), pybind11::arg("sfa_lo"), pybind11::arg("b_lo"),
          pybind11::arg("sfb_lo"), pybind11::arg("bias"), pybind11::arg("alpha") = 1.0,
          pybind11::arg("swz_w") = 0, pybind11::arg("serpentine") = 0);
    m.def("run_prepacked", &run_prepacked, pybind11::arg("a_lo"), pybind11::arg("psfa_lo"),
          pybind11::arg("b_lo"), pybind11::arg("psfb_lo"), pybind11::arg("bias"),
          pybind11::arg("alpha") = 1.0, pybind11::arg("swz_w") = 0,
          pybind11::arg("serpentine") = 0);
    m.def("debug_gemm_attrs", &debug_gemm_attrs);
}
