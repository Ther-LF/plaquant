// act_mse.cuh — round-4 CHEAP activation NVFP4 block-scale MSE search.
//
// Shared by all fused activation producers (quant_pack_float_ext.cu,
// hadamard_pack_ext.cu, hadamard28_pack_ext.cu, quant_pack_fused_ext.cu,
// add_rmsnorm_pack_ext.cu).  ONLY the per-block SCALE SELECTION; the caller's
// final code-writing pass (true division + exact RTN grid) is untouched, so the
// emitted codes/SF for a given selected scale stay byte-exact to the torch twin.
//
// Round-3 shipped a 4-candidate search that re-divided the 16-wide block by each
// candidate scale (16 true FP32 divisions per candidate x 4 candidates + a final
// pass) — ~5x the per-block ALU of the no-MSE pack (8B pack 9.2->48.7 ms).
//
// The cost is the DIVISION: 16 true divisions per candidate.  Two levers here,
// compile-time selectable so a harness can A/B them against the round-3 variant:
//
//   PROMIX_ACT_MSE_FAST (default 1):
//     * reciprocal-multiply: compute inv = 1/scale ONCE per candidate (1 divide)
//       and evaluate x*inv (16 multiplies) instead of 16 divisions — selection
//       only, so the tiny reciprocal rounding never reaches the emitted codes;
//     * branchless nearest-FP4 VALUE (a short fsel cascade over the fixed E2M1
//       grid) instead of the data-dependent binary search — no warp divergence
//       across the 32 blocks a warp owns.
//     Not bit-identical to the round-3 true-division search (borderline blocks
//     may pick an adjacent E4M3 step) -> gated by PPL, not torch.equal.
//
//   PROMIX_ACT_MSE_FAST == 0:
//     Exact reproduction of the round-3 search (true division + tie-to-even RTN
//     over the same {0,.5,1,1.5,2,3,4,6} grid) -> bit-identical to round-3, used
//     to reproduce the accuracy-max PPL anchor from this same source.
//
//   PROMIX_ACT_MSE_STEPS (default 3): number of LOWER E4M3 grid steps tried
//     below the ceil base scale (round-2 showed depth saturates at 3).
#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#ifndef PROMIX_ACT_MSE_STEPS
#define PROMIX_ACT_MSE_STEPS 3
#endif
#ifndef PROMIX_ACT_MSE_FAST
#define PROMIX_ACT_MSE_FAST 1
#endif

// E4M3 byte -> float (hardware), local to the header (no per-file symbol).
__device__ __forceinline__ float promix_e4m3_to_float(unsigned char b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}

// Branchless nearest FP4 E2M1 grid VALUE for a >= 0 (used only inside the MSE
// selection).  Grid {0,.5,1,1.5,2,3,4,6}; midpoints 0.25/0.75/1.25/1.75/2.5/3.5/5.
// Ties resolved by the `<` cascade (measure-zero vs the exact tie-to-even RTN the
// caller uses for the final emitted codes).
__device__ __forceinline__ float promix_fp4_nearest_val(float a) {
  float v = 6.0f;
  v = (a < 5.00f) ? 4.0f : v;
  v = (a < 3.50f) ? 3.0f : v;
  v = (a < 2.50f) ? 2.0f : v;
  v = (a < 1.75f) ? 1.5f : v;
  v = (a < 1.25f) ? 1.0f : v;
  v = (a < 0.75f) ? 0.5f : v;
  v = (a < 0.25f) ? 0.0f : v;
  return v;
}

#if !PROMIX_ACT_MSE_FAST
// Exact tie-to-even RTN over the fixed FP4 grid (reproduces the round-3 rtn_idx).
__device__ __forceinline__ float promix_fp4_nearest_val_exact(float absx) {
  const float v[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  int lo = 0, hi = 8;
  while (lo < hi) { int mid = (lo + hi) >> 1; if (v[mid] < absx) lo = mid + 1; else hi = mid; }
  int insert = lo; if (insert > 7) insert = 7; if (insert < 0) insert = 0;
  int idx_hi = insert < 1 ? 1 : insert;
  int idx_lo = idx_hi - 1; if (idx_lo < 0) idx_lo = 0;
  float vlo = v[idx_lo], vhi = v[idx_hi];
  float dlo = fabsf(absx - vlo), dhi = fabsf(vhi - absx);
  bool pick_hi = dhi < dlo;
  bool is_tie = fabsf(dhi - dlo) < 1e-12f;
  bool even_lo = (idx_lo & 1) == 0;
  if (is_tie) pick_hi = !even_lo;
  return pick_hi ? vhi : vlo;
}
#endif

// Select the per-block NVFP4 scale minimizing dequant MSE over {base, up to
// PROMIX_ACT_MSE_STEPS lower E4M3 grid steps}.  xf[0..15] is the block (fp32);
// sfb/scale enter as the ceil base (E4M3 byte + decoded value) and are updated
// in place to the argmin-MSE candidate.
__device__ __forceinline__ void promix_act_mse_select(
    const float* __restrict__ xf, unsigned char& sfb, float& scale) {
#if PROMIX_ACT_MSE_FAST
  // --- fast path: reciprocal-multiply + branchless nearest ---
  float best_sc = scale; unsigned char best_b = sfb;
  float inv = 1.0f / best_sc;
  float best_err = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    float xs = fminf(fmaxf(xf[j] * inv, -6.0f), 6.0f);
    float g  = promix_fp4_nearest_val(fabsf(xs));
    float dq = (xs < 0.0f ? -g : g) * best_sc;
    float e  = dq - xf[j]; best_err += e * e;
  }
  unsigned char cb = sfb;
#pragma unroll
  for (int s = 0; s < PROMIX_ACT_MSE_STEPS; ++s) {
    if (cb <= 1) break;
    cb = (unsigned char)(cb - 1);
    float csc = promix_e4m3_to_float(cb); if (!(csc > 0.0f)) break;
    float cinv = 1.0f / csc;
    float err = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float xs = fminf(fmaxf(xf[j] * cinv, -6.0f), 6.0f);
      float g  = promix_fp4_nearest_val(fabsf(xs));
      float dq = (xs < 0.0f ? -g : g) * csc;
      float e  = dq - xf[j]; err += e * e;
    }
    if (err < best_err) { best_err = err; best_sc = csc; best_b = cb; }
  }
  scale = best_sc; sfb = best_b;
#else
  // --- exact path: reproduces the round-3 true-division search bit-for-bit ---
  float best_err = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    float xs = fminf(fmaxf(xf[j] / scale, -6.0f), 6.0f);
    float g  = promix_fp4_nearest_val_exact(fabsf(xs));
    float dq = (xs < 0.0f ? -1.0f : 1.0f) * g * scale;
    float e  = dq - xf[j]; best_err += e * e;
  }
  unsigned char cb = sfb;
  for (int s = 0; s < PROMIX_ACT_MSE_STEPS && cb > 1; ++s) {
    cb = (unsigned char)(cb - 1);
    float csc = promix_e4m3_to_float(cb); if (!(csc > 0.0f)) break;
    float err = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float xs = fminf(fmaxf(xf[j] / csc, -6.0f), 6.0f);
      float g  = promix_fp4_nearest_val_exact(fabsf(xs));
      float dq = (xs < 0.0f ? -1.0f : 1.0f) * g * csc;
      float e  = dq - xf[j]; err += e * e;
    }
    if (err < best_err) { best_err = err; scale = csc; sfb = cb; }
  }
#endif
}
