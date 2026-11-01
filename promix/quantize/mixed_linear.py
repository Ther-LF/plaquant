"""MixedQuantLinear — real packed NVFP4+MXFP8 fused-kernel Linear (SM120).

Drop-in replacement for the fake-quant forward of a rotated+reordered
`nn.Linear` in a ProMix decoder layer. Holds the offline-packed weight
(`PackedWeight`) and, per forward:

  1. splits the (already rotated+reordered) activation x into
     [low -> NVFP4 block-16 | high -> MXFP8 block-32] along K;
  2. quantizes each segment (kernel quantizers) and folds the pack + SF
     interleave into `cd.quant_pack_*` (ilv=True) -> packed codes + interleaved SFA;
  3. calls `cd.fused_gemm_prepacked` (activation + weight SF pre-interleaved) ->
     dequantized GEMM D (M,N);
  4. adds bias if present.

Scale convention matches the ProMix fake-quant path: the per-block FP scale
factors fully capture the quantization scale, so there is NO extra per-tensor
s_x / s_w multiply (s_x = s_w = 1). D is returned directly (+bias).

For M >= 2048 (the prefill regime this step targets) `fused_gemm_prepacked`'s
fixed 128x128 NV4/MX2 config IS the config the round-24 dispatcher
(`launch_best_codesign`) selects for every shape, so this path is both the
GEMM-only ceiling AND the dispatcher's choice. The dispatcher's
`launch_best_codesign` (raw-SF, internal interleave) is available via
`forward_dispatch` for cross-checking / small-M config selection.

The kernel modules (nv, mx, cd) are dependency-injected at construction.
"""

import torch
import torch.nn as nn

from promix.quantize.weight_pack import pack_weight_mixed, split_k, quantize_nvfp4_spec, quantize_mxfp8_spec
from promix.quantize.quant_utils import fake_quantize_nvfp4, fake_quantize_mxfp8

import os

# round-12 phase-adaptive decode: at M=1 the GEMV is bandwidth/latency-bound, so
# quantizing the activation (W4A4) buys no tensor-core throughput and only adds a
# pass + accuracy loss. "w4a16" (default) keeps the activation EXACT and dequantizes
# only the packed weights in-register; "w4a4" is the legacy quantize+pack decode.
# Prefill (M>1, compute-bound, tensor cores) is UNAFFECTED — it always runs W4A4.
def decode_mode():
    return os.environ.get("PROMIX_DECODE_MODE", "w4a16").strip().lower()


# vllm-perf round-1: largest flattened batch M routed through the decode GEMV.
# vLLM CUDA-graph capture pads decode batches to fixed sizes, so inside each
# captured graph M is static and this Python-side branch is capture-safe (the
# branch is evaluated once per captured size, never on-device). M=1 uses the
# round-12 single-row kernel; 2 <= M <= PROMIX_DECODE_M_MAX uses the batched
# W4A16 GEMV (decode_gemv_w4a16_mb), which amortizes the same packed weight
# stream over all M rows. Set PROMIX_DECODE_M_MAX=1 to restore the round-12
# M=1-only behavior; larger M always runs the fused tensor-core GEMM.
def decode_m_max():
    try:
        return int(os.environ.get("PROMIX_DECODE_M_MAX", "16"))
    except ValueError:
        return 16


# vllm-perf r2 lever 1: measured small-M GEMM config table. Keyed by (N, K);
# value = inclusive [m_lo, m_hi] window where cfg1 (128x64 NV4/MX2) beats the
# shipping 128x128 config, from promix/vllm_integ/gemm_probe.py under CUDA-graph
# replay on RTX Pro 6000 (r2_gemm_probe{,_rerun}.json; two independent runs
# agree). bit gate: all configs torch.equal to the shipping kernel. Outside the
# window (and for unknown shapes) the shipping config stays. The windows follow
# CTA-wave arithmetic (128x64 wins while ceil(M/128)*ceil(N/64) stays within one
# 188-SM wave AND M is large enough that doubled A/SFB re-reads amortize), but
# the encoded values are the MEASURED ones, not the formula.
#   (5120,3072)  3B qkv      M 64..256   (1.05-1.47x)
#   (3072,3072)  3B o        M 8..384    (1.15-1.49x)
#   (3072,8192)  3B down     M 8..384    (1.15-1.60x)
#   (6144,4096)  8B qkv      M 96..128   (1.44-1.47x; 2-wave above)
#   (4096,4096)  8B o        M 32..256   (1.05-1.49x)
#   (4096,14336) 8B down     M 32..256   (1.47-1.63x)
#   gate_up (N>=2K) never wins — N/128 CTAs already fill the machine.
_SMALLM_TABLE = {
    (5120, 3072): (64, 256),
    (3072, 3072): (8, 384),
    (3072, 8192): (8, 384),
    (6144, 4096): (96, 128),
    (4096, 4096): (32, 256),
    (4096, 14336): (32, 256),
}

# vllm-perf r3 lever 1: measured split-K/stream-K windows. Keyed by (N, K);
# value = list of (m_lo, m_hi, cfg, splits, decomp) windows (first match wins)
# for cd_splitk_ext's fused_gemm_prepacked_sk, measured by
# promix/vllm_integ/splitk_probe.py under CUDA-graph replay on RTX Pro 6000
# (r3_splitk_probe.json). cfg: 0=128x128, 1=128x64; decomp: 0=heuristic,
# 1=split-K with `splits`, 2=stream-K. Windows kept only where the probed
# endpoints beat the r2-best (min of shipping 128x128 and the _SMALLM_TABLE
# cfg1) by >=5%:
#   (3072,8192)   3B down   M 8..32 c0 splits=4 (1.06-1.16x vs r2-best),
#                           M 64..128 c1 heuristic (1.22-1.30x)
#   (4096,14336)  8B down   M 8..16 c0 splits=4 (1.48-1.66x!),
#                           M 32..512 c0 stream-K (1.07-1.35x)
#   (28672,4096)  8B gate_up M 8..256 c0 heuristic (1.09-1.21x)
#   qkv (both), o (both), 3B gate_up: split-K never beats the r2 config -> no
#   window (the serial-K latency there is already near the small-N CTA floor).
# Split-K takes priority over _SMALLM_TABLE inside its windows. NUMERICS: the
# FP32 partial sums are reduced in a different (deterministic) order than the
# serial mainloop -> NOT torch.equal, cosine ~1.0 (gated by splitk_probe + the
# T2 decode-route gate). The prefill path (M >= 1024 chunks) never routes
# here, so the T2 prefill gate stays byte-identical.
_SPLITK_TABLE = {
    (3072, 8192): [(8, 32, 0, 4, 1), (64, 128, 1, 1, 0)],
    (4096, 14336): [(8, 16, 0, 4, 1), (32, 512, 0, 1, 2)],
    (28672, 4096): [(8, 256, 0, 1, 0)],
}

# vllm-perf r4 lever 1: Marlin W4A16 tensor-core small-M windows. Keyed by
# (N, K); value = inclusive [m_lo, m_hi] window where the Marlin fp4 low-GEMM
# + decode_gemv_w8a16_hi_add high-companion route beats the best W4A4 GEMM
# route (r2 small-M cfg + r3 split-K tables INCLUDED in the baseline, plus its
# qpfm pack + fp16 cast), measured by promix/vllm_integ/marlin_probe.py under
# CUDA-graph replay on RTX Pro 6000. Takes PRIORITY over the batched GEMV
# inside its window (probe: Marlin wins 1.2-3.1x at M=2..4 too — same W4A16
# numerics class); M=1 keeps the round-12 single-row GEMV (byte-proven), and
# M <= 32 is the hi_add kernel bound (prefill M never routes here). W4A16
# numerics: the
# activation is NOT quantized on this route (same class as the GEMV routes —
# decode-route PPL moves in the improvement direction; T2-decode gated).
# Measured windows (r4_marlin_probe.json + _m2 rerun, graph-replay, >=5%-win
# rule like r3; o_proj windows use the gather-inclusive serve chains
# t_cur_gatherpack vs t_gather_marlin). The hi_add companion scales ~linearly
# in M (it is the batched-GEMV weakness in miniature), so every window closes
# by M=16-24 even though the Marlin low GEMM stays near-flat to M=32 — the
# high-segment TC companion is the named r5 lever.
# SERVE-LEVEL CORRECTION (honest probe/serve discrepancy): the kernel-level
# probe showed 8B M=16 wins (qkv 1.07x / o-chain 1.12x / gate_up 1.19x) but
# the same-session serve A/B measured 8B decode-b16 TPOT +3..+4% with M=16
# routing ON, reproduced across two sessions, and PROMIX_MARLIN_MMAX=8
# restores the control digits exactly (pm8b_mmax8 vs pm8b_r4ctl) — so the 8B
# windows are capped at the SERVE-measured m_hi=8. 3B keeps m_hi=16 (serve
# b16 -2%). Root cause not isolated (probe = idle-GPU back-to-back replay;
# engine = full-graph with interleaved attention/norm kernels).
# SERVE-LEVEL CORRECTION (honest probe/serve discrepancy): the kernel-level
# probe showed 8B M=16 wins (qkv 1.07x / o-chain 1.12x / gate_up 1.19x) but
# the same-session serve A/B measured 8B decode-b16 TPOT +3..+4% with M=16
# routing ON, reproduced across two sessions, and PROMIX_MARLIN_MMAX=8
# restores the control digits exactly (pm8b_mmax8 vs pm8b_r4ctl) — so the 8B
# windows were capped at the SERVE-measured m_hi=8 in r4.
# r5 UPDATE (r5_marlin_hi_probe2.json, graph-replay, >=5% rule): with the
# MXFP8-marlin high companion (which is near-flat in M like the low GEMM,
# replacing the linearly-scaling hi_add inside _MARLIN_HI_TABLE windows) the
# route beats the W4A4 GEMM out to M=32-48 on most shapes (e.g. 8B qkv M16
# 1.50x, M32 1.30x; 8B gate_up M32 1.73x, M48 1.35x). The windows ship at the
# probe values; the r4 8B serve cap is re-checked by the r5 serve A/B
# (PROMIX_MARLIN_MMAX remains the kill switch if the discrepancy persists).
# 3B o keeps (2,8): the r5 gather chain reads 1.00x at M16 (no win).
_MARLIN_TABLE = {
    (5120, 3072): (2, 48),      # 3B qkv   (M8 1.64x, M16 1.43x, M48 1.09x)
    (3072, 3072): (2, 8),       # 3B o     (chain 1.11-1.21x; M16 1.00 stop)
    (16384, 3072): (2, 32),     # 3B gate_up (M8 2.28x, M32 1.35x; M48 1.02 stop)
    (3072, 8192): (2, 48),      # 3B down  (M8 1.92x, M32 1.49x, M48 1.09x)
    (6144, 4096): (2, 48),      # 8B qkv   (M8 1.63x, M16 1.50x, M48 1.08x)
    (4096, 4096): (2, 32),      # 8B o     (chain 1.14-1.40x; M48 0.95 stop)
    (28672, 4096): (2, 48),     # 8B gate_up (M8 2.84x, M32 1.73x, M48 1.35x)
    (4096, 14336): (2, 32),     # 8B down  (M8 2.18x, M32 1.23x; M48 0.99 stop)
}

# vllm-perf r5 lever 1: per-shape smallest M where the MXFP8-marlin high
# companion (MarlinHigh.gemm + fused axpy) beats the hi_add GEMV at the ROUTE
# level (slice-cast included; the GEMV scales ~linearly in M — the r4
# bottleneck — while the companion is near-flat). Measured by
# promix/vllm_integ/marlin_hi_probe.py under CUDA-graph replay
# (r5_marlin_hi_probe2.json route_speedup_vs_r4). Below the crossover (and
# when PROMIX_MARLIN_HI=0 / prepack unavailable) the route keeps the hi_add
# GEMV; above M=32 (the hi_add kernel bound) the route REQUIRES the companion
# (marlin_route refuses otherwise).
_MARLIN_HI_TABLE = {
    (5120, 3072): 16,     # 3B qkv   (M16 1.01x tie, M24 1.30x)
    (3072, 3072): 16,     # 3B o     (M16 1.06x)
    (16384, 3072): 4,     # 3B gate_up (M4 1.08x, M8 1.37x)
    (3072, 8192): 8,      # 3B down  (M8 1.07x, M16 1.34x)
    (6144, 4096): 8,      # 8B qkv   (M8 1.14x, M16 1.43x)
    (4096, 4096): 16,     # 8B o     (M8 0.95, M16 1.10x)
    (28672, 4096): 2,     # 8B gate_up (M2 1.11x, M8 1.60x)
    (4096, 14336): 8,     # 8B down  (M8 1.23x, M16 1.48x)
}


def _marlin_env_mmax():
    """PROMIX_MARLIN_MMAX caps the Marlin window (0 disables routing)."""
    try:
        return int(os.environ.get("PROMIX_MARLIN_MMAX", "32"))
    except ValueError:
        return 32


def _splitk_env_scale():
    """PROMIX_SPLITK_MMAX caps the split-K window (0 disables routing)."""
    try:
        return int(os.environ.get("PROMIX_SPLITK_MMAX", "1024"))
    except ValueError:
        return 1024


class MixedQuantLinear(nn.Module):
    def __init__(self, packed, nv, mx, cd, out_dtype=torch.float16, qpf=None, dg=None,
                 qpfm=None, hpk=None, cd_bias=None, arn=None, hpk28=None, had28=None,
                 cd_sm=None, cd_sk=None):
        """
        Args:
            packed: PackedWeight from pack_weight_mixed.
            nv, mx, cd: kernel python modules.
            out_dtype: dtype of the returned activation (default fp16).
            qpf: optional quant_pack_float_ext module. When provided, the
                per-forward activation quant+pack uses the fused float-input
                CUDA kernel (quant_pack_float_fp4/fp8) — one pass float->packed
                codes + interleaved SF — instead of the slow torch quantize +
                cd.quant_pack. Bit-exact to the torch path (probe-verified), so
                the GEMM output is unchanged; it removes the 93-98%-of-e2e
                torch quantize cost. If None, falls back to the torch path.
            dg: optional decode_gemv_ext module. When provided AND the flattened
                batch M==1 (autoregressive DECODE), the forward uses the
                memory-bound W4A4 GEMV (decode_gemv_ext.cu) that streams the
                packed 4-bit/8-bit weights at HBM bandwidth — the prefill-oriented
                128x128 fused GEMM is ~1/128-utilized and bandwidth-wasteful at
                M=1. Bit-faithful to the fused path (dequant identical to W_deq).
                For M>1 the fused prefill GEMM is used regardless.
            cd_bias: optional codesign_fused_bias_ext module (round-26 lever A).
                When provided and this Linear has a bias, the prefill GEMM runs
                `fused_gemm_prepacked_bias`, folding `D.to(fp16)` AND `D + bias`
                into the GEMM epilogue with the reference arithmetic
                half(float(bf16(acc)) + float(bias)). Bit-identical to the
                two-pass path (probe-verified); removes 2 elementwise launches
                and ~8 B/element of HBM traffic per Linear per forward.
            arn: optional add_rmsnorm_pack_ext module (round-26 lever B). Used by
                `forward_norm_pack`, the fused residual-add + RMSNorm + quant/pack
                producer entry called by ActQuantWrapper when the wrapper has
                `norm_pack` enabled.
        """
        super().__init__()
        self.N = packed.N
        self.K = packed.K
        self.K_low = packed.K_low
        self.K_high = packed.K_high
        # K_high==0 -> pure-NVFP4 Linear (0 MXFP8 rings, e.g. rate-distortion low
        # budgets). The two-phase fused kernel cannot build an empty MXFP8 segment
        # (Phase-2 pack / TMA / SF layout assume K_high>0); instead we dispatch the
        # single-phase NVFP4 GEMM (cd.single_nvfp4_gemm_prepacked), which reuses the
        # SAME tuned NVFP4 Phase-1 collective with Phase-2 skipped at compile time.
        self.is_pure_nvfp4 = (self.K_high == 0)
        # K_low==0 (pure-MXFP8) has no single-phase MXFP8 real-kernel path -> refuse
        # clearly rather than silently serving a different split.
        if self.K_low == 0:
            raise ValueError(
                f"MixedQuantLinear: K_low==0 (pure-MXFP8, K_high={self.K_high}, "
                f"K={self.K}, N={self.N}) is unsupported — no single-phase MXFP8 "
                "kernel. Route to a pure-MXFP8 GEMM or allocate >=1 NVFP4 ring.")
        self.out_dtype = out_dtype
        # Round-6 two-level weight global (per-tensor FP32, pow2-or-arbitrary):
        # carried on the fused-GEMM epilogue alpha so D = g*(A@B). 1.0 = single-level.
        # The decode-GEMV path folds g into its FP32 scales (S_*_dec) instead.
        self.g_w = float(getattr(packed, "g_w", 1.0))
        # packed buffers (registered so .to(device) moves them)
        self.register_buffer("W_lo_packed", packed.W_lo_packed)
        self.register_buffer("physSFB_lo", packed.physSFB_lo)
        self.register_buffer("W_hi_bytes", packed.W_hi_bytes)
        self.register_buffer("physSFB_hi", packed.physSFB_hi)
        self.register_buffer("bias", packed.bias)
        # decode-GEMV buffers (optional; only used when dg is provided and M==1)
        self.register_buffer("W_lo_dec", packed.W_lo_dec)
        self.register_buffer("S_lo_dec", packed.S_lo_dec)
        self.register_buffer("W_hi_dec", packed.W_hi_dec)
        self.register_buffer("S_hi_dec", packed.S_hi_dec)
        # kernel modules (not buffers)
        self._nv = nv
        self._mx = mx
        self._cd = cd
        self._qpf = qpf
        self._qpfm = qpfm
        self._dg = dg
        self._hpk = hpk
        self._cd_bias = cd_bias
        self._arn = arn
        # 8B non-power-of-two (I=14336) fused SwiGLU+Hadamard+pack producer + its
        # 28x28 +-1 Hadamard matrix (kept on CPU; the kernel caches it to constant
        # memory on first call). Only used by forward_swiglu_pack when K==14336.
        self._hpk28 = hpk28
        self._had28 = had28
        # vllm-perf r2 lever 1: small-M fused-GEMM config ext (cd_smallm_ext).
        # The shipping 128x128 NV4/MX2 config leaves most SMs idle below its
        # 128-row M tile (N/128 CTAs); the 128x64 tiles double the CTA count.
        # Config per (shape, M) measured by promix/vllm_integ/gemm_probe.py
        # under CUDA-GRAPH REPLAY timing (r1 lesson) — see _smallm_cfg.
        self._cd_sm = cd_sm
        # vllm-perf r3 lever 1: split-K/stream-K fused-GEMM ext (cd_splitk_ext).
        # At M<=16 (decode b8/b16) the serial-K mainloop on N/128..N/64 CTAs is
        # the structural latency floor (r2); split-K partitions the union
        # [NV|MX] k-tile domain across CTAs and reduces once. Routed by the
        # measured _SPLITK_TABLE windows.
        self._cd_sk = cd_sk
        # vllm-perf r4 lever 1: Marlin W4A16 small-M route state (MarlinLow from
        # promix.vllm_integ.marlin_w4a16, attached by the vLLM plugin via
        # attach_marlin). Routed by the measured _MARLIN_TABLE windows; takes
        # priority over the W4A4 GEMM tables inside its window.
        self._marlin = None
        self._marlin_win = None
        # vllm-perf r5 lever 1: MXFP8-marlin high companion (MarlinHigh) + its
        # measured per-shape crossover vs the hi_add GEMV.
        self._marlin_hi = None
        self._marlin_hi_mlo = 8
        # decode-GEMV weight-stream warp count per shape (round-3 Win 4): large N
        # saturates HBM with more warps/block; small N stays launch-bound at 8.
        self._dec_warps = 24 if self.N >= 4096 else 8
        # batched (M>1) decode GEMV rows/block: w32 won the vllm-perf r1 sweep on
        # every shape at M>=2 (amortizes the shared activation staging hardest).
        self._dec_warps_mb = 32
        # Per-shape batched-decode crossover vs the fused tensor-core GEMM,
        # measured by promix/vllm_integ/decode_probe.py on RTX Pro 6000 under
        # CUDA-GRAPH REPLAY timing (the serving regime; eager timing overstates
        # the fused path's launch overhead and picks wrong thresholds — r1
        # lesson). Graph-timed: GEMV wins at M=1 (1.7-2.9x) and M=2 (1.2-2.1x)
        # on EVERY shape; at M=4 only where the margin survives the extra fp32
        # ->fp16 cast (small weights N*K<=17M at 1.35-1.50x, or K-heavy
        # down_proj-like at N*K<=26M where the fused pack scales with K); at
        # M>=8 the fused GEMM always wins. The GEMV cost grows ~linearly in
        # M (LDS/FMA-bound per row) while the fused GEMM is ~flat below its
        # 128-row tile.
        nk = self.N * self.K
        if nk <= 17e6 or (nk <= 26e6 and self.K >= 2 * self.N):
            self._dec_m_max_shape = 4
        else:
            self._dec_m_max_shape = 2
        # Set by wire_real_kernel when cd_bias is wired AND the epilogue-eligible
        # prefill path may use it (i.e. a real fp16 bias exists to fold).
        self.bias_in_epilogue = (cd_bias is not None)
        # A bias-less Linear never needed the extra pass anyway.
        if self.bias is None:
            self.bias_in_epilogue = True
        # vllm-perf r5: cached zero bias for the cast-fold epilogue (the ext
        # would otherwise torch.zeros per call for bias-less layers).
        self._zb = None

    def _bias_or_zero(self, dev):
        if self.bias is not None:
            return self.bias
        if self._zb is None or self._zb.device != dev:
            self._zb = torch.zeros(self.N, dtype=torch.float16, device=dev)
        return self._zb

    @classmethod
    def from_linear(cls, linear, nv, mx, cd, high_fraction=0.125, out_dtype=torch.float16, qpf=None, dg=None, qpfm=None, hpk=None, cd_bias=None, arn=None, hpk28=None, had28=None):
        """Build from a rotated+reordered nn.Linear (or ActQuantWrapper.module)."""
        W = linear.weight.data
        K_low, K_high = split_k(W.shape[1], high_fraction)
        bias = getattr(linear, "bias", None)
        bias = None if bias is None else bias.data
        packed = pack_weight_mixed(W, K_low, K_high, nv, mx, cd, bias=bias)
        return cls(packed, nv, mx, cd, out_dtype=out_dtype, qpf=qpf, dg=dg, qpfm=qpfm, hpk=hpk,
                   cd_bias=cd_bias, arn=arn, hpk28=hpk28, had28=had28)

    def _decode_gemv(self, x2d):
        """Memory-bound M=1 decode path. Two modes (PROMIX_DECODE_MODE):
          * "w4a16" (default): WEIGHT-ONLY — stream the packed 4/8-bit weights at
            HBM BW, dequant in-register, dot against the EXACT fp16 activation. No
            activation quantize (decode is bandwidth-bound, not compute-bound).
          * "w4a4": legacy SINGLE-LAUNCH quant-activation + weight-stream GEMV.
        Both stream identical weight bytes; both are bit-faithful to the fused path
        (same dequantized weight, FP32 accumulate). w4a16 additionally keeps the
        activation exact, so it is strictly more accurate at decode.

        vllm-perf r2 (secondary lever): when S_*_dec hold the checkpoint's RAW
        1-byte scales (uint8: E4M3 low / UE8M0 high), the _sf8 kernel variants
        stream those and fold the per-tensor g in-kernel — bit-exact to the fp32
        buffers (float(E4M3)*g and ldexpf(g, e-127) reproduce the precomputed
        products exactly; sf8_probe torch.equal-gated), ~18% less low-segment
        weight-stream traffic."""
        sf8 = (self.S_lo_dec is not None and self.S_lo_dec.dtype == torch.uint8)
        if decode_mode() == "w4a4":
            if sf8:
                raise RuntimeError(
                    "PROMIX_DECODE_MODE=w4a4 needs fp32 decode scales — set "
                    "PROMIX_DECODE_SF8=0 so the loader derives them")
            D = self._dg.decode_gemv_fused(x2d, self.W_lo_dec, self.S_lo_dec,
                                           self.W_hi_dec, self.S_hi_dec,
                                           self.N, self.K_low, self.K_high,
                                           self._dec_warps)
        elif sf8:
            D = self._dg.decode_gemv_w4a16_sf8(x2d, self.W_lo_dec, self.S_lo_dec,
                                               self.W_hi_dec, self.S_hi_dec,
                                               self.N, self.K_low, self.K_high,
                                               self._dec_warps, self.g_w)
        else:
            D = self._dg.decode_gemv_w4a16(x2d, self.W_lo_dec, self.S_lo_dec,
                                           self.W_hi_dec, self.S_hi_dec,
                                           self.N, self.K_low, self.K_high,
                                           self._dec_warps)
        return D.unsqueeze(0)      # (1, N)

    def _pack_activation(self, x2d):
        """x2d: (M, K) fp (any dtype) -> (A_lo_packed, physSFA_lo, A_hi_bytes, physSFA_hi).

        Pure-NVFP4 (K_high==0): pack ONLY the NVFP4 segment (the whole row) and
        return None for the MXFP8 pair — the single-phase GEMM ignores it. The
        qpf fp8 pack / qpfm mixed pack would launch a 0-block kernel otherwise."""
        M = x2d.shape[0]
        if self.is_pure_nvfp4:
            x2d = x2d.contiguous()
            if self._qpf is not None:
                a_lo = self._qpf.quant_pack_float_fp4(x2d, M, self.N, self.K_low,
                                                      row_stride=self.K, col_off=0)
                return a_lo[0], a_lo[1], None, None
            x_lo = x2d[:, :self.K_low].contiguous()
            codes_lo, SFA_lo, _ = quantize_nvfp4_spec(x_lo.float())
            a_lo = self._cd.quant_pack_fp4(codes_lo, SFA_lo, M, self.N, self.K_low, True)
            return a_lo[0], a_lo[1], None, None
        if self._qpfm is not None:
            # LEVER 1 (pack fusion): one fused launch for BOTH the NVFP4 (low) and
            # MXFP8 (high) segments -> halves the pack launch count. Bit-exact to the
            # two-launch qpf path (same quant math, probe-verified torch.equal).
            x2d = x2d.contiguous()
            a = self._qpfm.quant_pack_float_mixed(x2d, M, self.N, self.K_low, self.K_high)
            return a[0], a[1], a[2], a[3]
        if self._qpf is not None:
            # FAST PATH: fused float->packed-codes + interleaved-SF, one CUDA kernel each,
            # reading the low/high column slices of x2d IN PLACE (row_stride=K, col_off) —
            # no pre-slice or fp32-convert copies. fp16/bf16/fp32 all bit-exact.
            x2d = x2d.contiguous()
            a_lo = self._qpf.quant_pack_float_fp4(x2d, M, self.N, self.K_low,
                                                  row_stride=self.K, col_off=0)
            a_hi = self._qpf.quant_pack_float_fp8(x2d, M, self.N, self.K_high,
                                                  row_stride=self.K, col_off=self.K_low)
            return a_lo[0], a_lo[1], a_hi[0], a_hi[1]
        # SLOW PATH (fallback): torch quantize then fold pack + interleaved SF.
        x_lo = x2d[:, :self.K_low].contiguous()
        x_hi = x2d[:, self.K_low:].contiguous()
        codes_lo, SFA_lo, _ = quantize_nvfp4_spec(x_lo.float())
        bytes_hi, SFA_hi, _ = quantize_mxfp8_spec(x_hi.float())
        a_lo = self._cd.quant_pack_fp4(codes_lo, SFA_lo, M, self.N, self.K_low, True)
        a_hi = self._cd.quant_pack_fp8(bytes_hi, SFA_hi, M, self.N, self.K_high, True)
        return a_lo[0], a_lo[1], a_hi[0], a_hi[1]

    def _smallm_cfg(self, M):
        """vllm-perf r2: pick a cd_smallm_ext config for GEMM-routed M, or None
        for the shipping 128x128 kernel. Measured per shape x M under CUDA-graph
        replay (gemm_probe) — see _SMALLM_TABLE. M is static per captured vLLM
        CUDA graph, so the branch is capture-safe (r1 pattern)."""
        if self._cd_sm is None:
            return None
        win = _SMALLM_TABLE.get((self.N, self.K))
        if win is not None and win[0] <= M <= win[1]:
            return 1    # 128x64 NV4/MX2
        return None

    def _splitk_cfg(self, M):
        """vllm-perf r3: pick a cd_splitk_ext (cfg, splits, decomp) for
        GEMM-routed M, or None. Measured per shape x M under CUDA-graph replay
        (splitk_probe) — see _SPLITK_TABLE. M is static per captured vLLM CUDA
        graph, so the branch is capture-safe (r1 pattern)."""
        if self._cd_sk is None or self.is_pure_nvfp4:
            return None
        mcap = _splitk_env_scale()
        for m_lo, m_hi, cfg, splits, decomp in _SPLITK_TABLE.get((self.N, self.K), ()):
            if m_lo <= M <= min(m_hi, mcap):
                return cfg, splits, decomp
        return None

    def _gemm(self, A_lo, pSFA_lo, A_hi, pSFA_hi, M):
        """Packed-activation -> GEMM. Uses the bias+cast epilogue (lever A) when
        available, else the validated bf16 GEMM + torch cast/bias."""
        if self.is_pure_nvfp4:
            # Pure-NVFP4: single-phase NVFP4 GEMM (Phase-2 skipped at compile time),
            # reusing the fused kernel's tuned NVFP4 Phase-1. g_w rides the epilogue
            # alpha exactly as in the dual path; bias is applied in torch (the
            # single-phase entry has no bias-epilogue variant).
            D = self._cd.single_nvfp4_gemm_prepacked(
                A_lo, pSFA_lo, self.W_lo_packed, self.physSFB_lo,
                M, self.N, self.K_low, self.g_w)
            D = D.to(self.out_dtype)
            if self.bias is not None:
                D = D + self.bias.to(D.dtype)
            return D
        if self.bias_in_epilogue and self._cd_bias is not None and self.bias is not None:
            # A REAL bias: the epilogue fold wins everywhere (removes the cast
            # AND the bias add; r26 lever A, bit-identical by construction).
            return self._cd_bias.fused_gemm_prepacked_bias(
                A_lo, pSFA_lo, self.W_lo_packed, self.physSFB_lo,
                A_hi, pSFA_hi, self.W_hi_bytes, self.physSFB_hi,
                self.bias, M, self.N, self.K_low, self.K_high, self.g_w)
        skc = self._splitk_cfg(M)
        cfg = self._smallm_cfg(M)
        if skc is not None:
            # vllm-perf r3 lever 1: K split across CTAs + one FP32 workspace
            # reduction (deterministic). Same prepacked SF buffers (the SF atom
            # depends only on SFVecSize). Cosine~1 vs serial (reduction order),
            # gated by splitk_probe + T2 decode-route.
            D = self._cd_sk.fused_gemm_prepacked_sk(
                A_lo, pSFA_lo, self.W_lo_packed, self.physSFB_lo,
                A_hi, pSFA_hi, self.W_hi_bytes, self.physSFB_hi,
                M, self.N, self.K_low, self.K_high, self.g_w,
                skc[0], skc[1], skc[2])
        elif cfg is not None:
            # vllm-perf r2 lever 1: small-M tile config (same math, same prepacked
            # SF buffers — the interleaved SF atom depends only on SFVecSize, not
            # tile; torch.equal-gated vs the shipping kernel by gemm_probe).
            D = self._cd_sm.fused_gemm_prepacked_cfg(
                A_lo, pSFA_lo, self.W_lo_packed, self.physSFB_lo,
                A_hi, pSFA_hi, self.W_hi_bytes, self.physSFB_hi,
                M, self.N, self.K_low, self.K_high, self.g_w, cfg)
        else:
            if self._cd_bias is not None:
                # vllm-perf r5 lever 2 (cast fold): outside the r2/r3 windows
                # (= every prefill chunk M>=1024 and the no-window mid-M
                # points) the bias ext's half(float(bf16(acc))+0.0) epilogue
                # stores fp16 directly — BIT-IDENTICAL to the bf16 store +
                # .to(fp16) cast it replaces (r26 lever-A gate), one fewer
                # [M,N] round-trip per Linear (the r5 prefill profile's #1
                # actionable item: 3.3/6.8 ms per 3B/8B L2048 step).
                return self._cd_bias.fused_gemm_prepacked_bias(
                    A_lo, pSFA_lo, self.W_lo_packed, self.physSFB_lo,
                    A_hi, pSFA_hi, self.W_hi_bytes, self.physSFB_hi,
                    self._bias_or_zero(A_lo.device),
                    M, self.N, self.K_low, self.K_high, self.g_w)
            D = self._cd.fused_gemm_prepacked(
                A_lo, pSFA_lo, self.W_lo_packed, self.physSFB_lo,
                A_hi, pSFA_hi, self.W_hi_bytes, self.physSFB_hi,
                M, self.N, self.K_low, self.K_high, self.g_w,
            )
        D = D.to(self.out_dtype)
        if self.bias is not None:
            D = D + self.bias.to(D.dtype)
        return D

    def forward(self, x):
        """x: (..., K) -> (..., N). Flattens leading dims to M for the GEMM."""
        in_shape = x.shape
        x2d = x.reshape(-1, in_shape[-1])          # keep dtype (kernel reads fp16/bf16/fp32)
        M = x2d.shape[0]
        # Pure-NVFP4 (K_high==0) has no decode-GEMV (its weight-high buffers are
        # empty); route M==1 through the single-phase prefill GEMM (correct, just
        # not the bandwidth-optimized decode kernel).
        if self.marlin_route(M):
            # vllm-perf r4 lever 1: Marlin W4A16 tensor-core route (window per
            # measured _MARLIN_TABLE, priority over the GEMV + W4A4 GEMM
            # routes). Weight-only like the GEMV routes (activation consumed
            # exactly); fp16 output already — only bias remains.
            D = self._marlin_forward(x2d.contiguous())
        elif self.gemv_route(M):
            if M == 1:
                D = self._decode_gemv(x2d)         # memory-bound M=1 GEMV (decode)
            else:
                # vllm-perf r1: batched weight-only W4A16 GEMV for captured decode
                # batch sizes (M static per CUDA graph). w4a4 mode has no batched
                # GEMV — it falls through to the fused W4A4 tensor-core GEMM.
                # r2: _sf8 variant when the decode scales are raw 1-byte (see
                # _decode_gemv docstring).
                if self.S_lo_dec.dtype == torch.uint8:
                    D = self._dg.decode_gemv_w4a16_mb_sf8(
                        x2d.contiguous(), self.W_lo_dec, self.S_lo_dec,
                        self.W_hi_dec, self.S_hi_dec,
                        self.N, self.K_low, self.K_high, self._dec_warps_mb, self.g_w)
                else:
                    D = self._dg.decode_gemv_w4a16_mb(
                        x2d.contiguous(), self.W_lo_dec, self.S_lo_dec,
                        self.W_hi_dec, self.S_hi_dec,
                        self.N, self.K_low, self.K_high, self._dec_warps_mb)
        else:
            A_lo, pSFA_lo, A_hi, pSFA_hi = self._pack_activation(x2d)
            return self._gemm(A_lo, pSFA_lo, A_hi, pSFA_hi, M).reshape(*in_shape[:-1], self.N)
        D = D.to(self.out_dtype)
        if self.bias is not None:
            D = D + self.bias.to(D.dtype)
        return D.reshape(*in_shape[:-1], self.N)

    def forward_norm_pack(self, packed):
        """round-26 lever B entry: the activation was produced (residual-add +
        RMSNorm + quantize+pack, ONE launch) by a norm producer, and `packed` is
        the (A_lo, pSFA_lo, A_hi, pSFA_hi) tuple for THIS Linear's N_gemm. Runs
        only the packed fused GEMM (plus the epilogue bias/cast when wired)."""
        A_lo, pSFA_lo, A_hi, pSFA_hi = packed
        M = A_hi.shape[0]
        return self._gemm(A_lo, pSFA_lo, A_hi, pSFA_hi, M)

    def forward_swiglu_pack(self, gate, up):
        """down_proj FUSED entry: gate/up are the two [M, I] bf16 outputs of the
        gate_proj / up_proj GEMMs.  One kernel does silu*up -> Hadamard/sqrt(I)
        (fp32 in SMEM) -> fp16 -> NVFP4+MXFP8 pack; then the packed GEMM.
        Bit-exact to `self.forward(h)` where h = silu(gate)*up (probe-verified),
        minus the fp32 [M, I] HBM round-trips."""
        in_shape = gate.shape
        I = in_shape[-1]
        g2d = gate.reshape(-1, I).contiguous()
        u2d = up.reshape(-1, I).contiguous()
        M = g2d.shape[0]
        assert I == self.K, f"fused down_proj: I={I} != K={self.K}"
        if I == 14336:
            # 8B (Win 3): non-power-of-two width uses the 28x512 fused producer.
            assert self._hpk28 is not None, "I=14336 fused path needs the hpk28 kernel module"
            A_lo, pSFA_lo, A_hi, pSFA_hi = self._hpk28.hadamard28_swiglu_pack_mixed(
                g2d, u2d, self._had28, M, self.N, self.K_low, self.K_high)
        else:
            assert self._hpk is not None, "fused path needs the hpk kernel module"
            A_lo, pSFA_lo, A_hi, pSFA_hi = self._hpk.hadamard_swiglu_pack_mixed(
                g2d, u2d, M, self.N, self.K_low, self.K_high)
        return self._gemm(A_lo, pSFA_lo, A_hi, pSFA_hi, M).reshape(*in_shape[:-1], self.N)

    def gemv_route(self, M):
        """True iff forward(x) at this flattened M takes a decode-GEMV route.
        Used by the vLLM plugin for down_proj: the GEMV consumes an fp16
        activation (so the torch Hadamard must run first), while the GEMM route
        can use the fused Hadamard+pack producer instead."""
        use_dec = (self._dg is not None and not self.is_pure_nvfp4
                   and self.W_lo_dec is not None)
        if not use_dec:
            return False
        if M == 1:
            return True
        return (1 < M <= min(decode_m_max(), self._dec_m_max_shape)
                and decode_mode() == "w4a16"
                and hasattr(self._dg, "decode_gemv_w4a16_mb"))

    def attach_marlin(self, marlin_low, hi_warps=16, marlin_high=None):
        """vllm-perf r4 lever 1: attach the load-time Marlin repack of the low
        segment (promix.vllm_integ.marlin_w4a16.MarlinLow). The high segment
        rides decode_gemv_w8a16_hi_add on the decode-GEMV fp32 scale buffers,
        so the route additionally requires dg + fp32 S_hi_dec (default loader
        config; the opt-in sf8 experiment keeps uint8 scales and skips Marlin).

        r5 lever 1: `marlin_high` (MarlinHigh) is the MXFP8-marlin tensor-core
        high companion; inside the measured _MARLIN_HI_TABLE window it replaces
        the hi_add GEMV (which scales ~linearly in M). None keeps the pure-r4
        route."""
        if (self._dg is None or self.is_pure_nvfp4 or self.S_hi_dec is None
                or self.S_hi_dec.dtype != torch.float32
                or not hasattr(self._dg, "decode_gemv_w8a16_hi_add")):
            return False
        self._marlin = marlin_low
        self._marlin_win = _MARLIN_TABLE.get((self.N, self.K))
        self._marlin_hi_warps = int(hi_warps)
        self._marlin_hi = marlin_high
        self._marlin_hi_mlo = _MARLIN_HI_TABLE.get((self.N, self.K), 8)
        return True

    def _hi_marlin_route(self, M):
        """True iff the high segment at this M rides the MXFP8-marlin companion
        (r5) rather than the hi_add GEMV (r4). Capture-safe (M static per
        captured graph, r1 pattern)."""
        return self._marlin_hi is not None and M >= self._marlin_hi_mlo

    def marlin_route(self, M):
        """True iff forward(x) at this flattened M takes the Marlin W4A16
        tensor-core route (r4 lever 1). Window = measured _MARLIN_TABLE; takes
        PRIORITY over the batched GEMV inside the window (the probe shows
        Marlin beats the M=2..4 GEMV routes 1.2-3.1x — same W4A16 numerics
        class, different accumulation order). M=1 keeps the round-12 single-row
        GEMV (byte-proven path, and Marlin's M-tile padding buys nothing
        there). M is static per captured vLLM CUDA graph, so the branch is
        capture-safe (r1 pattern)."""
        if self._marlin is None or self._marlin_win is None or M == 1:
            return False
        if M > 32 and not self._hi_marlin_route(M):
            return False      # hi_add kernel bound (M<=32)
        m_lo, m_hi = self._marlin_win
        return m_lo <= M <= min(m_hi, _marlin_env_mmax())

    def _marlin_forward(self, x2d):
        """W4A16 small-M route: Marlin fp4 GEMM on the low columns + MXFP8
        high-segment add — the r5 marlin companion (TC, near-flat in M) inside
        its window, the r4 hi_add GEMV below it. x2d: contiguous (M, K)
        fp16/bf16/fp32. Returns fp16 (M, N). The activation is consumed
        EXACTLY (no quantization)."""
        xh = x2d if x2d.dtype == torch.float16 else x2d.to(torch.float16)
        x_lo = xh[:, :self.K_low].contiguous()
        Y = self._marlin.gemm(x_lo)
        if self._hi_marlin_route(xh.shape[0]):
            # bf16 slice-cast: marlin's e8m0 kernels are bf16-activation-only
            # (see MarlinHigh.gemm); the W4A4 reference quantizes this slice
            # to E4M3, so bf16 stays the more-accurate numerics class. The g
            # scale + cast + add ride the ext's fused axpy (one kernel).
            self._marlin_hi.gemm_add(
                Y, xh[:, self.K_low:].to(torch.bfloat16),
                axpy=getattr(self._dg, "axpy_bf16_into_half", None))
        else:
            self._dg.decode_gemv_w8a16_hi_add(
                Y, xh, self.W_hi_dec, self.S_hi_dec,
                self.N, self.K_low, self.K_high, self._marlin_hi_warps)
        return Y

    def fp16_route(self, M):
        """True iff forward(x) at this M consumes the RAW fp16 activation (GEMV
        or Marlin W4A16 routes) rather than the quantize+pack GEMM route. Used
        by the vLLM plugin: fp16 routes need the torch Hadamard (down_proj) /
        torch gather (o_proj) to run first, while the GEMM route uses the fused
        pack producers instead."""
        return self.gemv_route(M) or self.marlin_route(M)

    def can_had_pack(self):
        """True iff forward_had_pack is available for this Linear's K."""
        if self.is_pure_nvfp4:
            return False
        if self.K == 8192:
            return self._hpk is not None and hasattr(self._hpk, "hadamard_pack_mixed")
        if self.K == 14336:
            return (self._hpk28 is not None and self._had28 is not None
                    and hasattr(self._hpk28, "hadamard28_pack_mixed"))
        return False

    def forward_had_pack(self, x):
        """vllm-perf r2 lever 2: down_proj GEMM-route entry — fused online
        Hadamard + NVFP4/MXFP8 quant/pack in ONE kernel (replaces the
        fast_hadamard_transform launch + fp16 [M,K] HBM round-trip + the pack
        launches). The Hadamard runs in fp32 with the reference library's
        fp32-trait stage order (the hpk gate), vs the current torch path's
        fp16-input library call — numerically tighter, gated by cosine + PPL.
        Caller must have checked can_had_pack() and not gemv_route(M)."""
        in_shape = x.shape
        x2d = x.reshape(-1, self.K).contiguous()
        M = x2d.shape[0]
        if self.K == 14336:
            A = self._hpk28.hadamard28_pack_mixed(
                x2d, self._had28, M, self.N, self.K_low, self.K_high)
        else:
            A = self._hpk.hadamard_pack_mixed(x2d, M, self.N, self.K_low, self.K_high)
        return self._gemm(A[0], A[1], A[2], A[3], M).reshape(*in_shape[:-1], self.N)

    def can_gather_pack(self):
        """True iff forward_gather_pack is available (vllm-perf r3 lever 3)."""
        return (self._qpfm is not None and not self.is_pure_nvfp4
                and hasattr(self._qpfm, "quant_pack_float_mixed_gather"))

    def forward_gather_pack(self, x, col_order_i32):
        """o_proj GEMM-route entry (vllm-perf r3 lever 3): the static column
        reorder is folded into the qpfm pack's load (gidx indexing) — one
        kernel instead of a torch gather + pack, no fp16 [M,K] round trip.
        Byte-identical to forward(x[..., col_order]) on the GEMM route
        (gather_pack_probe torch.equal). Caller must have checked
        can_gather_pack() and not gemv_route(M) (the GEMV consumes fp16, so
        it keeps the torch gather)."""
        in_shape = x.shape
        x2d = x.reshape(-1, self.K).contiguous()
        M = x2d.shape[0]
        a = self._qpfm.quant_pack_float_mixed_gather(
            x2d, col_order_i32, M, self.N, self.K_low, self.K_high)
        return self._gemm(a[0], a[1], a[2], a[3], M).reshape(*in_shape[:-1], self.N)

    def pack_only(self, x):
        """Timing seam: run just the per-forward activation quant+pack (no GEMM)."""
        x2d = x.reshape(-1, x.shape[-1])
        return self._pack_activation(x2d)
