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


class MixedQuantLinear(nn.Module):
    def __init__(self, packed, nv, mx, cd, out_dtype=torch.float16, qpf=None, dg=None,
                 qpfm=None, hpk=None, cd_bias=None, arn=None, hpk28=None, had28=None):
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
        activation exact, so it is strictly more accurate at decode."""
        if decode_mode() == "w4a4":
            D = self._dg.decode_gemv_fused(x2d, self.W_lo_dec, self.S_lo_dec,
                                           self.W_hi_dec, self.S_hi_dec,
                                           self.N, self.K_low, self.K_high,
                                           self._dec_warps)
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
        if self.bias_in_epilogue and self._cd_bias is not None:
            return self._cd_bias.fused_gemm_prepacked_bias(
                A_lo, pSFA_lo, self.W_lo_packed, self.physSFB_lo,
                A_hi, pSFA_hi, self.W_hi_bytes, self.physSFB_hi,
                self.bias, M, self.N, self.K_low, self.K_high, self.g_w)
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
        use_dec = (self._dg is not None and not self.is_pure_nvfp4
                   and self.W_lo_dec is not None)
        if use_dec and M == 1:
            D = self._decode_gemv(x2d)             # memory-bound M=1 GEMV (decode)
        elif (use_dec and 1 < M <= min(decode_m_max(), self._dec_m_max_shape)
              and decode_mode() == "w4a16"
              and hasattr(self._dg, "decode_gemv_w4a16_mb")):
            # vllm-perf r1: batched weight-only W4A16 GEMV for captured decode
            # batch sizes (M static per CUDA graph). w4a4 mode has no batched
            # GEMV — it falls through to the fused W4A4 tensor-core GEMM.
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

    def pack_only(self, x):
        """Timing seam: run just the per-forward activation quant+pack (no GEMM)."""
        x2d = x.reshape(-1, x.shape[-1])
        return self._pack_activation(x2d)
