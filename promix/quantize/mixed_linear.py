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


class MixedQuantLinear(nn.Module):
    def __init__(self, packed, nv, mx, cd, out_dtype=torch.float16, qpf=None, dg=None):
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
        """
        super().__init__()
        self.N = packed.N
        self.K = packed.K
        self.K_low = packed.K_low
        self.K_high = packed.K_high
        self.out_dtype = out_dtype
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
        self._dg = dg

    @classmethod
    def from_linear(cls, linear, nv, mx, cd, high_fraction=0.125, out_dtype=torch.float16, qpf=None, dg=None):
        """Build from a rotated+reordered nn.Linear (or ActQuantWrapper.module)."""
        W = linear.weight.data
        K_low, K_high = split_k(W.shape[1], high_fraction)
        bias = getattr(linear, "bias", None)
        bias = None if bias is None else bias.data
        packed = pack_weight_mixed(W, K_low, K_high, nv, mx, cd, bias=bias)
        return cls(packed, nv, mx, cd, out_dtype=out_dtype, qpf=qpf, dg=dg)

    def _decode_gemv(self, x2d):
        """Memory-bound SINGLE-LAUNCH W4A4 M=1 path: one kernel quantizes the
        (tiny) activation to the spec grid (bit-exact to fake_quantize_*) and
        streams the packed 4-bit/8-bit weights at HBM BW. Bit-faithful to the
        fused kernel (same dequantized weight, FP32 accumulate)."""
        D = self._dg.decode_gemv_fused(x2d, self.W_lo_dec, self.S_lo_dec,
                                       self.W_hi_dec, self.S_hi_dec,
                                       self.N, self.K_low, self.K_high)
        return D.unsqueeze(0)      # (1, N)

    def _pack_activation(self, x2d):
        """x2d: (M, K) fp (any dtype) -> (A_lo_packed, physSFA_lo, A_hi_bytes, physSFA_hi)."""
        M = x2d.shape[0]
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

    def forward(self, x):
        """x: (..., K) -> (..., N). Flattens leading dims to M for the GEMM."""
        in_shape = x.shape
        x2d = x.reshape(-1, in_shape[-1])          # keep dtype (kernel reads fp16/bf16/fp32)
        M = x2d.shape[0]
        if self._dg is not None and M == 1:
            D = self._decode_gemv(x2d)             # memory-bound W4A4 GEMV (decode)
        else:
            A_lo, pSFA_lo, A_hi, pSFA_hi = self._pack_activation(x2d)
            D = self._cd.fused_gemm_prepacked(
                A_lo, pSFA_lo, self.W_lo_packed, self.physSFB_lo,
                A_hi, pSFA_hi, self.W_hi_bytes, self.physSFB_hi,
                M, self.N, self.K_low, self.K_high,
            )
        D = D.to(self.out_dtype)
        if self.bias is not None:
            D = D + self.bias.to(D.dtype)
        return D.reshape(*in_shape[:-1], self.N)

    def pack_only(self, x):
        """Timing seam: run just the per-forward activation quant+pack (no GEMM)."""
        x2d = x.reshape(-1, x.shape[-1])
        return self._pack_activation(x2d)
