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


class MixedQuantLinear(nn.Module):
    def __init__(self, packed, nv, mx, cd, out_dtype=torch.float16):
        """
        Args:
            packed: PackedWeight from pack_weight_mixed.
            nv, mx, cd: kernel python modules.
            out_dtype: dtype of the returned activation (default fp16).
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
        # kernel modules (not buffers)
        self._nv = nv
        self._mx = mx
        self._cd = cd

    @classmethod
    def from_linear(cls, linear, nv, mx, cd, high_fraction=0.125, out_dtype=torch.float16):
        """Build from a rotated+reordered nn.Linear (or ActQuantWrapper.module)."""
        W = linear.weight.data
        K_low, K_high = split_k(W.shape[1], high_fraction)
        bias = getattr(linear, "bias", None)
        bias = None if bias is None else bias.data
        packed = pack_weight_mixed(W, K_low, K_high, nv, mx, cd, bias=bias)
        return cls(packed, nv, mx, cd, out_dtype=out_dtype)

    def _pack_activation(self, x2d):
        """x2d: (M, K) fp -> (A_lo_packed, physSFA_lo, A_hi_bytes, physSFA_hi)."""
        M = x2d.shape[0]
        x_lo = x2d[:, :self.K_low].contiguous()
        x_hi = x2d[:, self.K_low:].contiguous()
        codes_lo, SFA_lo, _ = quantize_nvfp4_spec(x_lo)
        bytes_hi, SFA_hi, _ = quantize_mxfp8_spec(x_hi)
        # fold data-pack + interleaved-SF layout in one pass (ilv=True)
        a_lo = self._cd.quant_pack_fp4(codes_lo, SFA_lo, M, self.N, self.K_low, True)
        a_hi = self._cd.quant_pack_fp8(bytes_hi, SFA_hi, M, self.N, self.K_high, True)
        return a_lo[0], a_lo[1], a_hi[0], a_hi[1]

    def forward(self, x):
        """x: (..., K) -> (..., N). Flattens leading dims to M for the GEMM."""
        in_shape = x.shape
        x2d = x.reshape(-1, in_shape[-1]).to(torch.float32)
        M = x2d.shape[0]
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
        x2d = x.reshape(-1, x.shape[-1]).to(torch.float32)
        return self._pack_activation(x2d)
