"""MXFP8 (W8A8) quantized lm_head for the SM120 prefill path (lever 2).

lm_head is a shared ~10 ms FP16 GEMM (vocab 128256 x hidden 2048) that both the
real-kernel and FP16 models pay identically. W4A4-mixed on the un-rotated lm_head
input regresses wikitext-2 PPL +0.295 (fails the <=14.95 gate); W8 MXFP8-only
holds at +0.048 (14.898), so lm_head ships as pure MXFP8 (block-32 E4M3 elements,
E8M0 power-of-2 block scale), run on the SM120 block-scaled tensor cores (~2x FP16
throughput) instead of FP16 cuBLAS.

Per forward: pack the activation to MXFP8 codes + interleaved SFA (one
quant_pack_float_fp8 launch), then one pure-MXFP8 prepacked GEMM
(lmhead_mxfp8_ext.mxfp8_gemm_prepacked, activation + weight SF pre-interleaved).
Bit-faithful to the fake W8 twin (cosine gate >= 0.9999).
"""
import torch
import torch.nn as nn

from promix.quantize.weight_pack import quantize_mxfp8_spec


class MxFp8Linear(nn.Module):
    def __init__(self, W, lmh, qpf, bias=None, out_dtype=torch.float16):
        """W: (N, K) fp weight. lmh: lmhead_mxfp8_ext module. qpf:
        quant_pack_float_ext module (for the per-forward activation pack)."""
        super().__init__()
        self.N, self.K = W.shape
        assert self.K % 32 == 0, f"K={self.K} not MXFP8 block-32 aligned"
        self.out_dtype = out_dtype
        self._lmh = lmh
        self._qpf = qpf
        Wf = W.detach().to(torch.float32).contiguous()
        bytes_hi, SFB_hi, W_deq = quantize_mxfp8_spec(Wf)
        physSFB = lmh.prepack_sf_fp8_b(SFB_hi.contiguous(), self.N, self.K)
        self.register_buffer("W_bytes", bytes_hi.contiguous())
        self.register_buffer("physSFB", physSFB.contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().contiguous())
        # kept for the twin/cosine gate (not used in forward)
        self.register_buffer("W_deq", W_deq.contiguous())

    @classmethod
    def from_linear(cls, linear, lmh, qpf, out_dtype=torch.float16):
        b = getattr(linear, "bias", None)
        b = None if b is None else b.data
        return cls(linear.weight.data, lmh, qpf, bias=b, out_dtype=out_dtype)

    def forward(self, x):
        in_shape = x.shape
        x2d = x.reshape(-1, in_shape[-1]).contiguous()
        M = x2d.shape[0]
        A, SFA = self._qpf.quant_pack_float_fp8(x2d, M, self.N, self.K, -1, 0)
        D = self._lmh.mxfp8_gemm_prepacked(A, SFA, self.W_bytes, self.physSFB,
                                           M, self.N, self.K)
        D = D.to(self.out_dtype)
        if self.bias is not None:
            D = D + self.bias.to(D.dtype)
        return D.reshape(*in_shape[:-1], self.N)
