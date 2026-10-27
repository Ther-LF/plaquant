"""Round-10 Task-2 lever #2: Triton fused RoPE (neox/Llama style).

The round-10 per-op profile ranks RoPE the #2 non-GEMM prefill gap after RMSNorm:
HF `apply_rotary_pos_emb` runs `rotate_half` (a cat of two slices -> allocates) plus
several elementwise mul/add kernels, while vLLM's `rotary_embedding` is one in-place
kernel. This computes the IDENTICAL neox rotation in a single Triton kernel.

Llama/neox convention: emb = cat(freqs, freqs); rotate_half(x) = cat(-x2, x1) where
x1,x2 are the first/second halves. So for i < D/2:
  out[i]     = x[i]     * cos[i] - x[i+D/2] * sin[i]
  out[i+D/2] = x[i+D/2] * cos[i] + x[i]     * sin[i]
(cos[i] == cos[i+D/2] since emb duplicates freqs; we index the half-width table.)

`apply_rope_triton(q, k, cos, sin)` matches HF `apply_rotary_pos_emb(q,k,cos,sin)`:
q,k are [B,H,S,D], cos/sin are [B,S,D] (the full-width duplicated tables, as HF
builds them). We read only the first half of cos/sin per position.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(X, COS, SIN, OUT, n_heads, seq, D, HALF,
                 xs_b, xs_h, xs_s, cs_b, cs_s, BLOCK: tl.constexpr):
    row = tl.program_id(0)          # flattened (b, h, s)
    bh = row // seq
    s = row % seq
    b = bh // n_heads
    x_off = b * xs_b + (bh % n_heads) * xs_h + s * xs_s
    c_off = b * cs_b + s * cs_s
    cols = tl.arange(0, BLOCK)
    mask = cols < HALF
    x1 = tl.load(X + x_off + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X + x_off + HALF + cols, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(COS + c_off + cols, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(SIN + c_off + cols, mask=mask, other=0.0).to(tl.float32)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    tl.store(OUT + x_off + cols, o1.to(OUT.dtype.element_ty), mask=mask)
    tl.store(OUT + x_off + HALF + cols, o2.to(OUT.dtype.element_ty), mask=mask)


def _rope_one(x, cos, sin):
    B, H, S, D = x.shape
    HALF = D // 2
    out = torch.empty_like(x)
    xs = x.stride()
    cs = cos.stride()  # cos is [Bc,S,D] with Bc in {1, B}
    cs_b = cs[0] if cos.shape[0] == x.shape[0] else 0   # broadcast over batch if Bc==1
    BLOCK = triton.next_power_of_2(HALF)
    _rope_kernel[(B * H * S,)](
        x, cos, sin, out, H, S, D, HALF,
        xs[0], xs[1], xs[2], cs_b, cs[1], BLOCK=BLOCK,
        num_warps=4)
    return out


def apply_rope_triton(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Drop-in for transformers...apply_rotary_pos_emb. cos/sin arrive as [B,S,D]."""
    # cos/sin may be [B,S,D]; ensure contiguous for stride math
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()
    q_e = _rope_one(q.contiguous(), cos_c, sin_c)
    k_e = _rope_one(k.contiguous(), cos_c, sin_c)
    return q_e, k_e


def swap_rope():
    """Monkeypatch HF Llama apply_rotary_pos_emb -> Triton. Returns the original."""
    import transformers.models.llama.modeling_llama as ml
    orig = ml.apply_rotary_pos_emb
    ml.apply_rotary_pos_emb = apply_rope_triton
    return orig


def restore_rope(orig):
    import transformers.models.llama.modeling_llama as ml
    ml.apply_rotary_pos_emb = orig
