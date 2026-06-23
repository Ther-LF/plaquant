"""Core quantization utilities — ActQuantizer, ActQuantWrapper, STE functions.

Migrated from project-resq, stripped of GPTQ/weight quant/VLM code.
"""

import math

import torch
import torch.nn as nn

from promix.quantize.hadamard import get_hadK, matmul_hadU_cuda
from promix.utils import HadamardTransform


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0
    return minq, maxq


# Canonical strings ActQuantizer / ActQuantWrapper / WeightQuantizer accept
# in place of numeric `bits`. Adding a new FP block-scaled format means:
# (1) add the string here, (2) implement `fake_quantize_<name>`, (3) extend
# the dispatch in ActQuantizer.forward and WeightQuantizer.configure.
SUPPORTED_FP_FORMATS = ("mxfp8", "nvfp4")


def _quant_enabled(bits):
    """Predicate: is this `bits` value a request to actually quantize?

    Used at every call site that previously did `bits < 16` (which TypeErrors
    on string FP format identifiers). For numeric, true when below 16-bit
    pass-through. For string, true when the format is in SUPPORTED_FP_FORMATS.
    """
    if isinstance(bits, str):
        return bits in SUPPORTED_FP_FORMATS
    return bits < 16


def _apply_fp_format(x, fmt):
    """Dispatch a tensor segment through the spec-derived fake quantizer.

    Used by ActQuantizer.forward when bits / high_bits / low_bits are string
    FP format identifiers. Block-alignment is checked inside the underlying
    fake_quantize_* helper (it asserts shape[-1] % block_size == 0 with a
    clear error message).
    """
    if fmt == "mxfp8":
        return fake_quantize_mxfp8(x)
    if fmt == "nvfp4":
        return fake_quantize_nvfp4(x)
    raise ValueError(f"Unsupported FP format identifier: {fmt!r}")


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq):
        scale = scale.to(x.device)
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class AsymSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maxq):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class ActQuantizer(nn.Module):
    """Per-token activation quantizer with mixed-precision group support."""

    def __init__(self):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("maxq_h", torch.tensor(0))
        self.register_buffer("scale_h", torch.zeros(1))
        self.register_buffer("zero_h", torch.zeros(1))
        self.register_buffer("maxq_l", torch.tensor(0))
        self.register_buffer("scale_l", torch.zeros(1))
        self.register_buffer("zero_l", torch.zeros(1))

        self.bits = 16
        self.high_bits = 16
        self.low_bits = 16
        self.high_bits_length = 0
        self.low_bits_length = 0

    def free(self):
        self.zero = self.scale = None
        self.zero_h = self.scale_h = None
        self.zero_l = self.scale_l = None

    def configure(self, bits, groupsize=-1, sym=False, clip_ratio=1.0,
                  high_bits_length=0, high_bits=16, low_bits_length=0, low_bits=16, **kwargs):
        # FP block-scaled formats (per spec docs/specs/spec-mxfp8-nvfp4.md, plan
        # task9): when `bits` is the string "mxfp8" or "nvfp4", the forward
        # path dispatches to the spec-derived fake_quantize_* helpers and
        # skips the INT scale/zero/maxq machinery entirely (block scales are
        # computed internally per call).
        if isinstance(bits, str):
            self.bits = bits
        else:
            _, self.maxq = get_minq_maxq(bits, sym)
            self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        self.high_bits_length = high_bits_length
        self.high_bits = high_bits
        if isinstance(high_bits, str):
            pass  # no maxq_h needed for FP path
        else:
            _, self.maxq_h = get_minq_maxq(high_bits, sym)
        self.low_bits_length = low_bits_length
        self.low_bits = low_bits
        if isinstance(low_bits, str):
            pass  # no maxq_l needed for FP path
        else:
            _, self.maxq_l = get_minq_maxq(low_bits, sym)

    def forward(self, x):
        if self.bits == 16:
            return x

        # FP block-scaled fake quantization. String `bits` selects the
        # spec-derived helper; INT scale/zero/maxq state is ignored. When
        # high_bits / low_bits are also configured (mixed MXFP8+NVFP4),
        # apply the same [low | main | high] segment split as the INT
        # branch but route each segment to its own FP format.
        # Block-alignment of each segment length is asserted inside the
        # underlying helper (MXFP8 needs %32, NVFP4 needs %16).
        if isinstance(self.bits, str):
            x_dtype = x.dtype
            high_len = int(self.high_bits_length)
            low_len = int(self.low_bits_length)
            high_fmt = self.high_bits if isinstance(self.high_bits, str) else None
            low_fmt = self.low_bits if isinstance(self.low_bits, str) else None
            main_fmt = self.bits

            # Single-format path (no high/low split): quantize whole tensor.
            if high_len == 0 and low_len == 0:
                return _apply_fp_format(x, main_fmt).to(x_dtype)

            # Mixed: split [low_seg | main_seg | high_seg] along last dim.
            high_dim = x.shape[-1] - high_len
            x_l = x[..., :low_len] if low_len > 0 else None
            x_m = x[..., low_len:high_dim]
            x_h = x[..., high_dim:] if high_len > 0 else None

            parts = []
            if x_l is not None and low_fmt is not None:
                parts.append(_apply_fp_format(x_l, low_fmt))
            elif x_l is not None:
                # low segment configured numerically; not currently supported
                # alongside string main format. Surface a clear error rather
                # than silently mismatching.
                raise ValueError(
                    "Mixed FP/INT segment routing not supported: low_bits must be a "
                    "string FP format when bits is a string, got "
                    f"low_bits={self.low_bits!r}"
                )

            parts.append(_apply_fp_format(x_m, main_fmt))

            if x_h is not None and high_fmt is not None:
                parts.append(_apply_fp_format(x_h, high_fmt))
            elif x_h is not None:
                raise ValueError(
                    "Mixed FP/INT segment routing not supported: high_bits must be a "
                    "string FP format when bits is a string, got "
                    f"high_bits={self.high_bits!r}"
                )

            return torch.cat(parts, dim=-1).to(x_dtype)

        x_dtype = x.dtype

        if self.groupsize > 0:
            init_shape = x.shape
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // self.groupsize, self.groupsize)

        low_dim = self.low_bits_length
        high_dim = x.shape[-1] - self.high_bits_length
        x_l, x_m, x_h = x[..., :low_dim], x[..., low_dim:high_dim], x[..., high_dim:]

        if self.sym:
            x = STEQuantize.apply(x_m, self.scale, self.maxq)
            if self.high_bits_length != 0:
                x_h = STEQuantize.apply(x_h, self.scale_h, self.maxq_h)
                x = torch.cat([x, x_h], dim=-1).to(x_dtype)
            if self.low_bits_length != 0:
                x_l = STEQuantize.apply(x_l, self.scale_l, self.maxq_l)
                x = torch.cat([x_l, x], dim=-1).to(x_dtype)
        else:
            x = AsymSTEQuantize.apply(x_m, self.scale, self.zero, self.maxq)
            if self.high_bits_length != 0:
                x_h = AsymSTEQuantize.apply(x_h, self.scale_h, self.zero_h, self.maxq_h)
                x = torch.cat([x, x_h], dim=-1).to(x_dtype)
            if self.low_bits_length != 0:
                x_l = AsymSTEQuantize.apply(x_l, self.scale_l, self.zero_l, self.maxq_l)
                x = torch.cat([x_l, x], dim=-1).to(x_dtype)

        if self.groupsize > 0:
            x = x.reshape(init_shape)
        return x

    def find_params(self, x):
        # FP block-scaled paths compute their block scales inside the forward
        # call (per spec Section 6); there are no per-token / per-group scale
        # parameters to fit ahead of time, so find_params is a no-op when
        # bits is a string FP format identifier.
        if isinstance(self.bits, str):
            return

        if self.groupsize > 0:
            x_reshaped = x.reshape(
                x.shape[0], x.shape[1], x.shape[2] // self.groupsize, self.groupsize
            )
            low_dim = self.low_bits_length
            high_dim = x_reshaped.shape[-1] - self.high_bits_length
            x_l, x_m, x_h = x_reshaped[..., :low_dim], x_reshaped[..., low_dim:high_dim], x_reshaped[..., high_dim:]
            self.scale, self.zero = self._find_params_groupwise(x_m, self.maxq)
            if self.high_bits_length != 0:
                self.scale_h, self.zero_h = self._find_params_groupwise(x_h, self.maxq_h)
            if self.low_bits_length != 0:
                self.scale_l, self.zero_l = self._find_params_groupwise(x_l, self.maxq_l)
            return

        low_dim = self.low_bits_length
        high_dim = x.shape[-1] - self.high_bits_length
        x_l, x_m, x_h = x[..., :low_dim], x[..., low_dim:high_dim], x[..., high_dim:]
        self.scale, self.zero = self._find_params_per_token(x_m, self.maxq)
        if self.high_bits_length != 0:
            self.scale_h, self.zero_h = self._find_params_per_token(x_h, self.maxq_h)
        if self.low_bits_length != 0:
            self.scale_l, self.zero_l = self._find_params_per_token(x_l, self.maxq_l)

    def _find_params_per_token(self, x, maxq):
        """Per-token quantization params: scale/zero shape = (batch*seq, 1) broadcast."""
        dev = x.device
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))
        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = (xmax / maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            scale[tmp] = 1
            scale = scale.reshape(init_shape)
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)
            scale = scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            zero = zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

        return scale, zero

    def _find_params_groupwise(self, x, maxq):
        """Per-group quantization params (dim=3)."""
        xmax = torch.amax(x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = xmax / maxq
            scale[tmp] = 1
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)
        return scale, zero


class ActQuantWrapper(nn.Module):
    """Wraps a Linear layer with activation quantization + optional Hadamard rotation."""

    def __init__(self, module: nn.Linear):
        super().__init__()
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer("had_K", None)
        self.K = 1
        self.online_full_had = False
        self.fp32_had = False
        # Cache which extra kwargs the inner module's forward accepts so we
        # can route training-time R1/R2 args through wrappers around
        # QuantizeLinear without erroring on plain nn.Linear (e.g. lm_head).
        try:
            import inspect as _inspect
            self._inner_kwargs = set(_inspect.signature(module.forward).parameters.keys())
        except (ValueError, TypeError):
            self._inner_kwargs = set()

    def forward(self, x, column_order=None, **kwargs):
        x_dtype = x.dtype

        # Disambiguate the second positional arg: PTQ paths pass an int index
        # tensor (column_order); training paths in modeling_llama_train pass
        # a float rotation matrix R1 that needs to flow through to the inner
        # QuantizeLinear. Same param name kept for backward compat.
        col_order = None
        if column_order is not None:
            if column_order.dtype in (torch.long, torch.int, torch.int32, torch.int64, torch.uint8, torch.bool):
                col_order = column_order
            else:
                kwargs.setdefault("R1", column_order)

        if self.online_full_had:
            if self.fp32_had:
                x = matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:
                x = matmul_hadU_cuda(x, self.had_K, self.K)

        if _quant_enabled(self.quantizer.bits):
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        order = col_order if col_order is not None else getattr(self, '_column_order', None)
        if order is not None:
            x = x[..., order]

        # Filter kwargs to those the inner module accepts (lm_head etc. is
        # plain nn.Linear that doesn't take R1/R2).
        inner = {k: v for k, v in kwargs.items() if k in self._inner_kwargs}
        x = self.module(x, **inner).to(x_dtype)

        if _quant_enabled(self.out_quantizer.bits):
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


def add_actquant(module, name=""):
    """Recursively wrap all Linear layers with ActQuantWrapper."""
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if isinstance(tmp, nn.Linear):
            setattr(module, attr, ActQuantWrapper(tmp))
    for name1, child in module.named_children():
        add_actquant(child, name + "." + name1 if name != "" else name1)


def find_qlayers(module, layers=None, name=""):
    """Find all ActQuantWrapper (or specified) layers in a model."""
    if layers is None:
        layers = [nn.Linear, ActQuantWrapper]
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(child, layers, name + "." + name1 if name != "" else name1))
    return res


# ---------------------------------------------------------------------------
# MXFP8 / NVFP4 fake quantizers
# ---------------------------------------------------------------------------
#
# Pure PyTorch reference implementations of the block-scaled FP format spec
# at docs/specs/spec-mxfp8-nvfp4.md. Source of truth for fake quantization;
# the kernel epilogue / real packer must produce bit-identical outputs on
# the same input (verified by the fake-vs-real equivalence test).
#
# Both functions are quantize-then-dequantize ("fake quant"): the returned
# tensor has the same dtype/shape as the input but its values are constrained
# to what the format would represent after a real round-trip. This is what
# rotation training (optimize_rotation.py) and PPL eval need.
#
# Block scale rounding direction: UP (toward larger magnitude; smallest
# representable scale >= ideal). See spec Section 6 for rationale; round-DOWN
# would silently saturate outliers.

# FP8 E4M3: the 14 representable POSITIVE values (+0 and +finites; we mirror
# for negatives at quantize time). 256 total codes, NaN at the boundary.
# Source: NVIDIA / OCP FP8 spec.
_FP8_E4M3_POS = (
    [0.0]
    + [
        # subnormals (exp=0): 1, 2, 3, 4, 5, 6, 7 in mantissa -> 2^-9 .. 7*2^-9
        2.0 ** -9 * m for m in range(1, 8)
    ]
    + [
        # normals (exp=1..15, mantissa 0..7) but skip 1 NaN code (last bin)
        (2.0 ** (e - 7)) * (1.0 + m / 8.0)
        for e in range(1, 16)
        for m in range(8)
    ]
)
# Drop NaN code (last entry) and dedupe; max finite is 448
_FP8_E4M3_POS = sorted(set(v for v in _FP8_E4M3_POS if v <= 448.0))
FP8_E4M3_MAX = 448.0

# FP4 E2M1: 8 representable positive values {0, 0.5, 1, 1.5, 2, 3, 4, 6}
FP4_E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_E2M1_MAX = 6.0


def _round_to_nearest_value(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Round each element of x to the nearest value in `values` (1D, sorted ascending).

    Implements round-to-nearest-even per spec Section 6: at exact midpoints between
    two adjacent representable values, choose the one with the EVEN-indexed encoding
    (i.e., the smaller index, since encoded code 0..N-1; "even" = index%2==0). For
    formats whose representable set is enumerated in monotonically increasing order
    (FP4 E2M1 positive, FP8 E4M3 positive), this corresponds to "round to the
    representable whose encoded value has a 0 in its LSB", matching IEEE 754
    round-half-to-even semantics for these encodings.

    Spec Section 6 reference: ties-to-even at element quantization step.
    """
    # values: (V,) sorted ascending; x can be any shape
    values = values.to(x.device, dtype=x.dtype)
    sign = torch.sign(x)
    absx = x.abs()

    # For each absx, find the two adjacent representables that bracket it:
    # idx_lo such that values[idx_lo] <= absx <= values[idx_lo+1].
    # Use torch.searchsorted (returns position where absx would insert) and clamp.
    insert = torch.searchsorted(values, absx, right=False)  # (..., )
    insert = insert.clamp(min=0, max=values.numel() - 1)
    # Lower-bound index: largest representable <= absx (or smallest if absx <
    # values[0]). Upper-bound: insert clamped to V-1.
    idx_hi = insert.clamp(min=1, max=values.numel() - 1)
    idx_lo = (idx_hi - 1).clamp(min=0)
    # If absx is exactly a representable, both idx_lo and idx_hi may bracket it;
    # the standard "nearest" logic still picks correctly because diff_lo==0.

    val_lo = values[idx_lo]
    val_hi = values[idx_hi]
    diff_lo = (absx - val_lo).abs()
    diff_hi = (val_hi - absx).abs()

    # Default: pick the closer one
    pick_hi = diff_hi < diff_lo

    # Tie-break: at exact midpoint (diff_lo == diff_hi), pick the one whose
    # encoded index is EVEN. Index encoding: idx_lo and idx_hi are the
    # representable-set positions; even-indexed code = idx % 2 == 0.
    is_tie = (diff_hi - diff_lo).abs() < 1e-12  # exact midpoint up to fp precision
    even_idx_is_lo = (idx_lo % 2) == 0
    # When tie: pick lo if lo's index is even, otherwise pick hi (which is
    # idx_lo+1 and therefore the other parity — guaranteed even when lo is odd).
    pick_hi = torch.where(is_tie, ~even_idx_is_lo, pick_hi)

    chosen_idx = torch.where(pick_hi, idx_hi, idx_lo)
    return sign * values[chosen_idx]


def _ceil_to_value(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Round each element of x UP to the smallest value in `values` >= x.

    For positive x. Used for block-scale rounding (round UP per spec Section 6).
    Falls back to max(values) if x exceeds any representable value.
    """
    values = values.to(x.device, dtype=x.dtype)
    # idx of smallest value >= x; default to len(values)-1 if none satisfies
    cmp = values.unsqueeze(0) >= x.unsqueeze(-1)  # (..., V)
    has_geq = cmp.any(dim=-1)
    # set values that aren't >= x to a large sentinel so argmin picks the right one
    sentinel = torch.full_like(cmp, False)
    masked = torch.where(cmp, values.expand_as(cmp), torch.full_like(cmp, float("inf"), dtype=values.dtype))
    chosen = masked.min(dim=-1).values
    chosen = torch.where(has_geq, chosen, values.max())
    return chosen


def _e8m0_round_up(scale: torch.Tensor) -> torch.Tensor:
    """Round positive scale UP to nearest power-of-2 (E8M0 representation).

    E8M0 has 8-bit unsigned exponent, no mantissa, no sign. Representable
    values: 2**(exp - 127) for exp in 0..255. Rounding UP means
    chosen = 2**ceil(log2(scale)).
    """
    # Floor(log2(scale)) gives the largest power-of-2 <= scale; we want the
    # smallest power-of-2 >= scale, which is ceil(log2(scale)) unless scale is
    # already a power of 2.
    log2 = torch.log2(scale.clamp_min(torch.finfo(scale.dtype).tiny))
    exp_up = torch.ceil(log2)
    chosen = torch.pow(torch.tensor(2.0, device=scale.device, dtype=scale.dtype), exp_up)
    # Edge case: scale == 0 -> chosen = smallest representable (2^-127)
    chosen = torch.where(
        scale > 0,
        chosen,
        torch.tensor(2.0 ** -127, device=scale.device, dtype=scale.dtype),
    )
    return chosen


def fake_quantize_mxfp8(
    x: torch.Tensor,
    block_size: int = 32,
    elem_format: str = "e4m3",
) -> torch.Tensor:
    """Fake-quantize x as MXFP8 (block_size=32, FP8 E8M0 block scale, FP8 E4M3 elements).

    Per spec Section 6:
    - block scale = max_abs(block) / max_format_value, rounded UP to nearest power-of-2
    - elements = round-to-nearest representable FP8 value after dividing by block scale
    - return: dequantized values (same shape/dtype as input)

    Args:
        x: input tensor; quantization is along the last dim, which is treated as
           the K (contraction) axis. Shape (..., K) where K must be divisible
           by block_size.
        block_size: 32 for MXFP8.
        elem_format: "e4m3" (default) or "e5m2" (not implemented in this round).

    Returns:
        Tensor of same shape and dtype as x with values constrained to the
        MXFP8 representable set.
    """
    if elem_format != "e4m3":
        raise NotImplementedError("MXFP8 E4M3 only; E5M2 is a future option.")
    assert x.shape[-1] % block_size == 0, (
        f"MXFP8 quantization requires K ({x.shape[-1]}) divisible by block_size ({block_size})"
    )
    orig_dtype = x.dtype
    x = x.to(torch.float32)

    # Reshape last dim into (num_blocks, block_size)
    *prefix, k = x.shape
    num_blocks = k // block_size
    x_blk = x.view(*prefix, num_blocks, block_size)

    # Block scale: max abs / FP8_E4M3_MAX, then round UP to power-of-2
    block_max = x_blk.abs().amax(dim=-1, keepdim=True).clamp_min(0.0)
    ideal_scale = block_max / FP8_E4M3_MAX
    block_scale = _e8m0_round_up(ideal_scale.clamp_min(2.0 ** -127))

    # Element quantization: divide by block scale, round to nearest FP8 E4M3 value
    fp8_values = torch.tensor(_FP8_E4M3_POS, device=x.device, dtype=torch.float32)
    x_scaled = x_blk / block_scale
    x_scaled = x_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    q = _round_to_nearest_value(x_scaled, fp8_values)
    deq = q * block_scale
    return deq.view(*prefix, k).to(orig_dtype)


def fake_quantize_nvfp4(
    x: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """Fake-quantize x as NVFP4 (block_size=16, FP8 E4M3 block scale, FP4 E2M1 elements).

    Per spec Section 6:
    - block scale = max_abs(block) / max_format_value, rounded UP to nearest representable FP8 E4M3
    - elements = round-to-nearest representable FP4 E2M1 value after dividing by block scale
    - return: dequantized values

    Args:
        x: input tensor; quantization along the last dim. K must be divisible by 16.
        block_size: 16 for NVFP4.

    Returns:
        Tensor of same shape/dtype as x with values constrained to the
        NVFP4 representable set after the dual-scale composition.
    """
    assert x.shape[-1] % block_size == 0, (
        f"NVFP4 quantization requires K ({x.shape[-1]}) divisible by block_size ({block_size})"
    )
    orig_dtype = x.dtype
    x = x.to(torch.float32)

    *prefix, k = x.shape
    num_blocks = k // block_size
    x_blk = x.view(*prefix, num_blocks, block_size)

    # Block scale: max abs / FP4_E2M1_MAX, round UP to nearest FP8 E4M3 representable
    block_max = x_blk.abs().amax(dim=-1, keepdim=True)
    ideal_scale = (block_max / FP4_E2M1_MAX).clamp_min(2.0 ** -9)
    fp8_e4m3_values = torch.tensor(_FP8_E4M3_POS, device=x.device, dtype=torch.float32)
    block_scale = _ceil_to_value(ideal_scale, fp8_e4m3_values)

    # Element: round to nearest FP4 E2M1 representable value
    fp4_values_pos = torch.tensor(FP4_E2M1_POS, device=x.device, dtype=torch.float32)
    x_scaled = x_blk / block_scale
    x_scaled = x_scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
    q = _round_to_nearest_value(x_scaled, fp4_values_pos)
    deq = q * block_scale
    return deq.view(*prefix, k).to(orig_dtype)
