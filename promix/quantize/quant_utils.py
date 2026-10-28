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


def _quant_enabled(bits, *, strict: bool = False):
    """Predicate: is this `bits` value a request to actually quantize?

    Used at every call site that previously did `bits < 16` (which
    TypeErrors on string FP format identifiers). For numeric, true
    when below 16-bit pass-through. For string, true when the format
    is in `SUPPORTED_FP_FORMATS`.

    Default behavior (strict=False) is permissive: an unknown string
    silently returns False (i.e. quantization is disabled). That has
    been the contract since round 3 and many call sites rely on it
    failing-closed.

    `strict=True` raises `ValueError` on unknown strings, which is the
    right behavior for config-loading entry points where a typo like
    `"nvf4"` would silently produce an FP16 model labelled as
    quantized. Use `assert_quant_format(bits)` at config-load surfaces.
    """
    if isinstance(bits, str):
        if bits in SUPPORTED_FP_FORMATS:
            return True
        if strict:
            raise ValueError(
                f"unknown FP quantization format {bits!r}; expected one "
                f"of {SUPPORTED_FP_FORMATS} (or a numeric INT bit count "
                f"< 16). Common typo: 'nvf4' -> 'nvfp4'."
            )
        return False
    return bits < 16


def assert_quant_format(bits):
    """Validate a `bits` field at config-load time.

    Raises ValueError on unknown strings. Numeric values pass through
    silently (any int / bool / float behaves as in `_quant_enabled`).
    Use this at config-facing surfaces (`configure_quantizers`,
    `gptq_fwrd`) so a typo in `w_bits` / `a_bits` / `high_bits` /
    `low_bits` surfaces loudly instead of silently disabling
    quantization and producing an FP16 model labelled as quantized.
    """
    _quant_enabled(bits, strict=True)
    return True


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
        # FP block-scaled formats (per spec docs/specs/spec-mxfp8-nvfp4.md):
        # when `bits` is the string "mxfp8" or "nvfp4", the forward path
        # dispatches to the spec-derived fake_quantize_* helpers and skips
        # the INT scale/zero/maxq machinery entirely (block scales are
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
        # Prefill-fusion levers (set by full_network.wire_real_kernel for the
        # real packed-kernel path; default off keeps the fake-quant / PPL-twin
        # path byte-identical):
        #   fold_had_scale: fold the online Hadamard's 1/sqrt(n) into the
        #       transform kernel (kills the separate div kernel).
        #   had_out_fp32: keep the FP32 Hadamard output in FP32 and feed it
        #       STRAIGHT to the inner module (MixedQuantLinear's qpf pack reads
        #       fp32 directly), skipping the fp32->fp16 round-trip cast.
        self.fold_had_scale = False
        self.had_out_fp32 = False
        # FP per-head o_proj: apply the column reorder BEFORE quantization so
        # the quantizer sees the contiguous [low|main|high] layout whose
        # segment lengths align to the FP block grid (NVFP4 16 / MXFP8 32).
        self.reorder_before_quant = False
        # Fused down_proj producer path (real kernel only): the wrapper receives
        # the un-materialized SwiGLU operands via `mlp_gate`/`mlp_up` kwargs and
        # the single fused kernel does silu*up -> Hadamard/sqrt(I) -> fp16 ->
        # NVFP4+MXFP8 pack.  Set by full_network.wire_real_kernel; only enabled
        # when the inner module exposes forward_swiglu_pack.
        self.fused_swiglu_had = False
        # round-26 lever B: activation quant/pack fused into the RMSNorm producer.
        # `norm_pack` marks this Linear as a member of a norm->pack group; the
        # layer's norm adapter (full_network._NormProducer) then runs ONE
        # add_rmsnorm_pack_mixed launch per norm and publishes the packed codes +
        # per-N SFA layouts on `_norm_pack_group_obj` (a _NormPackGroup).
        # DEFAULT-OFF: the fused RMSNorm is not bit-exact to LlamaRMSNorm (fp32
        # reduction order), whole-model logit cos 0.9889 < 0.99999 -> see
        # docs/edge_fusion_report.md §2.
        self.norm_pack = False
        self._norm_pack_group_obj = None
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

        # Fused SwiGLU + Hadamard + pack producer path (down_proj, real kernel).
        # `x` is unused here: the MLP hands us the two un-materialized GEMM
        # outputs, so the [M, I] bf16 intermediate is never written to HBM.
        gate = kwargs.pop("mlp_gate", None)
        up = kwargs.pop("mlp_up", None)
        # The fused kernel's interleaved-SFA atom layout is defined for M>=128;
        # below that (decode / tiny smoke tests) fall through to the reference
        # path, which packs `h` exactly as before.
        if self.fused_swiglu_had and gate is not None and up is not None \
                and gate.reshape(-1, gate.shape[-1]).shape[0] >= 128 \
                and hasattr(self.module, "forward_swiglu_pack"):
            x = self.module.forward_swiglu_pack(gate, up).to(x_dtype)
            if _quant_enabled(self.out_quantizer.bits):
                self.out_quantizer.find_params(x)
                x = self.out_quantizer(x).to(x_dtype)
                self.out_quantizer.free()
            return x

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
                xf = matmul_hadU_cuda(x.float(), self.had_K, self.K,
                                      fold_scale=self.fold_had_scale)
                # Skip the fp32->fp16 cast when the inner module consumes fp32
                # directly (real kernel): the qpf pack is bit-exact-to-spec on
                # fp32 and this removes a full-tensor copy per down_proj.
                x = xf if self.had_out_fp32 else xf.to(x_dtype)
            else:
                # round-9 Task-3: low-precision online Hadamard. `had_mixed`
                # runs the fp16/bf16 WHT butterfly but the small K×K had_K
                # matmul in fp32 (fp32-accumulate variant); `had_dtype` optionally
                # forces the butterfly dtype (e.g. bfloat16). Plain else = full
                # fp16 Hadamard. PPL-gated, not bit-identical.
                had_dtype = getattr(self, "had_dtype", None)
                xin = x if had_dtype is None else x.to(had_dtype)
                xo = matmul_hadU_cuda(xin, self.had_K, self.K,
                                      fold_scale=self.fold_had_scale,
                                      kmat_fp32=getattr(self, "had_mixed", False))
                x = xo.to(x_dtype)

        order = col_order if col_order is not None else getattr(self, '_column_order', None)
        if order is not None and self.reorder_before_quant:
            # Reorder first so quantization segments ([low|main|high]) are
            # contiguous and FP-block-aligned; order is consumed here.
            x = x[..., order]
            order = None

        if _quant_enabled(self.quantizer.bits):
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

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


# ---------------------------------------------------------------------------
# Round-3 integration: MSE-optimal block-scale search on the native FP grid.
#
# The default block scale is a round-UP (ceil to E4M3 / power-of-2) that
# over-allocates headroom so the FP4/FP8 element grid under-resolves the bulk
# of each block. The MSE search tries the base scale plus a few LOWER grid
# steps and keeps, per block, the one minimizing dequant MSE (trading a little
# outlier clipping for much finer bulk resolution). The chosen scale STAYS on
# the native grid (E4M3 for NVFP4, power-of-2/E8M0 for MXFP8) so the packed
# layout, SFA/SFB interleave and the fused kernel are all unchanged — it is a
# re-pack (weights) / producer-kernel (activations) change only.
#
# Flags are module-level so a gate/ablation can toggle each lever; both default
# ON (the shipped round-3 "full stack": activation-MSE + weight-A2). Env
# overrides PROMIX_ACT_MSE / PROMIX_WEIGHT_MSE ("0" disables) let a detached
# baseline run reproduce the pre-integration numbers without code edits.
import os as _os
# Win 1 (activation NVFP4 MSE search) is DEFAULT-ON as of round-4: the round-3
# search cost ~2.1x the fused-producer pack (8B isolated 2.595->5.541 ms) which
# sank 8B prefill 1.32->0.78x. round-4 made it CHEAP (reciprocal-multiply +
# branchless nearest, see kernels/.../act_mse.cuh): 8B producer overhead drops to
# 1.10x (<1.15x gate) at the SAME 3-step depth, so the accuracy win (3B
# 8.53->8.38, 8B 6.80->6.71) now ships in the prefill path. PROMIX_ACT_MSE=0
# reproduces the pre-round-3 baseline.
ACT_MSE_SCALE_SEARCH = _os.environ.get("PROMIX_ACT_MSE", "1") != "0"      # Win 1 (round-4: SHIPPED, cheap)
# round-4: the activation search is now CHEAP (reciprocal-multiply + branchless
# nearest, see kernels/.../act_mse.cuh). PROMIX_ACT_MSE_FAST (default 1) selects
# the fast selection; "0" reproduces the round-3 true-division search bit-for-bit
# (the accuracy-max PPL anchor). PROMIX_ACT_MSE_STEPS (default 3) = lower E4M3
# grid steps tried below the ceil base. These ONLY affect the ACTIVATION NVFP4
# scale SELECTION; the weight-A2 path (_mse_best_scale) is untouched.
ACT_MSE_FAST = _os.environ.get("PROMIX_ACT_MSE_FAST", "1") != "0"
ACT_MSE_STEPS = int(_os.environ.get("PROMIX_ACT_MSE_STEPS", "3"))
# Win 2 (weight-A2) is DEFAULT-ON: it is a re-pack-only change (GPTQ frozen block
# scales), zero forward cost, free accuracy win.
WEIGHT_MSE_SCALE_SEARCH = _os.environ.get("PROMIX_WEIGHT_MSE", "1") != "0"  # Win 2 (shipped)
_MSE_STEPS = 3

# Round-6 (WEIGHT two-level scale, deployable). The NVFP4 weight block scale
# becomes `g * micro` where `g` is a per-tensor FP32 global (derived from the low
# segment's amax) and `micro` is the per-16 E4M3 byte stored in the SFB. A SINGLE
# epilogue-alpha scalar carries `g` across the fused kernel's NVFP4 low AND MXFP8
# high phases: the MXFP8 high micro = e8m0(ideal_hi/g) stays a power of 2 (E8M0
# exact) for ANY g, so the high segment is given the SAME two-level global (its
# effective scale g*e8m0(ideal/g) is strictly finer than stock single-level). This
# lifts tiny weight-block scales out of E4M3's subnormal range and, with NEAREST
# micro rounding + per-block micro-MSE, is worth 3B r=1/8 -0.08 / r=0 -0.18 over
# single-level weight-A2 (docs/round6_twolevel_weights.md). Element codes/deq are
# UNCHANGED vs passing the effective scale as the frozen scale; only the SFB stores
# micro=eff/g and the packer returns g for the epilogue. DEFAULT-ON for the sizes
# where it helps (gated per size like weight-A2). An arbitrary (non-pow2) global is
# used: it keeps the low segment bit-identical to round-5's arbitrary-g two-level
# (the pow2 rounding cost +0.016 at r=1/8); the high stays deployable regardless.
WEIGHT_TWO_LEVEL = _os.environ.get("PROMIX_WEIGHT_2LEVEL", "1") != "0"
WEIGHT_TWO_LEVEL_POW2 = _os.environ.get("PROMIX_WEIGHT_2LEVEL_POW2", "0") != "0"


def _weight_two_level_nvfp4_eff(Wblk, bmax, pow2=None):
    """NVFP4 two-level effective per-block scale: g (per-tensor) * micro (per-16
    E4M3). `Wblk`=(..., nb, 16), `bmax`=(..., nb) block absmax. Returns
    (eff (..., nb), g scalar). NEAREST E4M3 micro + (weight-A2) lower-step micro-MSE
    when WEIGHT_MSE_SCALE_SEARCH. g = amax/(448*6); pow2 rounds it UP to a power of
    2 (default off -> arbitrary FP32)."""
    dev = Wblk.device
    if pow2 is None:
        pow2 = WEIGHT_TWO_LEVEL_POW2
    # Empty block guard: a reordered/mixed-precision segment can be zero-width
    # (K_high==0 or K_low==0 for a given Linear after measured-MSE reorder on
    # some model dims, e.g. 1B/8B). `.amax()` on a 0-element tensor raises; an
    # empty segment has nothing to quantize, so return empty eff + unit global.
    # Zero-regression: only triggers when numel()==0 (previously a hard crash).
    if Wblk.numel() == 0:
        return bmax.clone(), torch.ones((), device=dev, dtype=torch.float32)
    e4 = torch.tensor(_FP8_E4M3_POS, device=dev, dtype=torch.float32)
    gamax = Wblk.abs().amax()                                    # per-tensor scalar
    gs = (gamax / (448.0 * FP4_E2M1_MAX)).clamp_min(2.0 ** -40)
    g = _e8m0_round_up(gs) if pow2 else gs                       # global (scalar)
    micro_ideal = (bmax / FP4_E2M1_MAX) / g                      # (..., nb)
    micro = _round_to_nearest_value(micro_ideal, e4).abs()
    micro = torch.clamp(micro, min=float(e4[1]))                 # avoid 0 scale
    if not WEIGHT_MSE_SCALE_SEARCH:
        return (micro * g), g
    fp4 = torch.tensor(FP4_E2M1_POS, device=dev, dtype=torch.float32)
    cur = micro
    cands = [cur]
    for _ in range(_MSE_STEPS):
        cur = _next_lower_grid(cur, e4)
        cands.append(cur)
    eff_cands = [(c * g).unsqueeze(-1) for c in cands]           # (..., nb, 1)
    best = _mse_best_scale(Wblk, eff_cands, FP4_E2M1_MAX, fp4)   # (..., nb, 1)
    return best.squeeze(-1), g


def _weight_two_level_mxfp8_eff(Wblk, bmax, g):
    """MXFP8 high two-level effective per-block scale sharing the NVFP4 low global
    `g`: micro = e8m0(ideal_hi / g) (a power of 2, E8M0-exact for ANY g), eff =
    g*micro. weight-A2 (base/2, base/4 pow2 steps) still applies. `g` lets ONE
    epilogue-alpha carry the global across both fused-GEMM phases."""
    dev = Wblk.device
    if Wblk.numel() == 0:                 # empty high segment -> nothing to scale
        return bmax.clone()
    ideal = (bmax / FP8_E4M3_MAX / g).clamp_min(2.0 ** -127)
    micro = _e8m0_round_up(ideal)
    if not WEIGHT_MSE_SCALE_SEARCH:
        return micro * g
    e4 = torch.tensor(_FP8_E4M3_POS, device=dev, dtype=torch.float32)
    mk = micro.unsqueeze(-1)
    eff_cands = [(mk * g), (mk / 2.0 * g), (mk / 4.0 * g)]
    return _mse_best_scale(Wblk, eff_cands, FP8_E4M3_MAX, e4).squeeze(-1)


def _next_lower_grid(sc: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Per-element, the largest grid value strictly below `sc` (or `sc` itself if
    none is lower). `grid` is 1D ascending. Matches the CUDA byte-decrement over
    the full E4M3 positive representable set (sorted==byte order)."""
    g = grid.view(*([1] * sc.dim()), -1)
    masked = torch.where(g < sc.unsqueeze(-1), g, torch.full_like(g, -1.0))
    best = masked.max(dim=-1).values
    return torch.where(best < 0, sc, best)


def _mse_best_scale(blk: torch.Tensor, cands, max_fmt: float,
                    elem_grid: torch.Tensor) -> torch.Tensor:
    """blk: (..., block). cands: list of (..., 1) candidate scales. Returns the
    per-block (..., 1) scale minimizing sum-squared dequant error on `elem_grid`."""
    best_err = best_sc = None
    for sc in cands:
        q = _round_to_nearest_value((blk / sc).clamp(-max_fmt, max_fmt), elem_grid)
        err = ((q * sc - blk) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_sc = err, sc
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_sc = torch.where(take, sc, best_sc)
    return best_sc


def _fp4_nearest_val(a: torch.Tensor) -> torch.Tensor:
    """Branchless nearest FP4 E2M1 grid VALUE for a >= 0. Mirrors the CUDA
    `promix_fp4_nearest_val` (`<` cascade over {0,.5,1,1.5,2,3,4,6}); used only
    inside the fast ACTIVATION scale selection."""
    v = torch.full_like(a, 6.0)
    v = torch.where(a < 5.00, torch.full_like(a, 4.0), v)
    v = torch.where(a < 3.50, torch.full_like(a, 3.0), v)
    v = torch.where(a < 2.50, torch.full_like(a, 2.0), v)
    v = torch.where(a < 1.75, torch.full_like(a, 1.5), v)
    v = torch.where(a < 1.25, torch.full_like(a, 1.0), v)
    v = torch.where(a < 0.75, torch.full_like(a, 0.5), v)
    v = torch.where(a < 0.25, torch.zeros_like(a), v)
    return v


def _mse_best_scale_act_fast(blk: torch.Tensor, cands) -> torch.Tensor:
    """Fast activation NVFP4 scale selection — the torch twin of the CUDA
    `promix_act_mse_select` fast path: reciprocal-multiply + branchless nearest
    (selection only). `cands` is a list of (..., 1) candidate scales. Returns the
    per-block (..., 1) argmin-MSE scale."""
    best_err = best_sc = None
    for sc in cands:
        inv = 1.0 / sc
        xs = (blk * inv).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
        g = _fp4_nearest_val(xs.abs())
        dq = torch.where(xs < 0, -g, g) * sc
        err = ((dq - blk) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_sc = err, sc
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_sc = torch.where(take, sc, best_sc)
    return best_sc



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


def select_block_scales_mxfp8(
    W: torch.Tensor, block_size: int = 32
) -> torch.Tensor:
    """Compute per-row-per-block MXFP8 scales using the spec-derived rule.

    Used by Section-11-style GPTQ (and any other consumer that needs the
    block scale separately from the rounding step). The scale is computed
    EXACTLY the way `fake_quantize_mxfp8` computes it: max_abs(block) /
    FP8_E4M3_MAX, then rounded UP to the nearest power of 2 (E8M0).

    Args:
        W: weight segment with shape (..., K) where K is along the
           contraction axis. K must be divisible by block_size.
        block_size: 32 for MXFP8 (per spec).

    Returns:
        Tensor with shape (..., K // block_size); float32.
    """
    assert W.shape[-1] % block_size == 0, (
        f"select_block_scales_mxfp8: K ({W.shape[-1]}) not divisible by block_size ({block_size})"
    )
    Wf = W.to(torch.float32)
    *prefix, k = Wf.shape
    num_blocks = k // block_size
    Wf_blk = Wf.reshape(*prefix, num_blocks, block_size)
    block_max = Wf_blk.abs().amax(dim=-1).clamp_min(0.0)
    ideal = block_max / FP8_E4M3_MAX
    base = _e8m0_round_up(ideal.clamp_min(2.0 ** -127))        # (..., nb)
    if not WEIGHT_MSE_SCALE_SEARCH:
        return base
    # Win 2 (weight-A2), high segment: the power-of-2 candidates base/2, base/4
    # stay on the E8M0 grid; keep the min-MSE one per block (re-pack only).
    e4m3 = torch.tensor(_FP8_E4M3_POS, device=Wf.device, dtype=torch.float32)
    bk = base.unsqueeze(-1)                                    # (..., nb, 1)
    cands = [bk, bk / 2.0, bk / 4.0]
    return _mse_best_scale(Wf_blk, cands, FP8_E4M3_MAX, e4m3).squeeze(-1)


def select_block_scales_nvfp4(
    W: torch.Tensor, block_size: int = 16
) -> torch.Tensor:
    """Compute per-row-per-block NVFP4 scales using the spec-derived rule.

    Same role as `select_block_scales_mxfp8` but for NVFP4. The scale is
    `max_abs(block) / FP4_E2M1_MAX`, then rounded UP to the nearest FP8
    E4M3 representable value.

    Args:
        W: weight segment, shape (..., K). K divisible by block_size.
        block_size: 16 for NVFP4 (per spec).

    Returns:
        Tensor with shape (..., K // block_size); float32.
    """
    assert W.shape[-1] % block_size == 0, (
        f"select_block_scales_nvfp4: K ({W.shape[-1]}) not divisible by block_size ({block_size})"
    )
    Wf = W.to(torch.float32)
    *prefix, k = Wf.shape
    num_blocks = k // block_size
    Wf_blk = Wf.reshape(*prefix, num_blocks, block_size)
    block_max = Wf_blk.abs().amax(dim=-1)
    if WEIGHT_TWO_LEVEL:
        # Round-6: two-level effective scale (g*micro). Returns the EFFECTIVE
        # per-block scale; the packer decomposes it into (E4M3 micro in the SFB,
        # pow2 g on the epilogue). GPTQ freezes/rounds against this effective
        # scale exactly as it did the single-level one.
        eff, _g = _weight_two_level_nvfp4_eff(Wf_blk, block_max, pow2=True)
        return eff
    ideal = (block_max / FP4_E2M1_MAX).clamp_min(2.0 ** -9)
    fp8_e4m3_values = torch.tensor(_FP8_E4M3_POS, device=Wf.device, dtype=torch.float32)
    base = _ceil_to_value(ideal, fp8_e4m3_values)              # (..., nb)
    if not WEIGHT_MSE_SCALE_SEARCH:
        return base
    # Win 2 (weight-A2): keep the base scale or a lower E4M3 grid step, whichever
    # minimizes per-block dequant MSE. Re-pack only; still on the E4M3 grid.
    fp4_values_pos = torch.tensor(FP4_E2M1_POS, device=Wf.device, dtype=torch.float32)
    cur = base.unsqueeze(-1)                                   # (..., nb, 1)
    cands = [cur]
    for _ in range(_MSE_STEPS):
        cur = _next_lower_grid(cur, fp8_e4m3_values)
        cands.append(cur)
    return _mse_best_scale(Wf_blk, cands, FP4_E2M1_MAX, fp4_values_pos).squeeze(-1)


def quantize_column_with_frozen_scale(
    w_col: torch.Tensor,
    frozen_scales: torch.Tensor,
    col_idx: int,
    fmt: str,
) -> torch.Tensor:
    """Round a single column of W to the nearest FP representable value
    given a pre-computed (frozen) per-row block scale for that column.

    This is the building block for spec Section 11 FP GPTQ: the block
    scale is frozen up-front from the original (uncorrected) weight,
    then GPTQ's per-column Hessian-corrected value is rounded against
    the SAME frozen scale.

    Args:
        w_col: shape (rows,); a single column of the segment.
        frozen_scales: shape (rows, n_blocks); pre-computed scales.
        col_idx: column index within the segment; selects the block via
                 col_idx // block_size for the segment's format.
        fmt: "mxfp8" (block_size=32, FP8 E4M3 element grid) or
             "nvfp4" (block_size=16, FP4 E2M1 element grid).

    Returns:
        Quantized column (same shape & dtype as w_col): each element
        is `representable_grid_value × frozen_scales[r, col_idx // block_size]`.
    """
    if fmt == "mxfp8":
        block_size = 32
        max_format = FP8_E4M3_MAX
        grid = torch.tensor(_FP8_E4M3_POS, device=w_col.device, dtype=torch.float32)
    elif fmt == "nvfp4":
        block_size = 16
        max_format = FP4_E2M1_MAX
        grid = torch.tensor(FP4_E2M1_POS, device=w_col.device, dtype=torch.float32)
    else:
        raise ValueError(f"unknown FP format {fmt!r}; expected 'mxfp8' or 'nvfp4'")

    block_idx = col_idx // block_size
    scale = frozen_scales[..., block_idx]  # (rows,)
    scaled = (w_col.to(torch.float32) / scale).clamp(-max_format, max_format)
    q = _round_to_nearest_value(scaled, grid)
    return (q * scale).to(w_col.dtype)


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
    base_scale = _ceil_to_value(ideal_scale, fp8_e4m3_values)
    fp4_values_pos = torch.tensor(FP4_E2M1_POS, device=x.device, dtype=torch.float32)

    if ACT_MSE_SCALE_SEARCH:
        # Win 1: per-block argmin-MSE over {base, N lower E4M3 steps}. round-4
        # makes this CHEAP (ACT_MSE_FAST, default): reciprocal-multiply +
        # branchless nearest, the twin of the CUDA producer's fast search. Set
        # PROMIX_ACT_MSE_FAST=0 to reproduce the round-3 true-division anchor.
        cur = base_scale
        cands = [cur]
        for _ in range(ACT_MSE_STEPS):
            cur = _next_lower_grid(cur, fp8_e4m3_values)
            cands.append(cur)
        if ACT_MSE_FAST:
            block_scale = _mse_best_scale_act_fast(x_blk, cands)
        else:
            block_scale = _mse_best_scale(x_blk, cands, FP4_E2M1_MAX, fp4_values_pos)
    else:
        block_scale = base_scale

    # Element: round to nearest FP4 E2M1 representable value
    x_scaled = x_blk / block_scale
    x_scaled = x_scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
    q = _round_to_nearest_value(x_scaled, fp4_values_pos)
    deq = q * block_scale
    return deq.view(*prefix, k).to(orig_dtype)
