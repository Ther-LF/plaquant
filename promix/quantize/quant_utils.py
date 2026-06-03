"""Core quantization utilities — activation quantizers, STE, quant/dequant functions.

Migrated from project-resq/fake_quant/utils/quant_utils.py with same logic,
cleaner organization.
"""

import math
import torch
import torch.nn as nn


def get_minq_maxq(bits, sym):
    """Get min/max quantization range for given bits and symmetry."""
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = torch.tensor(0)
    return minq, maxq


# ============================================================================
# Basic quant/dequant functions
# ============================================================================

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


# ============================================================================
# STE (Straight-Through Estimator) quantization for gradient-based optimization
# ============================================================================

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq, stoch=False):
        scale = scale.to(x.device)
        if stoch:
            q = torch.clamp(_stoch_round(x / scale), -(maxq + 1), maxq)
        else:
            q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class AsymSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maxq, stoch=False):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        if stoch:
            q = torch.clamp(_stoch_round(x / scale) + zero, 0, maxq)
        else:
            q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def _stoch_round(tensor):
    """Stochastic rounding."""
    floor_values = tensor.floor()
    fractional_part = tensor - floor_values
    random_values = torch.rand_like(tensor)
    return torch.where(random_values < fractional_part, tensor.ceil(), floor_values)


# ============================================================================
# ActQuantizer: per-token activation quantizer with mixed-precision support
# ============================================================================

class ActQuantizer(nn.Module):
    """Per-token activation quantizer supporting mixed precision (low/main/high channels)."""

    def __init__(self):
        super().__init__()
        # Main precision
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        # High precision
        self.register_buffer("maxq_h", torch.tensor(0))
        self.register_buffer("scale_h", torch.zeros(1))
        self.register_buffer("zero_h", torch.zeros(1))
        # Low precision
        self.register_buffer("maxq_l", torch.tensor(0))
        self.register_buffer("scale_l", torch.zeros(1))
        self.register_buffer("zero_l", torch.zeros(1))

        self.bits = 16
        self.high_bits = 16
        self.low_bits = 16
        self.high_bits_length = 0
        self.low_bits_length = 0

    def configure(self, bits, groupsize=-1, sym=False, clip_ratio=1.0,
                  high_bits_length=0, high_bits=16, low_bits_length=0, low_bits=16):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio

        self.high_bits_length = high_bits_length
        self.high_bits = high_bits
        _, self.maxq_h = get_minq_maxq(high_bits, sym)

        self.low_bits_length = low_bits_length
        self.low_bits = low_bits
        _, self.maxq_l = get_minq_maxq(low_bits, sym)

    def find_params(self, x):
        """Compute per-token quantization parameters for the input tensor."""
        if self.groupsize > 0:
            init_shape = x.shape
            x_reshaped = x.reshape(x.shape[0], x.shape[1], x.shape[2] // self.groupsize, self.groupsize)
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

    def _find_params_groupwise(self, x, maxq):
        """Per-group quantization params (for grouped attention like KV cache)."""
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

    def _find_params_per_token(self, x, maxq):
        """Per-token quantization params."""
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

    def forward(self, x):
        if self.bits == 16:
            return x

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

    def free(self):
        self.zero = None
        self.scale = None
        self.zero_h = None
        self.scale_h = None
        self.zero_l = None
        self.scale_l = None


# ============================================================================
# ActQuantWrapper: wraps a Linear layer with activation quantization
# ============================================================================

class ActQuantWrapper(nn.Module):
    """Wraps a Linear layer with input/output activation quantization + online Hadamard."""

    def __init__(self, module: nn.Linear):
        super().__init__()
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.hadK_quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer("had_K", None)
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False

    def forward(self, x, R1=None, R2=None, transpose=False, column_order=None):
        x_dtype = x.dtype

        # Online Hadamard rotation (if configured)
        if self.online_full_had:
            from .hadamard import matmul_hadU_cuda
            if self.fp32_had:
                x = matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:
                x = matmul_hadU_cuda(x, self.had_K, self.K)
        elif self.online_partial_had:
            x = self._partial_hadamard(x, x_dtype)

        # Quantize input activations
        if self.quantizer.bits < 16:
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        # Column reorder (for rotated weights)
        if column_order is not None:
            x = x[..., column_order]

        # Linear operation
        if R1 is not None:
            x = self.module(x, R1, R2, transpose).to(x_dtype)
        else:
            x = self.module(x).to(x_dtype)

        # Quantize output (if configured, e.g., for KV cache)
        if self.out_quantizer.bits < 16:
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x

    def _partial_hadamard(self, x, x_dtype):
        from .hadamard import HadamardTransform
        if self.fp32_had:
            x = x.float()
        init_shape = x.shape
        if self.K == 1:
            x = (HadamardTransform.apply(
                x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim).transpose(1, 2)
            ) / math.sqrt(init_shape[-1] // self.had_dim)).transpose(1, 2)
        else:
            x = (self.had_K.to(x.dtype) @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                 ) / math.sqrt(init_shape[-1] // self.had_dim)
        if self.fp32_had:
            x = x.to(x_dtype)
        return x.reshape(init_shape)


# ============================================================================
# Helper functions for adding/finding quantization wrappers in a model
# ============================================================================

def add_actquant(model):
    """Wrap all Linear layers in the model with ActQuantWrapper."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, ActQuantWrapper(module))


def find_qlayers(model):
    """Find all ActQuantWrapper layers in the model."""
    qlayers = {}
    for name, module in model.named_modules():
        if isinstance(module, ActQuantWrapper):
            qlayers[name] = module
    return qlayers
