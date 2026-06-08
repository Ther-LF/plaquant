"""KV cache quantization — online rotation + mixed-precision quantization for K cache.

K cache: requires runtime rotation (RoPE blocks fuse), then per-head mixed 4/8-bit.
V cache: handled by out_quantizer on v_proj (already in quant_utils.py).
"""

import copy
import functools
import math
import types

import torch
import torch.nn as nn

from promix.quantize.quant_utils import ActQuantizer, STEQuantize, AsymSTEQuantize, get_minq_maxq
from promix.utils import HadamardTransform


class WeightQuantizer(nn.Module):
    """Simple weight/matrix quantizer (used to quantize k_rotation matrix to INT8)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))

    def configure(self, bits, sym=True):
        self.bits = bits
        self.sym = sym
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)
        x_flat = x.flatten().unsqueeze(0)
        tmp = torch.zeros(x_flat.shape[0], device=dev)
        xmin = torch.minimum(x_flat.min(1)[0], tmp)
        xmax = torch.maximum(x_flat.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = 1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        shape = [-1] + [1] * (len(x.shape) - 1)
        self.scale = self.scale.repeat(x.shape[0]).reshape(shape)
        self.zero = self.zero.repeat(x.shape[0]).reshape(shape)

    def quantize(self, x):
        if self.bits < 16:
            if self.sym:
                return STEQuantize.apply(x, self.scale, self.maxq)
            return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq)
        return x


class QKRotationWrapper(nn.Module):
    """Wraps apply_rotary_pos_emb to add post-RoPE rotation and K quantization.

    After RoPE:
    1. Pre-quantize q,k to 8-bit (reduce rotation error)
    2. Rotate q,k by k_rotation (U_key_pos @ R2)
    3. Quantize K to mixed 4/8-bit per-head
    """

    def __init__(self, func, config, k_rotation, k_had, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        self.num_kv_groups = num_heads // config.num_key_value_heads
        self.func = func
        self.k_quantizer = ActQuantizer()
        self.pre_rotation_quantizer = ActQuantizer()
        self.k_bits = 16
        self.k_rotation = k_rotation
        self.k_had = k_had

        if kwargs:
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.high_bits_length = kwargs["high_bits_length"]

            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,
                sym=kwargs["k_sym"],
                clip_ratio=kwargs["k_clip_ratio"],
                high_bits_length=kwargs["high_bits_length"],
                high_bits=kwargs["k_bits_high"],
                low_bits_length=kwargs["low_bits_length"],
                low_bits=kwargs["k_bits_low"],
            )
            self.pre_rotation_quantizer.configure(
                bits=8,
                groupsize=-1,
                sym=kwargs["k_sym"],
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype

        if self.k_had:
            q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
            k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
        else:
            self.pre_rotation_quantizer.find_params(q)
            q = self.pre_rotation_quantizer(q).to(dtype)
            q = torch.matmul(q, self.k_rotation.to(q))
            self.pre_rotation_quantizer.free()

            self.pre_rotation_quantizer.find_params(k)
            k = self.pre_rotation_quantizer(k).to(dtype)
            k = torch.matmul(k, self.k_rotation.to(k))

        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:
            per_head_k = k.contiguous().view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        self.pre_rotation_quantizer.free()
        self.k_quantizer.free()
        return q, k


def _copy_func_with_new_globals(f, globals=None):
    """Copy a function with new globals dict."""
    if globals is None:
        globals = f.__globals__
    g = types.FunctionType(
        f.__code__, globals, name=f.__name__,
        argdefs=f.__defaults__, closure=f.__closure__,
    )
    g = functools.update_wrapper(g, f)
    g.__module__ = f.__module__
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g


def add_qk_rotation_wrapper(module, function_name, **kwargs):
    """Inject QKRotationWrapper after apply_rotary_pos_emb in the attention forward.

    Uses monkeypatch: replaces the function in the method's globals so that
    when forward() calls apply_rotary_pos_emb(), it actually calls our wrapper.
    """
    original_method = getattr(module, "forward").__func__
    method_globals = dict(original_method.__globals__)
    wrapper = QKRotationWrapper(method_globals[function_name], **kwargs)
    method_globals[function_name] = wrapper
    new_method = _copy_func_with_new_globals(original_method, globals=method_globals)
    setattr(module, "forward", new_method.__get__(module))
    setattr(module, f"{function_name}_qk_rotation_wrapper", wrapper)


def setup_k_quant(model, config, basis_path, rotation_path):
    """Setup K cache quantization with per-layer rotation + mixed-precision.

    Args:
        model: Model after fuse_basis + add_actquant
        config: YAML config dict
        basis_path: Path to U file
        rotation_path: Path to R file
    """
    qcfg = config['quantize']
    k_bits = qcfg.get('k_bits', 16)
    if k_bits >= 16:
        return

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    high_fraction = qcfg['high_fraction']
    low_fraction = qcfg.get('low_fraction', 0.0)
    high_bits_length = int(high_fraction * head_dim)
    low_bits_length = int(low_fraction * head_dim)

    k_quant_config = {
        "k_bits": k_bits,
        "k_bits_high": qcfg['high_bits'],
        "k_bits_low": qcfg['low_bits'],
        "k_groupsize": qcfg.get('k_groupsize', 64),
        "k_sym": not qcfg.get('k_asym', True),
        "k_clip_ratio": 1.0,
        "high_bits_length": high_bits_length,
        "low_bits_length": low_bits_length,
    }

    U_cpk = torch.load(basis_path, weights_only=False)
    R_dict = torch.load(rotation_path, weights_only=False)

    # Build R2 for key rotation
    if "R2_1" in R_dict:
        R2_1 = R_dict["R2_1"].cuda().to(torch.float64)
        R2_2 = R_dict["R2_2"].cuda().to(torch.float64)
        R2 = torch.block_diag(R2_1, R2_2)
        R2_0 = R_dict.get("R2_0")
        if R2_0 is not None:
            R2 = torch.block_diag(R2_0.cuda().to(torch.float64), R2)
    else:
        R2_1 = R_dict["model.layers.0.self_attn.R2_1"].cuda().to(torch.float64)
        R2_2 = R_dict["model.layers.0.self_attn.R2_2"].cuda().to(torch.float64)
        R2 = torch.block_diag(R2_1, R2_2)

    rope_function_name = "apply_rotary_pos_emb"
    layers = model.model.layers

    for idx, layer in enumerate(layers):
        # Per-layer key rotation = U_key_pos @ R2
        k_rotation = U_cpk[f"layer.{idx}.self_attn.key_pos"].cuda()

        # Handle per-layer R2 (trained format)
        if "R2_1" not in R_dict:
            r2_key = f"model.layers.{idx}.self_attn.R2_1"
            if r2_key in R_dict:
                R2_1 = R_dict[r2_key].cuda().to(torch.float64)
                R2_2 = R_dict[f"model.layers.{idx}.self_attn.R2_2"].cuda().to(torch.float64)
                R2 = torch.block_diag(R2_1, R2_2)

        k_rotation = torch.matmul(k_rotation, R2)

        # Quantize rotation matrix to INT8 (saves compute at runtime)
        quantizer = WeightQuantizer()
        quantizer.configure(8)
        quantizer.find_params(k_rotation)
        k_rotation = quantizer.quantize(k_rotation)

        add_qk_rotation_wrapper(
            layer.self_attn,
            rope_function_name,
            config=model.config,
            k_rotation=k_rotation,
            k_had=False,
            **k_quant_config,
        )
