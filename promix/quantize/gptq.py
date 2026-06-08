"""GPTQ weight quantization — Hessian-based optimal rounding.

Layer-by-layer weight quantization using calibration data to compute
Hessian matrices and optimally round weights to minimize output error.
Supports mixed-precision: different bit-widths for high/low/main groups.
"""

import copy
import logging
import math
import time

import torch
import torch.nn as nn

from promix.quantize.kv_quant import WeightQuantizer
from promix.quantize.quant_utils import find_qlayers, ActQuantWrapper
from promix.utils import cleanup_memory


class GPTQ:
    """GPTQ quantizer for a single linear layer."""

    def __init__(self, layer, mixed_precision=False, high_bits_length=0, low_bits_length=0):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.mixed_precision = mixed_precision
        self.high_bits_length = high_bits_length
        self.low_bits_length = low_bits_length
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.01, groupsize=-1, actorder=False):
        W = self.layer.weight.data.clone().float()
        high_dim = W.shape[-1] - self.high_bits_length
        low_dim = self.low_bits_length
        mp = self.mixed_precision

        if mp:
            W_l, W_m, W_h = W[:, :low_dim], W[:, low_dim:high_dim], W[:, high_dim:]
            if not self.quantizer.ready():
                self.quantizer.find_params(W_m)
            if self.high_bits_length != 0 and not self.high_quantizer.ready():
                self.high_quantizer.find_params(W_h)
            if self.low_bits_length != 0 and not self.low_quantizer.ready():
                self.low_quantizer.find_params(W_l)
        else:
            if not self.quantizer.ready():
                self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        Q_int = torch.zeros_like(W, dtype=torch.int16)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        try:
            H = torch.linalg.cholesky(H, upper=True)
        except torch._C._LinAlgError:
            epsilon = 1e-5
            H = H + epsilon * torch.eye(H.size(0), device=H.device)
            H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])

                if mp and (i1 + i) >= high_dim:
                    q = self.high_quantizer.quantize(w.unsqueeze(1)).flatten()
                    q_int_val, _, _ = self.high_quantizer.quantize_to_int(w.unsqueeze(1))
                    Q_int[:, i1 + i] = q_int_val.flatten()
                elif mp and (i1 + i) < low_dim:
                    q = self.low_quantizer.quantize(w.unsqueeze(1)).flatten()
                    q_int_val, _, _ = self.low_quantizer.quantize_to_int(w.unsqueeze(1))
                    Q_int[:, i1 + i] = q_int_val.flatten()
                else:
                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                    q_int_val, _, _ = self.quantizer.quantize_to_int(w.unsqueeze(1))
                    Q_int[:, i1 + i] = q_int_val.flatten()

                Q1[:, i] = q
                Losses[:, i1 + i] = (w - q) ** 2 / d ** 2
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]
            Q_int = Q_int[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        self.Q_int = Q_int.reshape(self.layer.weight.shape)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, config):
    """Run GPTQ weight quantization layer-by-layer.

    Args:
        model: Model after rotation + actquant wrapping
        dataloader: Calibration data [(input_ids, targets), ...]
        dev: Device
        config: YAML config dict

    Returns:
        quantizers: Dict of WeightQuantizer per layer
        int_weights: Dict of integer weight tensors per layer
    """
    logging.info("-----GPTQ Quantization-----")
    qcfg = config['quantize']
    w_bits = qcfg['w_bits']
    w_sym = not qcfg.get('w_asym', False)
    w_clip = qcfg.get('w_clip', True)
    w_groupsize = qcfg.get('w_groupsize', -1)
    percdamp = qcfg.get('percdamp', 0.01)
    act_order = qcfg.get('act_order', False)
    nsamples = len(dataloader)
    high_fraction = qcfg['high_fraction']
    low_fraction = qcfg.get('low_fraction', 0.0)
    model_dim = model.config.hidden_size

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs.get("position_embeddings", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    quantizers = {}
    int_weights = {}

    sequential = [
        ["self_attn.k_proj.module", "self_attn.v_proj.module", "self_attn.q_proj.module"],
        ["self_attn.o_proj.module"],
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
        ["mlp.down_proj.module"],
    ]

    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i].to(dev)
        full = find_qlayers(layer, layers=[nn.Linear])

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                layer_w_bits = w_bits
                if "lm_head" in name:
                    continue

                mixed_precision = False
                high_bits_length = 0
                low_bits_length = 0
                if "k_proj" in name or "q_proj" in name or "v_proj" in name \
                        or "up_proj" in name or "gate_proj" in name or "o_proj" in name:
                    mixed_precision = True
                    high_bits_length = int(high_fraction * model_dim)
                    low_bits_length = int(low_fraction * model_dim)

                gptq[name] = GPTQ(
                    subset[name],
                    mixed_precision=mixed_precision,
                    high_bits_length=high_bits_length,
                    low_bits_length=low_bits_length,
                )
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(layer_w_bits, perchannel=True, sym=w_sym, mse=w_clip)

                if mixed_precision and high_bits_length > 0:
                    gptq[name].high_quantizer = WeightQuantizer()
                    gptq[name].high_quantizer.configure(
                        qcfg['high_bits'], perchannel=True, sym=w_sym, mse=w_clip
                    )
                if mixed_precision and low_bits_length > 0:
                    gptq[name].low_quantizer = WeightQuantizer()
                    gptq[name].low_quantizer.configure(
                        qcfg['low_bits'], perchannel=True, sym=False, mse=w_clip
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                gptq[name].fasterquant(
                    percdamp=percdamp,
                    groupsize=w_groupsize,
                    actorder=act_order,
                )
                quantizers[f"model.layers.{i}.{name}"] = gptq[name].quantizer.cpu()
                if gptq[name].Q_int is not None:
                    int_weights[f"model.layers.{i}.{name}"] = gptq[name].Q_int.cpu()
                if mixed_precision and high_bits_length > 0:
                    quantizers[f"model.layers.{i}.{name},high_quantizer"] = gptq[name].high_quantizer.cpu()
                if mixed_precision and low_bits_length > 0:
                    quantizers[f"model.layers.{i}.{name},low_quantizer"] = gptq[name].low_quantizer.cpu()
                gptq[name].free()

        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

        layers[i] = layer.cpu()
        del layer, gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    cleanup_memory()
    logging.info("-----GPTQ Quantization Done-----")
    return quantizers, int_weights
