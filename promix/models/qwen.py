"""Qwen3 model plugin — supports the Qwen3 dense family (e.g. Qwen3-8B).

Qwen3 vs Llama architecture diffs that matter for ProMix/ResQ rotation:

- q_norm / k_norm: per-head RMSNorm applied to q and k AFTER q_proj/k_proj and
  BEFORE RoPE. These do NOT interact with the fused rotations: R1 is absorbed
  on the INPUT side of q/k/v_proj (their outputs are numerically unchanged, so
  q_norm/k_norm see identical tensors), and R2 is applied only on the value
  path (v_proj output / o_proj input), which has no norm. q_norm/k_norm are
  RMSNorm-over-head_dim modules (not nn.Linear) so add_actquant leaves them in
  fp16 — correct.
- key_pos caveat: basis.py collects H_key_pos from k_proj output + RoPE,
  WITHOUT k_norm, so the key_pos basis does not match Qwen3's real post-RoPE
  keys. It is only consumed by setup_k_quant (KV-cache quantization); W4A4
  configs (k_bits=16) never read it. KV4 on Qwen3 requires threading k_norm
  into the basis hook first.
- No biases anywhere (attention_bias=false). QKV bias is a Qwen2.5 property;
  supporting Qwen2.5 additionally requires rotating v_proj.bias by R2 in
  rotation._apply_had_to_linear (output path currently rotates weight only).
- explicit config.head_dim: Qwen3 defines head_dim independently of
  hidden_size // num_attention_heads. For Qwen3-8B they coincide (128); the
  plugin asserts this because rotation.py / ptq.py derive head_dim as
  hidden_size // num_heads throughout.
- tie_word_embeddings=False for 8B+ (loader's untie handling is a no-op).
- intermediate_size=12288 = 12 x 1024 -> down_proj online Hadamard uses had12
  (Paley I, q=11; added to hadamard_data.json).
"""

from typing import Dict, List
import torch.nn as nn

from .base import ModelPlugin


class Qwen3Plugin(ModelPlugin):
    model_type = "qwen3"

    def get_layers(self, model) -> List[nn.Module]:
        return list(model.model.layers)

    def get_projections(self, layer) -> Dict[str, nn.Module]:
        return {
            "q_proj": layer.self_attn.q_proj,
            "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj,
            "o_proj": layer.self_attn.o_proj,
            "gate_proj": layer.mlp.gate_proj,
            "up_proj": layer.mlp.up_proj,
            "down_proj": layer.mlp.down_proj,
        }

    def get_embedding(self, model) -> nn.Module:
        return model.model.embed_tokens

    def get_lm_head(self, model) -> nn.Module:
        return model.lm_head

    def get_norm(self, model) -> nn.Module:
        return model.model.norm

    def get_hidden_size(self, model) -> int:
        return model.config.hidden_size

    def get_intermediate_size(self, model) -> int:
        return model.config.intermediate_size

    def get_num_heads(self, model) -> int:
        return model.config.num_attention_heads

    def get_head_dim(self, model) -> int:
        implied = model.config.hidden_size // model.config.num_attention_heads
        explicit = getattr(model.config, "head_dim", implied)
        assert explicit == implied, (
            f"Qwen3 config.head_dim={explicit} != hidden//heads={implied}; "
            "rotation.py/ptq.py derive head_dim as hidden//heads, so this "
            "model size is NOT supported without generalizing those sites "
            "(e.g. Qwen3-4B: 2560/32=80 vs head_dim=128)."
        )
        return explicit
