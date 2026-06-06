"""Wikitext-2 perplexity evaluator — layer-by-layer offloading."""

import logging
import os

import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_ppl(model, testenc, dev, batch_size=8):
    """Evaluate wikitext-2 perplexity using layer-by-layer GPU offloading.

    This approach keeps only one layer on GPU at a time, enabling
    evaluation of large models that don't fit in GPU memory.
    """
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    input_ids = testenc.input_ids
    nsamples = input_ids.numel() // model.seqlen
    input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)
    input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)

    # Capture first-layer inputs
    inps = [0] * nbatches
    cache = {"i": 0}

    class Catcher(torch.nn.Module):
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
    for i in range(nbatches):
        try:
            model(input_ids[i])
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]
    torch.cuda.empty_cache()

    outs = [0] * nbatches

    # Process each layer
    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i].to(dev)
        for j in range(nbatches):
            outs[j] = layer(
                inps[j],
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # Final norm + lm_head
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in range(nbatches):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.cat(nlls).mean())
    model.config.use_cache = use_cache
    logging.info(f"WikiText2 PPL: {ppl.item():.3f}")
    return ppl.item()
