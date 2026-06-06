"""Dataset loading utilities for calibration and evaluation."""

import random

import datasets
import torch
import transformers


def get_wikitext2(nsamples=128, seed=0, seqlen=2048, tokenizer=None, eval_mode=False):
    """Load WikiText-2 dataset for calibration or evaluation."""
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    if eval_mode:
        testdata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]
        testenc = tokenizer(text="\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    else:
        traindata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]
        trainenc = tokenizer(text="\n\n".join(traindata["text"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
