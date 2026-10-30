"""Dataset loading utilities for calibration and evaluation."""

import random
from typing import Any, Dict

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


class CustomJsonDataset(torch.utils.data.IterableDataset):
    """Tokenized text dataset for rotation training (from HF dataset)."""

    def __init__(self, dataset, tokenizer, block_size=1024):
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized = [self._tokenize(d) for d in dataset]
        grouped = self._group_texts(tokenized)
        self.input_ids = grouped["input_ids"]
        self.labels = grouped["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, Any]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def _tokenize(self, examples):
        return self.tokenizer(examples["text"])

    def _group_texts(self, examples):
        concatenated = {}
        for d in examples:
            for key in d.keys():
                if key not in concatenated:
                    concatenated[key] = []
                concatenated[key].extend(d[key])
        total_length = len(concatenated["input_ids"])
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [t[i:i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


def get_c4(nsamples=256, seed=0, seqlen=2048, tokenizer=None, eval_mode=False):
    """Load C4 (allenai/c4, en) for calibration or evaluation.

    Follows the GPTQ/QuaRot/SpinQuant/ResQ-standard C4 PPL protocol: sample
    `nsamples` random windows of `seqlen` tokens from the first validation
    (eval) / train (calib) shard and concatenate them. eval_mode=True returns a
    lightweight wrapper exposing `.input_ids` (shape [1, nsamples*seqlen]) so it
    plugs straight into `evaluate_ppl`, which slices it into `nsamples` windows.

    The 256x2048 validation stream is the de-facto C4 perplexity set reported by
    the W4A4 quantization literature (QuaRot Table, SpinQuant, ResQ). seed is
    fixed so the window selection is reproducible across configs/models.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    if eval_mode:
        valdata = datasets.load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        random.seed(seed)
        valenc = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(valdata) - 1)
                enc = tokenizer(valdata[i]["text"], return_tensors="pt")
                if enc.input_ids.shape[1] > seqlen:
                    break
            s = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            valenc.append(enc.input_ids[:, s:s + seqlen])
        valenc = torch.hstack(valenc)

        class _C4TestEnc:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        return _C4TestEnc(valenc)
    else:
        traindata = datasets.load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                enc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if enc.input_ids.shape[1] > seqlen:
                    break
            s = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            inp = enc.input_ids[:, s:s + seqlen]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
