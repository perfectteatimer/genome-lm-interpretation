"""Attribution methods for the promoter sequence-classification model.

Unlike the per-nucleotide tasks, here a single label is assigned to the whole
sequence. We attribute the positive ("promoter") logit back to the input tokens
on correctly-classified promoters and aggregate the per-token scores into
*k*-mer importance values.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import pandas as pd
import torch
from captum.attr import IntegratedGradients, NoiseTunnel
from tqdm.auto import tqdm

from ..utils import get_device


def integrated_gradients_kmers_seqcls(
    model,
    tokenizer,
    dataset,
    k: int = 5,
    n_steps: int = 50,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """IG *k*-mer importance via ``transformers-interpret`` on true positives."""
    from transformers_interpret import SequenceClassificationExplainer

    device = device or get_device()
    explainer = SequenceClassificationExplainer(model, tokenizer)
    counts: dict = defaultdict(int)
    sums: dict = defaultdict(float)

    for example in tqdm(dataset, desc="IG k-mers (seq-cls)"):
        if example["label"] != 1:
            continue
        attributions = explainer(example["sequence"], n_steps=n_steps)
        if explainer.predicted_class_index != 1:
            continue

        tokens = [tok for tok, _ in attributions if tok not in tokenizer.all_special_tokens]
        scores = [score for tok, score in attributions if tok not in tokenizer.all_special_tokens]
        for i in range(len(tokens) - k + 1):
            kmer = "".join(tokens[i : i + k])
            counts[kmer] += 1
            sums[kmer] += sum(scores[i : i + k])

    return _to_kmer_frame(counts, sums)


def smoothgrad_kmers_seqcls(
    model,
    tokenizer,
    dataset,
    k: int = 5,
    nt_samples: int = 50,
    stdevs: float = 0.02,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """SmoothGrad (captum IG + NoiseTunnel) *k*-mer importance on true positives."""
    device = device or get_device()
    embedding_layer = model.get_input_embeddings()
    counts: dict = defaultdict(int)
    sums: dict = defaultdict(float)

    def forward(inputs_embeds, attention_mask):
        return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

    ig = IntegratedGradients(forward)
    noise_tunnel = NoiseTunnel(ig)

    for example in tqdm(dataset, desc="SmoothGrad k-mers (seq-cls)"):
        if example["label"] != 1:
            continue
        enc = tokenizer(example["sequence"], return_tensors="pt", truncation=True).to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        with torch.no_grad():
            pred = model(**enc).logits.argmax(-1).item()
        if pred != 1:
            continue

        inputs_embeds = embedding_layer(input_ids)
        attributions = noise_tunnel.attribute(
            inputs=inputs_embeds,
            nt_type="smoothgrad",
            nt_samples=nt_samples,
            stdevs=stdevs,
            target=1,
            additional_forward_args=(attention_mask,),
        )
        scores = attributions.sum(dim=-1)[0].detach().cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        keep = [
            (tok, score)
            for tok, score in zip(tokens, scores)
            if tok not in tokenizer.all_special_tokens
        ]
        toks = [t for t, _ in keep]
        vals = [s for _, s in keep]
        for i in range(len(toks) - k + 1):
            kmer = "".join(toks[i : i + k])
            counts[kmer] += 1
            sums[kmer] += sum(vals[i : i + k])

    return _to_kmer_frame(counts, sums)


def _to_kmer_frame(counts: dict, sums: dict) -> pd.DataFrame:
    records = [
        {"kmer": kmer, "count": count, "avg_score": sums[kmer] / count}
        for kmer, count in counts.items()
    ]
    return pd.DataFrame(records).sort_values("avg_score", ascending=False)
