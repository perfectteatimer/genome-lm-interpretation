"""Attribution / Explainable-AI methods for the fine-tuned HyenaDNA models.

Two families of methods are implemented:

* **Token-classification** (Z-DNA, G4) -- gradient attributions over the input
  embeddings, accumulated per nucleotide (token) and per *k*-mer, restricted to
  true-positive predictions. Methods: Integrated Gradients, SmoothGrad
  (Saliency + NoiseTunnel), and a model-free true-positive-rate baseline.
* **Sequence-classification** (promoter) -- Integrated Gradients (via
  ``transformers-interpret``) and SmoothGrad (captum) over correctly predicted
  promoter sequences, aggregated into *k*-mer importance scores.

Each function returns a pandas DataFrame ranked by importance.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import pandas as pd
import torch
from captum.attr import IntegratedGradients, NoiseTunnel, Saliency
from tqdm.auto import trange

from ..utils import get_device


def _make_tp_forward(model, tp_mask):
    """Forward that sums the positive-class logits over true-positive tokens."""

    def forward(inputs_embeds, attention_mask):
        logits = model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True
        ).logits[..., 1]
        return (logits * tp_mask).sum(dim=-1)

    return forward


def _predict_tp_mask(model, input_ids, attention_mask, labels, device):
    """Return per-token predictions and the true-positive mask for a window."""
    with torch.no_grad():
        out = model(
            inputs_embeds=model.get_input_embeddings()(input_ids),
            attention_mask=attention_mask,
            return_dict=True,
        )
    preds = out.logits.argmax(dim=-1)[0]
    lbl = torch.tensor(labels, device=device).unsqueeze(0)
    tp_mask = ((preds == 1) & (lbl == 1)).float()
    return preds, tp_mask


def integrated_gradients_kmers(
    model,
    tokenizer,
    dataset,
    k: int = 5,
    n_steps: int = 50,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Integrated-Gradients *k*-mer importance over true-positive windows."""
    device = device or get_device()
    ig = IntegratedGradients(lambda x, y: x.sum(dim=-1))
    counts: dict = defaultdict(int)
    sums: dict = defaultdict(float)

    for i in trange(len(dataset), desc="IG k-mers"):
        sample = dataset[i]
        seq = sample["seq"]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        labels = sample["labels"].tolist()

        preds, tp_mask = _predict_tp_mask(model, input_ids, attention_mask, labels, device)
        if tp_mask.sum() == 0:
            continue

        inputs_embeds = model.get_input_embeddings()(input_ids)
        ig.forward_func = _make_tp_forward(model, tp_mask)
        attributions = ig.attribute(
            inputs=inputs_embeds,
            baselines=torch.zeros_like(inputs_embeds),
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
        )
        token_scores = attributions.sum(dim=-1)[0].detach().cpu().tolist()

        preds = preds.cpu().tolist()
        for start in range(len(seq) - k + 1):
            window_lbls = labels[start : start + k]
            window_preds = preds[start : start + k]
            if all(l == 1 and p == 1 for l, p in zip(window_lbls, window_preds)):
                kmer = seq[start : start + k]
                counts[kmer] += 1
                sums[kmer] += sum(token_scores[start : start + k])

    return _to_ranked_frame(counts, sums, score_col="AvgScore")


def smoothgrad_kmers(
    model,
    tokenizer,
    dataset,
    k: int = 5,
    nt_samples: int = 50,
    stdevs: float = 0.1,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """SmoothGrad (Saliency + NoiseTunnel) *k*-mer importance over true positives."""
    device = device or get_device()
    saliency = Saliency(model)
    noise_tunnel = NoiseTunnel(saliency)
    counts: dict = defaultdict(int)
    sums: dict = defaultdict(float)

    for i in trange(len(dataset), desc="SmoothGrad k-mers"):
        sample = dataset[i]
        seq = sample["seq"]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        labels = sample["labels"].tolist()

        preds, tp_mask = _predict_tp_mask(model, input_ids, attention_mask, labels, device)
        if tp_mask.sum() == 0:
            continue

        inputs_embeds = model.get_input_embeddings()(input_ids)
        saliency.forward_func = _make_tp_forward(model, tp_mask)
        attributions = noise_tunnel.attribute(
            inputs=inputs_embeds,
            nt_type="smoothgrad",
            nt_samples=nt_samples,
            nt_samples_batch_size=10,
            stdevs=stdevs,
            additional_forward_args=(attention_mask,),
        )
        token_scores = attributions.sum(dim=-1)[0].detach().cpu().tolist()

        preds = preds.cpu().tolist()
        for start in range(len(seq) - k + 1):
            window_lbls = labels[start : start + k]
            window_preds = preds[start : start + k]
            if all(l == 1 and p == 1 for l, p in zip(window_lbls, window_preds)):
                kmer = seq[start : start + k]
                counts[kmer] += 1
                sums[kmer] += sum(token_scores[start : start + k])

    return _to_ranked_frame(counts, sums, score_col="AvgScore")


def tp_rate_kmers(
    model,
    dataset,
    k: int = 5,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Model-free baseline: fraction of each *k*-mer's occurrences that are TP."""
    device = device or get_device()
    tp_counts: dict = defaultdict(int)
    occ_counts: dict = defaultdict(int)

    for i in trange(len(dataset), desc="TP-rate k-mers"):
        sample = dataset[i]
        seq = sample["seq"]
        labels = sample["labels"].tolist()
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(
                inputs_embeds=model.get_input_embeddings()(input_ids),
                attention_mask=attention_mask,
                return_dict=True,
            )
        preds = out.logits.argmax(dim=-1)[0].cpu().tolist()

        for start in range(len(seq) - k + 1):
            kmer = seq[start : start + k]
            occ_counts[kmer] += 1
            window_lbls = labels[start : start + k]
            window_preds = preds[start : start + k]
            if all(l == 1 and p == 1 for l, p in zip(window_lbls, window_preds)):
                tp_counts[kmer] += 1

    records = [
        {
            "kmer": kmer,
            "TP_count": tp_counts.get(kmer, 0),
            "Occurrences": total,
            "Importance": tp_counts.get(kmer, 0) / total,
        }
        for kmer, total in occ_counts.items()
    ]
    return pd.DataFrame(records).sort_values("Importance", ascending=False)


def _to_ranked_frame(counts: dict, sums: dict, score_col: str = "AvgScore") -> pd.DataFrame:
    """Build a ``kmer / Count / AvgScore`` frame sorted by mean attribution."""
    records = [
        {"kmer": kmer, "Count": count, score_col: sums[kmer] / count}
        for kmer, count in counts.items()
    ]
    return pd.DataFrame(records).sort_values(score_col, ascending=False)
