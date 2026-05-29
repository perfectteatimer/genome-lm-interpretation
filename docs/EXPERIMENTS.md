# Experiments & results

This document summarises the experiments originally explored in the notebooks
(`notebooks/`) and distilled into the package + configs. All runs fine-tune the
`LongSafari/hyenadna-tiny-1k-seqlen-hf` backbone.

The primary metrics are **binary F1** and **Matthews correlation coefficient
(MCC)**, both robust under strong class imbalance.

---

## 1. Z-DNA — per-nucleotide (token) classification

The genome is sliced into 100-bp windows; negatives are undersampled to a 3:1
ratio. HyenaDNA has no native token-classification head, so a custom
`HyenaDNAForTokenClassification` head is added and the pretrained Hyena backbone
is transferred into it (the sequence-classification `score` projection seeds the
token `classifier`).

A wide sweep was run over sequence length (100–1024), batch size (1–128), loss
weighting (class-weighted CE and Focal Loss), schedulers (linear, cosine,
constant-with-warmup, ReduceLROnPlateau), reverse-complement and overlap
augmentation, tokenisation with/without special tokens, and a 5-fold stratified
cross-validation.

**Best configuration — two-stage fine-tuning, F1 ≈ 0.64:**

| Stage | Epochs | Learning rate | What trains |
|-------|--------|---------------|-------------|
| 1 — frozen backbone | 3 | 1e-3 (head) | classification head only |
| 2 — full fine-tune | 6 | 1e-5 backbone / 5e-4 head | whole model (discriminative LRs) |

Loss: class-weighted CrossEntropy with weights `[0.7, 2.0]`; window length 100,
batch size 64, classic stratified 60/20/20 split. Discriminative learning rates
prevent catastrophic forgetting of the pretrained backbone while letting the
head adapt aggressively. Config: [`configs/zdna.yaml`](../configs/zdna.yaml).

## 2. G-quadruplex (G4) — per-nucleotide (token) classification

Same pipeline and data treatment as Z-DNA, with G4 labels (`data/g4.pkl`).

**Best configuration — single-stage, F1 ≈ 0.58:** batch size 32, lr 6e-4,
linear schedule with 10% warmup, 12 epochs. Config:
[`configs/g4.yaml`](../configs/g4.yaml).

## 3. Promoter — whole-sequence classification

Uses the public [GUE benchmark](https://huggingface.co/datasets/leannmlindsey/GUE)
(`prom_core_all` config), a binary promoter / non-promoter task. Here HyenaDNA's
native `AutoModelForSequenceClassification` head is fine-tuned directly.

**Best configuration — F1 ≈ 0.82:** 10 epochs, lr 6e-4, linear schedule, batch
size 64. Config: [`configs/promoter.yaml`](../configs/promoter.yaml).

---

## Parameter-efficient fine-tuning (LoRA)

LoRA adapters were applied to different module groups (full, mixer-only,
FFN-only, short-filter-only, embeddings-only) at various ranks. Because the
Hyena operator is already sub-quadratic, LoRA gave **no wall-clock speed-up** and
no quality gain over full fine-tuning on these tasks — a useful negative result.

---

## Interpretation (Explainable AI)

For each best model, token / k-mer importance is computed on correctly predicted
(true-positive) positions and aggregated into ranked 5-mer tables.

| Task | Methods | Output CSVs (`results/`) |
|------|---------|--------------------------|
| Z-DNA | Integrated Gradients, SmoothGrad, TP-rate baseline, consensus ranking | `z-dna/top_kmers_IG_final.csv`, `z-dna/top_kmers_smoothgrad_final.csv`, `z-dna/top_kmers_TP_importance_final.csv`, `z-dna/full_ranking_IG_vs_SG.csv` |
| Promoter | Integrated Gradients, SmoothGrad, consensus ranking | `promoter/5mers_interpretation_IGseqclass.csv`, `promoter/5mers_interpretation_smoothgrad.csv`, `promoter/ranking2_IG_vs_SG_corrected.csv` |

The consensus ranking expresses each method's scores as a percentage deviation
from its own mean and averages them, giving a method-agnostic ordering of the
most influential k-mers — which are then checked against known biological motifs.
