# Interpreting Genomic Language Models

> **Identifying Biological Relations in the Human Genome Through Interpretation of Language Models**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model: HyenaDNA](https://img.shields.io/badge/model-HyenaDNA-8A2BE2.svg)](https://arxiv.org/abs/2306.15794)
[![HSE best project](https://img.shields.io/badge/HSE-best%20project-red.svg)](https://cs.hse.ru/cppr/best_projects/genome_lm_interpretation)

Fine-tuning and **interpreting** the [HyenaDNA][hyenadna-paper] genomic language
model on specialised DNA-structure tasks, and using Explainable-AI (XAI)
attribution methods to verify that the model's predictions rest on biologically
meaningful sequence motifs.

> 🏆 Selected as one of the **best coursework projects** of the year at the
> HSE Faculty of Computer Science.
> Project showcase & defence talk:
> **<https://cs.hse.ru/cppr/best_projects/genome_lm_interpretation>**

---

## Overview

DNA can fold into non-canonical secondary structures (Z-DNA, G-quadruplexes)
and contains regulatory regions (promoters) that are central to gene expression.
This project asks two questions:

1. **Can a pretrained genomic language model recognise these structures?**
   We fine-tune HyenaDNA on three tasks and report F1 / MCC.
2. **Does it do so for the right reasons?**
   We apply Integrated Gradients, SmoothGrad and Saliency to attribute the
   model's predictions back to individual nucleotides and *k*-mers, then compare
   the most influential motifs against known biology.

### Tasks

| Task | Type | Data | Best F1 |
|------|------|------|:------:|
| **Z-DNA** | per-nucleotide (token) classification | hg38 + Z-DNA label track | **≈ 0.64** |
| **G-quadruplex (G4)** | per-nucleotide (token) classification | hg38 + G4 label track | **≈ 0.58** |
| **Promoter** | whole-sequence classification | [GUE benchmark][gue] (`prom_core_all`) | **≈ 0.82** |

HyenaDNA ships no token-classification head, so for the per-nucleotide tasks we
add a custom `HyenaDNAForTokenClassification` head and transfer the pretrained
Hyena backbone into it. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the
full sweep (sequence length, batch size, loss weighting, schedulers,
augmentation, 5-fold CV, and a LoRA/PEFT study).

---

## Repository structure

```
.
├── configs/                  # YAML experiment configs (one per task)
│   ├── zdna.yaml
│   ├── g4.yaml
│   └── promoter.yaml
├── scripts/                  # Thin CLI entry points
│   ├── train.py
│   └── interpret.py
├── src/genome_lm_interp/     # Installable package
│   ├── models/               # HyenaDNA port + classification heads
│   ├── data/                 # genome windowing, datasets, promoter loader
│   ├── training/             # model factory, metrics, training pipelines
│   ├── interpretation/       # IG / SmoothGrad / TP-rate + k-mer ranking
│   ├── config.py             # typed config loaded from YAML
│   └── cli.py                # console entry points
├── notebooks/                # Original research notebooks (see notebooks/README.md)
├── results/                  # Attribution CSVs (k-mer importance rankings)
├── report/                   # Coursework paper & defence slides
├── data/                     # Small assets (G4 + blacklist tracks); large data ignored
└── docs/EXPERIMENTS.md       # Detailed experiment write-up
```

---

## Installation

```bash
git clone https://github.com/perfectteatimer/interpreting-genomic-language-models.git
cd interpreting-genomic-language-models

python -m venv .venv && source .venv/bin/activate    # optional but recommended
```

Editable install with all extras (training + interpretation + genomic data utils):

```bash
pip install -e ".[interpret,data]"
```

Or install the pinned requirements directly:

```bash
pip install -r requirements.txt
```

> **Python ≥ 3.10** is required. A CUDA-capable GPU is strongly recommended for
> training; inference and interpretation run on CPU but are slow.

---

## Data & model weights

Small label tracks are included in the repo:

- `data/g4.pkl` — per-nucleotide G-quadruplex labels
- `data/blacklist_hg38_v2.pkl` — ENCODE-style blacklist mask

The following are **not** version-controlled (size) and must be obtained or
regenerated — paths are configurable in `configs/*.yaml`:

| Asset | Expected location | Source |
|-------|-------------------|--------|
| hg38 DNA fragments | `data/z_dna/hg38_dna/` | [`z_dna`][zdna-repo] preprocessing |
| Z-DNA label track | `data/z_dna/hg38_zdna/sparse/ZDNA_cousine.pkl` | [`z_dna`][zdna-repo] |
| Fine-tuned checkpoints | `data/models/...` | produced by `scripts/train.py` |

The Z-DNA tracks are serialised with [`Sparse_vector`][sparse-repo]; it is
installed by the `[data]` extra and loaded transparently. The promoter task needs
no local data — the GUE dataset is pulled from the Hugging Face Hub on first run.

---

## Usage

### Train

```bash
# Per-nucleotide tasks (require the hg38 assembly under data/, see above)
python scripts/train.py --config configs/zdna.yaml
python scripts/train.py --config configs/g4.yaml

# Whole-sequence task (downloads the GUE dataset automatically)
python scripts/train.py --config configs/promoter.yaml
```

The task family is chosen automatically from `data.task_type` in the config.
Checkpoints and tokenizer are written to `training.output_dir`; test-set metrics
are printed as JSON. The console scripts `genome-lm-train` / `genome-lm-interpret`
are equivalent after install.

### Interpret

```bash
python scripts/interpret.py --config configs/zdna.yaml \
    --model-dir results/zdna/two_stage \
    --output-dir results/z-dna
```

This writes ranked *k*-mer importance tables (Integrated Gradients, SmoothGrad,
and — for token tasks — a true-positive-rate baseline) plus a consensus
IG-vs-SmoothGrad ranking. Pre-computed outputs live in [`results/`](results/).

### As a library

```python
from genome_lm_interp.config import ExperimentConfig
from genome_lm_interp.training import train_from_config

cfg = ExperimentConfig.from_yaml("configs/promoter.yaml")
metrics = train_from_config(cfg)
print(metrics)
```

---

## Method notes

- **Token classification.** 100-bp windows, negatives undersampled 3:1, labels
  aligned one-per-nucleotide token with `-100` on special/padding positions.
  The best Z-DNA model uses **two-stage discriminative fine-tuning** (frozen-head
  warm-up → full fine-tune with backbone LR 1e-5, head LR 5e-4).
- **Class imbalance** is handled with weighted CrossEntropy (default `[0.7, 2.0]`,
  configurable via `model.class_weights`).
- **Interpretation** is restricted to true-positive predictions, so attributions
  describe *why the model was correct*, and aggregated to 5-mers for comparison
  with known motifs.
- **LoRA/PEFT** was evaluated but gave no speed-up (the Hyena operator is already
  sub-quadratic) — documented as a negative result in `docs/EXPERIMENTS.md`.

---

## Citation

If you use this work, please cite it (see [`CITATION.cff`](CITATION.cff)) and the
underlying model:

```bibtex
@article{nguyen2023hyenadna,
  title   = {HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution},
  author  = {Nguyen, Eric and Poli, Michael and Faizi, Marjan and others},
  journal = {arXiv preprint arXiv:2306.15794},
  year    = {2023}
}
```

## Acknowledgments

- The authors of [HyenaDNA][hyenadna-paper] for the long-range genomic backbone.
- [`z_dna`][zdna-repo] and [`Sparse_vector`][sparse-repo] for the Z-DNA data tooling.
- HSE Faculty of Computer Science.

## License

Released under the MIT License — see [LICENSE](LICENSE).

[hyenadna-paper]: https://arxiv.org/abs/2306.15794
[gue]: https://huggingface.co/datasets/leannmlindsey/GUE
[zdna-repo]: https://github.com/vladislareon/z_dna
[sparse-repo]: https://github.com/Nazar1997/Sparse_vector
