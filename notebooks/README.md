# Notebooks

These notebooks are the **original research record** — the exploratory work
behind the project, including the full hyper-parameter sweeps, PEFT/LoRA
experiments and interpretation prototyping.

- `hyenadna_zdna_experiments.ipynb` — Z-DNA token classification + XAI
- `hyenadna_g4_experiments.ipynb` — G-quadruplex token classification
- `hyenadna_promoter_experiments.ipynb` — promoter sequence classification + XAI

## Relationship to the package

The reusable logic from these notebooks has been refactored into the installable
`genome_lm_interp` package (`src/`), driven by configs in `configs/` and the CLI
scripts in `scripts/`. **For reproducing or extending the work, prefer the
package and CLI** (see the top-level [README](../README.md)).

> Note: the notebooks predate the `src/` → `src/genome_lm_interp/` reorganisation,
> so their inline `sys.path.append("../src")` + `from modeling_hyena import ...`
> imports point at the old flat layout. After `pip install -e .`, import the
> equivalents from the package instead, e.g.:
>
> ```python
> from genome_lm_interp.models import HyenaDNAForTokenClassification
> from genome_lm_interp.data import DNATokenClassificationDataset
> from genome_lm_interp.interpretation import integrated_gradients_kmers
> ```
