#!/usr/bin/env python
"""Fine-tune HyenaDNA on a genomic task described by a YAML config.

Examples
--------
    python scripts/train.py --config configs/zdna.yaml
    python scripts/train.py --config configs/g4.yaml
    python scripts/train.py --config configs/promoter.yaml

The task family (per-nucleotide vs whole-sequence) is selected automatically
from ``data.task_type`` in the config. Equivalent console script: ``genome-lm-train``.
"""

from genome_lm_interp.cli import train_main

if __name__ == "__main__":
    train_main()
