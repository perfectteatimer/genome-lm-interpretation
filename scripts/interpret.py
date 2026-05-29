#!/usr/bin/env python
"""Run Explainable-AI attribution for a fine-tuned HyenaDNA checkpoint.

Produces per-k-mer importance CSVs (Integrated Gradients, SmoothGrad, and,
for token tasks, a true-positive-rate baseline) plus a consensus ranking.

Examples
--------
    python scripts/interpret.py --config configs/zdna.yaml \
        --model-dir results/zdna/two_stage --output-dir results/z-dna

    python scripts/interpret.py --config configs/promoter.yaml \
        --model-dir results/promoter/finetune --output-dir results/promoter

Equivalent console script: ``genome-lm-interpret``.
"""

from genome_lm_interp.cli import interpret_main

if __name__ == "__main__":
    interpret_main()
