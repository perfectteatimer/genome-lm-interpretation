"""Interpreting genomic language models.

A small, reproducible toolkit for fine-tuning the HyenaDNA genomic language
model on specialised DNA-structure tasks (Z-DNA, G-quadruplexes, promoters)
and interpreting its predictions with Explainable-AI attribution methods
(Integrated Gradients, SmoothGrad, Saliency).

The public API mirrors the project's pipeline stages:

* ``genome_lm_interp.models``        -- HyenaDNA port and classification heads
* ``genome_lm_interp.data``          -- genome windowing, datasets, loaders
* ``genome_lm_interp.training``      -- model factory, metrics, training loops
* ``genome_lm_interp.interpretation``-- attribution methods and k-mer ranking
"""

from .models import (
    HyenaConfig,
    HyenaDNAForSequenceClassification,
    HyenaDNAForTokenClassification,
)

__version__ = "0.1.0"

__all__ = [
    "HyenaConfig",
    "HyenaDNAForSequenceClassification",
    "HyenaDNAForTokenClassification",
    "__version__",
]
