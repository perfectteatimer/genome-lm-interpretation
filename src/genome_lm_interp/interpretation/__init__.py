"""Explainable-AI attribution methods and consensus k-mer ranking."""

from .attributions import (
    integrated_gradients_kmers,
    smoothgrad_kmers,
    tp_rate_kmers,
)
from .attributions_seqcls import (
    integrated_gradients_kmers_seqcls,
    smoothgrad_kmers_seqcls,
)
from .pipelines import (
    interpret_from_config,
    interpret_sequence_task,
    interpret_token_task,
)
from .ranking import compare_methods, get_ranked_features

__all__ = [
    "integrated_gradients_kmers",
    "smoothgrad_kmers",
    "tp_rate_kmers",
    "integrated_gradients_kmers_seqcls",
    "smoothgrad_kmers_seqcls",
    "get_ranked_features",
    "compare_methods",
    "interpret_from_config",
    "interpret_token_task",
    "interpret_sequence_task",
]
