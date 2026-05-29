"""HyenaDNA model port and task-specific classification heads.

This subpackage vendors the official HyenaDNA Hugging Face port
(`configuration_hyena.py`, `modeling_hyena.py`) and exposes the model
variants used throughout the project, including a custom
``HyenaDNAForTokenClassification`` head for per-nucleotide prediction.
"""

from .configuration_hyena import HyenaConfig
from .modeling_hyena import (
    HyenaDNAForCausalLM,
    HyenaDNAForSequenceClassification,
    HyenaDNAForTokenClassification,
    HyenaDNAModel,
    HyenaDNAPreTrainedModel,
)

__all__ = [
    "HyenaConfig",
    "HyenaDNAModel",
    "HyenaDNAPreTrainedModel",
    "HyenaDNAForCausalLM",
    "HyenaDNAForSequenceClassification",
    "HyenaDNAForTokenClassification",
]
