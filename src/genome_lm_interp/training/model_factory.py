"""Model construction and weight transfer for the HyenaDNA classification heads.

HyenaDNA ships pretrained sequence-classification weights but no
token-classification head. For the per-nucleotide tasks (Z-DNA, G4) we build a
:class:`HyenaDNAForTokenClassification`, copy the pretrained Hyena backbone into
it, and reuse the sequence-classification ``score`` projection as the initial
token ``classifier``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..models import HyenaDNAForTokenClassification
from ..utils import get_device

logger = logging.getLogger(__name__)


def build_token_model_from_pretrained(
    base_model_name: str = "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    num_labels: int = 2,
    class_weights: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[HyenaDNAForTokenClassification, PreTrainedTokenizer]:
    """Create a token-classification model seeded from pretrained HyenaDNA.

    Returns the model (on ``device``) and its tokenizer.
    """
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    config.num_labels = num_labels
    if class_weights is not None:
        config.class_weights = class_weights

    seq_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, config=config, trust_remote_code=True
    )
    token_model = HyenaDNAForTokenClassification(config)

    # Copy the Hyena backbone verbatim and rename score -> classifier.
    seq_sd = seq_model.state_dict()
    token_sd = token_model.state_dict()
    for key, value in seq_sd.items():
        if key.startswith("hyena."):
            token_sd[key] = value.clone()
    token_sd["classifier.weight"] = seq_sd["score.weight"].clone()

    missing, unexpected = token_model.load_state_dict(token_sd, strict=False)
    logger.info("Token model loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected))

    return token_model.to(device), tokenizer


def load_token_model(
    model_dir: str,
    num_labels: int = 2,
    class_weights: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[HyenaDNAForTokenClassification, PreTrainedTokenizer]:
    """Load a fine-tuned token-classification checkpoint from ``model_dir``."""
    device = device or get_device()
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config.num_labels = num_labels
    if class_weights is not None:
        config.class_weights = class_weights

    model = HyenaDNAForTokenClassification(config)
    state_dict = torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("Checkpoint loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    _patch_special_tokens(tokenizer)
    return model.to(device).eval(), tokenizer


def build_sequence_model(
    base_model_name: str = "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    num_labels: int = 2,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Build a HyenaDNA sequence-classification model + tokenizer (promoter)."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, trust_remote_code=True, num_labels=num_labels
    )
    return model, tokenizer


def _patch_special_tokens(tokenizer: PreTrainedTokenizer) -> None:
    """Ensure CLS/SEP wrapping is applied (needed for some saved tokenizers)."""

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens.__get__(
        tokenizer, type(tokenizer)
    )
