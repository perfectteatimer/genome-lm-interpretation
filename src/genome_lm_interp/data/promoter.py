"""Promoter sequence-classification data from the GUE benchmark.

The promoter task uses the public ``leannmlindsey/GUE`` dataset (config
``prom_core_all``), a binary promoter / non-promoter sequence-classification
benchmark. Sequences are tokenised with the HyenaDNA tokenizer.
"""

from __future__ import annotations

from typing import Tuple

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizer


def load_promoter_dataset(
    hf_dataset: str = "leannmlindsey/GUE",
    hf_config: str = "prom_core_all",
) -> DatasetDict:
    """Load the raw promoter dataset (``train`` / ``dev`` / ``test`` splits)."""
    return load_dataset(hf_dataset, hf_config)


def build_tokenized_datasets(
    tokenizer: PreTrainedTokenizer,
    hf_dataset: str = "leannmlindsey/GUE",
    hf_config: str = "prom_core_all",
) -> Tuple[DatasetDict, DatasetDict, DatasetDict]:
    """Return tokenised ``(train, val, test)`` datasets ready for the Trainer.

    The GUE ``dev`` split is used for validation.
    """
    raw = load_promoter_dataset(hf_dataset, hf_config)

    def preprocess(examples):
        return tokenizer(examples["sequence"], truncation=True, padding=True)

    tokenized = raw.map(preprocess, batched=True)
    return tokenized["train"], tokenized["dev"], tokenized["test"]
