"""PyTorch datasets for per-nucleotide (token) classification of DNA."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils import data
from transformers import PreTrainedTokenizer


class DNATokenClassificationDataset(data.Dataset):
    """Per-nucleotide labelled DNA windows for HyenaDNA token classification.

    Each item tokenises a DNA window and aligns the per-nucleotide labels to the
    resulting tokens. HyenaDNA uses single-nucleotide tokenisation, so character
    labels map one-to-one to non-special tokens; special tokens (CLS/SEP/PAD)
    and any truncated tail receive the ``-100`` ignore index.
    """

    def __init__(
        self,
        chroms: List[str],
        dna_source: Dict[str, str],
        labels_source: Dict[str, torch.Tensor],
        intervals: List[Tuple[str, int, int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ):
        self.chroms = chroms
        self.dna_source = dna_source
        self.labels_source = labels_source
        self.intervals = intervals
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.intervals)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        chrom, start, end = self.intervals[idx][:3]
        seq = self.dna_source[chrom][start:end].upper()
        char_labels = self.labels_source[chrom][start:end]  # shape (L,)

        enc = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_length + 1,
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        special_tokens_mask = torch.tensor(enc["special_tokens_mask"], dtype=torch.long)

        # Align one character label per non-special token; -100 elsewhere.
        labels_by_tok: List[int] = []
        char_ptr = 0
        for is_special in special_tokens_mask.tolist():
            if is_special:
                labels_by_tok.append(-100)
            elif char_ptr < len(char_labels):
                labels_by_tok.append(int(char_labels[char_ptr]))
                char_ptr += 1
            else:
                labels_by_tok.append(-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels_by_tok, dtype=torch.long),
            "seq": seq,
        }
