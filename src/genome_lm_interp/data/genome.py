"""Genome assembly, windowing and labelling for token-classification tasks.

The Z-DNA and G-quadruplex pipelines operate on the full hg38 assembly. This
module loads per-chromosome DNA sequences and label tracks, slices them into
fixed-width windows, filters out low-quality / blacklisted regions, and
produces balanced train/val/test interval lists.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm, trange

from ..utils import install_sparse_vector_aliases

# An interval is (chromosome, start, end, label).
Interval = Tuple[str, int, int, int]


def load_chrom_sequence(chrom: str, dna_dir: str) -> str:
    """Load and concatenate all joblib fragments for a single chromosome.

    Sequence fragments are stored as ``{chrom}_<n>`` files inside ``dna_dir``
    and concatenated in sorted order to reconstruct the full chromosome string.
    """
    files = sorted(f for f in os.listdir(dna_dir) if f.startswith(f"{chrom}_"))
    if not files:
        raise FileNotFoundError(
            f"No DNA fragments for {chrom!r} found in {dna_dir!r}. "
            "See the README 'Data' section for how to obtain the hg38 assembly."
        )
    return "".join(load(os.path.join(dna_dir, f)) for f in files)


def load_genome(chroms: Sequence[str], dna_dir: str) -> Dict[str, str]:
    """Return ``{chrom: sequence}`` for every requested chromosome."""
    return {chrom: load_chrom_sequence(chrom, dna_dir) for chrom in tqdm(chroms, desc="Loading DNA")}


def load_label_track(path: str) -> Dict[str, object]:
    """Load a pickled per-nucleotide label/blacklist track keyed by chromosome.

    Handles the ``Sparse_vector`` unpickling shim transparently.
    """
    install_sparse_vector_aliases()
    return load(path)


def build_labeled_windows(
    chroms: Sequence[str],
    dna: Dict[str, str],
    labels: Dict[str, object],
    blacklist: Dict[str, object],
    width: int = 100,
    seed: int = 10,
) -> Tuple[List[Interval], List[Interval]]:
    """Slice the genome into non-overlapping windows and label them.

    A window is labelled positive (1) if the label track contains any positive
    nucleotide inside it, negative (0) otherwise. Windows are discarded when
    more than half of their bases are ``N`` or when they overlap the blacklist.

    Returns ``(positives, negatives)`` interval lists.
    """
    np.random.seed(seed)
    positives: List[Interval] = []
    negatives: List[Interval] = []

    for chrom in chroms:
        chrom_len = labels[chrom].shape
        for start in trange(0, chrom_len - width, width, desc=f"Windows {chrom}", leave=False):
            end = min(start + width, chrom_len)
            n_count = sum(base == "N" for base in dna[chrom][start:end])
            bl_count = blacklist[chrom][start:end].sum()
            if n_count > width / 2 or bl_count > 0:
                continue
            label = 1 if labels[chrom][start:end].any() else 0
            (positives if label == 1 else negatives).append((chrom, int(start), int(end), label))

    return positives, negatives


def balance_intervals(
    positives: List[Interval],
    negatives: List[Interval],
    neg_pos_ratio: int = 3,
    seed: int = 10,
) -> List[Interval]:
    """Undersample negatives to ``neg_pos_ratio`` per positive and combine."""
    np.random.seed(seed)
    n_neg = min(len(positives) * neg_pos_ratio, len(negatives))
    chosen = np.random.choice(len(negatives), size=n_neg, replace=False)
    sampled_negatives = [negatives[i] for i in chosen]
    return positives + sampled_negatives


def split_intervals(
    intervals: List[Interval],
    test_size: float = 0.2,
    val_size: float = 0.25,
    seed: int = 42,
) -> Tuple[List[Interval], List[Interval], List[Interval]]:
    """Stratified train/val/test split by ``(label, chromosome)`` strata."""
    strata = [f"{label}_{chrom}" for chrom, _, _, label in intervals]
    train_val, test = train_test_split(
        intervals, test_size=test_size, stratify=strata, random_state=seed
    )
    train_val_strata = [f"{label}_{chrom}" for chrom, _, _, label in train_val]
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=train_val_strata, random_state=seed
    )
    return train, val, test
