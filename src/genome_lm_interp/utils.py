"""Small shared helpers: device selection, reproducibility, sparse-vector shim."""

from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return the best available torch device (CUDA if present, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def install_sparse_vector_aliases() -> None:
    """Register import aliases needed to unpickle the Z-DNA sparse arrays.

    The Z-DNA labels and genomic feature tracks were serialised with the
    ``Sparse_vector`` package (https://github.com/Nazar1997/Sparse_vector),
    whose importable module name is the lowercase ``sparse_vector``. We expose
    both spellings under ``sys.modules`` so that ``joblib.load`` can resolve the
    pickled class references regardless of the casing used at dump time.
    """
    try:
        import sparse_vector  # noqa: WPS433 (runtime optional dependency)
        import sparse_vector.sparse_vector  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - clear, actionable message
        raise ImportError(
            "The 'sparse_vector' package is required to load the Z-DNA data. "
            "Install it with: pip install "
            "git+https://github.com/Nazar1997/Sparse_vector.git"
        ) from exc

    sys.modules.setdefault("Sparse_vector", sparse_vector)
    sys.modules.setdefault("Sparse_vector.sparse_vector", sparse_vector.sparse_vector)
