"""Lightweight smoke tests that don't require GPUs, weights or genomic data."""

from pathlib import Path

import pandas as pd

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_package_imports():
    import genome_lm_interp  # noqa: F401
    from genome_lm_interp.models import HyenaDNAForTokenClassification  # noqa: F401

    assert genome_lm_interp.__version__


def test_configs_load():
    from genome_lm_interp.config import ExperimentConfig

    for name in ("zdna", "g4", "promoter"):
        cfg = ExperimentConfig.from_yaml(CONFIGS / f"{name}.yaml")
        assert cfg.name == name
        assert cfg.data.task_type in {"token", "sequence"}


def test_ranking_consensus():
    from genome_lm_interp.interpretation import get_ranked_features

    frame = pd.DataFrame({"kmer": ["AAAAA", "CCCCC"], "IG_imp": [2.0, 1.0]})
    ranked = get_ranked_features(frame)
    assert list(ranked["kmer"]) == ["AAAAA", "CCCCC"]
    assert "mean_dev" in ranked.columns
