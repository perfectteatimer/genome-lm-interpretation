"""Consensus ranking of *k*-mers across attribution methods.

To compare attribution methods (e.g. Integrated Gradients vs SmoothGrad) on a
common scale, each method's scores are converted to a percentage deviation from
that method's mean, and the per-*k*-mer deviations are averaged into a single
consensus rank.
"""

from __future__ import annotations

import pandas as pd


def get_ranked_features(frame: pd.DataFrame, id_col: str = "kmer") -> pd.DataFrame:
    """Rank rows by mean percentage deviation across all numeric score columns."""
    numeric = frame.drop(columns=[id_col])
    means = numeric.mean()
    deviation = numeric.sub(means).div(means) * 100
    deviation[id_col] = frame[id_col]
    deviation["mean_dev"] = deviation.drop(columns=[id_col]).mean(axis=1)
    return deviation[[id_col, "mean_dev"]].sort_values("mean_dev", ascending=False)


def compare_methods(
    df_ig: pd.DataFrame,
    df_sg: pd.DataFrame,
    ig_col: str = "IG_imp",
    sg_col: str = "SG_imp",
    id_col: str = "kmer",
    positive_only: bool = False,
) -> pd.DataFrame:
    """Merge IG and SmoothGrad scores and return a consensus ranking.

    When ``positive_only`` is set, only *k*-mers with a positive score under both
    methods are kept before ranking (used for the promoter comparison).
    """
    merged = df_ig[[id_col, ig_col]].merge(df_sg[[id_col, sg_col]], on=id_col)
    if positive_only:
        merged = merged[(merged[ig_col] > 0) & (merged[sg_col] > 0)]
    return get_ranked_features(merged, id_col=id_col)
