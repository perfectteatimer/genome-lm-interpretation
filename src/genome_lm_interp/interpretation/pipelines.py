"""End-to-end interpretation pipelines wired from an :class:`ExperimentConfig`.

Produces the k-mer attribution CSVs (Integrated Gradients, SmoothGrad,
TP-rate) and a consensus IG-vs-SmoothGrad ranking for a fine-tuned checkpoint.
"""

from __future__ import annotations

import logging
import os

from ..config import ExperimentConfig
from ..data import (
    DNATokenClassificationDataset,
    balance_intervals,
    build_labeled_windows,
    load_genome,
    load_label_track,
    load_promoter_dataset,
    split_intervals,
)
from ..training.model_factory import load_token_model
from ..utils import get_device, set_seed
from .attributions import integrated_gradients_kmers, smoothgrad_kmers, tp_rate_kmers
from .attributions_seqcls import (
    integrated_gradients_kmers_seqcls,
    smoothgrad_kmers_seqcls,
)
from .ranking import compare_methods

logger = logging.getLogger(__name__)


def _resolve(data_root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(data_root, path)


def interpret_token_task(cfg: ExperimentConfig, model_dir: str, output_dir: str, k: int = 5) -> None:
    """Run IG / SmoothGrad / TP-rate k-mer attribution for a token model."""
    set_seed(cfg.training.seed)
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    data_cfg = cfg.data

    dna = load_genome(data_cfg.chroms, _resolve(data_cfg.data_root, data_cfg.dna_dir))
    labels = load_label_track(_resolve(data_cfg.data_root, data_cfg.labels_file))
    blacklist = load_label_track(_resolve(data_cfg.data_root, data_cfg.blacklist_file))

    positives, negatives = build_labeled_windows(
        data_cfg.chroms, dna, labels, blacklist, width=data_cfg.window_width
    )
    intervals = balance_intervals(positives, negatives, neg_pos_ratio=data_cfg.neg_pos_ratio)
    _, _, test_iv = split_intervals(
        intervals, test_size=data_cfg.test_size, val_size=data_cfg.val_size, seed=cfg.training.seed
    )

    model, tokenizer = load_token_model(
        model_dir,
        num_labels=cfg.model.num_labels,
        class_weights=cfg.model.class_weights,
        device=device,
    )
    test_ds = DNATokenClassificationDataset(
        chroms=data_cfg.chroms,
        dna_source=dna,
        labels_source=labels,
        intervals=[(c, s, e) for c, s, e, _ in test_iv],
        tokenizer=tokenizer,
        max_length=cfg.model.max_length,
    )

    df_ig = integrated_gradients_kmers(model, tokenizer, test_ds, k=k, device=device)
    df_ig.to_csv(os.path.join(output_dir, "top_kmers_IG_final.csv"), index=False)

    df_sg = smoothgrad_kmers(model, tokenizer, test_ds, k=k, device=device)
    df_sg.to_csv(os.path.join(output_dir, "top_kmers_smoothgrad_final.csv"), index=False)

    df_tp = tp_rate_kmers(model, test_ds, k=k, device=device)
    df_tp.to_csv(os.path.join(output_dir, "top_kmers_TP_importance_final.csv"), index=False)

    ranking = compare_methods(
        df_ig.rename(columns={"AvgScore": "IG_imp"}),
        df_sg.rename(columns={"AvgScore": "SG_imp"}),
    )
    ranking.to_csv(os.path.join(output_dir, "full_ranking_IG_vs_SG.csv"), index=False)
    logger.info("Wrote attribution CSVs to %s", output_dir)


def interpret_sequence_task(cfg: ExperimentConfig, model_dir: str, output_dir: str, k: int = 5) -> None:
    """Run IG / SmoothGrad k-mer attribution for a promoter sequence model."""
    from transformers import AutoTokenizer

    from ..models import HyenaDNAForSequenceClassification

    set_seed(cfg.training.seed)
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()

    model = HyenaDNAForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    dev_split = load_promoter_dataset(cfg.data.hf_dataset, cfg.data.hf_config)["dev"]

    df_ig = integrated_gradients_kmers_seqcls(model, tokenizer, dev_split, k=k, device=device)
    df_ig.to_csv(os.path.join(output_dir, "5mers_interpretation_IGseqclass.csv"), index=False)

    df_sg = smoothgrad_kmers_seqcls(model, tokenizer, dev_split, k=k, device=device)
    df_sg.to_csv(os.path.join(output_dir, "5mers_interpretation_smoothgrad.csv"), index=False)

    ranking = compare_methods(
        df_ig.rename(columns={"avg_score": "IG_imp"}),
        df_sg.rename(columns={"avg_score": "SG_imp"}),
        positive_only=True,
    )
    ranking.to_csv(os.path.join(output_dir, "ranking2_IG_vs_SG_corrected.csv"), index=False)
    logger.info("Wrote attribution CSVs to %s", output_dir)


def interpret_from_config(cfg: ExperimentConfig, model_dir: str, output_dir: str, k: int = 5) -> None:
    """Dispatch to the right interpretation pipeline based on task type."""
    if cfg.data.task_type == "token":
        interpret_token_task(cfg, model_dir, output_dir, k=k)
    elif cfg.data.task_type == "sequence":
        interpret_sequence_task(cfg, model_dir, output_dir, k=k)
    else:
        raise ValueError(f"Unknown task_type: {cfg.data.task_type!r}")
