"""End-to-end training pipelines wired from an :class:`ExperimentConfig`.

These functions encapsulate the full path from raw data to a saved, evaluated
checkpoint, so the CLI entry points (``scripts/train.py``) remain a thin shell.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

from transformers import Trainer

from ..config import ExperimentConfig
from ..data import (
    DNATokenClassificationDataset,
    balance_intervals,
    build_labeled_windows,
    build_tokenized_datasets,
    load_genome,
    load_label_track,
    split_intervals,
)
from ..utils import get_device, set_seed
from .metrics import sequence_classification_metrics, token_classification_metrics
from .model_factory import build_sequence_model, build_token_model_from_pretrained
from .trainer import build_training_args, run_finetune, two_stage_finetune

logger = logging.getLogger(__name__)


def _resolve(data_root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(data_root, path)


def train_token_task(cfg: ExperimentConfig) -> Dict[str, float]:
    """Train + evaluate a per-nucleotide task (Z-DNA, G4). Returns test metrics."""
    set_seed(cfg.training.seed)
    device = get_device()
    data_cfg = cfg.data

    logger.info("Loading genome and label tracks ...")
    dna = load_genome(data_cfg.chroms, _resolve(data_cfg.data_root, data_cfg.dna_dir))
    labels = load_label_track(_resolve(data_cfg.data_root, data_cfg.labels_file))
    blacklist = load_label_track(_resolve(data_cfg.data_root, data_cfg.blacklist_file))

    logger.info("Building and balancing windows ...")
    positives, negatives = build_labeled_windows(
        data_cfg.chroms, dna, labels, blacklist, width=data_cfg.window_width
    )
    logger.info("Positives: %d | Negatives: %d", len(positives), len(negatives))
    intervals = balance_intervals(positives, negatives, neg_pos_ratio=data_cfg.neg_pos_ratio)
    train_iv, val_iv, test_iv = split_intervals(
        intervals, test_size=data_cfg.test_size, val_size=data_cfg.val_size, seed=cfg.training.seed
    )

    model, tokenizer = build_token_model_from_pretrained(
        cfg.model.base_model_name,
        num_labels=cfg.model.num_labels,
        class_weights=cfg.model.class_weights,
        device=device,
    )

    def make_ds(iv):
        return DNATokenClassificationDataset(
            chroms=data_cfg.chroms,
            dna_source=dna,
            labels_source=labels,
            intervals=[(c, s, e) for c, s, e, _ in iv],
            tokenizer=tokenizer,
            max_length=cfg.model.max_length,
        )

    train_ds, val_ds, test_ds = make_ds(train_iv), make_ds(val_iv), make_ds(test_iv)

    finetune = two_stage_finetune if cfg.training.two_stage else run_finetune
    trainer = finetune(
        model, train_ds, val_ds, tokenizer, token_classification_metrics, cfg.training
    )

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)

    metrics = trainer.evaluate(eval_dataset=test_ds)
    logger.info("Test metrics: %s", metrics)
    return metrics


def train_sequence_task(cfg: ExperimentConfig) -> Dict[str, float]:
    """Train + evaluate a whole-sequence task (promoter). Returns test metrics."""
    set_seed(cfg.training.seed)
    data_cfg = cfg.data

    model, tokenizer = build_sequence_model(cfg.model.base_model_name, cfg.model.num_labels)
    train_ds, val_ds, test_ds = build_tokenized_datasets(
        tokenizer, hf_dataset=data_cfg.hf_dataset, hf_config=data_cfg.hf_config
    )

    trainer = Trainer(
        model=model,
        args=build_training_args(cfg.training),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=sequence_classification_metrics,
    )
    trainer.train()

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)

    metrics = trainer.evaluate(eval_dataset=test_ds)
    logger.info("Test metrics: %s", metrics)
    return metrics


def train_from_config(cfg: ExperimentConfig) -> Dict[str, float]:
    """Dispatch to the right pipeline based on ``cfg.data.task_type``."""
    if cfg.data.task_type == "token":
        return train_token_task(cfg)
    if cfg.data.task_type == "sequence":
        return train_sequence_task(cfg)
    raise ValueError(f"Unknown task_type: {cfg.data.task_type!r} (expected 'token' or 'sequence')")
