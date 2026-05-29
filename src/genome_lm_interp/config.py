"""Typed configuration objects loaded from YAML.

Each experiment (Z-DNA, G4, promoter) is described by a single YAML file under
``configs/``. The dataclasses below give the configuration a documented schema
with sensible defaults, so the CLI scripts stay thin and every hyper-parameter
lives in version-controlled config rather than scattered across notebook cells.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml


@dataclass
class DataConfig:
    """Where the genomic data lives and how windows are built/labelled."""

    # Task family: "token" (per-nucleotide, e.g. Z-DNA, G4) or
    # "sequence" (whole-sequence, e.g. promoter via the HF GUE dataset).
    task_type: str = "token"

    # Root directory that holds the genomic assets (relative paths below are
    # resolved against it).
    data_root: str = "data"

    # --- token-classification (Z-DNA / G4) ---
    # Pickled per-nucleotide label track, keyed by chromosome.
    labels_file: Optional[str] = None
    # Pickled ENCODE-style blacklist mask, keyed by chromosome.
    blacklist_file: Optional[str] = None
    # Directory of per-chromosome DNA-sequence fragments (joblib dumps).
    dna_dir: Optional[str] = None
    chroms: List[str] = field(
        default_factory=lambda: [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]
    )
    window_width: int = 100
    # Negatives-to-positives ratio used to undersample the majority class.
    neg_pos_ratio: int = 3

    # --- sequence-classification (promoter) ---
    hf_dataset: Optional[str] = None  # e.g. "leannmlindsey/GUE"
    hf_config: Optional[str] = None  # e.g. "prom_core_all"

    # Split fractions (token-classification builds splits locally).
    test_size: float = 0.2
    val_size: float = 0.25  # fraction of the train+val remainder


@dataclass
class ModelConfig:
    """Backbone selection and head configuration."""

    base_model_name: str = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
    num_labels: int = 2
    # Per-class CrossEntropy weights for the imbalanced token task.
    class_weights: Optional[List[float]] = field(default_factory=lambda: [0.7, 2.0])
    max_length: int = 100


@dataclass
class TrainingConfig:
    """Hugging Face ``TrainingArguments`` mirrored as a typed config."""

    output_dir: str = "results/run"
    num_train_epochs: int = 12
    learning_rate: float = 5e-4
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.10
    fp16: bool = True
    save_total_limit: int = 3
    metric_for_best_model: str = "f1"
    seed: int = 42

    # Optional two-stage fine-tuning (frozen backbone -> full fine-tune with
    # discriminative learning rates). Used for the best Z-DNA / G4 runs.
    two_stage: bool = False
    stage1_epochs: int = 3
    stage1_lr: float = 1e-3
    stage2_epochs: int = 6
    backbone_lr: float = 1e-5
    head_lr: float = 5e-4


@dataclass
class ExperimentConfig:
    """Top-level config tying data, model and training together."""

    name: str = "experiment"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load an :class:`ExperimentConfig` from a YAML file."""
        with open(path, "r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls(
            name=raw.get("name", "experiment"),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view (useful for logging / serialisation)."""
        return dataclasses.asdict(self)
