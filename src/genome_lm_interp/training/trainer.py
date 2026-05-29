"""Training loops: single-stage and two-stage discriminative fine-tuning.

The best Z-DNA / G4 results came from a two-stage schedule: first train only the
classification head with a frozen backbone, then unfreeze and fine-tune the
whole model with discriminative learning rates (small for the backbone, larger
for the head). Both schedules are exposed here.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments

from ..config import TrainingConfig


def build_training_args(cfg: TrainingConfig, output_dir: Optional[str] = None, **overrides) -> TrainingArguments:
    """Create :class:`TrainingArguments` from a :class:`TrainingConfig`."""
    params = dict(
        output_dir=output_dir or cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        optim="adamw_torch",
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        save_safetensors=False,
        seed=cfg.seed,
        report_to="none",
    )
    params.update(overrides)
    return TrainingArguments(**params)


def run_finetune(
    model,
    train_dataset,
    eval_dataset,
    tokenizer,
    compute_metrics: Callable,
    cfg: TrainingConfig,
) -> Trainer:
    """Standard single-stage fine-tuning."""
    trainer = Trainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.can_return_loss = True
    trainer.train()
    return trainer


def two_stage_finetune(
    model,
    train_dataset,
    eval_dataset,
    tokenizer,
    compute_metrics: Callable,
    cfg: TrainingConfig,
) -> Trainer:
    """Two-stage fine-tuning (frozen head warm-up, then discriminative full FT).

    ``model`` must expose a ``hyena`` backbone and a ``classifier`` head
    (i.e. :class:`HyenaDNAForTokenClassification`).
    """
    # --- Stage 1: freeze backbone, train only the head ---
    for param in model.hyena.parameters():
        param.requires_grad = False

    stage1_args = build_training_args(
        cfg,
        output_dir=f"{cfg.output_dir}/stage1",
        num_train_epochs=cfg.stage1_epochs,
        learning_rate=cfg.stage1_lr,
    )
    Trainer(
        model=model,
        args=stage1_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    ).train()

    # --- Stage 2: unfreeze, fine-tune with discriminative learning rates ---
    for param in model.hyena.parameters():
        param.requires_grad = True

    optimizer = AdamW(
        [
            {"params": model.hyena.parameters(), "lr": cfg.backbone_lr},
            {"params": model.classifier.parameters(), "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    stage2_args = build_training_args(
        cfg,
        output_dir=f"{cfg.output_dir}/stage2",
        num_train_epochs=cfg.stage2_epochs,
        learning_rate=cfg.head_lr,
    )
    trainer = Trainer(
        model=model,
        args=stage2_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
    )
    trainer.can_return_loss = True
    trainer.train()
    return trainer
