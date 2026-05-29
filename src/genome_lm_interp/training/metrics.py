"""Metric functions for the Hugging Face ``Trainer``.

Across all tasks the primary metrics are binary **F1** and **Matthews
correlation coefficient (MCC)**, which are robust under the strong class
imbalance of the genomic-structure problems.
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def token_classification_metrics(eval_pred) -> dict:
    """Per-token metrics, ignoring positions labelled ``-100``."""
    preds = eval_pred.predictions.argmax(-1).flatten()
    labels = eval_pred.label_ids.flatten()
    mask = labels != -100
    preds, labels = preds[mask], labels[mask]

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "matthews": matthews_corrcoef(labels, preds),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }
    # ROC-AUC is undefined when only one class is present in the batch.
    if len(set(labels.tolist())) > 1:
        metrics["roc_auc"] = roc_auc_score(labels, preds)
    return metrics


def sequence_classification_metrics(eval_pred) -> dict:
    """Whole-sequence metrics for the promoter task."""
    preds = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "matthews": matthews_corrcoef(labels, preds),
    }
