"""Console entry points for training and interpretation.

Exposed as ``genome-lm-train`` and ``genome-lm-interpret`` (see pyproject), and
also invoked by the thin wrappers in ``scripts/``.
"""

from __future__ import annotations

import argparse
import json
import logging

from .config import ExperimentConfig

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def train_main(argv: list[str] | None = None) -> None:
    """Fine-tune HyenaDNA on the task described by a YAML config."""
    parser = argparse.ArgumentParser(description=train_main.__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override training.output_dir.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format=_LOG_FORMAT)

    from .training import train_from_config

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.output_dir:
        cfg.training.output_dir = args.output_dir

    logging.getLogger(__name__).info("Running experiment %r", cfg.name)
    metrics = train_from_config(cfg)
    print(json.dumps({k: float(v) for k, v in metrics.items()}, indent=2))


def interpret_main(argv: list[str] | None = None) -> None:
    """Run Explainable-AI attribution for a fine-tuned checkpoint."""
    parser = argparse.ArgumentParser(description=interpret_main.__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--model-dir", required=True, help="Fine-tuned checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Where to write attribution CSVs.")
    parser.add_argument("--k", type=int, default=5, help="k-mer length (default: 5).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format=_LOG_FORMAT)

    from .interpretation import interpret_from_config

    cfg = ExperimentConfig.from_yaml(args.config)
    interpret_from_config(cfg, args.model_dir, args.output_dir, k=args.k)
