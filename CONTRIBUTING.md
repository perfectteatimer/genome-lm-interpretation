# Contributing

Thanks for your interest in improving this project! The notes below keep the
codebase consistent and easy to review.

## Development setup

```bash
git clone https://github.com/perfectteatimer/genome-lm-interpretation.git
cd genome-lm-interpretation
python -m venv .venv && source .venv/bin/activate
make install-dev      # editable install + ruff/black/pytest
```

## Workflow

1. Create a feature branch: `git checkout -b feature/short-description`.
2. Make your change. Keep functions small and documented; match the style of the
   surrounding code.
3. Format and lint before committing:
   ```bash
   make format
   make lint
   ```
4. Run the tests: `make test`.
5. Open a pull request describing **what** changed and **why**.

## Project conventions

- **Source layout:** all importable code lives under `src/genome_lm_interp/`.
- **Configuration over code:** experiment hyper-parameters belong in
  `configs/*.yaml`, not hard-coded in scripts.
- **Notebooks** in `notebooks/` are the original research record. New reusable
  logic should go into the package, not into notebooks.
- **Line length:** 100 characters (`black` / `ruff` are configured in
  `pyproject.toml`).

## Reporting issues

Please include the command you ran, the full traceback, and your environment
(`python --version`, `pip freeze | grep -E "torch|transformers"`).
