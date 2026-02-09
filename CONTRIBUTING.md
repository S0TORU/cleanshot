# Contributing

## Quick setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -e .
```

## Run locally

```bash
cleanshot samples/example_small.csv example_clean.parquet
```

## Style

- Use Python 3.10+.
- Keep changes focused and minimal.
- Avoid adding new dependencies unless necessary.

## Pull requests

- Keep PRs small and clear.
- Describe the motivation and what changed.
- Include test steps if you ran anything.
