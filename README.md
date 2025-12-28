# CleanShot

CleanShot is my no-nonsense dataset cleaner. You point it at a CSV/Parquet/JSON file, it cleans the mess, and spits out a clean Parquet plus a markdown report of what happened.

```
        ┌───────────────────── engine: CleanShot ─────────────────────┐
        │                                                             │
        │   intake  ───►  filters  ───►  turbo-jet  ───►  exhaust      │
        │    CSV         null drop        impute          parquet      │
        │    JSON        date cast        dedupe          report.md    │
        │    Parquet     currency clean   polish          silence 🔧   │
        │                                                             │
        └─────────────────────────────────────────────────────────────┘

## What it does

- Drops garbage columns (mostly null or constant).
- Casts date-like strings to DATE.
- Cleans currency strings into numeric values.
- Imputes missing values (median for numeric, mode for categorical) with `ml_ready`.
- Deduplicates rows.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt

python cleanshot.py data.csv cleaned.parquet
```

## Example file

I included a trimmed example dataset at `samples/example_small.csv` (healthcare outpatient claims, ~5k rows).

```bash
python cleanshot.py samples/example_small.csv example_clean.parquet
```

## Interactive mode (recommended)

Guided flow with 3 choices and simple prompts.

```bash
python cleanshot.py --interactive
```

## Useful flags

```bash
# profile the dataset only
python cleanshot.py data.csv --profile

# markdown report only (no progress logs)
python cleanshot.py data.csv out.parquet --no-verbose

# skip imputation
python cleanshot.py data.csv out.parquet --preset basic

# force currency locale
python cleanshot.py data.csv out.parquet --currency-locale us
```

## Commands (copy/paste)

```bash
# basic
python cleanshot.py samples/example_small.csv example_clean.parquet

# interactive
python cleanshot.py --interactive

# profile only
python cleanshot.py samples/example_small.csv --profile

# dry run
python cleanshot.py samples/example_small.csv example_clean.parquet --dry-run

# skip imputation
python cleanshot.py samples/example_small.csv example_clean.parquet --preset basic

# ml_ready
python cleanshot.py samples/example_small.csv example_clean.parquet --preset ml_ready

# markdown report only
python cleanshot.py samples/example_small.csv example_clean.parquet --no-verbose

# force currency locale
python cleanshot.py samples/example_small.csv example_clean.parquet --currency-locale us
python cleanshot.py samples/example_small.csv example_clean.parquet --currency-locale eu

# threads + compression
python cleanshot.py samples/example_small.csv example_clean.parquet --threads 4 --compression zstd
```

## Notes

- Input can be a file, directory, or glob (example: `data/*.csv`).
- If you mess up the input, CleanShot tells you what to fix.
- A markdown report is printed at the end of every run.

## Author

Aanu Oshakuade  
LinkedIn: www.linkedin.com/in/aanu-oshakuade-26a2002ab

## License

MIT
