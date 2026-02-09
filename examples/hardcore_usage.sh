#!/usr/bin/env bash
set -euo pipefail

cleanshot samples/example_small.csv example_clean.parquet --preset ml_ready --threads 4
cleanshot samples/example_small.csv --profile
cleanshot data/big/*.csv cleaned.parquet --sample-rows 100000
cleanshot data/ cleaned.parquet
