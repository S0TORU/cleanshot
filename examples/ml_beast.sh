#!/usr/bin/env bash
set -euo pipefail

cleanshot samples/example_small.csv example_clean.parquet \
  --preset ml_ready \
  --outlier-mode percentile --outlier-pct 0.01,0.99 \
  --scale standardize \
  --encode onehot --encode-max-categories 25 \
  --split 0.8,0.1,0.1 \
  --schema-out schema.json
