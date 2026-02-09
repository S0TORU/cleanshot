#!/usr/bin/env bash
set -euo pipefail

cleanshot \
  --fusion-source imu=sensors/imu.csv \
  --fusion-source gps=sensors/gps.csv \
  --fusion-id-col device_id \
  --fusion-time-col timestamp \
  --fusion-time-tolerance 500ms \
  --output fused_clean.parquet
