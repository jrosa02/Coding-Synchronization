#!/usr/bin/env bash
# Known-good decode config for measurments/RefCurve_2026-08-18_1_140234.Wfm.csv
# (23/24 frames decoded, sync locked 8/8 on every kept frame).
#
# --eof-num 2, not 4: this file's inter-frame gaps never reach the eof-num=4 splitter
# threshold (4160 samples), so the splitter never finds a frame boundary at all.
# No --max-samples: 800k samples is under one 45-pulse frame's worth of data.
#
# Usage: ./decode_config.sh [path/to/waveform.csv]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -f .envrc ] && source .envrc

uv run decode_measurement.py "${1:-./measurments/RefCurve_2026-08-18_2_180749.Wfm.csv}" \
  --sample-rate 1e9 \
  --combine sub \
  --ppm-rank 10 \
  --dead-slots 16 \
  --sync-num 8 \
  --sync-value 0 \
  --metadata-num 5 \
  --data-num 16 \
  --ecc-num 16 \
  --eof-num 2 \
  --threshold 0.3 \
  --edge edge \
  --plot
