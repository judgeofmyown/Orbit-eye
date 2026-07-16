#!/usr/bin/env bash
# Pulls the raw datasets referenced in DATASETS.md via the Kaggle CLI.
#
# Prereqs:
#   pip install kaggle
#   place your API token at ~/.kaggle/kaggle.json  (Kaggle account -> Create New Token)
#   chmod 600 ~/.kaggle/kaggle.json
#
# NOTE: dataset slugs on Kaggle occasionally get renamed/removed by their owners.
# If a `kaggle datasets download` call 404s, search kaggle.com for the current slug
# and swap it in below.

set -euo pipefail

RAW_DIR="$(dirname "$0")/raw"
mkdir -p "$RAW_DIR"/{cloud,wildfire,oilspill}

echo "== Cloud cover dataset =="
# Using fxmikf/cloud-coverage-classification (~540MB, plain RGB, already
# classification-labeled) instead of the 17GB+ Sentinel-2 "On Cloud N" mirror, and
# instead of sakibahmed91/cloud2street-dataset which turned out to be Sentinel-1 SAR
# flood-mapping data (from the "Cloud to Street" org), not RGB cloud photos — a bad
# fit for TinyCloudNet's 128x128 RGB input. See DATASETS.md section 1 for the
# original tradeoff discussion; swap back to the Sentinel-2 mirror if you have the
# disk space and want the full 3-class (clear/partly_cloudy/overcast) version.
kaggle datasets download -d fxmikf/cloud-coverage-classification \
  -p "$RAW_DIR/cloud" --unzip

echo "== Wildfire dataset =="
kaggle datasets download -d abdelghaniaaba/wildfire-prediction-dataset \
  -p "$RAW_DIR/wildfire" --unzip

echo "== Oil spill (SAR) dataset =="
kaggle datasets download -d nabilsherif/oil-spill \
  -p "$RAW_DIR/oilspill" --unzip

echo "Done. Raw data is in $RAW_DIR."
echo "Next: write/run a prepare_cloud.py / prepare_events.py bucketing script to"
echo "reorganize these into data/cloud/{train,val}/<class>/ and data/events/{train,val}/<class>/"
echo "per the folder layout documented in DATASETS.md, then run the training scripts."
