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
kaggle datasets download -d willkoehrsen/sentinel2-drivendata-cloud-cover \
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
