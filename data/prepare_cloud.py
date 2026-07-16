"""
Buckets the raw fxmikf/cloud-coverage-classification dataset into the ImageFolder
layout expected by train_cloud_classifier.py: data/cloud/{train,val}/<class>/,
classes = CLOUD_CLASSES (clear, partly_cloudy, overcast).

The raw dataset is a flat images/ folder plus a CSV of human annotations with a
5-level ordinal "choice" column (Very Low..Very High cloud coverage), collapsed here
into the 3-class scheme:
  Very Low, Low -> "clear"
  Medium        -> "partly_cloudy"
  High, Very High -> "overcast"

A random 85/15 split (seeded) produces train/val.

Usage:
    python data/prepare_cloud.py
"""
import os
import random
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw", "cloud", "cloud_classification")
CSV_PATH = os.path.join(RAW_DIR, "cloud_classification_export.csv")
IMAGES_DIR = os.path.join(RAW_DIR, "images")
OUT = os.path.join(ROOT, "data", "cloud")

CLOUD_CLASSES = ["clear", "partly_cloudy", "overcast"]

CHOICE_MAP = {
    "Very Low": "clear",
    "Low": "clear",
    "Medium": "partly_cloudy",
    "High": "overcast",
    "Very High": "overcast",
}

VAL_FRACTION = 0.15
SEED = 42


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    for split in ("train", "val"):
        for label in CLOUD_CLASSES:
            os.makedirs(os.path.join(path, split, label), exist_ok=True)


def main():
    reset_dir(OUT)
    df = pd.read_csv(CSV_PATH)
    df["label"] = df["choice"].map(CHOICE_MAP)
    df = df.dropna(subset=["label"])

    rng = random.Random(SEED)
    rows = list(df.itertuples(index=False))
    rng.shuffle(rows)
    n_val = int(len(rows) * VAL_FRACTION)
    val_rows = set(id(r) for r in rows[:n_val])

    counts = {}
    missing = 0
    for row in rows:
        split = "val" if id(row) in val_rows else "train"
        src = os.path.join(RAW_DIR, row.image)
        if not os.path.exists(src):
            missing += 1
            continue
        dest_dir = os.path.join(OUT, split, row.label)
        shutil.copy(src, os.path.join(dest_dir, os.path.basename(row.image)))
        counts[(split, row.label)] = counts.get((split, row.label), 0) + 1

    if missing:
        print(f"WARNING: {missing} rows referenced images not found on disk")

    print()
    for split in ("train", "val"):
        print(f"-- cloud/{split} --")
        for label in CLOUD_CLASSES:
            print(f"  {label}: {counts.get((split, label), 0)}")


if __name__ == "__main__":
    main()
