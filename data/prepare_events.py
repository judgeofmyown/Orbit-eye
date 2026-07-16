"""
Buckets the raw wildfire + oil-spill datasets into the ImageFolder layout expected by
train_event_detector.py: data/events/{train,val}/<class>/, classes = EVENT_CLASSES.

Wildfire (abdelghaniaaba/wildfire-prediction-dataset): already shipped as
wildfire/nowildfire ImageFolder splits -> mapped directly (wildfire -> "wildfire",
nowildfire -> "none"). train+test combined into events/train for more volume,
valid -> events/val.

Oil spill (nabilsherif/oil-spill): ships as image + color-coded segmentation mask
pairs using the Krestenitis et al. 5-class palette:
  black (0,0,0)     = sea/background
  cyan  (0,255,255) = oil spill
  red   (255,0,0)   = look-alike (something that resembles oil but isn't)
  brown (153,76,0)  = ship
  green (0,153,0)   = land
Bucketed into a single classification label per image by taking whichever
foreground class (i.e. excluding sea) has the most pixels in the mask:
  - oil spill dominant   -> "oil_spill"
  - look-alike dominant  -> "other_anomaly" (closest available bucket for an
    anomalous water feature that isn't actually an oil spill)
  - ship/land dominant, or no foreground pixels at all -> "none"
train -> events/train, test -> events/val.

Usage:
    python data/prepare_events.py
"""
import os
import shutil

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "events")

WILDFIRE_DIR = os.path.join(RAW, "wildfire")
OILSPILL_DIR = os.path.join(RAW, "oilspill", "oil-spill")

EVENT_CLASSES = ["none", "wildfire", "oil_spill", "other_anomaly"]

OIL_SPILL = (0, 255, 255)
LOOK_ALIKE = (255, 0, 0)
SHIP = (153, 76, 0)
LAND = (0, 153, 0)


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    for split in ("train", "val"):
        for label in EVENT_CLASSES:
            os.makedirs(os.path.join(path, split, label), exist_ok=True)


def copy_frame(src, split, label, dest_name):
    dest_dir = os.path.join(OUT, split, label)
    shutil.copy(src, os.path.join(dest_dir, dest_name))


def prepare_wildfire():
    mapping = {"wildfire": "wildfire", "nowildfire": "none"}
    sources = [("train", "train"), ("test", "train"), ("valid", "val")]
    count = 0
    for src_split, dst_split in sources:
        for src_label, dst_label in mapping.items():
            src_dir = os.path.join(WILDFIRE_DIR, src_split, src_label)
            if not os.path.isdir(src_dir):
                continue
            for fname in os.listdir(src_dir):
                copy_frame(os.path.join(src_dir, fname), dst_split, dst_label,
                           dest_name=f"wf_{src_split}_{fname}")
                count += 1
    print(f"wildfire: copied {count} frames")


def bucket_oilspill_mask(mask_path):
    arr = np.array(Image.open(mask_path).convert("RGB"))
    flat = arr.reshape(-1, 3)
    counts = {
        OIL_SPILL: int(np.all(flat == OIL_SPILL, axis=1).sum()),
        LOOK_ALIKE: int(np.all(flat == LOOK_ALIKE, axis=1).sum()),
        SHIP: int(np.all(flat == SHIP, axis=1).sum()),
        LAND: int(np.all(flat == LAND, axis=1).sum()),
    }
    dominant_color, dominant_count = max(counts.items(), key=lambda kv: kv[1])
    if dominant_count == 0:
        return "none"
    if dominant_color == OIL_SPILL:
        return "oil_spill"
    if dominant_color == LOOK_ALIKE:
        return "other_anomaly"
    return "none"  # ship/land dominant, no oil/look-alike present


def prepare_oilspill():
    sources = [("train", "train"), ("test", "val")]
    count = 0
    for src_split, dst_split in sources:
        img_dir = os.path.join(OILSPILL_DIR, src_split, "images")
        lbl_dir = os.path.join(OILSPILL_DIR, src_split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            stem, _ = os.path.splitext(fname)
            mask_path = os.path.join(lbl_dir, stem + ".png")
            if not os.path.exists(mask_path):
                continue
            label = bucket_oilspill_mask(mask_path)
            copy_frame(os.path.join(img_dir, fname), dst_split, label,
                       dest_name=f"os_{src_split}_{fname}")
            count += 1
    print(f"oil spill: copied {count} frames")


def main():
    reset_dir(OUT)
    prepare_wildfire()
    prepare_oilspill()
    print()
    for split in ("train", "val"):
        print(f"-- events/{split} --")
        for label in EVENT_CLASSES:
            d = os.path.join(OUT, split, label)
            print(f"  {label}: {len(os.listdir(d))}")


if __name__ == "__main__":
    main()
