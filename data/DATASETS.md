# Datasets

All three tasks are framed as **image classification** (not segmentation/detection) to
keep training fast and inference cheap enough for an edge device. If a source dataset
only has masks, we bucket by pixel statistics into a classification label (steps below).

## 1. Cloud cover — `TinyCloudNet`

Classes: `clear` (0–10% cloud), `partly_cloudy` (10–60%), `overcast` (60–100%)

Primary option (Sentinel-2, optical, matches real CubeSat optical payloads):
- Kaggle mirror of the DrivenData "On Cloud N" challenge:
  - `willkoehrsen/sentinel2-drivendata-cloud-cover`
  - or `hmendonca/cloud-cover-detection`
- Ships as Sentinel-2 chips + binary cloud masks. Bucket each chip:
  ```python
  cloud_pct = mask.mean()  # mask is 0/1 per pixel
  if cloud_pct < 0.10: label = "clear"
  elif cloud_pct < 0.60: label = "partly_cloudy"
  else: label = "overcast"
  ```
- Simpler binary alternative if you want to skip bucketing entirely:
  `sakibahmed91/cloud2street-dataset` (already has cloud-free vs cloudy masks) — just
  collapse `TinyCloudNet` to 2 classes if you go this route (change `num_classes=2` in
  `models/cloud_classifier.py`).

Target folder layout expected by `train_cloud_classifier.py` (standard
`torchvision.datasets.ImageFolder` layout):
```
data/cloud/
├── train/
│   ├── clear/
│   ├── partly_cloudy/
│   └── overcast/
└── val/
    ├── clear/
    ├── partly_cloudy/
    └── overcast/
```

## 2. Event detection — `TinyEventNet`

Classes: `none`, `wildfire`, `oil_spill`, `other_anomaly`

- **Wildfire** (optical): `abdelghaniaaba/wildfire-prediction-dataset` (binary
  wildfire/nowildfire, 350×350 Canada satellite chips) — map `wildfire` → `wildfire`,
  `nowildfire` → `none`. Alternative: `elmadafri/the-wildfire-dataset`.
- **Oil spill** (SAR, Sentinel-1): `nabilsherif/oil-spill` or
  `harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset` — map spill-positive → `oil_spill`.
  Note: SAR imagery looks very different from optical (grayscale, speckle noise). If your
  CubeSat only carries an optical payload, either drop the oil-spill class or apply a
  SAR-specific preprocessing branch — flag this if you want that added.
- **`other_anomaly` / `none`**: sample "boring" clear-ground chips from the cloud
  dataset's `clear` bucket to serve as negative examples so the model doesn't just
  learn "textured photo = event."

Target folder layout:
```
data/events/
├── train/
│   ├── none/
│   ├── wildfire/
│   ├── oil_spill/
│   └── other_anomaly/
└── val/
    ├── none/
    ├── wildfire/
    ├── oil_spill/
    └── other_anomaly/
```

## 3. Getting the data

```bash
# one-time: put your Kaggle API token at ~/.kaggle/kaggle.json
bash data/download_datasets.sh
```

This pulls all datasets above into `data/raw/` — you'll still need to run the
bucketing/relabeling step (a `prepare_*.py` helper stub is called out in the script)
before pointing the training scripts at `data/cloud/` and `data/events/`.

## 4. Licensing note
Sentinel-1/2 data is distributed under the Copernicus Sentinel Data Terms and
Conditions (free for most uses, attribution required). Check each Kaggle dataset's
individual license/usage terms before using in anything beyond a hackathon demo.
