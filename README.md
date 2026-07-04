# OrbitEye-Filter

**Intelligent Bandwidth Optimization for SmallSats** — an onboard Edge-AI filter that
looks at every frame a CubeSat captures and decides, *before it ever touches the radio*,
whether it's garbage (clouds), routine (clear ground), or urgent (disaster event).

```
Camera → [ Edge AI container = simulated Jetson ] → Discard / Standard / Priority
                                                            │
                                                    shared "downlink" volume
                                                            │
                                              [ Ground Station Streamlit app ]
```

---

## 1. Project layout

```
orbiteye-filter/
├── docker/
│   ├── Dockerfile.edge          # simulated onboard Jetson container
│   ├── Dockerfile.ground        # ground station (Streamlit) container
│   └── docker-compose.yml       # wires both together with a shared volume
├── data/
│   ├── DATASETS.md              # dataset spec + links (see below)
│   └── download_datasets.sh     # kaggle CLI puller for all 3 datasets
├── models/
│   ├── cloud_classifier.py      # TinyCloudNet CNN definition (untrained)
│   ├── event_detector.py        # TinyEventNet CNN definition (untrained)
│   ├── train_cloud_classifier.py# training script skeleton — YOU run this
│   ├── train_event_detector.py  # training script skeleton — YOU run this
│   └── weights/                 # <-- drop your .pth files here when trained
├── edge/                        # everything that runs "onboard"
│   ├── camera_storage/          # simulated satellite photo storage (input)
│   ├── inference_engine.py      # loads models, classifies a frame
│   ├── queue_manager.py         # routes frames into Discard/Standard/Priority
│   ├── jetson_stats_sim.py      # simulates Jetson power/thermal/latency envelope
│   ├── config.yaml              # thresholds, compression rates, paths
│   └── run_edge_pipeline.py     # orchestrator entrypoint (the "satellite loop")
├── ground_station/
│   ├── app.py                   # Streamlit "Ground Station" dashboard
│   └── bandwidth_calculator.py  # raw vs. downlinked bytes, % saved
├── shared/
│   └── schemas.py                # shared dataclasses/JSON schema for telemetry
└── requirements.txt
```

## 2. The three-class edge decision

| Queue | Trigger | What's transmitted |
|---|---|---|
| **Discard** | cloud_score ≥ threshold (frame mostly cloud) | Nothing — 0 bytes, frame deleted from onboard storage |
| **Standard Downlink** | clear frame, no event detected | Compressed/downsampled thumbnail, low priority queue |
| **Priority Alert** | wildfire / oil-spill / other event detected | Full-resolution image, flagged for immediate downlink |

This is a **cascade**, not a single model: cloud check first (cheapest, runs on every
frame), event detector only runs on frames that pass the cloud filter (saves onboard
compute — real flight computers are power constrained).

## 3. Models — specs (train these yourself)

Both are deliberately tiny (target: runs on a Jetson Nano / Xavier NX class device,
not a datacenter GPU). Definitions are in `models/*.py`, placeholders — **no weights
are included**. Training scripts in `models/train_*.py` are skeletons for you to point
at real data and run.

### 3a. `TinyCloudNet` (cloud_classifier.py)
- Task: 3-class classification — `clear`, `partly_cloudy`, `overcast`
- Input: 128×128 RGB
- Architecture: 4 conv blocks (depthwise-separable convs to keep it edge-friendly) + GAP + FC
- Suggested dataset: **Sentinel-2 Cloud Cover Segmentation dataset (DrivenData "On Cloud N")**,
  mirrored on Kaggle as `hmendonca/cloud-cover-detection` / `willkoehrsen/sentinel2-drivendata-cloud-cover`.
  It's natively a segmentation mask, so for classification-style training bucket each
  chip by % cloud-pixel coverage into the 3 classes (see `data/DATASETS.md`).
  Alternative simpler dataset: `sakibahmed91/cloud2street-dataset` (binary cloud/clear masks).

### 3b. `TinyEventNet` (event_detector.py)
- Task: 4-class classification — `none`, `wildfire`, `oil_spill`, `other_anomaly`
- Input: 128×128 RGB
- Architecture: same depthwise-separable CNN backbone as TinyCloudNet (shared design
  so both can eventually be fused/quantized together for one Jetson engine)
- Suggested datasets (combine and relabel into one folder structure):
  - Wildfire: `abdelghaniaaba/wildfire-prediction-dataset` (binary wildfire/no-wildfire,
    350×350 satellite chips) or `elmadafri/the-wildfire-dataset`
  - Oil spill: `nabilsherif/oil-spill` or `harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset`
    (Sentinel-1 SAR imagery, so treat as a separate SAR-only training run if you don't
    want to mix optical + radar in one classifier)

Full details, class-bucketing logic, and folder conventions are in `data/DATASETS.md`.

## 4. Quickstart

### Without Docker (fastest for dev)
```bash
pip install -r requirements.txt

# 1. Run the edge pipeline once over the sample camera storage folder
python edge/run_edge_pipeline.py

# 2. Launch the ground station dashboard
streamlit run ground_station/app.py
```
Until you drop trained weights into `models/weights/`, the inference engine
automatically falls back to a **heuristic mode** (brightness/variance-based cloud proxy
+ a randomly-initialized event net) so the *entire pipeline is demoable end-to-end
today*, with a clearly logged `"mode": "heuristic_fallback"` flag in the telemetry so
nobody mistakes placeholder output for a real model result.

### With Docker (the actual point of this project)
```bash
docker compose -f docker/docker-compose.yml up --build
```
This starts two containers:
- `edge-sim` — resource-capped (CPU/RAM limited in `docker-compose.yml` to
  approximate a Jetson Nano's 4-core ARM CPU + 4GB RAM envelope) container that
  runs `run_edge_pipeline.py` on a loop, writing results to a shared Docker volume.
- `ground-station` — Streamlit container reading that same shared volume, exposed on
  `localhost:8501`.
