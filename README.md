# OrbitEye-Filter

**Intelligent Bandwidth Optimization for SmallSats** — an onboard Edge-AI filter that
looks at every frame a CubeSat captures and decides, *before it ever touches the radio*,
whether it's garbage (clouds), routine (clear ground), urgent (disaster event), or
worth a human's attention (models aren't sure / doesn't look like anything trained on).

```
Camera → [ Edge AI container = simulated Jetson ] → Discard / Standard / Review / Priority
                                                              │
                                                    orbit-pass-aware downlink
                                                     (shared volume + backlog)
                                                              │
                                                [ Ground Station Streamlit app ]
```

Trained end-to-end on real Kaggle imagery (not synthetic placeholders) — see
[section 5](#5-models--real-numbers) for exact datasets, architectures, and results.

---

## 1. Project layout

```
orbiteye-filter/
├── docker/
│   ├── Dockerfile.edge          # simulated onboard Jetson container
│   ├── Dockerfile.ground        # ground station (Streamlit) container
│   └── docker-compose.yml       # wires both together with a shared volume
├── data/
│   ├── DATASETS.md              # original dataset spec (see section 4 for what's actually used)
│   ├── download_datasets.sh     # kaggle CLI puller for all 3 datasets
│   ├── prepare_cloud.py         # buckets the cloud dataset into data/cloud/{train,val}/<class>/
│   └── prepare_events.py        # buckets wildfire+oil-spill into data/events/{train,val}/<class>/
├── models/
│   ├── cloud_classifier.py         # TinyCloudNet CNN definition
│   ├── event_detector.py           # TinyEventNet CNN definition
│   ├── fusion_net.py               # OrbitFusionNet — shared-trunk multi-task model
│   ├── anomaly_autoencoder.py      # TinyAutoencoder — out-of-distribution scorer
│   ├── train_cloud_classifier.py   # trains TinyCloudNet
│   ├── train_event_detector.py     # trains TinyEventNet (class-weighted loss)
│   ├── train_fusion.py             # trains OrbitFusionNet (partial-label multi-task)
│   ├── train_anomaly_autoencoder.py# trains TinyAutoencoder on "known-category" frames
│   ├── export_onnx.py              # ONNX export + INT8 quantization + latency benchmark
│   ├── finetune_from_corrections.py# applies ground-station corrections (incremental learning)
│   └── weights/                    # trained .pth checkpoints (committed — see section 5)
├── edge/                        # everything that runs "onboard"
│   ├── camera_storage/          # simulated satellite photo storage (input)
│   ├── inference_engine.py      # loads models, classifies a frame, auto-picks best backend
│   ├── queue_manager.py         # routes frames into Discard/Standard/Review/Priority
│   ├── jetson_stats_sim.py      # simulates Jetson power/thermal/latency envelope
│   ├── orbit_pass_sim.py        # simulates ground-station contact windows + downlink backlog
│   ├── gradcam.py                # Grad-CAM heatmap overlays for priority/review frames
│   ├── config.yaml              # thresholds, compression, orbit-pass, paths
│   └── run_edge_pipeline.py     # orchestrator entrypoint (the "satellite loop")
├── ground_station/
│   ├── app.py                   # Streamlit "Ground Station" dashboard
│   └── bandwidth_calculator.py  # raw vs. downlinked bytes, % saved
├── shared/
│   └── schemas.py                # shared dataclasses/JSON schema for telemetry
└── requirements.txt
```

## 2. The four-queue edge decision

| Queue | Trigger | What's transmitted |
|---|---|---|
| **Discard** | cloud_class = overcast, confidence ≥ 0.60 | Nothing — 0 bytes, frame deleted from onboard storage |
| **Standard Downlink** | clear/usable frame, no event detected | Real re-encoded JPEG thumbnail (320px, quality 35) |
| **Review** | both classifiers unsure (conf < 0.45), or autoencoder OOD score ≥ threshold | Real re-encoded JPEG (480px, quality 55) + Grad-CAM overlay |
| **Priority Alert** | wildfire / oil-spill / other event detected, confidence ≥ 0.55 | Full-resolution original + Grad-CAM overlay, immediate downlink |

This is a **cascade**, not a single model: cloud check first (cheapest, runs on every
frame), event detector only runs on frames that pass the cloud filter (saves onboard
compute — real flight computers are power constrained). Priority always wins over
discard — a wildfire glimpsed through partial cloud cover still gets downlinked.

Downlink itself is **orbit-pass-aware**: a real LEO satellite doesn't have continuous
downlink, only short ground-station contact windows once per orbit. Frames queue in a
backlog (`edge/orbit_pass_sim.py`) that only drains during a simulated contact window
at a bounded link rate — visible in the dashboard as "in contact" / "next contact in
Xs" plus a pending-backlog byte count.

## 3. Setup on a fresh machine

```bash
git clone https://github.com/judgeofmyown/Orbit-eye.git
cd Orbit-eye
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The repo ships with **trained weights already in `models/weights/`**
(`cloud_classifier.pth`, `event_detector.pth`, `fusion_net.pth`,
`anomaly_autoencoder.pth` — a few hundred KB each), so you can run the demo
immediately without training anything:

```bash
python edge/run_edge_pipeline.py          # process once and exit
# or
python edge/run_edge_pipeline.py --loop   # keep watching camera_storage for new frames

streamlit run ground_station/app.py       # dashboard at localhost:8501
```

`edge/camera_storage/` isn't populated by default — either generate synthetic demo
frames (`python edge/generate_sample_frames.py`), or drop real images in, or run the
full data pipeline below and copy a few files out of `data/cloud/val/` and
`data/events/val/`.

### Retraining from scratch on real data

If you want to reproduce the training rather than use the shipped weights:

```bash
# 1. Kaggle API auth (one-time)
#    kaggle auth login    — OR —   save a token to ~/.kaggle/access_token
pip install kaggle

# 2. Pull the raw datasets (~2.5GB total — see section 4 for what's in each)
bash data/download_datasets.sh

# 3. Bucket raw data into the ImageFolder layout the training scripts expect
python data/prepare_cloud.py
python data/prepare_events.py

# 4. Train (run from inside models/, or pass --data_dir explicitly)
cd models
python train_cloud_classifier.py --data_dir ../data/cloud --epochs 15
python train_event_detector.py --data_dir ../data/events --epochs 20
python train_fusion.py --cloud_dir ../data/cloud --events_dir ../data/events --epochs 15
python train_anomaly_autoencoder.py --data_root ../data --epochs 12

# 5. Optional: export to ONNX + INT8, and benchmark latency
python export_onnx.py --n_runs 200
```

A CUDA GPU is used automatically if available (`torch.cuda.is_available()`); all
training scripts fall back to CPU otherwise — the models are small enough (50-65K
params) that CPU training is viable, just slower.

### With Docker
```bash
docker compose -f docker/docker-compose.yml up --build
```
Starts two containers: `edge-sim` (resource-capped to approximate a Jetson Nano's
4-core ARM + 4GB RAM envelope, runs `run_edge_pipeline.py` on a loop) and
`ground-station` (Streamlit, reads the same shared volume, exposed on `localhost:8501`).

## 4. Datasets actually used

`data/DATASETS.md` documents the datasets originally proposed for this project; two of
the three didn't pan out on a normal dev machine (the primary cloud-cover mirror is
17GB of raw Sentinel-2 GeoTIFFs, and the "simpler" alternative turned out to be
mislabeled Sentinel-1 SAR flood data, not RGB cloud photos). What's actually wired up
in `download_datasets.sh`:

| Purpose | Kaggle dataset | Size | Notes |
|---|---|---|---|
| Cloud cover | `fxmikf/cloud-coverage-classification` | ~540MB | 718 RGB images, human-annotated 5-level coverage (Very Low → Very High), collapsed to 3 classes |
| Wildfire | `abdelghaniaaba/wildfire-prediction-dataset` | ~1.5GB | 42,850 usable 350×350 chips, already split wildfire/nowildfire × train/valid/test |
| Oil spill | `nabilsherif/oil-spill` | ~400MB | 1,112 image+segmentation-mask pairs (Krestenitis et al. 5-class palette), bucketed to a single dominant-class label per image |

## 5. Models — real numbers

All four models share the same depthwise-separable-conv design (MobileNet-style —
fewer FLOPs/params than full convs, no batch-norm tricks that complicate later
INT8/TensorRT quantization) and take **128×128 RGB** input.

| Model | Task | Classes | Params | Val accuracy / loss |
|---|---|---|---|---|
| `TinyCloudNet` | cloud cover | `clear`, `overcast`, `partly_cloudy` | 31,331 | **94.2%** |
| `TinyEventNet` | disaster event | `none`, `oil_spill`, `other_anomaly`, `wildfire` | 31,460 | **96.3%** (class-weighted loss — oil_spill/other_anomaly are 40-60x rarer than wildfire/none) |
| `OrbitFusionNet` | both, shared trunk, 1 forward pass | same as above, two heads | 49,895 (vs. 62,791 combined for the two separate models) | cloud **86.0%**, event **94.8%** |
| `TinyAutoencoder` | reconstruction / OOD score | n/a | 65,363 | val loss **0.00136**, calibrated OOD threshold **0.0024** (95th percentile) |

**Class ordering note**: torchvision's `ImageFolder` assigns label indices by
alphabetical folder name, not declaration order — `CLOUD_CLASSES`/`EVENT_CLASSES` in
the model files are ordered to match that (`clear, overcast, partly_cloudy` and
`none, oil_spill, other_anomaly, wildfire`), not the "natural" reading order used
elsewhere in this README/UI.

**Training hyperparameters**: AdamW, `lr=1e-3` (`1e-4` for fine-tuning), weight decay
`1e-4`, cosine annealing LR schedule, batch size 32. `TinyCloudNet`/`TinyEventNet`/
`OrbitFusionNet` train 15-20 epochs; `TinyAutoencoder` trains 12 epochs with MSE
reconstruction loss. `OrbitFusionNet` uses partial-label supervision — each cloud
batch only updates the cloud head + shared trunk, each event batch only the event
head + trunk, cycling the much smaller cloud dataset (718 images) to match the
events loader's length (~37,550 images) each epoch.

### Fusion vs. separate models — latency (ONNX Runtime, CPU, batch=1)

| Variant | fp32 | int8 |
|---|---|---|
| Two separate models (cloud + event back to back) | 0.163 ms/frame | 2.990 ms/frame |
| `OrbitFusionNet` (one pass) | 0.106 ms/frame | 2.101 ms/frame |
| **Fusion speedup** | **1.55x** | **1.42x** |

INT8 dynamic quantization shrinks the exported ONNX models **2.3-2.5x on disk** (e.g.
`event_detector.onnx` 127.2KB → 54.7KB int8), but is actually *slower* than fp32 at
this model size — these networks are small enough that per-op dequantization overhead
outweighs the compute savings. Worth measuring rather than assuming quantization is
free; the size reduction still matters for flash-constrained hardware even where
latency doesn't improve. Run `python models/export_onnx.py` to reproduce these numbers
on your machine.

## 6. Beyond the baseline

A few things layered on top of the original three-queue design, in
`edge/inference_engine.py`, `edge/queue_manager.py`, and `ground_station/app.py`:

- **Review queue + OOD detection** — frames that don't confidently match a known
  cloud/event category, or that the autoencoder flags as unlike anything in training,
  get held for human review instead of silently discarded or downlinked as routine.
- **Grad-CAM explanations** — every priority/review frame gets a class-activation
  heatmap overlay showing which part of the image drove the decision.
- **Orbit-pass-aware downlink** — bandwidth savings are simulated against a realistic
  ground-station contact schedule, not instant transmission.
- **On-orbit incremental learning (stub)** — flag a wrong label from the dashboard,
  then run `python models/finetune_from_corrections.py` to fine-tune the affected
  model on the correction (mixed with a replay sample of existing data to avoid
  overfitting/forgetting on a handful of new examples) — a stand-in for uplinking a
  small model update instead of a full retrain-and-redeploy cycle.
- **Event map** — frames whose source filename embeds GPS coordinates (the wildfire
  chip dataset does this natively) get plotted on the dashboard.

### Known limitations

- `OrbitFusionNet`'s cloud head trades some accuracy (86.0% vs. 94.2% dedicated) for
  the shared-compute win — a consequence of the 718-image cloud set being tiny next
  to the ~37,550-image event set it's jointly trained against.
- `finetune_from_corrections.py` only hot-patches the two separate-model checkpoints
  in place; if you're running on the fusion backend, corrections still get folded into
  `data/cloud/`/`data/events/` for the next full `train_fusion.py` run, but won't
  update `fusion_net.pth` directly.
- INT8 quantization is included for completeness/benchmarking but isn't a clear win
  at this model size — see the latency table above before wiring it into the default
  inference path.
