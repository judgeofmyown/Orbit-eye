# models/weights/

This folder is intentionally empty in the scaffold.

Expected files, once you've trained:
- `cloud_classifier.pth` — produced by `models/train_cloud_classifier.py`
- `event_detector.pth`   — produced by `models/train_event_detector.py`

Each checkpoint is saved as a dict: `{"model_state_dict": ..., "classes": [...], "val_acc": ...}`.

`edge/inference_engine.py` checks for these files at startup:
- If **both** are present -> loads real trained weights, runs in `"mode": "trained"`.
- If **either** is missing -> falls back to a heuristic/random-weight substitute for
  the missing one, and tags every telemetry record with `"mode": "heuristic_fallback"`
  so it's never silently mistaken for real model output.
