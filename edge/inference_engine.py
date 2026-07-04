"""
InferenceEngine — the "brain" that runs onboard. Loads TinyCloudNet + TinyEventNet if
trained weights exist in models/weights/; otherwise falls back to a heuristic so the
pipeline is runnable end-to-end before you've trained anything.

Heuristic fallback (used only when a .pth is missing):
- Cloud: brightness + low color-variance proxy (bright, flat, low-saturation regions
  look cloud-like) — NOT a real cloud detector, just enough signal to make the demo
  behave sensibly on arbitrary sample images.
- Event: a randomly-initialized TinyEventNet (i.e. random-ish logits) — clearly
  labeled "heuristic_fallback" in every result so it's never mistaken for a working
  detector.
"""
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from cloud_classifier import TinyCloudNet, CLOUD_CLASSES   # noqa: E402
from event_detector import TinyEventNet, EVENT_CLASSES     # noqa: E402

PREPROCESS = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@dataclass
class InferenceOutput:
    cloud_class: str
    cloud_confidence: float
    event_class: str
    event_confidence: float
    latency_ms: float
    mode: str  # "trained" or "heuristic_fallback"


class InferenceEngine:
    def __init__(self, cloud_weights_path: str, event_weights_path: str, device: str = "cpu"):
        self.device = device
        self.cloud_model, cloud_trained = self._load_cloud_model(cloud_weights_path)
        self.event_model, event_trained = self._load_event_model(event_weights_path)
        self.mode = "trained" if (cloud_trained and event_trained) else "heuristic_fallback"
        if self.mode == "heuristic_fallback":
            print(
                "[InferenceEngine] WARNING: one or both trained weight files were not "
                "found. Running in heuristic_fallback mode — results are placeholders, "
                "not real cloud/event detection. Train models and drop .pth files into "
                "models/weights/ to switch to real inference."
            )

    def _load_cloud_model(self, path: str):
        model = TinyCloudNet(num_classes=len(CLOUD_CLASSES)).to(self.device)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model, True
        model.eval()  # randomly initialized — only used as a shape-compatible fallback
        return model, False

    def _load_event_model(self, path: str):
        model = TinyEventNet(num_classes=len(EVENT_CLASSES)).to(self.device)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model, True
        model.eval()
        return model, False

    def _heuristic_cloud_score(self, img: Image.Image):
        """Brightness/flatness proxy used only when no trained cloud model is present."""
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        brightness = arr.mean()
        flatness = 1.0 - arr.std()  # low variance -> flatter/more cloud-like
        cloud_score = float(np.clip(0.6 * brightness + 0.4 * flatness, 0.0, 1.0))
        if cloud_score < 0.33:
            return "clear", cloud_score
        elif cloud_score < 0.66:
            return "partly_cloudy", cloud_score
        else:
            return "overcast", cloud_score

    def run(self, image_path: str) -> InferenceOutput:
        t0 = time.time()
        img = Image.open(image_path)
        tensor = PREPROCESS(img.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.mode == "trained":
                cloud_probs = self.cloud_model.predict_proba(tensor)[0]
                c_idx = int(torch.argmax(cloud_probs))
                cloud_class, cloud_conf = CLOUD_CLASSES[c_idx], float(cloud_probs[c_idx])
            else:
                cloud_class, cloud_conf = self._heuristic_cloud_score(img)

            # Cascade: only bother running the (more expensive) event model if the
            # frame isn't a hard discard-quality overcast frame — mirrors the real
            # onboard compute-saving design.
            if cloud_class == "overcast" and cloud_conf >= 0.6:
                event_class, event_conf = "none", 1.0
            else:
                event_probs = self.event_model.predict_proba(tensor)[0]
                e_idx = int(torch.argmax(event_probs))
                event_class, event_conf = EVENT_CLASSES[e_idx], float(event_probs[e_idx])

        latency_ms = (time.time() - t0) * 1000.0
        return InferenceOutput(
            cloud_class=cloud_class,
            cloud_confidence=cloud_conf,
            event_class=event_class,
            event_confidence=event_conf,
            latency_ms=latency_ms,
            mode=self.mode,
        )
