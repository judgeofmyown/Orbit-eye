"""
InferenceEngine — the "brain" that runs onboard.

Model backend, auto-detected from what's in models/weights/ (best available wins):
  - fusion_net.pth present            -> OrbitFusionNet, one shared-trunk forward
                                          pass for both cloud+event (mode="trained_fused")
  - cloud_classifier.pth AND
    event_detector.pth present         -> two separate TinyCloudNet/TinyEventNet
                                          forward passes (mode="trained")
  - anything missing                   -> heuristic fallback for the missing piece(s)
                                          (mode="heuristic_fallback")

anomaly_autoencoder.pth, if present, additionally scores every frame with an
out-of-distribution (OOD) score — how well it reconstructs against everything the
autoencoder learned to consider "a known category" (any cloud state, or event=none).
A high score means "doesn't look like anything in training", independent of what the
supervised classifiers guessed.

Heuristic fallback (used only when no relevant .pth is present):
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
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.dirname(__file__))
from cloud_classifier import TinyCloudNet, CLOUD_CLASSES   # noqa: E402
from event_detector import TinyEventNet, EVENT_CLASSES     # noqa: E402
from fusion_net import OrbitFusionNet                       # noqa: E402
from anomaly_autoencoder import TinyAutoencoder              # noqa: E402
from gradcam import GradCAM, overlay_heatmap                 # noqa: E402

PREPROCESS = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

RAW_TENSOR = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


@dataclass
class InferenceOutput:
    cloud_class: str
    cloud_confidence: float
    event_class: str
    event_confidence: float
    latency_ms: float
    mode: str  # "trained" | "trained_fused" | "heuristic_fallback"
    ood_score: float = 0.0
    gradcam_path: Optional[str] = None


class InferenceEngine:
    def __init__(self, cloud_weights_path: str, event_weights_path: str,
                 fusion_weights_path: Optional[str] = None,
                 autoencoder_weights_path: Optional[str] = None,
                 device: str = "cpu"):
        self.device = device
        self.backend = "separate"
        self.fusion_model = None
        self.cloud_model = None
        self.event_model = None

        if fusion_weights_path and os.path.exists(fusion_weights_path):
            self.fusion_model, fused_ok = self._load_fusion_model(fusion_weights_path)
            if fused_ok:
                self.backend = "fusion"

        if self.backend != "fusion":
            self.cloud_model, cloud_trained = self._load_cloud_model(cloud_weights_path)
            self.event_model, event_trained = self._load_event_model(event_weights_path)
            self.mode = "trained" if (cloud_trained and event_trained) else "heuristic_fallback"
        else:
            self.mode = "trained_fused"

        if self.mode == "heuristic_fallback":
            print(
                "[InferenceEngine] WARNING: one or both trained weight files were not "
                "found. Running in heuristic_fallback mode — results are placeholders, "
                "not real cloud/event detection. Train models and drop .pth files into "
                "models/weights/ to switch to real inference."
            )

        self.autoencoder, self.ood_threshold = self._load_autoencoder(autoencoder_weights_path)

    def _load_fusion_model(self, path: str):
        model = OrbitFusionNet(len(CLOUD_CLASSES), len(EVENT_CLASSES)).to(self.device)
        try:
            ckpt = torch.load(path, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model, True
        except Exception as e:
            print(f"[InferenceEngine] WARNING: failed to load fusion weights ({e}); "
                  f"falling back to separate models.")
            return None, False

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

    def _load_autoencoder(self, path: Optional[str]):
        if not path or not os.path.exists(path):
            return None, None
        model = TinyAutoencoder().to(self.device)
        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt["ood_threshold"]

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

    @torch.no_grad()
    def _compute_ood_score(self, tensor_raw: torch.Tensor) -> float:
        if self.autoencoder is None:
            return 0.0
        recon = self.autoencoder(tensor_raw)
        err = torch.mean((recon - tensor_raw) ** 2).item()
        return err / self.ood_threshold if self.ood_threshold else 0.0

    def run(self, image_path: str) -> InferenceOutput:
        t0 = time.time()
        img = Image.open(image_path)
        tensor = PREPROCESS(img.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.backend == "fusion":
                cloud_probs, event_probs_full = self.fusion_model.predict_proba(tensor)
                cloud_probs = cloud_probs[0]
                c_idx = int(torch.argmax(cloud_probs))
                cloud_class, cloud_conf = CLOUD_CLASSES[c_idx], float(cloud_probs[c_idx])

                if cloud_class == "overcast" and cloud_conf >= 0.6:
                    event_class, event_conf = "none", 1.0
                else:
                    event_probs = event_probs_full[0]
                    e_idx = int(torch.argmax(event_probs))
                    event_class, event_conf = EVENT_CLASSES[e_idx], float(event_probs[e_idx])

            elif self.mode == "trained":
                cloud_probs = self.cloud_model.predict_proba(tensor)[0]
                c_idx = int(torch.argmax(cloud_probs))
                cloud_class, cloud_conf = CLOUD_CLASSES[c_idx], float(cloud_probs[c_idx])

                # Cascade: only bother running the (more expensive) event model if the
                # frame isn't a hard discard-quality overcast frame — mirrors the real
                # onboard compute-saving design.
                if cloud_class == "overcast" and cloud_conf >= 0.6:
                    event_class, event_conf = "none", 1.0
                else:
                    event_probs = self.event_model.predict_proba(tensor)[0]
                    e_idx = int(torch.argmax(event_probs))
                    event_class, event_conf = EVENT_CLASSES[e_idx], float(event_probs[e_idx])
            else:
                cloud_class, cloud_conf = self._heuristic_cloud_score(img)
                if cloud_class == "overcast" and cloud_conf >= 0.6:
                    event_class, event_conf = "none", 1.0
                else:
                    event_probs = self.event_model.predict_proba(tensor)[0]
                    e_idx = int(torch.argmax(event_probs))
                    event_class, event_conf = EVENT_CLASSES[e_idx], float(event_probs[e_idx])

            tensor_raw = RAW_TENSOR(img.convert("RGB")).unsqueeze(0).to(self.device)
            ood_score = self._compute_ood_score(tensor_raw)

        latency_ms = (time.time() - t0) * 1000.0
        return InferenceOutput(
            cloud_class=cloud_class,
            cloud_confidence=cloud_conf,
            event_class=event_class,
            event_confidence=event_conf,
            latency_ms=latency_ms,
            mode=self.mode,
            ood_score=ood_score,
        )

    def explain(self, image_path: str, head: str, class_idx: int) -> Optional[Image.Image]:
        """
        Generates a Grad-CAM overlay explaining why `head` ("cloud" or "event")
        predicted `class_idx`. Only meaningful when trained weights are loaded —
        returns None in heuristic_fallback mode since there's no learned signal to
        explain. Runs a fresh forward pass with gradients enabled (the normal `run()`
        path uses no_grad for speed), so only call this for frames worth the extra
        compute — e.g. priority/review queue frames, not every frame.
        """
        if self.mode == "heuristic_fallback" and head == "cloud":
            return None  # heuristic cloud score isn't a differentiable model, nothing to explain

        img = Image.open(image_path).convert("RGB")
        tensor = PREPROCESS(img).unsqueeze(0).to(self.device)
        tensor.requires_grad_(False)

        if self.backend == "fusion":
            target_layer = (self.fusion_model.cloud_head_block if head == "cloud"
                             else self.fusion_model.event_head_block)
            cam_engine = GradCAM(self.fusion_model, target_layer)
            cloud_logits, event_logits = self.fusion_model(tensor)
            logits = cloud_logits if head == "cloud" else event_logits
        else:
            model = self.cloud_model if head == "cloud" else self.event_model
            target_layer = model.blocks[-1]
            cam_engine = GradCAM(model, target_layer)
            logits = model(tensor)

        cam = cam_engine.compute(logits, class_idx)
        return overlay_heatmap(img, cam)
