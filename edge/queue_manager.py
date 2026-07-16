"""
QueueManager — takes an InferenceOutput + the captured frame and decides which of the
four downlink queues the frame belongs to, and how many bytes would actually be
transmitted for it.

Compression is real, not a guessed multiplier: standard/review queues get an actual
PIL JPEG re-encode (resized + quality-reduced), and the byte count comes from the
size of that re-encoded buffer. Priority frames go down at full resolution/quality —
"no compromise" per the original design intent — so they're sent as the raw file
bytes unchanged.
"""
import io
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from schemas import QUEUE_DISCARD, QUEUE_STANDARD, QUEUE_PRIORITY, QUEUE_REVIEW  # noqa: E402


class QueueManager:
    def __init__(self, thresholds: dict, compression: dict):
        self.cloud_discard_confidence = thresholds["cloud_discard_confidence"]
        self.event_priority_confidence = thresholds["event_priority_confidence"]
        self.review_confidence_floor = thresholds.get("review_confidence_floor", 0.45)

        self.ood_review_threshold = thresholds.get("ood_review_threshold", 1.0)

        self.standard_max_dim = compression.get("standard_max_dim", 320)
        self.standard_jpeg_quality = compression.get("standard_jpeg_quality", 35)
        self.review_max_dim = compression.get("review_max_dim", 480)
        self.review_jpeg_quality = compression.get("review_jpeg_quality", 55)

    def _compress(self, image_path: str, max_dim: int, quality: int) -> bytes:
        """Re-encodes the frame as a downsized JPEG and returns the actual bytes."""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            return buf.getvalue()

    def decide_queue(self, cloud_class: str, cloud_confidence: float,
                      event_class: str, event_confidence: float, ood_score: float = 0.0) -> str:
        """Queue decision only, no byte accounting — reused by Grad-CAM/UI code that
        needs to know routing without re-touching disk."""
        if event_class != "none" and event_confidence >= self.event_priority_confidence:
            return QUEUE_PRIORITY

        # Neither model is confident about anything, OR the frame reconstructs badly
        # against every known category (autoencoder OOD score) — flag for a human to
        # look at, rather than silently guessing "routine" or "discard".
        if (cloud_confidence < self.review_confidence_floor and event_confidence < self.review_confidence_floor) \
                or ood_score >= self.ood_review_threshold:
            return QUEUE_REVIEW

        if cloud_class == "overcast" and cloud_confidence >= self.cloud_discard_confidence:
            return QUEUE_DISCARD

        return QUEUE_STANDARD

    def route(self, cloud_class: str, cloud_confidence: float,
              event_class: str, event_confidence: float,
              raw_bytes: int, image_path: str, ood_score: float = 0.0):
        """
        Returns (queue_name, downlinked_bytes, compressed_bytes).

        compressed_bytes is the actual re-encoded JPEG bytes to write to the
        downlink queue folder (None for discard/priority — priority is saved by
        copying the original file untouched; discard saves nothing).

        Priority always wins: a wildfire glimpsed through partial cloud cover is
        still worth downlinking at full resolution. Review is checked next so a
        frame the models are genuinely unsure about — or that looks like nothing in
        training, per the OOD score — doesn't get silently discarded.
        """
        queue = self.decide_queue(cloud_class, cloud_confidence, event_class, event_confidence, ood_score)

        if queue == QUEUE_PRIORITY:
            return queue, raw_bytes, None  # full resolution, no compromise
        if queue == QUEUE_DISCARD:
            return queue, 0, None
        if queue == QUEUE_REVIEW:
            data = self._compress(image_path, self.review_max_dim, self.review_jpeg_quality)
            return queue, len(data), data
        data = self._compress(image_path, self.standard_max_dim, self.standard_jpeg_quality)
        return queue, len(data), data
