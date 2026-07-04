"""
QueueManager — takes an InferenceOutput + raw frame size and decides which of the
three downlink queues the frame belongs to, and how many bytes would actually be
transmitted for it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from schemas import QUEUE_DISCARD, QUEUE_STANDARD, QUEUE_PRIORITY  # noqa: E402


class QueueManager:
    def __init__(self, thresholds: dict, compression: dict):
        self.cloud_discard_confidence = thresholds["cloud_discard_confidence"]
        self.event_priority_confidence = thresholds["event_priority_confidence"]
        self.discard_mult = compression["discard_size_multiplier"]
        self.standard_mult = compression["standard_size_multiplier"]
        self.priority_mult = compression["priority_size_multiplier"]

    def route(self, cloud_class: str, cloud_confidence: float,
              event_class: str, event_confidence: float, raw_bytes: int):
        """
        Returns (queue_name, downlinked_bytes).

        Priority always wins over discard: a wildfire glimpsed through partial cloud
        cover is still worth downlinking at full resolution.
        """
        if event_class != "none" and event_confidence >= self.event_priority_confidence:
            return QUEUE_PRIORITY, int(raw_bytes * self.priority_mult)

        if cloud_class == "overcast" and cloud_confidence >= self.cloud_discard_confidence:
            return QUEUE_DISCARD, int(raw_bytes * self.discard_mult)

        return QUEUE_STANDARD, int(raw_bytes * self.standard_mult)
