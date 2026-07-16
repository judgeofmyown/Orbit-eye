"""
Shared data contracts between the edge (satellite) side and the ground station side.
Both processes read/write this exact JSON shape via the shared volume, so keep this
file identical on both ends (that's why it lives in shared/, imported by both).
"""
from dataclasses import dataclass, asdict, field
from typing import Optional
import json
import time

QUEUE_DISCARD = "discard"
QUEUE_STANDARD = "standard_downlink"
QUEUE_PRIORITY = "priority_alert"
QUEUE_REVIEW = "review"


@dataclass
class FrameResult:
    frame_id: str                 # original filename
    timestamp: float               # unix time frame was processed onboard
    cloud_class: str                # clear | partly_cloudy | overcast
    cloud_confidence: float
    event_class: str                # none | wildfire | oil_spill | other_anomaly
    event_confidence: float
    queue: str                      # discard | standard_downlink | priority_alert | review
    raw_bytes: int                  # size of the original captured frame
    downlinked_bytes: int           # size actually scheduled for transmission (0 if discarded)
    inference_latency_ms: float
    mode: str                       # "trained" | "heuristic_fallback"
    ood_score: float = 0.0          # autoencoder reconstruction-error anomaly score (higher = more unusual)
    gradcam_path: Optional[str] = None   # path to saved Grad-CAM overlay, if generated
    latitude: Optional[float] = None     # parsed from filename when the source dataset embeds GPS coords
    longitude: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "FrameResult":
        return FrameResult(**json.loads(s))


@dataclass
class TelemetrySnapshot:
    """One tick of simulated onboard 'tegrastats'-style device telemetry."""
    timestamp: float = field(default_factory=time.time)
    cpu_util_pct: float = 0.0
    sim_power_draw_watts: float = 0.0
    sim_temp_celsius: float = 0.0
    frames_processed_total: int = 0
    frames_discarded_total: int = 0
    frames_standard_total: int = 0
    frames_priority_total: int = 0
    frames_review_total: int = 0
    raw_bytes_total: int = 0
    downlinked_bytes_total: int = 0
    in_contact_window: bool = True         # whether a ground-station pass is active right now
    seconds_to_next_event: float = 0.0     # seconds until contact starts (if idle) or ends (if in contact)
    backlog_bytes_pending: int = 0         # downlinked-queue bytes not yet sent (waiting for a pass)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "TelemetrySnapshot":
        return TelemetrySnapshot(**json.loads(s))
