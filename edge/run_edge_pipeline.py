"""
run_edge_pipeline.py — the "onboard satellite loop."

Watches edge/camera_storage/ for image frames, runs each through the InferenceEngine,
routes it via QueueManager, and writes:
  - one JSON-lines record per frame to <shared_output>/frame_log.jsonl
  - a rolling telemetry snapshot to <shared_output>/telemetry.json
  - the actual image bytes into <shared_output>/queues/{standard_downlink,priority_alert}/
    (discarded frames are NOT copied — that's the entire point: they never leave the
    satellite)

This is the script that runs inside the "edge-sim" Docker container to simulate an
onboard Jetson-class device.

Usage:
    python edge/run_edge_pipeline.py                 # process once and exit
    python edge/run_edge_pipeline.py --loop           # keep watching for new frames
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from inference_engine import InferenceEngine  # noqa: E402
from queue_manager import QueueManager        # noqa: E402
from jetson_stats_sim import JetsonStatsSim   # noqa: E402
from orbit_pass_sim import OrbitPassSim       # noqa: E402
from schemas import FrameResult, QUEUE_DISCARD, QUEUE_PRIORITY, QUEUE_REVIEW  # noqa: E402
from cloud_classifier import CLOUD_CLASSES    # noqa: E402
from event_detector import EVENT_CLASSES      # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Some source datasets (e.g. the wildfire chips) embed capture coordinates directly
# in the filename, like "-113.917...,50.901....jpg" -> (lon, lat).
GPS_FILENAME_RE = re.compile(r"(-?\d+\.\d+),(-?\d+\.\d+)")


def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def parse_gps(frame_id: str):
    m = GPS_FILENAME_RE.search(frame_id)
    if not m:
        return None, None
    lon, lat = float(m.group(1)), float(m.group(2))
    return lat, lon


def generate_gradcam(engine, frame_path, frame_id, result, queue, shared_output):
    """Explains whichever prediction is relevant to why this frame got flagged:
    always the event head for priority (that's literally what triggered it — cloud
    and event confidences aren't on a comparable scale, so picking by raw magnitude
    would often wrongly pick cloud just because it happened to be more confident);
    whichever head is more confident for review (neither triggered a clear decision,
    so show the closer call)."""
    if queue == QUEUE_PRIORITY:
        head, class_name, classes = "event", result.event_class, EVENT_CLASSES
    elif result.event_class != "none" and result.event_confidence >= result.cloud_confidence:
        head, class_name, classes = "event", result.event_class, EVENT_CLASSES
    else:
        head, class_name, classes = "cloud", result.cloud_class, CLOUD_CLASSES

    overlay = engine.explain(frame_path, head, classes.index(class_name))
    if overlay is None:
        return None

    dest_dir = os.path.join(shared_output, "gradcam")
    os.makedirs(dest_dir, exist_ok=True)
    stem, _ = os.path.splitext(frame_id)
    dest_path = os.path.join(dest_dir, f"{stem}_{head}.jpg")
    overlay.save(dest_path, quality=80)
    return dest_path


def process_frame(frame_path, engine, qmgr, stats_sim, orbit_sim, shared_output):
    result = engine.run(frame_path)
    raw_bytes = os.path.getsize(frame_path)
    queue, downlinked_bytes, compressed_bytes = qmgr.route(
        result.cloud_class, result.cloud_confidence,
        result.event_class, result.event_confidence,
        raw_bytes, frame_path, result.ood_score,
    )

    frame_id = os.path.basename(frame_path)
    lat, lon = parse_gps(frame_id)

    gradcam_path = None
    if queue in (QUEUE_PRIORITY, QUEUE_REVIEW):
        gradcam_path = generate_gradcam(engine, frame_path, frame_id, result, queue, shared_output)

    frame_result = FrameResult(
        frame_id=frame_id,
        timestamp=time.time(),
        cloud_class=result.cloud_class,
        cloud_confidence=result.cloud_confidence,
        event_class=result.event_class,
        event_confidence=result.event_confidence,
        queue=queue,
        raw_bytes=raw_bytes,
        downlinked_bytes=downlinked_bytes,
        inference_latency_ms=result.latency_ms,
        mode=result.mode,
        ood_score=result.ood_score,
        gradcam_path=gradcam_path,
        latitude=lat,
        longitude=lon,
    )

    # append to the frame log (ground truth of what happened onboard)
    log_path = os.path.join(shared_output, "frame_log.jsonl")
    with open(log_path, "a") as f:
        f.write(frame_result.to_json() + "\n")

    # queue the actual bytes that would be downlinked: real re-encoded JPEG for
    # standard/review, the untouched original for priority, nothing for discard
    if queue != QUEUE_DISCARD:
        dest_dir = os.path.join(shared_output, "queues", queue)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, frame_id)
        if compressed_bytes is not None:
            with open(dest_path, "wb") as f:
                f.write(compressed_bytes)
        else:
            shutil.copy(frame_path, dest_path)

    orbit_sim.tick()
    if queue != QUEUE_DISCARD:
        orbit_sim.enqueue(downlinked_bytes)
    backlog_bytes = orbit_sim.drain_if_in_contact()

    stats_sim.record_frame(queue, raw_bytes, downlinked_bytes)
    telemetry = stats_sim.snapshot(is_actively_inferring=True)
    telemetry.in_contact_window = orbit_sim.in_contact_window
    telemetry.seconds_to_next_event = orbit_sim.seconds_to_next_event
    telemetry.backlog_bytes_pending = backlog_bytes
    with open(os.path.join(shared_output, "telemetry.json"), "w") as f:
        f.write(telemetry.to_json())

    print(f"[{frame_result.frame_id}] cloud={result.cloud_class}({result.cloud_confidence:.2f}) "
          f"event={result.event_class}({result.event_confidence:.2f}) -> {queue} "
          f"({raw_bytes}B -> {downlinked_bytes}B, {result.latency_ms:.1f}ms, mode={result.mode}) "
          f"[{'IN CONTACT' if orbit_sim.in_contact_window else 'no contact'}, "
          f"backlog={backlog_bytes}B]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="keep watching camera_storage for new frames")
    args = parser.parse_args()

    cfg = load_config()
    camera_storage = resolve(cfg["paths"]["camera_storage"])
    shared_output = resolve(cfg["paths"]["shared_output"])
    os.makedirs(shared_output, exist_ok=True)
    os.makedirs(camera_storage, exist_ok=True)

    engine = InferenceEngine(
        cloud_weights_path=resolve(cfg["paths"]["cloud_weights"]),
        event_weights_path=resolve(cfg["paths"]["event_weights"]),
        fusion_weights_path=resolve(cfg["paths"]["fusion_weights"]),
        autoencoder_weights_path=resolve(cfg["paths"]["autoencoder_weights"]),
    )
    qmgr = QueueManager(cfg["thresholds"], cfg["compression"])
    stats_sim = JetsonStatsSim(cfg["jetson_sim"])
    orbit_sim = OrbitPassSim(cfg.get("orbit_pass", {}))

    processed = set()

    def process_new_frames():
        patterns = ["*.jpg", "*.jpeg", "*.png"]
        frames = sorted(set(sum((glob.glob(os.path.join(camera_storage, p)) for p in patterns), [])))
        new_frames = [f for f in frames if f not in processed]
        for frame_path in new_frames:
            process_frame(frame_path, engine, qmgr, stats_sim, orbit_sim, shared_output)
            processed.add(frame_path)
            time.sleep(cfg["pipeline"]["loop_interval_seconds"])
        return len(new_frames)

    if not args.loop:
        n = process_new_frames()
        if n == 0:
            print(f"No frames found in {camera_storage}. "
                  f"Run `python edge/generate_sample_frames.py` for a quick demo set, "
                  f"or drop real images there.")
        return

    print(f"Watching {camera_storage} for new frames (Ctrl+C to stop)...")
    try:
        while True:
            n = process_new_frames()
            if n == 0:
                # idle tick: still emit a telemetry snapshot so the dashboard shows "idle"
                orbit_sim.tick()
                backlog_bytes = orbit_sim.drain_if_in_contact()
                telemetry = stats_sim.snapshot(is_actively_inferring=False)
                telemetry.in_contact_window = orbit_sim.in_contact_window
                telemetry.seconds_to_next_event = orbit_sim.seconds_to_next_event
                telemetry.backlog_bytes_pending = backlog_bytes
                with open(os.path.join(shared_output, "telemetry.json"), "w") as f:
                    f.write(telemetry.to_json())
                time.sleep(cfg["pipeline"]["loop_interval_seconds"])
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
