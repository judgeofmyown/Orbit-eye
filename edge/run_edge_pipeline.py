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
import shutil
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))

from inference_engine import InferenceEngine  # noqa: E402
from queue_manager import QueueManager        # noqa: E402
from jetson_stats_sim import JetsonStatsSim   # noqa: E402
from schemas import FrameResult               # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def process_frame(frame_path, engine, qmgr, stats_sim, shared_output):
    result = engine.run(frame_path)
    raw_bytes = os.path.getsize(frame_path)
    queue, downlinked_bytes = qmgr.route(
        result.cloud_class, result.cloud_confidence,
        result.event_class, result.event_confidence,
        raw_bytes,
    )

    frame_result = FrameResult(
        frame_id=os.path.basename(frame_path),
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
    )

    # append to the frame log (ground truth of what happened onboard)
    log_path = os.path.join(shared_output, "frame_log.jsonl")
    with open(log_path, "a") as f:
        f.write(frame_result.to_json() + "\n")

    # only copy bytes for anything that would actually be downlinked
    if queue != "discard":
        dest_dir = os.path.join(shared_output, "queues", queue)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(frame_path, os.path.join(dest_dir, os.path.basename(frame_path)))

    stats_sim.record_frame(queue, raw_bytes, downlinked_bytes)
    telemetry = stats_sim.snapshot(is_actively_inferring=True)
    with open(os.path.join(shared_output, "telemetry.json"), "w") as f:
        f.write(telemetry.to_json())

    print(f"[{frame_result.frame_id}] cloud={result.cloud_class}({result.cloud_confidence:.2f}) "
          f"event={result.event_class}({result.event_confidence:.2f}) -> {queue} "
          f"({raw_bytes}B -> {downlinked_bytes}B, {result.latency_ms:.1f}ms, mode={result.mode})")


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
    )
    qmgr = QueueManager(cfg["thresholds"], cfg["compression"])
    stats_sim = JetsonStatsSim(cfg["jetson_sim"])

    processed = set()

    def process_new_frames():
        patterns = ["*.jpg", "*.jpeg", "*.png"]
        frames = sorted(set(sum((glob.glob(os.path.join(camera_storage, p)) for p in patterns), [])))
        new_frames = [f for f in frames if f not in processed]
        for frame_path in new_frames:
            process_frame(frame_path, engine, qmgr, stats_sim, shared_output)
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
                telemetry = stats_sim.snapshot(is_actively_inferring=False)
                with open(os.path.join(shared_output, "telemetry.json"), "w") as f:
                    f.write(telemetry.to_json())
                time.sleep(cfg["pipeline"]["loop_interval_seconds"])
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
