"""
OrbitEye-Filter — Ground Station dashboard.

Reads the shared volume that the edge pipeline writes to (frame_log.jsonl,
telemetry.json, the queues/ folders of actually-downlinked images, and gradcam/
overlays) and shows what a real ground station would have received.

Run:
    streamlit run ground_station/app.py
"""
import glob
import json
import os
import sys
import time

import pandas as pd
import streamlit as st
import yaml
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.dirname(__file__))

from schemas import FrameResult, TelemetrySnapshot          # noqa: E402
from bandwidth_calculator import summarize, human_bytes     # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLOUD_CLASSES = ["clear", "overcast", "partly_cloudy"]
EVENT_CLASSES = ["none", "oil_spill", "other_anomaly", "wildfire"]


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


@st.cache_data(ttl=1)
def load_config():
    with open(os.path.join(PROJECT_ROOT, "edge", "config.yaml")) as f:
        return yaml.safe_load(f)


def load_frame_log(shared_output: str):
    log_path = os.path.join(shared_output, "frame_log.jsonl")
    results = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(FrameResult.from_json(line))
    return results


def load_telemetry(shared_output: str):
    path = os.path.join(shared_output, "telemetry.json")
    if os.path.exists(path):
        with open(path) as f:
            return TelemetrySnapshot.from_json(f.read())
    return None


def find_frame_source(frame_id: str, camera_storage: str, shared_output: str):
    """Locates the original frame bytes for re-use by the correction/fine-tune flow —
    prefer the still-present camera_storage copy, fall back to whatever queue folder
    it landed in."""
    direct = os.path.join(camera_storage, frame_id)
    if os.path.exists(direct):
        return direct
    for q in ("priority_alert", "review", "standard_downlink"):
        p = os.path.join(shared_output, "queues", q, frame_id)
        if os.path.exists(p):
            return p
    return None


st.set_page_config(page_title="OrbitEye Ground Station", layout="wide")

cfg = load_config()
shared_output = resolve(cfg["paths"]["shared_output"])
camera_storage = resolve(cfg["paths"]["camera_storage"])

st.title("🛰️ OrbitEye-Filter — Ground Station")
st.caption(
    "This view shows only what the onboard Edge AI decided was worth transmitting. "
    "Discarded (cloudy) frames never left the satellite and never touched the radio."
)

auto_refresh = st.sidebar.checkbox("Auto-refresh (every 3s)", value=False)
if st.sidebar.button("Refresh now") or True:
    pass  # every script rerun reloads from disk anyway

frame_results = load_frame_log(shared_output)
telemetry = load_telemetry(shared_output)

if not frame_results:
    st.info(
        "No frames processed yet. On the edge side, run:\n\n"
        "```\npython edge/generate_sample_frames.py\npython edge/run_edge_pipeline.py\n```"
    )
else:
    if frame_results[0].mode == "heuristic_fallback":
        st.warning(
            "⚠️ Running in **heuristic_fallback** mode — no trained model weights were "
            "found onboard. These classifications are placeholder heuristics, not real "
            "cloud/event detection. Train the models and drop weights into "
            "`models/weights/` for real results.",
            icon="⚠️",
        )
    elif frame_results[0].mode == "trained_fused":
        st.success(
            "⚡ Running on **OrbitFusionNet** — one shared-trunk model doing both "
            "cloud + event detection in a single forward pass (lower onboard compute "
            "than the two separate networks).",
            icon="⚡",
        )

    summary = summarize(frame_results)

    st.subheader("Bandwidth impact")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames captured", summary.total_frames)
    c2.metric("Raw data (onboard)", human_bytes(summary.raw_bytes_total))
    c3.metric("Actually downlinked", human_bytes(summary.downlinked_bytes_total))
    c4.metric("Bandwidth saved", f"{summary.pct_saved:.1f}%",
              delta=f"-{human_bytes(summary.bytes_saved)}")

    st.subheader("Bandwidth over time")
    running_raw, running_down = 0, 0
    rows = []
    for i, r in enumerate(frame_results):
        running_raw += r.raw_bytes
        running_down += r.downlinked_bytes
        rows.append({"frame #": i + 1, "raw (cumulative)": running_raw,
                      "downlinked (cumulative)": running_down})
    chart_df = pd.DataFrame(rows).set_index("frame #")
    st.line_chart(chart_df)
    st.caption(
        "The gap between the two lines *is* the pitch — everything between them is "
        "bandwidth that never touched the radio."
    )

    st.subheader("Onboard device telemetry (simulated Jetson envelope)")
    if telemetry:
        t1, t2, t3 = st.columns(3)
        t1.metric("Simulated power draw", f"{telemetry.sim_power_draw_watts} W")
        t2.metric("Simulated temperature", f"{telemetry.sim_temp_celsius} °C")
        t3.metric("CPU utilization", f"{telemetry.cpu_util_pct}%")
        st.caption(
            "Simulated, not read from real Jetson hardware (no `tegrastats`/`jtop` "
            "access in this environment) — see README section 6 for how to wire up "
            "real telemetry on actual hardware."
        )

        st.markdown("**Downlink pass status**")
        p1, p2, p3 = st.columns(3)
        if telemetry.in_contact_window:
            p1.metric("Ground station contact", "🟢 IN CONTACT")
            p2.metric("Contact ends in", f"{telemetry.seconds_to_next_event:.0f}s (sim)")
        else:
            p1.metric("Ground station contact", "🔴 no contact")
            p2.metric("Next contact in", f"{telemetry.seconds_to_next_event:.0f}s (sim)")
        p3.metric("Backlog pending downlink", human_bytes(telemetry.backlog_bytes_pending))
        st.caption(
            "A real LEO CubeSat only has downlink during short ground-station passes, "
            "not continuously — queued frames wait in the backlog until the next "
            "contact window. See edge/orbit_pass_sim.py."
        )
    else:
        st.caption("No telemetry snapshot yet.")

    st.subheader("Downlink queues")
    q_discard, q_standard, q_priority, q_review = st.columns(4)

    with q_discard:
        st.markdown(f"### 🗑️ Discard ({summary.discarded})")
        st.caption("Never transmitted — too cloudy to be useful.")
        for r in [r for r in frame_results if r.queue == "discard"][-6:]:
            st.text(f"{r.frame_id} — {r.cloud_class} ({r.cloud_confidence:.0%})")

    with q_standard:
        st.markdown(f"### 📡 Standard ({summary.standard})")
        st.caption("Clear ground, routine — compressed thumbnail sent.")
        img_dir = os.path.join(shared_output, "queues", "standard_downlink")
        for path in sorted(glob.glob(os.path.join(img_dir, "*")))[-4:]:
            st.image(path, caption=os.path.basename(path), width=180)

    with q_priority:
        priority_count = sum(1 for r in frame_results if r.queue == "priority_alert")
        st.markdown(f"### 🚨 Priority ({priority_count})")
        st.caption("Event detected — full resolution, immediate downlink.")
        priority_frames = [r for r in frame_results if r.queue == "priority_alert"][-4:]
        for r in priority_frames:
            if r.gradcam_path and os.path.exists(r.gradcam_path):
                st.image(r.gradcam_path, caption=f"{r.frame_id} (Grad-CAM: {r.event_class})", width=180)
            else:
                path = os.path.join(shared_output, "queues", "priority_alert", r.frame_id)
                if os.path.exists(path):
                    st.image(Image.open(path), caption=r.frame_id, width=180)

    with q_review:
        review_count = sum(1 for r in frame_results if r.queue == "review")
        st.markdown(f"### 🔍 Review ({review_count})")
        st.caption("Models weren't confident, or the frame looked unlike anything in training.")
        review_frames = [r for r in frame_results if r.queue == "review"][-4:]
        for r in review_frames:
            if r.gradcam_path and os.path.exists(r.gradcam_path):
                st.image(r.gradcam_path,
                          caption=f"{r.frame_id} (ood={r.ood_score:.2f})", width=180)
            else:
                path = os.path.join(shared_output, "queues", "review", r.frame_id)
                if os.path.exists(path):
                    st.image(path, caption=f"{r.frame_id} (ood={r.ood_score:.2f})", width=180)

    geo_frames = [r for r in frame_results if r.latitude is not None and r.longitude is not None]
    if geo_frames:
        st.subheader("Event map")
        st.caption(
            "Frames whose source filename embeds GPS coordinates (the wildfire chip "
            "dataset does this natively). Color: red = priority alert, others = grey."
        )
        map_df = pd.DataFrame([
            {
                "lat": r.latitude, "lon": r.longitude,
                "color": [220, 40, 40] if r.queue == "priority_alert" else [120, 120, 120],
            }
            for r in geo_frames
        ])
        st.map(map_df, latitude="lat", longitude="lon", color="color", size=200)

    st.subheader("Full frame log")
    st.dataframe(
        [
            {
                "frame_id": r.frame_id,
                "cloud_class": r.cloud_class,
                "cloud_conf": round(r.cloud_confidence, 2),
                "event_class": r.event_class,
                "event_conf": round(r.event_confidence, 2),
                "ood_score": round(r.ood_score, 2),
                "queue": r.queue,
                "raw_bytes": r.raw_bytes,
                "downlinked_bytes": r.downlinked_bytes,
                "latency_ms": round(r.inference_latency_ms, 1),
                "mode": r.mode,
            }
            for r in reversed(frame_results)
        ],
        use_container_width=True,
    )

    st.subheader("🛠️ Correct a misclassification (simulated model-update uplink)")
    st.caption(
        "Flag a wrong label from the ground. Corrections queue up in "
        "shared_data/corrections.jsonl; run `python models/finetune_from_corrections.py` "
        "to fine-tune the onboard model on them and simulate uplinking the update to "
        "the satellite — a stand-in for real on-orbit incremental learning."
    )
    recent = frame_results[-30:][::-1]
    frame_choice = st.selectbox(
        "Frame", options=[r.frame_id for r in recent],
        format_func=lambda fid: f"{fid} (cloud={next(r.cloud_class for r in recent if r.frame_id==fid)}, "
                                 f"event={next(r.event_class for r in recent if r.frame_id==fid)})",
    )
    head_choice = st.radio("What was wrong?", ["event", "cloud"], horizontal=True)
    label_options = EVENT_CLASSES if head_choice == "event" else CLOUD_CLASSES
    correct_label = st.selectbox("Correct label", label_options)

    if st.button("Submit correction"):
        source_path = find_frame_source(frame_choice, camera_storage, shared_output)
        if source_path is None:
            st.error(f"Could not locate source image for {frame_choice} — it may have "
                      "been discarded onboard and never downlinked, so there's nothing "
                      "to correct against.")
        else:
            corrections_path = os.path.join(shared_output, "corrections.jsonl")
            entry = {
                "frame_id": frame_choice,
                "source_path": source_path,
                "head": head_choice,
                "corrected_label": correct_label,
                "timestamp": time.time(),
            }
            with open(corrections_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            st.success(f"Queued correction: {frame_choice} -> {head_choice}={correct_label}")

    corrections_path = os.path.join(shared_output, "corrections.jsonl")
    if os.path.exists(corrections_path):
        with open(corrections_path) as f:
            n_pending = sum(1 for _ in f)
        st.caption(f"{n_pending} correction(s) queued for the next fine-tune pass.")

if auto_refresh:
    time.sleep(3)
    st.rerun()
