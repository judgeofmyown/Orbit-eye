"""
OrbitEye-Filter — Ground Station dashboard.

Reads the shared volume that the edge pipeline writes to (frame_log.jsonl,
telemetry.json, and the queues/ folders of actually-downlinked images) and shows
what a real ground station would have received.

Run:
    streamlit run ground_station/app.py
"""
import glob
import os
import sys
import time

import streamlit as st
import yaml
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.dirname(__file__))

from schemas import FrameResult, TelemetrySnapshot          # noqa: E402
from bandwidth_calculator import summarize, human_bytes     # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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


st.set_page_config(page_title="OrbitEye Ground Station", layout="wide")

cfg = load_config()
shared_output = resolve(cfg["paths"]["shared_output"])

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

    summary = summarize(frame_results)

    st.subheader("Bandwidth impact")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames captured", summary.total_frames)
    c2.metric("Raw data (onboard)", human_bytes(summary.raw_bytes_total))
    c3.metric("Actually downlinked", human_bytes(summary.downlinked_bytes_total))
    c4.metric("Bandwidth saved", f"{summary.pct_saved:.1f}%",
              delta=f"-{human_bytes(summary.bytes_saved)}")

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
    else:
        st.caption("No telemetry snapshot yet.")

    st.subheader("Downlink queues")
    q_discard, q_standard, q_priority = st.columns(3)

    with q_discard:
        st.markdown(f"### 🗑️ Discard ({summary.discarded})")
        st.caption("Never transmitted — too cloudy to be useful.")
        for r in [r for r in frame_results if r.queue == "discard"][-6:]:
            st.text(f"{r.frame_id} — {r.cloud_class} ({r.cloud_confidence:.0%})")

    with q_standard:
        st.markdown(f"### 📡 Standard downlink ({summary.standard})")
        st.caption("Clear ground, routine priority — compressed thumbnail sent.")
        img_dir = os.path.join(shared_output, "queues", "standard_downlink")
        for path in sorted(glob.glob(os.path.join(img_dir, "*")))[-4:]:
            st.image(path, caption=os.path.basename(path), width=180)

    with q_priority:
        st.markdown(f"### 🚨 Priority alert ({summary.priority})")
        st.caption("Event detected — full resolution, immediate downlink.")
        img_dir = os.path.join(shared_output, "queues", "priority_alert")
        for path in sorted(glob.glob(os.path.join(img_dir, "*")))[-4:]:
            img = Image.open(path)
            st.image(img, caption=os.path.basename(path), width=180)

    st.subheader("Full frame log")
    st.dataframe(
        [
            {
                "frame_id": r.frame_id,
                "cloud_class": r.cloud_class,
                "cloud_conf": round(r.cloud_confidence, 2),
                "event_class": r.event_class,
                "event_conf": round(r.event_confidence, 2),
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

if auto_refresh:
    time.sleep(3)
    st.rerun()
