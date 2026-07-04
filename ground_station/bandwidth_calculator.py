"""
Pure functions for turning the frame log into the bandwidth-savings numbers the
pitch is built around ("up to 60% reduction in downlink bandwidth").
"""
from dataclasses import dataclass
from typing import List

from schemas import FrameResult


@dataclass
class BandwidthSummary:
    total_frames: int
    discarded: int
    standard: int
    priority: int
    raw_bytes_total: int
    downlinked_bytes_total: int
    bytes_saved: int
    pct_saved: float


def summarize(frame_results: List[FrameResult]) -> BandwidthSummary:
    total = len(frame_results)
    discarded = sum(1 for r in frame_results if r.queue == "discard")
    standard = sum(1 for r in frame_results if r.queue == "standard_downlink")
    priority = sum(1 for r in frame_results if r.queue == "priority_alert")

    raw_total = sum(r.raw_bytes for r in frame_results)
    downlinked_total = sum(r.downlinked_bytes for r in frame_results)
    saved = raw_total - downlinked_total
    pct_saved = (saved / raw_total * 100.0) if raw_total > 0 else 0.0

    return BandwidthSummary(
        total_frames=total,
        discarded=discarded,
        standard=standard,
        priority=priority,
        raw_bytes_total=raw_total,
        downlinked_bytes_total=downlinked_total,
        bytes_saved=saved,
        pct_saved=pct_saved,
    )


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"
