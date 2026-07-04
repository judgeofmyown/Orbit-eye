"""
JetsonStatsSim — produces plausible-looking onboard device telemetry (power draw,
temperature, CPU utilization) so the ground station dashboard has something realistic
to show, without needing actual Jetson hardware.

This is explicitly a simulation, not a reading from `tegrastats` or `jtop`. If you
later deploy this on a real Jetson, swap this module for a wrapper around `jtop`
(https://github.com/rbonghi/jetson_stats) and keep the same TelemetrySnapshot fields
so the ground station code needs zero changes.
"""
import random
import time

from schemas import TelemetrySnapshot  # noqa


class JetsonStatsSim:
    def __init__(self, cfg: dict):
        self.idle_power = cfg["idle_power_watts"]
        self.active_power = cfg["active_power_watts"]
        self.max_temp = cfg["max_temp_celsius"]
        self.ambient_temp = cfg["ambient_temp_celsius"]
        self._temp = self.ambient_temp
        self._running_totals = dict(
            frames_processed_total=0,
            frames_discarded_total=0,
            frames_standard_total=0,
            frames_priority_total=0,
            raw_bytes_total=0,
            downlinked_bytes_total=0,
        )

    def record_frame(self, queue: str, raw_bytes: int, downlinked_bytes: int):
        self._running_totals["frames_processed_total"] += 1
        self._running_totals["raw_bytes_total"] += raw_bytes
        self._running_totals["downlinked_bytes_total"] += downlinked_bytes
        key = {
            "discard": "frames_discarded_total",
            "standard_downlink": "frames_standard_total",
            "priority_alert": "frames_priority_total",
        }[queue]
        self._running_totals[key] += 1

    def snapshot(self, is_actively_inferring: bool) -> TelemetrySnapshot:
        # Simple thermal model: temp drifts toward a "load-dependent" target with noise.
        target_temp = self.max_temp if is_actively_inferring else self.ambient_temp + 8
        self._temp += (target_temp - self._temp) * 0.15 + random.uniform(-0.5, 0.5)
        power = (self.active_power if is_actively_inferring else self.idle_power)
        power += random.uniform(-0.3, 0.3)
        cpu_util = random.uniform(55, 85) if is_actively_inferring else random.uniform(5, 15)

        return TelemetrySnapshot(
            timestamp=time.time(),
            cpu_util_pct=round(cpu_util, 1),
            sim_power_draw_watts=round(power, 2),
            sim_temp_celsius=round(self._temp, 1),
            **self._running_totals,
        )
