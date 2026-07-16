"""
OrbitPassSim — simulates the fact that a real CubeSat does NOT have continuous
downlink. It only talks to a ground station during short contact windows that recur
once per orbit (a typical LEO orbit is ~90 minutes, with only a handful of minutes of
actual ground-station visibility per pass).

Frames the QueueManager decides to keep (standard/review/priority) don't leave the
satellite immediately — they queue in a "backlog" that only drains during a contact
window, at a bounded downlink bitrate. This is what makes the dashboard's backlog
number behave like a real ops queue instead of an instant teleport.

Runs on a virtual clock scaled by `time_scale` so a demo run (a few seconds of wall
time) can still show multiple simulated orbits and contact windows, instead of
requiring you to sit through a real 90-minute period.
"""
import time


class OrbitPassSim:
    def __init__(self, cfg: dict):
        self.orbit_period_seconds = cfg.get("orbit_period_seconds", 5400)      # 90 min LEO orbit
        self.contact_window_seconds = cfg.get("contact_window_seconds", 600)   # 10 min pass
        self.time_scale = cfg.get("time_scale", 60)                            # demo speed-up
        self.max_downlink_bps = cfg.get("max_downlink_bytes_per_second", 250_000)  # ~2Mbps-ish link

        self._virtual_time = 0.0
        self._last_wall = time.time()
        self._backlog_bytes = 0
        self.in_contact_window = False
        self.seconds_to_next_event = self.orbit_period_seconds

        self._update_phase()

    def _update_phase(self):
        phase = self._virtual_time % self.orbit_period_seconds
        self.in_contact_window = phase < self.contact_window_seconds
        if self.in_contact_window:
            self.seconds_to_next_event = self.contact_window_seconds - phase
        else:
            self.seconds_to_next_event = self.orbit_period_seconds - phase

    def tick(self):
        """Advances the virtual orbital clock by however much wall time has passed."""
        now = time.time()
        dt_wall = max(now - self._last_wall, 0.0)
        self._last_wall = now
        self._virtual_time += dt_wall * self.time_scale
        self._update_phase()
        return dt_wall * self.time_scale

    def enqueue(self, downlinked_bytes: int):
        """Adds a frame's downlinked bytes to the pending backlog."""
        self._backlog_bytes += max(downlinked_bytes, 0)

    def drain_if_in_contact(self) -> int:
        """
        If currently in a contact window, drains backlog at the configured link
        rate (bounded by how much virtual time elapsed since the last tick).
        Returns the remaining backlog in bytes either way.
        """
        if self.in_contact_window and self._backlog_bytes > 0:
            # Approximate: assume ~1 simulated second of link time became available
            # since the last tick (tick() is called every frame, so this keeps the
            # drain rate tied to the configured bitrate without needing a separate
            # elapsed-time argument here).
            drained = min(self._backlog_bytes, self.max_downlink_bps)
            self._backlog_bytes -= drained
        return self._backlog_bytes
