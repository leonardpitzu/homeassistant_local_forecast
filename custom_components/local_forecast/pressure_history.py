"""Hourly sea-level-pressure ring buffer.

Keeps one pressure sample per hour over a rolling 24-hour window so the
integration can serve the WMO 3-hour tendency and a 24-hour synoptic mean
without a chain of statistics/derivative helpers.

The existing StateEstimator ring buffer is sampled at sensor cadence and
only spans a few hours; this buffer is sampled hourly and is persisted
across restarts by the pressure-tendency sensor (RestoreEntity), so both
derived sensors survive a Home Assistant reboot instead of warming up for
three hours.

Pure Python — no Home Assistant dependencies, so it is unit-testable in
isolation.
"""

from __future__ import annotations

import math
from collections import deque

# One sample per hour, keep a little over 24 h.
SAMPLE_INTERVAL_S: float = 3600.0
WINDOW_S: float = 24 * 3600.0
# A sample counts as "3 h ago" if it lands within this of the target.
NEAREST_TOLERANCE_S: float = 3600.0
# 24 h window + one hour margin for the ~25 hourly samples.
_MAXLEN: int = 26


class PressureHistory:
    """Rolling buffer of hourly (timestamp, sea-level pressure) samples."""

    def __init__(
        self,
        *,
        window_s: float = WINDOW_S,
        sample_interval_s: float = SAMPLE_INTERVAL_S,
        tolerance_s: float = NEAREST_TOLERANCE_S,
        maxlen: int = _MAXLEN,
    ) -> None:
        self._window = window_s
        self._interval = sample_interval_s
        self._tolerance = tolerance_s
        self._maxlen = maxlen
        self._samples: deque[tuple[float, float]] = deque(maxlen=maxlen)

    # ------------------------------------------------------------------
    #  Ingest
    # ------------------------------------------------------------------

    def record(self, timestamp: float, pressure: float | None) -> None:
        """Store an hourly sample, ignoring sub-hourly and invalid values."""
        if pressure is None or not math.isfinite(pressure):
            return
        if self._samples and (
            timestamp - self._samples[-1][0] < self._interval * 0.9
        ):
            return
        self._samples.append((timestamp, pressure))
        self._prune(timestamp)

    def _prune(self, now_ts: float) -> None:
        while self._samples and (
            now_ts - self._samples[0][0] > self._window + self._interval
        ):
            self._samples.popleft()

    # ------------------------------------------------------------------
    #  Derived quantities
    # ------------------------------------------------------------------

    def tendency_per_hour(
        self, now_ts: float, current_pressure: float | None
    ) -> float | None:
        """WMO 3-hour tendency as hPa/h, or None during warmup.

        Uses the buffered sample nearest to 3 h ago and divides by the
        actual elapsed time so an off-by-a-few-minutes sample still yields
        an honest per-hour rate.
        """
        if current_pressure is None:
            return None
        ref = self._nearest(now_ts - 3 * 3600.0)
        if ref is None:
            return None
        dt_h = (now_ts - ref[0]) / 3600.0
        if dt_h <= 0:
            return None
        return (current_pressure - ref[1]) / dt_h

    def mean(
        self, now_ts: float, current_pressure: float | None = None
    ) -> float | None:
        """Rolling mean of samples in the 24 h window, or None if empty."""
        values = [p for ts, p in self._samples if now_ts - ts <= self._window]
        if current_pressure is not None and math.isfinite(current_pressure):
            values.append(current_pressure)
        if not values:
            return None
        return sum(values) / len(values)

    def _nearest(self, target_ts: float) -> tuple[float, float] | None:
        best: tuple[float, float] | None = None
        best_diff = float("inf")
        for ts, p in self._samples:
            diff = abs(ts - target_ts)
            if diff < best_diff:
                best_diff = diff
                best = (ts, p)
        if best is None or best_diff > self._tolerance:
            return None
        return best

    # ------------------------------------------------------------------
    #  Persistence (RestoreEntity)
    # ------------------------------------------------------------------

    def dump(self) -> list[list[float]]:
        """Serialise samples to a JSON-friendly list of [ts, pressure]."""
        return [[ts, p] for ts, p in self._samples]

    def load(self, samples: list) -> None:
        """Merge persisted samples with any live ones, thinning to hourly."""
        combined = list(self._samples)
        for item in samples:
            try:
                ts, p = float(item[0]), float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(ts) and math.isfinite(p):
                combined.append((ts, p))

        combined.sort(key=lambda s: s[0])
        thinned: list[tuple[float, float]] = []
        for ts, p in combined:
            if thinned and ts - thinned[-1][0] < self._interval * 0.5:
                continue
            thinned.append((ts, p))

        self._samples = deque(thinned, maxlen=self._maxlen)
        if self._samples:
            self._prune(self._samples[-1][0])
