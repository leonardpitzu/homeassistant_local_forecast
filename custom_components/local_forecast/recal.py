"""Self-correction: the station checks the models' forecasts against itself.

The models were trained on airports.  A backyard station sits lower or higher,
in a hollow or on a roof, in the sun at noon or in a cold pocket at night; its
rain sensor counts rain differently from an observer, and its anemometer is not
on a 10 m mast.  So every forecast the station can verify is verified, once the
hour (or day) it was for has been observed, and a small correction is learned
per forecast range:

  temperature, dew point   an additive bias per 3 hours of the (solar) day
  wind speed               a scale factor (0.2x .. 2x)
  rain chance              a shift in log-odds, only with a rain sensor
  daily high / low         an additive bias per day ahead
  rain day chance          a shift in log-odds per day ahead

Tuned on 24 stations the models never saw, on four continents, plus a
backyard station (forecaster harness, heldout.py recal): one replay per
station and setting, a fresh install every January 1st.  The rates below are
the ones that help on average and cost no station more than ~0.7 % (a faster
temperature rate gained 0.2 % more on average but cost one station 1.6 %).
Mean error change: temperature -5 to -3 % (backyard -10 to -4 %), dew point
-3 %, wind -3 to -4 % (backyard -80 %: its anemometer sees 40 % of an
airport's wind), rain chance -1 % (backyard -3 to -4 %).  A single
temperature bias, without the time of day, made 7-24 h forecasts worse.

Rates are the share of the gap closed per hour of verified forecasts in a
range.  Every correction is bounded, so a broken sensor cannot teach the
forecast nonsense, and starts neutral, so a new install runs the models as
trained.

No Home Assistant dependencies — pure Python + math.
"""

from __future__ import annotations

import math
from typing import Any

# Forecast ranges that share a correction (hours ahead, inclusive).
BUCKETS = ((1, 3), (4, 6), (7, 12), (13, 24))
DAY_PARTS = 8  # temperature bias per 3 h of solar time

SHIFT_RATE = 0.01  # per hour of verified forecasts
SCALE_RATE = 0.005
LOGIT_RATE = 0.02
DAILY_RATE = 0.03  # per verified day
MAX_SHIFT_C = 3.0
SCALE_RANGE = (0.2, 2.0)
MAX_LOGIT = 3.0


def bucket(lead: int) -> int:
    for i, (a, b) in enumerate(BUCKETS):
        if a <= lead <= b:
            return i
    return len(BUCKETS) - 1


def _per_update(rate: float, lead: int) -> float:
    a, b = BUCKETS[bucket(lead)]
    return rate / (b - a + 1)


def _part(solar_hour: float) -> int:
    return int((solar_hour % 24.0) // (24 / DAY_PARTS))


def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class Recalibrator:
    """Bounded online corrections, persisted with the history."""

    def __init__(self) -> None:
        n = len(BUCKETS)
        self.shift: dict[str, list[list[float]]] = {k: [[0.0] * DAY_PARTS for _ in range(n)] for k in ("t", "td")}
        self.scale: list[float] = [1.0] * n
        self.logit: list[float] = [0.0] * n
        self.day_shift: dict[str, list[float]] = {"tmax": [0.0, 0.0], "tmin": [0.0, 0.0]}
        self.day_logit: list[float] = [0.0, 0.0]
        self.verified = 0

    # -- apply -------------------------------------------------------------

    def hourly(self, target: str, lead: int, raw: float, solar_hour: float) -> float:
        """Corrected hourly forecast for an hour at ``solar_hour`` (``raw`` = the model's)."""
        if raw != raw:
            return raw
        i = bucket(lead)
        if target in self.shift:
            return raw + self.shift[target][i][_part(solar_hour)]
        if target == "ws":
            return raw * self.scale[i]
        if target == "wet":
            return _sigmoid(_logit(raw) + self.logit[i])
        return raw

    def daily(self, target: str, offset: int, raw: float) -> float:
        if raw != raw:
            return raw
        j = min(max(offset, 1), 2) - 1
        if target in self.day_shift:
            return raw + self.day_shift[target][j]
        if target == "wetday":
            return _sigmoid(_logit(raw) + self.day_logit[j])
        return raw

    # -- learn -------------------------------------------------------------

    def learn_hourly(self, target: str, lead: int, raw: float, observed: float, solar_hour: float) -> None:
        """One verified hourly forecast: what the model said, what happened."""
        if raw != raw or observed != observed:
            return
        i = bucket(lead)
        if target in self.shift:
            row = self.shift[target][i]
            k = _part(solar_hour)
            err = observed - (raw + row[k])
            row[k] = _clamp(row[k] + _per_update(SHIFT_RATE, lead) * err, -MAX_SHIFT_C, MAX_SHIFT_C)
        elif target == "ws":
            if raw > 0.5:  # calm hours say nothing about scale
                gap = observed / raw - self.scale[i]
                self.scale[i] = _clamp(self.scale[i] + _per_update(SCALE_RATE, lead) * gap, *SCALE_RANGE)
        elif target == "wet":
            p = _sigmoid(_logit(raw) + self.logit[i])
            step = _per_update(LOGIT_RATE, lead) * (observed - p)
            self.logit[i] = _clamp(self.logit[i] + step, -MAX_LOGIT, MAX_LOGIT)
        self.verified += 1

    def learn_daily(self, target: str, offset: int, raw: float, observed: float) -> None:
        if raw != raw or observed != observed:
            return
        j = min(max(offset, 1), 2) - 1
        if target in self.day_shift:
            err = observed - (raw + self.day_shift[target][j])
            self.day_shift[target][j] = _clamp(self.day_shift[target][j] + DAILY_RATE * err, -MAX_SHIFT_C, MAX_SHIFT_C)
        elif target == "wetday":
            p = _sigmoid(_logit(raw) + self.day_logit[j])
            self.day_logit[j] = _clamp(self.day_logit[j] + DAILY_RATE * (observed - p), -MAX_LOGIT, MAX_LOGIT)

    # -- persistence -------------------------------------------------------

    def dump(self) -> dict[str, Any]:
        return {
            "shift": self.shift,
            "scale": self.scale,
            "logit": self.logit,
            "day_shift": self.day_shift,
            "day_logit": self.day_logit,
            "verified": self.verified,
        }

    def load(self, data: Any) -> None:
        """Restore a saved state; anything malformed leaves the current one."""
        if not isinstance(data, dict):
            return
        n = len(BUCKETS)
        try:
            shift = {k: [[float(x) for x in row] for row in data["shift"][k]] for k in ("t", "td")}
            scale = [float(x) for x in data["scale"]]
            logit = [float(x) for x in data["logit"]]
            day_shift = {k: [float(x) for x in data["day_shift"][k]] for k in ("tmax", "tmin")}
            day_logit = [float(x) for x in data["day_logit"]]
        except (KeyError, TypeError, ValueError):
            return
        if any(len(rows) != n or any(len(r) != DAY_PARTS for r in rows) for rows in shift.values()):
            return
        if len(scale) != n or len(logit) != n or len(day_logit) != 2:
            return
        self.shift, self.scale, self.logit = shift, scale, logit
        self.day_shift, self.day_logit = day_shift, day_logit
        self.verified = int(data.get("verified", 0))
