"""Atmospheric tide - the daily pressure waves that carry no weather.

Surface pressure rises and falls every day for reasons that are not weather:

  S1 (24 h)  surface heating.  Local and seasonal: 0.15 hPa in December and
             ~1 hPa in summer at a Brasov station, peaking before sunrise.
  S2 (12 h)  the global solar tide, ~1.16 cos^3(lat) hPa, peaking ~10:00.
  S3 (8 h)   the shape long winter nights give the daily cycle.

Left in, their slopes read as a pressure tendency.  Nothing here is tuned to a
place; the model learns the station it runs on:

  - S1 is the station's own daily temperature cycle times a site gain.  The
    temperature cycle is refitted with a 3-day memory, so season, day length,
    cloud and climate arrive through the thermometer; the gain (topography:
    basin, plain, coast) has a 14-day memory, as do S2 and S3.
  - Fits run on 3-hour changes, which weather does not bias toward any hour,
    and shrink toward a world-average prior, so a new install or a move starts
    sensible and adapts on its own.
  - Whatever shape that leaves (a sea breeze, a valley wind, S4) is caught in
    24 solar-hour bins of the residual against a centred 24 h mean.  They start
    at zero, so the model begins as the physics alone.

Scored causally at ten sites on four continents (2019-2023: coast, plain,
alpine valley, high plains, desert, both hemispheres) and on 6.2 years at
Brasov, the daily cycle left in the 3 h tendency averages 0.215 hPa/h under a
fixed climatology, 0.049 with the physics alone and 0.033 with the bins; it
improves at every site.

Time is local solar time from longitude, never the timezone.  Pure Python.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import math
from typing import Any

_W = 2.0 * math.pi

# S2 prior: Haurwitz's cos^3 law at the observed 10.0 h phase (not the 9.5 h
# often quoted).
S2_AMP_EQUATOR_HPA = 1.16
S2_AMP_DEFAULT_HPA = 0.41  # ~45 deg, when the latitude is unknown
S2_PHASE_H = 10.0

# S1 prior: pressure trough 1.5 h after the temperature peak, 0.05 hPa per K of
# daily temperature amplitude - the zonal-mean S1 of a typical land cycle, not
# any one site's value.
PRIOR_GAIN_HPA_PER_K = 0.05
PRIOR_LAG_H = 1.5

TEMPERATURE_MEMORY_DAYS = 3.0
PRESSURE_MEMORY_DAYS = 14.0
# What the prior is worth, in days of hourly samples.
TEMPERATURE_PRIOR_DAYS = 0.1
PRESSURE_PRIOR_DAYS = 7.0

PAIR_SPAN_S = 3 * 3600.0
PAIR_TOLERANCE_S = 1800.0
# A 3 h change beyond this is a sensor fault, not anything a tide explains.
MAX_PRESSURE_CHANGE_HPA = 15.0

# Residual bins: one update per bin per day, so 0.05 is a ~20-day memory.
RESIDUAL_ALPHA = 0.05
RESIDUAL_WINDOW_S = 24 * 3600.0
# The centred mean needs a whole day with no hole in it.
RESIDUAL_MAX_GAP_S = 5400.0
# A residual this large is a front passing through the window, not a tide.
MAX_RESIDUAL_HPA = 4.0
# Samples kept: the residual window plus the hour the 3 h pairing may reach past it.
_KEEP_S = RESIDUAL_WINDOW_S + 3600.0


def s2_amplitude(latitude_deg: float | None) -> float:
    """Semidiurnal amplitude in hPa: A = 1.16 hPa * cos^3(lat)."""
    if latitude_deg is None:
        return S2_AMP_DEFAULT_HPA
    c = math.cos(math.radians(latitude_deg))
    return S2_AMP_EQUATOR_HPA * c * c * c


def solar_hour(timestamp: float, longitude_deg: float | None) -> float:
    """Local mean solar time in hours, from an epoch timestamp.

    Deliberately derived from UTC and longitude rather than the configured
    timezone: the tide is phased to the sun, and a timezone can sit over an hour
    away from it (further still under DST).
    """
    utc_hour = (timestamp / 3600.0) % 24.0
    if longitude_deg is None:
        return utc_hour
    return (utc_hour + longitude_deg / 15.0) % 24.0


def _harmonics(timestamp: float, longitude_deg: float | None) -> tuple[float, ...]:
    """cos/sin of the 24, 12 and 8 h waves at the solar time of ``timestamp``."""
    h = solar_hour(timestamp, longitude_deg)
    a1, a2, a3 = _W * h / 24.0, _W * h / 12.0, _W * h / 8.0
    return (math.cos(a1), math.sin(a1), math.cos(a2), math.sin(a2), math.cos(a3), math.sin(a3))


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gauss-Jordan; the prior on the diagonal keeps the system positive definite."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(n):
        pivot = m[i][i]
        m[i] = [v / pivot for v in m[i]]
        for k in range(n):
            if k != i and (f := m[k][i]):
                m[k] = [x - f * y for x, y in zip(m[k], m[i], strict=True)]
    return [m[i][n] for i in range(n)]


def _wave(c: float, s: float, period_h: float) -> dict[str, float]:
    """Amplitude and peak time of ``c cos + s sin`` over one period."""
    peak = (math.atan2(s, c) / _W * period_h) % period_h
    return {"amplitude": round(math.hypot(c, s), 3), "peak_solar_hour": round(peak, 2)}


class _Fit:
    """Least squares with exponential forgetting in time, shrunk toward a prior."""

    def __init__(self, prior: Sequence[float], memory_days: float, prior_days: float) -> None:
        n = len(prior)
        self._prior = list(prior)
        self._memory_s = memory_days * 86400.0
        self._prior_weight = prior_days * 12.0  # hourly samples of unit-scale regressors
        self._a = [[0.0] * n for _ in range(n)]
        self._b = [0.0] * n
        self._last: float | None = None
        self.evidence = 0.0  # samples still remembered
        self.coef = list(self._prior)

    def learn(self, timestamp: float, x: Sequence[float], y: float) -> None:
        keep = 1.0 if self._last is None else math.exp(-max(0.0, timestamp - self._last) / self._memory_s)
        self._last = timestamp
        for i, xi in enumerate(x):
            self._b[i] = keep * self._b[i] + xi * y
            row = self._a[i]
            for j, xj in enumerate(x):
                row[j] = keep * row[j] + xi * xj
        self.evidence = keep * self.evidence + 1.0
        self._refresh()

    def _refresh(self) -> None:
        w = self._prior_weight
        a = [[v + (w if i == j else 0.0) for j, v in enumerate(row)] for i, row in enumerate(self._a)]
        self.coef = _solve(a, [bi + w * p for bi, p in zip(self._b, self._prior, strict=True)])

    def dump(self) -> dict[str, Any]:
        return {"a": self._a, "b": self._b, "last": self._last, "evidence": self.evidence}

    def load(self, data: Any) -> None:
        try:
            a = [[float(v) for v in row] for row in data["a"]]
            b = [float(v) for v in data["b"]]
            last = None if data["last"] is None else float(data["last"])
            evidence = float(data["evidence"])
        except (KeyError, TypeError, ValueError):
            return
        n = len(self._prior)
        values = [*b, *(v for row in a for v in row), evidence, 0.0 if last is None else last]
        if len(b) != n or len(a) != n or any(len(row) != n for row in a) or not all(map(math.isfinite, values)):
            return
        self._a, self._b, self._last, self.evidence = a, b, last, evidence
        self._refresh()


class LocalTide:
    """S1 + S2 + S3 at one station, learned from its own pressure and temperature."""

    def __init__(self, latitude_deg: float | None, longitude_deg: float | None) -> None:
        self._longitude = longitude_deg
        s2 = s2_amplitude(latitude_deg)
        lag = _W * PRIOR_LAG_H / 24.0
        s2_phase = _W * S2_PHASE_H / 12.0
        # The temperature cycle as S1 + S2; only its S1 drives pressure.
        self._temperature = _Fit((0.0, 0.0, 0.0, 0.0), TEMPERATURE_MEMORY_DAYS, TEMPERATURE_PRIOR_DAYS)
        # Site gain on that S1 (in phase, quadrature), then S2 and S3.
        self._pressure = _Fit(
            (
                -PRIOR_GAIN_HPA_PER_K * math.cos(lag),
                -PRIOR_GAIN_HPA_PER_K * math.sin(lag),
                s2 * math.cos(s2_phase),
                s2 * math.sin(s2_phase),
                0.0,
                0.0,
            ),
            PRESSURE_MEMORY_DAYS,
            PRESSURE_PRIOR_DAYS,
        )
        self._residual = [0.0] * 24  # hPa at each whole solar hour
        self._residual_last: float | None = None  # centre of the last residual folded
        # Hourly samples (~55 min apart), pruned by age rather than count.
        self._recent: deque[tuple[float, float, float]] = deque(maxlen=64)

    def _features(self, timestamp: float) -> tuple[float, ...]:
        c1, s1, c2, s2, c3, s3 = _harmonics(timestamp, self._longitude)
        a, b = self._temperature.coef[0], self._temperature.coef[1]
        return (a * c1 + b * s1, a * s1 - b * c1, c2, s2, c3, s3)

    def _physics_hpa(self, timestamp: float) -> float:
        return sum(c * f for c, f in zip(self._pressure.coef, self._features(timestamp), strict=True))

    def _residual_hpa(self, timestamp: float) -> float:
        h = solar_hour(timestamp, self._longitude)
        i = int(h)
        f = h - i
        return (1.0 - f) * self._residual[i % 24] + f * self._residual[(i + 1) % 24]

    def hpa_at(self, timestamp: float) -> float:
        """Tide in hPa at an epoch timestamp, ready to subtract."""
        return self._physics_hpa(timestamp) + self._residual_hpa(timestamp)

    def learn(self, timestamp: float, pressure_hpa: float, temperature_c: float) -> None:
        """Learn from an hourly sample, paired with the one ~3 h before it."""
        sample = (timestamp, pressure_hpa, temperature_c)
        target = timestamp - PAIR_SPAN_S
        partner = min(self._recent, key=lambda r: abs(r[0] - target), default=None)
        self._recent.append(sample)
        while timestamp - self._recent[0][0] > _KEEP_S:
            self._recent.popleft()
        if partner is not None and abs(partner[0] - target) <= PAIR_TOLERANCE_S:
            self._learn_change(sample, partner)
        self._learn_residual(timestamp)

    def _learn_change(self, sample: tuple[float, float, float], partner: tuple[float, float, float]) -> None:
        """Fit the physics on one 3 h change."""
        (now, pressure, temperature), (then, pressure_then, temperature_then) = sample, partner
        change = pressure - pressure_then
        if abs(change) > MAX_PRESSURE_CHANGE_HPA:
            return
        # Pressure first, on the temperature cycle as known before this hour.
        x = [a - b for a, b in zip(self._features(now), self._features(then), strict=True)]
        self._pressure.learn(now, x, change)
        now_h, then_h = _harmonics(now, self._longitude), _harmonics(then, self._longitude)
        x_t = [a - b for a, b in zip(now_h[:4], then_h[:4], strict=True)]
        self._temperature.learn(now, x_t, temperature - temperature_then)

    def _learn_residual(self, now: float) -> None:
        """Fold the sample 12 h back against the mean of the day centred on it."""
        day = [r for r in self._recent if now - r[0] <= RESIDUAL_WINDOW_S]
        times = [r[0] for r in day]
        if times[-1] - times[0] < RESIDUAL_WINDOW_S - RESIDUAL_MAX_GAP_S or any(
            b - a > RESIDUAL_MAX_GAP_S for a, b in zip(times, times[1:], strict=False)
        ):
            return
        centre, pressure, _ = min(day, key=lambda r: abs(r[0] - (now - RESIDUAL_WINDOW_S / 2)))
        if abs(centre - (now - RESIDUAL_WINDOW_S / 2)) > PAIR_TOLERANCE_S:
            return
        if self._residual_last is not None and centre <= self._residual_last:
            return
        self._residual_last = centre
        r = pressure - sum(p for _, p, _ in day) / len(day) - self._physics_hpa(centre)
        if abs(r) > MAX_RESIDUAL_HPA:
            return
        i = round(solar_hour(centre, self._longitude)) % 24
        self._residual[i] += RESIDUAL_ALPHA * (r - self._residual[i])

    def describe(self, timestamp: float) -> dict[str, Any]:
        """What has been learned, for diagnostics."""
        a, b = self._temperature.coef[0], self._temperature.coef[1]
        g1, g2, c2, s2, c3, s3 = self._pressure.coef
        return {
            "tide_now_hpa": round(self.hpa_at(timestamp), 3),
            "s1_hpa": _wave(g1 * a - g2 * b, g1 * b + g2 * a, 24.0),
            "s2_hpa": _wave(c2, s2, 12.0),
            "s3_hpa": _wave(c3, s3, 8.0),
            "temperature_cycle_k": _wave(a, b, 24.0),
            "site_gain_hpa_per_k": round(math.hypot(g1, g2), 4),
            "trough_after_temperature_peak_h": round((math.atan2(-g2, -g1) / _W * 24.0) % 24.0, 2),
            "learned_days": round(self._pressure.evidence / 24.0, 1),
            "residual_by_solar_hour_hpa": [round(v, 3) for v in self._residual],
        }

    def dump(self) -> dict[str, Any]:
        return {
            "temperature": self._temperature.dump(),
            "pressure": self._pressure.dump(),
            "residual": list(self._residual),
            "residual_last": self._residual_last,
            "recent": [list(r) for r in self._recent],
        }

    def load(self, data: Any) -> None:
        """Restore learned state; anything malformed leaves the prior in place."""
        if not isinstance(data, dict):
            return
        self._temperature.load(data.get("temperature"))
        self._pressure.load(data.get("pressure"))
        try:
            residual = [float(v) for v in data["residual"]]
            last = data["residual_last"]
            last = None if last is None else float(last)
        except (KeyError, TypeError, ValueError):
            pass
        else:
            if len(residual) == 24 and all(map(math.isfinite, residual)) and (last is None or math.isfinite(last)):
                self._residual, self._residual_last = residual, last
        for item in data.get("recent") or []:
            try:
                ts, p, t = (float(v) for v in item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(ts) and math.isfinite(p) and math.isfinite(t):
                self._recent.append((ts, p, t))
