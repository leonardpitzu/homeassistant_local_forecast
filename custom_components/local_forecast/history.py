"""Hourly history of the station's own channels — what the models read.

Every reading is assigned to the nearest whole hour (a report at 10:20 counts
for 10:00, one at 10:40 for 11:00) and averaged there: the same rule the
training data was built with.  Wind direction is a vector mean; the wet flag
is 1 when any rain reading in the hour is above zero.

Thirty days are kept (features.HISTORY_HOURS) and persisted, so a restart
loses nothing.  A fresh install can be seeded from the recorder's long-term
statistics (hourly means), so the forecast starts with history, not blind.

No Home Assistant dependencies — pure Python + math.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

from .features import CHANNELS, HISTORY_HOURS, NAN, dew_point

_HOUR = 3600


def hour_of(ts: float) -> int:
    """The whole hour (epoch seconds) a reading at ``ts`` belongs to."""
    return int((ts + _HOUR / 2) // _HOUR) * _HOUR


def issue_hour(ts: float) -> int:
    """The last hour whose window (hour +/- 30 min) has fully closed by ``ts``."""
    return int((ts - _HOUR / 2) // _HOUR) * _HOUR


class _Bin:
    __slots__ = ("n", "s", "wet")

    def __init__(self) -> None:
        self.n: dict[str, int] = {}
        self.s: dict[str, float] = {}
        self.wet: float = NAN

    def add(self, key: str, value: float | None) -> None:
        if value is None or value != value:
            return
        self.n[key] = self.n.get(key, 0) + 1
        self.s[key] = self.s.get(key, 0.0) + value

    def mean(self, key: str) -> float:
        n = self.n.get(key, 0)
        return self.s[key] / n if n else NAN


class HourlyHistory:
    """Thirty days of hourly means per channel, newest last."""

    def __init__(self) -> None:
        self._bins: dict[int, _Bin] = {}

    def add(
        self,
        ts: float,
        *,
        temperature: float | None,
        humidity: float | None = None,
        wind_speed: float | None = None,
        wind_direction: float | None = None,
        pressure: float | None = None,
        rain_rate: float | None = None,
    ) -> None:
        """Fold one reading (canonical units, sea-level pressure) into its hour."""
        b = self._bins.setdefault(hour_of(ts), _Bin())
        b.add("t", temperature)
        if temperature is not None and humidity is not None:
            b.add("td", dew_point(temperature, humidity))
        b.add("ws", wind_speed)
        if wind_direction is not None and (wind_speed is None or wind_speed > 0):
            b.add("u", math.sin(math.radians(wind_direction)))
            b.add("v", math.cos(math.radians(wind_direction)))
        b.add("p", pressure)
        if rain_rate is not None:
            b.wet = max(0.0 if b.wet != b.wet else b.wet, 1.0 if rain_rate > 0 else 0.0)
        self._prune(ts)

    def _prune(self, now: float) -> None:
        oldest = hour_of(now) - (HISTORY_HOURS + 1) * _HOUR
        if self._bins and min(self._bins) < oldest:
            for h in [h for h in self._bins if h < oldest]:
                del self._bins[h]

    def has(self, hour: int) -> bool:
        return hour in self._bins

    def series(self, issue: int) -> dict[str, list[float]]:
        """Channels as lists ending at ``issue``; NaN where nothing was read."""
        hours = range(issue - (HISTORY_HOURS - 1) * _HOUR, issue + _HOUR, _HOUR)
        out: dict[str, list[float]] = {c: [] for c in CHANNELS}
        for h in hours:
            b = self._bins.get(h)
            if b is None:
                for c in CHANNELS:
                    out[c].append(NAN)
                continue
            out["t"].append(b.mean("t"))
            out["td"].append(b.mean("td"))
            out["ws"].append(b.mean("ws"))
            u, v = b.s.get("u"), b.s.get("v")
            out["wd"].append(math.degrees(math.atan2(u, v)) % 360.0 if u is not None and (u or v) else NAN)
            out["p"].append(b.mean("p"))
            out["wet"].append(b.wet)
        return out

    def observed(self, start: int, end: int) -> list[tuple[int, float]]:
        """(hour, temperature) for whole hours in [start, end)."""
        return [(h, b.mean("t")) for h, b in sorted(self._bins.items()) if start <= h < end and b.n.get("t")]

    def seed(self, rows: Iterable[tuple[int, dict[str, float | None]]]) -> int:
        """Fill hours we have never seen from recorder hourly means; returns count."""
        added = 0
        for hour, vals in rows:
            if hour in self._bins:
                continue
            b = _Bin()
            t, rh = vals.get("t"), vals.get("rh")
            b.add("t", t)
            if t is not None and rh is not None:
                b.add("td", dew_point(t, rh))
            b.add("ws", vals.get("ws"))
            b.add("p", vals.get("p"))
            rain = vals.get("rain")
            if rain is not None:
                b.wet = 1.0 if rain > 0 else 0.0
            if b.n:
                self._bins[hour] = b
                added += 1
        return added

    # ------------------------------------------------------------------
    #  Persistence: [hour, t, td, ws, u, v, p, wet] per hour, compact
    # ------------------------------------------------------------------

    def dump(self) -> list[list[float | None]]:
        def r(x: float, nd: int) -> float | None:
            return None if x != x else round(x, nd)

        out = []
        for h, b in sorted(self._bins.items()):
            u, v = b.mean("u"), b.mean("v")
            out.append([h, r(b.mean("t"), 2), r(b.mean("td"), 2), r(b.mean("ws"), 2), r(u, 3), r(v, 3),
                        r(b.mean("p"), 2), r(b.wet, 0)])  # fmt: skip
        return out

    def load(self, rows: Any) -> None:
        self._bins.clear()
        if not isinstance(rows, list):
            return
        for row in rows:
            try:
                h, t, td, ws, u, v, p, wet = row
                b = _Bin()
                for key, val in (("t", t), ("td", td), ("ws", ws), ("u", u), ("v", v), ("p", p)):
                    b.add(key, None if val is None else float(val))
                b.wet = NAN if wet is None else float(wet)
                self._bins[int(h)] = b
            except (TypeError, ValueError):
                continue
