"""Hourly history of the station's own channels — what the models read.

Every reading is assigned to the nearest whole hour (a report at 10:20 counts
for 10:00, one at 10:40 for 11:00) and averaged there: the same rule the
training data was built with.  Wind direction is a vector mean.  Rain is the
mean rain rate over the hour, i.e. mm in that hour, and the hour is wet from
features.WET_MM up - the gauge-based label the rain models learned.  Solar
radiation is the hour's mean in W/m^2.  Each hour also keeps its highest and
lowest raw temperature reading: the models work on hourly means, but the card's
"today" shows the real measured peak.

Thirty days are kept (features.HISTORY_HOURS) and persisted, so a restart
loses nothing.  The recorder's long-term statistics (hourly mean / max / min)
fill whatever a channel is missing - a fresh install starts with weeks of
history, and an upgrade gets the channels its saved history lacked.

No Home Assistant dependencies — pure Python + math.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

from .features import CHANNELS, HISTORY_HOURS, NAN, WET_MM, dew_point

_HOUR = 3600


def hour_of(ts: float) -> int:
    """The whole hour (epoch seconds) a reading at ``ts`` belongs to."""
    return int((ts + _HOUR / 2) // _HOUR) * _HOUR


def issue_hour(ts: float) -> int:
    """The last hour whose window (hour +/- 30 min) has fully closed by ``ts``."""
    return int((ts - _HOUR / 2) // _HOUR) * _HOUR


class _Bin:
    __slots__ = ("hi", "lo", "n", "s")

    def __init__(self) -> None:
        self.n: dict[str, int] = {}
        self.s: dict[str, float] = {}
        self.hi = NAN  # highest raw temperature reading in the hour
        self.lo = NAN

    def extremes(self, hi: float | None, lo: float | None) -> None:
        if hi is not None and hi == hi:
            self.hi = hi if self.hi != self.hi else max(self.hi, hi)
        if lo is not None and lo == lo:
            self.lo = lo if self.lo != self.lo else min(self.lo, lo)

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
        solar: float | None = None,
    ) -> None:
        """Fold one reading (canonical units, sea-level pressure) into its hour."""
        b = self._bins.setdefault(hour_of(ts), _Bin())
        b.add("t", temperature)
        b.extremes(temperature, temperature)
        if temperature is not None and humidity is not None:
            b.add("td", dew_point(temperature, humidity))
        b.add("ws", wind_speed)
        if wind_direction is not None and (wind_speed is None or wind_speed > 0):
            b.add("u", math.sin(math.radians(wind_direction)))
            b.add("v", math.cos(math.radians(wind_direction)))
        b.add("p", pressure)
        b.add("rain", None if rain_rate is None else max(0.0, rain_rate))
        b.add("solar", None if solar is None else max(0.0, solar))
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
            rain = b.mean("rain")
            out["rain"].append(rain)
            out["wet"].append(NAN if rain != rain else (1.0 if rain >= WET_MM else 0.0))
            out["solar"].append(b.mean("solar"))
        return out

    def peaks(self, start: int, end: int) -> tuple[float, float] | None:
        """Highest and lowest raw temperature read in hours [start, end)."""
        his, los = [], []
        for h, b in self._bins.items():
            if start <= h < end and b.n.get("t"):
                his.append(b.hi if b.hi == b.hi else b.mean("t"))
                los.append(b.lo if b.lo == b.lo else b.mean("t"))
        return (max(his), min(los)) if his else None

    def observed(self, start: int, end: int) -> list[tuple[int, float, float]]:
        """(hour, temperature, rain mm) for whole hours in [start, end) with a temperature."""
        return [
            (h, b.mean("t"), b.mean("rain")) for h, b in sorted(self._bins.items()) if start <= h < end and b.n.get("t")
        ]

    def seed(self, rows: Iterable[tuple[int, dict[str, float | None]]]) -> int:
        """Fill, from recorder hourly statistics, every channel an hour lacks.

        Channels already read live are never touched, so seeding twice (every
        restart) changes nothing.  Returns how many hours gained something.
        """
        added = 0
        for hour, vals in rows:
            b = self._bins.get(hour) or _Bin()
            before = sum(b.n.values())
            t, rh = vals.get("t"), vals.get("rh")
            fill = {
                "t": t,
                "ws": vals.get("ws"),
                "p": vals.get("p"),
                "rain": vals.get("rain"),
                "solar": vals.get("solar"),
            }
            if t is not None and rh is not None:
                fill["td"] = dew_point(t, rh)
            for key, value in fill.items():
                if not b.n.get(key):
                    b.add(key, value)
            if b.hi != b.hi:
                b.extremes(vals.get("tmax", t), vals.get("tmin", t))
            if sum(b.n.values()) > before:
                self._bins[hour] = b
                added += 1
        return added

    # ------------------------------------------------------------------
    #  Persistence: [hour, t, td, ws, u, v, p, rain, solar, t_high, t_low]
    # ------------------------------------------------------------------

    def dump(self) -> list[list[float | None]]:
        def r(x: float, nd: int) -> float | None:
            return None if x != x else round(x, nd)

        return [
            [
                h,
                r(b.mean("t"), 2),
                r(b.mean("td"), 2),
                r(b.mean("ws"), 2),
                r(b.mean("u"), 3),
                r(b.mean("v"), 3),
                r(b.mean("p"), 2),
                r(b.mean("rain"), 2),
                r(b.mean("solar"), 1),
                r(b.hi, 2),
                r(b.lo, 2),
            ]  # fmt: skip
            for h, b in sorted(self._bins.items())
        ]

    def load(self, rows: Any) -> None:
        self._bins.clear()
        if not isinstance(rows, list):
            return
        keys = ("t", "td", "ws", "u", "v", "p", "rain", "solar")
        for row in rows:
            try:
                if len(row) == 8:  # 2.0.0 kept a wet flag, not mm: the rain is unknown
                    row = [*row[:7], None, None, None, None]
                h, *vals, hi, lo = row
                b = _Bin()
                for key, val in zip(keys, vals, strict=True):
                    b.add(key, None if val is None else float(val))
                b.extremes(None if hi is None else float(hi), None if lo is None else float(lo))
                self._bins[int(h)] = b
            except (TypeError, ValueError):
                continue
