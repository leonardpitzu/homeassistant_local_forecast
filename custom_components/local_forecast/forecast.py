"""The forecast: trained models over the station's own hourly history.

Once per hour (an "issue") every model is evaluated on the features of the
last closed hour: 24 hourly leads and the next two calendar days.  Between
issues the published hours are re-anchored on the current time by linear
interpolation between leads, so entries stay at "now + k hours" as the cards
have always shown them.

What each model says, and how it reaches the card:
  temperature, dew point, pressure   a change from now (pressure with the
                                      station's tide removed, added back for
                                      the hour it is shown at)
  wind speed                          the value itself
  rain                                chance of a wet hour, and mm if wet
  cloud, fog, thunder                 cover 0..1 and chances, for the icon
  daily high / low, rain day          from their own daily models, not from
                                      aggregated hours

Every hourly and daily number the station can verify is corrected by the
bounded self-correction in recal.py.

No Home Assistant dependencies — pure Python + math.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math

from .const import RAIN_HEAVY, WET_BULB_MIX_UPPER, WET_BULB_SNOW, WIND_STRONG
from .features import base, daily_row, hourly_row, rh_from, solar
from .history import HourlyHistory
from .model import ModelSet
from .recal import Recalibrator
from .state_estimator import wet_bulb_stull

LEADS = 24
_HOUR = 3600


# Icon thresholds, set on out-of-fold forecasts for 230 stations the models
# never saw (weather_lab/forecaster/score.py).  Rain and cloud are frequency-
# matched: the rainy / cloudy icon shows as often as rainy / cloudy weather
# happens, so the icons are neither too wet nor too dry.  Fog and thunder
# cannot be placed in the right hour (at best ~30 % and ~12 % of such icons
# would be right), so they show only when the risk is high; a thunder DAY is
# predictable (AUC 0.86) and shows from a 30 % chance.
@dataclass(frozen=True)
class Thresholds:
    rain: float
    thunder: float
    fog: float
    cloudy: float
    partly: float


HOURLY_ICONS = Thresholds(rain=0.25, thunder=0.30, fog=0.30, cloudy=0.53, partly=0.45)
DAILY_ICONS = Thresholds(rain=0.40, thunder=0.30, fog=1.01, cloudy=0.62, partly=0.44)
HEAVY_DAY_MM = 25.0  # a wet day this heavy shows as pouring

# The estimator calls fog on saturated, calm air; at 230 stations that alone is
# fog 8 % of the time.  When the fog model gives the next hour under 5 %, the
# call is vetoed: two thirds of the false fog goes, three in four real fogs stay.
FOG_VETO = 0.05


@dataclass
class HourForecast:
    """One hour of forecast output."""

    hours_ahead: int
    condition: str  # HA condition string (drives the icon)
    temperature: float  # °C
    humidity: float | None  # %, None without a humidity sensor
    pressure: float  # hPa
    precipitation_probability: int  # 0-100
    precipitation_amount: float  # mm expected in this hour
    wind_speed: float | None  # m/s, None without a wind-speed sensor
    wind_bearing: float | None  # degrees, None without a wind-direction sensor
    is_daytime: bool = True  # day/night flag for icon variants


@dataclass
class DayForecast:
    """A whole local calendar day."""

    date: int  # proleptic ordinal of the local date
    condition: str
    temperature: float
    templow: float
    precipitation_probability: int
    precipitation_amount: float
    wind_speed: float | None


@dataclass
class Sensors:
    """Which optional channels the station really has."""

    humidity: bool = False
    wind: bool = False
    wind_direction: bool = False
    rain: bool = False


@dataclass
class Issue:
    """Everything the models said at one issue hour."""

    ts: int
    local_date: int  # ordinal of the local date the issue hour falls on
    t0: float
    td0: float
    p0: float  # tide removed
    leads: dict[str, list[float]] = field(default_factory=dict)  # index = lead (0 unused)
    raw: dict[str, list[float]] = field(default_factory=dict)  # before self-correction
    days: dict[int, dict[str, float]] = field(default_factory=dict)
    raw_days: dict[int, dict[str, float]] = field(default_factory=dict)
    verified_to: int = 0  # last lead already learned from
    solar_hours: list[float] = field(default_factory=list)  # local solar time of each lead


def condition(
    *,
    rain: float,
    thunder: float,
    fog: float,
    cloud: float,
    heavy: bool,
    temperature: float,
    humidity: float | None,
    wind: float | None,
    daytime: bool,
    icons: Thresholds = HOURLY_ICONS,
) -> str:
    """The icon for one hour (or one day) from the models' outputs."""
    if rain >= icons.rain:
        if thunder >= icons.thunder:
            return "lightning-rainy"
        wb = wet_bulb_stull(temperature, humidity if humidity is not None else 90.0)
        if wb <= WET_BULB_SNOW:
            return "snowy"
        if wb <= WET_BULB_MIX_UPPER:
            return "snowy-rainy"
        return "pouring" if heavy else "rainy"
    if fog >= icons.fog:
        return "fog"
    if wind is not None and wind >= WIND_STRONG:
        return "windy"
    if cloud >= icons.cloudy:
        return "cloudy"
    if cloud >= icons.partly:
        return "partlycloudy"
    return "sunny" if daytime else "clear-night"


class Forecaster:
    """Trained-model forecaster for one station."""

    def __init__(self, models: ModelSet, lat: float, lon: float, elev_m: float) -> None:
        self.models = models
        self.lat, self.lon, self.elev = lat, lon, elev_m
        self.recal = Recalibrator()
        self.issue: Issue | None = None
        self._pending: list[Issue] = []  # issues waiting to be verified
        # Latest forecast made for each (local date, days ahead): verified once
        # that date is over.  Not persisted; a restart only skips a day.
        self._day_snaps: dict[tuple[int, int], tuple[float, dict[str, float]]] = {}

    # ------------------------------------------------------------------
    #  Issue: evaluate every model once per hour
    # ------------------------------------------------------------------

    def run_issue(
        self,
        history: HourlyHistory,
        ts: int,
        tide: Callable[[float], float],
        hours_to_midnight: float,
        local_date: int,
        sensors: Sensors,
    ) -> Issue | None:
        """Evaluate the models at issue hour ``ts`` (blocking; a few hundred ms)."""
        series = history.series(ts)
        f = base(series, ts, self.lat, self.lon, self.elev, tide)
        if f["t0"] != f["t0"] or f["p0"] != f["p0"]:
            return None
        hourly = self.models.hourly
        wanted = ["t", "p", "wet", "rain", "cloud", "fog", "ts"]
        if sensors.humidity:
            wanted.append("td")
        if sensors.wind:
            wanted.append("ws")
        issue = Issue(ts=ts, local_date=local_date, t0=f["t0"], td0=f["td0"], p0=f["p0"])
        issue.solar_hours = [solar(ts + lead * _HOUR, self.lat, self.lon)[0] for lead in range(LEADS + 1)]
        for k in wanted:
            issue.raw[k] = [math.nan] * (LEADS + 1)
        for lead in range(1, LEADS + 1):
            row = hourly_row(f, ts, lead, self.lat, self.lon)
            for k in wanted:
                issue.raw[k][lead] = hourly[k].predict(row)
        sh = issue.solar_hours
        issue.leads = {
            k: [self.recal.hourly(k, lead, v, sh[lead]) for lead, v in enumerate(vals)] for k, vals in issue.raw.items()
        }
        daily = self.models.daily
        # Day 3 covers the half hour after midnight, when the newest closed hour
        # still belongs to yesterday and "the day after tomorrow" is 3 days out.
        for offset in (1, 2, 3):
            row = daily_row(f, ts, offset, hours_to_midnight, self.lat, self.lon)
            raw = {k: m.predict(row) for k, m in daily.items() if k != "wsday" or sensors.wind}
            issue.raw_days[offset] = raw
            issue.days[offset] = {k: self.recal.daily(k, offset, v) for k, v in raw.items()}
            if offset <= 2:
                self._day_snaps[(local_date + offset, offset)] = (issue.t0, raw)
        self._day_snaps = {k: v for k, v in self._day_snaps.items() if k[0] >= local_date - 1}
        self.issue = issue
        self._pending.append(issue)
        self._pending = [i for i in self._pending if ts - i.ts <= LEADS * _HOUR]
        return issue

    def fog_unlikely(self) -> bool:
        """Whether the fog model vetoes a fog call for the current hour."""
        issue = self.issue
        return issue is not None and "fog" in issue.leads and issue.leads["fog"][1] < FOG_VETO

    # ------------------------------------------------------------------
    #  Self-correction: verify what has been observed since
    # ------------------------------------------------------------------

    def verify(self, history: HourlyHistory, now_issue: int, sensors: Sensors) -> None:
        """Learn from every pending forecast whose hour has now been observed."""
        series = history.series(now_issue)
        n = len(series["t"])

        def obs(ch: str, hour: int) -> float:
            k = (now_issue - hour) // _HOUR
            return series[ch][n - 1 - k] if 0 <= k < n else math.nan

        for issue in self._pending:
            last = min(LEADS, (now_issue - issue.ts) // _HOUR)
            for lead in range(issue.verified_to + 1, last + 1):
                hour = issue.ts + lead * _HOUR
                sh = issue.solar_hours[lead]
                learn = self.recal.learn_hourly
                learn("t", lead, issue.t0 + issue.raw["t"][lead], obs("t", hour), sh)
                if "td" in issue.raw and issue.td0 == issue.td0:
                    learn("td", lead, issue.td0 + issue.raw["td"][lead], obs("td", hour), sh)
                if "ws" in issue.raw:
                    learn("ws", lead, issue.raw["ws"][lead], obs("ws", hour), sh)
                if sensors.rain:
                    learn("wet", lead, issue.raw["wet"][lead], obs("wet", hour), sh)
            issue.verified_to = max(issue.verified_to, last)

    def verify_day(self, local_date: int, high: float, low: float, wet: float | None) -> None:
        """Learn from a finished calendar day, once per forecast range."""
        for offset in (1, 2):
            snap = self._day_snaps.pop((local_date, offset), None)
            if snap is None:
                continue
            t0, raw = snap
            self.recal.learn_daily("tmax", offset, t0 + raw["tmax"], high)
            self.recal.learn_daily("tmin", offset, t0 + raw["tmin"], low)
            if wet is not None:
                self.recal.learn_daily("wetday", offset, raw["wetday"], wet)

    # ------------------------------------------------------------------
    #  Publish: hours re-anchored on "now", days as forecast
    # ------------------------------------------------------------------

    def hours(
        self,
        now: float,
        tide: Callable[[float], float],
        sensors: Sensors,
        wind_bearing: float | None,
        hours: int = LEADS,
    ) -> list[HourForecast]:
        issue = self.issue
        if issue is None:
            return []
        out: list[HourForecast] = []
        for k in range(1, hours + 1):
            target = now + k * _HOUR
            x = max(1.0, min(float(LEADS), (target - issue.ts) / _HOUR))
            lo = int(x)
            hi = min(LEADS, lo + 1)
            w = x - lo

            def at(key: str, _lo: int = lo, _hi: int = hi, _w: float = w) -> float:
                vals = issue.leads[key]
                return vals[_lo] + _w * (vals[_hi] - vals[_lo])

            temp = issue.t0 + at("t")
            humidity = None
            if "td" in issue.leads and issue.td0 == issue.td0:
                td = min(temp, issue.td0 + at("td"))
                humidity = max(1.0, min(100.0, rh_from(temp, td)))
            wind = max(0.0, at("ws")) if "ws" in issue.leads else None
            rain = min(1.0, max(0.0, at("wet")))
            mm_if_wet = max(0.0, at("rain"))
            _h, sun, *_ = solar(target, self.lat, self.lon)
            daytime = sun > -0.833
            out.append(
                HourForecast(
                    hours_ahead=k,
                    condition=condition(
                        rain=rain,
                        thunder=at("ts"),
                        fog=at("fog"),
                        cloud=at("cloud"),
                        heavy=mm_if_wet >= RAIN_HEAVY,
                        temperature=temp,
                        humidity=humidity,
                        wind=wind,
                        daytime=daytime,
                    ),
                    temperature=round(temp, 1),
                    humidity=None if humidity is None else round(humidity),
                    pressure=round(issue.p0 + at("p") + tide(target), 1),
                    precipitation_probability=round(100 * rain),
                    precipitation_amount=round(rain * mm_if_wet, 1),
                    wind_speed=None if wind is None else round(wind, 1),
                    wind_bearing=wind_bearing if sensors.wind_direction else None,
                    is_daytime=daytime,
                )
            )
        return out

    def days(self, sensors: Sensors, humidity_now: float | None, today: int) -> list[DayForecast]:
        """Tomorrow and the day after, as calendar dates (``today`` is an ordinal)."""
        issue = self.issue
        if issue is None:
            return []
        out = []
        for date in (today + 1, today + 2):
            d = issue.days.get(date - issue.local_date)
            if d is None:
                continue
            high, low = issue.t0 + d["tmax"], issue.t0 + d["tmin"]
            if low > high:
                high = low = (high + low) / 2
            rain = min(1.0, max(0.0, d["wetday"]))
            wind = max(0.0, d["wsday"]) if "wsday" in d else None
            out.append(
                DayForecast(
                    date=date,
                    condition=condition(
                        rain=rain,
                        thunder=d["tsday"],
                        fog=0.0,
                        cloud=d["cloudday"],
                        heavy=d["rainday"] >= HEAVY_DAY_MM,
                        temperature=(high + low) / 2,
                        humidity=humidity_now if sensors.humidity else None,
                        wind=wind,
                        daytime=True,
                        icons=DAILY_ICONS,
                    ),
                    temperature=round(high, 1),
                    templow=round(low, 1),
                    precipitation_probability=round(100 * rain),
                    precipitation_amount=round(rain * max(0.0, d["rainday"]), 1),
                    wind_speed=None if wind is None else round(wind, 1),
                )
            )
        return out

    # ------------------------------------------------------------------
    #  Persistence of the self-correction
    # ------------------------------------------------------------------

    def dump(self) -> dict:
        return self.recal.dump()

    def load(self, data: object) -> None:
        self.recal.load(data)
