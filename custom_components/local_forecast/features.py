"""Model inputs — the one definition shared by Home Assistant and training.

The forecast models were trained on exactly these numbers, computed from an
hourly history of the station's own channels.  The training harness
(``weather_lab/forecaster``) carries a vectorised twin of this file and a
parity test that holds the two together; change one, change both.

History convention: ``hist[ch][-1]`` is the hour ending at the issue time,
``hist[ch][-1 - k]`` the hour ``k`` hours earlier.  Missing hours are NaN.
A channel the station does not have is all NaN, and every feature built on it
is NaN — the models were trained with whole channels hidden, so that is a
normal input, not an error.

Pressure is QNH with the station's own learned tide removed (``tide``), so a
tendency here carries weather only.  Time of day is local solar time from
longitude, never the timezone.  Rain is measured rain: ``rain`` is mm in the
hour, ``wet`` is 1 from WET_MM up (the gauge-based label the rain models were
trained on).  Sunshine enters as the clear-sky index: measured radiation over
what a cloudless sky would give at that sun height, divided by the station's
own clear-sky reference (the 95th percentile of its daylight index over the
last 30 days), so a sensor that reads 25 % low - shading, calibration, a light
sensor standing in for a pyranometer - looks the same as a perfect one.

No Home Assistant dependencies — pure Python + math.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

NAN = float("nan")
HISTORY_HOURS = 720  # 30 days: the longest window any feature reads

CHANNELS = ("t", "td", "ws", "wd", "p", "wet", "rain", "solar")
WET_MM = 0.2  # a wet hour, and a wet day: at least this much measured rain
CLEAR_SKY_MIN_ELEV = 10.0  # below this the clear-sky index is too noisy to use
SUN_REF_QUANTILE = 0.95  # of the last 30 days' daylight index: "a clear hour here"
SUN_REF_MIN_SAMPLES = 60
SUN_REF_RANGE = (0.5, 1.5)

BASE = (
    # where and when
    "hsin", "hcos", "ssin", "scos", "alat", "elev", "sun", "noon", "daylen",
    # temperature
    "t0", "dt1", "dt3", "dt6", "dt12", "dt24", "tmax24", "tmin24", "tmean24",
    "tanom7", "tmax7", "tmin7", "tmean30",
    # moisture
    "td0", "dd0", "rh0", "dtd3", "dtd12", "dtd24", "ddmin24",
    # pressure (tide removed)
    "p0", "dp1", "dp3", "dp6", "dp12", "dp24", "dp3acc", "panom7", "pz30",
    # wind
    "ws0", "ws3", "ws24", "dws3", "wsin", "wcos",
    # precipitation
    "wet0", "wet3", "wet24", "since", "wetfrac30", "rain24",
    # sunshine (clear-sky index)
    "kc0", "kc3", "kc24", "kclast",
)  # fmt: skip
HOURLY = (*BASE, "lead", "thsin", "thcos", "tsun")
DAILY = (*BASE, "dofs", "hleft", "hspan", "tdaylen", "tnoon")

_J2000 = 946728000.0  # 2000-01-01 12:00 UTC


def solar(ts: float, lat: float, lon: float) -> tuple[float, float, float, float, float]:
    """(solar hour, sun elevation deg, noon elevation deg, day length h, season rad).

    Low-precision almanac (about 0.01 deg in declination), plenty for weather.
    Season is the sun's ecliptic longitude, shifted half a year south of the
    equator so the same value always means the same season.
    """
    d = (ts - _J2000) / 86400.0
    mean_lon = math.radians((280.459 + 0.98564736 * d) % 360.0)
    g = math.radians((357.529 + 0.98560028 * d) % 360.0)
    lam = mean_lon + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 3.6e-7 * d)
    decl = math.asin(math.sin(eps) * math.sin(lam))
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    eot_h = (((mean_lon - ra) + math.pi) % (2 * math.pi) - math.pi) * 12.0 / math.pi
    hour = ((ts / 3600.0) % 24.0 + lon / 15.0 + eot_h) % 24.0
    phi = math.radians(lat)
    ha = math.radians(15.0 * (hour - 12.0))
    sin_el = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(ha)
    elev = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
    noon = 90.0 - abs(lat - math.degrees(decl))
    x = max(-1.0, min(1.0, -math.tan(phi) * math.tan(decl)))
    daylen = 2.0 * math.degrees(math.acos(x)) / 15.0
    season = lam + (math.pi if lat < 0 else 0.0)
    return hour, elev, noon, daylen, season


def rh_from(t: float, td: float) -> float:
    """Relative humidity (%) from temperature and dew point (Magnus)."""
    return 100.0 * math.exp(17.625 * td / (243.04 + td) - 17.625 * t / (243.04 + t))


def dew_point(t: float, rh: float) -> float:
    """Dew point (C) from temperature and relative humidity (Magnus)."""
    rh = max(1.0, min(100.0, rh))
    g = math.log(rh / 100.0) + 17.625 * t / (243.04 + t)
    return 243.04 * g / (17.625 - g)


def clear_sky(elev_deg: float) -> float:
    """Global radiation under a cloudless sky, W/m^2 (Haurwitz 1945)."""
    if elev_deg <= 0:
        return 0.0
    c = math.sin(math.radians(elev_deg))
    return 1098.0 * c * math.exp(-0.057 / c)


def clear_index(radiation: float, elev_deg: float) -> float:
    """Measured over clear-sky radiation (0 overcast .. ~1 clear); NaN when the sun is low."""
    if radiation != radiation or elev_deg < CLEAR_SKY_MIN_ELEV:
        return NAN
    return max(0.0, min(1.5, radiation / clear_sky(elev_deg)))


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile (numpy's default method)."""
    v = sorted(values)
    x = q * (len(v) - 1)
    i = int(x)
    return v[i] if i + 1 >= len(v) else v[i] + (x - i) * (v[i + 1] - v[i])


def _ok(v: float) -> bool:
    return v == v  # not NaN


def _tail(x: Sequence[float], k: int, min_n: int) -> list[float] | None:
    vals = [v for v in x[-k:] if _ok(v)]
    return vals if len(vals) >= min_n else None


def _mean(x: Sequence[float], k: int, min_n: int) -> float:
    vals = _tail(x, k, min_n)
    return sum(vals) / len(vals) if vals else NAN


def _ago(x: Sequence[float], k: int) -> float:
    return x[-1 - k] if len(x) > k else NAN


def _daily_extreme(x: Sequence[float], fn: Callable[[list[float]], float]) -> float:
    """Mean over the last 7 days of each 24 h block's max (or min)."""
    blocks = []
    for k in range(7):
        end = len(x) - 24 * k
        vals = [v for v in x[max(0, end - 24) : end] if _ok(v)] if end > 0 else []
        if len(vals) >= 18:
            blocks.append(fn(vals))
    return sum(blocks) / len(blocks) if len(blocks) >= 3 else NAN


def base(
    hist: dict[str, Sequence[float]],
    ts: float,
    lat: float,
    lon: float,
    elev_m: float,
    tide: Callable[[float], float] | None = None,
) -> dict[str, float]:
    """Every BASE feature at issue time ``ts`` (the end of the last full hour)."""
    f: dict[str, float] = {}
    hour, sun, noon, daylen, season = solar(ts, lat, lon)
    f["hsin"], f["hcos"] = math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)
    f["ssin"], f["scos"] = math.sin(season), math.cos(season)
    f["alat"], f["elev"] = abs(lat), elev_m / 1000.0
    f["sun"], f["noon"], f["daylen"] = sun, noon, daylen

    t = hist["t"]
    t0 = _ago(t, 0)
    f["t0"] = t0
    for k in (1, 3, 6, 12, 24):
        f[f"dt{k}"] = t0 - _ago(t, k)
    last = _tail(t, 24, 18)
    f["tmax24"], f["tmin24"] = (max(last), min(last)) if last else (NAN, NAN)
    f["tmean24"] = _mean(t, 24, 18)
    f["tanom7"] = t0 - _mean(t, 168, 72)
    f["tmax7"], f["tmin7"] = _daily_extreme(t, max), _daily_extreme(t, min)
    f["tmean30"] = _mean(t, 720, 168)

    td = hist["td"]
    td0 = _ago(td, 0)
    f["td0"], f["dd0"] = td0, t0 - td0
    f["rh0"] = rh_from(t0, td0) if _ok(t0) and _ok(td0) else NAN
    for k in (3, 12, 24):
        f[f"dtd{k}"] = td0 - _ago(td, k)
    dd = [a - b for a, b in zip(t[-24:], td[-24:], strict=False)]
    near = _tail(dd, 24, 18)
    f["ddmin24"] = min(near) if near else NAN

    p = hist["p"]

    def detided(k: int) -> float:
        v = _ago(p, k)
        return v - tide(ts - 3600.0 * k) if tide is not None and _ok(v) else v

    p0, p3, p6 = detided(0), detided(3), detided(6)
    f["p0"] = p0
    for k in (1, 3, 6, 12, 24):
        f[f"dp{k}"] = p0 - detided(k)
    f["dp3acc"] = (p0 - p3) - (p3 - p6)
    f["panom7"] = p0 - _mean(p, 168, 72)
    month = _tail(p, 720, 168)
    if month:
        m = sum(month) / len(month)
        sd = math.sqrt(sum((v - m) ** 2 for v in month) / len(month))
        f["pz30"] = (p0 - m) / max(1.0, sd)
    else:
        f["pz30"] = NAN

    ws = hist["ws"]
    ws0 = _ago(ws, 0)
    f["ws0"] = ws0
    f["ws3"] = _mean(ws, 3, 2)
    f["ws24"] = _mean(ws, 24, 18)
    f["dws3"] = ws0 - _ago(ws, 3)
    wd = hist["wd"]
    u = [math.sin(math.radians(v)) for v in wd[-3:] if _ok(v)]
    w = [math.cos(math.radians(v)) for v in wd[-3:] if _ok(v)]
    if u:
        su, sw = sum(u), sum(w)
        norm = math.hypot(su, sw)
        f["wsin"], f["wcos"] = (su / norm, sw / norm) if norm > 1e-6 else (NAN, NAN)
    else:
        f["wsin"] = f["wcos"] = NAN

    wet = hist["wet"]
    f["wet0"] = _ago(wet, 0)
    f["wet3"] = _mean(wet, 3, 2)
    f["wet24"] = _mean(wet, 24, 18)
    since = NAN
    seen = 0
    for k in range(min(168, len(wet))):
        v = wet[-1 - k]
        if not _ok(v):
            continue
        seen += 1
        if v > 0:
            since = float(k)
            break
    else:
        if seen >= 72:
            since = 168.0
    f["since"] = since
    f["wetfrac30"] = _mean(wet, 720, 168)
    rain24 = _tail(hist["rain"], 24, 18)
    f["rain24"] = sum(rain24) if rain24 else NAN

    sol = hist["solar"]
    kc = [NAN] * min(HISTORY_HOURS, len(sol))
    for k in range(len(kc)):
        if _ok(sol[-1 - k]):
            kc[k] = clear_index(sol[-1 - k], solar(ts - 3600.0 * k, lat, lon)[1])
    seen = [v for v in kc if _ok(v)]
    ref = 1.0
    if len(seen) >= SUN_REF_MIN_SAMPLES:
        ref = max(SUN_REF_RANGE[0], min(SUN_REF_RANGE[1], quantile(seen, SUN_REF_QUANTILE)))
    day = [v / ref if _ok(v) else NAN for v in kc[:24]] + [NAN] * max(0, 24 - len(kc))
    f["kc0"] = day[0]
    f["kc3"] = _mean(day[2::-1], 3, 1)
    f["kc24"] = _mean(day[::-1], 24, 3)
    f["kclast"] = next((v for v in day[:18] if _ok(v)), NAN)
    return f


def hourly_row(f: dict[str, float], ts: float, lead: int, lat: float, lon: float) -> list[float]:
    """Inputs for an hourly model ``lead`` hours after issue time ``ts``."""
    hour, sun, _noon, _daylen, _season = solar(ts + 3600.0 * lead, lat, lon)
    extra = {
        "lead": float(lead),
        "thsin": math.sin(2 * math.pi * hour / 24),
        "thcos": math.cos(2 * math.pi * hour / 24),
        "tsun": sun,
    }
    return [f[k] if k in f else extra[k] for k in HOURLY]


def daily_row(
    f: dict[str, float], ts: float, offset: int, hours_to_midnight: float, lat: float, lon: float
) -> list[float]:
    """Inputs for a daily model: local day ``offset`` after ``ts``.

    Offset 0 is the rest of today (from now to midnight), 1 tomorrow, 2 the day
    after.  ``hours_to_midnight`` runs from the issue time to the next local
    midnight.
    """
    if offset == 0:
        hleft, hspan, middle = 0.0, hours_to_midnight, hours_to_midnight - 12.0
    else:
        hleft, hspan = hours_to_midnight + 24.0 * (offset - 1), 24.0
        middle = hleft + 12.0
    _hour, _sun, noon, daylen, _season = solar(ts + 3600.0 * middle, lat, lon)
    extra = {"dofs": float(offset), "hleft": hleft, "hspan": hspan, "tdaylen": daylen, "tnoon": noon}
    return [f[k] if k in f else extra[k] for k in DAILY]
