"""Model inputs: sun geometry, windows, missing channels."""

from datetime import UTC, datetime
import math

import pytest

from local_forecast.features import (
    BASE,
    CHANNELS,
    DAILY,
    HOURLY,
    base,
    clear_index,
    clear_sky,
    daily_row,
    dew_point,
    hourly_row,
    rh_from,
    solar,
)

NAN = float("nan")


def _ts(*args: int) -> float:
    return datetime(*args, tzinfo=UTC).timestamp()


def test_equinox_noon_sun_stands_at_ninety_minus_latitude():
    _hour, _el, noon, daylen, _season = solar(_ts(2026, 3, 20, 12), 45.0, 0.0)
    assert noon == pytest.approx(45.0, abs=0.5)
    assert daylen == pytest.approx(12.0, abs=0.15)


def test_solar_noon_follows_longitude_not_the_timezone():
    # 25.6 E: the sun is highest ~1 h 42 min before 12:00 UTC.
    hour, *_ = solar(_ts(2026, 6, 1, 10, 18), 45.6, 25.6)
    assert hour == pytest.approx(12.0, abs=0.1)


def test_season_is_mirrored_south_of_the_equator():
    *_, north = solar(_ts(2026, 7, 1), 45.0, 0.0)
    *_, south = solar(_ts(2026, 7, 1), -45.0, 0.0)
    assert math.cos(north - south) == pytest.approx(-1.0, abs=1e-9)


def test_humidity_round_trip():
    assert rh_from(20.0, dew_point(20.0, 65.0)) == pytest.approx(65.0, abs=1e-6)


def _history(hours: int, **channels: list[float]) -> dict[str, list[float]]:
    out = {c: [NAN] * hours for c in CHANNELS}
    out.update(channels)
    return out


def test_minimal_station_gets_nan_not_errors():
    """Temperature and pressure only, one hour of history: every other input is NaN."""
    f = base(_history(1, t=[12.0], p=[1015.0]), _ts(2026, 5, 1, 12), 45.6, 25.6, 544)
    assert set(f) == set(BASE)
    assert f["t0"] == 12.0 and f["p0"] == 1015.0
    for k in ("td0", "rh0", "ws0", "wet0", "dt3", "tmean30", "pz30", "since", "rain24", "kc0", "kclast"):
        assert math.isnan(f[k]), k


def test_tendencies_use_the_tide_as_known_now():
    p = [1010.0] * 5
    f = base(_history(5, t=[10.0] * 5, p=p), 1.8e9, 45.0, 25.0, 0, tide=lambda ts: (ts - 1.8e9) / 3600.0)
    # Raw pressure is flat; the tide fell 3 hPa over 3 h, so the weather rose 3.
    assert f["p0"] == 1010.0
    assert f["dp3"] == pytest.approx(-3.0)


def test_rows_line_up_with_the_model_inputs():
    f = base(_history(48, t=[10.0] * 48, p=[1012.0] * 48), 1.8e9, 45.0, 25.0, 300)
    h = hourly_row(f, 1.8e9, 6, 45.0, 25.0)
    d = daily_row(f, 1.8e9, 2, 5.0, 45.0, 25.0)
    assert len(h) == len(HOURLY) and h[HOURLY.index("lead")] == 6.0
    assert len(d) == len(DAILY) and d[DAILY.index("hleft")] == 29.0 and d[DAILY.index("hspan")] == 24.0


def test_rest_of_today_window():
    """Offset 0 starts now and runs to midnight."""
    f = base(_history(24, t=[10.0] * 24, p=[1012.0] * 24), 1.8e9, 45.0, 25.0, 300)
    d = daily_row(f, 1.8e9, 0, 7.0, 45.0, 25.0)
    assert (d[DAILY.index("dofs")], d[DAILY.index("hleft")], d[DAILY.index("hspan")]) == (0.0, 0.0, 7.0)


def test_clear_sky_index():
    assert clear_sky(-1.0) == 0.0
    assert 850 < clear_sky(60.0) < 930  # Haurwitz: ~890 W/m^2 at 60 deg
    assert clear_index(clear_sky(40.0), 40.0) == pytest.approx(1.0)
    assert clear_index(0.25 * clear_sky(40.0), 40.0) == pytest.approx(0.25)
    assert math.isnan(clear_index(100.0, 5.0))  # sun too low to judge cloud by
    assert math.isnan(clear_index(math.nan, 40.0))


def test_sunshine_inputs_remember_the_day_after_dark():
    """At night the index is undefined, but the last daylight value carries on."""
    ts = _ts(2026, 6, 21, 21)  # after sunset at 45 N, 0 E
    n = 30
    sol = [0.0] * n
    for k in range(n):  # an overcast day: a quarter of the clear-sky radiation
        el = solar(ts - 3600.0 * (n - 1 - k), 45.0, 0.0)[1]
        sol[k] = 0.25 * clear_sky(el)
    f = base(_history(n, t=[15.0] * n, p=[1012.0] * n, solar=sol), ts, 45.0, 0.0, 0)
    assert math.isnan(f["kc0"])
    assert f["kclast"] == pytest.approx(0.25, abs=1e-6)
    assert f["kc24"] == pytest.approx(0.25, abs=1e-6)


def test_hours_since_rain():
    wet = [0.0] * 100
    wet[-8] = 1.0
    rain = [0.0] * 100
    rain[-8] = 2.5
    f = base(_history(100, t=[5.0] * 100, p=[1000.0] * 100, wet=wet, rain=rain), 1.8e9, 45.0, 25.0, 0)
    assert f["since"] == 7.0
    assert f["wet24"] == pytest.approx(1 / 24)
    assert f["rain24"] == pytest.approx(2.5)
