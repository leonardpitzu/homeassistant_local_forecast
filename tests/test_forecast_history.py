"""Hourly history: binning, persistence, seeding."""

import math

import pytest

from local_forecast.features import HISTORY_HOURS
from local_forecast.history import HourlyHistory, hour_of, issue_hour

H = 3600
T0 = 1_800_000_000 - 1_800_000_000 % H  # a whole hour


def test_readings_count_for_the_nearest_hour():
    assert hour_of(T0 + 20 * 60) == T0
    assert hour_of(T0 + 40 * 60) == T0 + H


def test_issue_hour_is_the_last_closed_window():
    # At 10:42 the 11:00 window (10:30-11:30) is still open; 10:00 is closed.
    assert issue_hour(T0 + 42 * 60) == T0
    assert issue_hour(T0 + 25 * 60) == T0 - H


def test_means_and_flags():
    h = HourlyHistory()
    h.add(T0 - 600, temperature=10.0, humidity=50.0, wind_speed=2.0, wind_direction=350, pressure=1010.0, rain_rate=0)
    h.add(T0 + 600, temperature=12.0, humidity=50.0, wind_speed=4.0, wind_direction=10, pressure=1012.0, rain_rate=0.4)
    s = h.series(T0)
    assert len(s["t"]) == HISTORY_HOURS
    assert s["t"][-1] == pytest.approx(11.0)
    assert s["p"][-1] == pytest.approx(1011.0)
    assert s["ws"][-1] == pytest.approx(3.0)
    assert min(s["wd"][-1], 360 - s["wd"][-1]) < 1e-6  # vector mean of 350 and 10 is north
    assert s["wet"][-1] == 1.0
    assert math.isnan(s["t"][-2])


def test_missing_channels_stay_missing():
    h = HourlyHistory()
    h.add(T0, temperature=5.0, pressure=1000.0)
    s = h.series(T0)
    assert math.isnan(s["td"][-1]) and math.isnan(s["ws"][-1]) and math.isnan(s["wet"][-1])


def test_dump_load_round_trip():
    h = HourlyHistory()
    for k in range(5):
        h.add(T0 + k * H, temperature=float(k), humidity=80.0, wind_speed=1.0, wind_direction=90, pressure=1000.0)
    g = HourlyHistory()
    g.load(h.dump())
    a, b = h.series(T0 + 4 * H), g.series(T0 + 4 * H)
    for c in a:
        for x, y in zip(a[c], b[c], strict=True):
            assert (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y, abs=0.01)


def test_old_hours_are_pruned():
    h = HourlyHistory()
    h.add(T0, temperature=1.0, pressure=1000.0)
    h.add(T0 + (HISTORY_HOURS + 5) * H, temperature=2.0, pressure=1000.0)
    assert not h.has(T0)


def test_seed_fills_only_unseen_hours():
    h = HourlyHistory()
    h.add(T0, temperature=10.0, pressure=1000.0)
    added = h.seed([(T0, {"t": 99.0}), (T0 - H, {"t": 8.0, "rh": 70.0, "p": 1001.0, "rain": 0.0})])
    assert added == 1
    s = h.series(T0)
    assert s["t"][-1] == 10.0 and s["t"][-2] == 8.0 and s["wet"][-2] == 0.0
