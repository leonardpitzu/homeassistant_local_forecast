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
    h.add(T0 + 600, temperature=12.0, humidity=50.0, wind_speed=4.0, wind_direction=10, pressure=1012.0, rain_rate=0.6)
    s = h.series(T0)
    assert len(s["t"]) == HISTORY_HOURS
    assert s["t"][-1] == pytest.approx(11.0)
    assert s["p"][-1] == pytest.approx(1011.0)
    assert s["ws"][-1] == pytest.approx(3.0)
    assert min(s["wd"][-1], 360 - s["wd"][-1]) < 1e-6  # vector mean of 350 and 10 is north
    assert s["rain"][-1] == pytest.approx(0.3)  # mean rate over the hour = mm in it
    assert s["wet"][-1] == 1.0
    assert math.isnan(s["t"][-2])


def test_a_wet_hour_is_measured_rain_not_a_trace():
    """The models learned >= 0.2 mm in the hour; a passing drizzle is not wet."""
    h = HourlyHistory()
    h.add(T0 - 600, temperature=10.0, pressure=1010.0, rain_rate=0.0)
    h.add(T0 + 600, temperature=10.0, pressure=1010.0, rain_rate=0.3)
    assert h.series(T0)["wet"][-1] == 0.0


def test_solar_is_kept():
    h = HourlyHistory()
    h.add(T0, temperature=20.0, pressure=1015.0, solar=640.0)
    h.add(T0 + 60, temperature=20.0, pressure=1015.0, solar=660.0)
    assert h.series(T0)["solar"][-1] == pytest.approx(650.0)


def test_missing_channels_stay_missing():
    h = HourlyHistory()
    h.add(T0, temperature=5.0, pressure=1000.0)
    s = h.series(T0)
    for ch in ("td", "ws", "wet", "rain", "solar"):
        assert math.isnan(s[ch][-1]), ch


def test_dump_load_round_trip():
    h = HourlyHistory()
    for k in range(5):
        h.add(T0 + k * H, temperature=float(k), humidity=80.0, wind_speed=1.0, wind_direction=90, pressure=1000.0,
              rain_rate=0.1 * k, solar=100.0 * k)  # fmt: skip
    g = HourlyHistory()
    g.load(h.dump())
    a, b = h.series(T0 + 4 * H), g.series(T0 + 4 * H)
    for c in a:
        for x, y in zip(a[c], b[c], strict=True):
            assert (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y, abs=0.01)


def test_a_2_0_0_save_loads_without_inventing_rain():
    g = HourlyHistory()
    g.load([[T0, 10.0, 5.0, 2.0, 0.0, 1.0, 1012.0, 1.0]])  # 2.0.0 kept a wet flag
    s = g.series(T0)
    assert s["t"][-1] == 10.0 and s["p"][-1] == 1012.0
    assert math.isnan(s["rain"][-1]) and math.isnan(s["wet"][-1])


def test_old_hours_are_pruned():
    h = HourlyHistory()
    h.add(T0, temperature=1.0, pressure=1000.0)
    h.add(T0 + (HISTORY_HOURS + 5) * H, temperature=2.0, pressure=1000.0)
    assert not h.has(T0)


def test_seed_fills_only_unseen_hours():
    h = HourlyHistory()
    h.add(T0, temperature=10.0, pressure=1000.0)
    added = h.seed([(T0, {"t": 99.0}), (T0 - H, {"t": 8.0, "rh": 70.0, "p": 1001.0, "rain": 1.2, "solar": 50.0})])
    assert added == 1
    s = h.series(T0)
    assert s["t"][-1] == 10.0 and s["t"][-2] == 8.0
    assert s["wet"][-2] == 1.0 and s["solar"][-2] == 50.0


def test_seed_fills_the_channels_an_hour_lacks():
    """An hour read live before the rain sensor existed still gets its rain."""
    h = HourlyHistory()
    h.add(T0, temperature=10.0, pressure=1000.0)
    assert h.seed([(T0, {"t": 99.0, "p": 900.0, "rain": 0.8, "solar": 120.0})]) == 1
    s = h.series(T0)
    assert s["t"][-1] == 10.0 and s["p"][-1] == 1000.0  # live readings win
    assert s["rain"][-1] == 0.8 and s["wet"][-1] == 1.0 and s["solar"][-1] == 120.0
    assert h.seed([(T0, {"rain": 5.0})]) == 0  # seeding again changes nothing
    assert h.series(T0)["rain"][-1] == 0.8


def test_peaks_are_the_raw_readings_not_hourly_means():
    h = HourlyHistory()
    for minutes, t in ((-25, 10.0), (0, 14.5), (25, 12.0)):
        h.add(T0 + minutes * 60, temperature=t, pressure=1000.0)
    h.add(T0 + H, temperature=11.0, pressure=1000.0)
    assert h.series(T0)["t"][-1] == pytest.approx(12.17, abs=0.01)
    assert h.peaks(T0, T0 + 2 * H) == (14.5, 10.0)
    assert h.peaks(T0 + 2 * H, T0 + 3 * H) is None


def test_seeded_hours_carry_the_recorder_max_and_min():
    h = HourlyHistory()
    h.seed([(T0, {"t": 12.0, "tmax": 15.0, "tmin": 9.0}), (T0 + H, {"t": 13.0})])
    assert h.peaks(T0, T0 + 2 * H) == (15.0, 9.0)


def test_peaks_survive_a_restart():
    h = HourlyHistory()
    h.add(T0 - 600, temperature=8.0, pressure=1000.0)
    h.add(T0 + 600, temperature=16.0, pressure=1000.0)
    g = HourlyHistory()
    g.load(h.dump())
    assert g.peaks(T0, T0 + H) == (16.0, 8.0)


def test_a_2_0_0_save_has_no_peaks_so_means_stand_in():
    g = HourlyHistory()
    g.load([[T0, 10.0, 5.0, 2.0, 0.0, 1.0, 1012.0, 1.0]])
    assert g.peaks(T0, T0 + H) == (10.0, 10.0)
    g.seed([(T0, {"t": 10.0, "tmax": 13.0, "tmin": 7.0, "rain": 0.4})])  # the upgrade backfill
    assert g.peaks(T0, T0 + H) == (13.0, 7.0)
    assert g.series(T0)["rain"][-1] == 0.4
