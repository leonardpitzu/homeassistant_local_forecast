"""Tests for local_forecast.pressure_history — hourly buffer, tendency, mean."""

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "custom_components"),
)

from local_forecast.pressure_history import PressureHistory

HOUR = 3600.0


def _fill_hourly(buf, start_ts, pressures):
    """Record one sample per hour starting at start_ts."""
    for i, p in enumerate(pressures):
        buf.record(start_ts + i * HOUR, p)


class TestRecord:
    def test_records_first_sample(self):
        buf = PressureHistory()
        buf.record(1000.0, 1013.0)
        assert buf.dump() == [[1000.0, 1013.0]]

    def test_rejects_sub_hourly_samples(self):
        buf = PressureHistory()
        buf.record(1000.0, 1013.0)
        buf.record(1000.0 + 600, 1014.0)  # 10 min later — ignored
        assert len(buf.dump()) == 1

    def test_rejects_non_finite(self):
        buf = PressureHistory()
        buf.record(1000.0, float("nan"))
        buf.record(2000.0, float("inf"))
        buf.record(3000.0, None)
        assert buf.dump() == []

    def test_prunes_beyond_window(self):
        buf = PressureHistory()
        _fill_hourly(buf, 0.0, [1013.0] * 30)  # 30 h of samples
        # Window is 24 h (+1 h margin); oldest should be dropped.
        samples = buf.dump()
        newest = samples[-1][0]
        assert all(newest - ts <= 25 * HOUR for ts, _ in samples)


class TestTendency:
    def test_warmup_returns_none(self):
        buf = PressureHistory()
        buf.record(0.0, 1013.0)
        assert buf.tendency_per_hour(0.0, 1013.0) is None

    def test_three_hour_tendency(self):
        buf = PressureHistory()
        # Samples at 0,1,2,3 h; pressure falling 3 hPa over 3 h.
        _fill_hourly(buf, 0.0, [1016.0, 1015.0, 1014.0, 1013.0])
        now = 3 * HOUR
        tendency = buf.tendency_per_hour(now, 1013.0)
        # p_now 1013 vs p(0h) 1016 over 3 h → -1.0 hPa/h
        assert tendency is not None
        assert abs(tendency - (-1.0)) < 1e-6

    def test_none_current_pressure(self):
        buf = PressureHistory()
        _fill_hourly(buf, 0.0, [1016.0, 1015.0, 1014.0, 1013.0])
        assert buf.tendency_per_hour(3 * HOUR, None) is None


class TestMean:
    def test_empty_returns_none(self):
        buf = PressureHistory()
        assert buf.mean(1000.0) is None

    def test_mean_over_window(self):
        buf = PressureHistory()
        _fill_hourly(buf, 0.0, [1010.0, 1012.0, 1014.0])
        mean = buf.mean(2 * HOUR)
        assert abs(mean - 1012.0) < 1e-6

    def test_mean_includes_current(self):
        buf = PressureHistory()
        buf.record(0.0, 1010.0)
        # current 1014 → mean(1010, 1014) = 1012
        assert abs(buf.mean(HOUR, 1014.0) - 1012.0) < 1e-6

    def test_excludes_samples_older_than_window(self):
        buf = PressureHistory(maxlen=100)
        # One old sample far outside window plus recent ones.
        buf.record(0.0, 900.0)
        _fill_hourly(buf, 30 * HOUR, [1013.0, 1013.0])
        mean = buf.mean(31 * HOUR)
        assert abs(mean - 1013.0) < 1e-6


class TestPersistence:
    def test_dump_load_roundtrip(self):
        buf = PressureHistory()
        _fill_hourly(buf, 0.0, [1010.0, 1011.0, 1012.0])
        dumped = buf.dump()

        restored = PressureHistory()
        restored.load(dumped)
        assert restored.dump() == dumped

    def test_load_merges_with_live_sample(self):
        buf = PressureHistory()
        buf.record(3 * HOUR, 1013.0)  # live sample after restart
        buf.load([[0.0, 1010.0], [HOUR, 1011.0], [2 * HOUR, 1012.0]])
        timestamps = [ts for ts, _ in buf.dump()]
        assert timestamps == [0.0, HOUR, 2 * HOUR, 3 * HOUR]

    def test_load_thins_duplicates(self):
        buf = PressureHistory()
        buf.record(0.0, 1010.0)
        # Near-duplicate (few minutes off) should be dropped on merge.
        buf.load([[300.0, 1010.5], [HOUR, 1011.0]])
        timestamps = [ts for ts, _ in buf.dump()]
        assert timestamps == [0.0, HOUR]

    def test_load_ignores_malformed(self):
        buf = PressureHistory()
        buf.load([[0.0, 1010.0], ["bad"], [None, None], [HOUR, 1011.0]])
        assert buf.dump() == [[0.0, 1010.0], [HOUR, 1011.0]]
