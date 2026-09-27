"""Tests for the learned atmospheric tide and its removal from the pressure trend."""

import math
import os
import random
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "custom_components"),
)

from local_forecast.state_estimator import SensorReading, StateEstimator
from local_forecast.tide import (
    S2_AMP_EQUATOR_HPA,
    LocalTide,
    s2_amplitude,
    solar_hour,
)

BRASOV_LAT = 45.65
BRASOV_LON = 25.60
W = 2 * math.pi
START = 1_790_000_000.0


def _station(days, *, gain, t_amp, lon=BRASOV_LON, seed=1, start=START):
    """Hourly (ts, pressure, temperature, true tide) from a synthetic station.

    Pressure S1 is the temperature cycle, inverted and 2 h late, times ``gain``;
    S2 and S3 are fixed; weather is a random walk that knows no hour of day.
    """
    rng = random.Random(seed)
    weather = 0.0
    for i in range(int(days * 24)):
        ts = start + i * 3600.0
        h = solar_hour(ts, lon)
        amp = t_amp(i / 24.0) if callable(t_amp) else t_amp
        temperature = 10.0 + amp * math.cos(W * (h - 15.0) / 24.0) + rng.gauss(0, 0.3)
        weather += rng.gauss(0, 0.3)
        tide = (
            -gain * amp * math.cos(W * (h - 17.0) / 24.0)
            + 0.5 * math.cos(W * (h - 9.5) / 12.0)
            + 0.15 * math.cos(W * (h - 2.0) / 8.0)
        )
        yield ts, 1015.0 + weather + tide, temperature, tide


def _rms_error(model, samples):
    """RMS of (model - truth) with each series' daily mean removed."""
    errors = [model.hpa_at(ts) - truth for ts, _, _, truth in samples]
    mean = sum(errors) / len(errors)
    return math.sqrt(sum((e - mean) ** 2 for e in errors) / len(errors))


def _trained(days, **kw):
    tide = LocalTide(BRASOV_LAT, BRASOV_LON)
    last = []
    for sample in _station(days, **kw):
        ts, p, t, _ = sample
        tide.learn(ts, p, t)
        last.append(sample)
    return tide, last[-48:]


class TestS2Amplitude:
    def test_follows_haurwitz_cos_cubed(self) -> None:
        for lat in (0.0, 16.0, 32.0, 48.0, -32.0):
            expected = S2_AMP_EQUATOR_HPA * math.cos(math.radians(lat)) ** 3
            assert math.isclose(s2_amplitude(lat), expected, rel_tol=1e-12)

    def test_unknown_latitude_falls_back_to_mid_latitude(self) -> None:
        assert 0.3 < s2_amplitude(None) < 0.55

    def test_vanishes_at_the_pole(self) -> None:
        assert s2_amplitude(90.0) < 1e-9


class TestSolarHour:
    def test_greenwich_solar_time_is_utc(self) -> None:
        assert solar_hour(0.0, 0.0) == 0.0

    def test_longitude_shifts_by_four_minutes_per_degree(self) -> None:
        assert math.isclose(solar_hour(0.0, 15.0), 1.0)
        assert math.isclose(solar_hour(0.0, -15.0), 23.0)

    def test_unknown_longitude_falls_back_to_utc(self) -> None:
        assert solar_hour(3600.0, None) == 1.0


class TestPrior:
    def test_new_install_starts_from_the_global_semidiurnal_tide(self) -> None:
        """Nothing learned yet: S2 by Haurwitz, peaking at 10:00 solar."""
        tide = LocalTide(BRASOV_LAT, BRASOV_LON)
        described = tide.describe(START)
        assert math.isclose(described["s2_hpa"]["amplitude"], s2_amplitude(BRASOV_LAT), abs_tol=1e-3)
        assert math.isclose(described["s2_hpa"]["peak_solar_hour"], 10.0, abs_tol=0.01)
        assert described["s1_hpa"]["amplitude"] == 0.0  # no temperature cycle seen yet
        assert described["learned_days"] == 0.0


class TestLearning:
    def test_learns_the_station_it_runs_on(self) -> None:
        tide, recent = _trained(30, gain=0.12, t_amp=6.0)
        # Across seeds the learned tide lands 0.05-0.12 hPa from the truth; the prior is >0.4 off.
        assert _rms_error(tide, recent) < 0.15
        assert _rms_error(LocalTide(BRASOV_LAT, BRASOV_LON), recent) > 0.4
        described = tide.describe(recent[-1][0])
        assert math.isclose(described["site_gain_hpa_per_k"], 0.12, rel_tol=0.25)
        assert math.isclose(described["trough_after_temperature_peak_h"], 2.0, abs_tol=0.75)
        assert math.isclose(described["s2_hpa"]["amplitude"], 0.5, abs_tol=0.1)

    def test_daily_heating_cycle_follows_the_thermometer_within_days(self) -> None:
        """A cloudy spell shrinks S1 long before the 14-day site memory could."""

        def cycle(day):
            return 6.0 if day < 30 else 2.0

        tide = LocalTide(BRASOV_LAT, BRASOV_LON)
        before = after = None
        for ts, p, t, _ in _station(34, gain=0.12, t_amp=cycle):
            tide.learn(ts, p, t)
            if ts == START + 30 * 86400:
                before = tide.describe(ts)["s1_hpa"]["amplitude"]
        after = tide.describe(ts)["s1_hpa"]["amplitude"]
        assert after < 0.55 * before

    def test_adapts_to_a_move_with_no_reset(self) -> None:
        """A basin station moved to a plain: the site gain relearns on its own."""
        tide = LocalTide(BRASOV_LAT, BRASOV_LON)
        for ts, p, t, _ in _station(30, gain=0.15, t_amp=6.0, seed=2):
            tide.learn(ts, p, t)
        recent = []
        for sample in _station(45, gain=0.03, t_amp=6.0, seed=3, start=START + 31 * 86400):
            tide.learn(*sample[:3])
            recent.append(sample)
        assert tide.describe(recent[-1][0])["site_gain_hpa_per_k"] < 0.08

    def test_sensor_glitch_is_not_learned(self) -> None:
        tide, _ = _trained(10, gain=0.12, t_amp=6.0)
        before = tide.dump()
        last = before["recent"][-3]
        tide.learn(last[0] + 3 * 3600.0, last[1] + 40.0, last[2])
        assert tide.dump()["pressure"] == before["pressure"]


class TestPersistence:
    def test_round_trip_keeps_what_was_learned(self) -> None:
        tide, recent = _trained(20, gain=0.12, t_amp=6.0)
        restored = LocalTide(BRASOV_LAT, BRASOV_LON)
        restored.load(tide.dump())
        for ts, *_ in recent:
            assert math.isclose(restored.hpa_at(ts), tide.hpa_at(ts), abs_tol=1e-9)

    def test_malformed_state_falls_back_to_the_prior(self) -> None:
        prior = LocalTide(BRASOV_LAT, BRASOV_LON)
        for junk in (
            None,
            [],
            "x",
            {"pressure": {"a": [[1.0]], "b": [1.0], "last": None, "evidence": 1.0}},
            {"temperature": {"a": "nope"}},
            {"recent": [["a", 1, 2], [1.0]]},
        ):
            tide = LocalTide(BRASOV_LAT, BRASOV_LON)
            tide.load(junk)
            assert tide.hpa_at(START) == prior.hpa_at(START)


def _feed(est: StateEstimator, tide: LocalTide, *, hours: float, rate_hpa_h: float, with_tide: bool) -> None:
    """Feed a linear ramp, optionally carrying the tide, at 2 min spacing."""
    t0 = 1_754_600_000.0
    steps = int(hours * 30)
    for i in range(steps + 1):
        t = t0 + i * 120.0
        p = 1013.0 + rate_hpa_h * (i * 120.0) / 3600.0
        if with_tide:
            p += tide.hpa_at(t)
        est.update(SensorReading(timestamp=t, pressure_hpa=p, temperature_c=15.0, humidity_pct=60.0))


class TestTendencyIsDetided:
    def setup_method(self) -> None:
        self.tide, _ = _trained(20, gain=0.12, t_amp=6.0)

    def test_tide_bearing_flat_series_reads_as_steady(self) -> None:
        est = StateEstimator(tide_hpa=self.tide.hpa_at)
        _feed(est, self.tide, hours=1.5, rate_hpa_h=0.0, with_tide=True)
        assert abs(est.state.dp_dt) < 0.02

    def test_uncorrected_estimator_is_fooled_by_the_same_series(self) -> None:
        naive = StateEstimator()
        _feed(naive, self.tide, hours=1.5, rate_hpa_h=0.0, with_tide=True)
        fixed = StateEstimator(tide_hpa=self.tide.hpa_at)
        _feed(fixed, self.tide, hours=1.5, rate_hpa_h=0.0, with_tide=True)
        assert abs(fixed.state.dp_dt) < abs(naive.state.dp_dt)

    def test_real_tendency_survives_the_correction(self) -> None:
        est = StateEstimator(tide_hpa=self.tide.hpa_at)
        _feed(est, self.tide, hours=1.5, rate_hpa_h=-1.5, with_tide=True)
        assert math.isclose(est.state.dp_dt, -1.5, abs_tol=0.05)

    def test_reported_pressure_keeps_its_tide(self) -> None:
        """Only the tendency is de-tided; the barometer reading is what it is."""
        est = StateEstimator(tide_hpa=self.tide.hpa_at)
        _feed(est, self.tide, hours=1.5, rate_hpa_h=0.0, with_tide=True)
        assert math.isclose(est.state.pressure, 1013.0 + self.tide.hpa_at(1_754_600_000.0 + 5400.0), abs_tol=0.15)
