"""Tests for the atmospheric tide model and its effect on the pressure tendency."""

import math
import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "custom_components"),
)

from local_forecast.state_estimator import SensorReading, StateEstimator
from local_forecast.tide import (
    S2_AMP_EQUATOR_PA,
    s1_tide,
    s2_amplitude,
    solar_hour,
    tide_pa,
)

BRASOV_LAT = 45.65
BRASOV_LON = 25.60


class TestS2Amplitude:
    def test_follows_haurwitz_cos_cubed(self) -> None:
        for lat in (0.0, 16.0, 32.0, 48.0, -32.0):
            expected = S2_AMP_EQUATOR_PA * math.cos(math.radians(lat)) ** 3
            assert math.isclose(s2_amplitude(lat), expected, rel_tol=1e-12)

    def test_unknown_latitude_falls_back_to_mid_latitude(self) -> None:
        assert 30.0 < s2_amplitude(None) < 55.0

    def test_vanishes_at_the_pole(self) -> None:
        assert s2_amplitude(90.0) < 1e-9


class TestS1Tide:
    def test_reproduces_the_zonal_means(self) -> None:
        # Node values from Dai & Wang (1999); the table is those, so hitting a
        # node exactly must return them.
        amp, phase = s1_tide(0.0)
        assert round(amp, 1) == 56.2
        assert round(phase, 2) == 6.28

    def test_vanishes_at_the_poles(self) -> None:
        assert s1_tide(90.0)[0] < 1e-9
        assert s1_tide(-90.0)[0] < 1e-9

    def test_interpolates_between_nodes(self) -> None:
        # Halfway between the 30 and 45 deg nodes, so bounded by both.
        mid = s1_tide(37.5)[0]
        assert min(s1_tide(30.0)[0], s1_tide(45.0)[0]) <= mid <= max(s1_tide(30.0)[0], s1_tide(45.0)[0])

    def test_phase_never_wraps_negative(self) -> None:
        for lat in range(-90, 91, 5):
            assert 0.0 <= s1_tide(float(lat))[1] < 24.0


class TestSolarHour:
    def test_greenwich_solar_time_is_utc(self) -> None:
        assert solar_hour(0.0, 0.0) == 0.0

    def test_longitude_shifts_by_four_minutes_per_degree(self) -> None:
        assert math.isclose(solar_hour(0.0, 15.0), 1.0)
        assert math.isclose(solar_hour(0.0, -15.0), 23.0)

    def test_unknown_longitude_falls_back_to_utc(self) -> None:
        assert solar_hour(3600.0, None) == 1.0


class TestTideMagnitude:
    def test_slope_can_rival_the_steady_threshold(self) -> None:
        """The reason this module exists: the tide alone reaches the 0.3 hPa/h dead zone."""
        worst = max(
            abs(tide_pa((h + 1) / 60.0, BRASOV_LAT) - tide_pa(h / 60.0, BRASOV_LAT)) * 60.0 for h in range(24 * 60)
        )
        assert worst / 100.0 > 0.25  # hPa/h

    def test_tropics_exceed_it_outright(self) -> None:
        worst = max(abs(tide_pa((h + 1) / 60.0, 0.0) - tide_pa(h / 60.0, 0.0)) * 60.0 for h in range(24 * 60))
        assert worst / 100.0 > 0.5


def _feed(est: StateEstimator, *, hours: float, rate_hpa_h: float, with_tide: bool, lat, lon) -> None:
    """Feed a linear ramp, optionally carrying the real tide, at 2 min spacing."""
    t0 = 1_754_600_000.0
    steps = int(hours * 30)
    for i in range(steps + 1):
        t = t0 + i * 120.0
        p = 1013.0 + rate_hpa_h * (i * 120.0) / 3600.0
        if with_tide:
            p += tide_pa(solar_hour(t, lon), lat) / 100.0
        est.update(SensorReading(timestamp=t, pressure_hpa=p, temperature_c=15.0, humidity_pct=60.0))


class TestTendencyIsDetided:
    def test_tide_bearing_flat_series_reads_as_steady(self) -> None:
        """A tide with no weather must not look like a tendency."""
        est = StateEstimator(latitude=BRASOV_LAT, longitude=BRASOV_LON)
        _feed(est, hours=1.5, rate_hpa_h=0.0, with_tide=True, lat=BRASOV_LAT, lon=BRASOV_LON)
        assert abs(est.state.dp_dt) < 0.05

    def test_uncorrected_estimator_is_fooled_by_the_same_series(self) -> None:
        """Guards the fix: without a position the tide survives into dp_dt."""
        est = StateEstimator()
        _feed(est, hours=1.5, rate_hpa_h=0.0, with_tide=True, lat=BRASOV_LAT, lon=BRASOV_LON)
        naive = abs(est.state.dp_dt)

        est_fixed = StateEstimator(latitude=BRASOV_LAT, longitude=BRASOV_LON)
        _feed(est_fixed, hours=1.5, rate_hpa_h=0.0, with_tide=True, lat=BRASOV_LAT, lon=BRASOV_LON)
        assert abs(est_fixed.state.dp_dt) < naive

    def test_real_tendency_survives_the_correction(self) -> None:
        est = StateEstimator(latitude=BRASOV_LAT, longitude=BRASOV_LON)
        _feed(est, hours=1.5, rate_hpa_h=-1.5, with_tide=True, lat=BRASOV_LAT, lon=BRASOV_LON)
        assert math.isclose(est.state.dp_dt, -1.5, abs_tol=0.1)

    def test_reported_pressure_keeps_its_tide(self) -> None:
        """Only the tendency is de-tided; the barometer reading is what it is."""
        est = StateEstimator(latitude=BRASOV_LAT, longitude=BRASOV_LON)
        _feed(est, hours=1.5, rate_hpa_h=0.0, with_tide=True, lat=BRASOV_LAT, lon=BRASOV_LON)
        assert 1012.0 < est.state.pressure < 1014.0
