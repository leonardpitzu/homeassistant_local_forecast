"""Tests for local_forecast.state_estimator — sensor fusion, trends, classification."""

import sys
import os
import time

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "custom_components"),
)

from local_forecast.state_estimator import SensorReading, SmoothedState, StateEstimator
from local_forecast.const import (
    HA_CONDITIONS,
    S_CLEAR,
    S_CLEAR_NIGHT,
    S_CLOUDY,
    S_FOG,
    S_LIGHTNING_RAINY,
    S_PARTLY_CLOUDY,
    S_POURING,
    S_RAINY,
    S_SNOWY,
    S_SNOWY_RAINY,
    S_WINDY,
)


def _reading(
    ts=None,
    pressure=1013.25,
    temp=20.0,
    humidity=50.0,
    wind=3.0,
    wind_dir=180.0,
    solar=None,
    rain=None,
):
    """Helper to build a SensorReading with defaults."""
    return SensorReading(
        timestamp=ts or time.time(),
        pressure_hpa=pressure,
        temperature_c=temp,
        humidity_pct=humidity,
        wind_speed_ms=wind,
        wind_direction_deg=wind_dir,
        solar_radiation_wm2=solar,
        rain_rate_mmh=rain,
    )


class TestSensorReading:
    """SensorReading data container."""

    def test_required_fields(self):
        r = SensorReading(
            timestamp=1000.0,
            pressure_hpa=1013.0,
            temperature_c=20.0,
        )
        assert r.pressure_hpa == 1013.0
        assert r.humidity_pct is None

    def test_optional_fields(self):
        r = _reading(humidity=85.0, rain=2.5)
        assert r.humidity_pct == 85.0
        assert r.rain_rate_mmh == 2.5


class TestStateEstimatorBasic:
    """StateEstimator initialisation and single-update behaviour."""

    def test_initial_state_is_zeroed(self):
        est = StateEstimator()
        s = est.state
        assert s.pressure == 0.0 or s.pressure >= 0  # initialised
        assert isinstance(s, SmoothedState)

    def test_single_update(self):
        """After several updates at the same value, Kalman converges."""
        est = StateEstimator()
        base = time.time()
        for i in range(15):
            est.update(_reading(ts=base + i * 60, pressure=1015.0, temp=22.0, humidity=60.0))
        s = est.state
        assert 1010 < s.pressure < 1020
        assert 20 < s.temperature < 24
        assert 50 < s.humidity < 70

    def test_multiple_updates_smooth(self):
        """Kalman filter should smooth out a spike."""
        est = StateEstimator()
        base = time.time()
        for i in range(10):
            est.update(_reading(ts=base + i * 60, pressure=1013.0, temp=20.0))
        # Inject spike
        est.update(_reading(ts=base + 660, pressure=1020.0, temp=20.0))
        # Should not jump to 1020 — Kalman damps it
        assert est.state.pressure < 1020.0


class TestTrends:
    """Pressure and temperature trend computation."""

    def test_falling_pressure_trend(self):
        est = StateEstimator()
        base = time.time()
        # Simulate 4 hours of falling pressure (1 reading per 10 min)
        for i in range(24):
            p = 1020.0 - i * 0.5  # -0.5 hPa per 10 min = -3 hPa/h
            est.update(_reading(ts=base + i * 600, pressure=p, temp=15.0))
        assert est.state.dp_dt < 0  # negative trend

    def test_rising_pressure_trend(self):
        est = StateEstimator()
        base = time.time()
        # Warm up Kalman at starting pressure
        for i in range(10):
            est.update(_reading(ts=base + i * 60, pressure=1000.0, temp=15.0))
        # Now apply a rising trend over 4 hours
        for i in range(24):
            p = 1000.0 + i * 0.5
            est.update(_reading(ts=base + 600 + i * 600, pressure=p, temp=15.0))
        assert est.state.dp_dt > 0


class TestClassifyClearConditions:
    """State classification for clear/cloudy conditions."""

    def test_clear_day(self):
        est = StateEstimator()
        base = time.time()
        for i in range(5):
            est.update(_reading(
                ts=base + i * 60,
                pressure=1020.0, temp=25.0, humidity=30.0,
                wind=2.0, solar=800.0, rain=0.0,
            ))
        est.state.is_night = False
        idx = est.classify()
        # Should be sunny, partlycloudy, or clear — not a precip state
        assert HA_CONDITIONS[idx] in ("sunny", "partlycloudy", "cloudy")

    def test_clear_night(self):
        est = StateEstimator()
        base = time.time()
        for i in range(5):
            est.update(_reading(
                ts=base + i * 60,
                pressure=1020.0, temp=10.0, humidity=40.0,
                wind=1.0, solar=0.0, rain=0.0,
            ))
        est.state.is_night = True
        idx = est.classify()
        cond = HA_CONDITIONS[idx]
        assert cond in ("clear-night", "partlycloudy", "cloudy", "fog")


class TestClassifyPrecipitation:
    """Classification when precipitation is active."""

    def test_rain_detected(self):
        est = StateEstimator()
        base = time.time()
        for i in range(5):
            est.update(_reading(
                ts=base + i * 60,
                pressure=1005.0, temp=15.0, humidity=90.0,
                wind=5.0, rain=3.0,
            ))
        est.state.is_night = False
        idx = est.classify()
        cond = HA_CONDITIONS[idx]
        assert cond in ("rainy", "pouring", "lightning-rainy")

    def test_heavy_rain_is_pouring(self):
        est = StateEstimator()
        base = time.time()
        for i in range(5):
            est.update(_reading(
                ts=base + i * 60,
                pressure=1000.0, temp=18.0, humidity=95.0,
                wind=8.0, rain=10.0,
            ))
        est.state.is_night = False
        idx = est.classify()
        cond = HA_CONDITIONS[idx]
        assert cond in ("pouring", "lightning-rainy")

    def test_snow_when_cold(self):
        """Rain rate + cold temp → snow (wet-bulb < -2)."""
        est = StateEstimator()
        base = time.time()
        for i in range(5):
            est.update(_reading(
                ts=base + i * 60,
                pressure=1005.0, temp=-5.0, humidity=85.0,
                wind=3.0, rain=2.0,
            ))
        est.state.is_night = False
        idx = est.classify()
        cond = HA_CONDITIONS[idx]
        assert cond in ("snowy", "snowy-rainy")

    def test_sleet_near_zero(self):
        """Near-zero wet-bulb → mixed precipitation."""
        est = StateEstimator()
        base = time.time()
        for i in range(5):
            est.update(_reading(
                ts=base + i * 60,
                pressure=1005.0, temp=0.5, humidity=90.0,
                wind=3.0, rain=2.0,
            ))
        est.state.is_night = False
        idx = est.classify()
        cond = HA_CONDITIONS[idx]
        assert cond in ("snowy-rainy", "snowy", "rainy")


class TestClassifyFog:
    """Fog classification."""

    def test_fog_low_dew_depression(self):
        """High humidity + low wind + small T-Td → fog."""
        est = StateEstimator()
        base = time.time()
        for i in range(5):
            est.update(_reading(
                ts=base + i * 60,
                pressure=1015.0, temp=10.0, humidity=98.0,
                wind=1.0, rain=0.0,
            ))
        est.state.is_night = True
        idx = est.classify()
        cond = HA_CONDITIONS[idx]
        assert cond in ("fog", "cloudy", "clear-night")


class TestClassifyWindy:
    """Windy state classification."""

    def test_strong_wind_no_precip(self):
        est = StateEstimator()
        base = time.time()
        for i in range(5):
            est.update(_reading(
                ts=base + i * 60,
                pressure=1010.0, temp=12.0, humidity=45.0,
                wind=15.0, rain=0.0,
            ))
        est.state.is_night = False
        idx = est.classify()
        cond = HA_CONDITIONS[idx]
        # Wind > 10 m/s with no precip → windy or cloudy
        assert cond in ("windy", "cloudy", "partlycloudy")


class TestWetBulbCalculation:
    """Wet-bulb temperature computation."""

    def test_wet_bulb_less_than_dry(self):
        """Wet-bulb should always be ≤ dry-bulb temperature."""
        est = StateEstimator()
        est.update(_reading(temp=25.0, humidity=50.0))
        assert est.state.wet_bulb <= 25.0

    def test_wet_bulb_equals_dry_at_100_rh(self):
        """At 100% RH, wet-bulb ≈ dry-bulb (after Kalman convergence)."""
        est = StateEstimator()
        base = time.time()
        for i in range(20):
            est.update(_reading(ts=base + i * 60, temp=20.0, humidity=100.0))
        assert abs(est.state.wet_bulb - 20.0) < 3.0

    def test_wet_bulb_much_lower_in_dry_air(self):
        est = StateEstimator()
        est.update(_reading(temp=35.0, humidity=15.0))
        assert est.state.wet_bulb < 25.0  # significant depression


class TestDewPoint:
    """Dew-point calculation."""

    def test_dew_point_below_temp(self):
        est = StateEstimator()
        est.update(_reading(temp=20.0, humidity=60.0))
        assert est.state.dew_point < 20.0

    def test_dew_point_near_temp_at_high_rh(self):
        est = StateEstimator()
        base = time.time()
        for i in range(20):
            est.update(_reading(ts=base + i * 60, temp=15.0, humidity=98.0))
        assert est.state.dew_point > 13.0


class TestFrontalDetection:
    """Frontal passage detection from trends."""

    def test_no_front_in_steady_conditions(self):
        est = StateEstimator()
        base = time.time()
        for i in range(20):
            est.update(_reading(
                ts=base + i * 600,
                pressure=1015.0, temp=18.0, humidity=55.0,
                wind=3.0, wind_dir=180.0,
            ))
        s = est.state
        # Steady conditions — no fronts
        assert not s.front_warm
        assert not s.front_cold
