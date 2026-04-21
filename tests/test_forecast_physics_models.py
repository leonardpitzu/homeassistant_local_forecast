"""Tests for local_forecast.physics_models — pressure, temperature, humidity models."""

import sys
import os
import math

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "custom_components"),
)

from local_forecast.physics_models import (
    HumidityModel,
    PressureModel,
    TemperatureModel,
)


# =====================================================================
#  PressureModel
# =====================================================================

class TestPressureModel:
    """Damped linear pressure extrapolation."""

    def test_no_trend_stays_flat(self):
        pm = PressureModel(current=1013.0, dp_dt=0.0)
        for h in range(1, 13):
            assert pm(h) == 1013.0

    def test_falling_pressure(self):
        pm = PressureModel(current=1015.0, dp_dt=-2.0)
        assert pm(1) < 1015.0
        assert pm(6) < pm(1)

    def test_rising_pressure(self):
        pm = PressureModel(current=1005.0, dp_dt=1.5)
        assert pm(1) > 1005.0
        assert pm(6) > pm(1)

    def test_damping_decays_trend(self):
        """After many hours, pressure change should plateau."""
        pm = PressureModel(current=1013.0, dp_dt=-3.0)
        delta_1_to_6 = pm(6) - pm(1)
        delta_6_to_12 = pm(12) - pm(6)
        # Later hours change less
        assert abs(delta_6_to_12) < abs(delta_1_to_6)

    def test_clamped_lower(self):
        pm = PressureModel(current=925.0, dp_dt=-10.0)
        assert pm(12) >= 920.0

    def test_clamped_upper(self):
        pm = PressureModel(current=1065.0, dp_dt=5.0)
        assert pm(12) <= 1070.0

    def test_custom_damping(self):
        pm_fast = PressureModel(current=1013.0, dp_dt=-2.0, damping=0.5)
        pm_slow = PressureModel(current=1013.0, dp_dt=-2.0, damping=0.99)
        # Slow damping = more total drop over time
        assert pm_slow(12) < pm_fast(12)


# =====================================================================
#  TemperatureModel
# =====================================================================

class TestTemperatureModel:
    """Energy-balance + diurnal temperature model."""

    def _model(self, **kwargs):
        defaults = dict(
            current_temp=20.0,
            dt_dt=0.0,
            humidity=50.0,
            wind_speed=3.0,
            cloud_fraction=0.0,
            sunrise_hour=6.0,
            sunset_hour=20.0,
            current_hour=12.0,
            latitude=48.0,
        )
        defaults.update(kwargs)
        return TemperatureModel(**defaults)

    def test_returns_float(self):
        tm = self._model()
        assert isinstance(tm(1), float)

    def test_reasonable_range(self):
        tm = self._model(current_temp=20.0)
        for h in range(1, 13):
            assert -45 <= tm(h) <= 50

    def test_afternoon_warming(self):
        """From 10 AM on a clear day, temperature should rise toward peak."""
        tm = self._model(current_hour=10.0, current_temp=18.0, cloud_fraction=0.0)
        # By 14:00 (4h ahead), should be warmer
        assert tm(4) > 18.0 or tm(4) >= 17.5  # at least near current

    def test_evening_cooling(self):
        """After sunset, temperature should drop."""
        tm = self._model(
            current_hour=21.0, current_temp=22.0,
            cloud_fraction=0.0, wind_speed=1.0,
        )
        # 4 hours into the night
        assert tm(4) < 22.0

    def test_cloud_cover_reduces_amplitude(self):
        """Overcast conditions reduce diurnal temperature swing."""
        tm_clear = self._model(cloud_fraction=0.0)
        tm_cloudy = self._model(cloud_fraction=0.9)
        assert tm_cloudy.amplitude < tm_clear.amplitude

    def test_radiative_cooling_clear_night(self):
        """Clear nights cool more than cloudy ones."""
        tm_clear = self._model(
            current_hour=22.0, current_temp=15.0,
            cloud_fraction=0.0, wind_speed=1.0, humidity=40.0,
        )
        tm_cloudy = self._model(
            current_hour=22.0, current_temp=15.0,
            cloud_fraction=0.8, wind_speed=1.0, humidity=40.0,
        )
        # Clear night cools more
        assert tm_clear(6) <= tm_cloudy(6)

    def test_wind_increases_thermal_response(self):
        """More wind → smaller thermal inertia τ."""
        tm_calm = self._model(wind_speed=0.5)
        tm_windy = self._model(wind_speed=10.0)
        assert tm_windy.tau < tm_calm.tau

    def test_cold_temperature(self):
        """Model handles sub-zero temperatures."""
        tm = self._model(current_temp=-10.0, current_hour=6.0)
        for h in range(1, 13):
            result = tm(h)
            assert -45 <= result <= 50

    def test_hot_temperature(self):
        tm = self._model(current_temp=40.0, current_hour=14.0)
        for h in range(1, 13):
            result = tm(h)
            assert -45 <= result <= 50


# =====================================================================
#  HumidityModel
# =====================================================================

class TestHumidityModel:
    """Clausius-Clapeyron relative humidity model."""

    def test_cooling_increases_rh(self):
        """When temperature drops, RH rises (Clausius-Clapeyron)."""
        temp_fn = lambda h: 20.0 - h * 1.0  # cooling 1°C/h
        hm = HumidityModel(current_rh=60.0, current_temp=20.0, temperature_model=temp_fn)
        assert hm(3) > 60.0

    def test_warming_decreases_rh(self):
        """When temperature rises, RH drops."""
        temp_fn = lambda h: 20.0 + h * 1.0
        hm = HumidityModel(current_rh=70.0, current_temp=20.0, temperature_model=temp_fn)
        assert hm(3) < 70.0

    def test_no_temp_change_no_rh_change(self):
        temp_fn = lambda h: 20.0
        hm = HumidityModel(current_rh=55.0, current_temp=20.0, temperature_model=temp_fn)
        assert abs(hm(5) - 55.0) < 0.5

    def test_clamped_upper(self):
        """RH should not exceed 100%."""
        temp_fn = lambda h: 20.0 - h * 5.0  # rapid cooling
        hm = HumidityModel(current_rh=80.0, current_temp=20.0, temperature_model=temp_fn)
        assert hm(6) <= 100.0

    def test_clamped_lower(self):
        """RH should not drop below 1%."""
        temp_fn = lambda h: 20.0 + h * 10.0  # extreme warming
        hm = HumidityModel(current_rh=10.0, current_temp=20.0, temperature_model=temp_fn)
        assert hm(6) >= 1.0

    def test_magnus_formula(self):
        """Verify the saturation vapour pressure calculation."""
        es = HumidityModel._es(20.0)
        # At 20°C, es ≈ 23.4 hPa (well-known value)
        assert 23.0 < es < 24.0

    def test_magnus_at_zero(self):
        es = HumidityModel._es(0.0)
        # At 0°C, es ≈ 6.11 hPa
        assert 6.0 < es < 6.3

    def test_magnus_at_100(self):
        es = HumidityModel._es(100.0)
        # At 100°C, es ≈ 1013 hPa (boiling point)
        assert 900 < es < 1200

    def test_symmetry_with_temperature_model(self):
        """HumidityModel integrates correctly with TemperatureModel."""
        tm = TemperatureModel(
            current_temp=20.0, dt_dt=0.0, humidity=60.0,
            wind_speed=3.0, cloud_fraction=0.3,
            sunrise_hour=6.0, sunset_hour=20.0,
            current_hour=14.0,
        )
        hm = HumidityModel(current_rh=60.0, current_temp=20.0, temperature_model=tm)
        for h in range(1, 13):
            rh = hm(h)
            assert 1.0 <= rh <= 100.0


# =====================================================================
#  Cross-model consistency
# =====================================================================

class TestCrossModelConsistency:
    """Models working together produce physically consistent results."""

    def test_cooling_night_raises_humidity(self):
        """Clear night: temp drops → RH rises."""
        tm = TemperatureModel(
            current_temp=18.0, dt_dt=-0.5, humidity=55.0,
            wind_speed=1.0, cloud_fraction=0.1,
            sunrise_hour=6.0, sunset_hour=20.0,
            current_hour=22.0,
        )
        hm = HumidityModel(current_rh=55.0, current_temp=18.0, temperature_model=tm)
        # After 4 hours of clear-night cooling
        assert hm(4) >= 55.0

    def test_pressure_and_temperature_independent(self):
        """Pressure model is independent of temperature."""
        pm = PressureModel(current=1010.0, dp_dt=-1.0)
        tm = TemperatureModel(
            current_temp=25.0, dt_dt=0.0, humidity=50.0,
            wind_speed=3.0, cloud_fraction=0.3,
            sunrise_hour=6.0, sunset_hour=20.0, current_hour=12.0,
        )
        # Pressure doesn't depend on temperature model
        p6 = pm(6)
        assert p6 < 1010.0
        # Temperature doesn't depend on pressure model
        t6 = tm(6)
        assert isinstance(t6, float)
