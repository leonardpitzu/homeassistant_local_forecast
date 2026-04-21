"""Tests for local_forecast.bayesian_forecaster — Markov + Bayesian evidence."""

import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "custom_components"),
)

from local_forecast.bayesian_forecaster import BayesianForecaster, HourForecast
from local_forecast.state_estimator import SmoothedState
from local_forecast.const import (
    FORECAST_HOURS,
    HA_CONDITIONS,
    NUM_STATES,
    S_CLEAR,
    S_CLEAR_NIGHT,
    S_CLOUDY,
    S_FOG,
    S_POURING,
    S_RAINY,
    S_SNOWY,
    S_SNOWY_RAINY,
)


def _smoothed(
    pressure=1013.0, temperature=20.0, humidity=50.0,
    wind_speed=3.0, wind_direction=180.0,
    dp_dt=0.0, d2p_dt2=0.0, dt_dt=0.0, dh_dt=0.0,
    dew_point=10.0, dew_depression=10.0, wet_bulb=15.0,
    is_night=False,
    front_warm=False, front_cold=False, front_occluded=False,
    rain_rate=0.0, solar=0.0,
    dd_trend=0.0,
):
    """Build a SmoothedState with sensible defaults."""
    return SmoothedState(
        pressure=pressure,
        temperature=temperature,
        humidity=humidity,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        dp_dt=dp_dt,
        d2p_dt2=d2p_dt2,
        dt_dt=dt_dt,
        dh_dt=dh_dt,
        dew_point=dew_point,
        dew_depression=dew_depression,
        wet_bulb=wet_bulb,
        is_night=is_night,
        front_warm=front_warm,
        front_cold=front_cold,
        front_occluded=front_occluded,
        rain_rate=rain_rate,
        solar_radiation=solar,
        dd_trend=dd_trend,
    )


class TestForecasterBasic:
    """Basic forecaster behaviour."""

    def test_returns_correct_number_of_hours(self):
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_CLEAR,
            smoothed=_smoothed(),
            hours=FORECAST_HOURS,
        )
        assert len(result) == FORECAST_HOURS

    def test_all_hours_have_valid_condition(self):
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_CLOUDY,
            smoothed=_smoothed(),
        )
        for hf in result:
            assert hf.condition in HA_CONDITIONS

    def test_hours_ahead_sequential(self):
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_CLEAR,
            smoothed=_smoothed(),
        )
        for i, hf in enumerate(result):
            assert hf.hours_ahead == i + 1

    def test_precip_probability_bounded(self):
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_RAINY,
            smoothed=_smoothed(humidity=90.0, dp_dt=-1.5),
        )
        for hf in result:
            assert 0 <= hf.precipitation_probability <= 100

    def test_temperature_reasonable(self):
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_CLEAR,
            smoothed=_smoothed(temperature=22.0),
            predict_temperature=lambda h: 22.0 - 0.5 * h,
        )
        for hf in result:
            assert -50 < hf.temperature < 60


class TestDayNightSwap:
    """Sunny ↔ clear-night swap based on forecast hour."""

    def test_daytime_hours_not_clear_night(self):
        """During daytime, should never show clear-night as clear state."""
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_CLEAR,
            smoothed=_smoothed(is_night=False),
            sunrise_hour=6.0,
            sunset_hour=20.0,
            current_hour=10.0,  # mid-morning
        )
        # First few hours are definitely daytime
        for hf in result[:6]:
            if hf.condition in ("sunny", "clear-night"):
                assert hf.condition == "sunny"

    def test_nighttime_hours_not_sunny(self):
        """During nighttime, clear state should be clear-night."""
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_CLEAR_NIGHT,
            smoothed=_smoothed(is_night=True),
            sunrise_hour=6.0,
            sunset_hour=20.0,
            current_hour=22.0,  # late evening
        )
        # First few hours are nighttime
        for hf in result[:6]:
            if hf.condition in ("sunny", "clear-night"):
                assert hf.condition == "clear-night"


class TestSnowConstraint:
    """No snow above temperature threshold."""

    def test_no_snow_when_warm(self):
        """At 15°C, snow states must have zero probability."""
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_RAINY,
            smoothed=_smoothed(temperature=15.0, humidity=85.0, dp_dt=-1.0),
            predict_temperature=lambda h: 15.0,
        )
        for hf in result:
            assert hf.condition not in ("snowy", "snowy-rainy")

    def test_snow_possible_when_cold(self):
        """At -5°C, snow states should be reachable."""
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_SNOWY,
            smoothed=_smoothed(
                temperature=-5.0, humidity=85.0, wet_bulb=-7.0,
                dp_dt=-0.5, rain_rate=2.0,
            ),
            predict_temperature=lambda h: -5.0,
        )
        conditions = [hf.condition for hf in result]
        # At least some hours should be snow-related
        assert any(c in ("snowy", "snowy-rainy") for c in conditions)


class TestPrecipitationProbability:
    """Precipitation probability reflects wet-state probabilities."""

    def test_clear_sky_low_precip(self):
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_CLEAR,
            smoothed=_smoothed(
                pressure=1025.0, humidity=30.0, dp_dt=0.5,
            ),
        )
        # First hour from clear sky: very low precip probability
        assert result[0].precipitation_probability < 30

    def test_rainy_high_precip(self):
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_RAINY,
            smoothed=_smoothed(
                humidity=90.0, dp_dt=-1.5, rain_rate=3.0,
            ),
        )
        assert result[0].precipitation_probability > 40


class TestPhysicsModelIntegration:
    """Forecaster with physics model callables."""

    def test_temperature_model_used(self):
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_CLEAR,
            smoothed=_smoothed(temperature=20.0),
            predict_temperature=lambda h: 20.0 + h,  # warming
        )
        # Hour 6 should be around 26°C
        assert result[5].temperature > 24

    def test_pressure_model_used(self):
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_CLOUDY,
            smoothed=_smoothed(pressure=1010.0),
            predict_pressure=lambda h: 1010.0 - h * 0.5,  # falling
        )
        assert result[5].pressure < 1010.0

    def test_humidity_model_used(self):
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_CLOUDY,
            smoothed=_smoothed(humidity=60.0),
            predict_humidity=lambda h: min(100, 60.0 + h * 3),  # rising
        )
        assert result[5].humidity > 70


class TestFrontalEvidence:
    """Frontal flags influence forecast."""

    def test_warm_front_increases_rain(self):
        fc = BayesianForecaster()
        # Without warm front
        r_no_front = fc.forecast(
            current_state_idx=S_CLOUDY,
            smoothed=_smoothed(dp_dt=-1.0, humidity=75.0),
        )
        # With warm front
        r_front = fc.forecast(
            current_state_idx=S_CLOUDY,
            smoothed=_smoothed(
                dp_dt=-1.0, humidity=75.0, front_warm=True,
            ),
        )
        # Warm front should increase precip probability
        pp_no = sum(h.precipitation_probability for h in r_no_front[:4])
        pp_yes = sum(h.precipitation_probability for h in r_front[:4])
        assert pp_yes >= pp_no

    def test_cold_front_shifts_conditions(self):
        fc = BayesianForecaster()
        result = fc.forecast(
            current_state_idx=S_CLOUDY,
            smoothed=_smoothed(
                dp_dt=-2.0, d2p_dt2=-0.8,
                dt_dt=-1.5, front_cold=True,
                humidity=70.0, wind_speed=8.0,
            ),
        )
        # Cold front with strong signals should produce some precip probability
        total_precip = sum(h.precipitation_probability for h in result)
        assert total_precip > 0  # at least some precip signal


class TestTransitionMatrix:
    """Transition matrix properties."""

    def test_all_rows_sum_to_one(self):
        fc = BayesianForecaster()
        for i, row in enumerate(fc._T):
            total = sum(row)
            assert abs(total - 1.0) < 1e-6, f"Row {i} sums to {total}"

    def test_matrix_is_square(self):
        fc = BayesianForecaster()
        assert len(fc._T) == NUM_STATES
        for row in fc._T:
            assert len(row) == NUM_STATES

    def test_no_negative_probabilities(self):
        fc = BayesianForecaster()
        for row in fc._T:
            for val in row:
                assert val >= 0.0
