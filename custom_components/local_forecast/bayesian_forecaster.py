"""Bayesian Forecaster — probabilistic weather-state prediction.

Maintains a probability distribution over the 12 HA weather states and
advances it hour-by-hour using:
  1. A Markov transition matrix  (climatological prior)
  2. Bayesian evidence updates   (sensor-derived likelihoods)
  3. Physical constraints         (temperature → precip type, day/night)

The output is a probability vector per forecast hour.  The weather entity
picks argmax for the condition icon, and uses the probability of
precipitation states for precipitation_probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .const import (
    FOG_DEW_DEPRESSION,
    FORECAST_HOURS,
    HA_CONDITIONS,
    NUM_STATES,
    RAIN_HEAVY,
    RAIN_LIGHT,
    S_CLEAR,
    S_CLEAR_NIGHT,
    S_CLOUDY,
    S_EXCEPTIONAL,
    S_FOG,
    S_LIGHTNING_RAINY,
    S_PARTLY_CLOUDY,
    S_POURING,
    S_RAINY,
    S_SNOWY,
    S_SNOWY_RAINY,
    S_WINDY,
    WET_BULB_MIX_UPPER,
    WET_BULB_SNOW,
)
from .state_estimator import SmoothedState


# ---------------------------------------------------------------------------
#  Hourly forecast output
# ---------------------------------------------------------------------------

@dataclass
class HourForecast:
    """One hour of forecast output."""

    hours_ahead: int
    condition: str              # HA condition string (drives the icon)
    temperature: float          # °C
    humidity: float             # %
    pressure: float             # hPa
    precipitation_probability: int   # 0-100
    precipitation_amount: float      # mm expected in this hour
    wind_speed: float           # m/s
    wind_bearing: float         # degrees

    # Full probability vector (for diagnostics / attributes)
    state_probs: list[float] | None = None


# ---------------------------------------------------------------------------
#  Transition matrix
# ---------------------------------------------------------------------------

def _build_transition_matrix() -> list[list[float]]:
    """Build a 12×12 Markov transition matrix.

    Each row sums to 1.0.  Entry T[i][j] = P(next_hour = j | current = i).

    This encodes the physical reality that weather evolves gradually:
    - Clear skies don't jump to pouring in one hour
    - Snow doesn't appear at 25 °C
    - Fog tends to persist for hours
    - Frontal passages drive rapid transitions along specific paths

    These are *prior* probabilities, overridden by Bayesian evidence.
    """
    T = [[0.0] * NUM_STATES for _ in range(NUM_STATES)]

    # Helper: distribute probability across targets
    def _row(src: int, targets: dict[int, float]) -> None:
        total = sum(targets.values())
        for dst, p in targets.items():
            T[src][dst] = p / total  # normalise

    # S_CLEAR (sunny)
    _row(S_CLEAR, {
        S_CLEAR: 0.70, S_CLEAR_NIGHT: 0.05, S_PARTLY_CLOUDY: 0.20,
        S_CLOUDY: 0.03, S_FOG: 0.01, S_WINDY: 0.01,
    })

    # S_CLEAR_NIGHT
    _row(S_CLEAR_NIGHT, {
        S_CLEAR_NIGHT: 0.65, S_CLEAR: 0.10, S_PARTLY_CLOUDY: 0.15,
        S_CLOUDY: 0.04, S_FOG: 0.05, S_WINDY: 0.01,
    })

    # S_PARTLY_CLOUDY
    _row(S_PARTLY_CLOUDY, {
        S_CLEAR: 0.15, S_CLEAR_NIGHT: 0.05, S_PARTLY_CLOUDY: 0.45,
        S_CLOUDY: 0.25, S_RAINY: 0.05, S_FOG: 0.02, S_WINDY: 0.03,
    })

    # S_CLOUDY
    _row(S_CLOUDY, {
        S_PARTLY_CLOUDY: 0.15, S_CLOUDY: 0.45, S_RAINY: 0.20,
        S_FOG: 0.05, S_POURING: 0.05, S_SNOWY: 0.03,
        S_SNOWY_RAINY: 0.02, S_LIGHTNING_RAINY: 0.02, S_WINDY: 0.03,
    })

    # S_FOG
    _row(S_FOG, {
        S_FOG: 0.60, S_CLOUDY: 0.20, S_PARTLY_CLOUDY: 0.10,
        S_CLEAR: 0.05, S_CLEAR_NIGHT: 0.05,
    })

    # S_RAINY
    _row(S_RAINY, {
        S_RAINY: 0.50, S_CLOUDY: 0.20, S_POURING: 0.10,
        S_PARTLY_CLOUDY: 0.05, S_SNOWY_RAINY: 0.05,
        S_LIGHTNING_RAINY: 0.05, S_SNOWY: 0.03, S_FOG: 0.02,
    })

    # S_POURING
    _row(S_POURING, {
        S_POURING: 0.40, S_RAINY: 0.30, S_LIGHTNING_RAINY: 0.10,
        S_CLOUDY: 0.10, S_SNOWY_RAINY: 0.05, S_SNOWY: 0.05,
    })

    # S_SNOWY
    _row(S_SNOWY, {
        S_SNOWY: 0.55, S_SNOWY_RAINY: 0.15, S_CLOUDY: 0.15,
        S_RAINY: 0.05, S_PARTLY_CLOUDY: 0.05, S_FOG: 0.05,
    })

    # S_SNOWY_RAINY (sleet)
    _row(S_SNOWY_RAINY, {
        S_SNOWY_RAINY: 0.35, S_SNOWY: 0.20, S_RAINY: 0.20,
        S_CLOUDY: 0.15, S_POURING: 0.05, S_FOG: 0.05,
    })

    # S_LIGHTNING_RAINY
    _row(S_LIGHTNING_RAINY, {
        S_LIGHTNING_RAINY: 0.30, S_POURING: 0.25, S_RAINY: 0.25,
        S_CLOUDY: 0.10, S_WINDY: 0.05, S_PARTLY_CLOUDY: 0.05,
    })

    # S_WINDY
    _row(S_WINDY, {
        S_WINDY: 0.40, S_PARTLY_CLOUDY: 0.20, S_CLOUDY: 0.20,
        S_CLEAR: 0.10, S_RAINY: 0.05, S_CLEAR_NIGHT: 0.05,
    })

    # S_EXCEPTIONAL
    _row(S_EXCEPTIONAL, {
        S_EXCEPTIONAL: 0.30, S_POURING: 0.25, S_LIGHTNING_RAINY: 0.20,
        S_RAINY: 0.15, S_WINDY: 0.10,
    })

    return T


TRANSITION_MATRIX: list[list[float]] = _build_transition_matrix()


# ---------------------------------------------------------------------------
#  Bayesian Forecaster
# ---------------------------------------------------------------------------

class BayesianForecaster:
    """Advance a probability vector through time with evidence updates."""

    def __init__(self) -> None:
        self._T = TRANSITION_MATRIX

    def forecast(
        self,
        current_state_idx: int,
        smoothed: SmoothedState,
        *,
        hours: int = FORECAST_HOURS,
        sunrise_hour: float = 6.0,
        sunset_hour: float = 20.0,
        current_hour: float = 12.0,
        predict_temperature: object | None = None,
        predict_pressure: object | None = None,
        predict_humidity: object | None = None,
    ) -> list[HourForecast]:
        """Produce hourly forecasts from 1..hours.

        Args:
            current_state_idx: Index from StateEstimator.classify()
            smoothed: Current SmoothedState (for evidence calculation)
            hours: Forecast horizon
            sunrise_hour / sunset_hour: Decimal hours UTC for day/night
            current_hour: Current decimal hour UTC
            predict_temperature: Optional callable(hours_ahead) → °C
            predict_pressure: Optional callable(hours_ahead) → hPa
            predict_humidity: Optional callable(hours_ahead) → %

        Returns:
            List of HourForecast, one per hour ahead.
        """
        # Start with a peaked distribution at the observed state
        prob = [0.0] * NUM_STATES
        prob[current_state_idx] = 1.0

        results: list[HourForecast] = []

        for h in range(1, hours + 1):
            # --- Step 1: Markov transition (prior) ---
            prob = self._transition(prob)

            # --- Step 2: Predict physical quantities ---
            forecast_hour = (current_hour + h) % 24
            is_night = not (sunrise_hour <= forecast_hour < sunset_hour)

            if predict_temperature is not None:
                temp_h = predict_temperature(h)
            else:
                temp_h = smoothed.temperature + smoothed.dt_dt * min(h, 3)

            if predict_pressure is not None:
                pres_h = predict_pressure(h)
            else:
                # Damped linear extrapolation
                damp = 0.9 ** h
                pres_h = smoothed.pressure + smoothed.dp_dt * h * damp

            if predict_humidity is not None:
                hum_h = predict_humidity(h)
            else:
                hum_h = smoothed.humidity

            # --- Step 3: Bayesian evidence update ---
            prob = self._apply_evidence(
                prob, smoothed, temp_h, pres_h, hum_h, is_night, h
            )

            # --- Step 4: Physical constraints (hard overrides) ---
            prob = self._apply_constraints(prob, temp_h, is_night)

            # --- Step 5: Normalise ---
            prob = self._normalise(prob)

            # --- Output ---
            best = max(range(NUM_STATES), key=lambda i: prob[i])
            condition = HA_CONDITIONS[best]

            # Precipitation probability = sum of all wet states
            precip_prob = sum(
                prob[i]
                for i in (S_RAINY, S_POURING, S_SNOWY, S_SNOWY_RAINY, S_LIGHTNING_RAINY)
            )

            # Expected precipitation (rough: map probability to mm/h)
            precip_mm = self._estimate_precip_amount(prob, smoothed.rain_rate)

            results.append(HourForecast(
                hours_ahead=h,
                condition=condition,
                temperature=round(temp_h, 1),
                humidity=round(max(1, min(100, hum_h)), 0),
                pressure=round(pres_h, 1),
                precipitation_probability=round(min(100, precip_prob * 100)),
                precipitation_amount=round(precip_mm, 1),
                wind_speed=round(smoothed.wind_speed, 1),
                wind_bearing=round(smoothed.wind_direction),
                state_probs=[round(p, 3) for p in prob],
            ))

        return results

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _transition(self, prob: list[float]) -> list[float]:
        """Multiply probability vector by transition matrix."""
        new = [0.0] * NUM_STATES
        for j in range(NUM_STATES):
            for i in range(NUM_STATES):
                new[j] += prob[i] * self._T[i][j]
        return new

    def _apply_evidence(
        self,
        prob: list[float],
        s: SmoothedState,
        temp: float,
        pressure: float,
        humidity: float,
        is_night: bool,
        hours_ahead: int,
    ) -> list[float]:
        """Bayesian update: multiply by likelihood of each state given evidence.

        Evidence signals (each is a soft multiplier, not a hard override):
          - Pressure tendency  → favours improving or deteriorating states
          - Pressure acceleration → frontal approach
          - Humidity level → cloud/precip states
          - Dew-point depression convergence → imminent precip/fog
          - Frontal flags → specific transition patterns
          - Wind speed → windy state

        The strength of evidence decays with forecast horizon because
        sensor readings become less predictive further out.
        """
        likelihood = [1.0] * NUM_STATES
        # Evidence weight decays with time (half-life ~4 h)
        w = math.exp(-hours_ahead / 6.0)

        # --- Pressure tendency ---
        if s.dp_dt < -2.0:
            # Rapidly falling → boost rainy/pouring/lightning
            for idx in (S_RAINY, S_POURING, S_LIGHTNING_RAINY):
                likelihood[idx] *= 1.0 + 0.8 * w
            for idx in (S_CLEAR, S_CLEAR_NIGHT, S_PARTLY_CLOUDY):
                likelihood[idx] *= max(0.2, 1.0 - 0.5 * w)
        elif s.dp_dt < -1.0:
            # Moderately falling → boost cloudy/rainy
            for idx in (S_CLOUDY, S_RAINY):
                likelihood[idx] *= 1.0 + 0.5 * w
        elif s.dp_dt > 2.0:
            # Rising fast → boost clearing
            for idx in (S_CLEAR, S_CLEAR_NIGHT, S_PARTLY_CLOUDY):
                likelihood[idx] *= 1.0 + 0.6 * w
            for idx in (S_RAINY, S_POURING, S_LIGHTNING_RAINY):
                likelihood[idx] *= max(0.3, 1.0 - 0.4 * w)
        elif s.dp_dt > 1.0:
            # Moderately rising → slight clearing bias
            for idx in (S_PARTLY_CLOUDY, S_CLEAR, S_CLEAR_NIGHT):
                likelihood[idx] *= 1.0 + 0.3 * w

        # --- Pressure acceleration (frontal approach) ---
        if s.d2p_dt2 < -0.5:
            # Accelerating pressure drop → storm approaching
            for idx in (S_POURING, S_LIGHTNING_RAINY, S_EXCEPTIONAL):
                likelihood[idx] *= 1.0 + 0.6 * w

        # --- Humidity ---
        if humidity > 85:
            for idx in (S_RAINY, S_POURING, S_FOG, S_SNOWY, S_SNOWY_RAINY):
                likelihood[idx] *= 1.0 + 0.4 * w
            for idx in (S_CLEAR, S_CLEAR_NIGHT):
                likelihood[idx] *= max(0.3, 1.0 - 0.4 * w)
        elif humidity < 40:
            for idx in (S_CLEAR, S_CLEAR_NIGHT, S_PARTLY_CLOUDY):
                likelihood[idx] *= 1.0 + 0.3 * w
            for idx in (S_RAINY, S_POURING, S_FOG, S_SNOWY, S_SNOWY_RAINY):
                likelihood[idx] *= max(0.2, 1.0 - 0.5 * w)

        # --- Dew-point depression convergence → imminent precip/fog ---
        if s.dd_trend < -1.0 and s.dew_depression < 4.0:
            for idx in (S_RAINY, S_FOG, S_SNOWY, S_SNOWY_RAINY):
                likelihood[idx] *= 1.0 + 0.5 * w

        # --- Warm front flag ---
        if s.front_warm:
            # Classic warm-front sequence: cloud thickening → steady rain
            for idx in (S_CLOUDY, S_RAINY):
                likelihood[idx] *= 1.0 + 0.6 * w

        # --- Cold front flag ---
        if s.front_cold:
            # Post-frontal clearing but initially blustery
            for idx in (S_WINDY, S_PARTLY_CLOUDY):
                likelihood[idx] *= 1.0 + 0.5 * w
            # Dying rain
            likelihood[S_RAINY] *= max(0.4, 1.0 - 0.3 * w)

        # --- Occluded front ---
        if s.front_occluded:
            for idx in (S_RAINY, S_POURING, S_CLOUDY):
                likelihood[idx] *= 1.0 + 0.5 * w

        # --- Wind ---
        if s.wind_speed > 10.0:
            likelihood[S_WINDY] *= 1.0 + 0.4 * w
            likelihood[S_FOG] *= max(0.2, 1.0 - 0.6 * w)

        # Multiply prior × likelihood
        return [p * l for p, l in zip(prob, likelihood)]

    def _apply_constraints(
        self, prob: list[float], temp: float, is_night: bool
    ) -> list[float]:
        """Hard physical constraints — zero out impossible states.

        These are not soft biases, they are physical laws:
          - Cannot snow when wet-bulb > 4 °C (even generous margin)
          - Cannot be "sunny" at night
          - Cannot be "clear-night" during day
        """
        out = list(prob)

        # --- Temperature: kill snow if way too warm ---
        if temp > 6.0:
            out[S_SNOWY] = 0.0
            out[S_SNOWY_RAINY] = 0.0

        # --- Day/night swap ---
        if is_night:
            # Move any "sunny" probability to "clear-night"
            out[S_CLEAR_NIGHT] += out[S_CLEAR]
            out[S_CLEAR] = 0.0
        else:
            # Move any "clear-night" probability to "sunny"
            out[S_CLEAR] += out[S_CLEAR_NIGHT]
            out[S_CLEAR_NIGHT] = 0.0

        return out

    def _normalise(self, prob: list[float]) -> list[float]:
        total = sum(prob)
        if total < 1e-10:
            return [1.0 / NUM_STATES] * NUM_STATES
        return [p / total for p in prob]

    def _estimate_precip_amount(
        self, prob: list[float], current_rain_rate: float
    ) -> float:
        """Rough expected precipitation in mm for this hour."""
        p_light = prob[S_RAINY] + prob[S_SNOWY] + prob[S_SNOWY_RAINY]
        p_heavy = prob[S_POURING] + prob[S_LIGHTNING_RAINY]

        if current_rain_rate > 0.5:
            return max(0.0, current_rain_rate) * (p_light + p_heavy)
        return p_light * 1.5 + p_heavy * 8.0
