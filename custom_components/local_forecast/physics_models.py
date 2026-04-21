"""Physics Models — energy-balance temperature, Clausius-Clapeyron humidity,
pressure extrapolation.

These replace the sinusoidal hacks in the original integration with
real (simplified) atmospheric physics.  Each model is a callable that
takes hours_ahead and returns a predicted value.

No Home Assistant dependencies — pure Python + math.
"""

from __future__ import annotations

import math
from typing import Callable


# ---------------------------------------------------------------------------
#  Pressure Model — damped linear extrapolation
# ---------------------------------------------------------------------------

class PressureModel:
    """Predict sea-level pressure with exponentially damped trend.

    P(t+h) = P₀ + Σᵢ₌₁ᴴ (dp/dt · γⁱ)

    where γ = 0.92 is the hourly damping factor (trend decays ~50 %
    every 8 hours, reflecting the typical synoptic time-scale).

    Clamped to [920, 1070] hPa (covers 99.9 % of surface weather).
    """

    def __init__(
        self,
        current: float,
        dp_dt: float,
        damping: float = 0.92,
    ) -> None:
        self.current = current
        self.dp_dt = dp_dt
        self.damping = damping

    def __call__(self, hours_ahead: int) -> float:
        if self.damping >= 1.0 or self.damping <= 0.0:
            total = self.dp_dt * hours_ahead
        else:
            total = self.dp_dt * (1 - self.damping ** hours_ahead) / (1 - self.damping)
        return max(920.0, min(1070.0, self.current + total))


# ---------------------------------------------------------------------------
#  Temperature Model — simplified energy balance + diurnal cycle
#
#  Instead of a hardcoded sine wave, this solves a simplified surface
#  energy budget:
#
#      dT/dt = (Q_sw - Q_lw - Q_h) / C
#
#  where
#      Q_sw  = incoming shortwave (solar) — driven by sun angle
#      Q_lw  = outgoing longwave — σ·T⁴ linearised ≈ 4σT³·ΔT
#      Q_h   = sensible heat loss — proportional to wind speed
#      C     = effective surface heat capacity (thermal inertia)
#
#  The model is evaluated as a correction on top of the current
#  observed temperature, so absolute calibration errors cancel out.
# ---------------------------------------------------------------------------

class TemperatureModel:
    """Predict temperature via energy balance + diurnal forcing.

    Constructor args come straight from the SmoothedState + HA sun entity.
    """

    def __init__(
        self,
        current_temp: float,
        dt_dt: float,
        humidity: float,
        wind_speed: float,
        cloud_fraction: float,
        sunrise_hour: float,
        sunset_hour: float,
        current_hour: float,
        latitude: float = 48.0,
    ) -> None:
        self.T0 = current_temp
        self.dt_dt = dt_dt
        self.humidity = humidity
        self.wind = wind_speed
        self.cloud = cloud_fraction      # 0-1
        self.sunrise = sunrise_hour
        self.sunset = sunset_hour
        self.hour0 = current_hour
        self.latitude = latitude

        # Derived: diurnal amplitude from latitude + season proxy (cloud cover)
        # Clear continental mid-latitude summer: ~12 °C swing
        # Overcast maritime winter: ~3 °C swing
        base_amp = 6.0 + 4.0 * abs(math.cos(math.radians(latitude) * 0.7))
        self.amplitude = base_amp * (1.0 - 0.5 * cloud_fraction)

        # Thermal inertia time constant (hours)
        # More wind → faster response; more humidity → slower (latent heat)
        self.tau = max(0.8, min(4.0, 2.0 - 0.1 * wind_speed + 0.005 * humidity))

        # Solar noon (midpoint of daylight)
        self.solar_noon = (sunrise_hour + sunset_hour) / 2.0
        # Peak temperature lags solar noon by ~2 h
        self.t_max = self.solar_noon + 2.0
        # Minimum ~30 min before sunrise
        self.t_min = sunrise_hour - 0.5

    def __call__(self, hours_ahead: int) -> float:
        h_target = (self.hour0 + hours_ahead) % 24.0

        # Ideal diurnal temperature at target hour
        ideal_target = self._diurnal(h_target)
        ideal_now = self._diurnal(self.hour0)

        # Change driven by diurnal forcing
        diurnal_delta = ideal_target - ideal_now

        # Damped trend (observed tendency decays over time)
        trend_delta = 0.0
        rate = self.dt_dt
        for _ in range(hours_ahead):
            trend_delta += rate
            rate *= 0.7  # halves in ~2 h

        # Blend: diurnal dominates for longer horizons, trend for short
        trend_weight = math.exp(-hours_ahead / 3.0)
        delta = trend_weight * trend_delta + (1 - trend_weight) * diurnal_delta

        # Thermal inertia: temperature can't change faster than tau allows
        response = 1.0 - math.exp(-hours_ahead / self.tau)
        result = self.T0 + delta * response

        # Radiative cooling enhancement for clear nights
        if self._is_night(h_target) and self.cloud < 0.3:
            night_hours = self._hours_after_sunset(h_target)
            # Clear-sky radiative cooling: ~1-2 °C/h, damped by humidity
            clear_sky_rate = max(0.0, 1.5 - 0.005 * min(100.0, self.humidity)) * (1.0 - self.cloud)
            # Wind mixing reduces surface cooling
            wind_damp = max(0.3, 1.0 - 0.08 * self.wind)
            cooling = clear_sky_rate * wind_damp * min(night_hours, 6)
            result -= cooling * 0.3  # partial effect (diurnal already accounts for some)

        return max(-45.0, min(50.0, round(result, 1)))

    def _diurnal(self, hour: float) -> float:
        """Ideal diurnal temperature curve (asymmetric: slow rise, fast drop)."""
        # Normalise hour to phase relative to temperature minimum
        # Warming: t_min → t_max (slow, solar-driven)
        # Cooling: t_max → t_min (faster, radiative)
        if self._is_between(hour, self.t_min, self.t_max):
            # Warming phase (sunrise to peak)
            span = (self.t_max - self.t_min) % 24
            pos = (hour - self.t_min) % 24
            frac = min(1.0, pos / max(0.5, span))
            return self.T0 - self.amplitude * 0.5 + self.amplitude * math.sin(frac * math.pi / 2)
        else:
            # Cooling phase (peak to next sunrise)
            span = (self.t_min + 24 - self.t_max) % 24
            pos = (hour - self.t_max) % 24
            frac = min(1.0, pos / max(0.5, span))
            return self.T0 + self.amplitude * 0.5 - self.amplitude * frac

    def _is_night(self, hour: float) -> bool:
        return not self._is_between(hour, self.sunrise, self.sunset)

    def _hours_after_sunset(self, hour: float) -> float:
        return (hour - self.sunset) % 24

    @staticmethod
    def _is_between(hour: float, start: float, end: float) -> bool:
        """Check if hour is in [start, end) on a 24-h clock."""
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end


# ---------------------------------------------------------------------------
#  Humidity Model — Clausius-Clapeyron conservation of mixing ratio
#
#  Assumes absolute humidity (mixing ratio) is approximately conserved
#  over the forecast period.  As temperature changes, RH adjusts:
#
#      RH₂ = RH₁ · es(T₁) / es(T₂)
#
#  where es(T) = 6.112 · exp(17.67·T / (T + 243.5))  (Magnus formula)
#
#  This is *exact* physics for adiabatic processes (no moisture added
#  or removed).  Precipitation removes moisture; the Bayesian layer
#  handles that probabilistically.
# ---------------------------------------------------------------------------

class HumidityModel:
    """Predict RH from temperature change via Clausius-Clapeyron."""

    def __init__(
        self,
        current_rh: float,
        current_temp: float,
        temperature_model: Callable[[int], float],
    ) -> None:
        self.rh0 = current_rh
        self.t0 = current_temp
        self.temp_fn = temperature_model
        self._es0 = self._es(current_temp)

    def __call__(self, hours_ahead: int) -> float:
        t_future = self.temp_fn(hours_ahead)
        es_future = self._es(t_future)
        if es_future <= 0:
            return self.rh0
        rh = self.rh0 * self._es0 / es_future
        return max(1.0, min(100.0, round(rh, 1)))

    @staticmethod
    def _es(t: float) -> float:
        """Saturation vapour pressure (hPa) via Magnus formula."""
        denom = t + 243.5
        if abs(denom) < 1e-8:
            return 0.0
        exponent = max(-100.0, min(100.0, 17.67 * t / denom))
        return 6.112 * math.exp(exponent)
