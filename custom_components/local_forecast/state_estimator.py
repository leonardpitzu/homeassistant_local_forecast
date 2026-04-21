"""State Estimator — sensor fusion, trend analysis and current-state classification.

Reads raw sensor values, applies a 1-D Kalman smoother per channel,
computes derivatives (dp/dt, d²p/dt², dT/dt, dew-point depression trend),
detects frontal passages and classifies the current weather state into
one of the 12 HA condition indices.

No Home Assistant dependencies — pure Python + math.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .const import (
    FOG_DEW_DEPRESSION,
    FOG_MAX_WIND,
    HISTORY_MAX_RECORDS,
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
    STORM_HUMIDITY,
    STORM_PRESSURE_DROP,
    STORM_WIND,
    WET_BULB_MIX_UPPER,
    WET_BULB_SNOW,
    WIND_STRONG,
)


# ---------------------------------------------------------------------------
#  Data containers
# ---------------------------------------------------------------------------

@dataclass
class SensorReading:
    """Single snapshot from all available sensors."""

    timestamp: float                          # epoch seconds
    pressure_hpa: float                       # sea-level pressure
    temperature_c: float
    humidity_pct: Optional[float] = None      # 0-100
    wind_speed_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    solar_radiation_wm2: Optional[float] = None
    rain_rate_mmh: Optional[float] = None


@dataclass
class SmoothedState:
    """Kalman-filtered state + derived quantities."""

    pressure: float = 1013.25
    temperature: float = 15.0
    humidity: float = 50.0
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    solar_radiation: float = 0.0
    rain_rate: float = 0.0

    # Derivatives (per hour)
    dp_dt: float = 0.0           # hPa/h  — pressure tendency
    d2p_dt2: float = 0.0         # hPa/h² — pressure acceleration
    dt_dt: float = 0.0           # °C/h
    dh_dt: float = 0.0           # %/h

    # Moisture
    dew_point: float = 10.0
    dew_depression: float = 5.0  # T − Td
    dd_trend: float = 0.0        # dew-depression change (°C/h)
    wet_bulb: float = 10.0       # Tw for precip-type decisions

    # Frontal flags
    front_warm: bool = False
    front_cold: bool = False
    front_occluded: bool = False

    # Day/night (set by the HA layer before calling classify)
    is_night: bool = False


@dataclass
class _KalmanChannel:
    """Per-variable 1-D Kalman state."""

    x: float = 0.0
    p: float = 1.0
    q: float = 0.01             # process noise
    r: float = 0.1              # measurement noise


# ---------------------------------------------------------------------------
#  State Estimator
# ---------------------------------------------------------------------------

class StateEstimator:
    """Fuses sensor readings into a clean state with trends and frontal flags."""

    def __init__(self, *, history_size: int = HISTORY_MAX_RECORDS) -> None:
        self._history: deque[SensorReading] = deque(maxlen=history_size)
        self._kf: dict[str, _KalmanChannel] = {
            "pressure":    _KalmanChannel(q=0.005, r=0.15),
            "temperature": _KalmanChannel(q=0.02,  r=0.3),
            "humidity":    _KalmanChannel(q=0.05,  r=1.0),
            "wind_speed":  _KalmanChannel(q=0.1,   r=0.5),
        }
        self._state = SmoothedState()
        self._prev_dp_dt: Optional[float] = None
        self._prev_dd: Optional[float] = None
        self._wind_history: deque[tuple[float, float]] = deque(maxlen=60)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def update(self, reading: SensorReading) -> SmoothedState:
        """Ingest one reading, return updated smoothed state."""
        self._history.append(reading)

        # --- Kalman update per channel ---
        self._state.pressure = self._kalman("pressure", reading.pressure_hpa)
        self._state.temperature = self._kalman("temperature", reading.temperature_c)

        if reading.humidity_pct is not None:
            self._state.humidity = self._kalman("humidity", reading.humidity_pct)
        if reading.wind_speed_ms is not None:
            self._state.wind_speed = self._kalman("wind_speed", reading.wind_speed_ms)
        if reading.wind_direction_deg is not None:
            self._state.wind_direction = reading.wind_direction_deg
            self._wind_history.append(
                (reading.timestamp, reading.wind_direction_deg)
            )
        if reading.solar_radiation_wm2 is not None:
            self._state.solar_radiation = reading.solar_radiation_wm2
        if reading.rain_rate_mmh is not None:
            self._state.rain_rate = max(0.0, reading.rain_rate_mmh)

        # --- Derived quantities ---
        self._compute_trends()
        self._compute_moisture()
        self._detect_fronts()
        return self._state

    @property
    def state(self) -> SmoothedState:
        return self._state

    @property
    def history(self) -> list[SensorReading]:
        return list(self._history)

    # ------------------------------------------------------------------
    #  Classify current weather into one of 12 HA condition indices
    # ------------------------------------------------------------------

    def classify(self) -> int:
        """Return the state index that best describes current conditions.

        Priority chain (highest first):
          1. Active heavy precipitation  → pouring / snowy / snowy-rainy
          2. Active light precipitation  → rainy / snowy / snowy-rainy
          3. Thunderstorm proxy          → lightning-rainy
          4. Fog                         → fog
          5. Exceptional (bomb cyclone)  → exceptional
          6. Strong wind                 → windy
          7. Cloud cover                 → clear / partly cloudy / cloudy
          8. Day vs night                → sunny vs clear-night
        """
        s = self._state

        # --- 1 & 2: Active precipitation (rain sensor) ---
        if s.rain_rate >= RAIN_LIGHT:
            return self._precip_state(s)

        # --- Also check: is precipitation *imminent*? ---
        # Dew depression closing fast + pressure falling → precip about to start
        # (but don't show rain icon yet — only when sensor confirms it)

        # --- 3: Thunderstorm proxy ---
        if (
            s.dp_dt < STORM_PRESSURE_DROP
            and s.humidity > STORM_HUMIDITY
            and s.wind_speed > STORM_WIND
        ):
            return S_LIGHTNING_RAINY

        # --- 4: Fog ---
        if s.dew_depression < FOG_DEW_DEPRESSION and s.wind_speed < FOG_MAX_WIND:
            return S_FOG

        # --- 5: Exceptional (bomb cyclone: pressure drop > 24 hPa / 24h) ---
        if s.dp_dt < -4.0:  # ~24 hPa/6h extrapolated
            return S_EXCEPTIONAL

        # --- 6: Strong wind (only if no precipitation) ---
        if s.wind_speed >= WIND_STRONG:
            return S_WINDY

        # --- 7 & 8: Cloud cover + day/night ---
        cloud = self._estimate_cloud_fraction()

        if cloud < 0.15:
            return S_CLEAR_NIGHT if s.is_night else S_CLEAR
        if cloud < 0.50:
            return S_PARTLY_CLOUDY
        return S_CLOUDY

    # ------------------------------------------------------------------
    #  Precipitation type from wet-bulb temperature
    # ------------------------------------------------------------------

    def _precip_state(self, s: SmoothedState) -> int:
        """Map rain rate + wet-bulb temp to the correct precip icon."""
        tw = s.wet_bulb

        if tw < WET_BULB_SNOW:
            # Frozen → snowflake icon
            return S_SNOWY

        if tw < WET_BULB_MIX_UPPER:
            # Transition zone → mixed snow/rain icon
            return S_SNOWY_RAINY

        # Liquid
        if s.rain_rate >= RAIN_HEAVY:
            return S_POURING
        return S_RAINY

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _kalman(self, channel: str, measurement: float) -> float:
        k = self._kf[channel]
        if not math.isfinite(measurement):
            return k.x
        k.p += k.q
        denom = k.p + k.r
        if denom <= 0:
            denom = 1e-6
        gain = min(1.0, max(0.0, k.p / denom))
        k.x += gain * (measurement - k.x)
        k.p *= 1.0 - gain
        return k.x

    def _compute_trends(self) -> None:
        """Compute dp/dt, d²p/dt², dT/dt, dH/dt from history ring buffer."""
        if len(self._history) < 2:
            return

        now = self._history[-1]
        t_now = now.timestamp

        ref = self._find_nearest(t_now - 3600)  # ~1 h ago
        if ref is not None:
            dt_h = max(0.1, (t_now - ref.timestamp) / 3600)
            self._state.dp_dt = (self._state.pressure - ref.pressure_hpa) / dt_h
            self._state.dt_dt = (self._state.temperature - ref.temperature_c) / dt_h
            if ref.humidity_pct is not None:
                self._state.dh_dt = (self._state.humidity - ref.humidity_pct) / dt_h

        # Pressure acceleration from 3-h window
        ref3 = self._find_nearest(t_now - 10800)
        if ref3 is not None and self._prev_dp_dt is not None:
            self._state.d2p_dt2 = self._state.dp_dt - self._prev_dp_dt
        self._prev_dp_dt = self._state.dp_dt

    def _compute_moisture(self) -> None:
        """Dew point (Magnus), wet-bulb (Stull 2011), depression trend."""
        T = self._state.temperature
        RH = max(1.0, min(100.0, self._state.humidity))

        # --- Dew point (Magnus) ---
        a, b = 17.27, 237.7
        alpha = (a * T) / (b + T) + math.log(RH / 100.0)
        if abs(a - alpha) < 1e-8:
            Td = T  # Saturated air
        else:
            Td = (b * alpha) / (a - alpha)

        old_dd = self._state.dew_depression
        self._state.dew_point = round(Td, 1)
        self._state.dew_depression = round(T - Td, 1)

        # --- Wet-bulb (Stull 2011, ±0.3 °C for normal meteo range) ---
        wb_raw = (
            T * math.atan(0.151977 * math.sqrt(RH + 8.313659))
            + math.atan(T + RH)
            - math.atan(RH - 1.676331)
            + 0.00391838 * (RH ** 1.5) * math.atan(0.023101 * RH)
            - 4.686035
        )
        self._state.wet_bulb = round(max(Td, min(T, wb_raw)), 1)

        # --- Depression trend (°C/h) ---
        if self._prev_dd is not None and len(self._history) >= 2:
            dt = self._history[-1].timestamp - self._history[-2].timestamp
            if dt > 0:
                self._state.dd_trend = (
                    (self._state.dew_depression - self._prev_dd) / (dt / 3600)
                )
        self._prev_dd = self._state.dew_depression

    def _detect_fronts(self) -> None:
        """Detect warm / cold / occluded frontal signatures."""
        s = self._state
        ws = self._wind_shift_rate()

        # Warm front: steady pressure fall + backing wind + rising humidity
        s.front_warm = (
            s.dp_dt < -1.0 and s.dh_dt > 2.0 and ws < -10.0
        )

        # Cold front: pressure trough (accelerating up) + temp drop + veering
        s.front_cold = (
            s.d2p_dt2 > 0.5 and s.dt_dt < -1.0 and ws > 15.0
        )

        # Occluded: strong pressure fall + big wind shift + narrow depression
        s.front_occluded = (
            s.dp_dt < -2.0 and abs(ws) > 20.0 and s.dew_depression < 2.0
        )

    def _wind_shift_rate(self) -> float:
        """Degrees/hour change in wind direction.  Positive = veering (CW)."""
        if len(self._wind_history) < 2:
            return 0.0
        oldest = self._wind_history[0]
        newest = self._wind_history[-1]
        dt_h = (newest[0] - oldest[0]) / 3600
        if dt_h < 0.1:
            return 0.0
        diff = (newest[1] - oldest[1] + 180) % 360 - 180  # signed shortest arc
        return max(-180.0, min(180.0, diff / dt_h))

    def _find_nearest(self, target_ts: float) -> Optional[SensorReading]:
        best: Optional[SensorReading] = None
        best_diff = float("inf")
        for r in self._history:
            d = abs(r.timestamp - target_ts)
            if d < best_diff:
                best_diff = d
                best = r
        if best_diff > 1800:  # reject if >30 min from target
            return None
        return best

    def _estimate_cloud_fraction(self) -> float:
        """Blend solar-radiation and humidity signals into 0-1 cloud fraction."""
        sol = self._state.solar_radiation
        hum = self._state.humidity

        # Solar-based estimate (only meaningful during daytime)
        if sol > 10.0:
            # Crude clear-sky ceiling — good enough for fraction estimate.
            # A proper model would use latitude+time, but we only need
            # "is it significantly dimmer than it should be?"
            ratio = min(1.0, sol / 800.0)
            solar_cloud = 1.0 - ratio
        else:
            solar_cloud = None

        # Humidity-based estimate (works day and night)
        if hum < 40:
            hum_cloud = 0.05
        elif hum < 60:
            hum_cloud = 0.05 + (hum - 40) / 20 * 0.25
        elif hum < 75:
            hum_cloud = 0.30 + (hum - 60) / 15 * 0.30
        elif hum < 90:
            hum_cloud = 0.60 + (hum - 75) / 15 * 0.25
        else:
            hum_cloud = 0.85 + (hum - 90) / 10 * 0.15

        if solar_cloud is not None:
            return max(0.0, min(1.0, 0.7 * solar_cloud + 0.3 * hum_cloud))
        return max(0.0, min(1.0, hum_cloud))
