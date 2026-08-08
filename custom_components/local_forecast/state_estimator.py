"""State Estimator — sensor fusion, trend analysis and current-state classification.

Reads raw sensor values, applies a 1-D Kalman smoother per channel,
computes derivatives (dp/dt, d²p/dt², dT/dt, dew-point depression trend),
detects frontal passages and classifies the current weather state into
one of the 12 HA condition indices.

No Home Assistant dependencies — pure Python + math.
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass
import math

from .const import (
    FOG_DEW_DEPRESSION,
    FOG_MAX_WIND,
    HISTORY_MAX_RECORDS,
    HISTORY_SECONDS,
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
from .tide import tide_hpa_at

# ---------------------------------------------------------------------------
#  Data containers
# ---------------------------------------------------------------------------


@dataclass
class SensorReading:
    """Single snapshot from all available sensors."""

    timestamp: float  # epoch seconds
    pressure_hpa: float  # sea-level pressure
    temperature_c: float
    humidity_pct: float | None = None  # 0-100
    wind_speed_ms: float | None = None
    wind_direction_deg: float | None = None
    solar_radiation_wm2: float | None = None
    rain_rate_mmh: float | None = None


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
    dp_dt: float = 0.0  # hPa/h  — pressure tendency
    d2p_dt2: float = 0.0  # hPa/h² — pressure acceleration
    dt_dt: float = 0.0  # °C/h
    dh_dt: float = 0.0  # %/h

    # Moisture
    dew_point: float = 10.0
    dew_depression: float = 5.0  # T − Td
    dd_trend: float = 0.0  # dew-depression change (°C/h)
    wet_bulb: float = 10.0  # Tw for precip-type decisions

    # Frontal flags
    front_warm: bool = False
    front_cold: bool = False
    front_occluded: bool = False

    # Which optional channels are backed by a real sensor.  Humidity and wind
    # default to sensible numbers, and classifying off those defaults invents
    # fog and calm that were never measured.
    has_humidity: bool = True
    has_wind: bool = True

    # Day/night (set by the HA layer before calling classify)
    is_night: bool = False


@dataclass
class _KalmanChannel:
    """Per-variable 1-D Kalman state."""

    x: float = 0.0
    p: float = 1.0
    q: float = 0.01  # process noise
    r: float = 0.1  # measurement noise
    initialized: bool = False  # seed x from the first valid reading


# ---------------------------------------------------------------------------
#  State Estimator
# ---------------------------------------------------------------------------

# Rain persistence: seconds after last detected rain before clearing
# the rain icon.  Showers come in bursts — without this the icon
# flip-flops every minute.
RAIN_PERSIST_SECONDS: float = 1200.0  # 20 minutes

# Cloud classification hysteresis band (fraction).  To switch from
# partly-cloudy → cloudy the fraction must exceed the threshold by
# this amount; to switch back it must drop below by the same amount.
CLOUD_HYSTERESIS: float = 0.06

# After rain ends, how long (seconds) to keep a cloud-fraction floor.
# Clouds don't vanish the instant the rain gauge dries.
POST_RAIN_CLOUD_SECONDS: float = 1800.0  # 30 minutes
POST_RAIN_CLOUD_FLOOR: float = 0.40

# Wind-direction smoothing factor (exponential, applied to unit vectors).
# 0.3 ≈ a ~3-sample time constant: damps light-wind jitter without
# lagging genuine wind shifts.
WIND_DIR_SMOOTH_ALPHA: float = 0.3


def dew_point_magnus(temperature_c: float, humidity_pct: float) -> float:
    """Dew point in °C from temperature and relative humidity (Magnus)."""
    rh = max(1.0, min(100.0, humidity_pct))
    a, b = 17.27, 237.7
    alpha = (a * temperature_c) / (b + temperature_c) + math.log(rh / 100.0)
    if abs(a - alpha) < 1e-8:
        return temperature_c  # saturated air
    return (b * alpha) / (a - alpha)


def wet_bulb_stull(temperature_c: float, humidity_pct: float) -> float:
    """Wet-bulb temperature in °C (Stull 2011, ±0.3 °C for normal ranges)."""
    t = temperature_c
    rh = max(1.0, min(100.0, humidity_pct))
    raw = (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh**1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )
    return max(dew_point_magnus(t, rh), min(t, raw))


class StateEstimator:
    """Fuses sensor readings into a clean state with trends and frontal flags."""

    def __init__(
        self,
        *,
        history_size: int = HISTORY_MAX_RECORDS,
        history_seconds: float = HISTORY_SECONDS,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> None:
        self._history: deque[SensorReading] = deque(maxlen=history_size)
        self._history_seconds = history_seconds
        self._latitude = latitude
        self._longitude = longitude
        self._kf: dict[str, _KalmanChannel] = {
            "pressure": _KalmanChannel(q=0.005, r=0.15),
            "temperature": _KalmanChannel(q=0.02, r=0.3),
            "humidity": _KalmanChannel(q=0.05, r=1.0),
            "wind_speed": _KalmanChannel(q=0.1, r=0.5),
        }
        self._state = SmoothedState()
        self._prev_dd: float | None = None
        self._wind_history: deque[tuple[float, float]] = deque(maxlen=60)
        # Circular (vector) smoothing of wind bearing
        self._wind_dir_sin: float | None = None
        self._wind_dir_cos: float | None = None
        self._last_rain_ts: float | None = None  # epoch of last rain_rate >= RAIN_LIGHT
        self._prev_cloud_state: str = "clear"  # "clear" | "partly" | "cloudy"

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def update(self, reading: SensorReading) -> SmoothedState:
        """Ingest one reading, return updated smoothed state."""
        self._history.append(reading)
        cutoff = reading.timestamp - self._history_seconds
        while len(self._history) > 1 and self._history[0].timestamp < cutoff:
            self._history.popleft()

        # --- Kalman update per channel ---
        self._state.pressure = self._kalman("pressure", reading.pressure_hpa)
        self._state.temperature = self._kalman("temperature", reading.temperature_c)

        if reading.humidity_pct is not None:
            self._state.humidity = self._kalman("humidity", reading.humidity_pct)
        if reading.wind_speed_ms is not None:
            self._state.wind_speed = self._kalman("wind_speed", reading.wind_speed_ms)
        if reading.wind_direction_deg is not None:
            self._state.wind_direction = self._smooth_wind_direction(reading.wind_direction_deg)
            self._wind_history.append((reading.timestamp, reading.wind_direction_deg))
        if reading.solar_radiation_wm2 is not None:
            self._state.solar_radiation = reading.solar_radiation_wm2
        if reading.rain_rate_mmh is not None:
            self._state.rain_rate = max(0.0, reading.rain_rate_mmh)
            if self._state.rain_rate >= RAIN_LIGHT:
                self._last_rain_ts = reading.timestamp

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

    def cloud_fraction(self, sun_elevation_deg: float = 90.0) -> float:
        """Public accessor for the blended 0-1 cloud-fraction estimate."""
        return self._estimate_cloud_fraction(sun_elevation_deg)

    # ------------------------------------------------------------------
    #  Classify current weather into one of 12 HA condition indices
    # ------------------------------------------------------------------

    def classify(
        self,
        sun_elevation_deg: float = 90.0,
        *,
        cloud_fraction: float | None = None,
    ) -> int:
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
            self._prev_cloud_state = "cloudy"  # rain implies overcast
            return self._precip_state(s)

        # --- 1b: Rain persistence — keep rain icon for a while after ---
        # Rain comes in bursts.  Without this the icon flips sunny/rainy
        # every minute during a shower.
        if self._last_rain_ts is not None and len(self._history) > 0:
            age = self._history[-1].timestamp - self._last_rain_ts
            if age < RAIN_PERSIST_SECONDS:
                self._prev_cloud_state = "cloudy"  # still in rain episode
                return self._precip_state(s)

        # --- 3: Thunderstorm proxy ---
        if s.dp_dt < STORM_PRESSURE_DROP and s.humidity > STORM_HUMIDITY and s.wind_speed > STORM_WIND:
            return S_LIGHTNING_RAINY

        # --- 4: Fog ---
        if s.has_humidity and s.dew_depression < FOG_DEW_DEPRESSION and s.wind_speed < FOG_MAX_WIND:
            return S_FOG

        # --- 5: Exceptional (bomb cyclone: pressure drop > 24 hPa / 24h) ---
        if s.dp_dt < -4.0:  # ~24 hPa/6h extrapolated
            return S_EXCEPTIONAL

        # --- 6: Strong wind (only if no precipitation) ---
        if s.has_wind and s.wind_speed >= WIND_STRONG:
            return S_WINDY

        # --- 7 & 8: Cloud cover + day/night (with hysteresis) ---
        # Reuse a precomputed base fraction when the caller already has one
        # (the weather entity needs the same value for its temperature model,
        # so computing it twice per tick is pure waste).
        cloud = self._estimate_cloud_fraction(sun_elevation_deg) if cloud_fraction is None else cloud_fraction

        # Apply post-rain cloud floor: clouds linger after showers
        if self._last_rain_ts is not None and len(self._history) > 0:
            rain_age = self._history[-1].timestamp - self._last_rain_ts
            if rain_age < POST_RAIN_CLOUD_SECONDS:
                # Linearly decay the floor from POST_RAIN_CLOUD_FLOOR → 0
                decay = 1.0 - rain_age / POST_RAIN_CLOUD_SECONDS
                cloud = max(cloud, POST_RAIN_CLOUD_FLOOR * decay)

        # Hysteresis: require crossing threshold ± band to change state
        h = CLOUD_HYSTERESIS
        prev = self._prev_cloud_state

        if prev == "cloudy":
            # Stay cloudy unless cloud drops well below 0.50
            if cloud < 0.50 - h:
                new = "clear" if cloud < 0.15 - h else "partly"
            else:
                new = "cloudy"
        elif prev == "partly":
            if cloud >= 0.50 + h:
                new = "cloudy"
            elif cloud < 0.15 - h:
                new = "clear"
            else:
                new = "partly"
        else:  # "clear"
            if cloud >= 0.50 + h:
                new = "cloudy"
            elif cloud >= 0.15 + h:
                new = "partly"
            else:
                new = "clear"

        self._prev_cloud_state = new

        if new == "clear":
            return S_CLEAR_NIGHT if s.is_night else S_CLEAR
        if new == "partly":
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
        # Seed from the first valid reading instead of ramping up from
        # x=0.  Ramping published physically-impossible values (e.g. a
        # sea-level pressure climbing 0 -> 970 -> 983 -> ...) for minutes
        # after every restart; seeding makes the first sample already true.
        if not k.initialized:
            k.x = measurement
            k.p = k.r
            k.initialized = True
            return k.x
        k.p += k.q
        denom = k.p + k.r
        if denom <= 0:
            denom = 1e-6
        gain = min(1.0, max(0.0, k.p / denom))
        k.x += gain * (measurement - k.x)
        k.p *= 1.0 - gain
        return k.x

    def _smooth_wind_direction(self, bearing_deg: float) -> float:
        """Circular exponential smoothing of the wind bearing.

        Averaging raw degrees is wrong across the 0/360 wrap, so smooth
        the unit-vector (sin, cos) components and recover the angle.
        """
        rad = math.radians(bearing_deg)
        s_comp, c_comp = math.sin(rad), math.cos(rad)
        if self._wind_dir_sin is None or self._wind_dir_cos is None:
            self._wind_dir_sin, self._wind_dir_cos = s_comp, c_comp
        else:
            a = WIND_DIR_SMOOTH_ALPHA
            self._wind_dir_sin += a * (s_comp - self._wind_dir_sin)
            self._wind_dir_cos += a * (c_comp - self._wind_dir_cos)
        return math.degrees(math.atan2(self._wind_dir_sin, self._wind_dir_cos)) % 360.0

    def _compute_trends(self) -> None:
        """Compute dp/dt, d²p/dt², dT/dt, dH/dt from history ring buffer."""
        if len(self._history) < 2:
            return

        # Snapshot the deque once; timestamps are monotonic (append order)
        # so nearest-sample lookups can bisect instead of scanning 3× each.
        hist = list(self._history)
        times = [r.timestamp for r in hist]
        now = hist[-1]
        t_now = now.timestamp

        # Slopes over the last hour by least squares: a two-point endpoint
        # difference throws away every sample in between and carries the full
        # sensor noise, while the regression cuts it by ~sqrt(N).
        window_start = t_now - 3600.0
        idx = bisect.bisect_left(times, window_start)
        if len(hist) - idx < 3:
            idx = 0  # startup: use whatever history exists rather than nothing
        window = hist[idx:]
        if len(window) >= 2 and (t_now - window[0].timestamp) >= 300.0:
            w_times = [(r.timestamp - t_now) / 3600.0 for r in window]
            self._state.dp_dt = self._slope(w_times, [self._detided(r) for r in window])
            self._state.dt_dt = self._slope(w_times, [r.temperature_c for r in window])
            hum = [(t, r.humidity_pct) for t, r in zip(w_times, window, strict=True) if r.humidity_pct is not None]
            if len(hum) >= 2:
                self._state.dh_dt = self._slope([t for t, _ in hum], [h for _, h in hum])

        # Pressure acceleration over a ~3 h span, from the *actual* sample
        # spacing.  The samples the buffer returns are only approximately at
        # the requested offsets, so a fixed Δ² divisor mis-scales the result.
        #   d²P/dt² ≈ 2·(s₂ − s₁) / (t₂ − t₀)   with sᵢ the two secant slopes
        ref_mid = self._nearest_sorted(times, hist, t_now - 5400)  # ~1.5 h ago
        ref_old = self._nearest_sorted(times, hist, t_now - 10800)  # ~3 h ago
        self._state.d2p_dt2 = 0.0
        if ref_mid is not None and ref_old is not None:
            t0 = (ref_old.timestamp - t_now) / 3600.0
            t1 = (ref_mid.timestamp - t_now) / 3600.0
            span_recent, span_old = -t1, t1 - t0
            if span_recent > 0.1 and span_old > 0.1:
                s_recent = (self._detided(now) - self._detided(ref_mid)) / span_recent
                s_old = (self._detided(ref_mid) - self._detided(ref_old)) / span_old
                self._state.d2p_dt2 = 2.0 * (s_recent - s_old) / (-t0)

    def _detided(self, reading: SensorReading) -> float:
        """Pressure with the solar tide removed, for trend fitting only.

        The reported pressure keeps its tide: that is what the barometer actually
        reads. Only the tendency needs it gone, and there the tide is not a small
        correction — its own slope rivals the steady/moving threshold.
        """
        return reading.pressure_hpa - tide_hpa_at(reading.timestamp, self._latitude, self._longitude)

    @staticmethod
    def _slope(times_h: list[float], values: list[float]) -> float:
        """Least-squares slope of ``values`` against ``times_h`` (per hour)."""
        n = len(times_h)
        mean_t = sum(times_h) / n
        mean_v = sum(values) / n
        denom = sum((t - mean_t) ** 2 for t in times_h)
        if denom <= 1e-12:
            return 0.0
        num = sum((t - mean_t) * (v - mean_v) for t, v in zip(times_h, values, strict=True))
        return num / denom

    def _compute_moisture(self) -> None:
        """Dew point (Magnus), wet-bulb (Stull 2011), depression trend."""
        T = self._state.temperature
        RH = max(1.0, min(100.0, self._state.humidity))

        Td = dew_point_magnus(T, RH)
        self._state.dew_point = round(Td, 1)
        self._state.dew_depression = round(T - Td, 1)
        self._state.wet_bulb = round(wet_bulb_stull(T, RH), 1)

        # --- Depression trend (°C/h) ---
        if self._prev_dd is not None and len(self._history) >= 2:
            dt = self._history[-1].timestamp - self._history[-2].timestamp
            if dt > 0:
                self._state.dd_trend = (self._state.dew_depression - self._prev_dd) / (dt / 3600)
        self._prev_dd = self._state.dew_depression

    def _detect_fronts(self) -> None:
        """Detect warm / cold / occluded frontal signatures."""
        s = self._state
        ws = self._wind_shift_rate()

        # Warm front: steady pressure fall + backing wind + rising humidity
        s.front_warm = s.dp_dt < -1.0 and s.dh_dt > 2.0 and ws < -10.0

        # Cold front: pressure trough (accelerating up) + temp drop + veering
        s.front_cold = s.d2p_dt2 > 0.5 and s.dt_dt < -1.0 and ws > 15.0

        # Occluded: strong pressure fall + big wind shift + narrow depression
        s.front_occluded = s.dp_dt < -2.0 and abs(ws) > 20.0 and s.dew_depression < 2.0

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

    @staticmethod
    def _nearest_sorted(
        times: list[float],
        hist: list[SensorReading],
        target_ts: float,
        tolerance_s: float = 1800.0,
    ) -> SensorReading | None:
        """Nearest reading to ``target_ts`` via bisect on ascending timestamps.

        ``times`` must be sorted ascending and parallel to ``hist``.  Returns
        None if the closest sample is more than ``tolerance_s`` away.
        """
        if not times:
            return None
        idx = bisect.bisect_left(times, target_ts)
        best: SensorReading | None = None
        best_diff = float("inf")
        for i in (idx - 1, idx):
            if 0 <= i < len(times):
                d = abs(times[i] - target_ts)
                if d < best_diff:
                    best_diff = d
                    best = hist[i]
        if best is None or best_diff > tolerance_s:
            return None
        return best

    def _estimate_cloud_fraction(self, sun_elevation_deg: float = 90.0) -> float:
        """Blend solar-radiation and dew-depression signals into 0-1 cloud fraction.

        Solar path uses Beer-Lambert clear-sky irradiance scaled by sun
        elevation so that morning/evening readings are not mistaken for
        overcast skies.  The fallback (night or no solar sensor) uses
        dew-point depression, which is a better cloud proxy than raw
        humidity because it is independent of temperature-sensor bias.
        """
        sol = self._state.solar_radiation
        dd = self._state.dew_depression

        # --- Solar-based estimate (Beer-Lambert clear-sky model) ---
        solar_cloud: float | None = None
        if sol > 10.0 and sun_elevation_deg > 3.0:
            el_rad = math.radians(sun_elevation_deg)
            air_mass = 1.0 / math.sin(el_rad)
            # I_clear = S₀ × τ^(AM^0.678) × sin(α)
            # S₀=1361 W/m², τ=0.72 (typical clear-sky transmittance)
            clear_sky = 1361.0 * (0.72 ** (air_mass**0.678)) * math.sin(el_rad)
            clear_sky = max(50.0, clear_sky)
            ratio = min(1.0, sol / clear_sky)
            solar_cloud = 1.0 - ratio

        # --- Dew-depression-based estimate (works day and night) ---
        # Dew depression (T − Td) is a direct proxy for saturation deficit.
        # Small dd → air near saturation → cloud / fog likely.
        # Large dd → dry air → clear sky.
        if dd > 8.0:
            dd_cloud = 0.05
        elif dd > 5.0:
            dd_cloud = 0.05 + (8.0 - dd) / 3.0 * 0.20
        elif dd > 3.0:
            dd_cloud = 0.25 + (5.0 - dd) / 2.0 * 0.30
        elif dd > 1.5:
            dd_cloud = 0.55 + (3.0 - dd) / 1.5 * 0.25
        else:
            dd_cloud = 0.80 + max(0.0, 1.5 - dd) / 1.5 * 0.20

        if solar_cloud is not None:
            return max(0.0, min(1.0, 0.6 * solar_cloud + 0.4 * dd_cloud))
        if not self._state.has_humidity:
            # No solar, no humidity: nothing observed says anything about the
            # sky.  Sit in the middle rather than assert a default-driven one.
            return 0.3
        return max(0.0, min(1.0, dd_cloud))
