"""Forecast engine for Local Weather Forecast.

Owns the whole sensor -> StateEstimator -> BayesianForecaster pipeline and
publishes one immutable result per run.  Entities render that result and
nothing else; no entity reaches into the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial
import logging
import time
from typing import Any, Final

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
)
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import (
    PressureConverter,
    SpeedConverter,
    TemperatureConverter,
)

from .bayesian_forecaster import BayesianForecaster, HourForecast
from .const import (
    CONF_ELEVATION,
    CONF_HUMIDITY_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_PRESSURE_TYPE,
    CONF_RAIN_RATE_SENSOR,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    DEFAULT_ELEVATION,
    DEFAULT_PRESSURE_TYPE,
    DOMAIN,
    FORECAST_HOURS,
    GRAVITY_EXPONENT,
    HA_CONDITIONS,
    HISTORY_SECONDS,
    KELVIN_OFFSET,
    LAPSE_RATE,
    PRESSURE_RELATIVE,
    STORAGE_VERSION,
    UPDATE_DEBOUNCE_SECONDS,
    UPDATE_INTERVAL_MINUTES,
)
from .physics_models import HumidityModel, PressureModel, TemperatureModel, WindModel
from .pressure_history import PressureHistory
from .state_estimator import SensorReading, StateEstimator

_LOGGER = logging.getLogger(__name__)

# Unit handling is delegated to Home Assistant's own converters so every unit
# HA knows about is accepted, not just the handful worth string-matching.
_CONVERTERS: Final = {
    CONF_PRESSURE_SENSOR: (PressureConverter, UnitOfPressure.HPA),
    CONF_TEMPERATURE_SENSOR: (TemperatureConverter, UnitOfTemperature.CELSIUS),
    CONF_WIND_SPEED_SENSOR: (SpeedConverter, UnitOfSpeed.METERS_PER_SECOND),
}

_SENSOR_KEYS: Final = (
    CONF_PRESSURE_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    CONF_HUMIDITY_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_RAIN_RATE_SENSOR,
)

_BEAUFORT_NAMES: Final = (
    "Calm",
    "Light air",
    "Light breeze",
    "Gentle breeze",
    "Moderate breeze",
    "Fresh breeze",
    "Strong breeze",
    "Near gale",
    "Gale",
    "Strong gale",
    "Storm",
    "Violent storm",
    "Hurricane force",
)
# WMO Beaufort upper bounds (m/s) for scales 0-11; at/above the last -> 12.
_BEAUFORT_THRESHOLDS: Final = (
    0.3,
    1.6,
    3.4,
    5.5,
    8.0,
    10.8,
    13.9,
    17.2,
    20.8,
    24.5,
    28.5,
    32.7,
)

# Sea-level pressure window every published value must fall inside.
_SEA_LEVEL_MIN: Final = 870.0
_SEA_LEVEL_MAX: Final = 1090.0


def _beaufort(wind_ms: float) -> int:
    """Convert wind speed in m/s to Beaufort scale (0-12)."""
    for i, threshold in enumerate(_BEAUFORT_THRESHOLDS):
        if wind_ms < threshold:
            return i
    return 12


@dataclass(slots=True, frozen=True)
class ForecastResult:
    """One complete pipeline run — everything any entity needs to render."""

    generated: datetime
    condition: str
    hourly: list[HourForecast]
    attributes: dict[str, Any]
    temperature: float
    apparent_temperature: float
    pressure: float
    humidity: float | None
    wind_speed: float | None
    wind_bearing: float | None
    dew_point: float | None
    hourly_dicts: list[dict[str, Any]] = field(default_factory=list)


class LocalForecastCoordinator(DataUpdateCoordinator[ForecastResult | None]):
    """Runs the forecast pipeline and pushes the result to every entity."""

    config_entry: LocalForecastConfigEntry

    def __init__(self, hass: HomeAssistant, entry: LocalForecastConfigEntry) -> None:
        """Set up the pipeline, its persistence and its refresh policy."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=timedelta(minutes=UPDATE_INTERVAL_MINUTES),
            setup_method=self._async_backfill_history,
            # Coalescing window, not a reset-on-event debounce: a station that
            # updates faster than the cooldown must still get a refresh.
            request_refresh_debouncer=Debouncer(hass, _LOGGER, cooldown=UPDATE_DEBOUNCE_SECONDS, immediate=False),
        )
        # Options are a full re-submission of the config form, so when they
        # exist they are authoritative — merging them over data would resurrect
        # optional sensors the user has just cleared.
        self._config: dict[str, Any] = dict(entry.options or entry.data)

        self._estimator = StateEstimator(
            latitude=hass.config.latitude,
            longitude=hass.config.longitude,
        )
        self._forecaster = BayesianForecaster()
        self._has_data = False

        self.pressure_history = PressureHistory()
        self._store: Store[dict[str, Any]] = Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}.pressure")

        # Sample timestamps must be monotonic — the history buffer is bisected
        # on them — but also comparable to recorder timestamps, so anchor a
        # monotonic clock onto the wall clock at startup.
        self._clock_wall = time.time()
        self._clock_mono = time.monotonic()

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    async def async_load_pressure_history(self) -> None:
        """Restore the persisted sea-level pressure buffer."""
        if (saved := await self._store.async_load()) and isinstance(saved, dict):
            self.pressure_history.load(saved.get("samples", []))

    @callback
    def async_track_sources(self, entry: ConfigEntry) -> None:
        """Refresh whenever one of the configured source sensors changes."""
        if ids := [self._config[k] for k in _SENSOR_KEYS if self._config.get(k)]:
            entry.async_on_unload(async_track_state_change_event(self.hass, ids, self._async_source_changed))

    @callback
    def _async_source_changed(self, _event: Event) -> None:
        self.config_entry.async_create_task(self.hass, self.async_request_refresh(), eager_start=False)

    # ------------------------------------------------------------------
    #  Config helpers
    # ------------------------------------------------------------------

    def has_source(self, key: str) -> bool:
        """Whether an optional sensor is configured for this entry."""
        return bool(self._config.get(key))

    def _now_ts(self) -> float:
        """Epoch-like timestamp immune to wall-clock steps (NTP, no RTC)."""
        return self._clock_wall + (time.monotonic() - self._clock_mono)

    # ------------------------------------------------------------------
    #  Pipeline
    # ------------------------------------------------------------------

    async def _async_update_data(self) -> ForecastResult | None:
        """Ingest the sensors and rebuild the forecast."""
        self._ingest_sensors()
        if not self._has_data:
            return None
        return self._run_forecast()

    # ------------------------------------------------------------------
    #  Recorder backfill
    # ------------------------------------------------------------------

    async def _async_backfill_history(self) -> None:
        """Seed the estimator from the recorder so trends are live at boot.

        Pressure tendency is the single most important input; without this
        it reads zero for the first hour after every restart.
        """
        try:
            await self._async_backfill()
        except Exception:  # never let a cold-start optimisation break setup
            _LOGGER.debug("Recorder backfill failed", exc_info=True)

    async def _async_backfill(self) -> None:
        if "recorder" not in self.hass.config.components:
            return
        pressure_id = self._config.get(CONF_PRESSURE_SENSOR)
        temp_id = self._config.get(CONF_TEMPERATURE_SENSOR)
        if not pressure_id or not temp_id:
            return

        from homeassistant.components.recorder import get_instance, history

        ids = [pressure_id, temp_id]
        humidity_id = self._config.get(CONF_HUMIDITY_SENSOR)
        if humidity_id:
            ids.append(humidity_id)

        end = dt_util.utcnow()
        start = end - timedelta(seconds=HISTORY_SECONDS)
        try:
            past = await get_instance(self.hass).async_add_executor_job(
                partial(
                    history.get_significant_states,
                    self.hass,
                    start,
                    end,
                    ids,
                    include_start_time_state=True,
                    significant_changes_only=False,
                    no_attributes=True,
                )
            )
        except Exception:  # recorder is best-effort; never block setup
            _LOGGER.debug("Recorder backfill unavailable", exc_info=True)
            return

        series = {
            key: self._series(past.get(sid, []), key)
            for key, sid in (
                (CONF_PRESSURE_SENSOR, pressure_id),
                (CONF_TEMPERATURE_SENSOR, temp_id),
                (CONF_HUMIDITY_SENSOR, humidity_id),
            )
            if sid
        }
        pressures = series.get(CONF_PRESSURE_SENSOR, [])
        temps = series.get(CONF_TEMPERATURE_SENSOR, [])
        humidities = series.get(CONF_HUMIDITY_SENSOR, [])
        if len(pressures) < 3 or not temps:
            return

        # All three series are sorted, so walk them together: rescanning each
        # one per pressure sample is quadratic, and a chatty station easily
        # puts thousands of samples in a four-hour window.
        count = 0
        temp_cur = hum_cur = 0
        for ts, value in pressures:
            temp_cur, temp = self._align(temps, temp_cur, ts)
            if temp is None:
                continue
            hum_cur, humidity = self._align(humidities, hum_cur, ts)
            self._estimator.update(
                SensorReading(
                    timestamp=ts,
                    pressure_hpa=self._to_sea_level(value, temp),
                    temperature_c=temp,
                    humidity_pct=humidity,
                )
            )
            count += 1
        if count:
            self._has_data = True
            _LOGGER.debug("Backfilled %d historical readings", count)

    def _series(self, states, config_key: str) -> list[tuple[float, float]]:
        """Convert recorder states to (timestamp, value) in canonical units."""
        live = self.hass.states.get(self._config.get(config_key) or "")
        unit = live.attributes.get(ATTR_UNIT_OF_MEASUREMENT) if live else None
        out: list[tuple[float, float]] = []
        for state in states:
            value = self._parse(getattr(state, "state", None), unit, config_key)
            if value is not None:
                out.append((state.last_updated.timestamp(), value))
        out.sort(key=lambda item: item[0])
        return out

    @staticmethod
    def _align(series: list[tuple[float, float]], cursor: int, ts: float) -> tuple[int, float | None]:
        """Advance ``cursor`` to the last sample at or before ``ts``.

        Callers must query with non-decreasing ``ts``; the cursor only ever
        moves forward, so a whole merge costs one pass.
        """
        while cursor < len(series) and series[cursor][0] <= ts:
            cursor += 1
        return cursor, series[cursor - 1][1] if cursor else None

    # ------------------------------------------------------------------
    #  Sensor ingestion
    # ------------------------------------------------------------------

    def _ingest_sensors(self) -> bool:
        """Read all configured sensors and feed a SensorReading."""
        pressure = self._read_float(CONF_PRESSURE_SENSOR)
        temperature = self._read_float(CONF_TEMPERATURE_SENSOR)
        if pressure is None or temperature is None:
            return False

        pressure = self._to_sea_level(pressure, temperature)
        # Validate the sea-level value, not the station value: above roughly
        # 1250 m a perfectly healthy QFE reading is below 870 hPa.
        if not _SEA_LEVEL_MIN <= pressure <= _SEA_LEVEL_MAX:
            _LOGGER.debug("Rejecting sea-level pressure %.1f hPa", pressure)
            return False

        self._estimator.update(
            SensorReading(
                timestamp=self._now_ts(),
                pressure_hpa=pressure,
                temperature_c=temperature,
                humidity_pct=self._read_float(CONF_HUMIDITY_SENSOR),
                wind_speed_ms=self._read_float(CONF_WIND_SPEED_SENSOR),
                wind_direction_deg=self._read_float(CONF_WIND_DIRECTION_SENSOR),
                solar_radiation_wm2=self._read_float(CONF_SOLAR_RADIATION_SENSOR),
                rain_rate_mmh=self._read_float(CONF_RAIN_RATE_SENSOR),
            )
        )
        self._has_data = True
        return True

    def _to_sea_level(self, pressure: float, temperature: float) -> float:
        """Convert station pressure (QFE) to sea level (QNH) when needed."""
        if self._config.get(CONF_PRESSURE_TYPE, DEFAULT_PRESSURE_TYPE) == PRESSURE_RELATIVE:
            return pressure
        elevation = self._config.get(CONF_ELEVATION, DEFAULT_ELEVATION)
        if not elevation:
            return pressure
        temp_kelvin = max(200.0, temperature + KELVIN_OFFSET)
        return pressure * (1 - LAPSE_RATE * elevation / temp_kelvin) ** -GRAVITY_EXPONENT

    def _read_float(self, config_key: str) -> float | None:
        sid = self._config.get(config_key)
        if not sid:
            return None
        state = self.hass.states.get(sid)
        if state is None or state.state in ("unknown", "unavailable", ""):
            return None
        return self._parse(state.state, state.attributes.get(ATTR_UNIT_OF_MEASUREMENT), config_key)

    @staticmethod
    def _parse(raw: Any, unit: str | None, config_key: str) -> float | None:
        """Parse a state string into a validated value in canonical units."""
        try:
            val = float(raw)
        except (ValueError, TypeError):
            return None

        converter = _CONVERTERS.get(config_key)
        if converter is not None and unit:
            cls, target = converter
            if unit != target and unit in cls.VALID_UNITS:
                try:
                    val = cls.convert(val, unit, target)
                except (ValueError, TypeError):
                    return None

        # Reject physically impossible values after conversion.  The pressure
        # bound is deliberately wide here: this is station pressure, which is
        # only checked against the sea-level window after reduction.
        if config_key == CONF_PRESSURE_SENSOR and not (300.0 <= val <= 1100.0):
            return None
        if config_key == CONF_TEMPERATURE_SENSOR and not (-60.0 <= val <= 60.0):
            return None
        if config_key == CONF_HUMIDITY_SENSOR:
            val = max(0.0, min(100.0, val))
        if config_key == CONF_WIND_SPEED_SENSOR:
            val = max(0.0, min(60.0, val))

        return val

    # ------------------------------------------------------------------
    #  Forecast
    # ------------------------------------------------------------------

    def _run_forecast(self) -> ForecastResult:
        """Classify current state, build physics models, run Bayesian forecast."""
        s = self._estimator.state
        s.has_humidity = self.has_source(CONF_HUMIDITY_SENSOR)
        s.has_wind = self.has_source(CONF_WIND_SPEED_SENSOR)

        sunrise_h, sunset_h = self._sun_hours()
        sun_el = self._sun_elevation()
        now_local = dt_util.now()
        now_h = now_local.hour + now_local.minute / 60.0
        sun = self.hass.states.get("sun.sun")
        # sun.sun already answers this correctly at every latitude and across
        # midnight; re-deriving it from decimal hours does not.
        s.is_night = sun.state == "below_horizon" if sun is not None else not (sunrise_h <= now_h < sunset_h)

        # Cloud fraction: computed once and reused for both the classifier
        # (section 7 hysteresis) and the temperature model below.
        cloud = self._estimator.cloud_fraction(sun_el)
        current_idx = self._estimator.classify(sun_el, cloud_fraction=cloud)

        _LOGGER.debug(
            "Pipeline: P=%.1f dp/dt=%.2f T=%.1f RH=%.0f wind=%.1f state=%s night=%s",
            s.pressure,
            s.dp_dt,
            s.temperature,
            s.humidity,
            s.wind_speed,
            HA_CONDITIONS[current_idx],
            s.is_night,
        )

        temp_model = TemperatureModel(
            current_temp=s.temperature,
            dt_dt=s.dt_dt,
            humidity=s.humidity,
            wind_speed=s.wind_speed,
            cloud_fraction=cloud,
            sunrise_hour=sunrise_h,
            sunset_hour=sunset_h,
            current_hour=now_h,
            latitude=self.hass.config.latitude or 48.0,
        )
        hourly = self._forecaster.forecast(
            current_state_idx=current_idx,
            smoothed=s,
            hours=FORECAST_HOURS,
            sunrise_hour=sunrise_h,
            sunset_hour=sunset_h,
            current_hour=now_h,
            predict_temperature=temp_model,
            predict_pressure=PressureModel(s.pressure, s.dp_dt),
            predict_humidity=HumidityModel(s.humidity, s.temperature, temp_model),
            predict_wind=WindModel(s.wind_speed, s.wind_direction, s.dp_dt),
        )

        if hourly:
            h1 = hourly[0]
            _LOGGER.debug(
                "Forecast: %d hours, +1h=%s %.1f°C %d%% precip",
                len(hourly),
                h1.condition,
                h1.temperature,
                h1.precipitation_probability,
            )

        self._record_pressure(s.pressure)

        return ForecastResult(
            generated=now_local,
            condition=HA_CONDITIONS[current_idx],
            hourly=hourly,
            attributes=self._build_attributes(hourly),
            temperature=round(s.temperature, 1),
            apparent_temperature=self._apparent_temperature(),
            pressure=round(s.pressure, 1),
            humidity=(round(s.humidity) if self.has_source(CONF_HUMIDITY_SENSOR) else None),
            wind_speed=(round(s.wind_speed, 1) if self.has_source(CONF_WIND_SPEED_SENSOR) else None),
            wind_bearing=(
                round((s.wind_direction + 360) % 360) if self.has_source(CONF_WIND_DIRECTION_SENSOR) else None
            ),
            dew_point=(s.dew_point if self.has_source(CONF_HUMIDITY_SENSOR) else None),
            hourly_dicts=_hourly_dicts(hourly, now_local),
        )

    def _record_pressure(self, pressure: float) -> None:
        """Append an hourly sea-level sample for the tendency/barometer sensors."""
        if not _SEA_LEVEL_MIN <= pressure <= _SEA_LEVEL_MAX:
            return
        buffer = self.pressure_history
        before = len(buffer.dump())
        # Wall clock on purpose: this buffer is persisted across restarts.
        buffer.record(time.time(), pressure)
        if len(buffer.dump()) != before:
            self._store.async_delay_save(lambda: {"samples": buffer.dump()}, 60)

    def _apparent_temperature(self) -> float:
        """Feels-like temperature — wind chill or heat index."""
        s = self._estimator.state
        temp, wind, humidity = s.temperature, s.wind_speed, s.humidity
        # Wind chill (Environment Canada formula, T < 10 °C, W > 4.8 km/h)
        wind_kmh = wind * 3.6
        if self.has_source(CONF_WIND_SPEED_SENSOR) and temp <= 10.0 and wind_kmh > 4.8:
            return round(
                13.12 + 0.6215 * temp - 11.37 * wind_kmh**0.16 + 0.3965 * temp * wind_kmh**0.16,
                1,
            )
        # Heat index (Steadman, T > 27 °C)
        if self.has_source(CONF_HUMIDITY_SENSOR) and temp >= 27.0 and humidity >= 40:
            return round(
                -8.785
                + 1.611 * temp
                + 2.339 * humidity
                - 0.1461 * temp * humidity
                - 0.01231 * temp * temp
                - 0.01642 * humidity * humidity
                + 0.002212 * temp * temp * humidity
                + 0.000725 * temp * humidity * humidity
                - 0.000003582 * temp * temp * humidity * humidity,
                1,
            )
        return round(temp, 1)

    def _build_attributes(self, hourly: list[HourForecast]) -> dict[str, Any]:
        s = self._estimator.state
        attrs: dict[str, Any] = {
            "pressure_trend": round(s.dp_dt, 2),
            "pressure_acceleration": round(s.d2p_dt2, 2),
            "front_warm": s.front_warm,
            "front_cold": s.front_cold,
            "front_occluded": s.front_occluded,
        }
        # Moisture is derived from relative humidity; without that sensor the
        # numbers would be pure fiction, so publish nothing instead.
        if self.has_source(CONF_HUMIDITY_SENSOR):
            attrs["dew_point"] = s.dew_point
            attrs["dew_depression"] = s.dew_depression
            attrs["wet_bulb"] = s.wet_bulb
        if self.has_source(CONF_WIND_SPEED_SENSOR):
            force = _beaufort(s.wind_speed)
            attrs["wind_force"] = force
            attrs["wind_force_description"] = _BEAUFORT_NAMES[force]
        if hourly:
            h1 = hourly[0]
            attrs["next_hour_condition"] = h1.condition
            attrs["next_hour_precip_probability"] = h1.precipitation_probability
            # Aggregate precipitation probability over the next 6 h.
            # Hourly probabilities are strongly correlated (not independent),
            # so compounding 1-Π(1-pᵢ) badly overstates the risk — a sunny
            # day reads ~35%.  The window maximum is the honest "chance of
            # rain in the next 6 hours" for a non-technical dashboard.
            if len(hourly) >= 6:
                attrs["precip_probability_6h"] = max(hf.precipitation_probability for hf in hourly[:6])
        return attrs

    # ------------------------------------------------------------------
    #  Sun
    # ------------------------------------------------------------------

    def _sun_elevation(self) -> float:
        """Return current sun elevation in degrees from sun.sun entity."""
        if sun := self.hass.states.get("sun.sun"):
            try:
                if (elevation := sun.attributes.get("elevation")) is not None:
                    return float(elevation)
            except (ValueError, TypeError):
                pass
        return 45.0  # fallback: assume mid-sky

    def _sun_hours(self) -> tuple[float, float]:
        """Return (sunrise_hour, sunset_hour) in local decimal hours."""
        if sun := self.hass.states.get("sun.sun"):
            try:
                rising = dt_util.parse_datetime(sun.attributes.get("next_rising", ""))
                setting = dt_util.parse_datetime(sun.attributes.get("next_setting", ""))
                if rising and setting:
                    rise_local = dt_util.as_local(rising)
                    set_local = dt_util.as_local(setting)
                    return (
                        rise_local.hour + rise_local.minute / 60.0,
                        set_local.hour + set_local.minute / 60.0,
                    )
            except (ValueError, TypeError):
                pass
        return (6.0, 20.0)


def _hourly_dicts(hourly: list[HourForecast], generated: datetime) -> list[dict[str, Any]]:
    """Hourly forecast as plain dicts for the meteogram sensor attribute.

    Mirrors what ``get_forecasts`` returns but in the entity's native units,
    so a dashboard meteogram card can read it straight off the sensor without
    a websocket subscription.
    """
    return [
        {
            "datetime": (generated + timedelta(hours=hf.hours_ahead)).isoformat(),
            "condition": hf.condition,
            "temperature": round(hf.temperature, 1),
            "humidity": hf.humidity,
            "pressure": round(hf.pressure, 1),
            "precipitation_probability": hf.precipitation_probability,
            "precipitation": round(hf.precipitation_amount, 1),
            "wind_speed": round(hf.wind_speed, 1),
            "wind_bearing": round(hf.wind_bearing),
            "is_daytime": hf.is_daytime,
        }
        for hf in hourly
    ]


type LocalForecastConfigEntry = ConfigEntry[LocalForecastCoordinator]
