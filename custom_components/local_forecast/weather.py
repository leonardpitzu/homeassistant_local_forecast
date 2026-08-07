"""Weather entity for Local Weather Forecast.

This is the single entity the integration exposes.  It drives:
  - The weather card on dashboards (phone / tablet)
  - Hourly and daily forecast services
  - All attributes visible in Developer Tools

The entity listens to configured sensor state changes, feeds them into
the StateEstimator → BayesianForecaster → PhysicsModels pipeline, and
publishes the result as a standard HA WeatherEntity.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Final

from homeassistant.components.weather import (
    Forecast,
    WeatherEntity,
    WeatherEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
)
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.loader import async_get_integration
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
    SIGNAL_UPDATE,
)
from .physics_models import HumidityModel, PressureModel, TemperatureModel, WindModel
from .state_estimator import SensorReading, StateEstimator

_LOGGER = logging.getLogger(__name__)

# Unit handling is delegated to Home Assistant's own converters so every unit
# HA knows about is accepted, not just the handful worth string-matching.
_CONVERTERS: Final = {
    CONF_PRESSURE_SENSOR: (PressureConverter, UnitOfPressure.HPA),
    CONF_TEMPERATURE_SENSOR: (TemperatureConverter, UnitOfTemperature.CELSIUS),
    CONF_WIND_SPEED_SENSOR: (SpeedConverter, UnitOfSpeed.METERS_PER_SECOND),
}

# Severity ranking used to pick the "worst" condition of a day for the daily
# forecast card (higher = more severe, what a non-technical family cares about).
_CONDITION_SEVERITY: dict[str, int] = {
    "lightning-rainy": 11,
    "exceptional": 10,
    "pouring": 9,
    "snowy": 8,
    "snowy-rainy": 7,
    "rainy": 6,
    "fog": 5,
    "windy": 4,
    "cloudy": 3,
    "partlycloudy": 2,
    "clear-night": 1,
    "sunny": 0,
}

# Day+2 extrapolation regresses severe conditions toward milder ones.
_CONDITION_DECAY: dict[str, str] = {
    "pouring": "rainy",
    "lightning-rainy": "rainy",
    "exceptional": "cloudy",
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Local Weather Forecast weather entity."""
    integration = await async_get_integration(hass, DOMAIN)
    sw_version = str(integration.version) if integration.version else None
    entity = LocalForecastWeather(hass, entry, sw_version=sw_version)
    hass.data[DOMAIN][entry.entry_id]["weather_entity"] = entity
    async_add_entities([entity], True)


# ---------------------------------------------------------------------------
#  Weather Entity
# ---------------------------------------------------------------------------

class LocalForecastWeather(WeatherEntity):
    """Bayesian local weather forecast entity."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_should_poll = False
    _attr_native_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_native_pressure_unit = UnitOfPressure.HPA
    _attr_native_wind_speed_unit = UnitOfSpeed.METERS_PER_SECOND
    _attr_supported_features = (
        WeatherEntityFeature.FORECAST_HOURLY
        | WeatherEntityFeature.FORECAST_DAILY
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, sw_version: str | None = None
    ) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_weather"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Local Weather Forecast",
            manufacturer="Local Weather Forecast",
            model="Bayesian Forecaster",
            sw_version=sw_version,
        )

        # Core pipeline
        self._estimator = StateEstimator()
        self._forecaster = BayesianForecaster()

        # Cached forecasts
        self._hourly: list[HourForecast] = []
        self._condition: str | None = None
        self._attrs: dict[str, Any] = {}
        self._forecast_ts: datetime | None = None
        self._debounce_cancel: Any = None
        self._pending_since: float | None = None
        self._min_interval: float = 30.0   # debounce quiet period
        self._max_wait: float = 120.0      # never postpone longer than this
        self._has_data = False

        # Sample timestamps must be monotonic — the history buffer is
        # bisected on them — but also comparable to recorder timestamps, so
        # anchor a monotonic clock onto the wall clock at startup.
        self._clock_wall = time.time()
        self._clock_mono = time.monotonic()

    def _now_ts(self) -> float:
        """Epoch-like timestamp immune to wall-clock steps (NTP, no RTC)."""
        return self._clock_wall + (time.monotonic() - self._clock_mono)

    # ------------------------------------------------------------------
    #  Config helpers
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default: Any = None) -> Any:
        return self._entry.options.get(key, self._entry.data.get(key, default))

    def _has(self, key: str) -> bool:
        """Whether an optional sensor is configured for this entry."""
        return bool(self._cfg(key))

    def _sensor_ids(self) -> list[str]:
        """All configured sensor entity_ids (for state tracking)."""
        keys = [
            CONF_PRESSURE_SENSOR, CONF_TEMPERATURE_SENSOR,
            CONF_HUMIDITY_SENSOR, CONF_WIND_SPEED_SENSOR,
            CONF_WIND_DIRECTION_SENSOR, CONF_SOLAR_RADIATION_SENSOR,
            CONF_RAIN_RATE_SENSOR,
        ]
        return [self._cfg(k) for k in keys if self._cfg(k)]

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    async def async_added_to_hass(self) -> None:
        """Subscribe to sensor changes."""
        await super().async_added_to_hass()

        try:
            await self._async_backfill_history()
        except Exception:  # never let a cold-start optimisation break setup
            _LOGGER.debug("Recorder backfill failed", exc_info=True)

        ids = self._sensor_ids()
        if ids:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, ids, self._on_sensor_change
                )
            )
        # Safety net: the entity does not poll, so guarantee the forecast is
        # refreshed even while every source sensor sits perfectly still.
        self.async_on_remove(
            async_track_time_interval(
                self.hass, self._on_interval, timedelta(minutes=5)
            )
        )
        self.async_on_remove(self._cancel_debounce)

    @callback
    def _cancel_debounce(self) -> None:
        if self._debounce_cancel:
            self._debounce_cancel()
            self._debounce_cancel = None

    async def _on_interval(self, _now) -> None:
        await self._async_recalculate()

    @callback
    def _on_sensor_change(self, event: Event) -> None:
        """Handle sensor state change — debounced with a hard ceiling.

        A pure reset-on-event debounce starves forever when any source
        sensor updates faster than the quiet period.
        """
        now = self._now_ts()
        if self._pending_since is None:
            self._pending_since = now
        elif now - self._pending_since >= self._max_wait:
            self._cancel_debounce()
            self._debounce_fire(None)
            return
        self._cancel_debounce()
        self._debounce_cancel = async_call_later(
            self.hass, self._min_interval, self._debounce_fire
        )

    @callback
    def _debounce_fire(self, _now) -> None:
        """Fire after debounce interval — all sensors are current."""
        self._debounce_cancel = None
        self._pending_since = None
        self._entry.async_create_task(
            self.hass, self._async_recalculate(), eager_start=False
        )

    async def _async_recalculate(self) -> None:
        """Run the full pipeline and write state."""
        self._ingest_sensors()
        self._run_forecast()
        self.async_write_ha_state()
        async_dispatcher_send(self.hass, SIGNAL_UPDATE.format(self._entry.entry_id))
        await self.async_update_listeners(None)

    async def async_update(self) -> None:
        """Refresh on demand (initial add, or an explicit update request)."""
        self._ingest_sensors()
        self._run_forecast()
        async_dispatcher_send(self.hass, SIGNAL_UPDATE.format(self._entry.entry_id))
        await self.async_update_listeners(None)

    async def _async_backfill_history(self) -> None:
        """Seed the estimator from the recorder so trends are live at boot.

        Pressure tendency is the single most important input; without this
        it reads zero for the first hour after every restart.
        """
        if "recorder" not in self.hass.config.components:
            return
        pressure_id = self._cfg(CONF_PRESSURE_SENSOR)
        temp_id = self._cfg(CONF_TEMPERATURE_SENSOR)
        if not pressure_id or not temp_id:
            return

        from homeassistant.components.recorder import get_instance, history

        ids = [pressure_id, temp_id]
        humidity_id = self._cfg(CONF_HUMIDITY_SENSOR)
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
        if len(pressures) < 3 or not temps:
            return

        count = 0
        for ts, value in pressures:
            temp = self._latest_before(temps, ts)
            if temp is None:
                continue
            reading = SensorReading(
                timestamp=ts,
                pressure_hpa=self._to_sea_level(value, temp),
                temperature_c=temp,
                humidity_pct=self._latest_before(
                    series.get(CONF_HUMIDITY_SENSOR, []), ts
                ),
            )
            self._estimator.update(reading)
            count += 1
        if count:
            self._has_data = True
            _LOGGER.debug("Backfilled %d historical readings", count)

    def _series(self, states, config_key: str) -> list[tuple[float, float]]:
        """Convert recorder states to (timestamp, value) in canonical units."""
        live = self.hass.states.get(self._cfg(config_key) or "")
        unit = live.attributes.get(ATTR_UNIT_OF_MEASUREMENT) if live else None
        out: list[tuple[float, float]] = []
        for state in states:
            value = self._parse(getattr(state, "state", None), unit, config_key)
            if value is not None:
                out.append((state.last_updated.timestamp(), value))
        out.sort(key=lambda item: item[0])
        return out

    @staticmethod
    def _latest_before(
        series: list[tuple[float, float]], ts: float
    ) -> float | None:
        """Most recent value in ``series`` at or before ``ts``."""
        best: float | None = None
        for sample_ts, value in series:
            if sample_ts > ts:
                break
            best = value
        return best

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
        if not 870.0 <= pressure <= 1090.0:
            _LOGGER.debug("Rejecting sea-level pressure %.1f hPa", pressure)
            return False

        reading = SensorReading(
            timestamp=self._now_ts(),
            pressure_hpa=pressure,
            temperature_c=temperature,
            humidity_pct=self._read_float(CONF_HUMIDITY_SENSOR),
            wind_speed_ms=self._read_float(CONF_WIND_SPEED_SENSOR),
            wind_direction_deg=self._read_float(CONF_WIND_DIRECTION_SENSOR),
            solar_radiation_wm2=self._read_float(CONF_SOLAR_RADIATION_SENSOR),
            rain_rate_mmh=self._read_float(CONF_RAIN_RATE_SENSOR),
        )
        self._estimator.update(reading)
        self._has_data = True
        return True

    def _to_sea_level(self, pressure: float, temperature: float) -> float:
        """Convert station pressure (QFE) to sea level (QNH) when needed."""
        if self._cfg(CONF_PRESSURE_TYPE, DEFAULT_PRESSURE_TYPE) == PRESSURE_RELATIVE:
            return pressure
        elevation = self._cfg(CONF_ELEVATION, DEFAULT_ELEVATION)
        if not elevation:
            return pressure
        temp_kelvin = max(200.0, temperature + KELVIN_OFFSET)
        return pressure * (1 - LAPSE_RATE * elevation / temp_kelvin) ** -GRAVITY_EXPONENT

    def _read_float(self, config_key: str) -> float | None:
        sid = self._cfg(config_key)
        if not sid:
            return None
        state = self.hass.states.get(sid)
        if state is None or state.state in ("unknown", "unavailable", ""):
            return None
        return self._parse(
            state.state, state.attributes.get(ATTR_UNIT_OF_MEASUREMENT), config_key
        )

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
    #  Forecast pipeline
    # ------------------------------------------------------------------

    def _run_forecast(self) -> None:
        """Classify current state, build physics models, run Bayesian forecast."""
        if not self._has_data:
            return
        s = self._estimator.state
        s.has_humidity = self._has(CONF_HUMIDITY_SENSOR)
        s.has_wind = self._has(CONF_WIND_SPEED_SENSOR)

        # Day/night from sun entity
        sunrise_h, sunset_h = self._sun_hours()
        sun_el = self._sun_elevation()
        now_local = dt_util.now()
        now_h = now_local.hour + now_local.minute / 60.0
        sun = self.hass.states.get("sun.sun")
        # sun.sun already answers this correctly at every latitude and across
        # midnight; re-deriving it from decimal hours does not.
        s.is_night = (
            sun.state == "below_horizon"
            if sun is not None
            else not (sunrise_h <= now_h < sunset_h)
        )

        # Cloud fraction: computed once and reused for both the classifier
        # (section 7 hysteresis) and the temperature model below.
        cloud = self._estimator.cloud_fraction(sun_el)

        # Current state classification.  Cache it for the `condition`
        # property so that reading entity state never mutates the
        # estimator's cloud-hysteresis / rain-persistence state machine.
        current_idx = self._estimator.classify(sun_el, cloud_fraction=cloud)
        self._condition = HA_CONDITIONS[current_idx]

        _LOGGER.debug(
            "Pipeline: P=%.1f dp/dt=%.2f T=%.1f RH=%.0f wind=%.1f "
            "state=%s night=%s",
            s.pressure, s.dp_dt, s.temperature, s.humidity,
            s.wind_speed, HA_CONDITIONS[current_idx], s.is_night,
        )

        # Physics models
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
        pres_model = PressureModel(s.pressure, s.dp_dt)
        hum_model = HumidityModel(s.humidity, s.temperature, temp_model)
        wind_model = WindModel(s.wind_speed, s.wind_direction, s.dp_dt)

        # Bayesian forecast
        self._hourly = self._forecaster.forecast(
            current_state_idx=current_idx,
            smoothed=s,
            hours=FORECAST_HOURS,
            sunrise_hour=sunrise_h,
            sunset_hour=sunset_h,
            current_hour=now_h,
            predict_temperature=temp_model,
            predict_pressure=pres_model,
            predict_humidity=hum_model,
            predict_wind=wind_model,
        )

        if self._hourly:
            h1 = self._hourly[0]
            _LOGGER.debug(
                "Forecast: %d hours, +1h=%s %.1f°C %d%% precip",
                len(self._hourly), h1.condition,
                h1.temperature, h1.precipitation_probability,
            )

        # Build the attribute dict once per pipeline run.  Several sensor
        # entities read `extra_state_attributes` on every weather update, so
        # rebuilding it per reader was the main avoidable work in the hot path.
        self._attrs = self._build_attributes()

        # Timestamp of this forecast generation (state of the hourly-forecast
        # sensor) and hourly sea-level-pressure sample for the pressure ring
        # buffer that feeds the tendency / synoptic / barometer sensors.
        self._forecast_ts = now_local
        entry_data = self.hass.data[DOMAIN][self._entry.entry_id]
        buffer = entry_data.get("pressure_history")
        if buffer is not None and 870.0 <= s.pressure <= 1090.0:
            # Wall clock on purpose: this buffer is persisted across restarts.
            before = len(buffer.dump())
            buffer.record(time.time(), s.pressure)
            store = entry_data.get("pressure_store")
            if store is not None and len(buffer.dump()) != before:
                store.async_delay_save(lambda: {"samples": buffer.dump()}, 60)

    def _sun_elevation(self) -> float:
        """Return current sun elevation in degrees from sun.sun entity."""
        sun = self.hass.states.get("sun.sun")
        if sun:
            try:
                el = sun.attributes.get("elevation")
                if el is not None:
                    return float(el)
            except (ValueError, TypeError):
                pass
        return 45.0  # fallback: assume mid-sky

    def _sun_hours(self) -> tuple[float, float]:
        """Return (sunrise_hour, sunset_hour) in local decimal hours."""
        sun = self.hass.states.get("sun.sun")
        if sun:
            try:
                sr = dt_util.parse_datetime(
                    sun.attributes.get("next_rising", "")
                )
                ss = dt_util.parse_datetime(
                    sun.attributes.get("next_setting", "")
                )
                if sr and ss:
                    sr_local = dt_util.as_local(sr)
                    ss_local = dt_util.as_local(ss)
                    return (
                        sr_local.hour + sr_local.minute / 60.0,
                        ss_local.hour + ss_local.minute / 60.0,
                    )
            except (ValueError, TypeError):
                pass
        # Fallback
        return (6.0, 20.0)

    # ------------------------------------------------------------------
    #  WeatherEntity properties — current conditions
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Only claim a state once real sensor data has been ingested."""
        return self._has_data

    @property
    def condition(self) -> str | None:
        return self._condition

    @property
    def native_temperature(self) -> float | None:
        return round(self._estimator.state.temperature, 1)

    @property
    def humidity(self) -> float | None:
        if not self._has(CONF_HUMIDITY_SENSOR):
            return None
        return round(self._estimator.state.humidity)

    @property
    def native_pressure(self) -> float | None:
        return round(self._estimator.state.pressure, 1)

    @property
    def native_wind_speed(self) -> float | None:
        if not self._has(CONF_WIND_SPEED_SENSOR):
            return None
        return round(self._estimator.state.wind_speed, 1)

    @property
    def wind_bearing(self) -> float | None:
        if not self._has(CONF_WIND_DIRECTION_SENSOR):
            return None
        return round((self._estimator.state.wind_direction + 360) % 360)

    @property
    def native_apparent_temperature(self) -> float | None:
        """Apparent (feels-like) temperature — wind chill or heat index."""
        s = self._estimator.state
        T, W, RH = s.temperature, s.wind_speed, s.humidity
        # Wind chill (Environment Canada formula, T < 10 °C, W > 4.8 km/h)
        W_kmh = W * 3.6
        if self._has(CONF_WIND_SPEED_SENSOR) and T <= 10.0 and W_kmh > 4.8:
            wc = (
                13.12 + 0.6215 * T
                - 11.37 * W_kmh ** 0.16
                + 0.3965 * T * W_kmh ** 0.16
            )
            return round(wc, 1)
        # Heat index (Steadman, T > 27 °C)
        if self._has(CONF_HUMIDITY_SENSOR) and T >= 27.0 and RH >= 40:
            hi = (
                -8.785 + 1.611 * T + 2.339 * RH
                - 0.1461 * T * RH - 0.01231 * T * T
                - 0.01642 * RH * RH + 0.002212 * T * T * RH
                + 0.000725 * T * RH * RH - 0.000003582 * T * T * RH * RH
            )
            return round(hi, 1)
        return round(T, 1)

    @property
    def native_dew_point(self) -> float | None:
        if not self._has(CONF_HUMIDITY_SENSOR):
            return None
        return self._estimator.state.dew_point

    # ------------------------------------------------------------------
    #  Extra state attributes (visible in Developer Tools, usable in
    #  templates / automations / pixel display)
    # ------------------------------------------------------------------

    _BEAUFORT_NAMES: Final = (
        "Calm", "Light air", "Light breeze", "Gentle breeze",
        "Moderate breeze", "Fresh breeze", "Strong breeze",
        "Near gale", "Gale", "Strong gale", "Storm",
        "Violent storm", "Hurricane force",
    )
    # WMO Beaufort upper bounds (m/s) for scales 0-11; at/above the last → 12.
    _BEAUFORT_THRESHOLDS: Final = (
        0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9,
        17.2, 20.8, 24.5, 28.5, 32.7,
    )

    @classmethod
    def _beaufort(cls, wind_ms: float) -> int:
        """Convert wind speed in m/s to Beaufort scale (0-12)."""
        for i, t in enumerate(cls._BEAUFORT_THRESHOLDS):
            if wind_ms < t:
                return i
        return 12

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the attribute dict cached by the last pipeline run."""
        return self._attrs

    def _build_attributes(self) -> dict[str, Any]:
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
        if self._has(CONF_HUMIDITY_SENSOR):
            attrs["dew_point"] = s.dew_point
            attrs["dew_depression"] = s.dew_depression
            attrs["wet_bulb"] = s.wet_bulb
        if self._has(CONF_WIND_SPEED_SENSOR):
            force = self._beaufort(s.wind_speed)
            attrs["wind_force"] = force
            attrs["wind_force_description"] = self._BEAUFORT_NAMES[force]
        if self._hourly:
            h1 = self._hourly[0]
            attrs["next_hour_condition"] = h1.condition
            attrs["next_hour_precip_probability"] = h1.precipitation_probability
            # Aggregate precipitation probability over the next 6 h.
            # Hourly probabilities are strongly correlated (not independent),
            # so compounding 1−Π(1−pᵢ) badly overstates the risk — a sunny
            # day reads ~35%.  The window maximum is the honest "chance of
            # rain in the next 6 hours" for a non-technical dashboard.
            if len(self._hourly) >= 6:
                attrs["precip_probability_6h"] = max(
                    hf.precipitation_probability for hf in self._hourly[:6]
                )
        return attrs

    # ------------------------------------------------------------------
    #  Forecast services (what weather cards call)
    # ------------------------------------------------------------------

    async def async_forecast_hourly(self) -> list[Forecast] | None:
        if not self._hourly:
            return None

        # Anchor on when the forecast was computed, not when a card happens to
        # ask, or every entry drifts by up to one refresh interval.
        now = self._forecast_ts or dt_util.now()
        result: list[Forecast] = []
        for hf in self._hourly:
            ft = now + timedelta(hours=hf.hours_ahead)
            result.append(
                Forecast(  # type: ignore[typeddict-unknown-key]
                    datetime=ft.isoformat(),
                    condition=hf.condition,
                    native_temperature=hf.temperature,
                    humidity=hf.humidity,
                    native_pressure=hf.pressure,
                    precipitation_probability=hf.precipitation_probability,
                    native_precipitation=hf.precipitation_amount,
                    native_wind_speed=hf.wind_speed,
                    wind_bearing=hf.wind_bearing,
                    is_daytime=hf.is_daytime,
                )
            )
        return result

    @property
    def forecast_generated(self) -> datetime | None:
        """Timestamp of the most recent forecast generation."""
        return self._forecast_ts

    def hourly_forecast_list(self) -> list[dict[str, Any]]:
        """Synchronous hourly forecast as plain dicts (for the meteogram sensor).

        Mirrors what ``get_forecasts`` returns but in the entity's native
        units, so a dashboard meteogram card can read it straight off the
        sensor's ``forecast`` attribute without a websocket subscription.
        """
        if not self._hourly:
            return []
        now = self._forecast_ts or dt_util.now()
        out: list[dict[str, Any]] = []
        for hf in self._hourly:
            ft = now + timedelta(hours=hf.hours_ahead)
            out.append(
                {
                    "datetime": ft.isoformat(),
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
            )
        return out

    async def async_forecast_daily(self) -> list[Forecast] | None:
        """Aggregate hourly into today / tomorrow / day-after-tomorrow.

        With a 12-hour horizon, tomorrow uses the tail hours and day+2
        extrapolates further toward climatological mean values.
        """
        if not self._hourly:
            return None

        now = self._forecast_ts or dt_util.now()
        hours_left_today = 24 - now.hour
        today_hours = [h for h in self._hourly if h.hours_ahead <= hours_left_today]
        tomorrow_hours = [h for h in self._hourly if h.hours_ahead > hours_left_today]

        # If all forecast hours fall in today, use the tail as proxy
        if not tomorrow_hours:
            tomorrow_hours = self._hourly[-min(6, len(self._hourly)):]
        if not today_hours:
            today_hours = self._hourly[:min(6, len(self._hourly))]

        days: list[Forecast] = []

        for offset_days, hours in enumerate([today_hours, tomorrow_hours]):
            if not hours:
                continue
            temps = [h.temperature for h in hours]
            condition = self._worst_condition(hours)

            # Daily forecasts are always daytime — swap clear-night to sunny
            if condition == "clear-night":
                condition = "sunny"

            # Today uses current time so it's never "in the past" for the
            # frontend card; future days use noon.
            day = now.date() + timedelta(days=offset_days)
            if offset_days == 0:
                dt_entry = now.replace(microsecond=0)
            else:
                dt_entry = datetime(
                    day.year, day.month, day.day, 12, 0, 0,
                    tzinfo=now.tzinfo,
                )

            days.append(
                Forecast(  # type: ignore[typeddict-unknown-key]
                    datetime=dt_entry.isoformat(),
                    condition=condition,
                    native_temperature=round(max(temps), 1),
                    native_templow=round(min(temps), 1),
                    precipitation_probability=max(
                        h.precipitation_probability for h in hours
                    ),
                    native_precipitation=round(
                        sum(h.precipitation_amount for h in hours), 1
                    ),
                    humidity=round(
                        sum(h.humidity for h in hours) / len(hours)
                    ),
                    native_pressure=round(
                        sum(h.pressure for h in hours) / len(hours), 1
                    ),
                    native_wind_speed=round(
                        sum(h.wind_speed for h in hours) / len(hours), 1
                    ),
                    wind_bearing=round(hours[0].wind_bearing),
                    is_daytime=True,
                )
            )

        # Day+2: extrapolate beyond 12h horizon toward climatological norms.
        # Decay precipitation probability, regress temps toward mean, etc.
        if tomorrow_hours:
            last = tomorrow_hours[-1]
            # Temperature regresses toward the daily mean (avg of today's
            # high/low) with ~24h time constant
            today_temps = [h.temperature for h in today_hours] if today_hours else [last.temperature]
            daily_mean = (max(today_temps) + min(today_temps)) / 2.0
            day2_temp_high = round(
                last.temperature * 0.6 + daily_mean * 0.4, 1
            )
            day2_temp_low = round(
                min(h.temperature for h in tomorrow_hours) * 0.6
                + (daily_mean - 4.0) * 0.4, 1
            )
            # Ensure high >= low
            if day2_temp_high < day2_temp_low:
                day2_temp_high, day2_temp_low = day2_temp_low, day2_temp_high

            # Precip probability decays toward base rate (~20%)
            last_precip = max(h.precipitation_probability for h in tomorrow_hours)
            day2_precip_prob = round(last_precip * 0.7 + 20 * 0.3)

            # Condition: if tomorrow had precip, day+2 is likely just cloudy
            day2_condition = self._worst_condition(tomorrow_hours)
            if day2_condition == "clear-night":
                day2_condition = "sunny"
            # Regress severe conditions toward milder
            day2_condition = _CONDITION_DECAY.get(day2_condition, day2_condition)

            # Humidity regresses toward 55% (continental mean)
            day2_humidity = round(
                (sum(h.humidity for h in tomorrow_hours) / len(tomorrow_hours))
                * 0.7 + 55 * 0.3
            )

            # Pressure continues gentle trend
            day2_pressure = round(last.pressure + last.pressure - tomorrow_hours[0].pressure, 1)
            day2_pressure = max(920.0, min(1070.0, day2_pressure))

            day2_date = now.date() + timedelta(days=2)
            dt_day2 = datetime(
                day2_date.year, day2_date.month, day2_date.day, 12, 0, 0,
                tzinfo=now.tzinfo,
            )

            days.append(
                Forecast(  # type: ignore[typeddict-unknown-key]
                    datetime=dt_day2.isoformat(),
                    condition=day2_condition,
                    native_temperature=day2_temp_high,
                    native_templow=day2_temp_low,
                    precipitation_probability=day2_precip_prob,
                    native_precipitation=round(
                        sum(h.precipitation_amount for h in tomorrow_hours) * 0.7, 1
                    ),
                    humidity=day2_humidity,
                    native_pressure=day2_pressure,
                    native_wind_speed=round(last.wind_speed * 0.8 + 0.5, 1),
                    wind_bearing=round(last.wind_bearing),
                    is_daytime=True,
                )
            )

        _LOGGER.debug("Daily forecast: %d day(s)", len(days))
        return days or None

    @staticmethod
    def _worst_condition(hours: list[HourForecast]) -> str:
        """Pick the most severe condition from a set of hourly forecasts.

        Severity order (what matters to a non-technical family):
        lightning-rainy > exceptional > pouring > snowy > snowy-rainy >
        rainy > fog > windy > cloudy > partlycloudy > clear-night > sunny
        """
        if not hours:
            return "cloudy"
        return max(
            (h.condition for h in hours),
            key=lambda c: _CONDITION_SEVERITY.get(c, 0),
        )
