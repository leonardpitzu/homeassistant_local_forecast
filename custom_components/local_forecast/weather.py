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
from typing import Any, Final

from homeassistant.components.weather import (
    Forecast,
    WeatherEntity,
    WeatherEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
)
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util import dt as dt_util

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
    KELVIN_OFFSET,
    LAPSE_RATE,
    PRESSURE_RELATIVE,
)
from .physics_models import HumidityModel, PressureModel, TemperatureModel
from .state_estimator import SensorReading, StateEstimator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Local Weather Forecast weather entity."""
    entity = LocalForecastWeather(hass, entry)
    hass.data[DOMAIN][entry.entry_id]["weather_entity"] = entity
    async_add_entities([entity], True)


# ---------------------------------------------------------------------------
#  Weather Entity
# ---------------------------------------------------------------------------

class LocalForecastWeather(WeatherEntity):
    """Bayesian local weather forecast entity."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_native_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_native_pressure_unit = UnitOfPressure.HPA
    _attr_native_wind_speed_unit = UnitOfSpeed.METERS_PER_SECOND
    _attr_supported_features = (
        WeatherEntityFeature.FORECAST_HOURLY
        | WeatherEntityFeature.FORECAST_DAILY
    )

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_weather"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Local Weather Forecast",
            manufacturer="Local Weather Forecast",
            model="Bayesian Forecaster",
            sw_version="1.0.0",
        )

        # Core pipeline
        self._estimator = StateEstimator()
        self._forecaster = BayesianForecaster()

        # Cached forecasts
        self._hourly: list[HourForecast] = []
        self._last_update: float = 0
        self._min_interval: float = 30.0  # seconds between full recalculations

    # ------------------------------------------------------------------
    #  Config helpers
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default: Any = None) -> Any:
        return self._entry.options.get(key, self._entry.data.get(key, default))

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

        ids = self._sensor_ids()
        if ids:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, ids, self._on_sensor_change
                )
            )

    @callback
    def _on_sensor_change(self, event: Event) -> None:
        """Handle sensor state change — throttled."""
        now = time.monotonic()
        if now - self._last_update < self._min_interval:
            return
        self._last_update = now
        self.hass.async_create_task(self._async_recalculate())

    async def _async_recalculate(self) -> None:
        """Run the full pipeline and write state."""
        self._ingest_sensors()
        self._run_forecast()
        self.async_write_ha_state()
        await self.async_update_listeners(None)

    async def async_update(self) -> None:
        """Periodic update (if HA polls us)."""
        self._ingest_sensors()
        self._run_forecast()
        await self.async_update_listeners(None)

    # ------------------------------------------------------------------
    #  Sensor ingestion
    # ------------------------------------------------------------------

    def _ingest_sensors(self) -> None:
        """Read all configured sensors and feed a SensorReading."""
        pressure = self._read_float(CONF_PRESSURE_SENSOR)
        temperature = self._read_float(CONF_TEMPERATURE_SENSOR)
        if pressure is None or temperature is None:
            return

        # QFE → QNH conversion if absolute pressure
        if self._cfg(CONF_PRESSURE_TYPE, DEFAULT_PRESSURE_TYPE) != PRESSURE_RELATIVE:
            elevation = self._cfg(CONF_ELEVATION, DEFAULT_ELEVATION)
            if elevation and elevation > 0:
                temp_kelvin = max(200.0, temperature + KELVIN_OFFSET)
                pressure = pressure * (
                    1 - LAPSE_RATE * elevation / temp_kelvin
                ) ** -GRAVITY_EXPONENT

        reading = SensorReading(
            timestamp=time.time(),
            pressure_hpa=pressure,
            temperature_c=temperature,
            humidity_pct=self._read_float(CONF_HUMIDITY_SENSOR),
            wind_speed_ms=self._read_float(CONF_WIND_SPEED_SENSOR),
            wind_direction_deg=self._read_float(CONF_WIND_DIRECTION_SENSOR),
            solar_radiation_wm2=self._read_float(CONF_SOLAR_RADIATION_SENSOR),
            rain_rate_mmh=self._read_float(CONF_RAIN_RATE_SENSOR),
        )
        self._estimator.update(reading)

    def _read_float(self, config_key: str) -> float | None:
        sid = self._cfg(config_key)
        if not sid:
            return None
        state = self.hass.states.get(sid)
        if state is None or state.state in ("unknown", "unavailable", ""):
            return None
        try:
            val = float(state.state)
        except (ValueError, TypeError):
            return None

        # Auto-convert common units
        unit = (state.attributes.get("unit_of_measurement") or "").lower()
        if config_key == CONF_PRESSURE_SENSOR:
            if "inhg" in unit:
                val *= 33.8639
            elif "mmhg" in unit:
                val *= 1.33322
            elif "kpa" in unit:
                val *= 10.0
            elif "psi" in unit:
                val *= 68.9476
        elif config_key == CONF_TEMPERATURE_SENSOR:
            if "°f" in unit or "fahrenheit" in unit.lower():
                val = (val - 32) * 5 / 9
        elif config_key == CONF_WIND_SPEED_SENSOR:
            if "km/h" in unit or "kph" in unit:
                val /= 3.6
            elif "mph" in unit:
                val *= 0.44704
            elif "kn" in unit or "knot" in unit:
                val *= 0.51444

        # Reject physically impossible values after conversion
        if config_key == CONF_PRESSURE_SENSOR and not (870.0 <= val <= 1090.0):
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
        s = self._estimator.state

        # Day/night from sun entity
        sunrise_h, sunset_h = self._sun_hours()
        now_h = dt_util.now().hour + dt_util.now().minute / 60.0
        s.is_night = not (sunrise_h <= now_h < sunset_h)

        # Current state classification
        current_idx = self._estimator.classify()

        _LOGGER.debug(
            "Pipeline: P=%.1f dp/dt=%.2f T=%.1f RH=%.0f wind=%.1f "
            "state=%s night=%s",
            s.pressure, s.dp_dt, s.temperature, s.humidity,
            s.wind_speed, HA_CONDITIONS[current_idx], s.is_night,
        )

        # Cloud fraction for temperature model
        cloud = self._estimator._estimate_cloud_fraction()

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
        )

        if self._hourly:
            h1 = self._hourly[0]
            _LOGGER.debug(
                "Forecast: %d hours, +1h=%s %.1f°C %d%% precip",
                len(self._hourly), h1.condition,
                h1.temperature, h1.precipitation_probability,
            )

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
    def condition(self) -> str | None:
        return HA_CONDITIONS[self._estimator.classify()]

    @property
    def native_temperature(self) -> float | None:
        return round(self._estimator.state.temperature, 1)

    @property
    def humidity(self) -> float | None:
        return round(self._estimator.state.humidity)

    @property
    def native_pressure(self) -> float | None:
        return round(self._estimator.state.pressure, 1)

    @property
    def native_wind_speed(self) -> float | None:
        return round(self._estimator.state.wind_speed, 1)

    @property
    def wind_bearing(self) -> float | None:
        return round((self._estimator.state.wind_direction + 360) % 360)

    @property
    def native_apparent_temperature(self) -> float | None:
        """Apparent (feels-like) temperature — wind chill or heat index."""
        s = self._estimator.state
        T, W, RH = s.temperature, s.wind_speed, s.humidity
        # Wind chill (Environment Canada formula, T < 10 °C, W > 4.8 km/h)
        W_kmh = W * 3.6
        if T <= 10.0 and W_kmh > 4.8:
            wc = (
                13.12 + 0.6215 * T
                - 11.37 * W_kmh ** 0.16
                + 0.3965 * T * W_kmh ** 0.16
            )
            return round(wc, 1)
        # Heat index (Steadman, T > 27 °C)
        if T >= 27.0 and RH >= 40:
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

    @staticmethod
    def _beaufort(wind_ms: float) -> int:
        """Convert wind speed in m/s to Beaufort scale (0-12)."""
        # WMO thresholds (m/s): 0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9,
        #                        17.2, 20.8, 24.5, 28.5, 32.7
        thresholds = (0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9,
                      17.2, 20.8, 24.5, 28.5, 32.7)
        for i, t in enumerate(thresholds):
            if wind_ms < t:
                return i
        return 12

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        s = self._estimator.state
        attrs: dict[str, Any] = {
            "pressure_trend": round(s.dp_dt, 2),
            "pressure_acceleration": round(s.d2p_dt2, 2),
            "dew_point": s.dew_point,
            "dew_depression": s.dew_depression,
            "wet_bulb": s.wet_bulb,
            "wind_force": self._beaufort(s.wind_speed),
            "wind_force_description": self._BEAUFORT_NAMES[self._beaufort(s.wind_speed)],
            "front_warm": s.front_warm,
            "front_cold": s.front_cold,
            "front_occluded": s.front_occluded,
        }
        if self._hourly:
            h1 = self._hourly[0]
            attrs["next_hour_condition"] = h1.condition
            attrs["next_hour_precip_probability"] = h1.precipitation_probability
            # Aggregate precipitation probability over next 6 h
            if len(self._hourly) >= 6:
                # Probability of at least one rainy hour in next 6
                p_no_rain = 1.0
                for hf in self._hourly[:6]:
                    p_no_rain *= 1.0 - hf.precipitation_probability / 100.0
                attrs["precip_probability_6h"] = round((1 - p_no_rain) * 100)
        return attrs

    # ------------------------------------------------------------------
    #  Forecast services (what weather cards call)
    # ------------------------------------------------------------------

    async def async_forecast_hourly(self) -> list[Forecast] | None:
        if not self._hourly:
            return None

        now = dt_util.now()
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
                    is_daytime=hf.condition != "clear-night",
                )
            )
        return result

    async def async_forecast_daily(self) -> list[Forecast] | None:
        """Aggregate hourly into today / tomorrow / day-after-tomorrow.

        With a 12-hour horizon, tomorrow and day+2 may share the same
        tail hours.  This is the best we can do without a cloud API.
        """
        if not self._hourly:
            return None

        now = dt_util.now()
        hours_left_today = 24 - now.hour
        today_hours = [h for h in self._hourly if h.hours_ahead <= hours_left_today]
        tomorrow_hours = [h for h in self._hourly if h.hours_ahead > hours_left_today]

        # If all forecast hours fall in today, use the tail as proxy
        if not tomorrow_hours:
            tomorrow_hours = self._hourly[-min(6, len(self._hourly)):]
        if not today_hours:
            today_hours = self._hourly[:min(6, len(self._hourly))]

        # Day+2: reuse the last few hours as best-guess extrapolation
        day2_hours = self._hourly[-min(4, len(self._hourly)):]

        days: list[Forecast] = []

        for hours, offset_days in [
            (today_hours, 0),
            (tomorrow_hours, 1),
            (day2_hours, 2),
        ]:
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

        _LOGGER.debug("Daily forecast: %d day(s)", len(days))
        return days or None

    @staticmethod
    def _worst_condition(hours: list[HourForecast]) -> str:
        """Pick the most severe condition from a set of hourly forecasts.

        Severity order (what matters to a non-technical family):
        lightning-rainy > exceptional > pouring > snowy > snowy-rainy >
        rainy > fog > windy > cloudy > partlycloudy > clear-night > sunny
        """
        severity = {
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
        if not hours:
            return "cloudy"
        return max(
            (h.condition for h in hours),
            key=lambda c: severity.get(c, 0),
        )
