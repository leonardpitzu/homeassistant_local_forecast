"""Sensor platform for Local Weather Forecast.

Exposes key forecast attributes as standalone sensor entities so they
have their own history, can be graphed in tile cards, and used in badges.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfPressure
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.restore_state import ExtraStoredData, RestoreEntity

from .const import DOMAIN
from .pressure_history import PressureHistory

# Sea-level pressure bands (hPa) for the barometer enum, low → high.
_BAROMETER_OPTIONS = ["stormy", "rain", "change", "fair", "very_dry"]
_BAROMETER_BANDS = (985.0, 1005.0, 1020.0, 1035.0)
# Tendency (hPa/h) beyond which the needle is read one category up/down.
_BAROMETER_TENDENCY_STEP = 0.5


def _barometer_state(
    pressure: float | None, tendency: float | None
) -> str | None:
    """Classify sea-level pressure into a tendency-aware barometer category."""
    if pressure is None:
        return None
    band = 0
    for threshold in _BAROMETER_BANDS:
        if pressure >= threshold:
            band += 1
    if tendency is not None:
        if tendency <= -_BAROMETER_TENDENCY_STEP:
            band -= 1
        elif tendency >= _BAROMETER_TENDENCY_STEP:
            band += 1
    band = max(0, min(len(_BAROMETER_OPTIONS) - 1, band))
    return _BAROMETER_OPTIONS[band]


# Pressure-tendency direction enum (companion to the numeric sensor).
# Thresholds in hPa/h: |t| < 0.3 steady, 0.3-1.0 slow, >= 1.0 fast.
_TENDENCY_DIRECTION_OPTIONS = [
    "falling_fast",
    "falling",
    "steady",
    "rising",
    "rising_fast",
]
_TENDENCY_STEADY = 0.3
_TENDENCY_FAST = 1.0


def _tendency_direction(tendency: float | None) -> str | None:
    """Classify a pressure tendency (hPa/h) into a direction enum."""
    if tendency is None:
        return None
    if tendency >= _TENDENCY_FAST:
        return "rising_fast"
    if tendency >= _TENDENCY_STEADY:
        return "rising"
    if tendency <= -_TENDENCY_FAST:
        return "falling_fast"
    if tendency <= -_TENDENCY_STEADY:
        return "falling"
    return "steady"


# Frontal-passage identity as a single enum.  The three signatures can
# overlap; report the most-developed one (occluded > cold > warm).
_FRONT_OPTIONS = ["none", "warm", "cold", "occluded"]


def _front_state(
    warm: bool | None, cold: bool | None, occluded: bool | None
) -> str:
    """Collapse the three frontal flags into one mutually-exclusive state."""
    if occluded:
        return "occluded"
    if cold:
        return "cold"
    if warm:
        return "warm"
    return "none"

# HA condition string → human-readable label
_CONDITION_LABELS: dict[str, str] = {
    "sunny": "Sunny",
    "clear-night": "Clear Night",
    "partlycloudy": "Partly Cloudy",
    "cloudy": "Cloudy",
    "fog": "Fog",
    "rainy": "Rainy",
    "pouring": "Pouring",
    "snowy": "Snowy",
    "snowy-rainy": "Snowy Rainy",
    "lightning-rainy": "Lightning Rainy",
    "windy": "Windy",
    "exceptional": "Exceptional",
}

# HA condition string → MDI icon
_CONDITION_ICONS: dict[str, str] = {
    "sunny": "mdi:weather-sunny",
    "clear-night": "mdi:weather-night",
    "partlycloudy": "mdi:weather-partly-cloudy",
    "cloudy": "mdi:weather-cloudy",
    "fog": "mdi:weather-fog",
    "rainy": "mdi:weather-rainy",
    "pouring": "mdi:weather-pouring",
    "snowy": "mdi:weather-snowy",
    "snowy-rainy": "mdi:weather-snowy-rainy",
    "lightning-rainy": "mdi:weather-lightning-rainy",
    "windy": "mdi:weather-windy",
    "exceptional": "mdi:alert-circle-outline",
}


# Below this probability (%) the precip badge shows a neutral icon
# instead of an alarming rain cloud under fair skies.
_DRY_PROBABILITY_THRESHOLD = 20


def _precip_icon(wet_bulb: float | None, probability: float | None = None) -> str:
    """Pick the precipitation icon.

    With a negligible probability, return a neutral 'dry' icon; otherwise
    choose rain / sleet / snow from the wet-bulb temperature.
    """
    if probability is not None and probability < _DRY_PROBABILITY_THRESHOLD:
        return "mdi:weather-partly-cloudy"
    if wet_bulb is None:
        return "mdi:weather-rainy"
    if wet_bulb < -2.0:
        return "mdi:weather-snowy"
    if wet_bulb < 1.0:
        return "mdi:weather-snowy-rainy"
    return "mdi:weather-rainy"


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Local Weather Forecast sensor entities."""
    data = hass.data[DOMAIN].get(entry.entry_id, {})
    weather_entity = data.get("weather_entity")
    if weather_entity is None:
        return

    pressure_history: PressureHistory = data["pressure_history"]

    device_info = DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
    )

    async_add_entities(
        [
            PrecipProbability6hSensor(entry, weather_entity, device_info),
            NextHourConditionSensor(entry, weather_entity, device_info),
            NextHourPrecipProbabilitySensor(entry, weather_entity, device_info),
            SeaLevelPressureSensor(entry, weather_entity, device_info),
            PressureTendencySensor(
                entry, weather_entity, device_info, pressure_history
            ),
            PressureTendencyDirectionSensor(
                entry, weather_entity, device_info, pressure_history
            ),
            PressureSynopticSensor(
                entry, weather_entity, device_info, pressure_history
            ),
            BarometerSensor(
                entry, weather_entity, device_info, pressure_history
            ),
            HourlyForecastSensor(entry, weather_entity, device_info),
            FrontSensor(entry, weather_entity, device_info),
        ],
        True,
    )


class _ForecastSensorBase(SensorEntity):
    """Base class for forecast sensors that read from the weather entity."""

    _attr_has_entity_name = True

    def __init__(
        self,
        entry: ConfigEntry,
        weather_entity,
        device_info: DeviceInfo,
    ) -> None:
        self._weather = weather_entity
        self._attr_device_info = device_info

    async def async_added_to_hass(self) -> None:
        """Update when the parent weather entity updates."""
        await super().async_added_to_hass()
        weather_id = self._weather.entity_id
        if weather_id:
            self.async_on_remove(
                async_track_state_change_event(
                    self.hass, [weather_id], self._on_weather_update
                )
            )

    @callback
    def _on_weather_update(self, _event) -> None:
        """Re-read value when the weather entity state changes."""
        self.async_schedule_update_ha_state(True)


class PrecipProbability6hSensor(_ForecastSensorBase):
    """Probability of precipitation in the next 6 hours."""

    _attr_name = "Precipitation probability"
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, entry, weather_entity, device_info) -> None:
        super().__init__(entry, weather_entity, device_info)
        self._attr_unique_id = f"{entry.entry_id}_precip_prob_6h"

    @property
    def native_value(self) -> int | None:
        attrs = self._weather.extra_state_attributes
        return attrs.get("precip_probability_6h")

    @property
    def icon(self) -> str:
        attrs = self._weather.extra_state_attributes
        return _precip_icon(
            attrs.get("wet_bulb"), attrs.get("precip_probability_6h")
        )


class NextHourConditionSensor(_ForecastSensorBase):
    """Forecast condition for the next hour."""

    _attr_name = "1h forecast"

    def __init__(self, entry, weather_entity, device_info) -> None:
        super().__init__(entry, weather_entity, device_info)
        self._attr_unique_id = f"{entry.entry_id}_next_hour_condition"

    @property
    def native_value(self) -> str | None:
        attrs = self._weather.extra_state_attributes
        val = attrs.get("next_hour_condition")
        if val:
            return _CONDITION_LABELS.get(val, val.replace("-", " ").title())
        return None

    @property
    def icon(self) -> str:
        attrs = self._weather.extra_state_attributes
        condition = attrs.get("next_hour_condition", "")
        return _CONDITION_ICONS.get(condition, "mdi:weather-partly-cloudy")


class NextHourPrecipProbabilitySensor(_ForecastSensorBase):
    """Precipitation probability for the next hour."""

    _attr_name = "Next hour precipitation probability"
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, entry, weather_entity, device_info) -> None:
        super().__init__(entry, weather_entity, device_info)
        self._attr_unique_id = f"{entry.entry_id}_next_hour_precip_prob"

    @property
    def native_value(self) -> int | None:
        attrs = self._weather.extra_state_attributes
        return attrs.get("next_hour_precip_probability")

    @property
    def icon(self) -> str:
        attrs = self._weather.extra_state_attributes
        return _precip_icon(
            attrs.get("wet_bulb"), attrs.get("next_hour_precip_probability")
        )


class SeaLevelPressureSensor(_ForecastSensorBase):
    """Sea-level (QNH) pressure — the value feeding the weather entity."""

    _attr_name = "Sea level pressure"
    _attr_device_class = SensorDeviceClass.ATMOSPHERIC_PRESSURE
    _attr_native_unit_of_measurement = UnitOfPressure.HPA
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 1

    def __init__(self, entry, weather_entity, device_info) -> None:
        super().__init__(entry, weather_entity, device_info)
        self._attr_unique_id = f"{entry.entry_id}_sea_level_pressure"

    @property
    def native_value(self) -> float | None:
        return self._weather.native_pressure


class PressureTendencySensor(_ForecastSensorBase, RestoreEntity):
    """WMO 3-hour pressure tendency (hPa/h).

    Owns persistence of the shared hourly pressure buffer so both this
    sensor and the synoptic mean survive a restart instead of warming up
    for three hours.
    """

    _attr_name = "Pressure tendency"
    _attr_native_unit_of_measurement = "hPa/h"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 2
    _attr_icon = "mdi:gauge"

    def __init__(
        self, entry, weather_entity, device_info, history: PressureHistory
    ) -> None:
        super().__init__(entry, weather_entity, device_info)
        self._history = history
        self._attr_unique_id = f"{entry.entry_id}_pressure_tendency"

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_extra_data()
        if last is not None:
            restored = _PressureHistoryExtraData.from_dict(last.as_dict())
            self._history.load(restored.samples)

    @property
    def extra_restore_state_data(self) -> ExtraStoredData:
        return _PressureHistoryExtraData(self._history.dump())

    @property
    def native_value(self) -> float | None:
        return self._history.tendency_per_hour(
            time.time(), self._weather.native_pressure
        )


class PressureTendencyDirectionSensor(_ForecastSensorBase):
    """Direction companion to the numeric tendency — an enum for badge icons."""

    _attr_translation_key = "pressure_tendency_direction"
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = _TENDENCY_DIRECTION_OPTIONS

    def __init__(
        self, entry, weather_entity, device_info, history: PressureHistory
    ) -> None:
        super().__init__(entry, weather_entity, device_info)
        self._history = history
        self._attr_unique_id = f"{entry.entry_id}_pressure_tendency_direction"

    @property
    def native_value(self) -> str | None:
        tendency = self._history.tendency_per_hour(
            time.time(), self._weather.native_pressure
        )
        return _tendency_direction(tendency)


class PressureSynopticSensor(_ForecastSensorBase):
    """24-hour rolling mean of sea-level pressure (hPa)."""

    _attr_name = "Pressure synoptic"
    _attr_device_class = SensorDeviceClass.ATMOSPHERIC_PRESSURE
    _attr_native_unit_of_measurement = UnitOfPressure.HPA
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 1
    _attr_icon = "mdi:gauge-low"

    def __init__(
        self, entry, weather_entity, device_info, history: PressureHistory
    ) -> None:
        super().__init__(entry, weather_entity, device_info)
        self._history = history
        self._attr_unique_id = f"{entry.entry_id}_pressure_synoptic"

    @property
    def native_value(self) -> float | None:
        return self._history.mean(time.time(), self._weather.native_pressure)


class BarometerSensor(_ForecastSensorBase):
    """Tendency-aware barometer needle as an enum."""

    _attr_translation_key = "barometer"
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = _BAROMETER_OPTIONS

    def __init__(
        self, entry, weather_entity, device_info, history: PressureHistory
    ) -> None:
        super().__init__(entry, weather_entity, device_info)
        self._history = history
        self._attr_unique_id = f"{entry.entry_id}_barometer"

    @property
    def native_value(self) -> str | None:
        pressure = self._weather.native_pressure
        tendency = self._history.tendency_per_hour(time.time(), pressure)
        return _barometer_state(pressure, tendency)


class HourlyForecastSensor(_ForecastSensorBase):
    """Meteogram feed: forecast generation time + full hourly list.

    The ``forecast`` attribute is excluded from the recorder — it is live
    data for cards, not history worth writing to the database.
    """

    _attr_name = "Hourly forecast"
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:chart-line"
    _unrecorded_attributes = frozenset({"forecast"})

    def __init__(self, entry, weather_entity, device_info) -> None:
        super().__init__(entry, weather_entity, device_info)
        self._attr_unique_id = f"{entry.entry_id}_hourly_forecast"

    @property
    def native_value(self) -> datetime | None:
        return self._weather.forecast_generated

    @property
    def extra_state_attributes(self) -> dict:
        return {"forecast": self._weather.hourly_forecast_list()}


class FrontSensor(_ForecastSensorBase):
    """Frontal-passage identity as a single mutually-exclusive enum."""

    _attr_translation_key = "front"
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = _FRONT_OPTIONS

    def __init__(self, entry, weather_entity, device_info) -> None:
        super().__init__(entry, weather_entity, device_info)
        self._attr_unique_id = f"{entry.entry_id}_front"

    @property
    def native_value(self) -> str:
        attrs = self._weather.extra_state_attributes
        return _front_state(
            attrs.get("front_warm"),
            attrs.get("front_cold"),
            attrs.get("front_occluded"),
        )


@dataclass
class _PressureHistoryExtraData(ExtraStoredData):
    """Persisted hourly pressure samples for RestoreEntity."""

    samples: list

    def as_dict(self) -> dict:
        return {"samples": self.samples}

    @classmethod
    def from_dict(cls, data: dict) -> "_PressureHistoryExtraData":
        return cls(data.get("samples", []))

