"""Sensor platform for Local Weather Forecast.

Exposes key forecast attributes as standalone sensor entities so they
have their own history, can be graphed in tile cards, and used in badges.
"""

from __future__ import annotations

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event

from .const import DOMAIN

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


def _precip_icon(wet_bulb: float | None) -> str:
    """Pick rain/snow/sleet icon based on wet-bulb temperature."""
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

    device_info = DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
    )

    async_add_entities(
        [
            PrecipProbability6hSensor(entry, weather_entity, device_info),
            NextHourConditionSensor(entry, weather_entity, device_info),
            NextHourPrecipProbabilitySensor(entry, weather_entity, device_info),
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
        return _precip_icon(attrs.get("wet_bulb"))


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
        return _precip_icon(attrs.get("wet_bulb"))
