"""Binary sensor platform for Local Weather Forecast.

Promotes the weather entity's frontal-detection attributes (warm / cold /
occluded front) to standalone binary sensors so they can annotate the
pressure chart and drive automations.  Disabled by default — they are for
meteorology nerds, not the average dashboard.
"""

from __future__ import annotations

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event

from .const import DOMAIN


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Local Weather Forecast binary sensor entities."""
    data = hass.data[DOMAIN].get(entry.entry_id, {})
    weather_entity = data.get("weather_entity")
    if weather_entity is None:
        return

    device_info = DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
    )

    async_add_entities(
        [
            FrontBinarySensor(
                entry, weather_entity, device_info, "front_warm"
            ),
            FrontBinarySensor(
                entry, weather_entity, device_info, "front_cold"
            ),
            FrontBinarySensor(
                entry, weather_entity, device_info, "front_occluded"
            ),
        ],
        True,
    )


class FrontBinarySensor(BinarySensorEntity):
    """A single frontal-detection flag from the weather entity."""

    _attr_has_entity_name = True
    _attr_entity_registry_enabled_default = False

    def __init__(
        self,
        entry: ConfigEntry,
        weather_entity,
        device_info: DeviceInfo,
        attribute: str,
    ) -> None:
        self._weather = weather_entity
        self._attribute = attribute
        self._attr_device_info = device_info
        self._attr_translation_key = attribute
        self._attr_unique_id = f"{entry.entry_id}_{attribute}"

    async def async_added_to_hass(self) -> None:
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
        self.async_schedule_update_ha_state(True)

    @property
    def is_on(self) -> bool | None:
        return bool(
            self._weather.extra_state_attributes.get(self._attribute)
        )
