"""Local Weather Forecast — integration entry point."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import CONF_ENABLE_MAP, DEFAULT_ENABLE_MAP, DOMAIN
from .map import LocalForecastMapView
from .pressure_history import PressureHistory

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.WEATHER,
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
]


def _map_enabled(entry: ConfigEntry) -> bool:
    """Return whether the optional satellite map is enabled for this entry."""
    return entry.options.get(
        CONF_ENABLE_MAP, entry.data.get(CONF_ENABLE_MAP, DEFAULT_ENABLE_MAP)
    )


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Local Weather Forecast from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        "config": entry.data,
        # Shared hourly sea-level-pressure buffer feeding the tendency,
        # synoptic-mean and barometer sensors.  The weather entity records
        # into it; the pressure-tendency sensor persists it across restarts.
        "pressure_history": PressureHistory(),
    }

    # Optional satellite map: serve the pan/zoom view only while at least one
    # entry enables it. The HTTP view is registered once for the lifetime of
    # Home Assistant (aiohttp routes cannot be removed) and gates itself on the
    # live `map_enabled` flag, so toggling it off makes the endpoint 404.
    hass.data[DOMAIN]["map_enabled"] = any(
        _map_enabled(e) for e in hass.config_entries.async_entries(DOMAIN)
    )
    if hass.data[DOMAIN]["map_enabled"] and not hass.data[DOMAIN].get(
        "map_view_registered"
    ):
        hass.http.register_view(LocalForecastMapView(hass))
        hass.data[DOMAIN]["map_view_registered"] = True

    # Weather must be set up before Sensor/Binary Sensor (they read weather_entity)
    await hass.config_entries.async_forward_entry_setups(entry, [Platform.WEATHER])
    await hass.config_entries.async_forward_entry_setups(
        entry, [Platform.SENSOR, Platform.BINARY_SENSOR]
    )
    entry.async_on_unload(entry.add_update_listener(_async_reload))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        hass.data[DOMAIN].pop(entry.entry_id, None)
        # Recompute the map flag from the entries that remain.
        hass.data[DOMAIN]["map_enabled"] = any(
            _map_enabled(e)
            for e in hass.config_entries.async_entries(DOMAIN)
            if e.entry_id != entry.entry_id
        )
    return unload_ok


async def _async_reload(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload on options change."""
    await hass.config_entries.async_reload(entry.entry_id)
