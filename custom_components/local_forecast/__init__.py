"""Local Weather Forecast — integration entry point."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback

from .const import CONF_ENABLE_MAP, DATA_MAP, DEFAULT_ENABLE_MAP, DOMAIN
from .coordinator import LocalForecastConfigEntry, LocalForecastCoordinator
from .map import (
    LocalForecastMapTimesView,
    LocalForecastMapView,
    MapState,
    async_register_static_assets,
)

PLATFORMS: list[Platform] = [Platform.WEATHER, Platform.SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: LocalForecastConfigEntry) -> bool:
    """Set up Local Weather Forecast from a config entry."""
    coordinator = LocalForecastCoordinator(hass, entry)
    await coordinator.async_load_pressure_history()
    await coordinator.async_config_entry_first_refresh()
    coordinator.async_track_sources(entry)
    entry.runtime_data = coordinator

    await _async_setup_map(hass)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: LocalForecastConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        _async_refresh_map_gate(hass, skip_entry_id=entry.entry_id)
    return unload_ok


async def _async_setup_map(hass: HomeAssistant) -> None:
    """Serve the pan/zoom satellite view while any entry asks for it.

    The HTTP views are registered once for the lifetime of Home Assistant —
    aiohttp routes cannot be removed — and gate themselves on the live
    ``enabled`` flag, so toggling the option off makes the endpoint 404.
    """
    state = _async_refresh_map_gate(hass)
    if not state.enabled or state.views_registered:
        return
    await async_register_static_assets(hass)
    hass.http.register_view(LocalForecastMapView(hass))
    hass.http.register_view(LocalForecastMapTimesView(hass))
    state.views_registered = True


@callback
def _async_refresh_map_gate(hass: HomeAssistant, skip_entry_id: str | None = None) -> MapState:
    """Recompute the map gate from the config entries that remain."""
    state = hass.data.setdefault(DATA_MAP, MapState())
    state.enabled = any(
        _map_enabled(entry) for entry in hass.config_entries.async_entries(DOMAIN) if entry.entry_id != skip_entry_id
    )
    return state


def _map_enabled(entry: ConfigEntry) -> bool:
    """Return whether the optional satellite map is enabled for this entry."""
    config = entry.options or entry.data
    return bool(config.get(CONF_ENABLE_MAP, DEFAULT_ENABLE_MAP))
