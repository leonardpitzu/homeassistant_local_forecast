"""Diagnostics for Local Weather Forecast: what the tide model has learned."""

from __future__ import annotations

import time
from typing import Any

from homeassistant.core import HomeAssistant

from .coordinator import LocalForecastConfigEntry


async def async_get_config_entry_diagnostics(hass: HomeAssistant, entry: LocalForecastConfigEntry) -> dict[str, Any]:
    """Return the learned tide and the state of the pressure buffer."""
    coordinator = entry.runtime_data
    return {
        "config": dict(entry.options or entry.data),
        "pressure_history_samples": len(coordinator.pressure_history.dump()),
        "tide": coordinator.tide.describe(time.time()),
    }
