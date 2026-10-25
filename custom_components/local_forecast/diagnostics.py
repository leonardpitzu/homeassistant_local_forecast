"""Diagnostics for Local Weather Forecast: what the station has learned."""

from __future__ import annotations

import time
from typing import Any

from homeassistant.core import HomeAssistant

from .coordinator import LocalForecastConfigEntry


async def async_get_config_entry_diagnostics(hass: HomeAssistant, entry: LocalForecastConfigEntry) -> dict[str, Any]:
    """Return the learned tide, the history held and the self-correction."""
    coordinator = entry.runtime_data
    forecaster = coordinator.forecaster
    return {
        "config": dict(entry.options or entry.data),
        "pressure_history_samples": len(coordinator.pressure_history.dump()),
        "tide": coordinator.tide.describe(time.time()),
        "history_hours": len(coordinator.history.dump()),
        "self_correction": forecaster.dump() if forecaster else None,
        "last_issue": forecaster.issue.ts if forecaster and forecaster.issue else None,
    }
