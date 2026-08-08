"""Sensor platform for Local Weather Forecast.

Exposes key forecast values as standalone sensor entities so they have their
own history, can be graphed in tile cards, and used in badges.  Every sensor
is the same class; only its description differs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import UnitOfPressure, UnitOfRatio
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .classifiers import (
    BAROMETER_OPTIONS,
    FRONT_OPTIONS,
    TENDENCY_DIRECTION_OPTIONS,
    barometer_state,
    front_state,
    tendency_direction,
)
from .const import DOMAIN
from .coordinator import (
    ForecastResult,
    LocalForecastConfigEntry,
    LocalForecastCoordinator,
)

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


def _precip_icon(wet_bulb: float | None, probability: float | None) -> str:
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


def _condition_label(data: ForecastResult) -> str | None:
    condition = data.attributes.get("next_hour_condition")
    if not condition:
        return None
    return _CONDITION_LABELS.get(condition, condition.replace("-", " ").title())


def _tendency(coordinator: LocalForecastCoordinator, data: ForecastResult):
    return coordinator.pressure_history.tendency_per_hour(time.time(), data.pressure)


@dataclass(frozen=True, kw_only=True)
class LocalForecastSensorEntityDescription(SensorEntityDescription):
    """Describes one Local Weather Forecast sensor.

    ``key`` doubles as the unique_id suffix; changing one renames an entity
    that has been on somebody's dashboard for years.
    """

    value_fn: Callable[[LocalForecastCoordinator, ForecastResult], StateType | datetime]
    icon_fn: Callable[[ForecastResult], str] | None = None
    attributes_fn: Callable[[ForecastResult], dict[str, Any]] | None = None


SENSORS: tuple[LocalForecastSensorEntityDescription, ...] = (
    LocalForecastSensorEntityDescription(
        key="precip_prob_6h",
        translation_key="precipitation_probability",
        native_unit_of_measurement=UnitOfRatio.PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda _c, d: d.attributes.get("precip_probability_6h"),
        icon_fn=lambda d: _precip_icon(d.attributes.get("wet_bulb"), d.attributes.get("precip_probability_6h")),
    ),
    LocalForecastSensorEntityDescription(
        key="next_hour_condition",
        translation_key="forecast_1h",
        value_fn=lambda _c, d: _condition_label(d),
        icon_fn=lambda d: _CONDITION_ICONS.get(
            d.attributes.get("next_hour_condition", ""), "mdi:weather-partly-cloudy"
        ),
    ),
    LocalForecastSensorEntityDescription(
        key="next_hour_precip_prob",
        translation_key="next_hour_precipitation_probability",
        native_unit_of_measurement=UnitOfRatio.PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda _c, d: d.attributes.get("next_hour_precip_probability"),
        icon_fn=lambda d: _precip_icon(
            d.attributes.get("wet_bulb"),
            d.attributes.get("next_hour_precip_probability"),
        ),
    ),
    LocalForecastSensorEntityDescription(
        key="sea_level_pressure",
        translation_key="sea_level_pressure",
        device_class=SensorDeviceClass.ATMOSPHERIC_PRESSURE,
        native_unit_of_measurement=UnitOfPressure.HPA,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=1,
        value_fn=lambda _c, d: d.pressure,
    ),
    LocalForecastSensorEntityDescription(
        key="pressure_tendency",
        translation_key="pressure_tendency",
        icon="mdi:gauge",
        native_unit_of_measurement="hPa/h",
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=2,
        value_fn=_tendency,
    ),
    LocalForecastSensorEntityDescription(
        key="pressure_tendency_direction",
        translation_key="pressure_tendency_direction",
        device_class=SensorDeviceClass.ENUM,
        options=TENDENCY_DIRECTION_OPTIONS,
        value_fn=lambda c, d: tendency_direction(_tendency(c, d)),
    ),
    LocalForecastSensorEntityDescription(
        key="pressure_synoptic",
        translation_key="pressure_synoptic",
        icon="mdi:gauge-low",
        device_class=SensorDeviceClass.ATMOSPHERIC_PRESSURE,
        native_unit_of_measurement=UnitOfPressure.HPA,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=1,
        value_fn=lambda c, d: c.pressure_history.mean(time.time(), d.pressure),
    ),
    LocalForecastSensorEntityDescription(
        key="barometer",
        translation_key="barometer",
        device_class=SensorDeviceClass.ENUM,
        options=BAROMETER_OPTIONS,
        value_fn=lambda c, d: barometer_state(d.pressure, _tendency(c, d)),
    ),
    LocalForecastSensorEntityDescription(
        key="hourly_forecast",
        translation_key="hourly_forecast",
        icon="mdi:chart-line",
        device_class=SensorDeviceClass.TIMESTAMP,
        value_fn=lambda _c, d: d.generated,
        attributes_fn=lambda d: {"forecast": d.hourly_dicts},
    ),
    LocalForecastSensorEntityDescription(
        key="front",
        translation_key="front",
        device_class=SensorDeviceClass.ENUM,
        options=FRONT_OPTIONS,
        value_fn=lambda _c, d: front_state(
            d.attributes.get("front_warm"),
            d.attributes.get("front_cold"),
            d.attributes.get("front_occluded"),
        ),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LocalForecastConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Local Weather Forecast sensor entities."""
    coordinator = entry.runtime_data
    async_add_entities(LocalForecastSensor(coordinator, description) for description in SENSORS)


class LocalForecastSensor(CoordinatorEntity[LocalForecastCoordinator], SensorEntity):
    """A single value taken from the latest forecast run."""

    _attr_has_entity_name = True
    entity_description: LocalForecastSensorEntityDescription
    # Only the hourly-forecast sensor carries it, and it is live data for
    # cards, not history worth writing to the database.
    _unrecorded_attributes = frozenset({"forecast"})

    def __init__(
        self,
        coordinator: LocalForecastCoordinator,
        description: LocalForecastSensorEntityDescription,
    ) -> None:
        """Bind the sensor to its coordinator and description."""
        super().__init__(coordinator)
        self.entity_description = description
        entry_id = coordinator.config_entry.entry_id
        self._attr_unique_id = f"{entry_id}_{description.key}"
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry_id)})

    @property
    def available(self) -> bool:
        return super().available and self.coordinator.data is not None

    @property
    def native_value(self) -> StateType | datetime:
        if (data := self.coordinator.data) is None:
            return None
        return self.entity_description.value_fn(self.coordinator, data)

    @property
    def icon(self) -> str | None:
        icon_fn = self.entity_description.icon_fn
        if icon_fn is None or (data := self.coordinator.data) is None:
            return super().icon
        return icon_fn(data)

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        attributes_fn = self.entity_description.attributes_fn
        if attributes_fn is None or (data := self.coordinator.data) is None:
            return None
        return attributes_fn(data)
