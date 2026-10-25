"""Weather entity for Local Weather Forecast.

The dashboard face of the integration: condition, current values, and the
hourly / daily forecasts weather cards ask for.  All numbers come from the
coordinator; this module only shapes them for Home Assistant.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging

from homeassistant.components.weather import (
    Forecast,
    WeatherEntity,
    WeatherEntityFeature,
)
from homeassistant.const import UnitOfPressure, UnitOfSpeed, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .coordinator import (
    ForecastResult,
    LocalForecastConfigEntry,
    LocalForecastCoordinator,
)
from .forecast import HourForecast

_LOGGER = logging.getLogger(__name__)

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


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LocalForecastConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Local Weather Forecast weather entity."""
    async_add_entities([LocalForecastWeather(entry.runtime_data)])


class LocalForecastWeather(CoordinatorEntity[LocalForecastCoordinator], WeatherEntity):
    """Local weather forecast entity."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_native_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_native_pressure_unit = UnitOfPressure.HPA
    _attr_native_wind_speed_unit = UnitOfSpeed.METERS_PER_SECOND
    _attr_supported_features = WeatherEntityFeature.FORECAST_HOURLY | WeatherEntityFeature.FORECAST_DAILY

    def __init__(self, coordinator: LocalForecastCoordinator) -> None:
        """Bind the entity to its coordinator and device."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.config_entry.entry_id}_weather"
        self._attr_device_info = coordinator.device_info

    @property
    def _data(self) -> ForecastResult | None:
        return self.coordinator.data

    @callback
    def _handle_coordinator_update(self) -> None:
        """Push both the state and any live forecast subscription."""
        self.coordinator.config_entry.async_create_task(self.hass, self.async_update_listeners(None), eager_start=False)
        super()._handle_coordinator_update()

    # ------------------------------------------------------------------
    #  Current conditions
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Only claim a state once real sensor data has been ingested."""
        return super().available and self._data is not None

    @property
    def condition(self) -> str | None:
        return self._data.condition if self._data else None

    @property
    def native_temperature(self) -> float | None:
        return self._data.temperature if self._data else None

    @property
    def native_apparent_temperature(self) -> float | None:
        return self._data.apparent_temperature if self._data else None

    @property
    def native_pressure(self) -> float | None:
        return self._data.pressure if self._data else None

    @property
    def humidity(self) -> float | None:
        return self._data.humidity if self._data else None

    @property
    def native_wind_speed(self) -> float | None:
        return self._data.wind_speed if self._data else None

    @property
    def wind_bearing(self) -> float | None:
        return self._data.wind_bearing if self._data else None

    @property
    def native_dew_point(self) -> float | None:
        return self._data.dew_point if self._data else None

    @property
    def extra_state_attributes(self) -> dict:
        """Attributes built once per pipeline run (Developer Tools, templates)."""
        return self._data.attributes if self._data else {}

    # ------------------------------------------------------------------
    #  Forecast services (what weather cards call)
    # ------------------------------------------------------------------

    async def async_forecast_hourly(self) -> list[Forecast] | None:
        """Return the 24-hour forecast."""
        if not (data := self._data) or not data.hourly:
            return None

        # Anchor on when the forecast was computed, not when a card happens to
        # ask, or every entry drifts by up to one refresh interval.
        return [
            Forecast(  # type: ignore[typeddict-unknown-key]
                datetime=(data.generated + timedelta(hours=hf.hours_ahead)).isoformat(),
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
            for hf in data.hourly
        ]

    async def async_forecast_daily(self) -> list[Forecast] | None:
        """Today, tomorrow and the day after.

        Today combines what has been measured since midnight with the hours
        still to come; tomorrow and the day after come from the daily models.
        """
        if not (data := self._data) or not data.hourly:
            return None

        now = data.generated
        today = [h for h in data.hourly if (now + timedelta(hours=h.hours_ahead)).date() == now.date()]
        if not today:  # the last minutes before midnight
            today = data.hourly[:1]
        temps = [h.temperature for h in today]
        if data.today_observed is not None:
            temps += list(data.today_observed)
        condition = _worst_condition(today)
        days: list[Forecast] = [
            Forecast(  # type: ignore[typeddict-unknown-key]
                datetime=now.replace(microsecond=0).isoformat(),
                # Daily forecasts are always daytime.
                condition="sunny" if condition == "clear-night" else condition,
                native_temperature=round(max(temps), 1),
                native_templow=round(min(temps), 1),
                precipitation_probability=max(h.precipitation_probability for h in today),
                native_precipitation=round(sum(h.precipitation_amount for h in today), 1),
                humidity=_mean([h.humidity for h in today]),
                native_pressure=_mean([h.pressure for h in today], 1),
                native_wind_speed=_mean([h.wind_speed for h in today], 1),
                wind_bearing=today[0].wind_bearing,
                is_daytime=True,
            )
        ]

        for day in data.daily:
            date = datetime.fromordinal(day.date).date()
            hours = [h for h in data.hourly if (now + timedelta(hours=h.hours_ahead)).date() == date]
            # Humidity and pressure have no daily model: the forecast hours that
            # reach into the day stand for it.
            near = hours or data.hourly[-1:]
            days.append(
                Forecast(  # type: ignore[typeddict-unknown-key]
                    datetime=datetime(date.year, date.month, date.day, 12, 0, 0, tzinfo=now.tzinfo).isoformat(),
                    condition=day.condition,
                    native_temperature=day.temperature,
                    native_templow=day.templow,
                    precipitation_probability=day.precipitation_probability,
                    native_precipitation=day.precipitation_amount,
                    humidity=_mean([h.humidity for h in near]),
                    native_pressure=_mean([h.pressure for h in near], 1),
                    native_wind_speed=day.wind_speed,
                    wind_bearing=today[0].wind_bearing,
                    is_daytime=True,
                )
            )

        _LOGGER.debug("Daily forecast: %d day(s)", len(days))
        return days


def _mean(values: list[float | None], ndigits: int | None = None) -> float | None:
    """Rounded mean of one channel, or None when that channel is not measured."""
    if not values or None in values:
        return None
    return round(sum(values) / len(values), ndigits)


def _worst_condition(hours: list[HourForecast]) -> str:
    """Pick the most severe condition from a set of hourly forecasts.

    Severity order (what matters to a non-technical family):
    lightning-rainy > exceptional > pouring > snowy > snowy-rainy > rainy >
    fog > windy > cloudy > partlycloudy > clear-night > sunny
    """
    if not hours:
        return "cloudy"
    return max(
        (h.condition for h in hours),
        key=lambda c: _CONDITION_SEVERITY.get(c, 0),
    )
