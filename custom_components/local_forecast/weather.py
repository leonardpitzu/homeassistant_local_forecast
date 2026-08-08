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
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.loader import async_get_integration

from .bayesian_forecaster import HourForecast
from .const import DOMAIN
from .coordinator import (
    ForecastResult,
    LocalForecastConfigEntry,
    LocalForecastCoordinator,
)

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

# Day+2 extrapolation regresses severe conditions toward milder ones.
_CONDITION_DECAY: dict[str, str] = {
    "pouring": "rainy",
    "lightning-rainy": "rainy",
    "exceptional": "cloudy",
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LocalForecastConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Local Weather Forecast weather entity."""
    integration = await async_get_integration(hass, DOMAIN)
    sw_version = str(integration.version) if integration.version else None
    async_add_entities([LocalForecastWeather(entry.runtime_data, sw_version)])


class LocalForecastWeather(CoordinatorEntity[LocalForecastCoordinator], WeatherEntity):
    """Bayesian local weather forecast entity."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_native_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_native_pressure_unit = UnitOfPressure.HPA
    _attr_native_wind_speed_unit = UnitOfSpeed.METERS_PER_SECOND
    _attr_supported_features = WeatherEntityFeature.FORECAST_HOURLY | WeatherEntityFeature.FORECAST_DAILY

    def __init__(self, coordinator: LocalForecastCoordinator, sw_version: str | None) -> None:
        """Bind the entity to its coordinator and device."""
        super().__init__(coordinator)
        entry_id = coordinator.config_entry.entry_id
        self._attr_unique_id = f"{entry_id}_weather"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry_id)},
            name="Local Weather Forecast",
            manufacturer="Local Weather Forecast",
            model="Bayesian Forecaster",
            sw_version=sw_version,
        )

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
        """Return the raw 12-hour probabilistic forecast."""
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
        """Aggregate hourly into today / tomorrow / day-after-tomorrow.

        With a 12-hour horizon, tomorrow uses the tail hours and day+2
        extrapolates further toward climatological mean values.
        """
        if not (data := self._data) or not data.hourly:
            return None

        now = data.generated
        hourly = data.hourly
        hours_left_today = 24 - now.hour
        today_hours = [h for h in hourly if h.hours_ahead <= hours_left_today]
        tomorrow_hours = [h for h in hourly if h.hours_ahead > hours_left_today]

        # If all forecast hours fall in today, use the tail as proxy
        if not tomorrow_hours:
            tomorrow_hours = hourly[-min(6, len(hourly)) :]
        if not today_hours:
            today_hours = hourly[: min(6, len(hourly))]

        days: list[Forecast] = []

        for offset_days, hours in enumerate([today_hours, tomorrow_hours]):
            if not hours:
                continue
            temps = [h.temperature for h in hours]
            condition = _worst_condition(hours)

            # Daily forecasts are always daytime — swap clear-night to sunny
            if condition == "clear-night":
                condition = "sunny"

            # Today uses current time so it's never "in the past" for the
            # frontend card; future days use noon.
            if offset_days == 0:
                dt_entry = now.replace(microsecond=0)
            else:
                day = now.date() + timedelta(days=offset_days)
                dt_entry = datetime(day.year, day.month, day.day, 12, 0, 0, tzinfo=now.tzinfo)

            days.append(
                Forecast(  # type: ignore[typeddict-unknown-key]
                    datetime=dt_entry.isoformat(),
                    condition=condition,
                    native_temperature=round(max(temps), 1),
                    native_templow=round(min(temps), 1),
                    precipitation_probability=max(h.precipitation_probability for h in hours),
                    native_precipitation=round(sum(h.precipitation_amount for h in hours), 1),
                    humidity=round(sum(h.humidity for h in hours) / len(hours)),
                    native_pressure=round(sum(h.pressure for h in hours) / len(hours), 1),
                    native_wind_speed=round(sum(h.wind_speed for h in hours) / len(hours), 1),
                    wind_bearing=round(hours[0].wind_bearing),
                    is_daytime=True,
                )
            )

        if tomorrow_hours:
            days.append(_extrapolate_day2(today_hours, tomorrow_hours, now))

        _LOGGER.debug("Daily forecast: %d day(s)", len(days))
        return days or None


def _extrapolate_day2(
    today_hours: list[HourForecast],
    tomorrow_hours: list[HourForecast],
    now: datetime,
) -> Forecast:
    """Extend beyond the 12 h horizon by regressing toward climatological norms."""
    last = tomorrow_hours[-1]

    # Temperature regresses toward the daily mean (avg of today's high/low)
    # with a ~24 h time constant.
    today_temps = [h.temperature for h in today_hours] or [last.temperature]
    daily_mean = (max(today_temps) + min(today_temps)) / 2.0
    temp_high = round(last.temperature * 0.6 + daily_mean * 0.4, 1)
    temp_low = round(
        min(h.temperature for h in tomorrow_hours) * 0.6 + (daily_mean - 4.0) * 0.4,
        1,
    )
    if temp_high < temp_low:
        temp_high, temp_low = temp_low, temp_high

    # Precip probability decays toward the ~20% base rate.
    last_precip = max(h.precipitation_probability for h in tomorrow_hours)

    condition = _worst_condition(tomorrow_hours)
    if condition == "clear-night":
        condition = "sunny"
    condition = _CONDITION_DECAY.get(condition, condition)

    # Pressure continues the gentle trend; humidity regresses toward the
    # 55% continental mean.
    pressure = round(last.pressure + last.pressure - tomorrow_hours[0].pressure, 1)
    humidity = round((sum(h.humidity for h in tomorrow_hours) / len(tomorrow_hours)) * 0.7 + 55 * 0.3)

    day = now.date() + timedelta(days=2)
    return Forecast(  # type: ignore[typeddict-unknown-key]
        datetime=datetime(day.year, day.month, day.day, 12, 0, 0, tzinfo=now.tzinfo).isoformat(),
        condition=condition,
        native_temperature=temp_high,
        native_templow=temp_low,
        precipitation_probability=round(last_precip * 0.7 + 20 * 0.3),
        native_precipitation=round(sum(h.precipitation_amount for h in tomorrow_hours) * 0.7, 1),
        humidity=humidity,
        native_pressure=max(920.0, min(1070.0, pressure)),
        native_wind_speed=round(last.wind_speed * 0.8 + 0.5, 1),
        wind_bearing=round(last.wind_bearing),
        is_daytime=True,
    )


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
