"""Config flow for Local Weather Forecast."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_ELEVATION,
    CONF_HUMIDITY_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_PRESSURE_TYPE,
    CONF_RAIN_RATE_SENSOR,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    DEFAULT_ELEVATION,
    DEFAULT_PRESSURE_TYPE,
    DOMAIN,
    PRESSURE_ABSOLUTE,
    PRESSURE_RELATIVE,
)

_LOGGER = logging.getLogger(__name__)

SENSOR_SELECTOR = selector.EntitySelector(
    selector.EntitySelectorConfig(domain="sensor")
)
OPTIONAL_SENSOR_SELECTOR = selector.EntitySelector(
    selector.EntitySelectorConfig(domain="sensor")
)


class LocalForecastConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Local Weather Forecast."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate required sensors exist
            for key in (CONF_PRESSURE_SENSOR, CONF_TEMPERATURE_SENSOR):
                sid = user_input.get(key)
                if sid and not self.hass.states.get(sid):
                    errors[key] = "sensor_not_found"

            if not -500 <= user_input.get(CONF_ELEVATION, 0) <= 9000:
                errors[CONF_ELEVATION] = "invalid_elevation"

            if not errors:
                await self.async_set_unique_id(
                    user_input[CONF_PRESSURE_SENSOR]
                )
                self._abort_if_unique_id_configured()

                # Strip empty optional fields
                cleaned = {
                    k: v for k, v in user_input.items() if v not in (None, "")
                }
                return self.async_create_entry(
                    title="Local Weather Forecast", data=cleaned
                )

        schema = vol.Schema(
            {
                vol.Required(CONF_PRESSURE_SENSOR): SENSOR_SELECTOR,
                vol.Required(CONF_TEMPERATURE_SENSOR): SENSOR_SELECTOR,
                vol.Optional(CONF_HUMIDITY_SENSOR): OPTIONAL_SENSOR_SELECTOR,
                vol.Optional(CONF_WIND_SPEED_SENSOR): OPTIONAL_SENSOR_SELECTOR,
                vol.Optional(CONF_WIND_DIRECTION_SENSOR): OPTIONAL_SENSOR_SELECTOR,
                vol.Optional(CONF_SOLAR_RADIATION_SENSOR): OPTIONAL_SENSOR_SELECTOR,
                vol.Optional(CONF_RAIN_RATE_SENSOR): OPTIONAL_SENSOR_SELECTOR,
                vol.Optional(
                    CONF_ELEVATION, default=DEFAULT_ELEVATION
                ): vol.Coerce(int),
                vol.Optional(
                    CONF_PRESSURE_TYPE, default=DEFAULT_PRESSURE_TYPE
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            selector.SelectOptionDict(
                                value=PRESSURE_ABSOLUTE, label="Absolute (QFE)"
                            ),
                            selector.SelectOptionDict(
                                value=PRESSURE_RELATIVE, label="Sea-level (QNH)"
                            ),
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="user", data_schema=schema, errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        return LocalForecastOptionsFlow(config_entry)


class LocalForecastOptionsFlow(config_entries.OptionsFlow):
    """Handle options (re-configure sensors)."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if user_input is not None:
            cleaned = {
                k: v for k, v in user_input.items() if v not in (None, "")
            }
            return self.async_create_entry(title="", data=cleaned)

        data = {**self._entry.data, **self._entry.options}
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_PRESSURE_SENSOR,
                    default=data.get(CONF_PRESSURE_SENSOR, ""),
                ): SENSOR_SELECTOR,
                vol.Required(
                    CONF_TEMPERATURE_SENSOR,
                    default=data.get(CONF_TEMPERATURE_SENSOR, ""),
                ): SENSOR_SELECTOR,
                vol.Optional(
                    CONF_HUMIDITY_SENSOR,
                    description={"suggested_value": data.get(CONF_HUMIDITY_SENSOR)},
                ): OPTIONAL_SENSOR_SELECTOR,
                vol.Optional(
                    CONF_WIND_SPEED_SENSOR,
                    description={"suggested_value": data.get(CONF_WIND_SPEED_SENSOR)},
                ): OPTIONAL_SENSOR_SELECTOR,
                vol.Optional(
                    CONF_WIND_DIRECTION_SENSOR,
                    description={"suggested_value": data.get(CONF_WIND_DIRECTION_SENSOR)},
                ): OPTIONAL_SENSOR_SELECTOR,
                vol.Optional(
                    CONF_SOLAR_RADIATION_SENSOR,
                    description={"suggested_value": data.get(CONF_SOLAR_RADIATION_SENSOR)},
                ): OPTIONAL_SENSOR_SELECTOR,
                vol.Optional(
                    CONF_RAIN_RATE_SENSOR,
                    description={"suggested_value": data.get(CONF_RAIN_RATE_SENSOR)},
                ): OPTIONAL_SENSOR_SELECTOR,
                vol.Optional(
                    CONF_ELEVATION,
                    default=data.get(CONF_ELEVATION, DEFAULT_ELEVATION),
                ): vol.Coerce(int),
                vol.Optional(
                    CONF_PRESSURE_TYPE,
                    default=data.get(CONF_PRESSURE_TYPE, DEFAULT_PRESSURE_TYPE),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            selector.SelectOptionDict(
                                value=PRESSURE_ABSOLUTE, label="Absolute (QFE)"
                            ),
                            selector.SelectOptionDict(
                                value=PRESSURE_RELATIVE, label="Sea-level (QNH)"
                            ),
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)
