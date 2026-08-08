"""Config flow for Local Weather Forecast."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithReload,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import selector
import voluptuous as vol

from .const import (
    CONF_ELEVATION,
    CONF_ENABLE_MAP,
    CONF_HUMIDITY_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_PRESSURE_TYPE,
    CONF_RAIN_RATE_SENSOR,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    DEFAULT_ELEVATION,
    DEFAULT_ENABLE_MAP,
    DEFAULT_PRESSURE_TYPE,
    DOMAIN,
    PRESSURE_ABSOLUTE,
    PRESSURE_RELATIVE,
)

SENSOR_SELECTOR = selector.EntitySelector(
    selector.EntitySelectorConfig(domain="sensor")
)

PRESSURE_TYPE_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(value=PRESSURE_ABSOLUTE, label="Absolute (QFE)"),
            selector.SelectOptionDict(value=PRESSURE_RELATIVE, label="Sea-level (QNH)"),
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)

OPTIONAL_SENSOR_KEYS = (
    CONF_HUMIDITY_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_RAIN_RATE_SENSOR,
)


def _validate(user_input: dict[str, Any], hass: HomeAssistant) -> dict[str, str]:
    """Return per-field errors for a submitted form."""
    errors: dict[str, str] = {}
    for key in (CONF_PRESSURE_SENSOR, CONF_TEMPERATURE_SENSOR):
        sid = user_input.get(key)
        if sid and not hass.states.get(sid):
            errors[key] = "sensor_not_found"
    if not -500 <= user_input.get(CONF_ELEVATION, 0) <= 9000:
        errors[CONF_ELEVATION] = "invalid_elevation"
    return errors


def _cleaned(user_input: dict[str, Any]) -> dict[str, Any]:
    """Drop optional fields the user left empty."""
    return {k: v for k, v in user_input.items() if v not in (None, "")}


def _schema(defaults: dict[str, Any]) -> vol.Schema:
    """Build the shared config/options form, pre-filled from ``defaults``."""
    fields: dict[Any, Any] = {
        vol.Required(
            CONF_PRESSURE_SENSOR,
            default=defaults.get(CONF_PRESSURE_SENSOR, vol.UNDEFINED),
        ): SENSOR_SELECTOR,
        vol.Required(
            CONF_TEMPERATURE_SENSOR,
            default=defaults.get(CONF_TEMPERATURE_SENSOR, vol.UNDEFINED),
        ): SENSOR_SELECTOR,
    }
    for key in OPTIONAL_SENSOR_KEYS:
        fields[
            vol.Optional(key, description={"suggested_value": defaults.get(key)})
        ] = SENSOR_SELECTOR
    fields[
        vol.Optional(
            CONF_ELEVATION, default=defaults.get(CONF_ELEVATION, DEFAULT_ELEVATION)
        )
    ] = vol.Coerce(int)
    fields[
        vol.Optional(
            CONF_PRESSURE_TYPE,
            default=defaults.get(CONF_PRESSURE_TYPE, DEFAULT_PRESSURE_TYPE),
        )
    ] = PRESSURE_TYPE_SELECTOR
    fields[
        vol.Optional(
            CONF_ENABLE_MAP, default=defaults.get(CONF_ENABLE_MAP, DEFAULT_ENABLE_MAP)
        )
    ] = selector.BooleanSelector()
    return vol.Schema(fields)


class LocalForecastConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Local Weather Forecast."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            errors = _validate(user_input, self.hass)
            if not errors:
                await self.async_set_unique_id(user_input[CONF_PRESSURE_SENSOR])
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="Local Weather Forecast", data=_cleaned(user_input)
                )

        return self.async_show_form(
            step_id="user",
            data_schema=_schema(user_input or {}),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlowWithReload:
        """Return the options flow, which reloads the entry when it finishes."""
        return LocalForecastOptionsFlow()


class LocalForecastOptionsFlow(OptionsFlowWithReload):
    """Handle options (re-configure sensors)."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Re-submit the whole configuration form."""
        errors: dict[str, str] = {}

        if user_input is not None:
            errors = _validate(user_input, self.hass)
            if not errors:
                return self.async_create_entry(data=_cleaned(user_input))

        # Pre-fill with the stored configuration, overlaid with whatever the
        # user just submitted so a validation error does not wipe the form.
        entry = self.config_entry
        defaults = {**(entry.options or entry.data), **(user_input or {})}
        return self.async_show_form(
            step_id="init", data_schema=_schema(defaults), errors=errors
        )
