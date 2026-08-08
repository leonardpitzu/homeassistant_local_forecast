"""Config and options flow tests.

The flows are the only place a user can wire the integration up, and the
options flow is the only place they can un-wire an optional sensor again.
"""

from __future__ import annotations

from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from local_forecast.const import (
    CONF_ELEVATION,
    CONF_HUMIDITY_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    DOMAIN,
)


@pytest.fixture(autouse=True)
def sources(hass: HomeAssistant):
    hass.states.async_set("sensor.pressure", "1013.2", {ATTR_UNIT_OF_MEASUREMENT: "hPa"})
    hass.states.async_set("sensor.temperature", "18.4", {ATTR_UNIT_OF_MEASUREMENT: "°C"})
    hass.states.async_set("sensor.humidity", "62", {ATTR_UNIT_OF_MEASUREMENT: "%"})


async def test_user_flow_creates_an_entry(hass: HomeAssistant):
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    assert result["type"] is FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_PRESSURE_SENSOR: "sensor.pressure",
            CONF_TEMPERATURE_SENSOR: "sensor.temperature",
            CONF_HUMIDITY_SENSOR: "sensor.humidity",
            CONF_ELEVATION: 600,
        },
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Local Weather Forecast"
    assert result["data"][CONF_HUMIDITY_SENSOR] == "sensor.humidity"
    assert result["data"][CONF_ELEVATION] == 600


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({CONF_PRESSURE_SENSOR: "sensor.nope"}, CONF_PRESSURE_SENSOR),
        ({CONF_ELEVATION: 12000}, CONF_ELEVATION),
    ],
)
async def test_user_flow_rejects_bad_input(hass: HomeAssistant, overrides, field):
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_PRESSURE_SENSOR: "sensor.pressure",
            CONF_TEMPERATURE_SENSOR: "sensor.temperature",
            **overrides,
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert field in result["errors"]


async def test_options_flow_can_clear_an_optional_sensor(hass: HomeAssistant):
    """Removing a sensor in the options form must actually remove it."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_PRESSURE_SENSOR: "sensor.pressure",
            CONF_TEMPERATURE_SENSOR: "sensor.temperature",
            CONF_HUMIDITY_SENSOR: "sensor.humidity",
        },
        entry_id="options",
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert "dew_point" in hass.states.get("weather.local_weather_forecast").attributes

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_PRESSURE_SENSOR: "sensor.pressure",
            CONF_TEMPERATURE_SENSOR: "sensor.temperature",
        },
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert CONF_HUMIDITY_SENSOR not in entry.options
    # OptionsFlowWithReload reloaded the entry, so the change is already live.
    attrs = hass.states.get("weather.local_weather_forecast").attributes
    assert "dew_point" not in attrs
    assert attrs.get("humidity") is None
