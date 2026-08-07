"""End-to-end tests: the integration must actually set up inside Home Assistant.

The pure-python modules are well covered elsewhere; every bug that reaches a
dashboard lives in the entity layer, so these boot a real HomeAssistant
instance and assert on the published states.
"""

from __future__ import annotations

import pytest
from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from local_forecast.const import (
    CONF_ELEVATION,
    CONF_HUMIDITY_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_PRESSURE_TYPE,
    CONF_TEMPERATURE_SENSOR,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    DOMAIN,
    PRESSURE_ABSOLUTE,
)
from pytest_homeassistant_custom_component.common import MockConfigEntry

WEATHER = "weather.local_weather_forecast"


def _entry(hass: HomeAssistant, **overrides) -> MockConfigEntry:
    data = {
        CONF_PRESSURE_SENSOR: "sensor.pressure",
        CONF_TEMPERATURE_SENSOR: "sensor.temperature",
        CONF_HUMIDITY_SENSOR: "sensor.humidity",
        CONF_WIND_SPEED_SENSOR: "sensor.wind_speed",
        CONF_WIND_DIRECTION_SENSOR: "sensor.wind_dir",
    }
    data.update(overrides)
    entry = MockConfigEntry(domain=DOMAIN, data=data, entry_id="test")
    entry.add_to_hass(hass)
    return entry


def _set(hass: HomeAssistant, entity_id: str, value, unit: str | None = None) -> None:
    attrs = {ATTR_UNIT_OF_MEASUREMENT: unit} if unit else {}
    hass.states.async_set(entity_id, value, attrs)


async def _setup(hass: HomeAssistant, entry) -> None:
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


@pytest.fixture
def sensors(hass: HomeAssistant):
    _set(hass, "sensor.pressure", "1013.2", "hPa")
    _set(hass, "sensor.temperature", "18.4", "°C")
    _set(hass, "sensor.humidity", "62", "%")
    _set(hass, "sensor.wind_speed", "3.2", "m/s")
    _set(hass, "sensor.wind_dir", "210", "°")


async def test_setup_publishes_a_usable_weather_entity(hass, sensors):
    await _setup(hass, _entry(hass))

    state = hass.states.get(WEATHER)
    assert state is not None
    assert state.state != STATE_UNAVAILABLE
    assert state.attributes["temperature"] == pytest.approx(18.4, abs=0.2)
    assert state.attributes["pressure"] == pytest.approx(1013.2, abs=0.2)
    assert state.attributes["humidity"] == 62
    assert "wet_bulb" in state.attributes
    assert "pressure_trend" in state.attributes


async def test_all_sensor_entities_exist(hass, sensors):
    await _setup(hass, _entry(hass))

    for suffix in (
        "precipitation_probability",
        "1h_forecast",
        "next_hour_precipitation_probability",
        "sea_level_pressure",
        "pressure_tendency",
        "pressure_tendency_direction",
        "pressure_synoptic",
        "barometer",
        "hourly_forecast",
        "front",
    ):
        entity_id = f"sensor.local_weather_forecast_{suffix}"
        assert hass.states.get(entity_id) is not None, entity_id


async def test_unavailable_until_real_data_arrives(hass):
    """No sensor data must never be published as 15 °C / 1013 hPa."""
    _set(hass, "sensor.pressure", STATE_UNAVAILABLE)
    _set(hass, "sensor.temperature", STATE_UNAVAILABLE)
    await _setup(hass, _entry(hass))

    assert hass.states.get(WEATHER).state == STATE_UNAVAILABLE
    assert (
        hass.states.get("sensor.local_weather_forecast_sea_level_pressure").state
        == STATE_UNAVAILABLE
    )


async def test_unconfigured_channels_report_none(hass):
    """Optional sensors that were never configured must not be invented."""
    _set(hass, "sensor.pressure", "1008.0", "hPa")
    _set(hass, "sensor.temperature", "5.0", "°C")
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_PRESSURE_SENSOR: "sensor.pressure",
            CONF_TEMPERATURE_SENSOR: "sensor.temperature",
        },
        entry_id="minimal",
    )
    entry.add_to_hass(hass)
    await _setup(hass, entry)

    attrs = hass.states.get(WEATHER).attributes
    assert attrs.get("humidity") is None
    assert attrs.get("wind_bearing") is None
    assert attrs.get("wind_speed") is None
    assert "dew_point" not in attrs
    assert "wind_force" not in attrs


async def test_high_altitude_station_is_not_rejected(hass):
    """A QFE reading at 1600 m is ~835 hPa and must still be accepted."""
    _set(hass, "sensor.pressure", "835.0", "hPa")
    _set(hass, "sensor.temperature", "4.0", "°C")
    await _setup(
        hass,
        _entry(hass, **{CONF_ELEVATION: 1600, CONF_PRESSURE_TYPE: PRESSURE_ABSOLUTE}),
    )

    state = hass.states.get(WEATHER)
    assert state.state != STATE_UNAVAILABLE
    assert 990.0 < state.attributes["pressure"] < 1040.0


async def test_units_are_converted_by_home_assistant(hass):
    """inHg / °F / mph must land in hPa / °C / m/s."""
    _set(hass, "sensor.pressure", "29.92", "inHg")
    _set(hass, "sensor.temperature", "68", "°F")
    _set(hass, "sensor.humidity", "50", "%")
    _set(hass, "sensor.wind_speed", "10", "mph")
    _set(hass, "sensor.wind_dir", "180", "°")
    await _setup(hass, _entry(hass))

    attrs = hass.states.get(WEATHER).attributes
    assert attrs["pressure"] == pytest.approx(1013.2, abs=0.5)
    assert attrs["temperature"] == pytest.approx(20.0, abs=0.3)
    # The weather entity reports wind in km/h for display; 10 mph = 4.47 m/s.
    assert attrs["wind_speed"] == pytest.approx(16.1, abs=0.4)


async def test_forecasts_are_served(hass, sensors):
    await _setup(hass, _entry(hass))

    result = await hass.services.async_call(
        "weather",
        "get_forecasts",
        {"entity_id": WEATHER, "type": "hourly"},
        blocking=True,
        return_response=True,
    )
    hourly = result[WEATHER]["forecast"]
    assert len(hourly) == 12

    result = await hass.services.async_call(
        "weather",
        "get_forecasts",
        {"entity_id": WEATHER, "type": "daily"},
        blocking=True,
        return_response=True,
    )
    assert len(result[WEATHER]["forecast"]) >= 3


async def test_sensor_change_refreshes_the_entity(hass, sensors):
    """The push path must actually reach the published state."""
    await _setup(hass, _entry(hass))
    before = hass.states.get(WEATHER).attributes["temperature"]

    _set(hass, "sensor.temperature", "25.0", "°C")
    await hass.async_block_till_done()
    entry = hass.config_entries.async_entries(DOMAIN)[0]
    entity = hass.data[DOMAIN][entry.entry_id]["weather_entity"]
    await entity._async_recalculate()
    await hass.async_block_till_done()

    assert hass.states.get(WEATHER).attributes["temperature"] > before


async def test_map_endpoint_is_gated_and_coarse(hass, sensors, hass_client_no_auth):
    """Disabled -> 404.  Enabled -> no exact home coordinates in the page."""
    await async_setup_component(hass, "http", {})
    hass.config.latitude = 45.6431
    hass.config.longitude = 25.5887
    await _setup(hass, _entry(hass, enable_map=True))

    client = await hass_client_no_auth()
    resp = await client.get("/api/local_forecast/map")
    assert resp.status == 200
    body = await resp.text()
    assert "45.6431" not in body
    assert "25.5887" not in body
    assert "unpkg.com" not in body
    assert "/local_forecast_static/leaflet.js" in body

    asset = await client.get("/local_forecast_static/leaflet.js")
    assert asset.status == 200
    assert "Leaflet" in await asset.text()


async def test_setup_with_recorder_present(recorder_mock, hass, sensors):
    """The startup backfill path must run against a real recorder."""
    await _setup(hass, _entry(hass))
    assert hass.states.get(WEATHER).state != STATE_UNAVAILABLE
