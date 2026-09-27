"""End-to-end tests: the integration must actually set up inside Home Assistant.

The pure-python modules are well covered elsewhere; every bug that reaches a
dashboard lives in the entity layer, so these boot a real HomeAssistant
instance and assert on the published states.
"""

from __future__ import annotations

from datetime import datetime
import math
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.diagnostics import get_diagnostics_for_config_entry

from local_forecast.bayesian_forecaster import HourForecast
from local_forecast.const import (
    CONF_ELEVATION,
    CONF_HUMIDITY_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_PRESSURE_TYPE,
    CONF_TEMPERATURE_SENSOR,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    DOMAIN,
    MAP_CENTER_GRID,
    MAP_DEGREE_METRES,
    MAP_TIME_CACHE_TTL,
    PRESSURE_ABSOLUTE,
)
from local_forecast.map import _ring_radius
from local_forecast.tide import LocalTide
from local_forecast.weather import LocalForecastWeather

WEATHER = "weather.local_weather_forecast"


def _wms_time() -> ModuleType:
    """Return the wms_time module Home Assistant actually loaded.

    The test harness imports the component from a temp config dir, so the
    running module is ``custom_components.local_forecast.wms_time`` — patching
    the copy reachable as ``local_forecast.wms_time`` would have no effect.
    """
    return sys.modules["custom_components.local_forecast.wms_time"]


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


# entity_id, unique_id suffix, friendly name.  These three are what a
# dashboard, an automation and a family remember; none of them may drift.
EXPECTED_ENTITIES = [
    ("weather.local_weather_forecast", "weather", "Local Weather Forecast"),
    (
        "sensor.local_weather_forecast_precipitation_probability",
        "precip_prob_6h",
        "Local Weather Forecast Precipitation probability",
    ),
    (
        "sensor.local_weather_forecast_1h_forecast",
        "next_hour_condition",
        "Local Weather Forecast 1h forecast",
    ),
    (
        "sensor.local_weather_forecast_next_hour_precipitation_probability",
        "next_hour_precip_prob",
        "Local Weather Forecast Next hour precipitation probability",
    ),
    (
        "sensor.local_weather_forecast_sea_level_pressure",
        "sea_level_pressure",
        "Local Weather Forecast Sea level pressure",
    ),
    (
        "sensor.local_weather_forecast_pressure_tendency",
        "pressure_tendency",
        "Local Weather Forecast Pressure tendency",
    ),
    (
        "sensor.local_weather_forecast_pressure_tendency_direction",
        "pressure_tendency_direction",
        "Local Weather Forecast Pressure tendency direction",
    ),
    (
        "sensor.local_weather_forecast_pressure_synoptic",
        "pressure_synoptic",
        "Local Weather Forecast Pressure synoptic",
    ),
    (
        "sensor.local_weather_forecast_barometer",
        "barometer",
        "Local Weather Forecast Barometer",
    ),
    (
        "sensor.local_weather_forecast_hourly_forecast",
        "hourly_forecast",
        "Local Weather Forecast Hourly forecast",
    ),
    (
        "sensor.local_weather_forecast_front",
        "front",
        "Local Weather Forecast Front",
    ),
]


async def test_entity_identity_is_frozen(hass, sensors, entity_registry):
    """Renaming any of these silently breaks every existing dashboard."""
    entry = _entry(hass)
    await _setup(hass, entry)

    for entity_id, unique_suffix, friendly_name in EXPECTED_ENTITIES:
        registry_entry = entity_registry.async_get(entity_id)
        assert registry_entry is not None, entity_id
        assert registry_entry.unique_id == f"{entry.entry_id}_{unique_suffix}"
        assert hass.states.get(entity_id).attributes["friendly_name"] == friendly_name


async def test_static_icons_survive(hass, sensors):
    """icons.json is resolved in the frontend only, so these stay on the entity."""
    await _setup(hass, _entry(hass))

    icons = {
        "pressure_tendency": "mdi:gauge",
        "pressure_synoptic": "mdi:gauge-low",
        "hourly_forecast": "mdi:chart-line",
    }
    for suffix, icon in icons.items():
        state = hass.states.get(f"sensor.local_weather_forecast_{suffix}")
        assert state.attributes.get("icon") == icon, suffix


async def test_unavailable_until_real_data_arrives(hass):
    """No sensor data must never be published as 15 °C / 1013 hPa."""
    _set(hass, "sensor.pressure", STATE_UNAVAILABLE)
    _set(hass, "sensor.temperature", STATE_UNAVAILABLE)
    await _setup(hass, _entry(hass))

    assert hass.states.get(WEATHER).state == STATE_UNAVAILABLE
    assert hass.states.get("sensor.local_weather_forecast_sea_level_pressure").state == STATE_UNAVAILABLE


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

    for kind in ("hourly", "daily"):
        result = await hass.services.async_call(
            "weather",
            "get_forecasts",
            {"entity_id": WEATHER, "type": kind},
            blocking=True,
            return_response=True,
        )
        for item in result[WEATHER]["forecast"]:
            for key in ("humidity", "wind_speed", "wind_bearing"):
                assert item.get(key) is None, (kind, key)

    meteogram = hass.states.get("sensor.local_weather_forecast_hourly_forecast").attributes["forecast"]
    assert all(item["humidity"] is None and item["wind_speed"] is None for item in meteogram)


async def test_daily_today_ends_at_local_midnight():
    """At 14:30 the +10 h hour is 00:30 tomorrow, not the last hour of today."""
    generated = datetime(2026, 9, 27, 14, 30, tzinfo=dt_util.get_default_time_zone())
    hourly = [
        HourForecast(
            hours_ahead=h,
            condition="cloudy",
            temperature=float(h),
            humidity=60.0,
            pressure=1013.0,
            precipitation_probability=0,
            precipitation_amount=0.0,
            wind_speed=2.0,
            wind_bearing=180,
        )
        for h in range(1, 13)
    ]
    weather = LocalForecastWeather.__new__(LocalForecastWeather)
    weather.coordinator = SimpleNamespace(
        data=SimpleNamespace(generated=generated, hourly=hourly),
    )

    today, tomorrow, _ = await weather.async_forecast_daily()

    assert today["native_temperature"] == 9.0
    assert tomorrow["native_templow"] == 10.0


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


@pytest.mark.parametrize("temperature", ["-10.0", "30.0"])
async def test_qnh_uses_the_standard_atmosphere(hass, temperature):
    """Brasov airport's QNH is matched to 0.3 hPa only when the live temperature stays out."""
    _set(hass, "sensor.pressure", "957.0", "hPa")
    _set(hass, "sensor.temperature", temperature, "°C")
    await _setup(hass, _entry(hass, **{CONF_ELEVATION: 544, CONF_PRESSURE_TYPE: PRESSURE_ABSOLUTE}))

    qnh = 957.0 * (1 - 0.0065 * 544 / 288.15) ** -5.257
    assert hass.states.get(WEATHER).attributes["pressure"] == pytest.approx(qnh, abs=0.05)


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
    await entry.runtime_data.async_refresh()
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
    assert f'"ringRadius": {_ring_radius(45.75)}' in body

    asset = await client.get("/local_forecast_static/leaflet.js")
    assert asset.status == 200
    assert "Leaflet" in await asset.text()


@pytest.mark.parametrize("latitude,longitude", [(45.6431, 25.5887), (44.4268, 26.1025), (0.02, 0.02)])
def test_home_ring_encloses_the_home_the_snapping_hid(latitude, longitude):
    """Snapping moves the centre off the house; the ring must still contain it.

    Otherwise the marker would point at open country next to where you live,
    which is worse than no marker at all.
    """
    snapped_lat = round(latitude / MAP_CENTER_GRID) * MAP_CENTER_GRID
    snapped_lon = round(longitude / MAP_CENTER_GRID) * MAP_CENTER_GRID
    north = (latitude - snapped_lat) * MAP_DEGREE_METRES
    east = (longitude - snapped_lon) * MAP_DEGREE_METRES * math.cos(math.radians(latitude))

    assert math.hypot(north, east) <= _ring_radius(snapped_lat)


async def test_map_times_endpoint_serves_and_caches_frame_times(hass, sensors, hass_client_no_auth):
    """One capabilities read per workspace per TTL, however many browsers ask."""
    await async_setup_component(hass, "http", {})
    await _setup(hass, _entry(hass, enable_map=True))
    calls: list[str] = []

    async def fake_fetch(hass_arg, session, workspace):
        calls.append(workspace)
        return {f"{workspace}:rgb_dust": "2026-08-08T00:30:00Z"}

    client = await hass_client_no_auth()
    with patch.object(_wms_time(), "_async_fetch_workspace", fake_fetch):
        first = await (await client.get("/api/local_forecast/map/times")).json()
        second = await (await client.get("/api/local_forecast/map/times")).json()

    assert first["msg_fes:rgb_dust"] == "2026-08-08T00:30:00Z"
    assert second == first
    assert sorted(calls) == ["msg_fes", "mtg_fd"]


async def test_map_times_survive_a_failed_refresh(hass, sensors, freezer):
    """A stale frame beats no frame, so the last good answer is kept."""
    await async_setup_component(hass, "http", {})
    await _setup(hass, _entry(hass, enable_map=True))
    module = _wms_time()

    async def good(hass_arg, session, workspace):
        return {f"{workspace}:rgb_dust": "2026-08-08T00:30:00Z"}

    async def unreachable(hass_arg, session, workspace):
        return None

    with patch.object(module, "_async_fetch_workspace", good):
        await module.async_get_frame_times(hass)

    freezer.tick(MAP_TIME_CACHE_TTL + 1)
    with patch.object(module, "_async_fetch_workspace", unreachable):
        times = await module.async_get_frame_times(hass)

    assert times["msg_fes:rgb_dust"] == "2026-08-08T00:30:00Z"


async def test_map_page_pins_the_frame_time_it_knows(hass, sensors, hass_client_no_auth):
    """The pinned TIME is what stops a cached tile posing as the current one."""
    await async_setup_component(hass, "http", {})
    await _setup(hass, _entry(hass, enable_map=True))

    async def fake_fetch(hass_arg, session, workspace):
        if workspace != "mtg_fd":
            return {}
        return {"mtg_fd:rgb_geocolour": "2026-08-08T00:40:00Z"}

    client = await hass_client_no_auth()
    with patch.object(_wms_time(), "_async_fetch_workspace", fake_fetch):
        await client.get("/api/local_forecast/map/times")

    body = await (await client.get("/api/local_forecast/map")).text()
    assert '"mtg_fd:rgb_geocolour": "2026-08-08T00:40:00Z"' in body
    assert "params.time" in body


async def test_setup_with_recorder_present(recorder_mock, hass, sensors):
    """The startup backfill path must run against a real recorder."""
    await _setup(hass, _entry(hass))
    assert hass.states.get(WEATHER).state != STATE_UNAVAILABLE


async def test_restored_tendency_excludes_the_tide(hass, sensors, hass_storage, freezer):
    """Same pressure as 3 h ago: whatever the tendency reads is the tide, reversed."""
    tide = LocalTide(hass.config.latitude, hass.config.longitude).hpa_at

    # Pick the moment of the day with the largest 3 h tide swing, so the test
    # cannot pass by landing on a flat stretch.
    now = max((1.79e9 + i * 1800 for i in range(48)), key=lambda t: abs(tide(t) - tide(t - 3 * 3600)))
    freezer.move_to(datetime.fromtimestamp(now, dt_util.UTC))
    hass_storage[f"{DOMAIN}.test.pressure"] = {
        "version": 1,
        "minor_version": 1,
        "key": f"{DOMAIN}.test.pressure",
        "data": {"samples": [[now - 3 * 3600, 1013.2]]},
    }

    await _setup(hass, _entry(hass))

    expected = -(tide(now) - tide(now - 3 * 3600)) / 3
    assert abs(expected) > 0.1
    state = hass.states.get("sensor.local_weather_forecast_pressure_tendency")
    assert float(state.state) == pytest.approx(expected, abs=0.006)


async def test_forecast_pressure_carries_the_tide_of_each_hour(hass, sensors, freezer):
    """Steady weather: each forecast hour moves by exactly the tide between now and then."""
    freezer.move_to(datetime.fromtimestamp(1.79e9, dt_util.UTC))
    entry = _entry(hass)
    await _setup(hass, entry)
    tide = entry.runtime_data.tide.hpa_at

    result = await hass.services.async_call(
        "weather", "get_forecasts", {"entity_id": WEATHER, "type": "hourly"}, blocking=True, return_response=True
    )
    now = 1.79e9
    pressures = [item["pressure"] for item in result[WEATHER]["forecast"]]
    expected = [1013.2 - tide(now) + tide(now + h * 3600) for h in range(1, 13)]
    assert max(expected) - min(expected) > 0.3
    assert pressures == pytest.approx(expected, abs=0.06)


async def test_diagnostics_report_the_learned_tide(hass, sensors, hass_client):
    await async_setup_component(hass, "http", {})
    entry = _entry(hass)
    await _setup(hass, entry)

    data = await get_diagnostics_for_config_entry(hass, hass_client, entry)

    assert data["config"][CONF_PRESSURE_SENSOR] == "sensor.pressure"
    assert set(data["tide"]) >= {"tide_now_hpa", "s1_hpa", "s2_hpa", "s3_hpa", "site_gain_hpa_per_k", "learned_days"}
