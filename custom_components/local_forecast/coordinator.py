"""Forecast engine for Local Weather Forecast.

Owns the whole pipeline and publishes one immutable result per run:

  sensors -> StateEstimator   current conditions, trends, fronts
          -> HourlyHistory    30 days of the station's own hourly means
          -> Forecaster       trained models, evaluated once per closed hour,
                              corrected by what the station has verified

Entities render that result and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
import logging
import time
from typing import Any, Final

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
)
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import (
    PressureConverter,
    SpeedConverter,
    TemperatureConverter,
)

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
    FORECAST_HOURS,
    GRAVITY_EXPONENT,
    HA_CONDITIONS,
    HISTORY_SECONDS,
    ISA_SEA_LEVEL_TEMP_K,
    LAPSE_RATE,
    PRESSURE_RELATIVE,
    S_CLOUDY,
    S_FOG,
    STORAGE_VERSION,
    UPDATE_DEBOUNCE_SECONDS,
    UPDATE_INTERVAL_MINUTES,
)
from .features import HISTORY_HOURS
from .forecast import DayForecast, Forecaster, HourForecast, Sensors
from .history import HourlyHistory, issue_hour
from .model import ModelSet
from .pressure_history import PressureHistory
from .state_estimator import SensorReading, StateEstimator
from .tide import LocalTide

_LOGGER = logging.getLogger(__name__)

# Unit handling is delegated to Home Assistant's own converters so every unit
# HA knows about is accepted, not just the handful worth string-matching.
_CONVERTERS: Final = {
    CONF_PRESSURE_SENSOR: (PressureConverter, UnitOfPressure.HPA),
    CONF_TEMPERATURE_SENSOR: (TemperatureConverter, UnitOfTemperature.CELSIUS),
    CONF_WIND_SPEED_SENSOR: (SpeedConverter, UnitOfSpeed.METERS_PER_SECOND),
}

_SENSOR_KEYS: Final = (
    CONF_PRESSURE_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    CONF_HUMIDITY_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_RAIN_RATE_SENSOR,
)

_BEAUFORT_NAMES: Final = (
    "Calm",
    "Light air",
    "Light breeze",
    "Gentle breeze",
    "Moderate breeze",
    "Fresh breeze",
    "Strong breeze",
    "Near gale",
    "Gale",
    "Strong gale",
    "Storm",
    "Violent storm",
    "Hurricane force",
)
# WMO Beaufort upper bounds (m/s) for scales 0-11; at/above the last -> 12.
_BEAUFORT_THRESHOLDS: Final = (
    0.3,
    1.6,
    3.4,
    5.5,
    8.0,
    10.8,
    13.9,
    17.2,
    20.8,
    24.5,
    28.5,
    32.7,
)

# Sea-level pressure window every published value must fall inside.
_SEA_LEVEL_MIN: Final = 870.0
_SEA_LEVEL_MAX: Final = 1090.0


def _beaufort(wind_ms: float) -> int:
    """Convert wind speed in m/s to Beaufort scale (0-12)."""
    for i, threshold in enumerate(_BEAUFORT_THRESHOLDS):
        if wind_ms < threshold:
            return i
    return 12


@dataclass(slots=True, frozen=True)
class ForecastResult:
    """One complete pipeline run — everything any entity needs to render."""

    generated: datetime
    condition: str
    hourly: list[HourForecast]
    attributes: dict[str, Any]
    temperature: float
    apparent_temperature: float
    pressure: float
    humidity: float | None
    wind_speed: float | None
    wind_bearing: float | None
    dew_point: float | None
    hourly_dicts: list[dict[str, Any]] = field(default_factory=list)
    daily: list[DayForecast] = field(default_factory=list)
    # (high, low) observed so far today, for the daily card's "today".
    today_observed: tuple[float, float] | None = None


class LocalForecastCoordinator(DataUpdateCoordinator[ForecastResult | None]):
    """Runs the forecast pipeline and pushes the result to every entity."""

    config_entry: LocalForecastConfigEntry

    def __init__(self, hass: HomeAssistant, entry: LocalForecastConfigEntry, sw_version: str | None = None) -> None:
        """Set up the pipeline, its persistence and its refresh policy."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=timedelta(minutes=UPDATE_INTERVAL_MINUTES),
            setup_method=self._async_backfill_history,
            # Coalescing window, not a reset-on-event debounce: a station that
            # updates faster than the cooldown must still get a refresh.
            request_refresh_debouncer=Debouncer(hass, _LOGGER, cooldown=UPDATE_DEBOUNCE_SECONDS, immediate=False),
        )
        # Options are a full re-submission of the config form, so when they
        # exist they are authoritative — merging them over data would resurrect
        # optional sensors the user has just cleared.
        self._config: dict[str, Any] = dict(entry.options or entry.data)

        # Platforms set up concurrently; whichever registers the device first names
        # it, and entity ids are minted from that name, so all must carry all of it.
        self.device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Local Weather Forecast",
            manufacturer="Local Weather Forecast",
            model="Trained forecaster",
            sw_version=sw_version,
        )

        # One tide for every consumer: trends subtract it, the forecast adds it back.
        self.tide = LocalTide(hass.config.latitude, hass.config.longitude)
        self._estimator = StateEstimator(tide_hpa=self.tide.hpa_at)
        self._has_data = False

        self.pressure_history = PressureHistory(tide_hpa=self.tide.hpa_at)
        self._store: Store[dict[str, Any]] = Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}.pressure")

        # The trained forecaster reads the station's own hourly history; both it
        # and the self-correction it has learned survive restarts.
        self.history = HourlyHistory()
        self.forecaster: Forecaster | None = None
        self._forecast_store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}.forecast"
        )
        self._local_date: int | None = None

        # Sample timestamps must be monotonic — the history buffer is bisected
        # on them — but also comparable to recorder timestamps, so anchor a
        # monotonic clock onto the wall clock at startup.
        self._clock_wall = time.time()
        self._clock_mono = time.monotonic()

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    async def async_restore(self) -> None:
        """Load the models and restore everything learned before a restart."""
        if (saved := await self._store.async_load()) and isinstance(saved, dict):
            self.pressure_history.load(saved.get("samples", []))
            self.tide.load(saved.get("tide"))
        models = await self.hass.async_add_executor_job(ModelSet.load)
        cfg = self.hass.config
        self.forecaster = Forecaster(models, cfg.latitude, cfg.longitude, float(self._config.get(CONF_ELEVATION, 0)))
        if (saved := await self._forecast_store.async_load()) and isinstance(saved, dict):
            self.history.load(saved.get("history"))
            self.forecaster.load(saved.get("recal"))

    @callback
    def async_track_sources(self, entry: ConfigEntry) -> None:
        """Refresh whenever one of the configured source sensors changes."""
        if ids := [self._config[k] for k in _SENSOR_KEYS if self._config.get(k)]:
            entry.async_on_unload(async_track_state_change_event(self.hass, ids, self._async_source_changed))

    @callback
    def _async_source_changed(self, _event: Event) -> None:
        self.config_entry.async_create_task(self.hass, self.async_request_refresh(), eager_start=False)

    # ------------------------------------------------------------------
    #  Config helpers
    # ------------------------------------------------------------------

    def has_source(self, key: str) -> bool:
        """Whether an optional sensor is configured for this entry."""
        return bool(self._config.get(key))

    def _now_ts(self) -> float:
        """Epoch-like timestamp immune to wall-clock steps (NTP, no RTC)."""
        return self._clock_wall + (time.monotonic() - self._clock_mono)

    # ------------------------------------------------------------------
    #  Pipeline
    # ------------------------------------------------------------------

    async def _async_update_data(self) -> ForecastResult | None:
        """Ingest the sensors and rebuild the forecast."""
        self._ingest_sensors()
        if not self._has_data:
            return None
        await self._async_issue_if_due()
        return self._run_forecast()

    # ------------------------------------------------------------------
    #  Recorder backfill
    # ------------------------------------------------------------------

    async def _async_backfill_history(self) -> None:
        """Seed the estimator and the hourly history from the recorder.

        Pressure tendency is the single most important input; without this
        it reads zero for the first hour after every restart.  The hourly
        means give a fresh install weeks of history instead of none.
        """
        try:
            await self._async_backfill()
        except Exception:  # never let a cold-start optimisation break setup
            _LOGGER.debug("Recorder backfill failed", exc_info=True)
        try:
            await self._async_seed_hourly()
        except Exception:
            _LOGGER.debug("Recorder statistics seeding failed", exc_info=True)

    async def _async_seed_hourly(self) -> None:
        """Fill hours the history has never seen from long-term statistics."""
        if "recorder" not in self.hass.config.components:
            return
        from homeassistant.components.recorder import get_instance, statistics

        keys = {
            "t": CONF_TEMPERATURE_SENSOR,
            "rh": CONF_HUMIDITY_SENSOR,
            "ws": CONF_WIND_SPEED_SENSOR,
            "p": CONF_PRESSURE_SENSOR,
            "rain": CONF_RAIN_RATE_SENSOR,
        }
        ids = {name: self._config[key] for name, key in keys.items() if self._config.get(key)}
        if not ids:
            return
        end = dt_util.utcnow()
        start = end - timedelta(hours=HISTORY_HOURS)
        stats = await get_instance(self.hass).async_add_executor_job(
            statistics.statistics_during_period,
            self.hass,
            start,
            end,
            set(ids.values()),
            "hour",
            None,
            {"mean"},
        )
        rows: dict[int, dict[str, float | None]] = {}
        for name, sid in ids.items():
            live = self.hass.states.get(sid)
            unit = live.attributes.get(ATTR_UNIT_OF_MEASUREMENT) if live else None
            for row in stats.get(sid, []):
                value = self._parse(row.get("mean"), unit, keys[name])
                if value is None:
                    continue
                if name == "p":
                    value = self._to_sea_level(value)
                    if not _SEA_LEVEL_MIN <= value <= _SEA_LEVEL_MAX:
                        continue
                start_ts = row["start"]
                if isinstance(start_ts, datetime):
                    start_ts = start_ts.timestamp()
                # A statistic covers [start, start + 1 h); it seeds the hour it ends on.
                rows.setdefault(int(start_ts) + 3600, {})[name] = value
        added = self.history.seed(sorted(rows.items()))
        _LOGGER.debug("Seeded %d hours of history from statistics", added)

    async def _async_backfill(self) -> None:
        if "recorder" not in self.hass.config.components:
            return
        pressure_id = self._config.get(CONF_PRESSURE_SENSOR)
        temp_id = self._config.get(CONF_TEMPERATURE_SENSOR)
        if not pressure_id or not temp_id:
            return

        from homeassistant.components.recorder import get_instance, history

        ids = [pressure_id, temp_id]
        humidity_id = self._config.get(CONF_HUMIDITY_SENSOR)
        if humidity_id:
            ids.append(humidity_id)

        end = dt_util.utcnow()
        start = end - timedelta(seconds=HISTORY_SECONDS)
        try:
            past = await get_instance(self.hass).async_add_executor_job(
                partial(
                    history.get_significant_states,
                    self.hass,
                    start,
                    end,
                    ids,
                    include_start_time_state=True,
                    significant_changes_only=False,
                    no_attributes=True,
                )
            )
        except Exception:  # recorder is best-effort; never block setup
            _LOGGER.debug("Recorder backfill unavailable", exc_info=True)
            return

        series = {
            key: self._series(past.get(sid, []), key)
            for key, sid in (
                (CONF_PRESSURE_SENSOR, pressure_id),
                (CONF_TEMPERATURE_SENSOR, temp_id),
                (CONF_HUMIDITY_SENSOR, humidity_id),
            )
            if sid
        }
        pressures = series.get(CONF_PRESSURE_SENSOR, [])
        temps = series.get(CONF_TEMPERATURE_SENSOR, [])
        humidities = series.get(CONF_HUMIDITY_SENSOR, [])
        if len(pressures) < 3 or not temps:
            return

        # All three series are sorted, so walk them together: rescanning each
        # one per pressure sample is quadratic, and a chatty station easily
        # puts thousands of samples in a four-hour window.
        count = 0
        temp_cur = hum_cur = 0
        for ts, value in pressures:
            temp_cur, temp = self._align(temps, temp_cur, ts)
            if temp is None:
                continue
            hum_cur, humidity = self._align(humidities, hum_cur, ts)
            self._estimator.update(
                SensorReading(
                    timestamp=ts,
                    pressure_hpa=self._to_sea_level(value),
                    temperature_c=temp,
                    humidity_pct=humidity,
                )
            )
            count += 1
        if count:
            self._has_data = True
            _LOGGER.debug("Backfilled %d historical readings", count)

    def _series(self, states, config_key: str) -> list[tuple[float, float]]:
        """Convert recorder states to (timestamp, value) in canonical units."""
        live = self.hass.states.get(self._config.get(config_key) or "")
        unit = live.attributes.get(ATTR_UNIT_OF_MEASUREMENT) if live else None
        out: list[tuple[float, float]] = []
        for state in states:
            value = self._parse(getattr(state, "state", None), unit, config_key)
            if value is not None:
                out.append((state.last_updated.timestamp(), value))
        out.sort(key=lambda item: item[0])
        return out

    @staticmethod
    def _align(series: list[tuple[float, float]], cursor: int, ts: float) -> tuple[int, float | None]:
        """Advance ``cursor`` to the last sample at or before ``ts``.

        Callers must query with non-decreasing ``ts``; the cursor only ever
        moves forward, so a whole merge costs one pass.
        """
        while cursor < len(series) and series[cursor][0] <= ts:
            cursor += 1
        return cursor, series[cursor - 1][1] if cursor else None

    # ------------------------------------------------------------------
    #  Sensor ingestion
    # ------------------------------------------------------------------

    def _ingest_sensors(self) -> bool:
        """Read all configured sensors and feed a SensorReading."""
        pressure = self._read_float(CONF_PRESSURE_SENSOR)
        temperature = self._read_float(CONF_TEMPERATURE_SENSOR)
        if pressure is None or temperature is None:
            return False

        pressure = self._to_sea_level(pressure)
        # Validate the sea-level value, not the station value: above roughly
        # 1250 m a perfectly healthy QFE reading is below 870 hPa.
        if not _SEA_LEVEL_MIN <= pressure <= _SEA_LEVEL_MAX:
            _LOGGER.debug("Rejecting sea-level pressure %.1f hPa", pressure)
            return False

        reading = SensorReading(
            timestamp=self._now_ts(),
            pressure_hpa=pressure,
            temperature_c=temperature,
            humidity_pct=self._read_float(CONF_HUMIDITY_SENSOR),
            wind_speed_ms=self._read_float(CONF_WIND_SPEED_SENSOR),
            wind_direction_deg=self._read_float(CONF_WIND_DIRECTION_SENSOR),
            solar_radiation_wm2=self._read_float(CONF_SOLAR_RADIATION_SENSOR),
            rain_rate_mmh=self._read_float(CONF_RAIN_RATE_SENSOR),
        )
        self._estimator.update(reading)
        # Wall clock: hours are calendar hours, shared with the recorder.
        self._add_to_history(time.time(), reading)
        self._has_data = True
        return True

    def _add_to_history(self, ts: float, r: SensorReading) -> None:
        self.history.add(
            ts,
            temperature=r.temperature_c,
            humidity=r.humidity_pct,
            wind_speed=r.wind_speed_ms,
            wind_direction=r.wind_direction_deg,
            pressure=r.pressure_hpa,
            rain_rate=r.rain_rate_mmh,
        )

    def _to_sea_level(self, pressure: float) -> float:
        """Convert station pressure (QFE) to QNH through the standard atmosphere."""
        if self._config.get(CONF_PRESSURE_TYPE, DEFAULT_PRESSURE_TYPE) == PRESSURE_RELATIVE:
            return pressure
        elevation = self._config.get(CONF_ELEVATION, DEFAULT_ELEVATION)
        return pressure * (1 - LAPSE_RATE * elevation / ISA_SEA_LEVEL_TEMP_K) ** -GRAVITY_EXPONENT

    def _read_float(self, config_key: str) -> float | None:
        sid = self._config.get(config_key)
        if not sid:
            return None
        state = self.hass.states.get(sid)
        if state is None or state.state in ("unknown", "unavailable", ""):
            return None
        return self._parse(state.state, state.attributes.get(ATTR_UNIT_OF_MEASUREMENT), config_key)

    @staticmethod
    def _parse(raw: Any, unit: str | None, config_key: str) -> float | None:
        """Parse a state string into a validated value in canonical units."""
        try:
            val = float(raw)
        except (ValueError, TypeError):
            return None

        converter = _CONVERTERS.get(config_key)
        if converter is not None and unit:
            cls, target = converter
            if unit != target and unit in cls.VALID_UNITS:
                try:
                    val = cls.convert(val, unit, target)
                except (ValueError, TypeError):
                    return None

        # Reject physically impossible values after conversion.  The pressure
        # bound is deliberately wide here: this is station pressure, which is
        # only checked against the sea-level window after reduction.
        if config_key == CONF_PRESSURE_SENSOR and not (300.0 <= val <= 1100.0):
            return None
        if config_key == CONF_TEMPERATURE_SENSOR and not (-60.0 <= val <= 60.0):
            return None
        if config_key == CONF_HUMIDITY_SENSOR:
            val = max(0.0, min(100.0, val))
        if config_key == CONF_WIND_SPEED_SENSOR:
            val = max(0.0, min(60.0, val))

        return val

    # ------------------------------------------------------------------
    #  Forecast
    # ------------------------------------------------------------------

    def _sensors(self) -> Sensors:
        return Sensors(
            humidity=self.has_source(CONF_HUMIDITY_SENSOR),
            wind=self.has_source(CONF_WIND_SPEED_SENSOR),
            wind_direction=self.has_source(CONF_WIND_DIRECTION_SENSOR),
            rain=self.has_source(CONF_RAIN_RATE_SENSOR),
        )

    async def _async_issue_if_due(self) -> None:
        """Evaluate the models once per closed hour (in an executor)."""
        forecaster = self.forecaster
        if forecaster is None:
            return
        now = time.time()
        issue = issue_hour(now)
        if forecaster.issue is not None and forecaster.issue.ts == issue:
            return
        if not self.history.has(issue):
            # The newest closed hour has no readings (a fresh install, or the
            # first minutes after a long outage): the current state stands in
            # for it, so there is always a forecast to show.
            s = self._estimator.state
            self._add_to_history(
                issue,
                SensorReading(
                    timestamp=issue,
                    pressure_hpa=s.pressure,
                    temperature_c=s.temperature,
                    humidity_pct=s.humidity if self.has_source(CONF_HUMIDITY_SENSOR) else None,
                    wind_speed_ms=s.wind_speed if self.has_source(CONF_WIND_SPEED_SENSOR) else None,
                    wind_direction_deg=s.wind_direction if self.has_source(CONF_WIND_DIRECTION_SENSOR) else None,
                    rain_rate_mmh=s.rain_rate if self.has_source(CONF_RAIN_RATE_SENSOR) else None,
                ),
            )
        local = dt_util.as_local(datetime.fromtimestamp(issue, UTC))
        local_date = local.date().toordinal()
        midnight = dt_util.start_of_local_day(local.date() + timedelta(days=1))
        hours_to_midnight = (midnight - local).total_seconds() / 3600.0
        finished = self._finished_days(local_date)
        sensors = self._sensors()

        def work() -> None:
            forecaster.verify(self.history, issue, sensors)
            for date, high, low, wet in finished:
                forecaster.verify_day(date, high, low, wet if sensors.rain else None)
            forecaster.run_issue(self.history, issue, self.tide.hpa_at, hours_to_midnight, local_date, sensors)

        await self.hass.async_add_executor_job(work)
        self._local_date = local_date
        self._forecast_store.async_delay_save(lambda: {"history": self.history.dump(), "recal": forecaster.dump()}, 300)

    def _finished_days(self, local_date: int) -> list[tuple[int, float, float, float | None]]:
        """Calendar days completed since the last issue: (date, high, low, wet)."""
        if self._local_date is None or local_date <= self._local_date:
            return []
        out = []
        for date in range(self._local_date, local_date):
            day = datetime.fromordinal(date).date()
            start = int(dt_util.start_of_local_day(day).timestamp())
            end = int(dt_util.start_of_local_day(day + timedelta(days=1)).timestamp())
            # An hour belongs to the day it ends in, as in training.
            temps = [t for _h, t in self.history.observed(start + 1, end + 1)]
            if len(temps) < 20:
                continue
            series = self.history.series(issue_hour(end + 1800))
            hours = (end - start) // 3600
            wet_vals = [w for w in series["wet"][-hours:] if w == w]
            wet = (1.0 if sum(wet_vals) >= 2 else 0.0) if len(wet_vals) >= 20 else None
            out.append((date, max(temps), min(temps), wet))
        return out

    def _run_forecast(self) -> ForecastResult:
        """Current conditions from the estimator, the forecast from the models."""
        s = self._estimator.state
        s.has_humidity = self.has_source(CONF_HUMIDITY_SENSOR)
        s.has_wind = self.has_source(CONF_WIND_SPEED_SENSOR)

        sun_el = self._sun_elevation()
        sun = self.hass.states.get("sun.sun")
        # sun.sun already answers this correctly at every latitude and across
        # midnight; the elevation is the fallback when it is missing.
        s.is_night = sun.state == "below_horizon" if sun is not None else sun_el < -0.833

        cloud = self._estimator.cloud_fraction(sun_el)
        current_idx = self._estimator.classify(sun_el, cloud_fraction=cloud)
        if current_idx == S_FOG and self.forecaster is not None and self.forecaster.fog_unlikely():
            current_idx = S_CLOUDY

        _LOGGER.debug(
            "Pipeline: P=%.1f dp/dt=%.2f T=%.1f RH=%.0f wind=%.1f state=%s night=%s",
            s.pressure,
            s.dp_dt,
            s.temperature,
            s.humidity,
            s.wind_speed,
            HA_CONDITIONS[current_idx],
            s.is_night,
        )

        now_local = dt_util.now()
        sensors = self._sensors()
        bearing = round((s.wind_direction + 360) % 360) if sensors.wind_direction else None
        hourly: list[HourForecast] = []
        daily: list[DayForecast] = []
        if self.forecaster is not None:
            hourly = self.forecaster.hours(now_local.timestamp(), self.tide.hpa_at, sensors, bearing, FORECAST_HOURS)
            daily = self.forecaster.days(
                sensors, s.humidity if sensors.humidity else None, now_local.date().toordinal()
            )

        if hourly:
            h1 = hourly[0]
            _LOGGER.debug(
                "Forecast: %d hours, +1h=%s %.1f°C %d%% precip",
                len(hourly),
                h1.condition,
                h1.temperature,
                h1.precipitation_probability,
            )

        self._record_pressure(s.pressure, s.temperature)

        midnight = int(dt_util.start_of_local_day(now_local).timestamp())
        seen = [t for _h, t in self.history.observed(midnight + 1, int(now_local.timestamp()) + 1)]

        return ForecastResult(
            generated=now_local,
            condition=HA_CONDITIONS[current_idx],
            hourly=hourly,
            attributes=self._build_attributes(hourly),
            temperature=round(s.temperature, 1),
            apparent_temperature=self._apparent_temperature(),
            pressure=round(s.pressure, 1),
            humidity=(round(s.humidity) if sensors.humidity else None),
            wind_speed=(round(s.wind_speed, 1) if sensors.wind else None),
            wind_bearing=bearing,
            dew_point=(s.dew_point if sensors.humidity else None),
            hourly_dicts=_hourly_dicts(hourly, now_local),
            daily=daily,
            today_observed=(round(max(seen), 1), round(min(seen), 1)) if seen else None,
        )

    def _record_pressure(self, pressure: float, temperature: float) -> None:
        """Append an hourly sea-level sample and let the tide learn from it."""
        if not _SEA_LEVEL_MIN <= pressure <= _SEA_LEVEL_MAX:
            return
        # Wall clock on purpose: both are persisted across restarts, and the tide
        # is phased to the real sun.
        now = time.time()
        if self.pressure_history.record(now, pressure):
            self.tide.learn(now, pressure, temperature)
            self._store.async_delay_save(
                lambda: {"samples": self.pressure_history.dump(), "tide": self.tide.dump()}, 60
            )

    def _apparent_temperature(self) -> float:
        """Feels-like temperature — wind chill or heat index."""
        s = self._estimator.state
        temp, wind, humidity = s.temperature, s.wind_speed, s.humidity
        # Wind chill (Environment Canada formula, T < 10 °C, W > 4.8 km/h)
        wind_kmh = wind * 3.6
        if self.has_source(CONF_WIND_SPEED_SENSOR) and temp <= 10.0 and wind_kmh > 4.8:
            return round(
                13.12 + 0.6215 * temp - 11.37 * wind_kmh**0.16 + 0.3965 * temp * wind_kmh**0.16,
                1,
            )
        # Heat index (Steadman, T > 27 °C)
        if self.has_source(CONF_HUMIDITY_SENSOR) and temp >= 27.0 and humidity >= 40:
            return round(
                -8.785
                + 1.611 * temp
                + 2.339 * humidity
                - 0.1461 * temp * humidity
                - 0.01231 * temp * temp
                - 0.01642 * humidity * humidity
                + 0.002212 * temp * temp * humidity
                + 0.000725 * temp * humidity * humidity
                - 0.000003582 * temp * temp * humidity * humidity,
                1,
            )
        return round(temp, 1)

    def _build_attributes(self, hourly: list[HourForecast]) -> dict[str, Any]:
        s = self._estimator.state
        attrs: dict[str, Any] = {
            "pressure_trend": round(s.dp_dt, 2),
            "pressure_acceleration": round(s.d2p_dt2, 2),
            "front_warm": s.front_warm,
            "front_cold": s.front_cold,
            "front_occluded": s.front_occluded,
        }
        # Moisture is derived from relative humidity; without that sensor the
        # numbers would be pure fiction, so publish nothing instead.
        if self.has_source(CONF_HUMIDITY_SENSOR):
            attrs["dew_point"] = s.dew_point
            attrs["dew_depression"] = s.dew_depression
            attrs["wet_bulb"] = s.wet_bulb
        if self.has_source(CONF_WIND_SPEED_SENSOR):
            force = _beaufort(s.wind_speed)
            attrs["wind_force"] = force
            attrs["wind_force_description"] = _BEAUFORT_NAMES[force]
        if hourly:
            h1 = hourly[0]
            attrs["next_hour_condition"] = h1.condition
            attrs["next_hour_precip_probability"] = h1.precipitation_probability
            # Aggregate precipitation probability over the next 6 h.
            # Hourly probabilities are strongly correlated (not independent),
            # so compounding 1-Π(1-pᵢ) badly overstates the risk — a sunny
            # day reads ~35%.  The window maximum is the honest "chance of
            # rain in the next 6 hours" for a non-technical dashboard.
            if len(hourly) >= 6:
                attrs["precip_probability_6h"] = max(hf.precipitation_probability for hf in hourly[:6])
        return attrs

    # ------------------------------------------------------------------
    #  Sun
    # ------------------------------------------------------------------

    def _sun_elevation(self) -> float:
        """Return current sun elevation in degrees from sun.sun entity."""
        if sun := self.hass.states.get("sun.sun"):
            try:
                if (elevation := sun.attributes.get("elevation")) is not None:
                    return float(elevation)
            except (ValueError, TypeError):
                pass
        return 45.0  # fallback: assume mid-sky


def _hourly_dicts(hourly: list[HourForecast], generated: datetime) -> list[dict[str, Any]]:
    """Hourly forecast as plain dicts for the meteogram sensor attribute.

    Mirrors what ``get_forecasts`` returns but in the entity's native units,
    so a dashboard meteogram card can read it straight off the sensor without
    a websocket subscription.  Values arrive already rounded.
    """
    return [
        {
            "datetime": (generated + timedelta(hours=hf.hours_ahead)).isoformat(),
            "condition": hf.condition,
            "temperature": hf.temperature,
            "humidity": hf.humidity,
            "pressure": hf.pressure,
            "precipitation_probability": hf.precipitation_probability,
            "precipitation": hf.precipitation_amount,
            "wind_speed": hf.wind_speed,
            "wind_bearing": hf.wind_bearing,
            "is_daytime": hf.is_daytime,
        }
        for hf in hourly
    ]


type LocalForecastConfigEntry = ConfigEntry[LocalForecastCoordinator]
