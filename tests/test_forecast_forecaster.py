"""The trained forecaster: shipped models, evaluation, icons, missing sensors."""

from datetime import date
import math

import pytest

from local_forecast.features import DAILY, HOURLY
from local_forecast.forecast import Forecaster, Sensors, condition
from local_forecast.history import HourlyHistory
from local_forecast.model import DAILY_TARGETS, HOURLY_TARGETS, Model, ModelSet

H = 3600
NOW = 1_790_000_000 - 1_790_000_000 % H  # a whole hour
LAT, LON, ELEV = 45.67, 25.60, 544.0
FLAT_TIDE = lambda ts: 0.0  # noqa: E731


@pytest.fixture(scope="module")
def models() -> ModelSet:
    return ModelSet.load()


def test_every_shipped_model_reads_exactly_these_inputs(models):
    """Training and Home Assistant must agree on every input, in order."""
    assert set(models.hourly) == set(HOURLY_TARGETS)
    assert set(models.daily) == set(DAILY_TARGETS)
    for m in models.hourly.values():
        assert m.features == HOURLY
    for m in models.daily.values():
        assert m.features == DAILY


def test_tree_walk():
    tree = [[0, -1, -1], [1.5, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0], [0.0, -1.0, 2.0]]
    m = Model({"features": ["x"], "link": "identity", "base": 10.0, "trees": [tree]})
    assert m.predict([1.0]) == 9.0
    assert m.predict([1.5]) == 9.0  # <= goes left
    assert m.predict([2.0]) == 12.0
    assert m.predict([math.nan]) == 12.0  # missing goes right here
    logit = Model({"features": ["x"], "link": "logit", "base": 0.0, "trees": []})
    assert logit.predict([0.0]) == 0.5


def _history(days: int = 4, humidity: bool = True, wind: bool = True, rain: bool = True) -> HourlyHistory:
    h = HourlyHistory()
    for k in range(days * 24, -1, -1):
        ts = NOW - k * H
        hour = (ts / H) % 24
        h.add(
            ts,
            temperature=12.0 + 6.0 * math.sin(2 * math.pi * (hour - 9) / 24),
            humidity=70.0 if humidity else None,
            wind_speed=3.0 if wind else None,
            wind_direction=200.0 if wind else None,
            pressure=1015.0 - 0.05 * (days * 24 - k) / 24,
            rain_rate=0.0 if rain else None,
        )
    return h


def _forecaster(models: ModelSet, sensors: Sensors, history: HourlyHistory) -> Forecaster:
    f = Forecaster(models, LAT, LON, ELEV)
    today = date(2026, 9, 21).toordinal()
    assert f.run_issue(history, NOW, FLAT_TIDE, 10.0, today, sensors) is not None
    return f


def test_full_station_forecast(models):
    sensors = Sensors(humidity=True, wind=True, wind_direction=True, rain=True)
    f = _forecaster(models, sensors, _history())
    hours = f.hours(NOW + 1200, FLAT_TIDE, sensors, 200)
    assert [h.hours_ahead for h in hours] == list(range(1, 25))
    for h in hours:
        assert -30 < h.temperature < 45
        assert 1 <= h.humidity <= 100
        assert 950 < h.pressure < 1080
        assert 0 <= h.precipitation_probability <= 100
        assert h.wind_speed >= 0 and h.wind_bearing == 200
    days = f.days(sensors, 70.0, date(2026, 9, 21).toordinal())
    assert [d.date for d in days] == [date(2026, 9, 22).toordinal(), date(2026, 9, 23).toordinal()]
    for d in days:
        assert d.templow <= d.temperature
        assert 0 <= d.precipitation_probability <= 100


def test_minimal_station_publishes_no_invented_channels(models):
    sensors = Sensors()
    f = _forecaster(models, sensors, _history(humidity=False, wind=False, rain=False))
    for h in f.hours(NOW + 600, FLAT_TIDE, sensors, None):
        assert h.humidity is None and h.wind_speed is None and h.wind_bearing is None
        assert not math.isnan(h.temperature)
    for d in f.days(sensors, None, date(2026, 9, 21).toordinal()):
        assert d.wind_speed is None


def test_brand_new_install_still_forecasts(models):
    """One hour of history: the models fall back on season and place, and still answer."""
    sensors = Sensors(humidity=True, wind=True, wind_direction=True, rain=True)
    f = _forecaster(models, sensors, _history(days=0))
    assert len(f.hours(NOW + 900, FLAT_TIDE, sensors, 90)) == 24
    assert len(f.days(sensors, 70.0, date(2026, 9, 21).toordinal())) == 2


def test_just_after_midnight_still_has_the_day_after_tomorrow(models):
    """Issued at 23:00 yesterday, it must still cover today + 2."""
    sensors = Sensors(humidity=True)
    f = _forecaster(models, sensors, _history())
    days = f.days(sensors, 70.0, date(2026, 9, 22).toordinal())
    assert [d.date for d in days] == [date(2026, 9, 23).toordinal(), date(2026, 9, 24).toordinal()]


def test_self_correction_learns_from_observed_hours(models):
    sensors = Sensors(humidity=True, wind=True, wind_direction=True, rain=True)
    history = _history()
    f = _forecaster(models, sensors, history)
    for k in range(1, 7):
        history.add(NOW + k * H, temperature=40.0, humidity=70.0, wind_speed=3.0, pressure=1015.0, rain_rate=0.0)
    f.verify(history, NOW + 6 * H, sensors)
    assert f.recal.verified >= 6 * 4
    assert max(f.recal.shift["t"][0]) > 0  # far warmer than forecast: the bias moved up


@pytest.mark.parametrize(
    ("kw", "icon"),
    [
        ({"rain": 0.9, "temperature": 15.0}, "rainy"),
        ({"rain": 0.9, "temperature": -5.0}, "snowy"),
        ({"rain": 0.9, "thunder": 0.8, "temperature": 25.0}, "lightning-rainy"),
        ({"rain": 0.9, "heavy": True, "temperature": 20.0}, "pouring"),
        ({"fog": 0.9}, "fog"),
        ({"cloud": 0.95}, "cloudy"),
        ({"cloud": 0.5}, "partlycloudy"),
        ({}, "sunny"),
        ({"daytime": False}, "clear-night"),
        ({"wind": 15.0}, "windy"),
    ],
)
def test_icons(kw, icon):
    args = {
        "rain": 0.05,
        "thunder": 0.0,
        "fog": 0.0,
        "cloud": 0.1,
        "heavy": False,
        "temperature": 15.0,
        "humidity": 60.0,
        "wind": 2.0,
        "daytime": True,
    }
    args.update(kw)
    assert condition(**args) == icon
