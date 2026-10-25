"""Self-correction: learns a station's bias, never past its bounds."""

import pytest

from local_forecast.recal import DAY_PARTS, MAX_LOGIT, MAX_SHIFT_C, SCALE_RANGE, Recalibrator

NOON, NIGHT = 12.5, 2.5  # solar hours


def test_starts_as_the_model():
    r = Recalibrator()
    assert r.hourly("t", 5, 12.3, NOON) == 12.3
    assert r.hourly("wet", 5, 0.3, NOON) == pytest.approx(0.3)
    assert r.daily("tmax", 1, 20.0) == 20.0


def test_learns_a_sensor_that_runs_warm_at_noon_only():
    r = Recalibrator()
    for _ in range(3000):
        r.learn_hourly("t", 2, 10.0, 11.5, NOON)
        r.learn_hourly("t", 2, 5.0, 5.0, NIGHT)
    assert r.hourly("t", 2, 10.0, NOON) == pytest.approx(11.5, abs=0.05)
    assert r.hourly("t", 2, 5.0, NIGHT) == pytest.approx(5.0, abs=0.01)
    assert r.hourly("t", 20, 10.0, NOON) == 10.0  # other ranges untouched


def test_a_sheltered_anemometer_is_learned():
    """A backyard sensor sees 40 % of the airport wind the models learned."""
    r = Recalibrator()
    for _ in range(5000):
        r.learn_hourly("ws", 2, 3.0, 1.2, NOON)
    assert r.hourly("ws", 2, 3.0, NOON) == pytest.approx(1.2, abs=0.05)


def test_a_broken_sensor_cannot_push_past_the_bounds():
    r = Recalibrator()
    for _ in range(50000):
        r.learn_hourly("t", 1, 10.0, 40.0, NOON)
        r.learn_hourly("ws", 1, 3.0, 30.0, NOON)
        r.learn_hourly("wet", 1, 0.5, 0.0, NOON)  # a blocked gauge: never rains
    assert r.hourly("t", 1, 10.0, NOON) == pytest.approx(10.0 + MAX_SHIFT_C)
    assert r.hourly("ws", 1, 3.0, NOON) == pytest.approx(3.0 * SCALE_RANGE[1])
    assert r.logit[0] == pytest.approx(-MAX_LOGIT)
    assert r.hourly("wet", 1, 0.5, NOON) > 0.04  # still forecasts rain, just less


def test_rain_chance_calibrates_to_the_station():
    r = Recalibrator()
    # Model says 30 %, the station's own gauge sees rain 10 % of the time.
    for i in range(30000):
        r.learn_hourly("wet", 3, 0.3, 1.0 if i % 10 == 0 else 0.0, NOON)
    assert r.hourly("wet", 3, 0.3, NOON) == pytest.approx(0.1, abs=0.02)


def test_persistence_round_trip_and_garbage():
    r = Recalibrator()
    for _ in range(100):
        r.learn_daily("tmin", 2, 0.0, -2.0)
        r.learn_hourly("td", 8, 3.0, 4.0, NIGHT)
    g = Recalibrator()
    g.load(r.dump())
    assert g.daily("tmin", 2, 0.0) == r.daily("tmin", 2, 0.0) != 0.0
    assert g.hourly("td", 8, 3.0, NIGHT) == r.hourly("td", 8, 3.0, NIGHT) != 3.0
    g.load({"shift": "nonsense"})
    g.load({**r.dump(), "shift": {"t": [[0.0] * (DAY_PARTS - 1)] * 4, "td": [[0.0] * DAY_PARTS] * 4}})
    assert g.daily("tmin", 2, 0.0) == r.daily("tmin", 2, 0.0)
