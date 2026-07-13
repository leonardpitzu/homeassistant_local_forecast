"""Tests for the pure classifier helpers in local_forecast.classifiers.

Barometer category, pressure-tendency direction and frontal identity are
pure functions of numeric inputs, so they are unit-tested directly without
any Home Assistant dependency.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "custom_components"),
)

from local_forecast.classifiers import (
    BAROMETER_OPTIONS,
    FRONT_OPTIONS,
    TENDENCY_DIRECTION_OPTIONS,
    barometer_state,
    front_state,
    tendency_direction,
)


class TestBarometerState:
    def test_none_pressure(self):
        assert barometer_state(None, None) is None

    def test_bands_no_tendency(self):
        assert barometer_state(980.0, None) == "stormy"
        assert barometer_state(995.0, None) == "rain"
        assert barometer_state(1012.0, None) == "change"
        assert barometer_state(1025.0, None) == "fair"
        assert barometer_state(1040.0, None) == "very_dry"

    def test_band_boundaries_inclusive_lower(self):
        # >= threshold moves up a band
        assert barometer_state(985.0, None) == "rain"
        assert barometer_state(984.99, None) == "stormy"
        assert barometer_state(1035.0, None) == "very_dry"

    def test_falling_tendency_demotes_one(self):
        # fair (1025) with a strong fall reads one worse -> change
        assert barometer_state(1025.0, -0.6) == "change"

    def test_rising_tendency_promotes_one(self):
        # change (1010) with a strong rise reads one better -> fair
        assert barometer_state(1010.0, 0.6) == "fair"

    def test_small_tendency_no_shift(self):
        assert barometer_state(1025.0, -0.4) == "fair"
        assert barometer_state(1010.0, 0.4) == "change"

    def test_shift_clamped_at_ends(self):
        assert barometer_state(980.0, -2.0) == "stormy"  # can't go below
        assert barometer_state(1040.0, 2.0) == "very_dry"  # can't exceed

    def test_options_order(self):
        assert BAROMETER_OPTIONS == [
            "stormy",
            "rain",
            "change",
            "fair",
            "very_dry",
        ]


class TestTendencyDirection:
    def test_none(self):
        assert tendency_direction(None) is None

    def test_bands(self):
        assert tendency_direction(1.5) == "rising_fast"
        assert tendency_direction(1.0) == "rising_fast"
        assert tendency_direction(0.5) == "rising"
        assert tendency_direction(0.3) == "rising"
        assert tendency_direction(0.1) == "steady"
        assert tendency_direction(0.0) == "steady"
        assert tendency_direction(-0.1) == "steady"
        assert tendency_direction(-0.3) == "falling"
        assert tendency_direction(-0.9) == "falling"
        assert tendency_direction(-1.0) == "falling_fast"
        assert tendency_direction(-2.5) == "falling_fast"

    def test_options_order(self):
        assert TENDENCY_DIRECTION_OPTIONS == [
            "falling_fast",
            "falling",
            "steady",
            "rising",
            "rising_fast",
        ]


class TestFrontState:
    def test_no_front(self):
        assert front_state(False, False, False) == "none"
        assert front_state(None, None, None) == "none"

    def test_single_flags(self):
        assert front_state(True, False, False) == "warm"
        assert front_state(False, True, False) == "cold"
        assert front_state(False, False, True) == "occluded"

    def test_priority_occluded_over_cold_over_warm(self):
        assert front_state(True, True, True) == "occluded"
        assert front_state(True, True, False) == "cold"
        assert front_state(True, False, True) == "occluded"

    def test_options_order(self):
        assert FRONT_OPTIONS == ["none", "warm", "cold", "occluded"]
