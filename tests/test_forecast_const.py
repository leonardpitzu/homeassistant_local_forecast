"""Tests for local_forecast.const — constant definitions and state mapping."""

import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "custom_components"),
)

from local_forecast.const import (
    DOMAIN,
    HA_CONDITIONS,
    NUM_STATES,
    S_CLEAR,
    S_CLEAR_NIGHT,
    S_CLOUDY,
    S_EXCEPTIONAL,
    S_FOG,
    S_LIGHTNING_RAINY,
    S_PARTLY_CLOUDY,
    S_POURING,
    S_RAINY,
    S_SNOWY,
    S_SNOWY_RAINY,
    S_WINDY,
    WET_BULB_MIX_UPPER,
    WET_BULB_SNOW,
)


class TestConstants:
    """Verify constant integrity."""

    def test_domain_name(self):
        assert DOMAIN == "local_forecast"

    def test_num_states_matches_conditions(self):
        assert NUM_STATES == 12
        assert len(HA_CONDITIONS) == NUM_STATES

    def test_state_indices_unique(self):
        indices = [
            S_CLEAR, S_CLEAR_NIGHT, S_PARTLY_CLOUDY, S_CLOUDY,
            S_FOG, S_RAINY, S_POURING, S_SNOWY, S_SNOWY_RAINY,
            S_LIGHTNING_RAINY, S_WINDY, S_EXCEPTIONAL,
        ]
        assert len(set(indices)) == NUM_STATES
        assert set(indices) == set(range(NUM_STATES))

    def test_ha_conditions_are_valid_strings(self):
        """Every condition must be a recognized HA weather condition."""
        valid = {
            "sunny", "clear-night", "partlycloudy", "cloudy", "fog",
            "rainy", "pouring", "snowy", "snowy-rainy",
            "lightning-rainy", "windy", "exceptional",
        }
        assert set(HA_CONDITIONS) == valid

    def test_condition_index_mapping(self):
        assert HA_CONDITIONS[S_CLEAR] == "sunny"
        assert HA_CONDITIONS[S_CLEAR_NIGHT] == "clear-night"
        assert HA_CONDITIONS[S_SNOWY] == "snowy"
        assert HA_CONDITIONS[S_SNOWY_RAINY] == "snowy-rainy"
        assert HA_CONDITIONS[S_LIGHTNING_RAINY] == "lightning-rainy"

    def test_wet_bulb_thresholds_ordered(self):
        assert WET_BULB_SNOW < WET_BULB_MIX_UPPER
