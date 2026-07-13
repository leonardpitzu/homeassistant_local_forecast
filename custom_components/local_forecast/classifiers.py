"""Pure display-state classifiers for Local Weather Forecast sensors.

Barometer category, pressure-tendency direction and frontal identity are
plain functions of numeric inputs — no Home Assistant state — so they live
here, testable in isolation like the rest of the estimator/physics layer.
The sensor platform imports the options lists and these functions.
"""

from __future__ import annotations

# --- Barometer (sea-level pressure bands, tendency-aware) ---
# Sea-level pressure bands (hPa) for the barometer enum, low → high.
BAROMETER_OPTIONS = ["stormy", "rain", "change", "fair", "very_dry"]
BAROMETER_BANDS = (985.0, 1005.0, 1020.0, 1035.0)
# Tendency (hPa/h) beyond which the needle is read one category up/down.
BAROMETER_TENDENCY_STEP = 0.5


def barometer_state(pressure: float | None, tendency: float | None) -> str | None:
    """Classify sea-level pressure into a tendency-aware barometer category."""
    if pressure is None:
        return None
    band = 0
    for threshold in BAROMETER_BANDS:
        if pressure >= threshold:
            band += 1
    if tendency is not None:
        if tendency <= -BAROMETER_TENDENCY_STEP:
            band -= 1
        elif tendency >= BAROMETER_TENDENCY_STEP:
            band += 1
    band = max(0, min(len(BAROMETER_OPTIONS) - 1, band))
    return BAROMETER_OPTIONS[band]


# --- Pressure-tendency direction (companion to the numeric sensor) ---
# Thresholds in hPa/h: |t| < 0.3 steady, 0.3-1.0 slow, >= 1.0 fast.
TENDENCY_DIRECTION_OPTIONS = [
    "falling_fast",
    "falling",
    "steady",
    "rising",
    "rising_fast",
]
TENDENCY_STEADY = 0.3
TENDENCY_FAST = 1.0


def tendency_direction(tendency: float | None) -> str | None:
    """Classify a pressure tendency (hPa/h) into a direction enum."""
    if tendency is None:
        return None
    if tendency >= TENDENCY_FAST:
        return "rising_fast"
    if tendency >= TENDENCY_STEADY:
        return "rising"
    if tendency <= -TENDENCY_FAST:
        return "falling_fast"
    if tendency <= -TENDENCY_STEADY:
        return "falling"
    return "steady"


# --- Frontal identity (single mutually-exclusive enum) ---
# The three signatures can overlap; report the most-developed one
# (occluded > cold > warm).
FRONT_OPTIONS = ["none", "warm", "cold", "occluded"]


def front_state(
    warm: bool | None, cold: bool | None, occluded: bool | None
) -> str:
    """Collapse the three frontal flags into one mutually-exclusive state."""
    if occluded:
        return "occluded"
    if cold:
        return "cold"
    if warm:
        return "warm"
    return "none"
