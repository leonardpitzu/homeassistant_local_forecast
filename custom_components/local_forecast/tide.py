"""Atmospheric tide — the twice- and once-daily pressure swing, in pascals.

Surface pressure carries a solar-driven oscillation that has nothing to do with
the weather: a semidiurnal S2 term (the global resonance, ~0.4 hPa at 45 deg,
peaking near 10:00 local solar time) and a diurnal S1 term (thermally driven and
strongly regional). Left in place it is indistinguishable from a real tendency —
its own slope reaches ~0.31 hPa/h at mid-latitudes and ~0.61 hPa/h in the
tropics, against a 0.3 hPa/h steady threshold.

Constants come from the Dai & Wang (1999) gridded climatology (22 years of
station data). S1 is carried as a zonal-mean table rather than a law because it
is largely non-migrating; the table is the vector zonal mean, which is the
least-squares optimum for any latitude-only model.

No Home Assistant dependencies — pure Python + math.
"""

from __future__ import annotations

import math

# S2 follows Haurwitz's cos^3 law; the phase is the observed 10.0 h, not the
# 9.5 h often quoted.
S2_AMP_EQUATOR_PA = 116.0
S2_AMP_DEFAULT_PA = 41.0  # ~45 deg latitude
S2_PHASE_H = 10.0

# Annual zonal-mean S1 as (real, imag) pascals every 15 deg from -90. Components
# are interpolated rather than amplitude and phase, because phase wraps at 24 h.
# The poles are zero: a sun-synchronous wave-1 pattern is degenerate there, so
# the grid's polar values are sparse-station noise.
S1_NODE_STEP_DEG = 15.0
S1_ZONAL_PA: tuple[tuple[float, float], ...] = (
    (0.00, 0.00),
    (0.00, 0.00),
    (25.25, 7.64),
    (-3.93, -11.95),
    (-7.56, 19.41),
    (-12.56, 43.97),
    (-4.09, 56.07),
    (-9.25, 44.95),
    (-18.34, 30.63),
    (-15.87, 15.46),
    (0.27, 7.11),
    (0.00, 0.00),
    (0.00, 0.00),
)
S1_DEFAULT_PA = 40.0
S1_DEFAULT_PHASE_H = 4.8


def s2_amplitude(latitude_deg: float | None) -> float:
    """Semidiurnal amplitude in Pa: A = 1.16 hPa * cos^3(lat)."""
    if latitude_deg is None:
        return S2_AMP_DEFAULT_PA
    c = math.cos(math.radians(latitude_deg))
    return S2_AMP_EQUATOR_PA * c * c * c


def s1_tide(latitude_deg: float | None) -> tuple[float, float]:
    """Diurnal amplitude in Pa and phase in hours of local solar time."""
    if latitude_deg is None:
        return S1_DEFAULT_PA, S1_DEFAULT_PHASE_H
    lat = max(-90.0, min(90.0, latitude_deg))
    span = (len(S1_ZONAL_PA) - 1) * S1_NODE_STEP_DEG
    x = (lat + 90.0) / span * (len(S1_ZONAL_PA) - 1)
    k = min(int(x), len(S1_ZONAL_PA) - 2)
    f = x - k
    re = S1_ZONAL_PA[k][0] + f * (S1_ZONAL_PA[k + 1][0] - S1_ZONAL_PA[k][0])
    im = S1_ZONAL_PA[k][1] + f * (S1_ZONAL_PA[k + 1][1] - S1_ZONAL_PA[k][1])
    return math.hypot(re, im), math.degrees(math.atan2(im, re)) / 15.0 % 24.0


def solar_hour(timestamp: float, longitude_deg: float | None) -> float:
    """Local mean solar time in hours, from an epoch timestamp.

    Deliberately derived from UTC and longitude rather than the configured
    timezone: the tide is phased to the sun, and a timezone can sit over an hour
    away from it (further still under DST).
    """
    utc_hour = (timestamp / 3600.0) % 24.0
    if longitude_deg is None:
        return utc_hour
    return (utc_hour + longitude_deg / 15.0) % 24.0


def tide_pa(hour: float, latitude_deg: float | None = None) -> float:
    """Combined S2 + S1 tide in pascals, at a local solar hour."""
    s1_amp, s1_phase = s1_tide(latitude_deg)
    return s2_amplitude(latitude_deg) * math.cos(2.0 * math.pi * (hour - S2_PHASE_H) / 12.0) + s1_amp * math.cos(
        2.0 * math.pi * (hour - s1_phase) / 24.0
    )


def tide_hpa_at(timestamp: float, latitude_deg: float | None, longitude_deg: float | None) -> float:
    """Tide contribution in hPa at an epoch timestamp, ready to subtract."""
    return tide_pa(solar_hour(timestamp, longitude_deg), latitude_deg) / 100.0
