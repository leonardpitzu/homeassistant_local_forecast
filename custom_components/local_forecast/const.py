"""Constants for the Local Weather Forecast integration."""

from typing import Final

DOMAIN: Final = "local_forecast"

# --- Config keys: required sensors ---
CONF_PRESSURE_SENSOR: Final = "pressure_sensor"
CONF_TEMPERATURE_SENSOR: Final = "temperature_sensor"

# --- Config keys: optional sensors ---
CONF_HUMIDITY_SENSOR: Final = "humidity_sensor"
CONF_WIND_SPEED_SENSOR: Final = "wind_speed_sensor"
CONF_WIND_DIRECTION_SENSOR: Final = "wind_direction_sensor"
CONF_SOLAR_RADIATION_SENSOR: Final = "solar_radiation_sensor"
CONF_RAIN_RATE_SENSOR: Final = "rain_rate_sensor"

# --- Config keys: station metadata ---
CONF_ELEVATION: Final = "elevation"
CONF_PRESSURE_TYPE: Final = "pressure_type"

# --- Pressure types ---
PRESSURE_ABSOLUTE: Final = "absolute"
PRESSURE_RELATIVE: Final = "relative"

# --- Defaults ---
DEFAULT_ELEVATION: Final = 0
DEFAULT_PRESSURE_TYPE: Final = PRESSURE_ABSOLUTE

# --- Physical constants ---
LAPSE_RATE: Final = 0.0065          # K/m  (ISA tropospheric lapse rate)
GRAVITY_EXPONENT: Final = 5.257     # g/(L·R) for barometric formula
KELVIN_OFFSET: Final = 273.15
STEFAN_BOLTZMANN: Final = 5.67e-8   # W/(m²·K⁴)

# ---------------------------------------------------------------------------
#  Internal weather states
#
#  These map 1:1 to the HA condition strings that drive dashboard icons.
#  Order matters — the index is used in the Bayesian transition matrix.
#
#  HA icon reference (what your family actually sees on cards/phone/tablet):
#    sunny            → ☀️   bright sun
#    clear-night      → 🌙   moon + stars
#    partlycloudy     → ⛅   sun behind small cloud
#    cloudy           → ☁️   thick cloud
#    fog              → 🌫️   three horizontal lines
#    rainy            → 🌧️   cloud with rain drops
#    pouring          → 🌧️⬇  cloud with heavy rain
#    snowy            → 🌨️   cloud with snowflake
#    snowy-rainy      → 🌨🌧  mixed snow/rain (sleet)
#    lightning-rainy  → ⛈️   cloud with lightning + rain
#    windy            → 💨   wind lines
#    exceptional      → ⚠️   warning triangle
# ---------------------------------------------------------------------------

S_CLEAR: Final = 0
S_CLEAR_NIGHT: Final = 1
S_PARTLY_CLOUDY: Final = 2
S_CLOUDY: Final = 3
S_FOG: Final = 4
S_RAINY: Final = 5
S_POURING: Final = 6
S_SNOWY: Final = 7
S_SNOWY_RAINY: Final = 8
S_LIGHTNING_RAINY: Final = 9
S_WINDY: Final = 10
S_EXCEPTIONAL: Final = 11

NUM_STATES: Final = 12

# Index → HA condition string  (this drives every icon your family sees)
HA_CONDITIONS: Final = [
    "sunny",              # 0  S_CLEAR
    "clear-night",        # 1  S_CLEAR_NIGHT
    "partlycloudy",       # 2  S_PARTLY_CLOUDY
    "cloudy",             # 3  S_CLOUDY
    "fog",                # 4  S_FOG
    "rainy",              # 5  S_RAINY
    "pouring",            # 6  S_POURING
    "snowy",              # 7  S_SNOWY
    "snowy-rainy",        # 8  S_SNOWY_RAINY
    "lightning-rainy",    # 9  S_LIGHTNING_RAINY
    "windy",              # 10 S_WINDY
    "exceptional",        # 11 S_EXCEPTIONAL
]

# ---------------------------------------------------------------------------
#  Precipitation type thresholds (wet-bulb based, WMO)
#
#  Wet-bulb temperature determines what falls from the sky better than
#  dry-bulb alone because it accounts for evaporative cooling as
#  precipitation descends through the atmosphere.
#
#  Tw < -2 °C  →  snow        (frozen all the way down)
#  -2 ≤ Tw < 1 →  sleet/mixed (partial melting)
#  Tw ≥ 1  °C  →  rain        (liquid)
# ---------------------------------------------------------------------------
WET_BULB_SNOW: Final = -2.0
WET_BULB_MIX_UPPER: Final = 1.0

# --- Rain intensity (mm/h) ---
RAIN_LIGHT: Final = 0.5
RAIN_MODERATE: Final = 2.5
RAIN_HEAVY: Final = 7.5

# --- Fog ---
FOG_DEW_DEPRESSION: Final = 1.5   # °C
FOG_MAX_WIND: Final = 3.0         # m/s

# --- Wind ---
WIND_STRONG: Final = 10.0         # m/s  (Beaufort 5-6)

# --- Thunderstorm proxy ---
STORM_PRESSURE_DROP: Final = -3.0  # hPa/h
STORM_HUMIDITY: Final = 80.0       # %
STORM_WIND: Final = 8.0            # m/s

# --- History ring-buffer ---
HISTORY_HOURS: Final = 6
HISTORY_MAX_RECORDS: Final = 360

# --- Forecast horizon ---
FORECAST_HOURS: Final = 12
