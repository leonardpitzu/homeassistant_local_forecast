"""Constants for the Local Weather Forecast integration."""

from typing import TYPE_CHECKING, Final

from homeassistant.util.hass_dict import HassKey

if TYPE_CHECKING:
    from .map import MapState

DOMAIN: Final = "local_forecast"

# Everything the integration keeps outside a config entry lives behind this one
# typed key: the map gate plus the shared EUMETView frame-time cache.
DATA_MAP: HassKey["MapState"] = HassKey(f"{DOMAIN}_map")

# Persisted sea-level pressure buffer.
STORAGE_VERSION: Final = 1

# Refresh policy. The interval is the safety net for a station whose sensors
# sit perfectly still; the debounce is the coalescing window for chatty ones.
UPDATE_INTERVAL_MINUTES: Final = 5
UPDATE_DEBOUNCE_SECONDS: Final = 30.0

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

# --- Config keys: optional satellite map ---
CONF_ENABLE_MAP: Final = "enable_map"

# --- Pressure types ---
PRESSURE_ABSOLUTE: Final = "absolute"
PRESSURE_RELATIVE: Final = "relative"

# --- Defaults ---
DEFAULT_ELEVATION: Final = 0
DEFAULT_PRESSURE_TYPE: Final = PRESSURE_ABSOLUTE
DEFAULT_ENABLE_MAP: Final = False

# --- Physical constants ---
LAPSE_RATE: Final = 0.0065  # K/m  (ISA tropospheric lapse rate)
GRAVITY_EXPONENT: Final = 5.257  # g/(L·R) for barometric formula
KELVIN_OFFSET: Final = 273.15

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
    "sunny",  # 0  S_CLEAR
    "clear-night",  # 1  S_CLEAR_NIGHT
    "partlycloudy",  # 2  S_PARTLY_CLOUDY
    "cloudy",  # 3  S_CLOUDY
    "fog",  # 4  S_FOG
    "rainy",  # 5  S_RAINY
    "pouring",  # 6  S_POURING
    "snowy",  # 7  S_SNOWY
    "snowy-rainy",  # 8  S_SNOWY_RAINY
    "lightning-rainy",  # 9  S_LIGHTNING_RAINY
    "windy",  # 10 S_WINDY
    "exceptional",  # 11 S_EXCEPTIONAL
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
RAIN_HEAVY: Final = 7.5

# --- Fog ---
FOG_DEW_DEPRESSION: Final = 1.5  # °C
FOG_MAX_WIND: Final = 3.0  # m/s

# --- Wind ---
WIND_STRONG: Final = 10.0  # m/s  (Beaufort 5-6)

# --- Thunderstorm proxy ---
STORM_PRESSURE_DROP: Final = -3.0  # hPa/h
STORM_HUMIDITY: Final = 80.0  # %
STORM_WIND: Final = 8.0  # m/s

# --- History ring-buffer ---
# The trend code looks back in *time* (1 h for slopes, 3 h for curvature), so
# the window is bounded by age; the record cap is only a memory backstop for
# pathologically fast sensors.
HISTORY_MAX_RECORDS: Final = 2000
HISTORY_SECONDS: Final = 4 * 3600.0

# --- Forecast horizon ---
FORECAST_HOURS: Final = 12

# ---------------------------------------------------------------------------
#  Optional satellite map view (Leaflet over EUMETView WMS)
#
#  When enabled, the integration serves an interactive pan/zoom satellite
#  viewer at /api/local_forecast/map, centred live on the Home Assistant home
#  location. The browser tiles EUMETView's public, anonymous WMS GetMap
#  endpoint directly — no polling, no entities, no token. Embed with an
#  `iframe` card. Disabled by default; toggle it in the integration options.
# ---------------------------------------------------------------------------
WMS_BASE_URL: Final = "https://view.eumetsat.int/geoserver/ows"
WMS_VERSION: Final = "1.3.0"  # WMS 1.3.0 + EPSG:4326 → lat,lon axis order.
MAP_VIEW_URL: Final = "/api/local_forecast/map"
MAP_STATIC_URL: Final = "/local_forecast_static"
MAP_DEFAULT_ZOOM: Final = 6
MAP_MAX_ZOOM: Final = 9

# The viewer is reachable without authentication (an `iframe` card cannot
# present a token), so the centre is snapped to this grid — enough to frame
# the right region at zoom 6, not enough to locate a house.
MAP_CENTER_GRID: Final = 0.25

# Fallback centre when Home Assistant has no home location configured
# (central Europe). Normally hass.config.latitude/longitude is used live.
MAP_FALLBACK_CENTER: Final = (45.0, 25.0)

# Curated EUMETView RGB layers (layer id, friendly name), taken from the live
# WMS GetCapabilities document. Shown in the viewer's layer switcher.
MAP_LAYERS: Final[list[tuple[str, str]]] = [
    ("mtg_fd:rgb_geocolour", "Geo Colour"),
    ("msg_fes:rgb_natural", "Natural Colour"),
    ("msg_fes:rgb_naturalenhncd", "Natural Colour Enhanced"),
    ("msg_fes:rgb_dust", "Dust"),
    ("msg_fes:rgb_airmass", "Airmass"),
    ("msg_fes:rgb_convection", "Convection"),
    ("msg_fes:rgb_ash", "Volcanic Ash"),
    ("msg_fes:rgb_fog", "Fog / Low Clouds"),
]

# ---------------------------------------------------------------------------
#  Frame pinning (see wms_time.py)
#
#  Without a TIME parameter every tile lives at a URL that never changes while
#  its content does, so browser caches keep re-serving the first frame they saw.
#  Pinning TIME gives each frame its own URL and the staleness cannot happen.
# ---------------------------------------------------------------------------

# GeoServer per-workspace virtual service. Its capabilities document is a
# fraction of the global one (~85 KB for both workspaces against 282 KB) and
# names layers without the workspace prefix.
WMS_WORKSPACE_URL: Final = "https://view.eumetsat.int/geoserver/{workspace}/ows"

MAP_TIMES_URL: Final = MAP_VIEW_URL + "/times"

# Serve one cadence slot behind the newest advertised frame; the very latest is
# occasionally still incomplete over part of the disc.
MAP_FRAME_LAG_SLOTS: Final = 1

# Frames land every 10 min (MTG) / 15 min (MSG) with ~20-25 min of
# dissemination lag, so re-reading the timeline more often than this buys
# nothing. The shorter retry applies when a capabilities fetch failed.
MAP_TIME_CACHE_TTL: Final = 300.0
MAP_TIME_RETRY_TTL: Final = 60.0

# How often the viewer asks Home Assistant whether a newer frame exists.
MAP_TIME_REFRESH_MS: Final = 300_000
