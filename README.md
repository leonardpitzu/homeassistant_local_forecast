# Local Weather Forecast

A fully local weather forecaster for Home Assistant, tied to your own weather
station.  It reads your sensors, estimates the current weather with physics,
and forecasts 24 hours and three days with models trained on ten years of
weather at 230 stations around the world - then keeps correcting itself
against what your station actually measures.  No cloud APIs, no internet
dependency, no API keys.

One station sees about 100 km, so this is not a weather service: the next
12 hours are good, tomorrow is decent, and the day after is an honest,
calibrated guess.  The numbers are below.

## Features

| Feature | Description |
|---|---|
| 12-state weather model | Maps 1:1 to HA condition strings (sunny, rainy, snowy, snowy-rainy, etc.) |
| Hourly forecast | 24 hours ahead: temperature, humidity, pressure, wind, rain chance and amount, icon |
| Daily forecast | Today (measured so far + forecast), tomorrow and the day after from their own daily models |
| Trained models | Gradient-boosted trees, trained on 230 stations on six continents, pure Python at run time, no extra dependencies |
| Self-correction | Every forecast your station can check is checked; bounded corrections for your site's bias (backyard vs airport) |
| Any set of sensors | Temperature and pressure are enough; humidity, wind, wind direction and rain sharpen it |
| Frontal detection | Warm, cold, occluded - exposed as a single `front` enum sensor |
| Precipitation typing | Wet-bulb temperature: rain vs sleet vs snow |
| Pressure suite | Sea-level pressure, 3-hour tendency with the atmospheric tide removed (+ direction enum), 24-hour synoptic mean, and a tendency-aware barometer enum - all standalone sensors |
| Learned atmospheric tide | The daily pressure waves are learned at your station from its own barometer and thermometer - no location tuning, adapts to seasons and to a move |
| Meteogram feed | Hourly-forecast sensor carrying the full 24h list as an attribute for chart cards - no MQTT publish/discovery workaround needed |
| Kalman sensor smoothing | Per-channel 1-D filter, rejects spikes, seeds from the first reading (correct on restart, no warm-up ramp) |
| Day/night awareness | Swaps sunny/clear-night from `sun.sun` entity |
| Signal smoothing | Rain persistence, cloud hysteresis, post-rain cloud memory - no icon flicker |
| Unit auto-conversion | Accepts hPa/inHg/kPa, C/F, m-s/km-h/mph/knots |
| Pixel display ready | Condition maps directly to ESPHome icon names |
| Satellite map (optional) | Interactive pan/zoom EUMETSAT view, centred on your home - off by default |

---

## Architecture

| Module | Purpose |
|---|---|
| `coordinator.py` | Runs the pipeline: reads and unit-converts the sensors, recorder backfill at startup, estimator + forecaster, persisted pressure buffer and history; publishes one immutable result per run |
| `state_estimator.py` | Sensor fusion, Kalman smoothing, trend computation, frontal detection, weather classification, signal smoothing (rain persistence, cloud hysteresis, post-rain cloud memory) |
| `tide.py` | Atmospheric tide (S1 + S2 + S3) learned at the station; removed from every pressure trend, added back to the forecast |
| `history.py` | Thirty days of the station's own hourly means, persisted; seeded from the recorder's statistics on a fresh install |
| `features.py` | The model inputs - one definition shared with the training harness |
| `model.py` | Loads and evaluates the shipped tree models (`models/*.json.gz`), pure Python |
| `forecast.py` | Evaluates the models once per hour, re-anchors the hours on "now", turns outputs into icons |
| `recal.py` | Bounded self-correction learned from the station's own verified forecasts |
| `weather.py` | HA WeatherEntity - renders the coordinator's result as condition, attributes and hourly / daily forecasts |
| `sensor.py` | Standalone sensor entities - precipitation, forecast condition, sea-level pressure, tendency (+ direction enum), synoptic mean, barometer enum, hourly-forecast meteogram feed, and the `front` enum |
| `pressure_history.py` | Hourly sea-level-pressure ring buffer (24h window), persisted across restarts; feeds the tendency, synoptic-mean, and barometer sensors |
| `classifiers.py` | Pure display-state classifiers - barometer category, pressure-tendency direction, and frontal identity (no HA dependency) |
| `map.py`, `wms_time.py` | Optional satellite viewer and the EUMETView frame-time cache behind it |

```
 Sensors           State Estimator          Hourly history          Forecaster
+-----------+    +-------------------+    +----------------+    +--------------------------+
| pressure  |-+  | Kalman smoother   |    | 30 days of     |    | trained models, once per |
| temp      |-+->| trends, tide      |    | hourly means   |--->| hour: 24 h + 3 days      |
| humidity  |-+  | wet bulb, fronts  |    | (seeded from   |    | self-correction from the |
| wind      |-+  | current condition |    |  the recorder) |    | station's own checks     |
| rain      |-+  +---------+---------+    +----------------+    +------------+-------------+
| solar     |-+            |                     ^                           |
+-----------+              +---------------------+                           |
                           |                                                 |
                 +---------v-------------------------------------------------v---------+
                 | weather.local_weather_forecast: condition, 24 h hourly, 3 daily     |
                 +---------------------------------------------------------------------+
```

`state_estimator.py`, `tide.py`, `history.py`, `features.py`, `model.py`,
`forecast.py`, `recal.py`, `pressure_history.py` and `classifiers.py` are pure
Python with no HA dependencies.

---

## The 12 weather states

The current condition resolves to exactly one of these, and so does every
forecast hour and day.  Each maps 1:1 to the HA condition string that
controls the dashboard icon.

| Index | Constant | HA Condition |
|:---:|---|---|
| 0 | `S_CLEAR` | `sunny` |
| 1 | `S_CLEAR_NIGHT` | `clear-night` |
| 2 | `S_PARTLY_CLOUDY` | `partlycloudy` |
| 3 | `S_CLOUDY` | `cloudy` |
| 4 | `S_FOG` | `fog` |
| 5 | `S_RAINY` | `rainy` |
| 6 | `S_POURING` | `pouring` |
| 7 | `S_SNOWY` | `snowy` |
| 8 | `S_SNOWY_RAINY` | `snowy-rainy` |
| 9 | `S_LIGHTNING_RAINY` | `lightning-rainy` |
| 10 | `S_WINDY` | `windy` |
| 11 | `S_EXCEPTIONAL` | `exceptional` |

---

## Physics

<details>
<summary>Sensor smoothing, pressure trends, atmospheric tide, QNH, wet-bulb, dew point, frontal detection</summary>

### Kalman smoother

Each sensor channel runs an independent 1-D Kalman filter:

$$\hat{x}_k = \hat{x}_{k-1} + K_k (z_k - \hat{x}_{k-1})$$

$$K_k = \frac{P_{k-1} + Q}{P_{k-1} + Q + R}$$

where $Q$ is process noise and $R$ is measurement noise, tuned per
sensor type (pressure: $Q{=}0.005, R{=}0.15$; temperature: $Q{=}0.02, R{=}0.3$;
humidity: $Q{=}0.05, R{=}1.0$; wind: $Q{=}0.1, R{=}0.5$).

Each channel **seeds from its first valid reading** instead of ramping up
from zero, so the first published value is the real measurement - no
multi-minute warm-up after a Home Assistant restart.  Out-of-range inputs
are rejected before they reach the filter, so the seed is never junk.

### Pressure trends

First derivative (tendency) - the least-squares slope over the last hour,
fitted after removing the atmospheric tide (see below):

$$\frac{dp}{dt} = \operatorname{slope}_{1\,\text{h}}\left(p - p_{\text{tide}}\right) \quad [\text{hPa/h}]$$

Second derivative (acceleration - detects approaching fronts), from the two
secant slopes $s_{\text{recent}}$ (1.5 h ago to now) and $s_{\text{old}}$
(3 h ago to 1.5 h ago):

$$\frac{d^2p}{dt^2} \approx \frac{2\,(s_{\text{recent}} - s_{\text{old}})}{3\,\text{h}} \quad [\text{hPa/h}^2]$$

Temperature, humidity and dew-depression trends use the same one-hour fit.

A negative $d^2p/dt^2$ with falling pressure signals an accelerating
low, which feeds the frontal detection.

### Atmospheric tide

Pressure rises and falls every day with no weather behind it, and left in,
those slopes read as a tendency:

| Wave | Cause | Behaviour |
|---|---|---|
| S1 (24 h) | Surface heating | Local and seasonal: 0.15 hPa in December, ~1 hPa in summer at a Brasov station; peaks before sunrise |
| S2 (12 h) | Global solar tide | $\approx 1.16\cos^3\varphi$ hPa, peaks near 10:00 and 22:00 solar time |
| S3 (8 h) | Shape of the day | Long winter nights make the daily cycle non-sinusoidal |

Nothing is tuned to a place - the model learns the station it runs on:

- **S1 follows the thermometer**: it is the station's own daily temperature
  cycle (3-day memory) times a site gain (14-day memory).  Season, day length,
  cloud and climate arrive through the temperature; the gain captures the
  topography (basin, plain, coast).
- **S2 and S3** start from Haurwitz's law and zero, and are refined with the
  same 14-day memory.
- Fits run on 3-hour changes, which weather does not bias toward any hour,
  and shrink toward a world-average prior, so a new install or a move starts
  sensible and adapts on its own.  Time is local solar time from longitude.
- **Whatever shape is left** - a sea breeze, a valley wind, a 6-hour wave -
  is caught in 24 solar-hour bins of the residual against a centred 24-hour
  mean (~20-day memory).  They start at zero, so a new install begins as the
  physics alone; a day with a hole in it, or a front through the window, is
  skipped.

Scored hour by hour (each hour is predicted before the model may learn from
it) at ten sites on four continents - Atlantic and Black Sea coast, plain,
two alpine valleys, high plains, desert, the southern hemisphere; NOAA ISD
2019-2023 - and on 6.2 years from three Brasov stations, the daily cycle left
in the 3-hour tendency, averaged per calendar month:

| Tide model | Leak, mean of 10 sites (hPa/h) | Brasov (hPa/h) |
|---|---|---|
| None | - | 0.183 |
| Fixed climatology | 0.215 | 0.104 |
| Learned, temperature-driven S1 + S2 + S3 | 0.049 | 0.029 |
| **+ residual bins** | **0.033** | **0.020** |

The bins improve every one of the ten sites.  A fixed climatology is worst
where it matters most: in the alpine valleys it leaves 0.43 hPa/h, the
learned model 0.04.

The same fit also learns much of a barometer's own thermal fault: over the
spring one of those stations' sensors was failing with a fake 3 hPa daily
swing, it cut the daily cycle left in the tendency from 0.68 to 0.15 hPa/h
(those months are excluded from the table above).

The tide is removed from every trend (1-hour slope, acceleration, 3-hour
tendency) and added back to the forecast pressure.  The pressure shown on
the dashboard keeps it - that is what a barometer reads.  The learned state
is persisted with the pressure buffer and is visible in the integration's
diagnostics download.

### QFE to QNH conversion

If your station reports absolute pressure (QFE), the integration
converts it to QNH through the ICAO standard atmosphere, exactly as airports do:

$$P_{\text{QNH}} = P_{\text{QFE}} \left(1 - \frac{L \cdot h}{288.15}\right)^{-5.257}$$

where $L = 0.0065$ K/m (ISA lapse rate) and $h$ is station elevation in metres.
The live temperature is deliberately left out: at 544 m it would add
0.24 hPa per degree, a fake ~3 hPa daily swing.  Against the nearest airport's
QNH the standard reduction agrees to 0.28 hPa (its rounding); the
live-temperature one scattered by 1.3 hPa.

### Wet-bulb temperature

Determines precipitation type using the Stull (2011) approximation:

$$T_w = T \cdot \arctan(0.151977\sqrt{RH + 8.313659}) + \arctan(T + RH) - \arctan(RH - 1.676331) + 0.00391838 \cdot RH^{3/2} \cdot \arctan(0.023101 \cdot RH) - 4.686035$$

| Wet-bulb range | Precipitation type |
|---|---|
| $T_w < -2$ C | Snow |
| $-2 \leq T_w < 1$ C | Sleet / mixed |
| $T_w \geq 1$ C | Rain |

### Dew point (Magnus formula)

$$T_d = \frac{243.04 \cdot \ln\!\left(\frac{RH}{100}\right) + \frac{17.625 \cdot T}{243.04 + T}}{17.625 - \ln\!\left(\frac{RH}{100}\right) - \frac{17.625 \cdot T}{243.04 + T}}$$

Fog is classified when dew-point depression $T - T_d < 1.5$ C and wind speed
$< 3$ m/s.  On its own that rule calls fog far too often: at the training
airports it says fog 8 % of the time against 1.5 % observed (wet overcast is
saturated and calm too).  So when the fog model gives the next hour under 5 %,
the call is vetoed: 60 % of the false fog goes, three in four real fogs stay.
A stricter veto removes more false fog but starts hiding real fog; one real fog
in four is the price kept.

### Frontal detection

| Front | Pressure | Wind | Temperature | Humidity |
|---|---|---|---|---|
| **Warm** | Falling $> 1$ hPa/h | Backing (CCW shift $< -10$ deg/h) | Rising | Rising $> 2$ %/h |
| **Cold** | Trough (accel $> 0.5$) | Veering (CW shift $> 15$ deg/h) | Dropping $> 1$ C/h | -- |
| **Occluded** | Falling $> 2$ hPa/h | Large shift ($\lvert\Delta\rvert > 20$ deg/h) | -- | Dew depression $< 2$ C |

Wind shift is the rotation rate of the mean wind vector, fitted by least
squares over the last 2 hours and weighted by wind speed, so calm,
directionless samples carry no weight - positive = veering (clockwise),
negative = backing.  A wind with no prevailing direction reports no shift.

### Signal smoothing

Three mechanisms prevent icon flicker on the dashboard:

- **Rain persistence** - rain state is held for 20 minutes after rain rate drops below the light-rain threshold.  Prevents the icon from flashing between rainy and cloudy during intermittent showers.
- **Cloud hysteresis** - a $\pm 0.06$ deadband on cloud-fraction thresholds.  If the current state is cloudy, the cloud fraction must drop below the threshold minus 0.06 to transition to partly-cloudy (and vice versa).  Eliminates rapid toggling at decision boundaries.
- **Post-rain cloud memory** - after rain ends, a cloud-fraction floor of 0.40 is applied, linearly decaying to zero over 30 minutes.  Real skies stay overcast after rain stops; this prevents an immediate jump to sunny.

</details>

---

## Trained forecaster

<details>
<summary>What the models are, what they were trained on, and how good they are</summary>

### What forecasts

Sixteen gradient-boosted tree ensembles (scikit-learn `HistGradientBoosting`,
tuned per family: up to 600 trees of up to 63 leaves), one per quantity:

| Hourly, 1-24 h | Daily: rest of today, tomorrow, the day after |
|---|---|
| temperature, dew point (humidity), pressure, wind speed | high, low |
| chance of rain (>= 0.2 mm in the hour), mm if wet | chance of rain (>= 0.2 mm in the window), mm if wet |
| cloud cover, fog chance, thunder chance | cloud cover, mean wind, thunder chance |

They read about sixty inputs computed from the station's own last 30 days of
hourly means (`features.py`): the current values, 1-24 h changes, 24 h and
7-day extremes and means, the pressure anomaly against the last month, rain in
the last day and hours since rain, sunshine, and the sun itself - local solar
time, sun elevation, day length and season (mirrored south of the equator).
Pressure is QNH with the station's learned atmospheric tide removed, and the
tide is added back to the forecast pressure of each hour.

Sunshine enters as the clear-sky index - measured radiation over what a
cloudless sky gives at that sun height - divided by the station's own
clear-sky reference (its 95th percentile over the last 30 days), so a sensor
that reads 25 % low, or a garden shaded in the morning, gives the same inputs
as a calibrated pyranometer.

The daily card's "today" joins what the station measured since midnight with
the rest-of-today model; tomorrow and the day after come from their own daily
models.  Every day is reconciled with the hourly temperatures that fall on it,
so the card never shows a high below one of its own hours.

The models run in pure Python inside Home Assistant (`models/*.bin.gz`,
`model.py`, trees held in typed arrays): once per hour, in an executor, about
0.5 s on a Raspberry Pi.

### Training data

| Source | What it gives |
|---|---|
| Iowa Environmental Mesonet METAR archive: 230 airports on six continents, sea level to 2.5 km, 2016-2025 | temperature, dew point, wind, QNH, cloud layers, observed fog and thunder; US rain gauges |
| NOAA GHCNh (public domain) | hourly rain-gauge totals for the airports outside the US |
| Deutscher Wetterdienst open data (CC BY 4.0), 50 stations | all of the above at one site, total cloud, a rain gauge, and global radiation at 25 of them |

Rain is a gauge everywhere: a wet hour, or a wet day, is at least 0.2 mm
measured - the same thing a Home Assistant rain sensor reports.  Airports
without a gauge count as stations without a rain sensor.  The author's own
station (12 years) joins only the final training run and never a scored one.

Any combination of sensors works: during training whole sensors (humidity,
wind, wind direction, rain, solar) were hidden at random, so a missing sensor
is a normal input, and a fresh install with one hour of history falls back on
place and season.

### How good it is

Scored on stations the models never saw, and never had a neighbour of in
training: three-fold cross-validation by region (whole 8 x 8 degree cells),
with the hyper-parameters tuned inside each fold.  Against what a single
station can do without a forecast ("persistence": nothing changes) and against
the 1.6.0 Bayesian forecaster replayed on the same hours:

| | 1-3 h | 4-6 h | 7-12 h | 13-24 h |
|---|---|---|---|---|
| Temperature, mean error | 0.9 C | 1.4 C | 1.7 C | 2.0 C |
| (persistence) | 1.4 C | 3.1 C | 4.5 C | 3.8 C |
| Rain (>= 0.2 mm), AUC | 0.91 | 0.85 | 0.81 | 0.75 |
| Rain, skill vs the station's climatology | +0.30 | +0.16 | +0.10 | +0.05 |

The rain chance beats climatology over the next six hours at every one of the
131 gauged stations (the worst still +0.04).

| | Rest of today | Tomorrow | Day after |
|---|---|---|---|
| High, mean error (persistence) | 1.3 C (2.9) | 2.3 C (2.8) | 2.8 C (3.4) |
| Low, mean error (persistence) | 1.3 C (2.6) | 2.0 C (2.5) | 2.5 C (3.1) |
| Rain (>= 0.2 mm), AUC | 0.87 | 0.77 | 0.70 |

Against 1.6.0 on identical hours: temperature 1-12 h 1.45 C vs 2.38 C; rain
chance averaging 6 % against 6 % observed (1.6.0: 22 %), Brier score 0.049 vs
0.092; tomorrow's high 2.3 C vs 5.2 C, low 2.0 C vs 3.9 C; rain day AUC 0.79
vs 0.61.

Sunshine, where a station measures it, improves cloud cover by 5-10 %,
tonight's low by 3 % and temperature by up to 1 %.  Fog and thunder are rare
and cannot be placed in the right hour from one station, so their icons show
less often than they happen; a thunder day is predictable (AUC 0.84) and shows
on the daily forecast.

### At one backyard station, twelve years

The final exam was the author's own station (Brasov, Romania, a 544 m basin):
12.8 years of hourly records, 2014-2026, five sensor generations - scored with
models trained WITHOUT it, every hour of the record as a forecast, the
self-correction replayed exactly as it runs in Home Assistant and restarted
every January 1st (a new install each year):

| | 1-3 h | 4-6 h | 7-12 h | 13-24 h |
|---|---|---|---|---|
| Temperature (persistence) | 1.2 C (1.9) | 1.8 C (4.1) | 2.1 C (5.9) | 2.4 C (4.8) |
| Dew point (persistence) | 0.9 C (0.9) | 1.3 C (1.7) | 1.6 C (2.2) | 2.0 C (2.4) |
| Wind (persistence) | 0.4 m/s (0.4) | 0.4 m/s (0.5) | 0.5 m/s (0.6) | 0.5 m/s (0.6) |
| Rain chance, skill vs climatology | +0.23 | +0.12 | +0.08 | +0.04 |

| | Rest of today | Tomorrow | Day after |
|---|---|---|---|
| High (persistence / climatology) | 2.3 C (5.6 / -) | 3.7 C (4.4 / 5.1) | 4.3 C (5.1 / 5.1) |
| Low (persistence / climatology) | 1.3 C (3.5 / -) | 1.8 C (2.3 / 2.9) | 2.2 C (2.8 / 2.9) |
| Rain day, skill vs climatology | | +0.10 | +0.005 |

This archive has no rain gauge - only 30-minute snapshots of a rain sensor -
so its rain is a different measure from the gauges the models learned; the
rain figures above are after the self-correction has adapted to it, which is
exactly what it does with a real sensor in Home Assistant.  Two days out, rain
is better than luck, but only just: one station sees about 100 km, and
weather two days away is still 500 km upwind.

Against 1.6.0 on the same hours: temperature +1 h 0.9 C vs 1.3 C, +12 h
2.2 C vs 3.8 C; rain chance 1-12 h skill +0.12 vs -0.75 with an average of
5 % against 5 % observed (1.6.0: 20 %); tomorrow's high 3.7 C vs 7.5 C.

### Self-correction

The models learned airports and weather services.  A backyard station sits
lower or higher, in a hollow or on a roof, in the sun at noon or in a cold
pocket at night, and its anemometer is not on a 10 m mast.  So every forecast
the station can verify is verified once its hour (or day) has been measured,
and a small correction is learned per forecast range (`recal.py`): a bias for
temperature and dew point per three hours of the day, a scale for wind, a
shift in log-odds for the rain chance (only with a rain sensor), and the same
for the daily high, low and rain day.  Every correction is bounded, so a
broken sensor cannot teach the forecast nonsense, and starts neutral.  It
starts over when the models change (an update that retrains them): a
correction learned for other models does not fit these.  The state, with the
fingerprint of the models it belongs to, is in the diagnostics download.

The learning rates were chosen on 24 held-out stations on four continents plus
the backyard one, replaying each as a fresh install every year: they help on
average and cost no station more than ~0.7 %.  Mean error change: temperature
-5 to -3 % (backyard -10 to -4 %), dew point -3 %, wind -3 to -4 % (backyard
-80 %: its anemometer sees 40 % of an airport's wind), rain chance -1 %
(backyard -3 to -4 %).

### History

The hourly history behind the inputs is kept for 30 days and persisted across
restarts.  On a new install it is seeded from the recorder's long-term
statistics (sensors with a `state_class`), so the forecast starts with weeks
of history instead of none.  Every restart also fills, per hour, the channels
the stored history lacks - a rain or solar sensor added later, or an upgrade -
and never overwrites a live reading.  Today's high and low on the daily card
are the highest and lowest readings since midnight (the recorder's hourly
max/min for seeded hours), not hourly means.  If the newest hour has no
readings yet, the current values stand in for it: there is always a forecast
to show.

### Retraining

The training harness is not part of this repository.  It downloads the
station archive, builds the inputs with a vectorised copy of `features.py`
(a parity test holds the two together), cross-validates, scores and exports
`models/`.

</details>

---

## Entity attributes

<details>
<summary>Extra state attributes exposed by the weather entity</summary>

| Attribute | Type | Description |
|---|---|---|
| `pressure_trend` | float | $dp/dt$ in hPa/h |
| `pressure_acceleration` | float | $d^2p/dt^2$ in hPa/h squared |
| `dew_point` | float | Degrees C (only with a humidity sensor) |
| `dew_depression` | float | $T - T_d$ in C (only with a humidity sensor) |
| `wet_bulb` | float | Degrees C (only with a humidity sensor) |
| `wind_force` | int | Beaufort number (0-12) (only with a wind-speed sensor) |
| `wind_force_description` | str | Beaufort name, e.g. "Gentle breeze" (only with a wind-speed sensor) |
| `front_warm` | bool | Warm front detected (also surfaced as the `front` enum sensor) |
| `front_cold` | bool | Cold front detected (also surfaced as the `front` enum sensor) |
| `front_occluded` | bool | Occluded front detected (also surfaced as the `front` enum sensor) |
| `next_hour_condition` | str | HA condition for hour +1 |
| `next_hour_precip_probability` | int | 0-100 for hour +1 |
| `precip_probability_6h` | int | Max hourly rain probability over next 6h |

Attributes derived from an optional sensor are omitted entirely when that
sensor is not configured, rather than published from a placeholder value.

Standard weather entity properties are also available:
`temperature`, `apparent_temperature`, `dew_point`, `humidity`,
`pressure`, `wind_speed`, `wind_bearing`, hourly forecast (24h),
daily forecast (today, tomorrow and the day after).  Forecast `humidity`,
`wind_speed` and `wind_bearing` are `null` when the matching sensor is not
configured.

</details>

---

## Sensor entities

<details>
<summary>Standalone sensors with history for tile cards, graphs, badges, and automations</summary>

Key forecast values are also exposed as standalone sensor entities.  Unlike
weather entity attributes, sensors have their own history in the HA recorder
and can be graphed in tile cards, used in badges, and referenced in automations.

All entity IDs are prefixed `sensor.local_weather_forecast_`.

| Entity (`sensor.local_weather_forecast_…`) | Type | Description |
|---|---|---|
| `precipitation_probability` | % | Max rain probability over the next 6 hours |
| `1h_forecast` | text | Forecast condition for +1h (human-readable) |
| `next_hour_precipitation_probability` | % | Precipitation probability for +1h |
| `sea_level_pressure` | hPa | Sea-level (QNH) pressure - the value feeding the weather entity |
| `pressure_tendency` | hPa/h | 3-hour pressure tendency with the atmospheric tide removed (for charts/automations) |
| `pressure_tendency_direction` | enum | `falling_fast` / `falling` / `steady` / `rising` / `rising_fast`, with arrow icons |
| `pressure_synoptic` | hPa | 24-hour rolling mean of sea-level pressure |
| `barometer` | enum | Tendency-aware needle: `stormy` / `rain` / `change` / `fair` / `very_dry` |
| `hourly_forecast` | timestamp | Generation time; the `forecast` attribute carries the full 24h list for meteogram cards |
| `front` | enum | Frontal passage: `none` / `warm` / `cold` / `occluded` |

The `barometer`, `pressure_tendency_direction`, and `front` sensors are
`device_class: enum` with per-state icons (from `icons.json`), so a single
badge shows the right symbol with no template logic.

The tendency differs from a raw WMO reading on purpose: the atmospheric tide
(see Physics) swings pressure by up to ~1 hPa within 3 hours with no weather
behind it, enough on its own to cross the 0.3 hPa/h `steady` band.  The
`sea_level_pressure` value itself keeps the tide - that is what the
barometer reads.

The pressure ring buffer behind `pressure_tendency` and `pressure_synoptic`
is **persisted across restarts** (in Home Assistant's `.storage`), so the 3-hour tendency
and 24-hour mean survive a reboot instead of warming up for hours.

These entities are designed to **retire common dashboard workarounds**: the
pressure sensors replace statistics/derivative helpers, and `hourly_forecast`
replaces the usual `weather.get_forecasts` -> MQTT publish -> discovery-sensor
chain used to feed meteogram cards.

All sensors update automatically when the weather entity recalculates.
The `%` and `hPa` sensors have `state_class: measurement` so HA records
long-term statistics - min/max/mean in the history panel and trend graphs
in tile cards.

</details>

---

## Satellite map (optional)

<details>
<summary>Interactive pan/zoom satellite view, centred live on your home location</summary>

An optional, self-contained pan/zoom satellite viewer for glancing at what is
actually heading your way - handy when a weather alert fires and you want to
see the incoming system. It is **off by default**; enable it under
**Settings -> Devices & Services -> Local Weather Forecast -> Configure** by
ticking **Enable satellite map**. When disabled, the endpoint is not served at
all.

When enabled, the integration serves a Leaflet viewer at
`/api/local_forecast/map`. Your browser tiles EUMETSAT's public
[EUMETView](https://view.eumetsat.int/) WMS endpoint directly - there are **no
extra entities, no polling, and no API key**. Leaflet itself is bundled with
the integration, so nothing is fetched from a third-party CDN. The view offers
a layer switcher for Geo Colour, Natural Colour, Dust, Airmass, Convection,
Volcanic Ash, and Fog / Low Clouds.

The endpoint is reachable without authentication, because an `iframe` card
cannot present a token. It therefore centres on a coarse grid near your home
location rather than the exact coordinates, so the page cannot be used to
locate your house. A red ring marks that area: it is drawn wide enough to
contain your home wherever the grid moved the centre to, and it sits above
every layer, so you keep a reference point when the ground below is completely
clouded over.

Add it to a dashboard with an `iframe` card:

```yaml
type: iframe
url: /api/local_forecast/map
aspect_ratio: 75%
```

</details>

---

## Installation and configuration

<details>
<summary>HACS, manual install, config flow fields, sensor recommendations</summary>

### Installation

**HACS (recommended)**

1. Add this repository as a custom repository in HACS
2. Search for "Local Weather Forecast", install
3. Restart Home Assistant

**Manual**

Copy `custom_components/local_weather_forecast/` into your HA `config/custom_components/` directory and restart.

### Configuration

Settings, Devices and Services, Add Integration, Local Weather Forecast.

| Field | Required | Description |
|---|---|---|
| Pressure sensor | yes | Any pressure entity (hPa, inHg, kPa - auto-converted) |
| Temperature sensor | yes | Outdoor temperature (C or F - auto-converted) |
| Humidity sensor | | Outdoor relative humidity (%) |
| Wind speed sensor | | m/s, km/h, mph, or knots - auto-converted |
| Wind direction sensor | | Degrees (0-360) |
| Solar radiation sensor | | W/m2 - improves cloud estimation |
| Rain rate sensor | | mm/h - enables live precipitation detection |
| Elevation | | Station elevation in metres (for QFE to QNH conversion) |
| Pressure type | | Absolute (QFE) or Sea-level (QNH) |
| Enable satellite map | | Serves an interactive pan/zoom satellite view at `/api/local_forecast/map` (off by default) |

Minimum viable setup: pressure + temperature.  Everything else improves accuracy.

Recommended setup: pressure + temperature + humidity + wind speed + wind direction.

### Sensor recommendations

| Sensor | Why it matters |
|---|---|
| **Pressure** | The single most important weather predictor: tendencies, the month's anomaly, frontal detection.  Use a dedicated barometric sensor (BME280/BMP388), not a phone app.  Required. |
| **Temperature** | The models' anchor, precipitation type (rain vs snow), feels-like temperature.  Required. |
| **Humidity** | Dew point and its trend are strong rain and fog inputs; enables the humidity forecast and wet-bulb snow/rain typing. |
| **Wind speed** | Forecast input and the wind forecast itself; the self-correction learns how sheltered your anemometer is. |
| **Wind direction** | Flow direction as a forecast input; frontal passage detection (backing/veering). |
| **Solar radiation** | Current cloud estimate for the present condition.  Not a forecast input. |
| **Rain rate** | Hours since rain is a strong input, and only with it can the rain chance calibrate itself to your station. |

</details>

---

## Dashboard cards

<details>
<summary>Native cards, tile graphs, badges, Mushroom layout, pressure trends and frontal chips</summary>

### Minimal setup (no extra cards needed)

The weather entity works with native HA cards directly.  No
conditional cards, no template icons - the entity already handles
precipitation type, day/night, and all 12 conditions internally.

```yaml
type: vertical-stack
cards:
  - type: weather-forecast
    entity: weather.local_weather_forecast
    show_current: true
    show_forecast: true
    forecast_type: hourly
    forecast_slots: 6

  - type: weather-forecast
    entity: weather.local_weather_forecast
    show_current: false
    show_forecast: true
    forecast_type: daily
```

### Tile card with precipitation trend graph

The 6h precipitation probability sensor has history, so the tile card
can show a trend graph natively - no extra integrations needed:

```yaml
type: tile
entity: sensor.local_weather_forecast_precipitation_probability
features:
  - type: graph
    days_to_show: 1
```

### Badge with next-hour forecast

Native HA badges appear at the top of a dashboard view:

```yaml
badges:
  - entity: sensor.local_weather_forecast_1h_forecast
  - entity: sensor.local_weather_forecast_precipitation_probability
  - entity: sensor.local_weather_forecast_barometer
  - entity: sensor.local_weather_forecast_pressure_tendency_direction
  - entity: sensor.local_weather_forecast_front
```

### Adding pressure trend and frontal alerts

For the extra attributes (pressure trend, frontal detection, 6h rain probability),
add a markdown card below the weather cards:

```yaml
  - type: markdown
    content: >-
      **Pressure:** {{ state_attr('weather.local_weather_forecast', 'pressure') | round(1) }} hPa
      ({{ state_attr('weather.local_weather_forecast', 'pressure_trend') | round(1) }} hPa/h)
      {% if state_attr('weather.local_weather_forecast', 'front_warm') %} | Warm front{% endif %}
      {% if state_attr('weather.local_weather_forecast', 'front_cold') %} | Cold front{% endif %}
      {% if state_attr('weather.local_weather_forecast', 'front_occluded') %} | Occluded front{% endif %}
      | Wet-bulb: {{ state_attr('weather.local_weather_forecast', 'wet_bulb') }}°C
      | 6h rain: {{ state_attr('weather.local_weather_forecast', 'precip_probability_6h') | default(0) }}%
```

### Standard weather card

Works as-is:

```yaml
type: weather-forecast
entity: weather.local_weather_forecast
forecast_type: hourly
```

### Mushroom cards

Requires [Mushroom Cards](https://github.com/piitaya/lovelace-mushroom)
and [Vertical Stack In Card](https://github.com/ofekashery/vertical-stack-in-card) via HACS.

```yaml
type: custom:vertical-stack-in-card
cards:
  - type: custom:mushroom-title-card
    title: Local Weather Forecast
    subtitle: >-
      {% set c = states('weather.local_weather_forecast') %}
      {% set labels = {'partlycloudy':'Partly Cloudy','clear-night':'Clear Night','snowy-rainy':'Snowy Rainy','lightning-rainy':'Lightning Rainy'} %}
      {{ labels.get(c, c | replace('-', ' ') | title) }}

  - type: horizontal-stack
    cards:
      - type: custom:mushroom-template-card
        primary: Now
        secondary: >-
          {{ state_attr('weather.local_weather_forecast', 'temperature') | round }}°C
        icon: >-
          {% set c = states('weather.local_weather_forecast') %}
          {% set m = {
            'sunny': 'mdi:weather-sunny',
            'clear-night': 'mdi:weather-night',
            'partlycloudy': 'mdi:weather-partly-cloudy',
            'cloudy': 'mdi:weather-cloudy',
            'fog': 'mdi:weather-fog',
            'rainy': 'mdi:weather-rainy',
            'pouring': 'mdi:weather-pouring',
            'snowy': 'mdi:weather-snowy',
            'snowy-rainy': 'mdi:weather-snowy-rainy',
            'lightning-rainy': 'mdi:weather-lightning-rainy',
            'windy': 'mdi:weather-windy',
            'exceptional': 'mdi:alert',
          } %}
          {{ m.get(c, 'mdi:weather-partly-cloudy') }}
        icon_color: >-
          {% set pp = state_attr('weather.local_weather_forecast', 'next_hour_precip_probability') | int(0) %}
          {% if pp > 70 %}red{% elif pp > 40 %}orange{% elif pp > 20 %}yellow{% else %}blue{% endif %}
        layout: vertical

      - type: custom:mushroom-template-card
        primary: +1h
        secondary: >-
          {% set c = state_attr('weather.local_weather_forecast', 'next_hour_condition') %}
          {% set labels = {'partlycloudy':'Partly Cloudy','clear-night':'Clear Night','snowy-rainy':'Snowy Rainy','lightning-rainy':'Lightning Rainy'} %}
          {{ labels.get(c, c | replace('-', ' ') | title) }}
          {{ state_attr('weather.local_weather_forecast', 'next_hour_precip_probability') }}% precip
        icon: mdi:clock-outline
        icon_color: orange
        layout: vertical

  - type: horizontal-stack
    cards:
      - type: custom:mushroom-template-card
        primary: Pressure
        secondary: >-
          {{ state_attr('weather.local_weather_forecast', 'pressure') | round(1) }} hPa
        icon: >-
          {% set dp = state_attr('weather.local_weather_forecast', 'pressure_trend') | float(0) %}
          {% if dp > 0.5 %}mdi:trending-up{% elif dp < -0.5 %}mdi:trending-down{% else %}mdi:trending-neutral{% endif %}
        icon_color: >-
          {% set dp = state_attr('weather.local_weather_forecast', 'pressure_trend') | float(0) %}
          {% if dp < -1.5 %}red{% elif dp < -0.5 %}orange{% elif dp > 0.5 %}green{% else %}blue{% endif %}
        layout: vertical

      - type: custom:mushroom-template-card
        primary: Humidity
        secondary: >-
          {{ state_attr('weather.local_weather_forecast', 'humidity') }}%
          (Dew {{ state_attr('weather.local_weather_forecast', 'dew_point') }}°C)
        icon: mdi:water-percent
        icon_color: teal
        layout: vertical

  - type: custom:mushroom-chips-card
    chips:
      - type: template
        content: >-
          {% if state_attr('weather.local_weather_forecast', 'front_warm') %}Warm front{% elif state_attr('weather.local_weather_forecast', 'front_cold') %}Cold front{% elif state_attr('weather.local_weather_forecast', 'front_occluded') %}Occluded{% else %}No front{% endif %}
      - type: template
        content: >-
          6h rain: {{ state_attr('weather.local_weather_forecast', 'precip_probability_6h') | default(0) }}%
      - type: template
        content: >-
          Feels {{ state_attr('weather.local_weather_forecast', 'apparent_temperature') | round }}°C
```

</details>

---

## Pixel display

<details>
<summary>ESPHome pixel display automation with icon mapping</summary>

The weather entity condition maps directly to ESPHome pixel icon names.
The integration already handles day/night (sunny vs clear-night), precipitation
type (rain/snow/sleet via wet-bulb temperature), and storms - so the template
is a simple one-to-one map with night overrides for conditions that have
dedicated night icons in the firmware (`weather_cloudy_night`, `weather_rainy_night`).

**Important:** The Jinja template uses `{%- -%}` whitespace control on every
tag.  Without it, the `{% if %}` / `{% set %}` blocks inject leading spaces
and newlines into the rendered icon name, and EspHoMaTriXv2 cannot match it.

```yaml
alias: pixel forecast
description: Weather icon and 1h forecast text for ESPHome pixel display
triggers:
  - entity_id: binary_sensor.pixel_status
    to: "on"
    trigger: state
  - entity_id: weather.local_weather_forecast
    trigger: state
  - minutes: /10
    trigger: time_pattern
actions:
  - action: esphome.pixel_icon_screen
    data:
      default_font: true
      icon_name: >-
        {%- set cond = states('weather.local_weather_forecast') -%}
        {%- set is_night = is_state('sun.sun', 'below_horizon') -%}
        {%- set map = {
          'sunny':           'weather_sunny',
          'clear-night':     'weather_clear_night',
          'partlycloudy':    'weather_partly_cloudy',
          'cloudy':          'weather_cloudy',
          'fog':             'weather_fog',
          'rainy':           'weather_rainy',
          'pouring':         'weather_pouring',
          'snowy':           'weather_snowy',
          'snowy-rainy':     'weather_snowy_rainy',
          'lightning-rainy': 'weather_lightning_rainy',
          'windy':           'weather_windy',
          'exceptional':     'weather_cloudy',
        } -%}
        {%- set night = {
          'partlycloudy': 'weather_cloudy_night',
          'rainy':        'weather_rainy_night',
        } -%}
        {%- set icon = night.get(cond, map.get(cond, 'weather_cloudy'))
            if is_night else map.get(cond, 'weather_cloudy') -%}
        {{ icon }}|forecast
      text: "{{ states('sensor.local_weather_forecast_1h_forecast') }}"
      lifetime: 1440
      screen_time: 5
      r: 200
      g: 200
      b: 200
mode: single
```

The `|forecast` suffix is the screen name on the pixel device.
Adjust `lifetime`, `screen_time`, and RGB values to taste.

</details>

---

## Troubleshooting

<details>
<summary>Common issues and fixes</summary>

**Icons show wrong condition**

- Check that your sensors are reporting correct values in Developer Tools, States
- Verify pressure type setting matches your sensor (QFE vs QNH)
- If precipitation type is wrong, check wet-bulb in attributes - should be below -2 C for snow

**Forecast does not update**

- Entity updates on sensor state changes (throttled to 30s, and never delayed
  more than 120s), plus a refresh every 5 minutes
- Check that sensor entities are not `unavailable` or `unknown`
- The weather entity stays `unavailable` until at least one valid reading has
  been ingested - it will not publish placeholder values

**Temperature seems off**

- Check `unit_of_measurement` on your sensor - the integration auto-converts F, but the attribute must be set correctly
- Elevation affects QFE-to-QNH which affects frontal detection - verify it is correct

**Exceptional shows too often**

- This state triggers on a very rapid pressure fall (steeper than about 4 hPa/h - a bomb-cyclone signature)
- If your elevation or pressure type is wrong, the QNH conversion can produce extreme trends that trip it

**Sensor values rejected**

- Pressure is validated *after* reduction to sea level, so high-altitude
  stations work: the station reading may be well below 870 hPa as long as the
  resulting QNH lands in 870-1090 hPa
- Temperature outside -60 to 60 C after conversion is rejected
- Humidity is clamped to 0-100%, wind speed to 0-60 m/s
- Check `unit_of_measurement` attribute on your sensor entities - unit
  conversion is handled by Home Assistant's own converters and relies on it

</details>

---

## Debugging

If you're having issues with the integration, there are two ways to enable debug logging.

### Option 1 - YAML configuration

Add the following to your `configuration.yaml` and restart Home Assistant:

```yaml
logger:
  default: info
  logs:
    custom_components.local_forecast: debug
```

### Option 2 - Home Assistant UI

1. Go to **Settings** -> **Devices & Services**.
2. Find the **Local Weather Forecast** integration and click the **⋮** menu.
3. Select **Enable debug logging**.
4. Reproduce the issue.
5. Click **Disable debug logging** - the browser will download a log file you can inspect or attach to a bug report.

Debug output includes sensor values after unit conversion, current weather state classification, and forecast summary for the next hour.

### Diagnostics

**Settings** -> **Devices & Services** -> **Local Weather Forecast** -> **⋮** ->
**Download diagnostics** returns what the atmospheric tide has learned: the
S1 / S2 / S3 amplitudes and peak solar hours, the station's daily temperature
cycle, the site gain and lag, the residual by solar hour, and how many days of
evidence it holds.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
