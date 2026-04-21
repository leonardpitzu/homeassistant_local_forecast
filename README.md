# Local Weather Forecast

A physics-based, fully local weather forecaster for Home Assistant.  Reads
your own sensors, applies Bayesian state estimation and atmospheric physics,
and produces a 12-hour probabilistic forecast with correct dashboard icons.
No cloud APIs, no internet dependency, no API keys.

Snow icons when it snows.  Rain when it rains.  Mixed when it's mixed.
Seasons, time of day, and conditions all matter.

## Features

| Feature | Description |
|---|---|
| 12-state weather model | Maps 1:1 to HA condition strings (sunny, rainy, snowy, snowy-rainy, etc.) |
| Hourly forecast | 12 hours ahead, probabilistic, with precipitation type |
| Daily forecast | Today + tomorrow + day after tomorrow aggregated from hourly |
| Frontal detection | Warm, cold, occluded — from pressure acceleration and wind shift |
| Precipitation typing | Wet-bulb temperature: rain vs sleet vs snow |
| Energy-balance temperature | Diurnal cycle, radiative cooling, thermal inertia |
| Clausius-Clapeyron humidity | Conservation of mixing ratio as temperature changes |
| Kalman sensor smoothing | Per-channel 1-D filter, rejects spikes |
| Day/night awareness | Swaps sunny/clear-night from `sun.sun` entity |
| Unit auto-conversion | Accepts hPa/inHg/kPa, C/F, m-s/km-h/mph/knots |
| Pixel display ready | Condition maps directly to ESPHome icon names |

---

## Architecture

| Module | Purpose |
|---|---|
| `state_estimator.py` | Sensor fusion, Kalman smoothing, trend computation, frontal detection, weather classification |
| `bayesian_forecaster.py` | Markov transition matrix + Bayesian evidence updates, hourly probability vectors |
| `physics_models.py` | Energy-balance temperature, Clausius-Clapeyron humidity, damped pressure extrapolation |
| `weather.py` | HA WeatherEntity — reads sensors, runs pipeline, serves forecasts and attributes |
| `sensor.py` | Standalone sensor entities (precipitation probability, 1h forecast) with history for tile cards and badges |

```
 Sensors                 State Estimator              Bayesian Forecaster
+-----------+          +-------------------+        +----------------------+
| pressure  |--+       | Kalman smoother   |        | 12x12 Markov matrix  |
| temp      |--+       | dp/dt, d2p/dt2    |        | Evidence updates     |
| humidity  |--+-----> | dT/dt, dH/dt      |------> | Physical constraints |
| wind      |--+       | Wet-bulb calc     |        | Physics models       |
| rain      |--+       | Frontal detection |        | Day/night swap       |
| solar     |--+       | State classify    |        |                      |
+-----------+          +-------------------+        +----------+-----------+
                                                               |
                                                    +----------v-----------+
                                                    | weather.local_forecast|
                                                    |                      |
                                                    | condition + icon     |
                                                    | 12h hourly forecast  |
                                                    | daily forecast       |
                                                    | rich attributes      |
                                                    +----------------------+
```

All modules except `weather.py` are pure Python with no HA dependencies.

---

## The 12 weather states

Every forecast hour resolves to exactly one of these.  The index drives
the 12x12 Markov transition matrix and maps 1:1 to the HA condition
string that controls the dashboard icon.

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
<summary>Sensor smoothing, pressure trends, wet-bulb, dew point, frontal detection, temperature and humidity models</summary>

### Kalman smoother

Each sensor channel runs an independent 1-D Kalman filter:

$$\hat{x}_k = \hat{x}_{k-1} + K_k (z_k - \hat{x}_{k-1})$$

$$K_k = \frac{P_{k-1} + Q}{P_{k-1} + Q + R}$$

where $Q$ is process noise and $R$ is measurement noise, tuned per
sensor type (pressure: $Q{=}0.01, R{=}0.5$; temperature: $Q{=}0.05, R{=}0.3$).

### Pressure trends

First derivative (tendency):

$$\frac{dp}{dt} = \frac{p_{\text{now}} - p_{3\text{h ago}}}{3} \quad [\text{hPa/h}]$$

Second derivative (acceleration — detects approaching fronts):

$$\frac{d^2p}{dt^2} = \frac{(dp/dt)_{\text{now}} - (dp/dt)_{1\text{h ago}}}{1} \quad [\text{hPa/h}^2]$$

A negative $d^2p/dt^2$ with falling pressure signals an accelerating
low — the Bayesian layer increases weight on precipitation states.

### QFE to QNH conversion

If your station reports absolute pressure (QFE), the integration
converts to sea-level equivalent (QNH) using the barometric formula:

$$P_{\text{QNH}} = P_{\text{QFE}} \left(1 - \frac{L \cdot h}{T + 273.15}\right)^{-5.257}$$

where $L = 0.0065$ K/m (ISA lapse rate) and $h$ is station elevation in metres.

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

Fog is classified when dew-point depression $T - T_d < 1.5$ C and wind speed $< 3$ m/s.

### Frontal detection

| Front | Pressure | Wind | Temperature | Humidity |
|---|---|---|---|---|
| **Warm** | Falling $> 1$ hPa/h | Backing (CCW shift) | Rising | Rising $> 2$ %/h |
| **Cold** | Trough (accel $< -0.5$) | Veering (CW shift) | Dropping $> 1$ C/h | -- |
| **Occluded** | Both warm + cold signals | -- | -- | -- |

Wind shift is computed from the cross-product of consecutive direction
vectors — positive = veering (clockwise), negative = backing.

### Energy-balance temperature model

Temperature forecasts use a simplified surface energy budget:

$$\frac{dT}{dt} = \frac{Q_{sw} - Q_{lw} - Q_h}{C}$$

The model evaluates corrections relative to the current observation:

- **Diurnal forcing** — asymmetric curve (slow solar warming, faster radiative cooling) with amplitude from latitude and cloud cover
- **Radiative cooling** — enhanced for clear nights, approximately 1.5 C/h damped by humidity and wind mixing
- **Thermal inertia** — time constant $\tau$ depends on wind (faster response) and humidity (slower, latent heat)

### Clausius-Clapeyron humidity model

Relative humidity adjusts to temperature change via conservation of mixing ratio:

$$RH_2 = RH_1 \cdot \frac{e_s(T_1)}{e_s(T_2)}$$

where saturation vapour pressure follows the Magnus formula:

$$e_s(T) = 6.112 \cdot \exp\!\left(\frac{17.67 \cdot T}{T + 243.5}\right) \quad [\text{hPa}]$$

Cooling raises RH (condensation, fog, precipitation).  Warming drops RH (clearing).

### Pressure extrapolation

Damped linear extrapolation with hourly decay factor $\gamma = 0.92$:

$$P(t+h) = P_0 + \sum_{i=1}^{h} \frac{dp}{dt} \cdot \gamma^i$$

Clamped to $[920, 1070]$ hPa.  The trend halves every approximately 8 hours, reflecting typical synoptic time-scales.

</details>

---

## Bayesian forecaster

<details>
<summary>Transition matrix, evidence updates, hard constraints</summary>

### Transition matrix

A 12x12 Markov matrix $\mathbf{T}$ encodes climatological state persistence and transitions:

$$\mathbf{p}_{t+1} = \mathbf{p}_t \cdot \mathbf{T}$$

Key properties:

- Diagonal entries are high (weather persists): $T_{ii} \in [0.50, 0.75]$
- Sunny/partly-cloudy and cloudy/rainy have the strongest off-diagonal couplings
- Snow states only connect to cold/wet states

### Evidence updates

Each hour, the prior from the Markov step is multiplied by likelihood factors derived from sensor trends:

$$p_j \leftarrow p_j \cdot L_j(\text{evidence})$$

| Factor | What it modulates |
|---|---|
| Pressure tendency $dp/dt$ | Falling: +precip; rising: +clear |
| Pressure acceleration $d^2p/dt^2$ | Negative: +severe weather |
| Humidity | High: +fog/precip; low: +clear |
| Dew depression trend | Converging: +fog/precip |
| Frontal flags | Warm: +rain; cold: +showers/wind |
| Wind speed | High: +windy state |

After evidence multiplication, the vector is renormalised to sum to 1.

### Hard constraints

Physical impossibilities are zeroed out:

- **No snow above 6 C** — $p(\text{snowy}) = p(\text{snowy-rainy}) = 0$ if $T > 6$
- **Day/night swap** — $p(\text{sunny}) \leftrightarrow p(\text{clear-night})$ based on whether the forecast hour falls between sunrise and sunset
- **Precipitation probability** — sum of all wet-state probabilities (rainy + pouring + snowy + snowy-rainy + lightning-rainy)

</details>

---

## Entity attributes

<details>
<summary>Extra state attributes exposed by the weather entity</summary>

| Attribute | Type | Description |
|---|---|---|
| `pressure_trend` | float | $dp/dt$ in hPa/h |
| `pressure_acceleration` | float | $d^2p/dt^2$ in hPa/h squared |
| `dew_point` | float | Degrees C |
| `dew_depression` | float | $T - T_d$ in C |
| `wet_bulb` | float | Degrees C |
| `front_warm` | bool | Warm front detected |
| `front_cold` | bool | Cold front detected |
| `front_occluded` | bool | Occluded front detected |
| `next_hour_condition` | str | HA condition for hour +1 |
| `next_hour_precip_probability` | int | 0-100 for hour +1 |
| `precip_probability_6h` | int | P(any rain in next 6h) |

Standard weather entity properties are also available:
`temperature`, `apparent_temperature`, `dew_point`, `humidity`,
`pressure`, `wind_speed`, `wind_bearing`, hourly forecast (12h),
daily forecast (today + tomorrow).

</details>

---

## Sensor entities

<details>
<summary>Standalone sensors with history for tile cards, graphs, badges, and automations</summary>

Key forecast values are also exposed as standalone sensor entities.  Unlike
weather entity attributes, sensors have their own history in the HA recorder
and can be graphed in tile cards, used in badges, and referenced in automations.

| Entity | Unit | Description |
|---|---|---|
| `sensor.local_forecast_precipitation_probability` | % | Probability of rain in the next 6 hours |
| `sensor.local_forecast_1h_forecast` | — | Forecast condition for +1h (human-readable text) |
| `sensor.local_forecast_next_hour_precipitation_probability` | % | Precipitation probability for +1h |

All sensors update automatically when the weather entity recalculates.
The `%` sensors have `state_class: measurement` so HA records long-term
statistics — you get min/max/mean in the history panel and trend graphs
in tile cards.

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

Copy `custom_components/local_forecast/` into your HA `config/custom_components/` directory and restart.

### Configuration

Settings, Devices and Services, Add Integration, Local Weather Forecast.

| Field | Required | Description |
|---|---|---|
| Pressure sensor | yes | Any pressure entity (hPa, inHg, kPa — auto-converted) |
| Temperature sensor | yes | Outdoor temperature (C or F — auto-converted) |
| Humidity sensor | | Outdoor relative humidity (%) |
| Wind speed sensor | | m/s, km/h, mph, or knots — auto-converted |
| Wind direction sensor | | Degrees (0-360) |
| Solar radiation sensor | | W/m2 — improves cloud estimation |
| Rain rate sensor | | mm/h — enables live precipitation detection |
| Elevation | | Station elevation in metres (for QFE to QNH conversion) |
| Pressure type | | Absolute (QFE) or Sea-level (QNH) |

Minimum viable setup: pressure + temperature.  Everything else improves accuracy.

Recommended setup: pressure + temperature + humidity + wind speed + wind direction.

### Sensor recommendations

| Sensor | Why it matters |
|---|---|
| **Pressure** | The single most important weather predictor.  All frontal detection, storm warning, and trend analysis depends on it.  Use a dedicated barometric sensor (BME280/BMP388), not a phone app. |
| **Temperature** | Drives precipitation type (rain vs snow), diurnal model, feels-like temperature. |
| **Humidity** | Enables dew point, wet-bulb (snow/rain classification), fog detection, Clausius-Clapeyron humidity forecast. |
| **Wind speed** | Frontal detection (backing/veering), windchill, radiative cooling damping. |
| **Wind direction** | Critical for frontal passage detection (warm front = backing, cold front = veering). |
| **Solar radiation** | Cloud fraction estimation when no other cloud data is available.  Falls back to humidity-based estimate without it. |
| **Rain rate** | Live precipitation detection.  Without it, precipitation is inferred from humidity + pressure trends. |

</details>

---

## Dashboard cards

<details>
<summary>Native cards, tile graphs, badges, Mushroom layout, pressure trends and frontal chips</summary>

### Minimal setup (no extra cards needed)

The weather entity works out of the box with native HA cards.  No
conditional cards, no template icons — the entity already handles
precipitation type, day/night, and all 12 conditions internally.

```yaml
type: vertical-stack
cards:
  - type: weather-forecast
    entity: weather.local_forecast
    show_current: true
    show_forecast: true
    forecast_type: hourly
    forecast_slots: 6

  - type: weather-forecast
    entity: weather.local_forecast
    show_current: false
    show_forecast: true
    forecast_type: daily
```

### Tile card with precipitation trend graph

The 6h precipitation probability sensor has history, so the tile card
can show a trend graph natively — no extra integrations needed:

```yaml
type: tile
entity: sensor.local_forecast_precipitation_probability
features:
  - type: graph
    days_to_show: 1
```

### Badge with next-hour forecast

Native HA badges appear at the top of a dashboard view:

```yaml
badges:
  - entity: sensor.local_forecast_1h_forecast
  - entity: sensor.local_forecast_precipitation_probability
```

### Adding pressure trend and frontal alerts

For the extra attributes (pressure trend, frontal detection, 6h rain probability),
add a markdown card below the weather cards:

```yaml
  - type: markdown
    content: >-
      **Pressure:** {{ state_attr('weather.local_forecast', 'pressure') | round(1) }} hPa
      ({{ state_attr('weather.local_forecast', 'pressure_trend') | round(1) }} hPa/h)
      {% if state_attr('weather.local_forecast', 'front_warm') %} | Warm front{% endif %}
      {% if state_attr('weather.local_forecast', 'front_cold') %} | Cold front{% endif %}
      {% if state_attr('weather.local_forecast', 'front_occluded') %} | Occluded front{% endif %}
      | Wet-bulb: {{ state_attr('weather.local_forecast', 'wet_bulb') }}°C
      | 6h rain: {{ state_attr('weather.local_forecast', 'precip_probability_6h') | default(0) }}%
```

### Standard weather card

Works out of the box:

```yaml
type: weather-forecast
entity: weather.local_forecast
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
      {{ states('weather.local_forecast') | replace('-', ' ') | title }}

  - type: horizontal-stack
    cards:
      - type: custom:mushroom-template-card
        primary: Now
        secondary: >-
          {{ state_attr('weather.local_forecast', 'temperature') | round }}°C
        icon: >-
          {% set c = states('weather.local_forecast') %}
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
          {% set pp = state_attr('weather.local_forecast', 'next_hour_precip_probability') | int(0) %}
          {% if pp > 70 %}red{% elif pp > 40 %}orange{% elif pp > 20 %}yellow{% else %}blue{% endif %}
        layout: vertical

      - type: custom:mushroom-template-card
        primary: +1h
        secondary: >-
          {{ state_attr('weather.local_forecast', 'next_hour_condition') | replace('-', ' ') | title }}
          {{ state_attr('weather.local_forecast', 'next_hour_precip_probability') }}% precip
        icon: mdi:clock-outline
        icon_color: orange
        layout: vertical

  - type: horizontal-stack
    cards:
      - type: custom:mushroom-template-card
        primary: Pressure
        secondary: >-
          {{ state_attr('weather.local_forecast', 'pressure') | round(1) }} hPa
        icon: >-
          {% set dp = state_attr('weather.local_forecast', 'pressure_trend') | float(0) %}
          {% if dp > 0.5 %}mdi:trending-up{% elif dp < -0.5 %}mdi:trending-down{% else %}mdi:trending-neutral{% endif %}
        icon_color: >-
          {% set dp = state_attr('weather.local_forecast', 'pressure_trend') | float(0) %}
          {% if dp < -1.5 %}red{% elif dp < -0.5 %}orange{% elif dp > 0.5 %}green{% else %}blue{% endif %}
        layout: vertical

      - type: custom:mushroom-template-card
        primary: Humidity
        secondary: >-
          {{ state_attr('weather.local_forecast', 'humidity') }}%
          (Dew {{ state_attr('weather.local_forecast', 'dew_point') }}°C)
        icon: mdi:water-percent
        icon_color: teal
        layout: vertical

  - type: custom:mushroom-chips-card
    chips:
      - type: template
        content: >-
          {% if state_attr('weather.local_forecast', 'front_warm') %}Warm front{% elif state_attr('weather.local_forecast', 'front_cold') %}Cold front{% elif state_attr('weather.local_forecast', 'front_occluded') %}Occluded{% else %}No front{% endif %}
      - type: template
        content: >-
          6h rain: {{ state_attr('weather.local_forecast', 'precip_probability_6h') | default(0) }}%
      - type: template
        content: >-
          Feels {{ state_attr('weather.local_forecast', 'apparent_temperature') | round }}°C
```

</details>

---

## Pixel display

<details>
<summary>ESPHome pixel display automation with icon mapping</summary>

The weather entity condition maps directly to ESPHome icon names:

```yaml
automation:
  - alias: "Pixel display weather icon"
    trigger:
      - platform: state
        entity_id: weather.local_forecast
    action:
      - action: esphome.pixel_icon_screen
        data:
          icon_name: >-
            {% set map = {
              'sunny':           'weather_sunny',
              'clear-night':     'weather_clear_night',
              'partlycloudy':    'weather_partly_cloudy',
              'cloudy':          'weather_cloudy',
              'fog':             'weather_fog',
              'rainy':           'weather_rainy',
              'pouring':         'weather_pouring',
              'snowy':           'weather_snowy',
              'snowy-rainy':     'weather_snowy_rainy',
              'lightning-rainy': 'weather_lightning',
              'windy':           'weather_windy',
              'exceptional':     'weather_exceptional',
            } %}
            {{ map.get(states('weather.local_forecast'), 'weather_cloudy') }}
          text: >-
            {{ state_attr('weather.local_forecast', 'temperature') | round }}°
          lifetime: 300
          switch_time: 10
```

Or shorter — use the condition directly since it already matches standard weather icon naming:

```yaml
          icon_name: "weather_{{ states('weather.local_forecast') | replace('-', '_') }}"
```

</details>

---

## Troubleshooting

<details>
<summary>Common issues and fixes</summary>

**Icons show wrong condition**

- Check that your sensors are reporting correct values in Developer Tools, States
- Verify pressure type setting matches your sensor (QFE vs QNH)
- If precipitation type is wrong, check wet-bulb in attributes — should be below -2 C for snow

**Forecast does not update**

- Entity updates on sensor state changes (throttled to 30s)
- Check that sensor entities are not `unavailable` or `unknown`

**Temperature seems off**

- Check `unit_of_measurement` on your sensor — the integration auto-converts F, but the attribute must be set correctly
- Elevation affects QFE-to-QNH which affects frontal detection — verify it is correct

**Exceptional shows too often**

- This state triggers on extreme pressure (below 970 or above 1050 hPa) or temperature (above 42 or below -25 C)
- If your elevation is wrong, the QNH conversion may produce extreme values

**Sensor values rejected**

- Pressure outside 870-1090 hPa after unit conversion is rejected (returns unavailable)
- Temperature outside -60 to 60 C after conversion is rejected
- Humidity is clamped to 0-100%, wind speed to 0-60 m/s
- Check `unit_of_measurement` attribute on your sensor entities — the integration relies on it for auto-conversion

</details>

---

## Debugging

If you're having issues with the integration, there are two ways to enable debug logging.

### Option 1 — YAML configuration

Add the following to your `configuration.yaml` and restart Home Assistant:

```yaml
logger:
  default: info
  logs:
    custom_components.local_forecast: debug
```

### Option 2 — Home Assistant UI

1. Go to **Settings** → **Devices & Services**.
2. Find the **Local Weather Forecast** integration and click the **⋮** menu.
3. Select **Enable debug logging**.
4. Reproduce the issue.
5. Click **Disable debug logging** — the browser will download a log file you can inspect or attach to a bug report.

Debug output includes sensor values after unit conversion, current weather state classification, and forecast summary for the next hour.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
