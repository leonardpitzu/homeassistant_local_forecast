"""Optional interactive pan/zoom satellite map view.

Serves a self-contained Leaflet page that tiles EUMETView's public WMS GetMap
endpoint directly from the browser, so a dashboard can pan and zoom live
satellite imagery without any polling, entities, or token. Leaflet itself is
vendored into the integration, so the viewer works with no internet access to
anything but the imagery server and pulls no third-party script.

The endpoint is unauthenticated because an `iframe` card cannot present a
token, so the map centre is snapped to a coarse grid: it frames the right
region without publishing the home location to anyone who can reach the port.

The view is registered only while at least one config entry has the map
enabled; otherwise it responds 404, so when the user does not want it, it is
not there. Embed it with an `iframe` card pointing at ``/api/local_forecast/map``.
"""

from __future__ import annotations

import json
from pathlib import Path

from aiohttp import web
from homeassistant.components.http import HomeAssistantView, StaticPathConfig
from homeassistant.core import HomeAssistant

from .const import (
    DOMAIN,
    MAP_CENTER_GRID,
    MAP_DEFAULT_ZOOM,
    MAP_FALLBACK_CENTER,
    MAP_LAYERS,
    MAP_MAX_ZOOM,
    MAP_STATIC_URL,
    MAP_TIME_REFRESH_MS,
    MAP_TIMES_URL,
    MAP_VIEW_URL,
    WMS_BASE_URL,
    WMS_VERSION,
)
from .wms_time import async_get_frame_times, cached_frame_times

LEAFLET_DIR = Path(__file__).parent / "leaflet"


async def async_register_static_assets(hass: HomeAssistant) -> None:
    """Expose the vendored Leaflet bundle."""
    await hass.http.async_register_static_paths(
        [StaticPathConfig(MAP_STATIC_URL, str(LEAFLET_DIR), True)]
    )


class LocalForecastMapView(HomeAssistantView):
    """Serve the Leaflet pan/zoom satellite viewer."""

    url = MAP_VIEW_URL
    name = "api:local_forecast:map"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        """Capture hass so the home location is read live per request."""
        self._hass = hass

    async def get(self, request: web.Request) -> web.Response:
        """Return the viewer HTML, or 404 when the map is disabled."""
        if not self._hass.data.get(DOMAIN, {}).get("map_enabled"):
            return web.Response(status=404)
        return web.Response(
            text=self._render(cached_frame_times(self._hass)),
            content_type="text/html",
            headers={
                "X-Frame-Options": "SAMEORIGIN",
                "Referrer-Policy": "no-referrer",
                # Safari restores an iframe from its back-forward cache without
                # re-running the page, which is one way a stale frame sticks.
                "Cache-Control": "no-store",
            },
        )

    def _render(self, times: dict[str, str]) -> str:
        """Build the self-contained Leaflet HTML for the current home region."""
        latitude = self._hass.config.latitude
        longitude = self._hass.config.longitude
        if latitude is None or longitude is None:
            center = list(MAP_FALLBACK_CENTER)
        else:
            grid = MAP_CENTER_GRID
            center = [
                round(round(latitude / grid) * grid, 4),
                round(round(longitude / grid) * grid, 4),
            ]
        config = json.dumps(
            {
                "base": WMS_BASE_URL,
                "version": WMS_VERSION,
                "layers": [{"id": lid, "name": name} for lid, name in MAP_LAYERS],
                "center": center,
                "zoom": MAP_DEFAULT_ZOOM,
                "maxZoom": MAP_MAX_ZOOM,
                "static": MAP_STATIC_URL,
                "times": times,
                "timesUrl": MAP_TIMES_URL,
                "refreshMs": MAP_TIME_REFRESH_MS,
            }
        )
        return _HTML_TEMPLATE.replace("__CONFIG__", config).replace(
            "__STATIC__", MAP_STATIC_URL
        )


class LocalForecastMapTimesView(HomeAssistantView):
    """Report the newest satellite frame available for each layer.

    Shares the viewer's auth posture because it exists to serve that page; it
    carries nothing but public EUMETSAT frame timestamps.
    """

    url = MAP_TIMES_URL
    name = "api:local_forecast:map:times"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        """Capture hass so the shared frame-time cache is read per request."""
        self._hass = hass

    async def get(self, request: web.Request) -> web.Response:
        """Return ``{layer_id: frame_time}``, or 404 when the map is disabled."""
        if not self._hass.data.get(DOMAIN, {}).get("map_enabled"):
            return web.Response(status=404)
        return web.json_response(
            await async_get_frame_times(self._hass),
            headers={"Cache-Control": "no-store"},
        )


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Local Weather Forecast</title>
<link rel="stylesheet" href="__STATIC__/leaflet.css" />
<script src="__STATIC__/leaflet.js"></script>
<style>
  html, body, #map { height: 100%; margin: 0; background: #000; }
  #stamp {
    position: fixed; right: 8px; bottom: 8px; z-index: 1000;
    padding: 3px 7px; border-radius: 4px; pointer-events: none;
    background: rgba(0, 0, 0, 0.6); color: #fff;
    font: 12px/1.4 system-ui, sans-serif;
  }
</style>
</head>
<body>
<div id="map"></div>
<div id="stamp"></div>
<script>
  const cfg = __CONFIG__;
  const map = L.map('map', { worldCopyJump: true, maxZoom: cfg.maxZoom });
  map.setView(cfg.center, cfg.zoom);

  const stamp = document.getElementById('stamp');
  const wmsLayers = [];
  let active = null;

  const baseLayers = {};
  cfg.layers.forEach((layer, i) => {
    const params = {
      layers: layer.id,
      format: 'image/png',
      transparent: false,
      version: cfg.version,
      detectRetina: true,
      maxZoom: cfg.maxZoom,
    };
    // Pinning TIME gives every frame its own URL, so a cached tile can never
    // masquerade as the current one. Omitted when the timeline is unknown,
    // which falls back to the server's own "latest".
    if (cfg.times[layer.id]) params.time = cfg.times[layer.id];
    const wms = L.tileLayer.wms(cfg.base, params);
    wms.layerId = layer.id;
    wmsLayers.push(wms);
    baseLayers[layer.name] = wms;
    if (i === 0) { active = wms; wms.addTo(map); }
  });
  L.control.layers(baseLayers, null, { collapsed: false }).addTo(map);
  map.on('baselayerchange', (e) => { active = e.layer; showStamp(); });

  function showStamp() {
    const t = active && active.wmsParams.time;
    stamp.textContent = t ? new Date(t).toLocaleString() : 'latest available';
  }

  function applyTimes(times) {
    wmsLayers.forEach((wms) => {
      const t = times[wms.layerId];
      if (t && t !== wms.wmsParams.time) {
        wms.setParams({ time: t }, !map.hasLayer(wms));
      }
    });
    showStamp();
  }

  async function refresh() {
    try {
      const resp = await fetch(cfg.timesUrl, { cache: 'no-store' });
      if (resp.ok) applyTimes(await resp.json());
    } catch (err) {
      // Keep displaying the last frame we know about.
    }
  }

  setInterval(refresh, cfg.refreshMs);
  // Safari serves an iframe straight out of its back-forward cache, so a
  // returning tab never re-runs the page unless pageshow is handled.
  window.addEventListener('pageshow', refresh);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) refresh();
  });
  showStamp();
</script>
</body>
</html>
"""
