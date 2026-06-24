"""Optional interactive pan/zoom satellite map view.

Serves a self-contained Leaflet page that tiles EUMETView's public WMS GetMap
endpoint directly from the browser, so a dashboard can pan and zoom live
satellite imagery without any polling, entities, or token. The viewer opens
centred on the Home Assistant home location (read live, so moving home in HA
settings recentres the map) and requests HiDPI tiles for a sharper image.

The view is registered only while at least one config entry has the map
enabled; otherwise it responds 404, so when the user does not want it, it is
not there. Embed it with an `iframe` card pointing at ``/api/local_forecast/map``.
"""

from __future__ import annotations

import json

from aiohttp import web
from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant

from .const import (
    DOMAIN,
    MAP_DEFAULT_ZOOM,
    MAP_FALLBACK_CENTER,
    MAP_LAYERS,
    MAP_MAX_ZOOM,
    MAP_VIEW_URL,
    WMS_BASE_URL,
    WMS_VERSION,
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
        return web.Response(text=self._render(), content_type="text/html")

    def _render(self) -> str:
        """Build the self-contained Leaflet HTML for the current home location."""
        latitude = self._hass.config.latitude
        longitude = self._hass.config.longitude
        if latitude is None or longitude is None:
            center = list(MAP_FALLBACK_CENTER)
        else:
            center = [latitude, longitude]
        config = json.dumps(
            {
                "base": WMS_BASE_URL,
                "version": WMS_VERSION,
                "layers": [{"id": lid, "name": name} for lid, name in MAP_LAYERS],
                "center": center,
                "zoom": MAP_DEFAULT_ZOOM,
                "maxZoom": MAP_MAX_ZOOM,
            }
        )
        return _HTML_TEMPLATE.replace("__CONFIG__", config)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Local Weather Forecast</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body, #map { height: 100%; margin: 0; background: #000; }
</style>
</head>
<body>
<div id="map"></div>
<script>
  const cfg = __CONFIG__;
  const map = L.map('map', { worldCopyJump: true, maxZoom: cfg.maxZoom });
  map.setView(cfg.center, cfg.zoom);
  const baseLayers = {};
  cfg.layers.forEach((layer, i) => {
    const wms = L.tileLayer.wms(cfg.base, {
      layers: layer.id,
      format: 'image/png',
      transparent: false,
      version: cfg.version,
      detectRetina: true,
      maxZoom: cfg.maxZoom,
    });
    baseLayers[layer.name] = wms;
    if (i === 0) wms.addTo(map);
  });
  L.control.layers(baseLayers, null, { collapsed: false }).addTo(map);
  L.circleMarker(cfg.center, {
    radius: 6, color: '#fff', weight: 2, fillColor: '#1e88e5', fillOpacity: 0.9,
  }).addTo(map);
</script>
</body>
</html>
"""
