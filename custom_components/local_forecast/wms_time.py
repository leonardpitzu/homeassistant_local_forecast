"""Newest published frame time for each EUMETView satellite layer.

EUMETView advertises every layer's frame timeline as a WMS time dimension::

    <Dimension name="time" default="2026-08-08T00:50:00Z" nearestValue="1">
      2024-09-23T00:00:00.000Z/2026-08-08T00:50:00.000Z/PT10M
    </Dimension>

The timeline has to be read rather than derived from the clock. ``nearestValue``
only snaps *within* the advertised range, so a request for a time past its end —
which "now" always is, given the ~20-25 min dissemination lag — comes back as a
ServiceException rather than the newest frame.

Capabilities are read from GeoServer's per-workspace virtual services and cached
for the whole Home Assistant instance, so every browser between them costs one
fetch per TTL rather than one per page load. A failed refresh keeps the previous
answer: a slightly old frame beats no frame.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import timedelta

import aiohttp
from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    MAP_FRAME_LAG_SLOTS,
    MAP_LAYERS,
    MAP_TIME_CACHE_TTL,
    MAP_TIME_RETRY_TTL,
    WMS_VERSION,
    WMS_WORKSPACE_URL,
)

_LOGGER = logging.getLogger(__name__)

_WMS_NS = "{http://www.opengis.net/wms}"
_LAYER_PATH = f"{_WMS_NS}Layer"
_NAME_PATH = f"{_WMS_NS}Name"
_TIME_DIM_PATH = f"{_WMS_NS}Dimension[@name='time']"

_CACHE_KEY = "map_frame_times"
_LOCK_KEY = "map_frame_times_lock"

_TIMEOUT = aiohttp.ClientTimeout(total=30, connect=10)

# ISO 8601 duration, covering the cadences EUMETView actually publishes.
_PERIOD_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


def _workspaces() -> list[str]:
    """Return the GeoServer workspaces the curated layers live in."""
    return sorted({layer_id.split(":", 1)[0] for layer_id, _ in MAP_LAYERS})


def _parse_period(domain: str) -> timedelta | None:
    """Return the cadence from a ``start/end/PERIOD`` dimension domain."""
    parts = domain.strip().split("/")
    if len(parts) != 3:
        return None
    match = _PERIOD_RE.match(parts[2])
    if not match:
        return None
    fields = {k: int(v) for k, v in match.groupdict(default="0").items()}
    period = timedelta(**fields)
    return period or None


def _frame_time(default: str, domain: str) -> str | None:
    """Return the frame to pin: *default*, stepped back by the configured lag."""
    moment = dt_util.parse_datetime(default)
    if moment is None:
        return None
    if (period := _parse_period(domain)) is not None:
        moment -= period * MAP_FRAME_LAG_SLOTS
    return dt_util.as_utc(moment).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_capabilities(xml: bytes, workspace: str) -> dict[str, str]:
    """Extract ``{"workspace:layer": frame_time}`` from a capabilities document.

    Virtual services name layers without their workspace prefix, and the same
    bare name (``rgb_dust``) exists in more than one workspace with a different
    timeline, so the prefix is put back to keep the two apart.
    """
    root = ET.fromstring(xml)
    times: dict[str, str] = {}
    for layer in root.iter(_LAYER_PATH):
        name = layer.find(_NAME_PATH)
        dimension = layer.find(_TIME_DIM_PATH)
        if name is None or not name.text or dimension is None:
            continue
        default = dimension.get("default")
        if not default:
            continue
        if (frame := _frame_time(default, dimension.text or "")) is not None:
            times[f"{workspace}:{name.text}"] = frame
    return times


async def _async_fetch_workspace(
    session: aiohttp.ClientSession, workspace: str
) -> dict[str, str] | None:
    """Read one workspace's layer timelines, or None if it could not be read."""
    try:
        async with session.get(
            WMS_WORKSPACE_URL.format(workspace=workspace),
            params={
                "service": "WMS",
                "version": WMS_VERSION,
                "request": "GetCapabilities",
            },
            timeout=_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            body = await resp.read()
    except (aiohttp.ClientError, TimeoutError) as err:
        _LOGGER.debug("Could not fetch %s capabilities: %s", workspace, err)
        return None

    try:
        return parse_capabilities(body, workspace)
    except (ET.ParseError, DefusedXmlException) as err:
        _LOGGER.debug("Could not parse %s capabilities: %s", workspace, err)
        return None


def cached_frame_times(hass: HomeAssistant) -> dict[str, str]:
    """Return the frame times already known, without going near the network.

    The viewer renders from this so the page never waits on EUMETView; it asks
    the times endpoint for anything missing as soon as it has loaded.
    """
    _, times = hass.data.get(DOMAIN, {}).get(_CACHE_KEY, (0.0, {}))
    return times


async def async_get_frame_times(hass: HomeAssistant) -> dict[str, str]:
    """Return ``{layer_id: frame_time}`` for the curated layers.

    Layers whose timeline could not be read are simply absent, and the viewer
    then omits ``TIME`` for them — which is exactly the pre-pinning behaviour.
    """
    store = hass.data.setdefault(DOMAIN, {})
    lock: asyncio.Lock = store.setdefault(_LOCK_KEY, asyncio.Lock())

    async with lock:
        expires, cached = store.get(_CACHE_KEY, (0.0, {}))
        if time.monotonic() < expires:
            return cached

        session = async_get_clientsession(hass)
        times = dict(cached)
        refreshed = False
        for workspace in _workspaces():
            if (parsed := await _async_fetch_workspace(session, workspace)) is not None:
                times.update(parsed)
                refreshed = True

        ttl = MAP_TIME_CACHE_TTL if refreshed else MAP_TIME_RETRY_TTL
        store[_CACHE_KEY] = (time.monotonic() + ttl, times)
        return times
