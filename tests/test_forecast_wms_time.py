"""Frame-time extraction from EUMETView's WMS capabilities.

These are the rules that break the satellite viewer when they are wrong: a
timeline that cannot be read means an unpinned (and therefore cacheable-stale)
tile URL, and a mis-keyed layer means one layer wearing another's timestamp.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from local_forecast.wms_time import _frame_time, _parse_period, parse_capabilities

WMS_NS = 'xmlns="http://www.opengis.net/wms"'


def _caps(*layers: str) -> bytes:
    """Wrap layer fragments in a minimal WMS capabilities document."""
    return (
        f'<?xml version="1.0"?><WMS_Capabilities {WMS_NS}><Capability><Layer>'
        + "".join(layers)
        + "</Layer></Capability></WMS_Capabilities>"
    ).encode()


def _layer(name: str, default: str, domain: str) -> str:
    return (
        f"<Layer><Name>{name}</Name>"
        f'<Dimension name="time" units="ISO8601" default="{default}" '
        f'nearestValue="1">{domain}</Dimension></Layer>'
    )


MSG_DOMAIN = "2020-09-01T00:00:00.000Z/2026-08-08T00:45:00.000Z/PT15M"
MTG_DOMAIN = "2024-09-23T00:00:00.000Z/2026-08-08T00:50:00.000Z/PT10M"


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        (MTG_DOMAIN, timedelta(minutes=10)),
        (MSG_DOMAIN, timedelta(minutes=15)),
        ("a/b/PT1H", timedelta(hours=1)),
        ("a/b/P1DT2H30M", timedelta(days=1, hours=2, minutes=30)),
        ("a/b/nonsense", None),
        ("no-slashes", None),
        ("a/b/PT0M", None),
    ],
)
def test_parse_period(domain, expected):
    assert _parse_period(domain) == expected


def test_frame_time_steps_back_one_cadence_slot():
    """The newest advertised frame can still be partial, so we serve the prior one."""
    assert _frame_time("2026-08-08T00:50:00Z", MTG_DOMAIN) == "2026-08-08T00:40:00Z"
    assert _frame_time("2026-08-08T00:45:00Z", MSG_DOMAIN) == "2026-08-08T00:30:00Z"


def test_frame_time_falls_back_to_default_when_cadence_is_unreadable():
    assert _frame_time("2026-08-08T00:50:00Z", "garbage") == "2026-08-08T00:50:00Z"


def test_frame_time_rejects_an_unparseable_default():
    assert _frame_time("not-a-timestamp", MTG_DOMAIN) is None


def test_parse_capabilities_restores_the_workspace_prefix():
    """Virtual services drop the prefix, but layer ids in const.py carry it."""
    times = parse_capabilities(_caps(_layer("rgb_dust", "2026-08-08T00:45:00Z", MSG_DOMAIN)), "msg_fes")

    assert times == {"msg_fes:rgb_dust": "2026-08-08T00:30:00Z"}


def test_parse_capabilities_keeps_same_named_layers_of_two_workspaces_apart():
    """rgb_dust exists in both workspaces with different timelines."""
    msg = parse_capabilities(_caps(_layer("rgb_dust", "2026-08-08T00:45:00Z", MSG_DOMAIN)), "msg_fes")
    mtg = parse_capabilities(_caps(_layer("rgb_dust", "2026-08-08T00:50:00Z", MTG_DOMAIN)), "mtg_fd")

    merged = {**msg, **mtg}
    assert merged["msg_fes:rgb_dust"] == "2026-08-08T00:30:00Z"
    assert merged["mtg_fd:rgb_dust"] == "2026-08-08T00:40:00Z"


def test_parse_capabilities_gives_each_layer_its_own_time():
    """Layers in one workspace run on independent timelines."""
    times = parse_capabilities(
        _caps(
            _layer("rgb_cloudphase", "2026-08-07T21:50:00Z", MTG_DOMAIN),
            _layer("rgb_geocolour", "2026-08-08T00:50:00Z", MTG_DOMAIN),
        ),
        "mtg_fd",
    )

    assert times["mtg_fd:rgb_cloudphase"] == "2026-08-07T21:40:00Z"
    assert times["mtg_fd:rgb_geocolour"] == "2026-08-08T00:40:00Z"


def test_parse_capabilities_skips_layers_without_a_usable_time_dimension():
    """Container layers and undated layers must not produce entries."""
    times = parse_capabilities(
        _caps(
            "<Layer><Name>rgb_notime</Name></Layer>",
            f'<Layer><Name>rgb_nodefault</Name><Dimension name="time">{MTG_DOMAIN}</Dimension></Layer>',
            _layer("rgb_ok", "2026-08-08T00:50:00Z", MTG_DOMAIN),
        ),
        "mtg_fd",
    )

    assert times == {"mtg_fd:rgb_ok": "2026-08-08T00:40:00Z"}
