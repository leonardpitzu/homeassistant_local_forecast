"""Shared fixtures for the Local Weather Forecast test-suite."""

import os
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

sys.path.insert(0, os.path.join(str(REPO_ROOT), "custom_components"))

pytest_plugins = "pytest_homeassistant_custom_component"


@pytest.fixture
def hass_config_dir(hass_tmp_config_dir: str) -> str:
    """Point the test instance at a config dir that sees our component."""
    root = pathlib.Path(hass_tmp_config_dir) / "custom_components"
    root.mkdir(exist_ok=True)
    link = root / "local_forecast"
    if not link.exists():
        link.symlink_to(
            REPO_ROOT / "custom_components" / "local_forecast",
            target_is_directory=True,
        )
    return hass_tmp_config_dir


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(request):
    """Let the HA test harness load custom_components/local_forecast.

    Resolved lazily so tests that need the recorder can still have it built
    before the ``hass`` instance exists.
    """
    if "hass" not in request.fixturenames:
        yield
        return
    if "recorder_mock" in request.fixturenames:
        request.getfixturevalue("recorder_mock")
    request.getfixturevalue("enable_custom_integrations")
    yield
