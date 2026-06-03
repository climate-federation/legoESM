"""Shared fixtures and pytest configuration for ocean fidelity tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--network",
        action="store_true",
        default=False,
        help="Enable tests that hit the live network (obs downloads, Veros pulls).",
    )
    parser.addoption(
        "--cmems",
        action="store_true",
        default=False,
        help="Enable CMEMS-gated tests (requires CMEMS_USER / CMEMS_PASS env vars).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: requires internet; deselect by default"
    )
    config.addinivalue_line(
        "markers", "veros: requires the [ocean-fidelity] extra installed"
    )
    config.addinivalue_line(
        "markers", "cmems: requires CMEMS_USER / CMEMS_PASS env vars"
    )


def pytest_collection_modifyitems(config, items):
    skip_network = not config.getoption("--network")
    skip_cmems = not config.getoption("--cmems")
    for item in items:
        if skip_network and "network" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="needs --network"))
        if skip_cmems and "cmems" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="needs --cmems"))


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch) -> Path:
    """Point the fidelity cache at a per-test tmp dir.

    Removes both ``LEGOESM_OCEAN_FIDELITY_CACHE`` and ``LEGOESM_CACHE_DIR``
    from the environment, then sets the primary env var to a fresh tmp dir
    so each test starts from a known-empty cache.
    """
    cache_root = tmp_path / "fidelity_cache"
    monkeypatch.setenv("LEGOESM_OCEAN_FIDELITY_CACHE", str(cache_root))
    monkeypatch.delenv("LEGOESM_CACHE_DIR", raising=False)
    return cache_root
