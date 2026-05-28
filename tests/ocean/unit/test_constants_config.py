"""Tests for the ocean-scoped ConstantsConfig (Phase G, G-C1).

G-C1 introduces ConstantsConfig and adds it as a field on
LatLonCGridOceanConfig with defaults referencing legoesm.constants — a
zero-behaviour change until call sites are migrated (G-C2+).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from legoesm import constants
from legoesm.ocean.constants_config import ConstantsConfig, VEROS_CONSTANTS_CONFIG
from legoesm.ocean.state import LatLonCGridOceanConfig


def test_defaults_reference_legoesm_constants():
    """Defaults are the canonical legoESM Earth constants (zero-behaviour)."""
    c = ConstantsConfig()
    assert c.g == constants.g
    assert c.rho_0 == constants.rho_ocean
    assert c.c_sw == constants.c_sw
    assert c.Omega == constants.Omega
    assert c.R_earth == constants.R_earth


def test_latlon_config_carries_default_constants():
    """LatLonCGridOceanConfig gains a `constants` field defaulting to the
    canonical values — so existing configs are numerically unchanged."""
    cfg = LatLonCGridOceanConfig()
    assert isinstance(cfg.constants, ConstantsConfig)
    assert cfg.constants.g == constants.g
    assert cfg.constants.rho_0 == constants.rho_ocean
    # The pre-existing top-level g/rho_0 defaults are untouched (no regression).
    assert cfg.g == constants.g
    assert cfg.rho_0 == constants.rho_ocean


def test_veros_constants_config_values():
    """The Veros-pinned bundle carries Veros's canonical values (for G-C4)."""
    v = VEROS_CONSTANTS_CONFIG
    assert v.g == 9.81
    assert v.rho_0 == 1024.0
    assert v.c_sw == 3994.0
    assert v.Omega == pytest.approx(7.292115e-5)
    assert v.R_earth == 6.370e6


def test_constants_field_is_overridable_via_public_api():
    """A recipe can pin constants through the public config API (the point of
    G-C1) without the override_constants monkey-patch."""
    cfg = LatLonCGridOceanConfig(constants=VEROS_CONSTANTS_CONFIG)
    assert cfg.constants.rho_0 == 1024.0
    assert cfg.constants.g == 9.81
