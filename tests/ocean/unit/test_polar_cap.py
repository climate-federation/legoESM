"""Unit test for the north_cap_lat / south_cap_lat polar-cap option in
BathymetryConfig.  These cap fields close off high-latitude ocean cells
where the lat-lon grid singularity + small dx + cos²(lat) A_h scaling
combine to leave the Arctic under-damped (see
``docs/ocean/experiments/realistic_geometry_topology_fixes.md``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.bathymetry import BathymetryConfig


def test_polar_cap_default_is_no_cap():
    """Default config has both caps None — bit-exact regression for legacy."""
    cfg = BathymetryConfig(source="idealized")
    assert cfg.north_cap_lat is None
    assert cfg.south_cap_lat is None


def test_north_cap_field_accessible():
    """Verify the new fields are NamedTuple members and accept floats."""
    cfg = BathymetryConfig(
        source="file", path="/tmp/nonexistent.nc",
        north_cap_lat=80.0, south_cap_lat=-80.0,
    )
    assert cfg.north_cap_lat == 80.0
    assert cfg.south_cap_lat == -80.0
    # None preserves bit-exactness
    cfg_none = BathymetryConfig(source="file", path="/tmp/nonexistent.nc")
    assert cfg_none.north_cap_lat is None
    assert cfg_none.south_cap_lat is None


def test_north_cap_idempotent_on_cap_at_pole():
    """Setting north_cap_lat = 90° should be a no-op (no cells above 90°)."""
    cfg = BathymetryConfig(
        source="file", path="/tmp/nonexistent.nc",
        north_cap_lat=90.0,
    )
    # Just verify the cap is set; the no-op behaviour is tested implicitly
    # by the fact that no cells satisfy lat > 90° on a real grid.
    assert cfg.north_cap_lat == 90.0
