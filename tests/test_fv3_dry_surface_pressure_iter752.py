"""FV3_3D iter 752: dry_surface_pressure_fv3 port.

ps_dry = ps - g * total_water_column.

Tests
-----

1. ``test_dry_ps_dry_atmosphere``.
2. ``test_dry_ps_with_vapor``.
3. ``test_dry_ps_matches_iter694_global``.
4. ``test_dry_ps_shapes_3d``.
5. ``test_dry_ps_finite``.
6. ``test_dry_ps_no_tracers_raises``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    area_weighted_mean_fv3,
    dry_surface_pressure_fv3,
    prt_mass_fv3,
)


def test_dry_ps_dry_atmosphere():
    """q=0 → ps_dry = ps."""
    km = 5
    ps = jnp.full((3,), 1.0e5)
    delp = jnp.full((3, km), 2.0e4)
    q = jnp.zeros((3, km))
    ps_dry = dry_surface_pressure_fv3(ps, delp, q_sphum=q)
    assert jnp.allclose(ps_dry, ps, atol=1e-8)


def test_dry_ps_with_vapor():
    """q=0.01 uniform, p_s=1e5 → ps_dry = ps - g · 0.01·1e5/g = ps - 1000."""
    km = 10
    ps = jnp.full((2,), 1.0e5)
    delp = jnp.full((2, km), 1.0e4)
    q = jnp.full((2, km), 0.01)
    ps_dry = dry_surface_pressure_fv3(ps, delp, q_sphum=q)
    expected = 1.0e5 - 0.01 * 1.0e5   # ps - water-mass-pressure
    assert jnp.allclose(ps_dry, expected, atol=1.0)


def test_dry_ps_matches_iter694_global():
    """Area-weighted mean of dry_surface_pressure matches iter-694
    dry_ps_mean."""
    n_x, n_y, km = 4, 4, 10
    ps = jnp.full((n_x, n_y), 1.0e5)
    delp = jnp.full((n_x, n_y, km), 1.0e4)
    area = jnp.ones((n_x, n_y))
    q = jnp.full((n_x, n_y, km), 0.005)
    ps_dry = dry_surface_pressure_fv3(ps, delp, q_sphum=q)
    mean_via_iter752 = float(area_weighted_mean_fv3(ps_dry, area))
    diag = prt_mass_fv3(ps, delp, {"sphum": q}, area)
    assert abs(mean_via_iter752 - diag["dry_ps_mean"]) < 1e-6


def test_dry_ps_shapes_3d():
    """3-D ps + 4-D delp → 3-D output (matching ps shape)."""
    rng = np.random.default_rng(seed=752)
    n_x, n_y, km = 4, 5, 20
    ps = jnp.asarray(rng.uniform(9.0e4, 1.05e5, size=(n_x, n_y)))
    delp = jnp.full((n_x, n_y, km), 5000.0)
    q = jnp.asarray(rng.uniform(0.0, 0.02, size=(n_x, n_y, km)))
    ps_dry = dry_surface_pressure_fv3(ps, delp, q_sphum=q)
    assert ps_dry.shape == (n_x, n_y)


def test_dry_ps_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=753)
    ps = jnp.asarray(rng.uniform(9.0e4, 1.05e5, size=(4, 4)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, 30)))
    q = jnp.asarray(rng.uniform(0.0, 0.025, size=(4, 4, 30)))
    ps_dry = dry_surface_pressure_fv3(ps, delp, q_sphum=q)
    assert jnp.all(jnp.isfinite(ps_dry))


def test_dry_ps_no_tracers_raises():
    """No tracers → raises (delegated to iter-749)."""
    ps = jnp.full((3,), 1.0e5)
    delp = jnp.full((3, 5), 2.0e4)
    with pytest.raises(ValueError):
        dry_surface_pressure_fv3(ps, delp)
