"""FV3_3D iter 694: prt_mass_fv3 port.

Faithful JAX port of FV3 ``prt_mass`` (tools/fv_diagnostics.F90:
4164-4263).  Global mass-budget diagnostic for surface pressure
+ water tracers.

Tests
-----

1. ``test_prt_mass_ps_mean_uniform``.
2. ``test_prt_mass_dry_no_tracers``.
3. ``test_prt_mass_known_column_water``.
4. ``test_prt_mass_multi_tracer_sum``.
5. ``test_prt_mass_dry_ps_consistency``.
6. ``test_prt_mass_area_weighting``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import prt_mass_fv3


def test_prt_mass_ps_mean_uniform():
    """Uniform ps → mean ps = same value."""
    n_x, n_y, km = 4, 4, 5
    ps = jnp.full((n_x, n_y), 1.0e5)
    delp = jnp.full((n_x, n_y, km), 1.0e5 / km)
    area = jnp.ones((n_x, n_y))
    q = jnp.zeros((n_x, n_y, km))
    diag = prt_mass_fv3(ps, delp, {"sphum": q}, area)
    assert abs(diag["ps_mean"] - 1.0e5) < 1e-8


def test_prt_mass_dry_no_tracers():
    """No tracers: total_water=0, dry_ps_mean = ps_mean."""
    n_x, n_y, km = 4, 4, 5
    ps = jnp.full((n_x, n_y), 9.8e4)
    delp = jnp.full((n_x, n_y, km), 9.8e4 / km)
    area = jnp.ones((n_x, n_y))
    diag = prt_mass_fv3(ps, delp, {}, area)
    assert diag["total_water"] == 0.0
    assert abs(diag["dry_ps_mean"] - diag["ps_mean"]) < 1e-8


def test_prt_mass_known_column_water():
    """Uniform q=0.01 (10 g/kg), uniform delp:
    column water = q · p_s / g per column.
    For ps=1e5, q=0.01: column = 0.01·1e5/9.81 ≈ 101.94 kg/m²."""
    n_x, n_y, km = 4, 4, 10
    ps = jnp.full((n_x, n_y), 1.0e5)
    delp = jnp.full((n_x, n_y, km), 1.0e4)
    area = jnp.ones((n_x, n_y))
    q = jnp.full((n_x, n_y, km), 0.01)
    diag = prt_mass_fv3(ps, delp, {"sphum": q}, area)
    expected = 0.01 * 1.0e5 / constants.g
    assert abs(diag["sphum"] - expected) / expected < 1e-10
    assert abs(diag["total_water"] - expected) / expected < 1e-10


def test_prt_mass_multi_tracer_sum():
    """Two tracers: total_water = sum of both."""
    n_x, n_y, km = 4, 4, 5
    ps = jnp.full((n_x, n_y), 1.0e5)
    delp = jnp.full((n_x, n_y, km), 2.0e4)
    area = jnp.ones((n_x, n_y))
    q1 = jnp.full((n_x, n_y, km), 0.005)
    q2 = jnp.full((n_x, n_y, km), 0.001)
    diag = prt_mass_fv3(ps, delp, {"sphum": q1, "liq_wat": q2}, area)
    expected1 = 0.005 * 1.0e5 / constants.g
    expected2 = 0.001 * 1.0e5 / constants.g
    assert abs(diag["sphum"] - expected1) / expected1 < 1e-10
    assert abs(diag["liq_wat"] - expected2) / expected2 < 1e-10
    assert abs(diag["total_water"] - (expected1 + expected2)) / (
        expected1 + expected2
    ) < 1e-10


def test_prt_mass_dry_ps_consistency():
    """dry_ps_mean = ps_mean - g · total_water."""
    n_x, n_y, km = 4, 4, 5
    ps = jnp.full((n_x, n_y), 1.0e5)
    delp = jnp.full((n_x, n_y, km), 2.0e4)
    area = jnp.ones((n_x, n_y))
    q = jnp.full((n_x, n_y, km), 0.015)
    diag = prt_mass_fv3(ps, delp, {"sphum": q}, area)
    assert abs(
        diag["dry_ps_mean"] - (diag["ps_mean"] - constants.g * diag["total_water"])
    ) < 1e-6


def test_prt_mass_area_weighting():
    """Spatial-weighted mean differs from arithmetic mean when area
    varies."""
    n_x, n_y, km = 2, 2, 3
    ps = jnp.array([[1.0e5, 9.0e4], [1.0e5, 9.0e4]])
    delp = jnp.full((n_x, n_y, km), 3.3e4)
    # Heavier weight on the (1e5) columns
    area = jnp.array([[2.0, 1.0], [2.0, 1.0]])
    diag = prt_mass_fv3(ps, delp, {}, area)
    # Weighted mean = (2·1e5 + 1·9e4)/(3) per row · 2 rows = (4·1e5+2·9e4)/6
    expected = (4.0 * 1.0e5 + 2.0 * 9.0e4) / 6.0
    assert abs(diag["ps_mean"] - expected) / expected < 1e-10
