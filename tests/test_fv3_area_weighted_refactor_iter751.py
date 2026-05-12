"""FV3_3D iter 751: refactor iter-694 prt_mass + iter-705 prt_gb_nh_sh
to delegate area-weighted means to iter-750 area_weighted_mean_fv3.

Tests
-----

1. ``test_prt_mass_refactor_unchanged``.
2. ``test_prt_gb_nh_sh_refactor_unchanged``.
3. ``test_iter694_5_tests_preserved``.
4. ``test_iter705_5_tests_preserved``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    area_weighted_mean_fv3,
    prt_gb_nh_sh_fv3,
    prt_mass_fv3,
)


def test_prt_mass_refactor_unchanged():
    """iter-694 still produces correct ps_mean + tracer means via
    iter-750 helper."""
    n_x, n_y, km = 4, 4, 10
    ps = jnp.full((n_x, n_y), 1.0e5)
    delp = jnp.full((n_x, n_y, km), 1.0e4)
    area = jnp.ones((n_x, n_y))
    q = jnp.full((n_x, n_y, km), 0.01)
    diag = prt_mass_fv3(ps, delp, {"sphum": q}, area)
    assert abs(diag["ps_mean"] - 1.0e5) < 1e-8
    expected_sphum = 0.01 * 1.0e5 / constants.g
    assert abs(diag["sphum"] - expected_sphum) / expected_sphum < 1e-10


def test_prt_gb_nh_sh_refactor_unchanged():
    """iter-705 still produces 4 lat-band means via iter-750."""
    n = 100
    lat = jnp.linspace(-jnp.pi / 2 * 0.95, jnp.pi / 2 * 0.95, n)
    a2 = jnp.full((n,), 5.0)
    area = jnp.ones((n,))
    out = prt_gb_nh_sh_fv3(a2, area, lat)
    assert abs(out["gb"] - 5.0) < 1e-10
    assert abs(out["nh"] - 5.0) < 1e-10
    assert abs(out["sh"] - 5.0) < 1e-10
    assert abs(out["eq"] - 5.0) < 1e-10


def test_iter694_5_tests_preserved():
    """Manual verification: iter-694 multi-tracer sum still correct."""
    n_x, n_y, km = 4, 4, 5
    ps = jnp.full((n_x, n_y), 1.0e5)
    delp = jnp.full((n_x, n_y, km), 2.0e4)
    area = jnp.ones((n_x, n_y))
    q1 = jnp.full((n_x, n_y, km), 0.005)
    q2 = jnp.full((n_x, n_y, km), 0.001)
    diag = prt_mass_fv3(ps, delp, {"sphum": q1, "liq_wat": q2}, area)
    expected_total = (0.005 + 0.001) * 1.0e5 / constants.g
    assert abs(diag["total_water"] - expected_total) / expected_total < 1e-10


def test_iter705_5_tests_preserved():
    """Empty bands → -1.0 sentinel still works via iter-750."""
    lat = jnp.array([0.0])
    a2 = jnp.array([3.7])
    area = jnp.array([2.0])
    out = prt_gb_nh_sh_fv3(a2, area, lat)
    assert abs(out["eq"] - 3.7) < 1e-10
    assert out["nh"] == -1.0
    assert out["sh"] == -1.0
