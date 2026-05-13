"""FV3_3D iter 760: column_mean_rh_fv3 port.

RH_col = sum(delp * RH) / sum(delp).

Tests
-----

1. ``test_rh_col_uniform_layer_rh``.
2. ``test_rh_col_known_value``.
3. ``test_rh_col_do_cmip_branch``.
4. ``test_rh_col_shapes_3d``.
5. ``test_rh_col_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants, thermo
from legoesm.grids.cubed_sphere import column_mean_rh_fv3


def test_rh_col_uniform_layer_rh():
    """Uniform per-layer RH=50 → column RH=50 (mass-weighted average)."""
    km = 10
    T = jnp.full((km,), 290.0)
    p = jnp.full((km,), 5.0e4)
    qs = thermo.saturation_mixing_ratio(T, p)
    qv = 0.5 * qs   # RH = 50% per layer
    delp = jnp.full((km,), 1.0e4)
    rh_col = column_mean_rh_fv3(p, T, qv, delp)
    assert abs(float(rh_col) - 50.0) < 1e-10


def test_rh_col_known_value():
    """Two layers: RH=80 + RH=20, equal delp → mean=50."""
    T = jnp.full((2,), 290.0)
    p = jnp.full((2,), 5.0e4)
    qs = thermo.saturation_mixing_ratio(T, p)
    qv = jnp.array([0.8 * float(qs[0]), 0.2 * float(qs[1])])
    delp = jnp.full((2,), 1.0e4)
    rh_col = column_mean_rh_fv3(p, T, qv, delp)
    assert abs(float(rh_col) - 50.0) < 1e-10


def test_rh_col_do_cmip_branch():
    """do_cmip=True uses blended sat → different from liquid-only."""
    km = 5
    T = jnp.full((km,), constants.T_freeze - 10.0)
    p = jnp.full((km,), 5.0e4)
    qv = jnp.full((km,), 1e-4)
    delp = jnp.full((km,), 1.0e4)
    rh_liq = column_mean_rh_fv3(p, T, qv, delp, do_cmip=False)
    rh_cmip = column_mean_rh_fv3(p, T, qv, delp, do_cmip=True)
    assert abs(float(rh_liq) - float(rh_cmip)) > 1e-6


def test_rh_col_shapes_3d():
    """3-D → 2-D output."""
    rng = np.random.default_rng(seed=760)
    n_x, n_y, km = 4, 5, 20
    p = jnp.asarray(rng.uniform(1.0e4, 1.0e5, size=(n_x, n_y, km)))
    T = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    qv = jnp.asarray(rng.uniform(0.0, 0.020, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    rh_col = column_mean_rh_fv3(p, T, qv, delp)
    assert rh_col.shape == (n_x, n_y)


def test_rh_col_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=761)
    p = jnp.asarray(rng.uniform(1.0e4, 1.0e5, size=(4, 4, 30)))
    T = jnp.asarray(rng.uniform(220.0, 300.0, size=(4, 4, 30)))
    qv = jnp.asarray(rng.uniform(1e-8, 0.025, size=(4, 4, 30)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, 30)))
    rh_col = column_mean_rh_fv3(p, T, qv, delp)
    assert jnp.all(jnp.isfinite(rh_col))
