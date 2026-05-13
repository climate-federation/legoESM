"""FV3_3D iter 762: saturation_deficit_column_fv3 port.

SatDef = Q_sat_col - PWV.

Tests
-----

1. ``test_satdef_saturated_zero``.
2. ``test_satdef_dry_atmosphere_positive``.
3. ``test_satdef_do_cmip_branch``.
4. ``test_satdef_shapes_3d``.
5. ``test_satdef_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants, thermo
from legoesm.grids.cubed_sphere import saturation_deficit_column_fv3


def test_satdef_saturated_zero():
    """qv = q_sat everywhere → SatDef = 0."""
    km = 10
    T = jnp.full((km,), 290.0)
    p = jnp.full((km,), 5.0e4)
    qv = thermo.saturation_mixing_ratio(T, p)
    delp = jnp.full((km,), 1.0e4)
    sat_def = saturation_deficit_column_fv3(p, T, qv, delp)
    assert abs(float(sat_def)) < 1e-10


def test_satdef_dry_atmosphere_positive():
    """qv=0 (dry) → SatDef = full column q_sat."""
    km = 10
    T = jnp.full((km,), 290.0)
    p = jnp.full((km,), 5.0e4)
    qv = jnp.zeros((km,))
    delp = jnp.full((km,), 1.0e4)
    sat_def = saturation_deficit_column_fv3(p, T, qv, delp)
    qs = thermo.saturation_mixing_ratio(T, p)
    expected = float(jnp.sum(delp * qs)) / constants.g
    assert abs(float(sat_def) - expected) / expected < 1e-10


def test_satdef_do_cmip_branch():
    """do_cmip differs from liquid-only at sub-freezing T."""
    km = 5
    T = jnp.full((km,), constants.T_freeze - 10.0)
    p = jnp.full((km,), 5.0e4)
    qv = jnp.full((km,), 1e-5)
    delp = jnp.full((km,), 1.0e4)
    sat_def_liq = saturation_deficit_column_fv3(p, T, qv, delp, do_cmip=False)
    sat_def_cmip = saturation_deficit_column_fv3(p, T, qv, delp, do_cmip=True)
    assert abs(float(sat_def_liq) - float(sat_def_cmip)) > 1e-8


def test_satdef_shapes_3d():
    """3-D → 2-D output."""
    rng = np.random.default_rng(seed=762)
    n_x, n_y, km = 4, 5, 20
    p = jnp.asarray(rng.uniform(1.0e4, 1.0e5, size=(n_x, n_y, km)))
    T = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    qv = jnp.asarray(rng.uniform(0.0, 0.005, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    sat_def = saturation_deficit_column_fv3(p, T, qv, delp)
    assert sat_def.shape == (n_x, n_y)


def test_satdef_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=763)
    p = jnp.asarray(rng.uniform(1.0e4, 1.0e5, size=(4, 4, 30)))
    T = jnp.asarray(rng.uniform(220.0, 300.0, size=(4, 4, 30)))
    qv = jnp.asarray(rng.uniform(0.0, 0.020, size=(4, 4, 30)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, 30)))
    sat_def = saturation_deficit_column_fv3(p, T, qv, delp)
    assert jnp.all(jnp.isfinite(sat_def))
