"""FV3_3D iter 744: internal_energy_column_fv3 port.

IE = sum_k delp * cv * pt / g  (column integral).

Tests
-----

1. ``test_ie_isothermal_dry_known_value``.
2. ``test_ie_zero_T_zero``.
3. ``test_ie_custom_cv_moist``.
4. ``test_ie_shapes_3d``.
5. ``test_ie_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    internal_energy_column_fv3,
    moist_cv_fv3,
)


def test_ie_isothermal_dry_known_value():
    """Dry isothermal column at p_s=1e5, T=280 → IE = cv·T·p_s/g."""
    km = 10
    pt = jnp.full((km,), 280.0)
    delp = jnp.full((km,), 1.0e4)   # p_s = 1e5
    ie = internal_energy_column_fv3(pt, delp)
    cv = constants.c_pd - constants.R_d
    expected = cv * 280.0 * 1.0e5 / constants.g
    assert abs(float(ie) - expected) / expected < 1e-10


def test_ie_zero_T_zero():
    """T = 0 → IE = 0."""
    km = 5
    pt = jnp.zeros((km,))
    delp = jnp.full((km,), 1.0e4)
    ie = internal_energy_column_fv3(pt, delp)
    assert abs(float(ie)) < 1e-15


def test_ie_custom_cv_moist():
    """Custom layer-varying cv (from iter-713 moist_cv_fv3) works."""
    km = 5
    pt = jnp.full((km,), 280.0)
    delp = jnp.full((km,), 5.0e3)
    q = jnp.full((km,), 0.01)
    cvm, _ = moist_cv_fv3(q_sphum=q)
    ie = internal_energy_column_fv3(pt, delp, cv=cvm)
    # Each layer: cvm·280·5e3/g
    expected_per_layer = float(cvm[0]) * 280.0 * 5.0e3 / constants.g
    expected = 5 * expected_per_layer
    assert abs(float(ie) - expected) / expected < 1e-10


def test_ie_shapes_3d():
    """3-D (n_x, n_y, km) → 2-D (n_x, n_y) output."""
    rng = np.random.default_rng(seed=744)
    n_x, n_y, km = 4, 5, 20
    pt = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    ie = internal_energy_column_fv3(pt, delp)
    assert ie.shape == (n_x, n_y)


def test_ie_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=745)
    pt = jnp.asarray(rng.uniform(200.0, 320.0, size=(4, 4, 30)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, 30)))
    ie = internal_energy_column_fv3(pt, delp)
    assert jnp.all(jnp.isfinite(ie))
