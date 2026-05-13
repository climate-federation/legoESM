"""FV3_3D iter 742: column_integral_delp_fv3 port.

Generic delp-weighted column integral.

Tests
-----

1. ``test_col_uniform_q_known_value``.
2. ``test_col_zero_field_zero_total``.
3. ``test_col_divide_by_g_flag``.
4. ``test_col_shapes_2d_3d``.
5. ``test_col_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import column_integral_delp_fv3


def test_col_uniform_q_known_value():
    """Uniform q=0.01, p_s=1e5 → col = 0.01·1e5/g ≈ 101.94 kg/m²."""
    km = 10
    field = jnp.full((km,), 0.01)
    delp = jnp.full((km,), 1.0e4)   # p_s = 1e5
    col = column_integral_delp_fv3(field, delp)
    expected = 0.01 * 1.0e5 / constants.g
    assert abs(float(col) - expected) / expected < 1e-10


def test_col_zero_field_zero_total():
    """field=0 → col=0."""
    km = 10
    field = jnp.zeros((km,))
    delp = jnp.full((km,), 1.0e4)
    col = column_integral_delp_fv3(field, delp)
    assert abs(float(col)) < 1e-15


def test_col_divide_by_g_flag():
    """divide_by_g=False → col = Σ delp·field (no /g)."""
    km = 5
    field = jnp.full((km,), 1.0)
    delp = jnp.full((km,), 1.0e4)
    col_with_g = column_integral_delp_fv3(field, delp, divide_by_g=True)
    col_no_g = column_integral_delp_fv3(field, delp, divide_by_g=False)
    expected_no_g = 5.0e4
    expected_with_g = expected_no_g / constants.g
    assert abs(float(col_no_g) - expected_no_g) < 1e-8
    assert abs(float(col_with_g) - expected_with_g) < 1e-10


def test_col_shapes_2d_3d():
    """3-D (n_x, n_y, km) → 2-D (n_x, n_y) output."""
    rng = np.random.default_rng(seed=742)
    n_x, n_y, km = 4, 5, 20
    field = jnp.asarray(rng.uniform(0.0, 0.02, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 5000.0)
    col = column_integral_delp_fv3(field, delp)
    assert col.shape == (n_x, n_y)


def test_col_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=743)
    field = jnp.asarray(rng.uniform(0.0, 0.025, size=(4, 4, 30)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, 30)))
    col = column_integral_delp_fv3(field, delp)
    assert jnp.all(jnp.isfinite(col))
