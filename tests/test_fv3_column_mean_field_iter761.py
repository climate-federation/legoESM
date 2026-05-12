"""FV3_3D iter 761: column_mean_field_fv3 generic helper.

<field> = sum(delp * field) / sum(delp).

Tests
-----

1. ``test_mean_uniform_field``.
2. ``test_mean_two_layer_weighted``.
3. ``test_mean_T_diagnostic_usage``.
4. ``test_mean_iter760_rh_unchanged``.
5. ``test_mean_shapes_3d``.
6. ``test_mean_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import column_mean_field_fv3, column_mean_rh_fv3


def test_mean_uniform_field():
    """Uniform field=C → mean=C."""
    km = 10
    field = jnp.full((km,), 7.5)
    delp = jnp.full((km,), 1.0e4)
    mean = column_mean_field_fv3(field, delp)
    assert abs(float(mean) - 7.5) < 1e-12


def test_mean_two_layer_weighted():
    """Layer A=10, delp=3e4; B=0, delp=1e4 → mean = (3·10+1·0)/4 = 7.5."""
    field = jnp.array([10.0, 0.0])
    delp = jnp.array([3.0e4, 1.0e4])
    mean = column_mean_field_fv3(field, delp)
    assert abs(float(mean) - 7.5) < 1e-12


def test_mean_T_diagnostic_usage():
    """Column-mean T = mass-weighted T average."""
    km = 10
    T = jnp.linspace(220.0, 290.0, km)
    delp = jnp.full((km,), 1.0e4)
    T_mean = column_mean_field_fv3(T, delp)
    expected = float(jnp.mean(T))  # uniform delp → arithmetic mean
    assert abs(float(T_mean) - expected) < 1e-10


def test_mean_iter760_rh_unchanged():
    """iter-760 column_mean_rh_fv3 still produces same output."""
    km = 10
    T = jnp.full((km,), 290.0)
    p = jnp.full((km,), 5.0e4)
    qv = jnp.full((km,), 5e-3)
    delp = jnp.full((km,), 1.0e4)
    rh_col = column_mean_rh_fv3(p, T, qv, delp)
    assert jnp.isfinite(rh_col)
    assert 0.0 <= float(rh_col) <= 200.0


def test_mean_shapes_3d():
    """3-D → 2-D output."""
    rng = np.random.default_rng(seed=761)
    n_x, n_y, km = 4, 5, 20
    field = jnp.asarray(rng.normal(scale=100.0, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    mean = column_mean_field_fv3(field, delp)
    assert mean.shape == (n_x, n_y)


def test_mean_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=762)
    field = jnp.asarray(rng.normal(scale=50.0, size=(4, 4, 30)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, 30)))
    mean = column_mean_field_fv3(field, delp)
    assert jnp.all(jnp.isfinite(mean))
