"""FV3_3D iter 770: dewpoint_depression_fv3 (T - T_d).

Composes iter-766 vapor_pressure_from_q_fv3 + iter-767 dew_point_fv3.

Tests
-----

1. ``test_depression_positive_subsat``: subsat air → T - T_d > 0.
2. ``test_depression_dry_large``: very dry air → large depression.
3. ``test_depression_monotonic_in_q``: ∂(T - T_d)/∂q < 0.
4. ``test_depression_compose_chain``: equals t - dew_point(vapor_pressure(p_mb,q)).
5. ``test_depression_shapes_3d``.
6. ``test_depression_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    dew_point_fv3,
    dewpoint_depression_fv3,
    vapor_pressure_from_q_fv3,
)


def test_depression_positive_subsat():
    """Subsaturated air: T - T_d > 0."""
    T = jnp.full((3,), 295.0)
    p = jnp.full((3,), 100_000.0)
    q = jnp.array([0.005, 0.008, 0.012])
    depr = dewpoint_depression_fv3(T, p, q)
    assert jnp.all(depr > 0.0)


def test_depression_dry_large():
    """Very dry → very large depression."""
    T = jnp.full((2,), 295.0)
    p = jnp.full((2,), 100_000.0)
    q_dry = jnp.array([1e-5, 1e-6])
    depr = dewpoint_depression_fv3(T, p, q_dry)
    # Dry air at 295 K, p=1000 mb gives T_d well below 240 K
    # → depression > 50 K
    assert jnp.all(depr > 50.0)


def test_depression_monotonic_in_q():
    """At fixed (T, p), ∂(T - T_d)/∂q < 0 (more moisture → less depression)."""
    T = jnp.full((4,), 295.0)
    p = jnp.full((4,), 100_000.0)
    q_lo = jnp.array([0.001, 0.003, 0.005, 0.010])
    q_hi = q_lo + 0.002
    depr_lo = dewpoint_depression_fv3(T, p, q_lo)
    depr_hi = dewpoint_depression_fv3(T, p, q_hi)
    assert jnp.all(depr_hi < depr_lo)


def test_depression_compose_chain():
    """Output equals t − dew_point(vapor_pressure(p_mb, q))."""
    T = jnp.array([285.0, 290.0, 295.0])
    p_pa = jnp.array([85_000.0, 95_000.0, 101_325.0])
    q = jnp.array([0.005, 0.008, 0.012])
    depr = dewpoint_depression_fv3(T, p_pa, q)
    # Manual chain
    p_mb = p_pa / 100.0
    e = vapor_pressure_from_q_fv3(p_mb, q)
    t_d = dew_point_fv3(e)
    expected = T - t_d
    np.testing.assert_allclose(np.asarray(depr), np.asarray(expected), atol=1e-12)


def test_depression_shapes_3d():
    """3-D shapes preserved."""
    rng = np.random.default_rng(seed=770)
    n_x, n_y, km = 4, 5, 20
    T = jnp.asarray(rng.uniform(240.0, 300.0, size=(n_x, n_y, km)))
    p = jnp.asarray(rng.uniform(10_000.0, 100_000.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.001, 0.020, size=(n_x, n_y, km)))
    depr = dewpoint_depression_fv3(T, p, q)
    assert depr.shape == (n_x, n_y, km)


def test_depression_finite():
    """No NaN/Inf for realistic atmospheric range."""
    rng = np.random.default_rng(seed=771)
    T = jnp.asarray(rng.uniform(220.0, 305.0, size=(8, 30)))
    p = jnp.asarray(rng.uniform(5_000.0, 105_000.0, size=(8, 30)))
    q = jnp.asarray(rng.uniform(1e-6, 0.025, size=(8, 30)))
    depr = dewpoint_depression_fv3(T, p, q)
    assert jnp.all(jnp.isfinite(depr))
