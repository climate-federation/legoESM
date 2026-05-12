"""FV3_3D iter 804: frost_point_depression_fv3 (T − T_frost).

Composes iter-803 ``frost_point_temperature_fv3``.

Tests
-----

1. ``test_fpd_subice_positive``: subice-sat air → T − T_frost > 0.
2. ``test_fpd_at_frost_zero``: T = T_frost → depression ≈ 0.
3. ``test_fpd_less_than_dew_depression``: below freezing T_frost > T_dew
   so frost depression < dew depression.
4. ``test_fpd_monotonic_in_q``: ↑q → ↓depression.
5. ``test_fpd_compose_chain``: equals t − T_frost(p, q).
6. ``test_fpd_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    dewpoint_depression_fv3,
    frost_point_depression_fv3,
    frost_point_temperature_fv3,
)


def test_fpd_subice_positive():
    """Cold subice-saturated air → T − T_frost > 0."""
    T = jnp.array([240.0, 250.0, 260.0])
    p = jnp.full((3,), 30_000.0)
    q = jnp.array([1e-5, 1e-4, 1e-3])
    depr = frost_point_depression_fv3(T, p, q)
    assert jnp.all(depr > 0.0)


def test_fpd_at_frost_zero():
    """T = T_frost → depression ≈ 0 (close to numerical zero)."""
    p = jnp.array([50_000.0])
    q = jnp.array([1e-4])
    t_frost = frost_point_temperature_fv3(p, q)
    depr = frost_point_depression_fv3(t_frost, p, q)
    np.testing.assert_allclose(np.asarray(depr), [0.0], atol=1e-12)


def test_fpd_less_than_dew_depression():
    """Below freezing: T - T_frost < T - T_dew at same q."""
    T = jnp.array([245.0])
    p = jnp.array([40_000.0])
    q = jnp.array([1e-4])
    fpd = frost_point_depression_fv3(T, p, q)
    dpd = dewpoint_depression_fv3(T, p, q)
    assert float(fpd[0]) < float(dpd[0])


def test_fpd_monotonic_in_q():
    """↑q → ↓frost-point depression at fixed (T, p)."""
    T = jnp.full((4,), 230.0)
    p = jnp.full((4,), 30_000.0)
    q_lo = jnp.array([1e-6, 1e-5, 5e-5, 1e-4])
    q_hi = q_lo * 2.0
    fpd_lo = frost_point_depression_fv3(T, p, q_lo)
    fpd_hi = frost_point_depression_fv3(T, p, q_hi)
    assert jnp.all(fpd_hi < fpd_lo)


def test_fpd_compose_chain():
    """Output equals t − T_frost(p, q)."""
    T = jnp.array([225.0, 240.0, 255.0])
    p = jnp.array([20_000.0, 40_000.0, 60_000.0])
    q = jnp.array([1e-5, 1e-4, 1e-3])
    fpd = frost_point_depression_fv3(T, p, q)
    t_frost = frost_point_temperature_fv3(p, q)
    expected = T - t_frost
    np.testing.assert_allclose(np.asarray(fpd), np.asarray(expected), atol=1e-12)


def test_fpd_shapes_3d_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=804)
    n_x, n_y, km = 4, 5, 20
    T = jnp.asarray(rng.uniform(200.0, 270.0, size=(n_x, n_y, km)))
    p = jnp.asarray(rng.uniform(10_000.0, 80_000.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(1e-7, 0.005, size=(n_x, n_y, km)))
    depr = frost_point_depression_fv3(T, p, q)
    assert depr.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(depr))
