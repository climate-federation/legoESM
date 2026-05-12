"""FV3_3D iter 695: rh_calc_fv3 port.

Faithful JAX port of FV3 ``rh_calc`` (tools/fv_diagnostics.F90:
5309-5339).  RH = 100 · qv / qs(T, p_full).  Reuses
``legoesm.thermo.saturation_mixing_ratio``.

Tests
-----

1. ``test_rh_zero_qv``.
2. ``test_rh_saturated``.
3. ``test_rh_half_saturation``.
4. ``test_rh_shapes_3d``.
5. ``test_rh_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import thermo
from legoesm.grids.cubed_sphere import rh_calc_fv3


def test_rh_zero_qv():
    """qv = 0 → RH = 0."""
    n = 5
    p_full = jnp.full((n,), 1.0e5)
    t = jnp.full((n,), 290.0)
    qv = jnp.zeros((n,))
    rh = rh_calc_fv3(p_full, t, qv)
    assert jnp.all(jnp.abs(rh) < 1e-12)


def test_rh_saturated():
    """qv = qs → RH = 100."""
    n = 5
    p_full = jnp.full((n,), 1.0e5)
    t = jnp.full((n,), 290.0)
    qs = thermo.saturation_mixing_ratio(t, p_full)
    rh = rh_calc_fv3(p_full, t, qs)
    assert jnp.all(jnp.abs(rh - 100.0) < 1e-10)


def test_rh_half_saturation():
    """qv = 0.5·qs → RH = 50."""
    n = 5
    p_full = jnp.full((n,), 1.0e5)
    t = jnp.full((n,), 290.0)
    qs = thermo.saturation_mixing_ratio(t, p_full)
    rh = rh_calc_fv3(p_full, t, 0.5 * qs)
    assert jnp.all(jnp.abs(rh - 50.0) < 1e-10)


def test_rh_shapes_3d():
    """3-D input → 3-D output."""
    rng = np.random.default_rng(seed=695)
    n_x, n_y, km = 4, 5, 20
    p_full = jnp.asarray(rng.uniform(2.0e4, 1.0e5, size=(n_x, n_y, km)))
    t = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    qv = jnp.asarray(rng.uniform(0.0, 0.020, size=(n_x, n_y, km)))
    rh = rh_calc_fv3(p_full, t, qv)
    assert rh.shape == (n_x, n_y, km)


def test_rh_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=696)
    n_x, n_y, km = 4, 4, 20
    p_full = jnp.asarray(rng.uniform(2.0e4, 1.0e5, size=(n_x, n_y, km)))
    t = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    qv = jnp.asarray(rng.uniform(1e-8, 0.025, size=(n_x, n_y, km)))
    rh = rh_calc_fv3(p_full, t, qv)
    assert jnp.all(jnp.isfinite(rh))
