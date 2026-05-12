"""FV3_3D iter 712: pv_entropy_fv3 PPME upgrade.

Iter-691 ``pv_entropy_fv3`` used 2nd-order linear edge average.
iter-712 adds opt-in ``use_ppme=True`` that switches to iter-711
``ppme_fv3`` (FV3-faithful, matches FV3 source which explicitly
calls ppme).

Tests
-----

1. ``test_pv_entropy_ppme_uniform``.
2. ``test_pv_entropy_ppme_isothermal_zero_epv``.
3. ``test_pv_entropy_ppme_changes_with_flag``.
4. ``test_pv_entropy_ppme_default_matches_iter691``.
5. ``test_pv_entropy_ppme_shapes_3d``.
6. ``test_pv_entropy_ppme_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import pv_entropy_fv3


def test_pv_entropy_ppme_uniform():
    """Uniform θ → EPV = 0 (both paths)."""
    km = 10
    theta = jnp.full((km,), 300.0)
    delp = jnp.full((km,), 1000.0)
    vort = jnp.zeros((km,))
    f_d = jnp.asarray(1e-4)
    epv_lin = pv_entropy_fv3(vort, f_d, theta, delp, use_ppme=False)
    epv_ppme = pv_entropy_fv3(vort, f_d, theta, delp, use_ppme=True)
    assert jnp.all(jnp.abs(epv_lin) < 1e-12)
    assert jnp.all(jnp.abs(epv_ppme) < 1e-12)


def test_pv_entropy_ppme_isothermal_zero_epv():
    """ζ + f = 0 → EPV = 0 regardless of θ profile or flag."""
    km = 15
    theta = jnp.linspace(280.0, 320.0, km)
    delp = jnp.full((km,), 1000.0)
    vort = jnp.zeros((km,))
    f_d = jnp.asarray(0.0)
    epv = pv_entropy_fv3(vort, f_d, theta, delp, use_ppme=True)
    assert jnp.all(jnp.abs(epv) < 1e-12)


def test_pv_entropy_ppme_changes_with_flag():
    """For a non-linear θ profile, PPME edges differ from linear avg
    → EPV differs noticeably."""
    rng = np.random.default_rng(seed=712)
    km = 20
    theta = jnp.asarray(rng.uniform(280.0, 350.0, size=(km,)))
    delp = jnp.asarray(rng.uniform(800.0, 1200.0, size=(km,)))
    vort = jnp.asarray(rng.normal(scale=1e-4, size=(km,)))
    f_d = jnp.asarray(1e-4)
    epv_lin = pv_entropy_fv3(vort, f_d, theta, delp, use_ppme=False)
    epv_ppme = pv_entropy_fv3(vort, f_d, theta, delp, use_ppme=True)
    diff = float(jnp.max(jnp.abs(epv_lin - epv_ppme)))
    assert diff > 1e-12


def test_pv_entropy_ppme_default_matches_iter691():
    """Default use_ppme=False keeps iter-691 behavior (backward-compat)."""
    rng = np.random.default_rng(seed=713)
    km = 20
    theta = jnp.asarray(rng.uniform(280.0, 320.0, size=(km,)))
    delp = jnp.full((km,), 1000.0)
    vort = jnp.asarray(rng.normal(scale=1e-4, size=(km,)))
    f_d = jnp.asarray(1e-4)
    epv_default = pv_entropy_fv3(vort, f_d, theta, delp)
    epv_explicit = pv_entropy_fv3(vort, f_d, theta, delp, use_ppme=False)
    assert jnp.allclose(epv_default, epv_explicit, atol=1e-15)


def test_pv_entropy_ppme_shapes_3d():
    """3-D input → 3-D EPV."""
    rng = np.random.default_rng(seed=714)
    n_x, n_y, km = 4, 5, 20
    theta = jnp.asarray(rng.uniform(280.0, 350.0, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    vort = jnp.asarray(rng.normal(scale=1e-4, size=(n_x, n_y, km)))
    f_d = jnp.asarray(rng.uniform(-1.5e-4, 1.5e-4, size=(n_x, n_y)))
    epv = pv_entropy_fv3(vort, f_d, theta, delp, use_ppme=True)
    assert epv.shape == (n_x, n_y, km)


def test_pv_entropy_ppme_finite():
    """No NaN/Inf with PPME on random inputs."""
    rng = np.random.default_rng(seed=715)
    n_x, n_y, km = 4, 4, 30
    theta = jnp.asarray(rng.uniform(280.0, 360.0, size=(n_x, n_y, km)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(n_x, n_y, km)))
    vort = jnp.asarray(rng.normal(scale=1e-4, size=(n_x, n_y, km)))
    f_d = jnp.asarray(rng.uniform(-1.5e-4, 1.5e-4, size=(n_x, n_y)))
    epv = pv_entropy_fv3(vort, f_d, theta, delp, use_ppme=True)
    assert jnp.all(jnp.isfinite(epv))
