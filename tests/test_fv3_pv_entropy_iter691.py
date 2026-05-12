"""FV3_3D iter 691: pv_entropy_fv3 port.

Faithful JAX port of FV3 ``pv_entropy`` (tools/fv_diagnostics.F90:
5111-5193).  Ertel potential vorticity diagnostic.

Tests
-----

1. ``test_pv_zero_vort_zero_f``.
2. ``test_pv_isothermal_d_theta_zero``.
3. ``test_pv_uniform_stratification``.
4. ``test_pv_shapes_3d``.
5. ``test_pv_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import pv_entropy_fv3


def test_pv_zero_vort_zero_f():
    """ζ + f = 0 → EPV = 0."""
    km = 20
    theta = jnp.linspace(280.0, 320.0, km)
    delp = jnp.full((km,), 1000.0)
    vort = jnp.zeros((km,))
    f_d = jnp.asarray(0.0)
    epv = pv_entropy_fv3(vort, f_d, theta, delp)
    assert jnp.all(jnp.abs(epv) < 1e-15)


def test_pv_isothermal_d_theta_zero():
    """Constant θ → d_theta = 0 → EPV = 0."""
    km = 20
    theta = jnp.full((km,), 300.0)
    delp = jnp.full((km,), 1000.0)
    vort = jnp.full((km,), 0.001)
    f_d = jnp.asarray(1e-4)
    epv = pv_entropy_fv3(vort, f_d, theta, delp)
    assert jnp.all(jnp.abs(epv) < 1e-15)


def test_pv_uniform_stratification():
    """Linear θ(k) → expected interior EPV value.

    With theta[k] linear in k (e.g. theta = 280 + k·dθ), edge
    averages give theta_edges[k] = theta[k-1]+theta[k])/2.
    For interior layers:
       d_theta[k] = theta_edges[k] - theta_edges[k+1]
                  = (θ[k-1] - θ[k+1])/2 = -dθ  (positive if θ
                  increases upward in FV3 indexing — k=0 top, but
                  here we use top-down: k=0 top has largest θ if
                  stable stratification).

    Use 1-D test column with monotonically decreasing θ from top
    (k=0) to bottom (k=km-1).  Expect EPV finite, single-signed
    in interior.
    """
    km = 30
    theta = jnp.linspace(320.0, 280.0, km)  # k=0 top (320), k=29 bot (280)
    delp = jnp.full((km,), 500.0)
    vort = jnp.zeros((km,))
    f_d = jnp.asarray(1e-4)
    epv = pv_entropy_fv3(vort, f_d, theta, delp)
    # Interior layers: d_theta should be (dθ) > 0 (θ_top - θ_bot)
    interior_epv = epv[1:-1]
    assert jnp.all(interior_epv > 0.0)


def test_pv_shapes_3d():
    """3-D input → 3-D EPV shape."""
    rng = np.random.default_rng(seed=691)
    n_x, n_y, km = 4, 5, 20
    theta = jnp.asarray(rng.uniform(280, 350, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    vort = jnp.asarray(rng.normal(scale=1e-4, size=(n_x, n_y, km)))
    f_d = jnp.asarray(rng.uniform(-1.5e-4, 1.5e-4, size=(n_x, n_y)))
    epv = pv_entropy_fv3(vort, f_d, theta, delp)
    assert epv.shape == (n_x, n_y, km)


def test_pv_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=692)
    km = 30
    theta = jnp.asarray(rng.uniform(280, 350, size=(4, 4, km)))
    delp = jnp.asarray(rng.uniform(100, 2000, size=(4, 4, km)))
    vort = jnp.asarray(rng.normal(scale=1e-4, size=(4, 4, km)))
    f_d = jnp.asarray(rng.uniform(-1.5e-4, 1.5e-4, size=(4, 4)))
    epv = pv_entropy_fv3(vort, f_d, theta, delp)
    assert jnp.all(jnp.isfinite(epv))
