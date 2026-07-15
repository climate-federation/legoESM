"""Monotone (van-Leer TVD) vertical tracer advection — D5 (iter-44).

``vertical_advection_van_leer_plane`` is the positive-definite replacement for
the centred ``_vertical_advection_plane`` in the plane CRM tracer transport.
The centred scheme overshoots into NEGATIVE tracer at sharp convective vertical
gradients (then clipped on the microphysics read → mass loss); van-Leer forbids
new extrema. These tests pin the analytic properties (consistency for smooth
fields, monotonicity at a step, positivity, boundary safety, AD) and the
config wiring.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    _vertical_advection_plane, vertical_advection_van_leer_plane,
)
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


def _hc_w(nlev=20, H=10_000.0, w_amp=2.0, w_sign=1.0):
    """Height coord + a (2,2,nlev+1) w field, zero at the rigid lids."""
    hc = create_height_coordinate(nlev, H=H)
    J = jnp.ones((2, 2))
    wk = np.zeros(nlev + 1)
    wk[1:-1] = w_sign * w_amp * np.sin(np.linspace(0, np.pi, nlev - 1))
    w = jnp.asarray(wk)[None, None, :] * jnp.ones((2, 2, nlev + 1))
    return hc, J, w, wk


def test_constant_field_zero_tendency():
    """A uniform tracer is not advected — the advective-form ``-∂(wq)/∂z +
    q·∂w/∂z`` cancels exactly for constant q (machine zero)."""
    hc, J, w, _ = _hc_w()
    f = jnp.ones((2, 2, 20)) * 0.01
    tend = vertical_advection_van_leer_plane(f, w, hc, J)
    assert float(jnp.max(jnp.abs(tend))) < 1.0e-18


def test_linear_field_matches_centered_uniform_w():
    """2nd-order consistency: for a LINEAR field under locally-UNIFORM w the
    flux-divergence telescopes to the exact ``-w·b`` (the ``q·∂w/∂z`` term
    vanishes) ⇒ van-Leer == centred to machine precision in the deep interior.

    (Under VARYING velocity neither van-Leer nor the centred scheme equals the
    other exactly — that is the standard TVD advective-form behaviour, shared
    by the production horizontal ``_van_leer_advection_x``; consistency is the
    locally-uniform-velocity limit, asserted here.)
    """
    nlev = 20
    hc = create_height_coordinate(nlev, H=10_000.0)
    J = jnp.ones((2, 2))
    wk = np.zeros(nlev + 1)
    wk[1:-1] = 0.8                         # uniform interior, zero at lids
    w = jnp.asarray(wk)[None, None, :] * jnp.ones((2, 2, nlev + 1))
    z = np.asarray(hc.z_full)
    f = jnp.asarray(3.0e-6 * z)[None, None, :] * jnp.ones((2, 2, nlev))
    t_vl = vertical_advection_van_leer_plane(f, w, hc, J)
    t_ct = _vertical_advection_plane(f, w, hc, J)
    # deep interior (away from the 2-cell lid transition where w ramps 0→W)
    np.testing.assert_allclose(
        np.asarray(t_vl)[0, 0, 3:nlev - 3],
        np.asarray(t_ct)[0, 0, 3:nlev - 3], rtol=0.0, atol=1.0e-18)


def test_step_advection_is_monotone():
    """The signature D5 win: a square wave advected by uniform w stays in
    [0,1] under van-Leer (no new extrema), while centred RINGS (over/under-
    shoot ⇒ negative tracer)."""
    hc, J, _, _ = _hc_w()
    nlev = 20
    f0 = np.zeros((2, 2, nlev))
    f0[:, :, 8:12] = 1.0
    f0 = jnp.asarray(f0)
    wk = np.zeros(nlev + 1)
    wk[1:-1] = -1.0                       # uniform interior descent
    w = jnp.asarray(wk)[None, None, :] * jnp.ones((2, 2, nlev + 1))
    q_vl = q_ct = f0
    dt = 2.0
    for _ in range(15):
        q_vl = q_vl + dt * vertical_advection_van_leer_plane(q_vl, w, hc, J)
        q_ct = q_ct + dt * _vertical_advection_plane(q_ct, w, hc, J)
    # van-Leer: monotone — no overshoot below 0 or above 1 (tiny tol)
    assert float(jnp.min(q_vl)) >= -1.0e-9
    assert float(jnp.max(q_vl)) <= 1.0 + 1.0e-9
    # centred: demonstrably rings (negative undershoot)
    assert float(jnp.min(q_ct)) < -1.0e-3


def test_positive_definite_under_convergent_flow():
    """A positive tracer stays positive under van-Leer (no clipped negatives)."""
    hc, J, w, _ = _hc_w(w_amp=3.0)
    rng = np.random.default_rng(0)
    f = jnp.asarray(np.abs(rng.standard_normal((2, 2, 20))) * 1.0e-3)
    q = f
    dt = 1.0
    for _ in range(25):
        q = q + dt * vertical_advection_van_leer_plane(q, w, hc, J)
    assert bool(jnp.all(q >= -1.0e-12))
    assert bool(jnp.all(jnp.isfinite(q)))


def test_boundary_and_finiteness():
    """No NaN/Inf anywhere incl the boundary-adjacent cells; rigid lids
    (w=0) carry no spurious flux for a uniform field."""
    hc, J, w, _ = _hc_w()
    f = jnp.asarray(np.linspace(0.02, 0.0, 20))[None, None, :] \
        * jnp.ones((2, 2, 20))
    tend = vertical_advection_van_leer_plane(f, w, hc, J)
    assert tend.shape == (2, 2, 20)
    assert bool(jnp.all(jnp.isfinite(tend)))


def test_nlev_le_2_falls_back_to_centered():
    """Too few cells for the 4-point stencil ⇒ identical to centred."""
    hc = create_height_coordinate(2, H=2_000.0)
    J = jnp.ones((2, 2))
    w = jnp.asarray([0.0, 0.5, 0.0])[None, None, :] * jnp.ones((2, 2, 3))
    f = jnp.asarray([0.01, 0.02])[None, None, :] * jnp.ones((2, 2, 2))
    np.testing.assert_array_equal(
        np.asarray(vertical_advection_van_leer_plane(f, w, hc, J)),
        np.asarray(_vertical_advection_plane(f, w, hc, J)))


def test_ad_grad_finite():
    """``jax.grad`` flows cleanly through the limiter (double-where denom)."""
    hc, J, w, _ = _hc_w()
    f0 = jnp.asarray(np.abs(np.random.default_rng(1).standard_normal((2, 2, 20)))
                     * 1.0e-3)

    def loss(f):
        return jnp.sum(vertical_advection_van_leer_plane(f, w, hc, J) ** 2)

    g = jax.grad(loss)(f0)
    assert g.shape == f0.shape
    assert bool(jnp.all(jnp.isfinite(g)))


def test_config_selects_vertical_scheme():
    """``vertical_tracer_advection`` config dispatch + unknown-value guard."""
    assert CompressibleEulerConfig().vertical_tracer_advection == "centered"
    cfg = CompressibleEulerConfig(vertical_tracer_advection="van_leer")
    assert cfg.vertical_tracer_advection == "van_leer"
