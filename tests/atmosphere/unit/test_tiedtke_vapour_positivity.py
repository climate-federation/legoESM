"""Tiedtke's convective vapour sink must not drive q_v negative.

The compensating-subsidence sink (M/rho)*dq_dz had no q_v/dt bound and drove
q_v to -6.8e-5 kg/kg at a near-dry level, failing SCM-RCE at all SSTs. The
column positivity scale caps it. These checks pin the guarantee and its
no-op-when-stable property, on a synthetic near-dry column where the raw sink
would overshoot.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.convection.tiedtke import (
    TiedtkeConfig, tiedtke_convection,
)
from legoesm import constants


def _column(nlev=40, dry=True):
    p_half = np.linspace(5_000.0, 100_000.0, nlev + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    sigma = p_full / p_full[-1]
    T = 200.0 + 100.0 * sigma ** 0.5
    # A conditionally unstable, upper-dry sounding: q_v ~ 1e-4 aloft.
    q_v = (0.018 if dry else 0.02) * sigma ** 4 + 1e-4
    return (jnp.asarray(T)[None, :], jnp.asarray(q_v)[None, :],
            jnp.asarray(p_full)[None, :], jnp.asarray(p_half)[None, :])


def _run(dt=600.0):
    T, q_v, p_full, p_half = _column()
    cfg = TiedtkeConfig()
    z = jnp.zeros_like(T)
    out = tiedtke_convection(
        T=T, q_v=q_v, p_full=p_full, p_half=p_half, u=z, v=z,
        conv_prog_profile=z, dt=dt, config=cfg)
    out = out if hasattr(out, "dq_v_dt") else out[0]
    return q_v, out, dt


def test_vapour_stays_nonnegative_over_the_step():
    q_v, out, dt = _run()
    q_new = np.asarray(q_v + dt * out.dq_v_dt)
    assert q_new.min() >= -1e-12, (
        f"tiedtke drove q_v negative: min={q_new.min():.3e}")
    assert np.abs(np.asarray(out.dq_v_dt)).max() > 1e-10, (
        "no convective tendency -- a zero-returning limiter would pass "
        "positivity vacuously")


def test_limiter_is_a_noop_when_no_level_overshoots():
    """On the SAME convecting column but a tiny dt, no level can go negative, so
    the positivity scale is 1 and the tendency is the raw kernel output.

    Proven non-vacuously: the tendency is compared against an even-tinier-dt run
    (both non-binding, so the RATE must be identical), and it must be
    substantially non-zero -- a limiter that scaled every tendency to zero would
    give an identical (zero) result and fail the magnitude check.
    """
    T, q_v, p_full, p_half = _column()   # the convecting column
    z = jnp.zeros_like(T)

    def run(dt):
        out = tiedtke_convection(
            T=T, q_v=q_v, p_full=p_full, p_half=p_half, u=z, v=z,
            conv_prog_profile=z, dt=dt, config=TiedtkeConfig())
        return out if hasattr(out, "dq_v_dt") else out[0]

    o1 = run(1.0)        # dt*|dq_v_dt| << q_v -> positivity cannot bind
    q_new = np.asarray(q_v + 1.0 * o1.dq_v_dt)
    assert q_new.min() >= 0.0
    assert np.abs(np.asarray(o1.dq_v_dt)).max() > 1e-10, (
        "tendency is ~zero: a scale-to-zero limiter would pass vacuously")
    # dq_v_dt is a RATE: with the limiter inactive it must not depend on dt.
    o2 = run(0.25)
    np.testing.assert_allclose(
        np.asarray(o1.dq_v_dt), np.asarray(o2.dq_v_dt), rtol=1e-9,
        err_msg="tendency changed with dt in the non-binding regime -- the "
                "limiter is scaling when it should be a no-op")


def test_tendency_is_finite_and_differentiable():
    T, q_v, p_full, p_half = _column()
    def loss(qv):
        z = jnp.zeros_like(T)
        out = tiedtke_convection(
            T=T, q_v=qv, p_full=p_full, p_half=p_half, u=z, v=z,
            conv_prog_profile=z, dt=600.0, config=TiedtkeConfig())
        out = out if hasattr(out, "dq_v_dt") else out[0]
        return jnp.sum(out.dq_v_dt ** 2)
    g = jax.grad(loss)(q_v)
    assert bool(jnp.all(jnp.isfinite(g))), "non-finite gradient through limiter"
