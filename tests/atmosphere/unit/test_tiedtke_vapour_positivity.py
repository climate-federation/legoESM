"""Tiedtke's convective vapour sink must not drive q_v negative.

The compensating-subsidence sink (M/rho)*dq_dz had no q_v/dt bound and drove
q_v to -6.8e-5 kg/kg at a near-dry level, failing SCM-RCE at all SSTs. Most
checks target the pure limiter ``vapour_positivity_column_scale`` with KNOWN
inputs -- the non-vacuous way to prove the logic, since the full scheme's
tendency depends on dt through its own mass-flux relaxation. One end-to-end
check confirms the scheme itself keeps q_v >= 0.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.convection.tiedtke import (
    TiedtkeConfig, tiedtke_convection, vapour_positivity_column_scale,
)


def test_scale_is_exactly_one_when_no_level_overshoots():
    q_v = jnp.asarray([[1e-2, 5e-3, 1e-3, 1e-4]])
    dq_v_dt = jnp.asarray([[-1e-6, -1e-7, 0.0, 1e-7]])  # weak sinks + a source
    scale = vapour_positivity_column_scale(q_v, dq_v_dt, dt=600.0)
    np.testing.assert_allclose(np.asarray(scale), 1.0, rtol=0, atol=0)


def test_scale_binds_to_the_tightest_level():
    # Level 3: q_v=1e-4, sink wants dt*1e-6=6e-4 -> ratio 1e-4/6e-4 = 1/6.
    q_v = jnp.asarray([[1e-2, 1e-4]])
    dq_v_dt = jnp.asarray([[-1e-8, -1e-6]])
    scale = float(vapour_positivity_column_scale(q_v, dq_v_dt, dt=600.0)[0, 0])
    assert scale == pytest.approx(1e-4 / (600.0 * 1e-6), rel=1e-9)
    # And it enforces positivity exactly at that level.
    q_new = np.asarray(q_v)[0] + 600.0 * scale * np.asarray(dq_v_dt)[0]
    assert q_new.min() >= -1e-18


def test_zero_vapour_source_level_does_not_zero_the_column():
    # A source (dq_v_dt >= 0) at q_v == 0 must NOT constrain the column
    # (the codex edge case): scale stays governed by the real sink elsewhere.
    q_v = jnp.asarray([[1e-2, 0.0, 1e-3]])
    dq_v_dt = jnp.asarray([[-1e-8, 1e-7, -1e-9]])  # middle level is a source
    scale = float(vapour_positivity_column_scale(q_v, dq_v_dt, dt=600.0)[0, 0])
    assert scale == pytest.approx(1.0, rel=1e-9)


def test_scale_is_differentiable():
    q_v = jnp.asarray([[1e-2, 1e-4]])
    dq = jnp.asarray([[-1e-8, -1e-6]])
    g = jax.grad(lambda x: float(
        vapour_positivity_column_scale(q_v, x, 600.0).sum()) if False
        else vapour_positivity_column_scale(q_v, x, 600.0).sum())(dq)
    assert bool(jnp.all(jnp.isfinite(g)))


def _column(nlev=40):
    p_half = np.linspace(5_000.0, 100_000.0, nlev + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    sigma = p_full / p_full[-1]
    T = 200.0 + 100.0 * sigma ** 0.5
    q_v = 0.018 * sigma ** 4 + 1e-4
    return (jnp.asarray(T)[None, :], jnp.asarray(q_v)[None, :],
            jnp.asarray(p_full)[None, :], jnp.asarray(p_half)[None, :])


def test_scheme_keeps_q_v_nonnegative_end_to_end():
    T, q_v, p_full, p_half = _column()
    z = jnp.zeros_like(T)
    out = tiedtke_convection(
        T=T, q_v=q_v, p_full=p_full, p_half=p_half, u=z, v=z,
        conv_prog_profile=z, dt=600.0, config=TiedtkeConfig())
    out = out if hasattr(out, "dq_v_dt") else out[0]
    q_new = np.asarray(q_v + 600.0 * out.dq_v_dt)
    assert q_new.min() >= -1e-12, f"q_v went negative: {q_new.min():.3e}"
