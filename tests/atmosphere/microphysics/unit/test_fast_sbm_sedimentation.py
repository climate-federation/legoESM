"""Fast-SBM per-bin sedimentation (oracle FALFLUXHUCM_Z via shared helper).

Level convention: index 0 = top, last = surface.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.microphysics.fast_sbm import (
    mass_doubling_grid,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.sedimentation import (
    sediment_bins,
)

jax.config.update("jax_enable_x64", True)

NCOL, NLEV, NKR = 2, 6, 33


def _column():
    rho = jnp.full((NCOL, NLEV), 1.0)
    dz = jnp.full((NCOL, NLEV), 200.0)
    q = jnp.zeros((NCOL, NLEV, NKR)).at[:, 2, 10].set(1.0e-3)
    return q, rho, dz


def test_zero_velocity_fixed_point():
    q, rho, dz = _column()
    dq_dt, precip = sediment_bins(q, rho, jnp.zeros(NKR), dz, 10.0)
    np.testing.assert_array_equal(np.asarray(dq_dt), 0.0)
    np.testing.assert_array_equal(np.asarray(precip), 0.0)


def test_mass_moves_down_and_conserves_with_precip():
    q, rho, dz = _column()
    v = jnp.full((NKR,), 2.0)         # 2 m/s, CFL_sub = 2*10/(200*4) = 0.025
    dt = 10.0
    dq_dt, precip = sediment_bins(q, rho, v, dz, dt)
    q1 = q + dt * dq_dt
    assert np.all(np.asarray(q1) >= -1e-20)
    # Column water + surface precipitation invariant.
    col0 = np.asarray(jnp.sum(q * rho[..., None] * dz[..., None],
                              axis=(1, 2)))
    col1 = np.asarray(jnp.sum(q1 * rho[..., None] * dz[..., None],
                              axis=(1, 2)))
    np.testing.assert_allclose(col1 + np.asarray(precip) * dt, col0,
                               rtol=1e-12)
    # Mass left the source level downward only.
    assert float(q1[0, 2, 10]) < 1.0e-3
    assert float(q1[0, 3, 10]) > 0.0
    assert float(q1[0, 1, 10]) == 0.0


def test_bottom_layer_precipitates():
    rho = jnp.full((NCOL, NLEV), 1.0)
    dz = jnp.full((NCOL, NLEV), 100.0)
    q = jnp.zeros((NCOL, NLEV, NKR)).at[:, -1, 20].set(5.0e-4)
    dt = 20.0
    v = jnp.zeros(NKR).at[20].set(5.0)    # fast rain bin
    dq_dt, precip = sediment_bins(q, rho, v, dz, dt)
    assert np.all(np.asarray(precip) > 0.0)
    # Everything that left the column shows up as precip.
    col_loss = -float(jnp.sum(dq_dt[0] * rho[0, :, None] * dz[0, :, None]))
    assert col_loss * dt == pytest.approx(float(precip[0]) * dt, rel=1e-12)


def test_positivity_at_high_cfl():
    rho = jnp.full((1, NLEV), 1.0)
    dz = jnp.full((1, NLEV), 50.0)
    q = jnp.zeros((1, NLEV, NKR)).at[0, 1, 30].set(1.0e-3)
    v = jnp.zeros(NKR).at[30].set(9.0)
    dt = 60.0                              # raw CFL = 10.8 >> 1
    dq_dt, precip = sediment_bins(q, rho, v, dz, dt, n_substeps=4)
    q1 = q + dt * dq_dt
    # Positivity up to cap-cancellation roundoff (~1e-19 of the 1e-3 load).
    assert np.all(np.asarray(q1) >= -1e-18)
    col0 = float(jnp.sum(q * rho[..., None] * dz[..., None]))
    col1 = float(jnp.sum(q1 * rho[..., None] * dz[..., None]))
    assert col1 + float(precip[0]) * dt == pytest.approx(col0, rel=1e-12)


def test_per_level_velocity_field():
    # v_term as (ncol, nlev, n_bins): each level can fall at its own speed.
    # Mass only leaves levels where v > 0 — confirms the field is used
    # per-level, not collapsed to a single profile.
    rho = jnp.full((1, NLEV), 1.0)
    dz = jnp.full((1, NLEV), 100.0)
    q = jnp.zeros((1, NLEV, NKR)).at[0, :, 12].set(1.0e-4)
    v = jnp.zeros((1, NLEV, NKR))
    v = v.at[0, 2, 12].set(3.0)        # only level 2 falls
    dt = 10.0
    dq_dt, precip = sediment_bins(q, rho, v, dz, dt)
    q1 = q + dt * dq_dt
    # Level 2 lost mass to level 3; the still levels (0,1,4,5) unchanged.
    assert float(q1[0, 2, 12]) < 1.0e-4
    assert float(q1[0, 3, 12]) > 1.0e-4      # gained from above
    for k in (0, 1, 4, 5):
        assert float(q1[0, k, 12]) == pytest.approx(1.0e-4, rel=1e-12)
    # Column conserved (no precip — nothing reached the surface).
    col0 = float(jnp.sum(q * 100.0))
    col1 = float(jnp.sum(q1 * 100.0))
    assert col1 + float(precip[0]) * dt == pytest.approx(col0, rel=1e-12)


def test_differentiable_in_velocity():
    q, rho, dz = _column()

    def precip_of(v_scalar):
        v = jnp.full((NKR,), v_scalar)
        _, precip = sediment_bins(q, rho, v, dz, 200.0)
        return jnp.sum(precip)

    g = jax.grad(precip_of)(jnp.asarray(1.0))
    assert np.isfinite(float(g))
    assert float(g) > 0.0     # faster fall → more surface precip
