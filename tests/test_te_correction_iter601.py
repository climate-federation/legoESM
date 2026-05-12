"""FV3_3D iter 601: NH total-energy-conserving correction test.

Tests
-----

1. ``test_te_correction_reduces_drift`` — after correction,
   TE(corrected) ≈ TE(state_old).
2. ``test_te_correction_no_op_when_states_match``.
3. ``test_te_correction_only_adjusts_theta_prime``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.diagnostics import (
    apply_te_correction_nh,
    te_drift_nh,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_height_coordinate


def _build_state(n=8, nlev=5, theta_amp=0.0, u_amp=0.0):
    grid = create_cubed_sphere(n)
    hc = create_height_coordinate(nlev, 30000.0)
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.full((6, n, n, nlev), u_amp, dtype=jnp.float64),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1), dtype=jnp.float64),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(
            data=jnp.full((6, n, n, nlev), theta_amp, dtype=jnp.float64),
            name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n), dtype=jnp.float64),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0), dtype=jnp.float64),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, state


def test_te_correction_reduces_drift(capsys):
    """After correction, TE drift should be reduced to ~float64."""
    grid, hc, state_old = _build_state(theta_amp=0.0)
    _, _, state_new = _build_state(theta_amp=10.0)
    drift_before = te_drift_nh(state_old, state_new, grid, hc)
    state_corr = apply_te_correction_nh(state_old, state_new, grid, hc)
    drift_after = te_drift_nh(state_old, state_corr, grid, hc)
    with capsys.disabled():
        print(f"\n[iter-601] TE drift before: {drift_before:.3e}")
        print(f"[iter-601] TE drift after:  {drift_after:.3e}")
        if abs(drift_before) > 0:
            print(f"[iter-601] reduction factor: "
                  f"{abs(drift_after) / abs(drift_before):.3e}")
    assert abs(drift_after) < abs(drift_before) * 1e-6, (
        f"correction should reduce drift by ~6 orders of magnitude: "
        f"before={drift_before:.3e}, after={drift_after:.3e}"
    )


def test_te_correction_no_op_when_states_match():
    grid, hc, state = _build_state(theta_amp=5.0)
    state_corr = apply_te_correction_nh(state, state, grid, hc)
    dtheta = float(jnp.abs(state_corr.theta_prime.data
                           - state.theta_prime.data).max())
    assert dtheta < 1e-10, (
        f"identical states should be no-op; max|Δθ′|={dtheta}"
    )


def test_te_correction_only_adjusts_theta_prime():
    grid, hc, state_old = _build_state(theta_amp=0.0)
    _, _, state_new = _build_state(theta_amp=10.0, u_amp=5.0)
    state_corr = apply_te_correction_nh(state_old, state_new, grid, hc)
    # u, v, w, rho_prime unchanged
    assert jnp.array_equal(state_corr.u.data, state_new.u.data)
    assert jnp.array_equal(state_corr.v.data, state_new.v.data)
    assert jnp.array_equal(state_corr.w.data, state_new.w.data)
    assert jnp.array_equal(state_corr.rho_prime.data,
                            state_new.rho_prime.data)
    # theta_prime changed
    assert not jnp.array_equal(state_corr.theta_prime.data,
                                state_new.theta_prime.data)
