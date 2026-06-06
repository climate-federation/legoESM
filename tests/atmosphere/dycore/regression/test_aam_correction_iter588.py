"""FV3_3D iter 588: FV3 consv_am correction for NH state.

Tests
-----

1. ``test_correction_zeros_amdt`` — after correction, AAM(corrected)
   ≈ AAM(old) to high precision.
2. ``test_correction_is_no_op_when_states_match`` — identical states
   → no-op (within float roundoff).
3. ``test_correction_returns_finite``.
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
    aam_drift_nh,
    aam_from_nh_state,
    apply_aam_correction_nh,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_height_coordinate


def _build_state(n=8, nlev=5, u_amp=0.0, v_amp=0.0):
    grid = create_cubed_sphere(n)
    hc = create_height_coordinate(nlev, 30000.0)
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.full((6, n, n, nlev), u_amp, dtype=jnp.float64),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.full((6, n, n, nlev), v_amp, dtype=jnp.float64),
                name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1), dtype=jnp.float64),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
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


def test_correction_zeros_amdt(capsys):
    """After applying correction, AAM(corrected) ≈ AAM(state_old)."""
    grid, hc, state_old = _build_state(u_amp=0.0)
    _, _, state_new = _build_state(u_amp=1.0)  # +1 m/s zonal wind
    drift_before = aam_drift_nh(state_old, state_new, grid, hc)
    state_corrected = apply_aam_correction_nh(
        state_old, state_new, grid, hc,
    )
    drift_after = aam_drift_nh(state_old, state_corrected, grid, hc)
    with capsys.disabled():
        print(f"\n[iter-588] drift before correction: {drift_before:.3e}")
        print(f"[iter-588] drift after correction:  {drift_after:.3e}")
        if abs(drift_before) > 0:
            print(f"[iter-588] reduction factor: {abs(drift_after) / abs(drift_before):.3e}")
    # Correction should reduce drift by at least 1e6 (float64 precision)
    assert abs(drift_after) < abs(drift_before) * 1e-6, (
        f"correction should zero AAM drift: "
        f"before={drift_before:.3e}, after={drift_after:.3e}"
    )


def test_correction_is_no_op_when_states_match():
    """state_old = state_new → correction is no-op (within float roundoff)."""
    grid, hc, state = _build_state(u_amp=1.0)
    state_corr = apply_aam_correction_nh(state, state, grid, hc)
    du = float(jnp.abs(state_corr.u.data - state.u.data).max())
    dv = float(jnp.abs(state_corr.v.data - state.v.data).max())
    # amdt = 0 → u0 = 0 → no correction
    assert du < 1e-10, f"u shouldn't change when states match: max|Δu|={du}"
    assert dv < 1e-10, f"v shouldn't change when states match: max|Δv|={dv}"


def test_correction_returns_finite():
    grid, hc, state_old = _build_state(u_amp=0.0)
    _, _, state_new = _build_state(u_amp=5.0, v_amp=2.0)
    state_corr = apply_aam_correction_nh(state_old, state_new, grid, hc)
    assert jnp.all(jnp.isfinite(state_corr.u.data))
    assert jnp.all(jnp.isfinite(state_corr.v.data))
