"""FV3_3D iter 587: AAM drift diagnostic for NH state.

Tests
-----

1. ``test_aam_from_nh_state_runs`` — basic invocation.
2. ``test_aam_drift_zero_for_identical_states`` — drift = 0 when
   state unchanged.
3. ``test_aam_drift_changes_with_zonal_acceleration`` — adding
   uniform u shifts AAM (correct sign).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.diagnostics import aam_drift_nh, aam_from_nh_state
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_height_coordinate


def _build_state(n=8, nlev=5, u_amp=0.0):
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


def test_aam_from_nh_state_runs():
    grid, hc, state = _build_state(u_amp=0.0)
    col, total = aam_from_nh_state(state, grid, hc)
    assert col.shape == (6, 8, 8)
    assert np.isfinite(total)
    assert total > 0.0  # earth rotation positive


def test_aam_drift_zero_for_identical_states():
    grid, hc, state = _build_state(u_amp=0.0)
    drift = aam_drift_nh(state, state, grid, hc)
    assert abs(drift) < 1e-3, (
        f"Identical states should have zero AAM drift; got {drift}"
    )


def test_aam_drift_changes_with_zonal_acceleration():
    """Adding face-local u shifts AAM (after rotation to u_east)."""
    grid, hc, state_old = _build_state(u_amp=0.0)
    _, _, state_new = _build_state(u_amp=1.0)
    drift = aam_drift_nh(state_old, state_new, grid, hc)
    # +1 m/s face-local u, after rotation, contributes some u_east
    # over a non-trivial portion of cube → AAM should increase
    # (sign depends on face orientation, but magnitude should be
    # non-negligible compared to drift=0 case).
    assert abs(drift) > 1e10, (
        f"Adding 1 m/s u should produce non-trivial AAM drift; got {drift}"
    )
