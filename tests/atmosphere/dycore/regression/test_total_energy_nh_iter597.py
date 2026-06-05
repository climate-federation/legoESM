"""FV3_3D iter 597: NH total-energy diagnostic test.

Tests
-----

1. ``test_te_at_rest_is_positive_finite``.
2. ``test_te_increases_with_kinetic_energy`` — adding u increases TE.
3. ``test_te_increases_with_temperature`` — adding θ' increases TE.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.diagnostics import compute_total_energy_nh
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_height_coordinate


def _build_state(n=8, nlev=5, u_amp=0.0, theta_amp=0.0):
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


def test_te_at_rest_is_positive_finite():
    grid, hc, state = _build_state(u_amp=0.0)
    col, total = compute_total_energy_nh(state, grid, hc)
    assert col.shape == (6, 8, 8)
    assert np.isfinite(total)
    assert total > 0, f"TE at rest should be positive (cv·T+g·z); got {total}"


def test_te_increases_with_kinetic_energy():
    grid, hc, state_rest = _build_state(u_amp=0.0)
    _, _, state_moving = _build_state(u_amp=10.0)
    _, te_rest = compute_total_energy_nh(state_rest, grid, hc)
    _, te_moving = compute_total_energy_nh(state_moving, grid, hc)
    delta = te_moving - te_rest
    # Δ_KE = 0.5 · u² · total_mass = 0.5 · 100 · M_total
    # Should be POSITIVE
    assert delta > 0, (
        f"Adding 10 m/s u should increase TE; got Δ={delta}"
    )


def test_te_increases_with_temperature():
    grid, hc, state_cold = _build_state(theta_amp=0.0)
    _, _, state_warm = _build_state(theta_amp=10.0)
    _, te_cold = compute_total_energy_nh(state_cold, grid, hc)
    _, te_warm = compute_total_energy_nh(state_warm, grid, hc)
    delta = te_warm - te_cold
    # Δ_IE = cv · ΔT · total_mass.  ΔT ~ theta_amp · exner_ref (varies
    # with k) — should be positive overall.
    assert delta > 0, (
        f"Adding 10K θ' should increase TE; got Δ={delta}"
    )
