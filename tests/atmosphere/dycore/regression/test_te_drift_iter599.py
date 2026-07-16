"""FV3_3D iter 599: TE drift diagnostics for NH and PE.

Tests
-----

1. ``test_te_drift_nh_zero_for_identical``.
2. ``test_te_drift_nh_positive_for_warmer``.
3. ``test_te_drift_pe_zero_for_identical``.
4. ``test_te_drift_pe_positive_for_warmer``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.diagnostics import te_drift_nh, te_drift_pe
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    create_height_coordinate,
    standard_hybrid_levels,
)


def _build_nh(n=8, nlev=5, theta_amp=0.0):
    grid = create_cubed_sphere(n)
    hc = create_height_coordinate(nlev, 30000.0)
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
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


def _build_pe(n=8, nlev=5, T_delta=0.0):
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_hs = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_hs, cdgrid)
    if T_delta != 0.0:
        state = state._replace(
            T=state.T.replace(data=state.T.data + T_delta),
        )
    return grid, coord, state


def test_te_drift_nh_zero_for_identical():
    grid, hc, state = _build_nh()
    drift = te_drift_nh(state, state, grid, hc)
    assert abs(drift) < 1e6, (
        f"Identical NH states should have zero TE drift; got {drift}"
    )


def test_te_drift_nh_positive_for_warmer():
    grid, hc, state_cold = _build_nh(theta_amp=0.0)
    _, _, state_warm = _build_nh(theta_amp=10.0)
    drift = te_drift_nh(state_cold, state_warm, grid, hc)
    assert drift > 0, f"+10K θ' should increase TE; got drift={drift:.3e}"


def test_te_drift_pe_zero_for_identical():
    grid, coord, state = _build_pe()
    drift = te_drift_pe(state, state, grid, coord)
    assert abs(drift) < 1e10, (
        f"Identical PE states should have zero TE drift; got {drift}"
    )


def test_te_drift_pe_positive_for_warmer():
    grid, coord, state_cold = _build_pe(T_delta=0.0)
    _, _, state_warm = _build_pe(T_delta=10.0)
    drift = te_drift_pe(state_cold, state_warm, grid, coord)
    assert drift > 0, f"+10K T should increase PE TE; got drift={drift:.3e}"
