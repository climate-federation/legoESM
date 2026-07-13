"""Tests for moist baroclinic-wave tracer transport on the cubed-sphere FV3
C-D grid dycore.

Covers the new cube tracer-advection feature:
  * the moist IC (`baroclinic_wave_init(..., moist=True)`) attaches q_v/q_c/q_r;
  * `fv3_hydrostatic_tendencies` returns advective-form `tracer_tendencies`
    (consistent with how the FV3 hydrostatic core transports temperature);
  * the full RK3 `step` advects the tracers (they change, stay finite, stay
    non-negative after the prognostic floor) and the global tracer mass drifts
    only slowly (advective form — bounded, like the core's T transport, not the
    machine-precision conservation of a flux-form scheme);
  * `make_kessler_forcing_cube` drives condensation through a real step.

These build + step the dycore, so they run on a compute node (JIT), not the
Ginsburg login node.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.operators import global_integral
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel,
    CDGridPrimitiveEquationConfig,
    fv3_hydrostatic_tendencies,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.kessler_forcing import make_kessler_forcing_cube
from tests.test_cases.baroclinic_wave import baroclinic_wave_init

_N = 8
_NLEV = 12
_DT = 300.0


def _grid_sigma():
    grid = create_cubed_sphere(_N)
    sigma = create_sigma_coordinate(_NLEV, dtype=jnp.float64)
    return grid, sigma


def test_moist_ic_attaches_tracers():
    grid, sigma = _grid_sigma()
    dry = baroclinic_wave_init(grid, sigma, perturbed=False)
    assert dry.tracers is None
    wet = baroclinic_wave_init(grid, sigma, perturbed=False, moist=True)
    assert wet.tracers is not None
    assert set(wet.tracers) == {"q_v", "q_c", "q_r"}
    assert wet.tracers["q_v"].data.shape == (6, _N, _N, _NLEV)
    assert jnp.all(wet.tracers["q_v"].data >= 0.0)
    assert jnp.any(wet.tracers["q_v"].data > 0.0)
    assert jnp.all(wet.tracers["q_c"].data == 0.0)
    assert jnp.all(wet.tracers["q_r"].data == 0.0)


def test_fv3_tendencies_carry_tracer_tendencies():
    """fv3_hydrostatic_tendencies must return advective tracer tendencies for a
    moist state (and None for a dry one)."""
    grid, sigma = _grid_sigma()
    cdgrid = create_cubed_sphere_cdgrid(grid)
    config = CDGridPrimitiveEquationConfig()

    dry = hydrostatic_to_fv3(
        baroclinic_wave_init(grid, sigma, perturbed=True), cdgrid)
    tend_dry = fv3_hydrostatic_tendencies(dry, grid, sigma, cdgrid, config)
    assert tend_dry.tracer_tendencies is None

    wet = hydrostatic_to_fv3(
        baroclinic_wave_init(grid, sigma, perturbed=True, moist=True), cdgrid)
    tend = fv3_hydrostatic_tendencies(wet, grid, sigma, cdgrid, config)
    assert tend.tracer_tendencies is not None
    assert set(tend.tracer_tendencies) == {"q_v", "q_c", "q_r"}
    for k in ("q_v", "q_c", "q_r"):
        assert tend.tracer_tendencies[k].data.shape == (6, _N, _N, _NLEV)
        assert jnp.all(jnp.isfinite(tend.tracer_tendencies[k].data))
    # q_v has a horizontal gradient (RH-tapered + perturbed jet) so its
    # advective tendency must be nonzero somewhere.
    assert jnp.any(jnp.abs(tend.tracer_tendencies["q_v"].data) > 0.0)


def _total_tracer_mass(state, sigma, grid):
    # mass-weighted, area-weighted global tracer integral (advective-form proxy).
    # Reuses the model's own area weighting via global_integral.
    dsigma = jnp.diff(sigma.sigma_half)
    tw = (state.tracers["q_v"].data + state.tracers["q_c"].data
          + state.tracers["q_r"].data)
    col = jnp.sum(tw * dsigma[None, None, None, :], axis=-1) * state.p_s.data
    field = Field(data=col, name="tw_col", dims=("face", "x", "y"), units="kg/m^2")
    return float(global_integral(field, grid))


def test_dycore_advects_tracers_and_mass_bounded():
    """Pure advection (physics_fn=None): tracers change, stay finite + >=0, and
    global tracer mass drifts only slowly (advective form, like T)."""
    grid, sigma = _grid_sigma()
    cdgrid = create_cubed_sphere_cdgrid(grid)
    config = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=True, fix_mass=True,
        anchor_mass_to_initial=True, zero_mean_ps_tendency=True,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, config)
    state = hydrostatic_to_fv3(
        baroclinic_wave_init(grid, sigma, perturbed=True, moist=True), cdgrid)

    qv0 = state.tracers["q_v"].data
    m0 = _total_tracer_mass(state, sigma, grid)
    # CDGrid model.step returns the new state only (the physics carry is stashed
    # on the model); FV3HydrostaticState is a NamedTuple, so do NOT unpack it.
    for _ in range(15):
        state = model.step(state, _DT, physics_fn=None)
    qv1 = state.tracers["q_v"].data
    m1 = _total_tracer_mass(state, sigma, grid)

    for k in ("q_v", "q_c", "q_r"):
        arr = state.tracers[k].data
        assert jnp.all(jnp.isfinite(arr)), f"{k} non-finite"
        assert float(jnp.min(arr)) >= -1e-12, f"{k} below floor"
    assert float(jnp.max(jnp.abs(qv1 - qv0))) > 0.0, "q_v never advected"
    rel = abs(m1 - m0) / max(m0, 1e-30)
    assert rel < 0.05, f"tracer mass drifted too much (advective): {rel}"


def test_moist_step_with_kessler_runs():
    """A full moist step with the cube Kessler forcing runs and stays finite;
    q_v changes (advection + condensation active)."""
    grid, sigma = _grid_sigma()
    cdgrid = create_cubed_sphere_cdgrid(grid)
    config = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=True, fix_mass=True,
        anchor_mass_to_initial=True, zero_mean_ps_tendency=True,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, config)
    state = hydrostatic_to_fv3(
        baroclinic_wave_init(grid, sigma, perturbed=True, moist=True), cdgrid)
    phys = make_kessler_forcing_cube(_DT)

    qv0 = state.tracers["q_v"].data
    for _ in range(5):
        state = model.step(state, _DT, physics_fn=phys)
    for k in ("q_v", "q_c", "q_r"):
        arr = state.tracers[k].data
        assert jnp.all(jnp.isfinite(arr)), f"{k} non-finite"
        assert float(jnp.min(arr)) >= -1e-12, f"{k} below floor"
    assert float(jnp.max(jnp.abs(state.tracers["q_v"].data - qv0))) > 0.0
