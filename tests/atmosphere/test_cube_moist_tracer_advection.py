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
from legoesm.atmosphere.forcing.idealized.kessler_forcing import make_kessler_forcing_cube
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


def _total_tracer_mass(state, sigma, grid):
    # mass-weighted, area-weighted global tracer integral (advective-form proxy).
    # Reuses the model's own area weighting via global_integral.
    dsigma = jnp.diff(sigma.sigma_half)
    tw = (state.tracers["q_v"].data + state.tracers["q_c"].data
          + state.tracers["q_r"].data)
    col = jnp.sum(tw * dsigma[None, None, None, :], axis=-1) * state.p_s.data
    field = Field(data=col, name="tw_col", dims=("face", "x", "y"), units="kg/m^2")
    return float(global_integral(field, grid))
