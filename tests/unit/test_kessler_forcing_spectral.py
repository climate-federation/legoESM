"""Direct tests for the spectral-PE Kessler physics adapter.

``make_kessler_forcing_spectral`` bridges the global spectral primitive-equation
state to the grid-agnostic column Kessler warm-rain scheme: inverse-SH the
temperature/pressure, run the SHARED column microphysics, forward-SH the latent
heating back to ``T_hat``, and return tracer rates in grid space. These tests
exercise the TENDENCY (not the time integration): condensation signs, total-water
conservation, latent-heat sign, zeroed dynamics tendencies, and the guards.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis, sh_synthesis_3d
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.forcing.idealized.kessler_forcing import make_kessler_forcing_spectral
from tests.test_cases.baroclinic_wave import baroclinic_wave_init_spectral

_DT = 600.0


def _supersaturated_state(grid, sigma, factor=2.0):
    """Dry JW spectral state + grid-space tracers with q_v = factor·q_sat (>1)."""
    state = baroclinic_wave_init_spectral(grid, sigma, perturbed=False)
    T_grid = sh_synthesis_3d(grid, state.T_hat.data)            # (n_lat,n_lon,nlev)
    p_s = jnp.exp(sh_synthesis(grid, state.lnps_hat.data))      # (n_lat,n_lon)
    p_full = p_s[..., None] * sigma.sigma_full                  # (n_lat,n_lon,nlev)
    q_sat = saturation_mixing_ratio(T_grid, p_full)
    q_v = factor * q_sat                                        # supersaturated
    q_zero = jnp.zeros_like(q_v)
    dims = ("lat", "lon", "level")
    tracers = {
        "q_v": Field(data=q_v, name="q_v", dims=dims, units="kg/kg"),
        "q_c": Field(data=q_zero, name="q_c", dims=dims, units="kg/kg"),
        "q_r": Field(data=q_zero, name="q_r", dims=dims, units="kg/kg"),
    }
    return state._replace(tracers=tracers)


@pytest.fixture(scope="module")
def grid():
    return create_gaussian_grid(10)


@pytest.fixture(scope="module")
def sigma():
    return create_sigma_coordinate(8)


def test_tendency_signs_and_conservation(grid, sigma):
    state = _supersaturated_state(grid, sigma)
    phys = make_kessler_forcing_spectral(_DT)
    tend = phys(state, grid, sigma)

    dq_v = tend.tracers["q_v"].data
    dq_c = tend.tracers["q_c"].data
    dq_r = tend.tracers["q_r"].data
    for name, arr in (("dq_v", dq_v), ("dq_c", dq_c), ("dq_r", dq_r)):
        assert jnp.all(jnp.isfinite(arr)), f"{name} not finite"

    # Supersaturated air -> condensation: vapor sink, cloud source somewhere.
    assert float(jnp.min(dq_v)) < 0.0, "no vapor removal under supersaturation"
    assert float(jnp.max(dq_c)) > 0.0, "no cloud formation under supersaturation"

    # Warm-rain microphysics is internal: total water q_v+q_c+q_r is conserved,
    # so the three tendencies sum to ~0 everywhere.
    total = dq_v + dq_c + dq_r
    scale = float(jnp.max(jnp.abs(dq_v))) + 1e-30
    assert float(jnp.max(jnp.abs(total))) / scale < 1e-9, "total water not conserved"

    # Latent heating: condensation warms -> dT > 0 somewhere (synthesize T_hat).
    dT_grid = sh_synthesis_3d(grid, tend.T_hat.data)
    assert jnp.all(jnp.isfinite(dT_grid))
    assert float(jnp.max(dT_grid)) > 0.0, "no latent heating under condensation"

    # Warm-rain has no momentum / surface-pressure / geopotential source.
    assert float(jnp.max(jnp.abs(tend.vor_hat.data))) == 0.0
    assert float(jnp.max(jnp.abs(tend.div_hat.data))) == 0.0
    assert float(jnp.max(jnp.abs(tend.lnps_hat.data))) == 0.0
    assert float(jnp.max(jnp.abs(tend.phis_hat.data))) == 0.0


def test_forward_step_keeps_vapor_nonnegative(grid, sigma):
    """q_v + dt·dq_v stays >= 0 (condensation cannot over-deplete vapor)."""
    state = _supersaturated_state(grid, sigma, factor=1.5)
    phys = make_kessler_forcing_spectral(_DT)
    tend = phys(state, grid, sigma)
    q_v_new = state.tracers["q_v"].data + _DT * tend.tracers["q_v"].data
    assert float(jnp.min(q_v_new)) >= -1e-12


def test_bound_dt_and_column_local(grid, sigma):
    phys = make_kessler_forcing_spectral(_DT)
    assert phys._bound_dt == _DT
    assert phys._column_local is True


def test_requires_tracers(grid, sigma):
    state = baroclinic_wave_init_spectral(grid, sigma, perturbed=False)
    phys = make_kessler_forcing_spectral(_DT)
    with pytest.raises(ValueError, match="requires state.tracers"):
        phys(state, grid, sigma)


def test_rejects_nonpositive_dt():
    with pytest.raises(ValueError, match="must be > 0"):
        make_kessler_forcing_spectral(0.0)


def test_moist_ic_attaches_tracers(grid, sigma):
    """moist=True attaches q_v/q_c/q_r; q_v in [0, q_sat], q_c=q_r=0."""
    state = baroclinic_wave_init_spectral(grid, sigma, perturbed=False, moist=True)
    assert state.tracers is not None
    for k in ("q_v", "q_c", "q_r"):
        assert k in state.tracers
    q_v = state.tracers["q_v"].data
    assert jnp.all(jnp.isfinite(q_v))
    assert float(jnp.min(q_v)) >= 0.0
    assert float(jnp.max(jnp.abs(state.tracers["q_c"].data))) == 0.0
    assert float(jnp.max(jnp.abs(state.tracers["q_r"].data))) == 0.0
    # q_v capped at saturation.
    T_grid = sh_synthesis_3d(grid, state.T_hat.data)
    p_full = jnp.exp(sh_synthesis(grid, state.lnps_hat.data))[..., None] * sigma.sigma_full
    q_sat = saturation_mixing_ratio(T_grid, p_full)
    assert float(jnp.max(q_v - q_sat)) <= 1e-12


def test_dry_ic_has_no_tracers(grid, sigma):
    state = baroclinic_wave_init_spectral(grid, sigma, perturbed=False, moist=False)
    assert state.tracers is None
