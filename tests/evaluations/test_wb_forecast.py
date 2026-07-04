"""Tests for the WB2 headline-field assembly (Stage 0, Task 5).

Runs on a compute node (builds T21 SH transforms):
    srun --account=glab --time=0:15:00 env JAX_ENABLE_X64=1 \
        <conda>/python -m pytest tests/evaluations/test_wb_forecast.py -v
"""
import numpy as np
import jax.numpy as jnp

from legoesm import constants
from evaluations.wb_forecast import diagnose_headline_fields, HEADLINE_FIELD_KEYS


def _rest_state():
    from legoesm.atmosphere.dynamics.spectral_pe import isothermal_rest_state_spectral
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_gaussian_grid(21)          # T21 — small but valid
    sigma = create_sigma_coordinate(8)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)
    return state, grid, sigma


def test_headline_fields_keys_shapes_finite():
    state, grid, sigma = _rest_state()
    out = diagnose_headline_fields(state, grid, sigma)
    assert set(out) == set(HEADLINE_FIELD_KEYS)
    for k, arr in out.items():
        assert arr.shape == (grid.n_lat, grid.n_lon), k
        assert bool(jnp.all(jnp.isfinite(arr))), k


def test_headline_fields_isothermal_rest_physical():
    # Isothermal T=300, p_s=p_ref, flat (phis=0), at rest -> known headline values.
    state, grid, sigma = _rest_state()
    out = diagnose_headline_fields(state, grid, sigma)
    z500_analytic = float(constants.R_d * 300.0 / constants.g * np.log(2.0))  # ~6086 m
    assert abs(float(jnp.mean(out["z500"])) - z500_analytic) < 100.0
    assert abs(float(jnp.mean(out["t850"])) - 300.0) < 4.0
    assert abs(float(jnp.mean(out["mslp"])) - float(constants.p_ref)) < 100.0  # phis=0 -> mslp=p_s
    assert float(jnp.max(jnp.abs(out["u500"]))) < 1e-2       # at rest
    assert float(jnp.max(jnp.abs(out["v500"]))) < 1e-2
    assert float(jnp.max(out["wind_speed_10m"])) < 1e-2
    assert abs(float(jnp.mean(out["t2m"])) - 300.0) < 4.0


def test_headline_10m_wind_not_stronger_than_lowest_level():
    # The neutral reduction must weaken (never amplify) the lowest-level wind.
    from legoesm.atmosphere.dynamics.spectral_pe import spectral_pe_to_grid
    state, grid, sigma = _rest_state()
    out = diagnose_headline_fields(state, grid, sigma)
    g = spectral_pe_to_grid(state, grid, sigma)
    speed_low = jnp.sqrt(g["u"][..., -1] ** 2 + g["v"][..., -1] ** 2)
    # allow a tiny epsilon from the AD-safe floor
    assert float(jnp.max(out["wind_speed_10m"] - speed_low)) < 1e-6
