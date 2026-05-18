"""Unit tests for the OMIP-2 surface-flux applicator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import importlib.util

import numpy as np
import pytest


def _matrix_module():
    if not hasattr(_matrix_module, "_mod"):
        repo_root = Path(__file__).resolve().parents[3]
        scripts_dir = repo_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        matrix_path = scripts_dir / "run_ocean_test_matrix.py"
        spec = importlib.util.spec_from_file_location(
            "_rom_for_omip2_apply_tests", matrix_path,
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _matrix_module._mod = mod
    return _matrix_module._mod


def _rest_state_latlon():
    matrix_mod = _matrix_module()
    from ocean_test_matrix.setup import _create_ocean_setup
    from ocean_test_matrix.testcase import TestCase
    tc = TestCase("omip_applicator", "latlon", "36x72", 1.0, 0.1)
    grid, z, _, model, _, _, _ = _create_ocean_setup(
        tc, H_max=5500.0, nlev=10,
    )
    state = matrix_mod._create_rest_state(tc, grid, z, H_max=5500.0)
    return state, grid, z, model


def test_applicator_returns_same_type():
    from legoesm.ocean.coupler import apply_omip2_surface_fluxes
    from legoesm.ocean.forcing import synthetic_ocean_forcing
    state, grid, z, _ = _rest_state_latlon()
    forcing = synthetic_ocean_forcing(2000, n_time=4, nlon=72, nlat=36)
    new_state = apply_omip2_surface_fluxes(
        state, forcing=forcing, idx_t=0,
        z_coord=z, grid=grid, grid_type="latlon", dt=1800.0,
    )
    assert type(new_state) is type(state)
    # ``T_max`` of the modified surface layer must remain finite.
    T_top = np.asarray(new_state.T.data)[..., 0]
    assert np.isfinite(T_top).all()


def test_applicator_injects_kinetic_energy_from_rest():
    """At rest, applying nonzero wind should produce nonzero top-cell u/v."""
    from legoesm.ocean.coupler import apply_omip2_surface_fluxes
    from legoesm.ocean.forcing import synthetic_ocean_forcing
    state, grid, z, _ = _rest_state_latlon()
    forcing = synthetic_ocean_forcing(2000, n_time=4, nlon=72, nlat=36)
    # Sanity: u/v exactly zero before.
    assert np.all(np.asarray(state.u.data) == 0.0)
    assert np.all(np.asarray(state.v.data) == 0.0)
    new_state = apply_omip2_surface_fluxes(
        state, forcing=forcing, idx_t=0,
        z_coord=z, grid=grid, grid_type="latlon", dt=1800.0,
    )
    u_top = np.asarray(new_state.u.data)[..., 0]
    v_top = np.asarray(new_state.v.data)[..., 0]
    # Wind stress should have moved at least some surface velocity.
    assert np.abs(u_top).max() > 1e-6
    assert np.abs(v_top).max() > 1e-6
    # Below the surface should stay at rest after a single step
    # (no vertical mixing applied by the applicator).
    assert np.all(np.asarray(new_state.u.data)[..., 1:] == 0.0)


def test_applicator_temperature_responds_to_heat_flux():
    """Top-cell T should drift toward the forcing temperature; for a
    warm-air over cold-ocean column the top cell heats up."""
    from legoesm.ocean.coupler import apply_omip2_surface_fluxes
    from legoesm.ocean.forcing import synthetic_ocean_forcing
    state, grid, z, _ = _rest_state_latlon()
    forcing = synthetic_ocean_forcing(2000, n_time=4, nlon=72, nlat=36)
    T_top_initial = np.asarray(state.T.data)[..., 0].copy()
    new_state = apply_omip2_surface_fluxes(
        state, forcing=forcing, idx_t=0,
        z_coord=z, grid=grid, grid_type="latlon", dt=1800.0,
    )
    T_top_after = np.asarray(new_state.T.data)[..., 0]
    delta = T_top_after - T_top_initial
    # At least one cell must change; magnitude must be sane (<1 K per
    # half-hour step is sane; flagging > 5 K would catch a runaway).
    assert np.abs(delta).max() > 1e-6
    assert np.abs(delta).max() < 5.0


def test_applicator_raises_on_unsupported_grid():
    from legoesm.ocean.coupler import apply_omip2_surface_fluxes
    from legoesm.ocean.forcing import synthetic_ocean_forcing
    state, grid, z, _ = _rest_state_latlon()
    forcing = synthetic_ocean_forcing(2000, n_time=4, nlon=72, nlat=36)
    with pytest.raises(NotImplementedError):
        apply_omip2_surface_fluxes(
            state, forcing=forcing, idx_t=0,
            z_coord=z, grid=grid, grid_type="cubed_sphere", dt=1800.0,
        )
