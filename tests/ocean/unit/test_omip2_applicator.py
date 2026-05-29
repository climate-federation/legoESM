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


def test_woa_synthetic_sst_in_realistic_range():
    """WOA synthetic SST falls inside the global-ocean range."""
    from legoesm.ocean.forcing import synthetic_woa_sst
    sst_K, lat, lon = synthetic_woa_sst(nlon=72, nlat=36)
    assert sst_K.shape == (36, 72)
    assert sst_K.min() >= 263.0
    assert sst_K.max() <= 305.0
    # Unweighted global mean is ~ 293 K for the synthetic profile
    # (uniform-dlat integration overweights mid-latitudes vs the
    # area-weighted WOA ~290 K).
    assert 285.0 < sst_K.mean() < 297.0


def test_load_woa_sst_synthetic_fallback():
    from legoesm.ocean.forcing import load_woa_sst
    sst_K, lat, lon = load_woa_sst(nlon=72, nlat=36)
    assert sst_K.shape == (36, 72)
    # Sanity: equator is warmer than poles.
    eq_band = sst_K[16:20, :].mean()
    pole_band = sst_K[0:4, :].mean()
    assert eq_band > pole_band + 10.0


def test_load_woa_sst_raises_when_synthetic_disabled(tmp_path):
    from legoesm.ocean.forcing import load_woa_sst
    with pytest.raises(FileNotFoundError):
        load_woa_sst(cache_dir=tmp_path, allow_synthetic=False)


def test_woa_into_sst_climatology_bias():
    """Plug WOA loader into the climate-bias diagnostic."""
    from legoesm.ocean.forcing import load_woa_sst
    from legoesm.ocean.diagnostics_climate import sst_climatology_bias
    sst_ref, _, _ = load_woa_sst(nlon=36, nlat=18)
    # Model = WOA + uniform 1 K warm bias.
    sst_model = sst_ref + 1.0
    area = np.ones_like(sst_ref)
    res = sst_climatology_bias(sst_model, sst_ref, area)
    assert res.bias_K == pytest.approx(1.0, abs=1e-12)
    assert res.rmse_K == pytest.approx(1.0, abs=1e-12)


def test_applicator_raises_on_unsupported_grid():
    """Spectral grid type is not supported by the applicator."""
    from legoesm.ocean.coupler import apply_omip2_surface_fluxes
    from legoesm.ocean.forcing import synthetic_ocean_forcing
    state, grid, z, _ = _rest_state_latlon()
    forcing = synthetic_ocean_forcing(2000, n_time=4, nlon=72, nlat=36)
    with pytest.raises(NotImplementedError):
        apply_omip2_surface_fluxes(
            state, forcing=forcing, idx_t=0,
            z_coord=z, grid=grid, grid_type="spectral", dt=1800.0,
        )


def _rest_state(grid_type, res, H_max=5500.0, nlev=10):
    import importlib.util
    repo_root = Path(__file__).resolve().parents[3]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from ocean_test_matrix.setup import _create_ocean_setup
    from ocean_test_matrix.testcase import TestCase
    matrix_mod = _matrix_module()
    tc = TestCase("omip_applicator", grid_type, res, 1.0, 0.1)
    grid, z, _, model, _, _, _ = _create_ocean_setup(
        tc, H_max=H_max, nlev=nlev,
    )
    state = matrix_mod._create_rest_state(tc, grid, z, H_max=H_max)
    return state, grid, z, model


def _uniform_wind_forcing(u_east=8.0, nlat=18, nlon=36):
    """OceanForcing with spatially uniform eastward wind, benign heat/moisture.

    Isolates wind-stress *direction*: a steady eastward wind must drive an
    eastward (positive-u) ocean response over wet cells.
    """
    from legoesm.ocean.forcing.jra55_do import OceanForcing
    lat = np.linspace(-89.0, 89.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)

    def fld(val):
        return np.full((1, nlat, nlon), float(val))

    return OceanForcing(
        lon=lon, lat=lat, time_s=np.array([0.0]),
        u10=fld(u_east), v10=fld(0.0),
        T_air=fld(288.0), q_air=fld(0.008),
        sw_down=fld(0.0), lw_down=fld(0.0),
        precip=fld(0.0), runoff=fld(0.0),
    )


def test_applicator_wind_stress_sign_latlon():
    """Eastward wind must accelerate the ocean EASTWARD (u_top > 0).

    Guards the ``air_sea_fluxes`` atmospheric-convention sign: it returns
    ``tau = -rho_air Cd |U| U`` (opposing the wind), so the ocean feels ``-tau``.
    Applying ``+tau`` (the previous applicator bug) would drive the surface
    WESTWARD -- this test would fail under that bug.
    """
    from legoesm.ocean.coupler import apply_omip2_surface_fluxes
    state, grid, z, _ = _rest_state_latlon()
    forcing = _uniform_wind_forcing(u_east=8.0)
    new = apply_omip2_surface_fluxes(
        state, forcing=forcing, idx_t=0,
        z_coord=z, grid=grid, grid_type="latlon", dt=1800.0,
    )
    u_top = np.asarray(new.u.data)[..., 0]
    v_top = np.asarray(new.v.data)[..., 0]
    u_mask = np.asarray(state.u_mask.data) > 0.5
    assert np.isfinite(u_top).all()
    assert u_mask.any()
    # Eastward wind -> eastward (positive) mean u over wet faces.
    assert u_top[u_mask].mean() > 0.0
    assert np.abs(u_top[u_mask]).max() > 1e-6
    # No meridional wind -> no meridional stress -> v stays at rest.
    assert np.abs(v_top).max() < 1e-9


def test_applicator_tripole_runs_and_sign():
    """Tripole branch: NN-sample 2-D forcing, rotate, apply on the C-grid.

    Uses a synthetic tripole (regular metrics, identity rotation), so an
    eastward wind must give a positive-u response like the lat-lon path.
    """
    from legoesm.ocean.coupler import apply_omip2_surface_fluxes
    from legoesm.grids.tripole import create_synthetic_tripole
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    geom = create_synthetic_tripole(n_lat=36)
    z = create_ocean_z_star(n_levels=10, H_max=5500.0)
    state = rest_state_latlon_cgrid_ocean(geom, z, H_max=5500.0)
    forcing = _uniform_wind_forcing(u_east=8.0)
    new = apply_omip2_surface_fluxes(
        state, forcing=forcing, idx_t=0,
        z_coord=z, grid=geom, grid_type="tripole", dt=1800.0,
    )
    assert type(new) is type(state)
    u_top = np.asarray(new.u.data)[..., 0]
    assert np.isfinite(u_top).all()
    u_mask = np.asarray(state.u_mask.data) > 0.5
    assert u_mask.any()
    assert u_top[u_mask].mean() > 0.0


@pytest.mark.parametrize("grid_type,res", [
    ("cubed_sphere", "C24"),
    ("mpas", "ico3"),
])
def test_applicator_on_cube_and_mpas(grid_type, res):
    """Applicator builds + steps cube + MPAS states without errors."""
    from legoesm.ocean.coupler import apply_omip2_surface_fluxes
    from legoesm.ocean.forcing import synthetic_ocean_forcing
    state, grid, z, _ = _rest_state(grid_type, res)
    forcing = synthetic_ocean_forcing(2000, n_time=4, nlon=72, nlat=36)
    new_state = apply_omip2_surface_fluxes(
        state, forcing=forcing, idx_t=0,
        z_coord=z, grid=grid, grid_type=grid_type, dt=1800.0,
    )
    assert type(new_state) is type(state)
    # u must respond: cube has collocated u, MPAS u on edges.
    u_new = np.asarray(new_state.u.data)
    assert np.isfinite(u_new).all()
    assert np.abs(u_new).max() > 0.0
    # T top-cell must respond.
    T_top = (np.asarray(new_state.T.data)[..., 0]
             if grid_type == "cubed_sphere"
             else np.asarray(new_state.T.data)[:, 0])
    T0_top = (np.asarray(state.T.data)[..., 0]
              if grid_type == "cubed_sphere"
              else np.asarray(state.T.data)[:, 0])
    assert (T_top != T0_top).any()
