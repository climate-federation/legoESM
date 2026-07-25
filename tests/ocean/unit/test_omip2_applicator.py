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
            sys.path.insert(0, str(scripts_dir / "matrix"))
        # Phase-4 script reorg moved the runner into scripts/matrix/; the
        # ocean_test_matrix package stayed at scripts/ (kept on sys.path above).
        # Accept either location so the loader survives an in-flight reorg.
        matrix_path = scripts_dir / "matrix" / "run_ocean_test_matrix.py"
        if not matrix_path.exists():
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
        sys.path.insert(0, str(scripts_dir / "matrix"))
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


def test_compute_omip2_surface_forcing_mpas():
    """compute_omip2_surface_forcing MPAS branch (the LIVE branch the faithful
    run loop uses): cell-centred (nCells,) finite tau/q_net, eastward stress for
    an eastward wind.  Guards the new elif added for the MPAS NEMO comparison."""
    from legoesm.ocean.coupler import compute_omip2_surface_forcing
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    mesh = create_voronoi_mesh(2, lloyd_iterations=2)   # 162 cells, tiny
    z = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_mpas_ocean(mesh, z, H_max=4000.0)
    forcing = _uniform_wind_forcing(u_east=8.0)
    sf = compute_omip2_surface_forcing(
        state, forcing=forcing, idx_t=0, grid=mesh, grid_type="mpas")
    n = mesh.nCells
    for name, f in (("tau_x", sf.tau_x), ("tau_y", sf.tau_y),
                    ("q_net", sf.q_net), ("sw_down", sf.sw_down)):
        arr = np.asarray(f)
        assert arr.shape == (n,), f"{name} shape {arr.shape} != ({n},)"
        assert np.all(np.isfinite(arr)), f"{name} non-finite"
    # tau_x is a definite nonzero zonal stress for a zonal wind...
    assert abs(float(np.mean(np.asarray(sf.tau_x)))) > 1e-3
    # ...and it tracks the wind DIRECTION (convention-agnostic): reversing the
    # wind reverses tau_x.
    sf_rev = compute_omip2_surface_forcing(
        state, forcing=_uniform_wind_forcing(u_east=-8.0), idx_t=0,
        grid=mesh, grid_type="mpas")
    assert (float(np.mean(np.asarray(sf.tau_x)))
            * float(np.mean(np.asarray(sf_rev.tau_x)))) < 0.0


def _emp_forcing(*, precip, q_air, u_east=6.0, nlat=18, nlon=36):
    """OceanForcing isolating the freshwater budget: prescribed precip [kg/m2/s]
    and air humidity ``q_air`` (drives evaporation via q_sfc - q_air), benign
    radiation, steady wind to give a finite exchange coefficient."""
    from legoesm.ocean.forcing.jra55_do import OceanForcing
    lat = np.linspace(-89.0, 89.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)

    def fld(val):
        return np.full((1, nlat, nlon), float(val))

    return OceanForcing(
        lon=lon, lat=lat, time_s=np.array([0.0]),
        u10=fld(u_east), v10=fld(0.0),
        T_air=fld(288.0), q_air=fld(q_air),
        sw_down=fld(0.0), lw_down=fld(0.0),
        precip=fld(precip), runoff=fld(0.0),
    )


def test_compute_omip2_freshwater_forcing_emp():
    """``compute_omip2_freshwater_forcing`` builds a FreshwaterForcing with
    P (prescribed precip), E (interactive NCAR bulk evaporation), and runoff,
    for delivery via the in-core ``model.step(freshwater=...)`` channel.
    Guards the OMIP-2 salt-budget closure (without P - E the ocean freshens
    under runoff alone).

    Checks: (1) precip field == sampled precip and evap == the evap returned
    by the SAME NCAR bulk call (= -lh/L_vap(SST) by construction) to fp tol;
    (2) net P - E freshens under heavy precip / saturated air, salinifies under
    dry air (strong evap); (3) runoff_R enters the runoff channel; (4) emp=False
    zeroes P and E; (5) ramp scales all components.
    """
    import jax.numpy as jnp
    from legoesm.ocean.coupler import compute_omip2_freshwater_forcing
    from legoesm.ocean.coupler.omip2_applicator import (
        air_sea_fluxes, _sample_forcing_latlon,
    )
    from legoesm.ocean.freshwater import net_freshwater_flux
    from legoesm import constants
    state, grid, z, _ = _rest_state_latlon()

    # (1) Exactness: precip + evap == the independently reconstructed bulk path.
    precip0 = 5.0e-5
    forcing = _emp_forcing(precip=precip0, q_air=0.010)
    fw = compute_omip2_freshwater_forcing(
        state, forcing=forcing, idx_t=0, grid=grid, grid_type="latlon")
    P = np.asarray(fw.precip)
    E = np.asarray(fw.evap)
    assert P.shape == np.asarray(state.T.data)[..., 0].shape
    assert np.isfinite(P).all() and np.isfinite(E).all()

    lat_deg = np.degrees(np.asarray(grid.lat))
    lon_deg = np.degrees(np.asarray(grid.lon))
    forc = _sample_forcing_latlon(forcing, 0, lat_deg, lon_deg)
    T_sfc_K = np.asarray(state.T.data, dtype=np.float64)[..., 0] + constants.T_freeze
    _, _, _, lh, evap_ref = air_sea_fluxes(
        u10=jnp.asarray(forc["u10"]), v10=jnp.asarray(forc["v10"]),
        T_air_K=jnp.asarray(forc["T_air"]), q_air=jnp.asarray(forc["q_air"]),
        T_sfc_K=jnp.asarray(T_sfc_K))
    assert np.allclose(P, np.asarray(forc["precip"]), rtol=1e-6, atol=1e-12)
    assert np.allclose(E, np.asarray(evap_ref), rtol=1e-6, atol=1e-12)
    # evap and lh are mutually consistent through L_vap at the POTENTIAL
    # SST (NEMO BULK_FORMULA pTs = zsspt; ~0.1% below L_vap(SST_abs)).
    from legoesm.ocean.bulk_flux_omip import (
        exner_potential_temperature, latent_heat_vaporization_sst,
    )
    theta_sst = exner_potential_temperature(
        jnp.asarray(T_sfc_K), jnp.asarray(float(constants.p_atm_std)))
    L_vap = np.asarray(latent_heat_vaporization_sst(theta_sst))
    assert np.allclose(np.asarray(evap_ref), -np.asarray(lh) / L_vap,
                       rtol=1e-9, atol=1e-15)

    # (2) net = P - E: heavy precip / saturated air -> freshening (>0); dry air
    # / no precip -> strong evap -> salinifying (<0).
    fw_wet = compute_omip2_freshwater_forcing(
        state, forcing=_emp_forcing(precip=2.0e-4, q_air=0.020),
        idx_t=0, grid=grid, grid_type="latlon")
    assert float(np.mean(np.asarray(net_freshwater_flux(fw_wet)))) > 0.0
    fw_dry = compute_omip2_freshwater_forcing(
        state, forcing=_emp_forcing(precip=0.0, q_air=0.001),
        idx_t=0, grid=grid, grid_type="latlon")
    assert float(np.mean(np.asarray(net_freshwater_flux(fw_dry)))) < 0.0

    # (3) runoff_R flows into the runoff channel (additive, + into ocean).
    R = np.full(P.shape, 1.0e-5)
    fw_r = compute_omip2_freshwater_forcing(
        state, forcing=forcing, idx_t=0, grid=grid, grid_type="latlon",
        runoff_R=R)
    assert np.allclose(np.asarray(fw_r.runoff), R, rtol=1e-10)

    # (4) emp=False zeroes P and E but keeps runoff.
    fw_noemp = compute_omip2_freshwater_forcing(
        state, forcing=forcing, idx_t=0, grid=grid, grid_type="latlon",
        runoff_R=R, emp=False)
    assert np.all(np.asarray(fw_noemp.precip) == 0.0)
    assert np.all(np.asarray(fw_noemp.evap) == 0.0)
    assert np.allclose(np.asarray(fw_noemp.runoff), R, rtol=1e-10)

    # (5) ramp scales every component linearly.
    fw_half = compute_omip2_freshwater_forcing(
        state, forcing=forcing, idx_t=0, grid=grid, grid_type="latlon",
        runoff_R=R, ramp=0.5)
    assert np.allclose(np.asarray(fw_half.precip), 0.5 * P, rtol=1e-10)
    assert np.allclose(np.asarray(fw_half.evap), 0.5 * E, rtol=1e-10)
    assert np.allclose(np.asarray(fw_half.runoff), 0.5 * R, rtol=1e-10)


@pytest.mark.parametrize("grid_type,res", [
    ("cubed_sphere", "C24"),
    ("mpas", "ico3"),
])
def test_compute_omip2_freshwater_forcing_grid_routing(grid_type, res):
    """``compute_omip2_freshwater_forcing`` routes through ``_sample_omip2_forcing``
    on the cube ((6,n,n)) and MPAS ((nCells,)) grids: shape matches the surface
    field, fields are finite, and the net P - E sign tracks the forcing (dry air
    -> net salinifying < 0; heavy precip -> net freshening > 0).  Guards the
    grid-routing the run loop relies on (cube folds net onto sf.freshwater; MPAS
    passes the FreshwaterForcing to step(freshwater=))."""
    from legoesm.ocean.coupler import compute_omip2_freshwater_forcing
    from legoesm.ocean.freshwater import net_freshwater_flux
    state, grid, z, _ = _rest_state(grid_type, res)
    surf_shape = (np.asarray(state.T.data)[..., 0].shape if grid_type == "cubed_sphere"
                  else np.asarray(state.T.data)[:, 0].shape)

    fw_dry = compute_omip2_freshwater_forcing(
        state, forcing=_emp_forcing(precip=0.0, q_air=0.001),
        idx_t=0, grid=grid, grid_type=grid_type)
    fw_wet = compute_omip2_freshwater_forcing(
        state, forcing=_emp_forcing(precip=2.0e-4, q_air=0.020),
        idx_t=0, grid=grid, grid_type=grid_type)
    for fw in (fw_dry, fw_wet):
        assert np.asarray(fw.precip).shape == surf_shape
        assert np.isfinite(np.asarray(net_freshwater_flux(fw))).all()
    assert float(np.mean(np.asarray(net_freshwater_flux(fw_dry)))) < 0.0
    assert float(np.mean(np.asarray(net_freshwater_flux(fw_wet)))) > 0.0


def _sw_forcing(*, sw=250.0, nlat=18, nlon=36):
    """OceanForcing with uniform downwelling shortwave (isolates the SW albedo)."""
    from legoesm.ocean.forcing.jra55_do import OceanForcing
    f = lambda v: np.full((1, nlat, nlon), float(v))
    return OceanForcing(
        lon=np.linspace(0.0, 360.0, nlon, endpoint=False),
        lat=np.linspace(-89.0, 89.0, nlat), time_s=np.array([0.0]),
        u10=f(4.0), v10=f(0.0), T_air=f(288.0), q_air=f(0.010),
        sw_down=f(sw), lw_down=f(0.0), precip=f(0.0), runoff=f(0.0))


def test_ice_albedo_reduces_sw_and_q_net():
    """``ice_albedo`` (prescribed siconc) reduces the returned sw_down AND q_net by
    the effective albedo a_ocean*(1-siconc)+a_ice*siconc, applied ONCE.  Checks the
    None=identity default, the siconc=0 (open-ocean 1-a_ocean) and siconc=1
    (1-a_ice) limits, the 0<=sw_net<=sw_down bound, and that more ice -> less SW."""
    from legoesm.ocean.coupler import compute_omip2_surface_forcing
    from legoesm import constants
    state, grid, z, _ = _rest_state_latlon()
    forcing = _sw_forcing(sw=250.0)
    shp = np.asarray(state.T.data)[..., 0].shape

    sf_none = compute_omip2_surface_forcing(
        state, forcing=forcing, idx_t=0, grid=grid, grid_type="latlon")
    sw_full = np.asarray(sf_none.sw_down)
    q_full = np.asarray(sf_none.q_net)
    assert (sw_full > 0).any()

    def sf(sic):
        return compute_omip2_surface_forcing(
            state, forcing=forcing, idx_t=0, grid=grid, grid_type="latlon",
            ice_albedo=np.full(shp, float(sic)))
    a_oc = float(constants.alpha_ocean_broadband)
    a_ice = float(constants.alpha_ice_broadband_cold)

    sf0, sf1, sfh = sf(0.0), sf(1.0), sf(0.5)
    # siconc=0 -> open-ocean albedo: sw_net = (1-a_ocean)*sw_full.
    assert np.allclose(np.asarray(sf0.sw_down), (1.0 - a_oc) * sw_full, rtol=1e-9)
    # siconc=1 -> sea-ice albedo: sw_net = (1-a_ice)*sw_full.
    assert np.allclose(np.asarray(sf1.sw_down), (1.0 - a_ice) * sw_full, rtol=1e-9)
    # siconc=0.5 -> linear blend.
    assert np.allclose(np.asarray(sfh.sw_down),
                       (1.0 - 0.5 * (a_oc + a_ice)) * sw_full, rtol=1e-9)
    # q_net drops by exactly the SW reduction (other terms unchanged).
    assert np.allclose(np.asarray(sf1.q_net) - q_full,
                       (1.0 - a_ice) * sw_full - sw_full, rtol=1e-9, atol=1e-9)
    # More ice -> strictly less absorbed SW; bound 0 <= sw_net <= sw_full.
    assert float(np.asarray(sf1.sw_down).mean()) < float(np.asarray(sf0.sw_down).mean())
    for s in (sf0, sf1, sfh):
        sn = np.asarray(s.sw_down)
        assert (sn >= -1e-9).all() and (sn <= sw_full + 1e-9).all()
    # None default = NO albedo: the returned sw_down is the sampled SW unchanged
    # (compare to the directly-sampled forcing, not a literal -- the conservative
    # regrid produces edge-cell partial values, e.g. a 125 at the grid boundary).
    from legoesm.ocean.coupler.omip2_applicator import _sample_omip2_forcing
    forc = _sample_omip2_forcing(forcing, 0, grid, "latlon")
    assert np.allclose(sw_full, np.asarray(forc["sw_down"]), rtol=1e-12)


# ---------------------------------------------------------------------------
# Prescribed-ice thermodynamic boundary (--ice-thermo) — codex HIGH dual-pole
# fix.  Pure helpers, testable without forcing/grid.
# ---------------------------------------------------------------------------

def test_ice_surface_heat_regimes():
    from legoesm.ocean.coupler.omip2_applicator import _ice_surface_heat
    from legoesm import constants
    a_oc = float(constants.alpha_ocean_broadband)
    sw = np.full((4,), 200.0)
    qn = np.full((4,), -50.0)            # net non-SW (e.g. ocean losing heat)

    # None -> legacy: no albedo, full non-SW.
    swo, q = _ice_surface_heat(sw, qn, None)
    assert np.allclose(swo, sw) and np.allclose(q, qn + sw)

    # under_ice=False, sic given -> albedo-only surrogate (matches _sw_albedo_factor).
    swo, q = _ice_surface_heat(sw, qn, np.full((4,), 1.0), under_ice=False)
    a_ice = float(constants.alpha_ice_broadband_cold)
    assert np.allclose(swo, sw * (1 - a_ice))
    assert np.allclose(q, qn + sw * (1 - a_ice))     # full non-SW retained

    # under_ice=True, sic=0 -> open water: same as albedo-only open water.
    swo0, q0 = _ice_surface_heat(sw, qn, np.zeros((4,)), under_ice=True)
    assert np.allclose(swo0, sw * (1 - a_oc))
    assert np.allclose(q0, sw * (1 - a_oc) + qn)

    # under_ice=True, sic=1 -> tiny SW + non-SW SUPPRESSED.
    swo1, q1 = _ice_surface_heat(sw, qn, np.ones((4,)), under_ice=True,
                                 tau_ice_sw=0.03)
    assert np.allclose(swo1, sw * 0.03)
    assert np.allclose(q1, sw * 0.03)                # (1-sic)*qn = 0
    # Under-ice ocean gets MUCH less heat than the albedo-only surrogate would
    # (0.35*sw + qn): this is the SH-warm-bias correction.
    assert float(q1.mean()) < float((sw * (1 - a_ice) + qn).mean())


def test_under_ice_freeze_relax_two_sided():
    from legoesm.ocean.coupler.omip2_applicator import under_ice_freeze_relax
    from legoesm import constants
    Tf = float(constants.T_freeze_ocean) - float(constants.T_freeze)   # ~ -1.8 C
    day = 86400.0

    # sic=0 -> untouched.
    T = np.array([3.0, -3.0, 10.0])
    assert np.allclose(under_ice_freeze_relax(T, np.zeros(3), day), T)

    # sic=1, warm cell -> COOLED toward freezing; cold (super-cooled) -> WARMED up.
    T = np.array([3.0, -3.0])
    out = under_ice_freeze_relax(T, np.ones(2), day, tau_ice_days=20.0)
    assert out[0] < 3.0 and out[0] > Tf          # warm cell cooled toward Tf
    assert out[1] > -3.0 and out[1] < Tf         # super-cooled cell warmed up to Tf
    # Convergence: many days at sic=1 -> Tf.
    Tc = np.array([5.0])
    for _ in range(2000):
        Tc = under_ice_freeze_relax(Tc, np.ones(1), day, tau_ice_days=20.0)
    assert abs(float(Tc[0]) - Tf) < 1e-3
    # dt >> tau -> clipped to full relaxation (no overshoot/instability).
    big = under_ice_freeze_relax(np.array([5.0]), np.ones(1), 100 * day,
                                 tau_ice_days=20.0)
    assert abs(float(big[0]) - Tf) < 1e-9


# ---------------------------------------------------------------------------
# NEMO-form q_ns assembly (Kirchhoff LW, snow fusion, heat-content terms)
# ---------------------------------------------------------------------------

def _qns_forcing(*, lw_down=0.0, precip=0.0, snow=None, slp=None,
                 q_air=0.010, T_air=288.0, nlat=18, nlon=36):
    """OceanForcing for q_ns-assembly tests (zero SW; optional snow/slp)."""
    from legoesm.ocean.forcing.jra55_do import OceanForcing
    lat = np.linspace(-89.0, 89.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)

    def fld(val):
        return np.full((1, nlat, nlon), float(val))

    return OceanForcing(
        lon=lon, lat=lat, time_s=np.array([0.0]),
        u10=fld(6.0), v10=fld(0.0),
        T_air=fld(T_air), q_air=fld(q_air),
        sw_down=fld(0.0), lw_down=fld(lw_down),
        precip=fld(precip), runoff=fld(0.0),
        snow=None if snow is None else fld(snow),
        slp=None if slp is None else fld(slp),
    )


def test_qnet_longwave_is_kirchhoff_form():
    """Delta(q_net) / Delta(LW_down) == emissivity_seawater_lw (0.98), NOT 1.0:
    the NEMO net-LW form weights ABSORPTION by the same eps as emission.
    The old assembly absorbed 100% of LW_down (-> ~+11 W/m2 spurious warming)."""
    from legoesm.ocean.coupler import compute_omip2_surface_forcing
    from legoesm import constants
    state, grid, z, _ = _rest_state_latlon()
    sf_a = compute_omip2_surface_forcing(
        state, forcing=_qns_forcing(lw_down=300.0), idx_t=0,
        grid=grid, grid_type="latlon")
    sf_b = compute_omip2_surface_forcing(
        state, forcing=_qns_forcing(lw_down=400.0), idx_t=0,
        grid=grid, grid_type="latlon")
    dq = np.asarray(sf_b.q_net) - np.asarray(sf_a.q_net)
    assert np.allclose(dq, constants.emissivity_seawater_lw * 100.0,
                       rtol=1e-9, atol=1e-9)


def test_qnet_snow_fusion_and_heat_content():
    """Adding snow at fixed TOTAL precip changes q_net by exactly
    s*(-L_fus + c_p_ice*min(thetaC,0) - c_p_seawater*thetaC): the fusion
    sink plus swapping rain heat content for snow heat content (NEMO
    blk_oce_2 form, heat-content terms at the POTENTIAL air temperature
    and NEMO rLfus/rcpi values)."""
    import jax.numpy as jnp
    from legoesm.ocean.coupler import compute_omip2_surface_forcing
    from legoesm.ocean.bulk_flux_omip import potential_air_temperature_10m
    from legoesm import constants
    state, grid, z, _ = _rest_state_latlon()
    P, S = 2.0e-4, 1.0e-4
    T_air = 275.0
    sf_rain = compute_omip2_surface_forcing(
        state, forcing=_qns_forcing(precip=P, snow=0.0, T_air=T_air),
        idx_t=0, grid=grid, grid_type="latlon")
    sf_snow = compute_omip2_surface_forcing(
        state, forcing=_qns_forcing(precip=P, snow=S, T_air=T_air),
        idx_t=0, grid=grid, grid_type="latlon")
    theta_air, _ = potential_air_temperature_10m(
        jnp.asarray(T_air), jnp.asarray(0.010))
    theta_C = float(theta_air) - float(constants.T_freeze)
    expected = S * (-float(constants.L_fus_nemo)
                    + float(constants.c_p_ice_nemo) * min(theta_C, 0.0)
                    - float(constants.c_p_seawater) * theta_C)
    dq = np.asarray(sf_snow.q_net) - np.asarray(sf_rain.q_net)
    assert np.allclose(dq, expected, rtol=1e-9, atol=1e-9)
    # snow COOLS: the net effect must be negative at any realistic T_air
    assert expected < 0.0


def test_qnet_slp_channel_changes_density_and_qsat():
    """Providing slp != standard atmosphere shifts the turbulent fluxes via
    rho_air(slp) and ssq(slp) -- guards that the optional channel is actually
    consumed (a dropped channel would leave q_net bit-identical)."""
    from legoesm.ocean.coupler import compute_omip2_surface_forcing
    state, grid, z, _ = _rest_state_latlon()
    sf_std = compute_omip2_surface_forcing(
        state, forcing=_qns_forcing(q_air=0.004), idx_t=0,
        grid=grid, grid_type="latlon")
    sf_low = compute_omip2_surface_forcing(
        state, forcing=_qns_forcing(q_air=0.004, slp=96000.0), idx_t=0,
        grid=grid, grid_type="latlon")
    assert not np.allclose(np.asarray(sf_std.q_net),
                           np.asarray(sf_low.q_net), rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# scaling-M2 honest-cost contract (codex batch4 HIGH): the per-step forcing
# builders must (a) slice the state leaf BEFORE host conversion — only the
# 2-D surface layer may cross to host, never the full 3-D (possibly lat-band-
# sharded) field — and (b) record each pull in the module host-pull ledger so
# run_omip_core2's persistent lane reports the true leaf-transfer cost.
# ---------------------------------------------------------------------------

class _NoFullHostConvertLeaf:
    """State-leaf stand-in whose FULL-array host conversion raises.

    Encodes the slice-before-convert contract mechanically: ``np.asarray`` on
    the whole leaf (the codex batch4 HIGH pattern
    ``np.asarray(state.T.data)[..., 0]``) fails loudly, while the
    ``[..., 0]`` surface child is a plain numpy slice and converts fine.
    """

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def __getitem__(self, idx):
        return self._arr[idx]

    def __array__(self, dtype=None, copy=None):
        raise AssertionError(
            "full-3-D host conversion of a state leaf in the per-step "
            "forcing path (slice-before-convert contract violated)")

    @property
    def ndim(self):
        return self._arr.ndim

    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype


def _duck_state_with_guarded_temp(t3d):
    """Minimal duck state: the builders read ONLY ``state.T.data``."""
    from types import SimpleNamespace
    return SimpleNamespace(T=SimpleNamespace(data=_NoFullHostConvertLeaf(t3d)))


def test_forcing_builders_count_surface_slice_pulls():
    """Each forcing-builder call records exactly ONE surface-slice host pull
    in the module ledger (the persistent-sharded driver reads the delta for
    its honest-cost done line)."""
    from legoesm.ocean.coupler import (
        compute_omip2_freshwater_forcing,
        compute_omip2_surface_forcing,
    )
    from legoesm.ocean.coupler.omip2_applicator import host_pull_ledger
    state, grid, z, _ = _rest_state_latlon()
    forcing = _uniform_wind_forcing(u_east=8.0)

    n0 = host_pull_ledger()["surface_slice_pulls"]
    compute_omip2_surface_forcing(
        state, forcing=forcing, idx_t=0, grid=grid, grid_type="latlon")
    n1 = host_pull_ledger()["surface_slice_pulls"]
    assert n1 - n0 == 1, "surface-forcing build must record ONE surface pull"

    compute_omip2_freshwater_forcing(
        state, forcing=forcing, idx_t=0, grid=grid, grid_type="latlon")
    n2 = host_pull_ledger()["surface_slice_pulls"]
    assert n2 - n1 == 1, "freshwater build must record ONE surface pull"
    # accessor returns a COPY: mutating it must not corrupt the ledger
    snap = host_pull_ledger()
    snap["surface_slice_pulls"] = -999
    assert host_pull_ledger()["surface_slice_pulls"] == n2


def test_forcing_builders_slice_before_convert():
    """The builders must never host-convert the FULL 3-D T leaf: with a leaf
    whose full-array conversion raises, both builders still run — and produce
    bit-identical fields to the plain-state call (the slice path is the same
    data)."""
    from legoesm.ocean.coupler import (
        compute_omip2_freshwater_forcing,
        compute_omip2_surface_forcing,
    )
    state, grid, z, _ = _rest_state_latlon()
    forcing = _uniform_wind_forcing(u_east=8.0)
    t3d = np.asarray(state.T.data)
    guarded = _duck_state_with_guarded_temp(t3d)

    sf_ref = compute_omip2_surface_forcing(
        state, forcing=forcing, idx_t=0, grid=grid, grid_type="latlon")
    sf_g = compute_omip2_surface_forcing(
        guarded, forcing=forcing, idx_t=0, grid=grid, grid_type="latlon")
    for name in ("tau_x", "tau_y", "q_net", "sw_down"):
        np.testing.assert_array_equal(
            np.asarray(getattr(sf_g, name)), np.asarray(getattr(sf_ref, name)),
            err_msg=f"guarded-leaf surface forcing diverged on {name}")

    fw_ref = compute_omip2_freshwater_forcing(
        state, forcing=forcing, idx_t=0, grid=grid, grid_type="latlon")
    fw_g = compute_omip2_freshwater_forcing(
        guarded, forcing=forcing, idx_t=0, grid=grid, grid_type="latlon")
    for name in ("precip", "evap", "runoff"):
        np.testing.assert_array_equal(
            np.asarray(getattr(fw_g, name)), np.asarray(getattr(fw_ref, name)),
            err_msg=f"guarded-leaf freshwater forcing diverged on {name}")


def test_no_full_host_convert_guard_is_not_vacuous():
    """Tripwire self-test: the guard leaf really does refuse a full-array
    conversion (so the contract test above cannot silently pass vacuously)."""
    guard = _NoFullHostConvertLeaf(np.zeros((4, 5, 3)))
    with pytest.raises(AssertionError, match="slice-before-convert"):
        np.asarray(guard)
    # ... while the sliced surface child converts fine and is the right slab.
    child = np.asarray(guard[..., 0])
    assert child.shape == (4, 5)


def _uniform_centres(n, lo, hi):
    """Centres of ``n`` uniform cells tiling ``[lo, hi]`` degrees.

    ``_edges_from_centers_deg`` re-derives the edges as ``c -/+ dc/2``, so the
    reconstructed outermost edges land exactly on ``lo``/``hi``.
    """
    half = 0.5 * (hi - lo) / n
    return np.linspace(lo + half, hi - half, n)


def test_conservative_regrid_seam_full_coverage_high_ratio():
    """A coarse dst cell that STRADDLES the 0/360 seam (dst 8x wider than
    src in lon, so a single ghost cannot span its half-width) must get full
    source coverage: a constant source regrids to the same constant even in
    the seam-straddling column. RED under the old single-ghost pad (seam
    cell weight-sum < 1 -> constant returns ~0.625 there); GREEN with
    ceil(dd/ds) ghosts."""
    from legoesm.ocean.coupler.omip2_applicator import (
        _conservative_regrid_to_latlon,
    )
    # src 8x deg-finer than dst in lon (ratio 8 so 1 ghost under-covers).
    # Latitudes ASCENDING (south->north): compute_overlap_weights requires
    # strictly-increasing edges (see _check_edges).
    n_src_lat, n_src_lon = 8, 64
    src_lat = _uniform_centres(n_src_lat, -90.0, 90.0)
    src_lon = _uniform_centres(n_src_lon, 0.0, 360.0)
    # dst centred ON the seam: first centre at 0 -> cell [-22.5, 22.5]
    # straddles the 360/0 wrap. dd = 45, ds = 5.625 -> dd/ds = 8.
    n_dst_lat, n_dst_lon = 4, 8
    dst_lat = _uniform_centres(n_dst_lat, -90.0, 90.0)
    dd = 360.0 / n_dst_lon
    dst_lon = np.linspace(0.0, 360.0 - dd, n_dst_lon)
    field = np.ones((n_src_lat, n_src_lon))
    out = _conservative_regrid_to_latlon(
        field, src_lat, src_lon, dst_lat, dst_lon,
    )
    np.testing.assert_allclose(out, 1.0, atol=1e-6)


def test_conservative_regrid_allows_real_non_polar_source():
    """A source whose outermost latitude row stops short of +-90 -- i.e. every
    real forcing dataset (CORE-II +-88.5, JRA55-do ~+-89.6) -- must regrid
    finitely, not raise, even though this caller sets
    ``require_attainable_coverage=True`` -- the reference is what the source CAN
    supply, so the physical polar taper is allowed while a real deficit raises.
    Fully-covered interior rows still reproduce a constant exactly, while the
    polar rows come back REDUCED, which is the physical answer.
    See regrid_polar_coverage_2026-07-24.md."""
    from legoesm.ocean.coupler.omip2_applicator import (
        _REGRID_WEIGHTS_CACHE, _conservative_regrid_to_latlon,
    )
    # The pin is that the WEIGHT BUILD does not raise, so it must actually run:
    # a warm cache entry under a colliding (shape, first-centre) key would make
    # this test vacuous.
    _REGRID_WEIGHTS_CACHE.clear()
    # src cells tile [-80, 80] only, so the reconstructed src edges stop 10 deg
    # short of the pole and the lat clamp is a no-op -- the dst polar rows
    # ([-90, -45] and [45, 90]) are then genuinely under-covered.
    n_src_lat, n_src_lon = 8, 16
    src_lat = _uniform_centres(n_src_lat, -80.0, 80.0)
    src_lon = _uniform_centres(n_src_lon, 0.0, 360.0)
    n_dst_lat, n_dst_lon = 4, 8
    dst_lat = _uniform_centres(n_dst_lat, -90.0, 90.0)
    dst_lon = _uniform_centres(n_dst_lon, 0.0, 360.0)
    field = np.ones((n_src_lat, n_src_lon))
    out = _conservative_regrid_to_latlon(
        field, src_lat, src_lon, dst_lat, dst_lon,
    )
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[1:-1, :], 1.0, atol=1e-6)
    # Pins that the geometry actually exercises the premise: without a genuine
    # polar deficit this test would pass even with the flag re-enabled.
    assert np.all(out[[0, -1], :] < 0.99), out[[0, -1], :]


def test_conservative_regrid_rejects_partial_longitude_source():
    """The raw-longitude PRECONDITION must be wired into this caller, not just
    available in the helper.

    The +-360 ghost pad below it would turn a source tiling only half the circle
    into one enormous cell spanning the whole missing sector, which then reports
    COMPLETE longitude coverage to the weight builder -- so the check has to run on
    the RAW axis, before the pad. Without this test, deleting that call leaves the
    whole suite green.
    """
    from legoesm.ocean.coupler.omip2_applicator import (
        _REGRID_WEIGHTS_CACHE, _conservative_regrid_to_latlon,
    )
    _REGRID_WEIGHTS_CACHE.clear()
    n_src_lat, n_src_lon = 8, 16
    src_lat = _uniform_centres(n_src_lat, -90.0, 90.0)
    src_lon = _uniform_centres(n_src_lon, 0.0, 180.0)      # HALF the circle
    dst_lat = _uniform_centres(4, -90.0, 90.0)
    dst_lon = _uniform_centres(8, 0.0, 360.0)
    field = np.ones((n_src_lat, n_src_lon))
    with pytest.raises(ValueError, match="source longitude"):
        _conservative_regrid_to_latlon(
            field, src_lat, src_lon, dst_lat, dst_lon,
        )
    # Anti-vacuity: the same call with a globe-tiling source builds fine.
    _REGRID_WEIGHTS_CACHE.clear()
    out = _conservative_regrid_to_latlon(
        field, src_lat, _uniform_centres(n_src_lon, 0.0, 360.0), dst_lat, dst_lon,
    )
    assert np.all(np.isfinite(out))
