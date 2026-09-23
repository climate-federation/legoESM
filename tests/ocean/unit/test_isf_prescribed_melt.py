"""Tests for the NEMO ISF 'spe' prescribed melt (isfparmlt spe case).

1. **Column budgets** — heat: ρ0·c_sw·Σ dT·h = fwf·(c_sw·T_frz − L_fus);
   salt: ρ0·Σ dS·h = −fwf·⟨S⟩_band; volume: η̇ = fwf/ρ0.
2. **Band distribution** — tendencies live only in cells overlapping
   [zmin, zmax], weighted by geometric overlap; zmin clamp; deep zmax
   truncates at the seafloor.
3. **Signs** — melt COOLS (latent dominates) and FRESHENS the band,
   RAISES eta (mandatory sign-convention walk).
4. **No-op** — fwf = 0 returns exactly zero tendencies; dry columns inert.
5. **Loader** — monthly Depoorter-layout file: passthrough, regrid melt
   totals preserved, zmin clamp / band floor, land mask.
6. **Applicator** — state update T/S/eta matches dt × tendencies.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import nemo_eos_fzp
from legoesm.ocean.physics.ice_shelf import (
    IceShelfConfig,
    isf_prescribed_melt_tendencies,
)


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


RHO0 = 1026.0
CP = float(constants.c_p_seawater)      # NEMO rcp
LF = float(constants.L_fus_isf_nemo)    # NEMO isf_oce rLfusisf


def _column(nlev=8, H=1600.0):
    dz = np.full(nlev, H / nlev)
    S = np.linspace(34.0, 34.8, nlev)
    wet = np.ones(nlev)
    return dz, S, wet


def _isftbl_reference(dz, zmin, zmax):
    """Scalar transliteration of isf_tbl_ktop + isf_tbl_lvl (isftbl.F90):
    returns (weights w_k with sum = htbl, htbl)."""
    z_bot = np.cumsum(dz)
    z_top = z_bot - dz
    col = z_bot[-1]
    zmax_c = min(zmax, col)
    ktop = int(np.sum(z_top <= zmin)) - 1          # cell containing zmin
    zmin_snap = z_top[ktop]                        # snapped UP to its top
    htbl = max(min(zmax_c - zmin_snap, col - zmin_snap), dz[ktop])
    kbot = ktop + 1
    while np.sum(dz[ktop:kbot]) < htbl - 1e-9:
        kbot += 1
    kbot -= 1
    w = np.zeros_like(dz)
    w[ktop:kbot] = dz[ktop:kbot]
    w[kbot] = htbl - dz[ktop:kbot].sum()
    return w, htbl


def test_column_budgets_close():
    dz, S, wet = _column()
    fwf = 5.0e-4
    zmin, zmax = 150.0, 750.0
    dT, dS, eta_dot = isf_prescribed_melt_tendencies(
        jnp.asarray(S), jnp.asarray(dz), jnp.asarray(wet),
        jnp.asarray(fwf), jnp.asarray(zmin), jnp.asarray(zmax),
        rho_0=RHO0, c_sw=CP, L_fus=LF)
    dT, dS = np.asarray(dT), np.asarray(dS)
    # NEMO isftbl weights (zmin=150 in cell0 -> snap to 0; htbl=750;
    # cells 0..2 full + 0.75 of cell 3)
    w, htbl = _isftbl_reference(dz, zmin, zmax)
    assert htbl == pytest.approx(750.0)
    np.testing.assert_allclose(w, [200, 200, 200, 150, 0, 0, 0, 0])
    z_bot = np.cumsum(dz)
    z_top = z_bot - dz
    tfrz_lvl = np.asarray(nemo_eos_fzp(
        jnp.asarray(S), jnp.asarray(0.5 * (z_top + z_bot))))
    T_frz = float((tfrz_lvl * w).sum() / htbl)
    S_w = float((S * w).sum() / htbl)
    # heat budget
    np.testing.assert_allclose(
        RHO0 * CP * (dT * dz).sum(), fwf * (CP * T_frz - LF), rtol=1e-12)
    # salt budget
    np.testing.assert_allclose(
        RHO0 * (dS * dz).sum(), -fwf * S_w, rtol=1e-12)
    # volume
    np.testing.assert_allclose(float(eta_dot), fwf / RHO0, rtol=1e-14)
    # distribution: outside-TBL cells exactly zero; FULL cells share one
    # uniform dT; the fractional bottom cell carries frac x that.
    outside = w == 0.0
    assert np.all(dT[outside] == 0.0) and np.all(dS[outside] == 0.0)
    np.testing.assert_allclose(dT[1], dT[0], rtol=1e-12)
    np.testing.assert_allclose(dT[3], 0.75 * dT[0], rtol=1e-12)
    # signs: latent dominates -> cooling; dilution -> freshening; eta rises.
    assert np.all(dT[~outside] < 0.0)
    assert np.all(dS[~outside] < 0.0)
    assert float(eta_dot) > 0.0


def test_zmin_snaps_to_cell_top_interface():
    """isf_tbl_ktop snaps zmin UP to the containing cell's top interface —
    a mid-cell zmin=250 on dz=200 columns starts the TBL at 200 m with
    cell 1 FULL (not geometrically overlapped)."""
    dz, S, wet = _column()          # dz = 200 uniform, H = 1600
    dT, _, _ = isf_prescribed_melt_tendencies(
        jnp.asarray(S), jnp.asarray(dz), jnp.asarray(wet),
        jnp.asarray(5.0e-4), jnp.asarray(250.0), jnp.asarray(750.0),
        rho_0=RHO0, c_sw=CP, L_fus=LF)
    dT = np.asarray(dT)
    w, htbl = _isftbl_reference(dz, 250.0, 750.0)
    assert htbl == pytest.approx(550.0)          # 750 - snapped 200
    np.testing.assert_allclose(w, [0, 200, 200, 150, 0, 0, 0, 0])
    assert dT[0] == 0.0
    np.testing.assert_allclose(dT[2], dT[1], rtol=1e-12)
    np.testing.assert_allclose(dT[3], 0.75 * dT[1], rtol=1e-12)


def test_band_clamps_at_surface_and_seafloor():
    dz, S, wet = _column(nlev=5, H=500.0)
    # zmin negative in the source file → clamp handled by the LOADER; the
    # tendency fn itself just intersects with the water column: a band
    # deeper than the seafloor deposits only down to the last wet cell.
    dT, dS, eta_dot = isf_prescribed_melt_tendencies(
        jnp.asarray(S), jnp.asarray(dz), jnp.asarray(wet),
        jnp.asarray(1.0e-4), jnp.asarray(400.0), jnp.asarray(2000.0),
        rho_0=RHO0, c_sw=CP, L_fus=LF)
    dT = np.asarray(dT)
    assert np.all(dT[:4] == 0.0)       # above the band
    assert dT[4] < 0.0                 # only the deepest cell takes it
    assert float(eta_dot) > 0.0


def test_zero_fwf_and_dry_column_noop():
    dz, S, wet = _column(nlev=4, H=400.0)
    for fwf, wetc in [(0.0, wet), (1e-4, np.zeros_like(wet))]:
        dT, dS, eta_dot = isf_prescribed_melt_tendencies(
            jnp.asarray(S), jnp.asarray(dz), jnp.asarray(wetc),
            jnp.asarray(fwf), jnp.asarray(100.0), jnp.asarray(300.0),
            rho_0=RHO0, c_sw=CP, L_fus=LF)
        assert float(jnp.max(jnp.abs(dT))) == 0.0
        assert float(jnp.max(jnp.abs(dS))) == 0.0
        assert float(eta_dot) == 0.0


def test_applicator_updates_state():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.coupler.ice_shelf_apply import (
        apply_isf_prescribed_melt_step,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=4, n_lon=6)
    z = create_ocean_z_star(n_levels=4, H_max=2000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=2.0, T_deep=1.0, S_uniform=34.5)
    shape2d = state.T.data.shape[:-1]
    fwf = np.zeros(shape2d)
    fwf[1, 2] = 3.0e-4
    dz_live = np.broadcast_to(
        np.asarray(z.dz_ref) * float(state.H_bathy.data[0, 0]) / 2000.0,
        state.T.data.shape)
    wet = np.ones_like(dz_live)
    dt = 1800.0
    s1 = apply_isf_prescribed_melt_step(
        state, fwf_kg_m2_s=jnp.asarray(fwf),
        zmin_m=jnp.asarray(np.full(shape2d, 100.0)),
        zmax_m=jnp.asarray(np.full(shape2d, 800.0)),
        dz_live=jnp.asarray(dz_live), wet_cell=jnp.asarray(wet),
        dt=dt, rho_0=RHO0)
    dT = np.asarray(s1.T.data) - np.asarray(state.T.data)
    dS = np.asarray(s1.S.data) - np.asarray(state.S.data)
    dEta = np.asarray(s1.eta.data) - np.asarray(state.eta.data)
    # only the melting column changes; it cools + freshens; eta rises there.
    changed = np.abs(dT).sum(axis=-1) > 0
    assert changed[1, 2] and changed.sum() == 1
    assert dT[1, 2].min() < 0 and dS[1, 2].min() < 0
    np.testing.assert_allclose(dEta[1, 2], 3.0e-4 / RHO0 * dt, rtol=1e-6)
    assert float(np.abs(dEta).sum()) == pytest.approx(dEta[1, 2], rel=1e-12)


def test_loader_depoorter_layout(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    from legoesm.ocean.forcing.isf_spe import load_isf_spe_forcing

    ny, nx = 10, 16
    lat1 = np.linspace(-78.0, -40.0, ny)
    lon1 = np.linspace(-180.0 + 11.25, 180.0 - 11.25, nx)
    lat2d, lon2d = np.meshgrid(lat1, lon1, indexing="ij")
    rng = np.random.default_rng(3)
    fwf = np.zeros((12, ny, nx))
    fwf[:, :3, :] = rng.uniform(1e-5, 1e-3, (12, 3, nx))   # shelf band
    zmin = np.where(fwf > 0, rng.uniform(-50.0, 200.0, fwf.shape), 0.0)
    zmax = zmin + rng.uniform(100.0, 500.0, fwf.shape)
    p = tmp_path / "isf.nc"
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        for n, a in (("nav_lon", lon2d), ("nav_lat", lat2d)):
            v = ds.createVariable(n, "f8", ("y", "x")); v[:] = a
        for n, a in (("sornfisf", fwf), ("sodepmin_isf", zmin),
                     ("sodepmax_isf", zmax)):
            v = ds.createVariable(n, "f8", ("time", "y", "x")); v[:] = a
    # passthrough
    f = load_isf_spe_forcing(str(p), lat2d, lon2d)
    np.testing.assert_allclose(f.fwf, fwf)
    assert np.all(f.zmin >= 0.0)
    assert np.all(f.zmax > f.zmin)
    # regrid to a different grid: monthly totals preserved
    lat_t = np.linspace(-80.0, -35.0, 14)
    lon_t = np.linspace(-180.0 + 9.0, 180.0 - 9.0, 20)
    g = load_isf_spe_forcing(str(p), lat_t, lon_t)
    lat2d_t, _ = np.meshgrid(lat_t, lon_t, indexing="ij")
    for m in [0, 6, 11]:
        src = float((fwf[m] * np.cos(np.deg2rad(lat2d))).sum())
        tgt = float((g.fwf[m] * np.cos(np.deg2rad(lat2d_t))).sum())
        np.testing.assert_allclose(tgt, src, rtol=1e-10)


def test_loader_paired_cells_mpas(tmp_path):
    """paired_cells=True: 1-D lat/lon are (nCells,) centres, not axes.

    Guards the MPAS wiring: no outer-product meshgrid, and the melt-total
    renorm uses true target areas against an estimated source area (both
    m^2) so the absolute total is preserved, not a cos-lat proxy.
    """
    netCDF4 = pytest.importorskip("netCDF4")
    from legoesm import constants
    from legoesm.ocean.forcing.isf_spe import load_isf_spe_forcing

    ny, nx = 10, 16
    lat1 = np.linspace(-78.0, -40.0, ny)
    lon1 = np.linspace(-180.0 + 11.25, 180.0 - 11.25, nx)
    lat2d, lon2d = np.meshgrid(lat1, lon1, indexing="ij")
    rng = np.random.default_rng(7)
    fwf = np.zeros((12, ny, nx))
    fwf[:, :3, :] = rng.uniform(1e-5, 1e-3, (12, 3, nx))
    zmin = np.where(fwf > 0, rng.uniform(-50.0, 200.0, fwf.shape), 0.0)
    zmax = zmin + rng.uniform(100.0, 500.0, fwf.shape)
    p = tmp_path / "isf.nc"
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        for n, a in (("nav_lon", lon2d), ("nav_lat", lat2d)):
            v = ds.createVariable(n, "f8", ("y", "x")); v[:] = a
        for n, a in (("sornfisf", fwf), ("sodepmin_isf", zmin),
                     ("sodepmax_isf", zmax)):
            v = ds.createVariable(n, "f8", ("time", "y", "x")); v[:] = a

    n_cells = 500
    cell_lat = rng.uniform(-80.0, -30.0, n_cells)
    cell_lon = rng.uniform(-180.0, 180.0, n_cells)
    cell_area = np.full(n_cells, 1.2e10)  # quasi-uniform Voronoi cells [m^2]

    g = load_isf_spe_forcing(str(p), cell_lat, cell_lon,
                             paired_cells=True, target_area=cell_area)
    # paired: (12, nCells), never the (12, nCells, nCells) meshgrid cross
    assert g.fwf.shape == (12, n_cells)
    assert g.zmin.shape == (12, n_cells)
    # melt-total preserved in m^2 units: sum(out*A_tgt) == sum(src*A_src_est)
    from legoesm.ocean.forcing.curvilinear_regrid import (
        estimate_curvilinear_cell_area,
    )
    src_area = estimate_curvilinear_cell_area(lat2d, lon2d)
    for m in [0, 6, 11]:
        src = float((fwf[m] * src_area).sum())
        tgt = float((g.fwf[m] * cell_area).sum())
        np.testing.assert_allclose(tgt, src, rtol=1e-10)
    # no melt smeared equatorward of the source shelf band (+2 deg)
    far = cell_lat > (lat1[2] + 2.0)
    assert np.all(g.fwf[:, far] == 0.0)
