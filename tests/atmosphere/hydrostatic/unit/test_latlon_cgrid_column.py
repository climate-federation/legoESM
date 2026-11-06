"""The lat-lon C-grid dycore as a column model (``dycore.column_lane`` on
``latlon_cgrid``): the shared column loop drives the native C-grid state
through :class:`LatLonCGridColumnModel`."""
import numpy as np
import pytest

import jax.numpy as jnp

from legoesm.atmosphere.dynamics.gcm.latlon_cgrid_column import (
    LatLonCGridColumnModel, LatLonColumnState, build_latlon_column_mesh)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel,
    hydrostatic_to_cgrid)
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticTendencies

RES, NLEV, DT = 12, 8, 600.0


@pytest.fixture(scope="module")
def setup():
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_latlon)
    from legoesm.grids.factory import create_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    g = create_grid("latlon", RES)
    s = create_sigma_coordinate(NLEV)
    hs = held_suarez_init_latlon(g, s, T_init=280.0)
    nat = hydrostatic_to_cgrid(hs, g)
    nat = nat._replace(tracers={"q_v": 5e-3 * jnp.ones_like(nat.T)})
    return g, s, nat


def _dyn(g, s):
    cfg = CGridLatLonPrimitiveEquationConfig(use_polar_filter=True,
                                             energy_fixer=True)
    return CGridLatLonPrimitiveEquationModel(g, s, cfg, dt=DT)


def _tend(T, p_s, *, du=1e-4, dT=1e-4, precip=None, lat=None, lon=None):
    z, z2 = jnp.zeros_like(T), jnp.zeros_like(p_s)
    if lat is not None:
        # cell-dependent, so a mis-ordered column<->grid map is visible
        z = z + 1e-4 * (jnp.sin(lat) * jnp.cos(lon))[..., None]
    f = lambda a, n: Field(data=a, name=n)  # noqa: E731
    return HydrostaticTendencies(
        du_dt=f(z + du, "du"), dT_dt=f(z + dT, "dT"), dp_s_dt=f(z2, "dps"),
        dphis_dt=f(z2, "dphis"), dv_dt=f(z, "dv"),
        precip=None if precip is None else f(z2 + precip, "pr"))


def test_mesh_is_the_row_major_cells_of_the_grid(setup):
    g, _, _ = setup
    m = build_latlon_column_mesh(g)
    assert m.nCells == g.n_lat * g.n_lon
    np.testing.assert_array_equal(np.asarray(m.latCell),
                                  np.asarray(g.lat2d).reshape(-1))
    np.testing.assert_array_equal(np.asarray(m.lonCell),
                                  np.asarray(g.lon2d).reshape(-1))
    np.testing.assert_array_equal(np.asarray(m.areaCell),
                                  np.asarray(g.area).reshape(-1))
    lat, lon = np.asarray(m.latCell), np.asarray(m.lonCell)
    clat, clon = np.asarray(m.cornerLat), np.asarray(m.cornerLon)
    assert clat.shape == (m.nCells, 4)
    assert (clat.min(axis=1) < lat).all() and (clat.max(axis=1) > lat).all()
    assert (clon.min(axis=1) < lon).all() and (clon.max(axis=1) > lon).all()


def test_dynamics_only_step_is_the_native_step_bitwise(setup):
    g, s, nat = setup
    ref = _dyn(g, s).step(nat, DT)
    col = LatLonCGridColumnModel(_dyn(g, s))
    out = col.step(col.from_bundle(nat), DT)
    assert isinstance(out, LatLonColumnState)
    for k in ("u", "v", "T", "p_s"):
        np.testing.assert_array_equal(np.asarray(getattr(out.native, k)),
                                      np.asarray(getattr(ref, k)), err_msg=k)
    np.testing.assert_array_equal(np.asarray(out.native.tracers["q_v"]),
                                  np.asarray(ref.tracers["q_v"]))


def test_column_physics_is_the_native_operator_split_bitwise(setup):
    """Column physics through the bridge == the same tendencies given to the
    native step on the grid (same increment applier, polar filter, rails),
    and the column-shaped surface diagnostics are stashed."""
    g, s, nat = setup

    def col_phys(cols, mesh, sigma, phys_state=None, forcing=None):
        assert cols.T.data.shape == (mesh.nCells, NLEV)
        return _tend(cols.T.data, cols.p_s.data, precip=1.0,
                     lat=mesh.latCell, lon=mesh.lonCell)

    def grid_phys(hs, grid, sigma):
        # the mesh holds the grid's coordinates in fp64
        return _tend(hs.T.data, hs.p_s.data,
                     lat=jnp.asarray(grid.lat2d, jnp.float64),
                     lon=jnp.asarray(grid.lon2d, jnp.float64))

    ref = _dyn(g, s).step(nat, DT, physics_fn=grid_phys)
    col = LatLonCGridColumnModel(_dyn(g, s))
    out = col.step(col.from_bundle(nat), DT, physics_fn=col_phys)
    for k in ("u", "v", "T", "p_s"):
        np.testing.assert_array_equal(np.asarray(getattr(out.native, k)),
                                      np.asarray(getattr(ref, k)), err_msg=k)
    # the physics moved the state (the comparison has power)
    noph = _dyn(g, s).step(nat, DT)
    assert np.abs(np.asarray(ref.T) - np.asarray(noph.T)).max() > 0.05
    assert np.abs(np.asarray(ref.u) - np.asarray(noph.u)).max() > 0.05
    assert col._sfc_diag is not None
    assert col._sfc_diag[2].data.shape == (g.n_lat * g.n_lon,)
    assert float(jnp.max(jnp.abs(col._sfc_diag[2].data - 1.0))) == 0.0


def test_refuses_physics_without_the_meridional_tendency(setup):
    g, s, nat = setup

    def no_v(cols, mesh, sigma, phys_state=None, forcing=None):
        return _tend(cols.T.data, cols.p_s.data)._replace(dv_dt=None)

    col = LatLonCGridColumnModel(_dyn(g, s))
    with pytest.raises(ValueError, match="dv_dt=None"):
        col.step(col.from_bundle(nat), DT, physics_fn=no_v)


def test_writes_back_T_and_tracers_and_refuses_wind_edits(setup):
    g, s, nat = setup
    col = LatLonCGridColumnModel(_dyn(g, s))
    st = col.from_bundle(nat)
    edited = st._replace(T=st.T.replace(data=st.T.data + 1.0), tracers={
        "q_v": st.tracers["q_v"].replace(data=st.tracers["q_v"].data * 2.0)})
    back = col.to_bundle(edited)
    np.testing.assert_array_equal(np.asarray(back.T), np.asarray(nat.T) + 1.0)
    np.testing.assert_array_equal(np.asarray(back.tracers["q_v"]),
                                  2.0 * np.asarray(nat.tracers["q_v"]))
    np.testing.assert_array_equal(np.asarray(back.u), np.asarray(nat.u))
    for nm in ("u", "v", "p_s"):
        bad = st._replace(**{nm: getattr(st, nm).replace(
            data=getattr(st, nm).data + 1.0)})
        with pytest.raises(ValueError, match=f"state.{nm} was edited"):
            col.to_bundle(bad)
    with pytest.raises(ValueError, match="not the one this model returned"):
        col.to_bundle(st._replace(native=nat._replace(T=nat.T + 0.0)))


# ---------------------------------------------------------------------
# the driver: _run_column with the lat-lon column model
# ---------------------------------------------------------------------

def _driver_cfg(tmp_path, **over):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
    dycore = DycoreConfig(
        model_type="hydrostatic", discretization="latlon_cgrid", dt=DT,
        column_lane=over.pop("column", True), use_polar_filter=True)
    base = dict(
        grid=GridConfig(grid_type="latlon", resolution=RES, nlev=NLEV),
        dycore=dycore, days=over.pop("days", 0.25),
        radiation="none", convection="none", microphysics="none",
        turbulence="none", gravity_wave_drag="none", precision="fp64",
        output=OutputConfig(diag_days=0, checkpoint_days=0,
                            output_dir=str(tmp_path)))
    base.update(over)
    return ExperimentConfig(**base)


def _driver(tmp_path, **over):
    from legoesm.driver.model_driver import ModelDriver
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = _driver_cfg(tmp_path, **over)
    drv = ModelDriver(cfg, output_dir=tmp_path)
    drv.setup()
    return drv


def test_driver_column_lane_steps_the_native_dycore(tmp_path):
    """``run()`` goes through the column loop with the lat-lon column model;
    its native state equals the bare dycore stepped from the same IC.  The
    loop calls its (all-"none") physics package every step, so the two
    programs differ by zero increments through the polar filter and rails:
    equal to 1e-10 absolute (measured 1.2e-12 m/s, 7e-13 K after 36 steps)."""
    drv = _driver(tmp_path)
    assert isinstance(drv.model, LatLonCGridColumnModel)
    assert isinstance(drv.state, LatLonColumnState)
    assert drv.grid is drv.model.mesh and drv.sigma is drv.model.sigma_coord
    x = drv.state.native
    assert drv.run() == "COMPLETED"
    dyn = CGridLatLonPrimitiveEquationModel(
        drv.model.grid, drv.model.sigma_coord, drv.model.config, dt=DT)
    for _ in range(int(round(0.25 * 86400.0 / DT))):
        x = dyn.step(x, DT)
    for k in ("u", "v", "T", "p_s"):
        a, b = np.asarray(getattr(drv.state.native, k)), np.asarray(getattr(x, k))
        assert np.abs(a - b).max() <= 1e-10 * max(1.0, np.abs(b).max()), k
    assert np.abs(np.asarray(x.T) - np.asarray(drv._state_template.native.T)).max() > 0


def test_driver_refusals(tmp_path):
    from legoesm.driver.config import OutputConfig
    from legoesm.driver.model_driver import ModelDriver
    ledger = OutputConfig(diag_days=0, checkpoint_days=0,
                          output_dir=str(tmp_path), budget_ledger=True)
    for over, frag in ((dict(sponge_enabled=True), "sponge_enabled"),
                       (dict(mpas_qv_smooth_del4_m4s=1e14), "Voronoi mesh"),
                       (dict(ensemble_size=2), "single device"),
                       (dict(output=ledger), "budget_ledger"),
                       (dict(convection="kuo"), "grid operator")):
        drv = ModelDriver(_driver_cfg(tmp_path, **over), output_dir=tmp_path)
        with pytest.raises(ValueError, match=frag):
            drv.setup()


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("phys", [dict(), dict(microphysics="kessler")])
def test_driver_restart_is_bitwise(tmp_path, phys):
    """Run A: 2 days with a checkpoint each day; run B: a fresh driver from
    A's day-1 checkpoint for the remaining day.  The native C-grid states
    (face winds included) must be BITWISE equal."""
    from legoesm.driver.model_driver import ModelDriver
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir(), dir_b.mkdir()
    cfg_a = _driver_cfg(dir_a, days=2, **phys)
    cfg_a = cfg_a._replace(output=cfg_a.output._replace(checkpoint_days=1))
    drv_a = ModelDriver(cfg_a, output_dir=dir_a)
    drv_a.setup()
    assert drv_a.run() == "COMPLETED"
    mid = dir_a / "checkpoint_day_0001.npz"
    with np.load(mid) as d:
        assert str(d["latloncol_schema"]) == "latloncol_ckpt_v1"
        u_mid = np.asarray(d["latloncol_u"])
    drv_b = ModelDriver(_driver_cfg(dir_b, days=1, **phys), output_dir=dir_b)
    drv_b.setup()
    step, day = drv_b.load_checkpoint(mid)
    assert day == pytest.approx(1.0)
    assert drv_b.run(start_step=step, start_day=day) == "COMPLETED"
    a, b = drv_a.state.native, drv_b.state.native
    for k in ("u", "v", "T", "p_s", "phis"):
        assert _same(getattr(a, k), getattr(b, k)), k
    assert sorted(a.tracers) == sorted(b.tracers)
    if phys:
        # the moist run carries its water on the dycore state
        assert "q_v" in a.tracers and float(np.max(a.tracers["q_v"])) > 0.0
    for nm in a.tracers:
        assert _same(a.tracers[nm], b.tracers[nm]), nm
    assert not _same(u_mid, a.u)        # the second day moved the state
    # the driver's lat-lon dycore fixes dry mass step to step (no anchor:
    # the factory never sets anchor_mass_to_initial), so nothing to restore
    assert drv_a.model._target_mass is None and drv_b.model._target_mass is None


def test_driver_refuses_a_checkpoint_without_the_cgrid_state(tmp_path):
    drv = _driver(tmp_path)
    bad = tmp_path / "checkpoint_day_0000.npz"
    np.savez(bad, u=np.zeros(1), T=np.zeros(1), p_s=np.zeros(1),
             phis=np.zeros(1), step=np.asarray(3), day=np.asarray(0.1))
    with pytest.raises(ValueError, match="carries no lat-lon C-grid state"):
        drv.load_checkpoint(bad)
    np.savez(bad, u=np.zeros(1), T=np.zeros(1), p_s=np.zeros(1),
             phis=np.zeros(1), step=np.asarray(3), day=np.asarray(0.1),
             latloncol_schema=np.asarray("latloncol_ckpt_v0"))
    with pytest.raises(ValueError, match="unknown lat-lon column schema"):
        drv.load_checkpoint(bad)
    np.savez(bad, u=np.zeros(1), T=np.zeros(1), p_s=np.zeros(1),
             phis=np.zeros(1), step=np.asarray(3), day=np.asarray(0.1),
             latloncol_schema=np.asarray("latloncol_ckpt_v1"))
    with pytest.raises(ValueError, match="state incomplete"):
        drv.load_checkpoint(bad)


def test_driver_calibrated_land_and_prognostic_clubb_run(tmp_path, monkeypatch):
    """The two configurations the PhysicsPipeline lat-lon lane refuses
    validate and run on the lat-lon column lane: the multilayer land's
    calibrated physics with its solved-flux handoff, and CLUBB carrying its
    moments in the physics state."""
    from legoesm.land.config import biophysics_lmip_two_leaf_setup
    from tests._land_surfdata import require_surfdata
    from tests.unit.test_mpas_multilayer_land_port import _patch_land_loaders
    _patch_land_loaders(monkeypatch)
    cal = biophysics_lmip_two_leaf_setup()["soil_grid"]
    over = dict(
        days=1200.0 / 86400.0, dataset="analytical", radiation="gray",
        turbulence="clubb", clubb_prognostic=True,
        land_mask_path="synthetic.nc", use_multilayer_land=True,
        multilayer_n_layers=cal.n_layers,
        multilayer_soil_depth=cal.total_depth, land_beta_soil=True,
        land_surface_scheme="two_leaf", land_stomatal_beta=False,
        land_calibrated_physics=True, snow_albedo_feedback=True,
        surfdata_path=require_surfdata())
    _driver_cfg(tmp_path, **dict(over)).validate_strict()
    with pytest.raises(ValueError, match="land_calibrated_physics=True is "
                                         "supported only on the MPAS"):
        _driver_cfg(tmp_path, column=False, **dict(over)).validate_strict()
    with pytest.raises(ValueError, match="clubb_prognostic=True is not wired"):
        _driver_cfg(tmp_path, column=False,
                    **{**over, "land_calibrated_physics": False}
                    ).validate_strict()
    drv = _driver(tmp_path, **over)
    assert drv.run() == "COMPLETED"
    ps = drv._mpas_phys_state
    assert ps is not None and ps.clubb_moments is not None
    mom = np.asarray(ps.clubb_moments)
    assert mom.shape[0] == RES * 2 * RES and np.isfinite(mom).all()
    assert np.abs(mom).max() > 0.0


def test_driver_terrain_is_the_lat_lon_lanes_product(tmp_path):
    """The column lane builds the terrain product (binning + the lat-lon
    neighbour filter) on the NATIVE grid: phis and the land fraction equal
    the PhysicsPipeline lat-lon lane's, flattened row-major, and the IC's
    surface geopotential is that terrain."""
    from tests.atmosphere.hydrostatic.unit.test_fv3_duo_column import (
        _write_elevation_file)
    elev = tmp_path / "elev.nc"
    _write_elevation_file(elev)
    over = dict(topography=str(elev), topo_smoothing=2)
    col = _driver(tmp_path / "col", **over)
    pipe = _driver(tmp_path / "pipe", column=False, **over)
    p_pipe = np.asarray(pipe._phis_data).reshape(-1)
    assert np.abs(p_pipe).max() > 1000.0
    np.testing.assert_array_equal(np.asarray(col._phis_data), p_pipe)
    np.testing.assert_array_equal(np.asarray(col._f_land),
                                  np.asarray(pipe._f_land).reshape(-1))
    np.testing.assert_array_equal(np.asarray(col.state.native.phis),
                                  np.asarray(pipe._phis_data))


def test_ice_curve_drain_is_admitted_on_the_column_lane(tmp_path):
    """The ice-curve drain lives in the column loop's post-step hook, so the
    lat-lon column lane accepts it; the PhysicsPipeline lat-lon lane, where
    it would be inert, still refuses it."""
    over = dict(microphysics="morrison", hard_saturation_adjustment=True,
                hard_sat_ice_curve=True)
    _driver_cfg(tmp_path, **over).validate_strict()
    with pytest.raises(ValueError, match="requires the column loop"):
        _driver_cfg(tmp_path, column=False, **over).validate_strict()


def test_driver_restart_carries_clubb_moments_and_land(tmp_path, monkeypatch):
    """Restart with the two carries this lane exists for: CLUBB's prognostic
    moments (PhysicsState) and the multilayer land state.  Run A: 2 days,
    checkpoint each day; run B from A's day-1 checkpoint.  Native state and
    CLUBB moments must be bitwise equal, and the moments must have evolved
    after the checkpoint (else the comparison has no power)."""
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.land.config import biophysics_lmip_two_leaf_setup
    from tests._land_surfdata import require_surfdata
    from tests.unit.test_mpas_multilayer_land_port import _patch_land_loaders
    _patch_land_loaders(monkeypatch)
    cal = biophysics_lmip_two_leaf_setup()["soil_grid"]
    over = dict(
        dataset="analytical", radiation="gray", turbulence="clubb",
        clubb_prognostic=True, land_mask_path="synthetic.nc",
        use_multilayer_land=True, multilayer_n_layers=cal.n_layers,
        multilayer_soil_depth=cal.total_depth, land_beta_soil=True,
        land_surface_scheme="two_leaf", land_stomatal_beta=False,
        land_calibrated_physics=True, snow_albedo_feedback=True,
        surfdata_path=require_surfdata())
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir(), dir_b.mkdir()
    cfg_a = _driver_cfg(dir_a, days=2, **over)
    cfg_a = cfg_a._replace(output=cfg_a.output._replace(checkpoint_days=1))
    drv_a = ModelDriver(cfg_a, output_dir=dir_a)
    drv_a.setup()
    assert drv_a.run() == "COMPLETED"
    mid = dir_a / "checkpoint_day_0001.npz"
    with np.load(mid) as d:
        mom_mid = np.asarray(d["physstate_clubb_moments"])
    drv_b = ModelDriver(_driver_cfg(dir_b, days=1, **over), output_dir=dir_b)
    drv_b.setup()
    step, day = drv_b.load_checkpoint(mid)
    assert drv_b.run(start_step=step, start_day=day) == "COMPLETED"
    a, b = drv_a.state.native, drv_b.state.native
    for k in ("u", "v", "T", "p_s"):
        assert _same(getattr(a, k), getattr(b, k)), k
    for nm in a.tracers:
        assert _same(a.tracers[nm], b.tracers[nm]), nm
    mom_a = np.asarray(drv_a._mpas_phys_state.clubb_moments)
    assert _same(mom_a, drv_b._mpas_phys_state.clubb_moments)
    assert mom_mid.tobytes() != mom_a.tobytes()


def test_driver_surface_data_and_cmip_output_are_the_lat_lon_lanes(tmp_path):
    """The column lane regrids the land surface data on its native grid (the
    lat-lon lane's conservative regrid) and writes CMIP fields through the
    lat-lon lane's structured weights: both equal the PhysicsPipeline lane's
    for the same terrain."""
    from legoesm.driver.config import OutputConfig
    from tests._land_surfdata import require_surfdata
    from tests.atmosphere.hydrostatic.unit.test_fv3_duo_column import (
        _write_elevation_file)
    elev = tmp_path / "elev.nc"
    _write_elevation_file(elev)
    drv = {}
    for name, col in (("col", True), ("pipe", False)):
        d = tmp_path / name
        out = OutputConfig(diag_days=0, checkpoint_days=0,
                           output_dir=str(d), cmip_output=True)
        drv[name] = _driver(d, column=col, topography=str(elev),
                            topo_smoothing=2, output=out)
    c, p = drv["col"], drv["pipe"]
    sd = require_surfdata()
    a_col = np.asarray(c._surfdata_land_albedo(sd, jnp.zeros(RES * 2 * RES)))
    a_pipe = np.asarray(p._surfdata_land_albedo(sd, jnp.zeros((RES, 2 * RES))))
    assert np.count_nonzero(a_pipe) > 10
    np.testing.assert_array_equal(a_col, a_pipe.reshape(-1))
    o_col = c.diagnostics._regrid_to_latlon_2d(np.asarray(c._phis_data))
    o_pipe = p.diagnostics._regrid_to_latlon_2d(np.asarray(p._phis_data))
    assert o_col is not None and o_col.shape == o_pipe.shape
    np.testing.assert_array_equal(o_col, o_pipe)
    t3 = np.asarray(c.state.T.data)
    o3_col = c.diagnostics._regrid_to_latlon_3d(t3)
    o3_pipe = p.diagnostics._regrid_to_latlon_3d(t3.reshape(RES, 2 * RES, -1))
    np.testing.assert_array_equal(o3_col, o3_pipe)


def test_driver_subgrid_orography_uses_the_native_grid_spacing(tmp_path,
                                                               monkeypatch):
    """The SSO file's scale check sees the native lat-lon spacing (180/n_lat),
    not the cell-count estimate of an unstructured mesh.  (The PhysicsPipeline
    lat-lon lane has no per-column SSO field and refuses the file.)"""
    import xarray as xr
    import legoesm.grids.topography as topo
    from legoesm.driver.model_driver import ModelDriver
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(0.0, 360.0, 2.0)
    sso = 100.0 + 50.0 * np.cos(np.deg2rad(lat))[:, None] * np.ones(lon.size)
    path = tmp_path / "sso.nc"
    xr.Dataset({"SSO_STDH": (("lat", "lon"), sso)},
               coords={"lat": lat, "lon": lon}).to_netcdf(path)
    seen = []
    real = topo._check_sso_scale_decomposition
    monkeypatch.setattr(topo, "_check_sso_scale_decomposition",
                        lambda built, dx, *a, **k: (seen.append(dx),
                                                    real(built, dx, *a, **k))[1])
    over = dict(subgrid_orography_path=str(path), gravity_wave_drag="mcfarlane")
    col = _driver(tmp_path / "col", **over)
    assert seen == [pytest.approx(180.0 / RES)]
    got = np.asarray(col.grid.subgrid_topo_stddev)
    assert got.shape == (RES * 2 * RES,) and got.min() > 100.0
    (tmp_path / "pipe").mkdir()
    pipe = ModelDriver(_driver_cfg(tmp_path / "pipe", column=False, **over),
                       output_dir=tmp_path / "pipe")
    with pytest.raises(ValueError, match="no subgrid_topo_stddev field"):
        pipe.setup()


def test_ice_curve_drain_acts_on_the_column_lane(tmp_path):
    """The mixed-phase drain runs through the column view: with it on, the
    post-step T / water edits reach the native state (the run differs from
    the same run with the drain off)."""
    base = dict(days=3 * DT / 86400.0, microphysics="morrison",
                morrison_warm_rain_scheme="kk2000",
                hard_saturation_adjustment=True, ic="standard", rh_init=1.0)
    on = _driver(tmp_path / "on", hard_sat_ice_curve=True, **base)
    off = _driver(tmp_path / "off", hard_sat_ice_curve=False, **base)
    for d in (on, off):
        # supersaturate (a driver-side column edit, written back by the
        # model) so the drain has something to remove
        st = d.state
        tr = dict(st.tracers)
        tr["q_v"] = tr["q_v"].replace(data=3.0 * tr["q_v"].data)
        d.state = st._replace(tracers=tr)
    assert on.run() == "COMPLETED" and off.run() == "COMPLETED"
    d_T = np.abs(np.asarray(on.state.native.T) - np.asarray(off.state.native.T)).max()
    d_qi = np.abs(np.asarray(on.state.native.tracers["q_i"])
                  - np.asarray(off.state.native.tracers["q_i"])).max()
    assert d_T > 0.0 and d_qi > 0.0


# ---------------------------------------------------------------------
# the lane's q_v del4 smoother (latlon_qv_smooth_del4_m4s)
# ---------------------------------------------------------------------

def test_latlon_qv_smoother_damps_noise_and_keeps_level_water(setup):
    from legoesm.driver.model_driver import make_latlon_qv_smooth_fn
    g, s, nat = setup
    col = LatLonCGridColumnModel(_dyn(g, s))
    n = g.n_lat * g.n_lon
    i, j = np.meshgrid(np.arange(g.n_lat), np.arange(g.n_lon), indexing="ij")
    chk = ((-1.0) ** (i + j)).reshape(n, 1) * np.ones((1, NLEV))
    q = jnp.asarray(4e-3 + 1e-3 * chk)
    p_s = jnp.asarray(1e5 - 3e3 * np.cos(np.arange(n) * 0.37))
    nu4 = 0.5 * 1e14
    out = make_latlon_qv_smooth_fn(col, nu4, DT, s)(q, p_s)
    dp = s.layer_thickness_dp(p_s)
    area = np.asarray(col.mesh.areaCell)[:, None]
    m0 = np.sum(area * np.asarray(dp) * np.asarray(q), axis=0)
    m1 = np.sum(area * np.asarray(dp) * np.asarray(out), axis=0)
    np.testing.assert_allclose(m1, m0, rtol=1e-12)
    amp = lambda x: np.abs(np.sum((np.asarray(x) - 4e-3) * chk, axis=0))  # noqa: E731
    assert np.all(amp(out) < amp(q))
    flat = jnp.full_like(q, 4e-3)
    np.testing.assert_allclose(np.asarray(make_latlon_qv_smooth_fn(
        col, nu4, DT, s)(flat, p_s)), 4e-3, rtol=1e-13)


def test_driver_latlon_qv_smoother_is_wired(tmp_path):
    a = _driver(tmp_path / "off", microphysics="kessler", days=DT / 86400.0)
    b = _driver(tmp_path / "on", microphysics="kessler", days=DT / 86400.0,
                latlon_qv_smooth_del4_m4s=5e14)
    assert a.run() == "COMPLETED" and b.run() == "COMPLETED"
    qa = np.asarray(a.model.to_bundle(a.state).tracers["q_v"])
    qb = np.asarray(b.model.to_bundle(b.state).tracers["q_v"])
    assert np.abs(qa - qb).max() > 0


def test_latlon_qv_smoother_needs_the_column_lane(tmp_path):
    _driver_cfg(tmp_path, latlon_qv_smooth_del4_m4s=1e14).validate_strict()
    for v, over, frag in ((1e14, dict(column=False), "COLUMN lane"),
                          (-1.0, dict(), "finite in")):
        with pytest.raises(ValueError, match=frag):
            _driver_cfg(tmp_path, latlon_qv_smooth_del4_m4s=v,
                        **over).validate_strict()


def test_driver_latlon_qv_smoother_refuses_an_unstable_coefficient(tmp_path):
    drv = _driver(tmp_path, microphysics="kessler", latlon_qv_smooth_del4_m4s=1e18)
    with pytest.raises(ValueError, match="stability bound"):
        drv.run()


def test_no_retrace_after_warm_up_under_a_changing_forcing_and_dt_with_div4_on(setup):
    """The forcing and dt are TRACED jit arguments of the adapter's step:
    after warm-up, a new SST every step and a different dt reuse one compiled
    program, with the per-row div-4 coefficient (0.01*area^2/dt) built from
    the traced dt.  Warm-up: the fixture state is fp32 and the fp64 forcing
    promotes it over the first two steps (two traces, div-4 on or off)."""
    g, s, nat = setup
    cfg = CGridLatLonPrimitiveEquationConfig(
        use_polar_filter=True, polar_filter_cutoff_deg=35.9041, nu_div4_scale=1.0)
    col = LatLonCGridColumnModel(CGridLatLonPrimitiveEquationModel(g, s, cfg, dt=DT))

    def phys(cols, mesh, sigma, phys_state=None, forcing=None):
        return _tend(cols.T.data, cols.p_s.data, du=0.0,
                     dT=1e-6 * forcing["sst"][:, None])

    # the adapter's jitted step looks _step_cgrid_impl up once per TRACE
    traces = []
    impl = col.dyn._step_cgrid_impl

    def counting(*a, **k):
        traces.append(1)
        return impl(*a, **k)
    col.dyn._step_cgrid_impl = counting
    n = col.mesh.nCells
    st = col.from_bundle(nat)
    for k in range(3):
        st = col.step(st, DT, physics_fn=phys,
                      forcing={"sst": jnp.full((n,), 0.0)})
    warm = len(traces)
    assert warm >= 1
    for k in range(1, 5):
        st = col.step(st, DT, physics_fn=phys,
                      forcing={"sst": jnp.full((n,), float(k))})
    st = col.step(st, 0.5 * DT, physics_fn=phys,
                  forcing={"sst": jnp.full((n,), 5.0)})
    assert len(traces) == warm
    assert np.isfinite(np.asarray(st.native.T)).all()
    # the forcing WAS read: a different SST gives a different T
    nat_k = st.native
    T1 = np.asarray(col.step(col.from_bundle(nat_k), DT, physics_fn=phys,
                             forcing={"sst": jnp.full((n,), 1.0)}).native.T)
    T9 = np.asarray(col.step(col.from_bundle(nat_k), DT, physics_fn=phys,
                             forcing={"sst": jnp.full((n,), 9.0)}).native.T)
    assert np.abs(T9 - T1).max() > 0.0 and len(traces) == warm


def test_production_deck_dt_is_not_clamped_and_keeps_the_cam6_cadence():
    """The lat-lon production deck's dt survives BOTH clamps (the factory's
    equatorial CFL and the driver's generic cfl_check_and_adjust, which cut
    900 s to 600 s and would have halved every cadence; codex r1 #1), and
    dt x the cadence counts are CAM6's 1800 s / 3600 s intervals."""
    import yaml
    from legoesm import constants
    from pathlib import Path
    from legoesm.core.cfl import cfl_check_and_adjust
    from legoesm.grids.latlon import create_latlon_grid
    deck = Path(__file__).resolve().parents[4] / "config" / "amip" / "amip_production_latlon48.yaml"
    d = yaml.safe_load(deck.read_text())
    dt, n_lat = float(d["dt"]), int(d["resolution"])
    grid = create_latlon_grid(n_lat=n_lat, radius=constants.R_earth, omega=constants.Omega)
    assert d["use_polar_filter"] is True and d["column_lane"] is True
    assert cfl_check_and_adjust(dt, n_lat, model_type="primitive_eq", radius=grid.radius,
                                grid_type="latlon", use_polar_filter=True) == dt
    assert dt * d["physics_update_steps"] == 1800.0
    assert dt * d["rad_update_steps"] == 3600.0
    # the deck's own rule: 0.82 x the printed clamp, rounded down to a divisor of 1800
    clamp = cfl_check_and_adjust(1e9, n_lat, model_type="primitive_eq", radius=grid.radius,
                                 grid_type="latlon", use_polar_filter=True, verbose=False)
    assert dt <= 0.82 * 604.9 + 1.0 and 1800.0 % dt == 0 and dt > clamp * 0.82 - 150.0


def test_div4_reaches_the_adapter_step_with_the_step_dt(setup):
    """With div-4 on, the adapter's step is the native step at the SAME dt
    bitwise (the 0.01*area^2/dt coefficient sees the dynamics dt, not a
    physics interval), and differs from the div-4-off step."""
    g, s, nat = setup

    def dyn(scale):
        return CGridLatLonPrimitiveEquationModel(g, s, CGridLatLonPrimitiveEquationConfig(
            use_polar_filter=True, polar_filter_cutoff_deg=35.9041,
            nu_div4_scale=scale), dt=DT)
    ref = dyn(1.0).step(nat, DT)
    col = LatLonCGridColumnModel(dyn(1.0))
    out = col.step(col.from_bundle(nat), DT).native
    for k in ("u", "v", "T", "p_s"):
        np.testing.assert_array_equal(np.asarray(getattr(out, k)),
                                      np.asarray(getattr(ref, k)), err_msg=k)
    off = LatLonCGridColumnModel(dyn(0.0))
    out_off = off.step(off.from_bundle(nat), DT).native
    assert np.abs(np.asarray(out.u) - np.asarray(out_off.u)).max() > 0.0
    # a different dt changes the damped step (dt reaches the coefficient)
    col2 = LatLonCGridColumnModel(dyn(1.0))
    half = col2.step(col2.from_bundle(nat), 0.5 * DT).native
    np.testing.assert_array_equal(np.asarray(half.u),
                                  np.asarray(dyn(1.0).step(nat, 0.5 * DT).u))


def test_driver_checkpoint_div4_stamp_refuses_a_different_scale(tmp_path):
    """Checkpoints record the run's div-4 scale; a file without the stamp
    counts as 0; a resume whose scale differs is refused (user decision
    2026-10-26), so a pre-stamp checkpoint refuses only a damping-on run."""
    from legoesm.driver.model_driver import ModelDriver

    def cfg(d, scale, **over):
        c = _driver_cfg(d, **over)
        return c._replace(dycore=c.dycore._replace(mpas_div_damp4_scale=scale))

    def written(scale):
        d = tmp_path / f"w{scale}"
        d.mkdir()
        c = cfg(d, scale, days=1)
        c = c._replace(output=c.output._replace(checkpoint_days=1))
        drv = ModelDriver(c, output_dir=d)
        drv.setup()
        assert drv.run() == "COMPLETED"
        f = d / "checkpoint_day_0001.npz"
        with np.load(f) as z:
            assert float(z["latloncol_div4_scale"]) == scale
            unstamped = {k: z[k] for k in z.files if k != "latloncol_div4_scale"}
        old = d / "checkpoint_day_0002.npz"
        np.savez(old, **unstamped)
        return f, old

    def resume(f, scale):
        d = tmp_path / f"r{scale}_{f.parent.name}_{f.stem}"
        d.mkdir()
        drv = ModelDriver(cfg(d, scale, days=1), output_dir=d)
        drv.setup()
        return drv.load_checkpoint(f)

    off, old = written(0.0)
    on, _ = written(1.0)
    assert resume(off, 0.0)[1] == pytest.approx(1.0)
    assert resume(old, 0.0)[1] == pytest.approx(1.0)   # no stamp == 0
    assert resume(on, 1.0)[1] == pytest.approx(1.0)
    for f, scale in ((off, 1.0), (old, 1.0), (on, 0.0)):
        with pytest.raises(ValueError, match="div-4 damping scale"):
            resume(f, scale)
