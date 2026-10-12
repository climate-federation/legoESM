"""The lat-lon C-grid PE model as a column model behind the shared column
loop's step contract (composable physics P3): the G-lane ladder of the design
(fv3_duo_gaps/fu/composable_physics_design.md, row P3) at n_lat 8, nlev 5.

(a) rung 1: ``physics_fn=None`` adapter == the native dycore, bitwise, 10 steps;
(b) uniform-tendency injection: the post-step increment is the analytic
    per-lane expectation (T exact, winds through the lane's own face
    interpolator, water mass through p_s per apply_physics_water_mass);
(c) physics identity: the column view the wrapper feeds the physics gives
    tendencies bitwise equal to a direct call on the same arrays;
(d) restart parity lives in tests/unit/test_column_lane_restart_digest.py;
plus the surface-diagnostic slot merge, the no-retrace guarantee under a
changing forcing, the driver lane (setup / run / mesh identity / refusals)
and the checkpoint's face winds (planted-violation: a file without them is
refused).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.dynamics.gcm.latlon_column import (  # noqa: E402
    LATLON_CKPT_KEYS, LatLonColumnModel, LatLonColumnState,
    build_latlon_column_mesh,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (  # noqa: E402
    CGridLatLonHydrostaticState, CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel, cgrid_to_hydrostatic,
)
from legoesm.core.conservation import apply_physics_water_mass  # noqa: E402
from legoesm.core.state import HydrostaticTendencies  # noqa: E402
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.grids.operators_latlon_cgrid import (  # noqa: E402
    interp_cell_to_uface, interp_cell_to_vface_halo,
)
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402

N_LAT, NLEV, DT = 8, 5, 600.0
NAMES = ("q_v", "q_c", "q_r")


@pytest.fixture(autouse=True)
def _drop_compiled_graphs():
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def grid():
    return create_latlon_grid(n_lat=N_LAT, radius=constants.R_earth,
                              omega=constants.Omega)


@pytest.fixture(scope="module")
def sigma():
    return create_sigma_coordinate(n_levels=NLEV, dtype=jnp.float64)


def _model(grid, sigma, **cfg):
    dyn = CGridLatLonPrimitiveEquationModel(
        grid, sigma, CGridLatLonPrimitiveEquationConfig(**cfg), dt=DT)
    return dyn, LatLonColumnModel(dyn, tracer_names=NAMES)


def _native_ic(grid, seed=0):
    n_lat, n_lon = grid.n_lat, grid.n_lon
    rng = np.random.default_rng(seed)
    lat = np.asarray(grid.lat)
    T = (280.0 - 30.0 * np.sin(lat)[:, None, None] ** 2
         + np.linspace(20.0, -20.0, NLEV)[None, None, :])
    T = jnp.asarray(np.broadcast_to(T, (n_lat, n_lon, NLEV))
                    + 0.1 * rng.standard_normal((n_lat, n_lon, NLEV)))
    q_v = jnp.asarray(0.01 * rng.uniform(0.3, 1.0, (n_lat, n_lon, NLEV)))
    return CGridLatLonHydrostaticState(
        u=jnp.asarray(5.0 * rng.standard_normal((n_lat, n_lon + 1, NLEV))),
        v=jnp.zeros((n_lat + 1, n_lon, NLEV)),
        T=T, p_s=jnp.asarray(1.0e5 + 500.0 * rng.standard_normal((n_lat, n_lon))),
        phis=jnp.zeros((n_lat, n_lon)),
        tracers={"q_v": q_v, "q_c": jnp.zeros_like(q_v),
                 "q_r": jnp.asarray(5e-3 * rng.uniform(0.6, 1.0, (n_lat, n_lon, NLEV)))})


def _tend(state, du, dv, dT, tracers=None, **extra):
    z = jnp.zeros_like(state.p_s.data)
    f = lambda like, d, nm: like.replace(data=d, name=nm)  # noqa: E731
    return HydrostaticTendencies(
        du_dt=f(state.u, du, "du_dt"), dv_dt=f(state.v, dv, "dv_dt"),
        dT_dt=f(state.T, dT, "dT_dt"), dp_s_dt=f(state.p_s, z, "dp_s"),
        dphis_dt=f(state.phis, z, "dphis"), tracer_tendencies=tracers, **extra)


def _zero_physics(state, mesh, coord, phys_state=None, forcing=None):
    return _tend(state, jnp.zeros_like(state.u.data),
                 jnp.zeros_like(state.v.data), jnp.zeros_like(state.T.data))


def _assert_native_equal(a, b, *, rel=0.0):
    for k in ("u", "v", "T", "p_s", "phis"):
        x, y = np.asarray(getattr(a, k)), np.asarray(getattr(b, k))
        np.testing.assert_allclose(x, y, rtol=0, atol=rel * np.abs(y).max(), err_msg=k)
    assert sorted(a.tracers) == sorted(b.tracers)
    for k in a.tracers:
        np.testing.assert_allclose(np.asarray(a.tracers[k]), np.asarray(b.tracers[k]),
                                   rtol=0, atol=rel * np.abs(np.asarray(b.tracers[k])).max(),
                                   err_msg=k)


# ---------------------------------------------------------------------
# the mesh and the view
# ---------------------------------------------------------------------

def test_mesh_is_the_grid_row_major_with_ring_corners(grid, sigma):
    _, col = _model(grid, sigma)
    m = col.mesh
    assert m.nCells == grid.n_lat * grid.n_lon and m.grid_shape_2d == (m.nCells,)
    np.testing.assert_array_equal(np.asarray(m.latCell), np.asarray(grid.lat2d).reshape(-1))
    np.testing.assert_array_equal(np.asarray(m.lonCell), np.asarray(grid.lon2d).reshape(-1))
    np.testing.assert_array_equal(np.asarray(m.areaCell), np.asarray(grid.area).reshape(-1))
    assert np.isclose(float(m.grid_total_area), float(grid.total_area))
    # corners: the v-face latitudes south/north, lon interfaces west/east,
    # ring order sw, se, ne, nw; every cell's centre inside its quad
    c_lat, c_lon = np.asarray(m.cornerLat), np.asarray(m.cornerLon)
    assert c_lat.shape == (m.nCells, 4)
    lat_v = np.asarray(grid.lat_v)
    j = np.repeat(np.arange(grid.n_lat), grid.n_lon)
    np.testing.assert_array_equal(c_lat[:, 0], lat_v[j])
    np.testing.assert_array_equal(c_lat[:, 2], lat_v[j + 1])
    assert np.all(c_lat[:, 0] < np.asarray(m.latCell)) and np.all(np.asarray(m.latCell) < c_lat[:, 2])
    dlon = float(grid.dlon)
    width = (c_lon[:, 1] - c_lon[:, 0]) % (2 * np.pi)
    np.testing.assert_allclose(width, dlon, rtol=1e-12)
    assert m.radius == grid.radius and m.omega == grid.omega
    # radians, east-positive [0, 2pi), areas in m^2 summing to the sphere
    assert np.abs(np.asarray(m.latCell)).max() < np.pi / 2
    assert np.asarray(m.lonCell).min() >= 0.0 and np.asarray(m.lonCell).max() < 2 * np.pi
    np.testing.assert_allclose(float(np.asarray(m.areaCell).sum()),
                               4 * np.pi * grid.radius ** 2, rtol=1e-6)
    assert col.sigma_coord is sigma and col.native_grid is grid
    m2 = build_latlon_column_mesh(grid)
    assert (m2.nCells, m2.radius, m2.omega) == (m.nCells, m.radius, m.omega)
    for k in ("latCell", "lonCell", "areaCell", "cornerLat", "cornerLon"):
        np.testing.assert_array_equal(np.asarray(getattr(m2, k)), np.asarray(getattr(m, k)))


def test_column_view_is_the_dycores_own_cell_winds(grid, sigma):
    _, col = _model(grid, sigma)
    nat = _native_ic(grid)
    st = col.from_bundle(nat)
    assert isinstance(st, LatLonColumnState) and st.native is nat
    hs = cgrid_to_hydrostatic(nat, grid)
    ncol = col.mesh.nCells
    for k in ("u", "v", "T", "p_s", "phis"):
        got = np.asarray(getattr(st, k).data)
        exp = np.asarray(getattr(hs, k).data).reshape((ncol,) + np.asarray(getattr(hs, k).data).shape[2:])
        np.testing.assert_array_equal(got, exp, err_msg=k)
    assert tuple(st.tracers) == NAMES
    np.testing.assert_array_equal(np.asarray(st.tracers["q_v"].data),
                                  np.asarray(nat.tracers["q_v"]).reshape(ncol, NLEV))
    # the driver edits T / tracers on the columns: to_bundle writes them
    # back exactly, and never touches the face winds
    st2 = st._replace(T=st.T.replace(data=st.T.data + 1.0),
                      tracers={**st.tracers, "q_v": st.tracers["q_v"].replace(
                          data=st.tracers["q_v"].data * 2.0)})
    nat2 = col.to_bundle(st2)
    np.testing.assert_array_equal(np.asarray(nat2.T), np.asarray(nat.T) + 1.0)
    np.testing.assert_array_equal(np.asarray(nat2.tracers["q_v"]), np.asarray(nat.tracers["q_v"]) * 2.0)
    assert nat2.u is nat.u and nat2.v is nat.v


# ---------------------------------------------------------------------
# G-lane (a): rung 1, physics_fn=None == the native dycore bitwise
# ---------------------------------------------------------------------

def test_rung1_no_physics_is_the_native_dycore_bitwise(grid, sigma):
    dyn, col = _model(grid, sigma, fix_mass=True, anchor_mass_to_initial=False)
    ref_dyn = CGridLatLonPrimitiveEquationModel(
        grid, sigma, CGridLatLonPrimitiveEquationConfig(
            fix_mass=True, anchor_mass_to_initial=False), dt=DT)
    nat = _native_ic(grid)
    ref = nat
    st = col.from_bundle(nat)
    for _ in range(10):
        ref = ref_dyn.step(ref, DT)
        st = col.step(st, DT)
    _assert_native_equal(st.native, ref)
    assert np.abs(np.asarray(st.native.T) - np.asarray(nat.T)).max() > 0.0


def test_zero_tendency_physics_is_the_no_physics_step_bitwise(grid, sigma):
    """Exact-zero increments are exact identities on every lane (the
    operator-split test's own invariant), through the wrapper too."""
    _, col = _model(grid, sigma, fix_mass=False)
    nat = _native_ic(grid)
    a = col.step(col.from_bundle(nat), DT)
    b = col.step(col.from_bundle(nat), DT, physics_fn=_zero_physics)
    _assert_native_equal(a.native, b.native)


# ---------------------------------------------------------------------
# G-lane (b): uniform-tendency injection, the analytic per-lane increment
# ---------------------------------------------------------------------

def test_uniform_tendencies_land_as_the_lanes_analytic_increment(grid, sigma):
    _, col = _model(grid, sigma, fix_mass=False, use_polar_filter=False)
    nat = _native_ic(grid)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    ncol = n_lat * n_lon
    du0, dv0, dT0, dq0 = 1e-4, -2e-4, 3e-4, -1e-7

    def physics(state, mesh, coord, phys_state=None, forcing=None):
        o3 = jnp.ones_like(state.T.data)
        return _tend(state, du0 * o3, dv0 * o3, dT0 * o3,
                     tracers={"q_v": state.T.replace(data=dq0 * o3, name="dq_v")})
    ref = col.step(col.from_bundle(nat), DT)
    out = col.step(col.from_bundle(nat), DT, physics_fn=physics)
    # T: exact (no filter, no rails): dT = dt * dT/dt everywhere
    np.testing.assert_allclose(np.asarray(out.native.T - ref.native.T), DT * dT0,
                               rtol=1e-12, atol=1e-13)
    # winds: cell tendencies through the lane's OWN face interpolators,
    # v walled at the poles by the dycore
    exp_du = DT * np.asarray(interp_cell_to_uface(jnp.full((n_lat, n_lon, NLEV), du0)))
    exp_dv = DT * np.asarray(interp_cell_to_vface_halo(jnp.full((n_lat, n_lon, NLEV), dv0)))
    exp_dv[0] = 0.0
    exp_dv[-1] = 0.0
    np.testing.assert_allclose(np.asarray(out.native.u - ref.native.u), exp_du, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(np.asarray(out.native.v - ref.native.v), exp_dv, rtol=1e-12, atol=1e-13)
    # water: the mass the vapour tendency removes leaves through p_s, the
    # shared routine's rule, evaluated on the pre-physics (post-dynamics)
    # layer masses: rebuild the expectation from the no-physics step
    dq = {"q_v": jnp.full((n_lat, n_lon, NLEV), dq0)}
    _, p_s_exp = apply_physics_water_mass(ref.native.tracers, dq, ref.native.p_s, sigma, DT)
    np.testing.assert_allclose(np.asarray(out.native.p_s), np.asarray(p_s_exp), rtol=1e-13, atol=1e-9)
    assert np.abs(np.asarray(out.native.p_s - ref.native.p_s)).max() > 0.0
    # and the columns the driver sees are the view of that native state
    np.testing.assert_array_equal(np.asarray(out.T.data).reshape(n_lat, n_lon, NLEV), np.asarray(out.native.T))
    assert out.T.data.shape == (ncol, NLEV) and out.p_s.data.shape == (ncol,)


# ---------------------------------------------------------------------
# G-lane (c): the physics identity through the wrapper
# ---------------------------------------------------------------------

def test_physics_sees_the_column_view_of_the_post_dynamics_state(grid, sigma):
    """The wrapper feeds the physics (ncol[, nlev]) columns of the
    post-dynamics state: bitwise the reshape of the dycore's own cell view
    of the no-physics step, with the mesh and the coordinate the model
    owns, and the forcing / carry it was handed."""
    _, col = _model(grid, sigma, fix_mass=False)
    nat = _native_ic(grid)
    ref = col.step(col.from_bundle(nat), DT)
    seen = {}
    forcing = {"sst": jnp.full((col.mesh.nCells,), 300.0)}

    def physics(state, mesh, coord, phys_state=None, forcing=None):
        seen["state"], seen["mesh"], seen["coord"], seen["forcing"] = state, mesh, coord, forcing
        return _zero_physics(state, mesh, coord)
    col.step(col.from_bundle(nat), DT, physics_fn=physics, forcing=forcing)
    assert seen["mesh"] is col.mesh and seen["coord"] is col.sigma_coord
    hs = cgrid_to_hydrostatic(ref.native, grid)
    ncol = col.mesh.nCells
    # traced inside the jit: compare through a direct jitted evaluation of
    # the same reshape (shapes are static facts; values via a second call
    # that returns the state it saw as its "tendency")
    st = seen["state"]
    assert st.T.shape == (ncol, NLEV) and st.p_s.shape == (ncol,) and st.v is not None
    assert tuple(st.tracers) == NAMES

    def echo(state, mesh, coord, phys_state=None, forcing=None):
        return _tend(state, state.u.data, state.v.data, state.T.data,
                     tracers={"q_v": state.tracers["q_v"]}, sw_net_sfc=forcing["sst"])
    out = col.step(col.from_bundle(nat), DT, physics_fn=echo, forcing=forcing)
    # the echoed T tendency lands as dt*T_view on T: recover T_view bitwise
    T_view = np.asarray(out.native.T - ref.native.T) / DT
    np.testing.assert_allclose(T_view, np.asarray(hs.T.data), rtol=1e-12, atol=1e-10)
    # the forcing reached the physics as given and its diagnostic came out
    # of the jit as columns
    np.testing.assert_array_equal(np.asarray(col._sfc_diag[0]), np.asarray(forcing["sst"]))


def test_surface_diagnostics_merge_slot_wise_and_carry_rides(grid, sigma):
    _, col = _model(grid, sigma, fix_mass=False)
    nat = _native_ic(grid)
    ncol = col.mesh.nCells

    def _physics(rad, precip):
        def fn(state, mesh, coord, phys_state=None, forcing=None):
            extra = {"precip": jnp.full((ncol,), precip)}
            if rad is not None:
                extra["sw_net_sfc"] = jnp.full((ncol,), rad)
                extra["lw_net_sfc"] = jnp.full((ncol,), -rad)
            return _tend(state, jnp.zeros_like(state.u.data), jnp.zeros_like(state.v.data),
                         jnp.zeros_like(state.T.data), **extra)
        return fn
    st = col.step(col.from_bundle(nat), DT, physics_fn=_physics(100.0, 1.0))
    assert float(col._sfc_diag[0][0]) == 100.0 and float(col._sfc_diag[2][0]) == 1.0
    col.step(st, DT, physics_fn=_physics(None, 2.0))
    # a held-radiation step keeps the last radiation slots, refreshes precip
    assert float(col._sfc_diag[0][0]) == 100.0 and float(col._sfc_diag[1][0]) == -100.0
    assert float(col._sfc_diag[2][0]) == 2.0


def test_no_retrace_after_warm_up_under_a_changing_forcing(grid, sigma):
    """The forcing is TRACED: a new SST every step, the same physics_fn
    object, one compiled program (the no-retrace requirement of the design)."""
    _, col = _model(grid, sigma, fix_mass=False)
    nat = _native_ic(grid)

    def physics(state, mesh, coord, phys_state=None, forcing=None):
        return _tend(state, jnp.zeros_like(state.u.data), jnp.zeros_like(state.v.data),
                     1e-6 * forcing["sst"][:, None] * jnp.ones_like(state.T.data))
    # the jitted _step_cgrid calls _step_cgrid_impl once per TRACE (self is
    # a static argument, so the instance attribute is looked up then)
    traces = []
    impl = col.dyn._step_cgrid_impl

    def counting(*a, **k):
        traces.append(1)
        return impl(*a, **k)
    col.dyn._step_cgrid_impl = counting
    st = col.from_bundle(nat)
    st = col.step(st, DT, physics_fn=physics, forcing={"sst": jnp.full((col.mesh.nCells,), 1.0)})
    assert len(traces) == 1 and len(col._wrapped) == 1
    for k in range(2, 6):
        st = col.step(st, DT, physics_fn=physics,
                      forcing={"sst": jnp.full((col.mesh.nCells,), float(k))})
    assert len(traces) == 1 and len(col._wrapped) == 1
    assert np.isfinite(np.asarray(st.native.T)).all()
    # and the forcing WAS read: the last step's heating differs from the first's
    T1 = np.asarray(col.step(st, DT, physics_fn=physics,
                             forcing={"sst": jnp.full((col.mesh.nCells,), 1.0)}).native.T)
    T9 = np.asarray(col.step(st, DT, physics_fn=physics,
                             forcing={"sst": jnp.full((col.mesh.nCells,), 9.0)}).native.T)
    assert np.abs(T9 - T1).max() > 0.0 and len(traces) == 1


def test_refuses_physics_without_the_meridional_tendency(grid, sigma):
    _, col = _model(grid, sigma, fix_mass=False)

    def no_v(state, mesh, coord, phys_state=None, forcing=None):
        return _zero_physics(state, mesh, coord)._replace(dv_dt=None)
    with pytest.raises(ValueError, match="dv_dt=None"):
        col.step(col.from_bundle(_native_ic(grid)), DT, physics_fn=no_v)


# ---------------------------------------------------------------------
# the driver lane
# ---------------------------------------------------------------------

def _driver_cfg(tmp_path, **over):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
    grid = GridConfig(grid_type="latlon", resolution=N_LAT, nlev=NLEV)
    dycore = DycoreConfig(
        model_type="hydrostatic", discretization="latlon_cgrid",
        dt=over.pop("dt", DT), use_polar_filter=over.pop("polar", True),
        column_lane=over.pop("column", True))
    base = dict(
        grid=grid, dycore=dycore, days=over.pop("days", 3 * DT / 86400.0),
        radiation="none", convection="none", microphysics="none",
        turbulence="none", gravity_wave_drag="none", precision="fp64",
        output=OutputConfig(diag_days=over.pop("diag_days", 0),
                            checkpoint_days=over.pop("checkpoint_days", 0),
                            output_dir=str(tmp_path)))
    base.update(over)
    return ExperimentConfig(**base)


def _run_driver(tmp_path, **over):
    from legoesm.driver.model_driver import ModelDriver
    cfg = _driver_cfg(tmp_path, **over)
    drv = ModelDriver(cfg, output_dir=tmp_path)
    drv.setup()
    status = drv.run()
    assert status == "COMPLETED", status
    return drv


def test_driver_column_lane_is_the_native_grids_mesh_and_runs(tmp_path):
    drv = _run_driver(tmp_path)
    assert isinstance(drv.model, LatLonColumnModel)
    assert isinstance(drv.state, LatLonColumnState)
    assert drv.grid is drv.model.mesh and drv.sigma is drv.model.sigma_coord
    assert drv.grid.nCells == N_LAT * 2 * N_LAT
    # the terrain / land fraction are columns of the native product
    assert np.asarray(drv._phis_data).shape == (drv.grid.nCells,)
    assert np.asarray(drv._f_land).shape == (drv.grid.nCells,)
    # the dycore was built on the NATIVE grid (its own CFL clamp and
    # coefficients), the state advanced
    assert drv.model.native_grid.n_lat == N_LAT
    assert np.isfinite(np.asarray(drv.state.native.T)).all()
    assert tuple(drv.state.tracers) == NAMES
    assert np.abs(np.asarray(drv.state.native.T) - np.asarray(drv._state_template.native.T)).max() > 0.0


def test_driver_column_lane_refusals(tmp_path):
    from legoesm.driver.model_driver import ModelDriver
    for over, frag in ((dict(sponge_enabled=True), "sponge_enabled"),
                       (dict(mpas_qv_smooth_del4_m4s=1e14), "MPAS-lane knob|smoothing|edge operator"),
                       (dict(held_suarez_forcing=True), "meridional drag|held_suarez"),
                       (dict(convection="kuo"), "grid operator"),
                       (dict(distributed=True), "single-process")):
        drv = ModelDriver(_driver_cfg(tmp_path, **over), output_dir=tmp_path)
        with pytest.raises((ValueError, NotImplementedError), match=frag):
            drv.setup()


def test_checkpoint_carries_the_face_winds_and_refuses_a_file_without_them(tmp_path):
    """The column view's cell winds cannot rebuild the C-grid faces: the
    checkpoint carries them, a fresh driver restores the native state
    bitwise, and a file stripped of them (an MPAS-lane checkpoint) is
    refused -- the planted violation of this test."""
    from legoesm.driver.model_driver import ModelDriver
    a = tmp_path / "a"
    a.mkdir()
    # dt 600 = the driver's generic CFL clamp at this grid (any larger dt
    # is cut to it): one day = 144 steps
    drv = _run_driver(a, days=1.0, dt=600.0, checkpoint_days=1)
    ck = a / "checkpoint_day_0001.npz"
    assert ck.is_file(), sorted(p.name for p in a.iterdir())
    with np.load(ck) as d:
        for k in LATLON_CKPT_KEYS:
            assert k in d.files
        np.testing.assert_array_equal(d[LATLON_CKPT_KEYS[0]], np.asarray(drv.state.native.u))
        np.testing.assert_array_equal(d[LATLON_CKPT_KEYS[1]], np.asarray(drv.state.native.v))
        files = {k: d[k] for k in d.files}
    b = tmp_path / "b"
    b.mkdir()
    drv_b = ModelDriver(_driver_cfg(b, days=1.0, dt=600.0), output_dir=b)
    drv_b.setup()
    step, day = drv_b.load_checkpoint(ck)
    assert step == 144
    _assert_native_equal(drv_b.state.native, drv.state.native)
    # planted violation: the same file without the face winds
    bad = tmp_path / "bad.npz"
    np.savez(bad, **{k: v for k, v in files.items() if k not in LATLON_CKPT_KEYS})
    drv_c = ModelDriver(_driver_cfg(tmp_path / "c", days=1.0, dt=600.0),
                        output_dir=tmp_path / "c")
    drv_c.setup()
    with pytest.raises(ValueError, match="face winds"):
        drv_c.load_checkpoint(bad)
    # and a file carrying ANOTHER tracer set (a nine-slot ice deck's) is
    # refused too, never silently reduced to the registered three (the
    # dropped water would become dry air under the retained total p_s)
    wide = tmp_path / "wide.npz"
    extra = {**files, "tracer_names": np.asarray(sorted(list(NAMES) + ["q_i"])),
             "trc_q_i": np.zeros_like(files["trc_q_v"])}
    np.savez(wide, **extra)
    drv_d = ModelDriver(_driver_cfg(tmp_path / "d", days=1.0, dt=600.0),
                        output_dir=tmp_path / "d")
    drv_d.setup()
    with pytest.raises(ValueError, match="tracer"):
        drv_d.load_checkpoint(wide)
    # and a file WITHOUT the tracer metadata is not a dry restart but a
    # refusal (codex r2 #1: it loaded with no tracers under the moist p_s)
    bare = tmp_path / "bare.npz"
    np.savez(bare, **{k: v for k, v in files.items()
                      if k != "tracer_names" and not k.startswith("trc_")})
    drv_e = ModelDriver(_driver_cfg(tmp_path / "e", days=1.0, dt=600.0),
                        output_dir=tmp_path / "e")
    drv_e.setup()
    with pytest.raises(ValueError, match="tracer_names"):
        drv_e.load_checkpoint(bare)


# ---------------------------------------------------------------------
# the land / radiation deck of the restart-digest gate (G-lane d)
# ---------------------------------------------------------------------

def _latlon_land_stress_cfg(tmp_path, monkeypatch, *, stress, days, **over):
    """The duo digest test's land-stress deck on the lat-lon column lane:
    RRTMGP, Louis turbulence, the multilayer land with its fluxes and
    stress handed to the boundary layer; land north of the equator only
    (loaders patched: no network, synthetic surface map)."""
    from tests.atmosphere.hydrostatic.unit.test_fv3_duo_column import (
        _fake_surface_map_by_latitude)
    import legoesm.grids.topography as topo
    import legoesm.land.clm_surface_map as clm
    monkeypatch.setattr(clm, "download_clm_surfdata", lambda *a, **k: "synthetic")
    monkeypatch.setattr(clm, "load_clm_surface", _fake_surface_map_by_latitude)
    monkeypatch.setattr(
        topo, "load_land_fraction",
        lambda grid, path, *a, **k: jnp.where(
            jnp.asarray(grid.grid_lat) > 0.0, 0.6, 0.0))
    return _driver_cfg(
        tmp_path, days=days, radiation="rrtmgp", turbulence="louis",
        land_mask_path="synthetic.nc", use_multilayer_land=True,
        multilayer_n_layers=6, multilayer_soil_depth=2.5,
        land_surface_scheme="simple_seb", land_params_refresh=False,
        land_beta_soil=True, land_stress_from_land=stress, **over)


def test_production_deck_dt_is_not_clamped_and_keeps_the_cam6_cadence():
    """The lat-lon production deck's dt survives BOTH clamps (the factory's
    equatorial CFL and the driver's generic cfl_check_and_adjust, which cut
    900 s to 600 s and would have halved every cadence; codex r1 #1), and
    dt x the cadence counts are CAM6's 1800 s / 3600 s intervals."""
    import yaml
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
