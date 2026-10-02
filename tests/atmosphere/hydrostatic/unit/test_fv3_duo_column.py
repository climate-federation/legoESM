"""FV3 duo as a column model behind the MPAS lane's step contract (route A,
M2): certification ladder rungs 1-3 (plumbing identity, Held-Suarez,
Kessler) at C12 km=5, plus the refusals and one real MPAS-physics package
(Louis turbulence) driven through it.

Every identity is against the CLOSED duo lane's own operators on the same
post-dynamics bundle (the certified twins), bitwise unless stated.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.core.state import HydrostaticTendencies  # noqa: E402
from legoesm.grids.factory import create_fv3_duo_grid  # noqa: E402

N, NG, KM = 12, 3, 5
DT = 120.0
CI = slice(NG, NG + N)
CE = slice(NG, NG + N + 1)


@pytest.fixture(autouse=True)
def _drop_compiled_graphs():
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def grid():
    return create_fv3_duo_grid(N, NG)


def _model(grid, **cfg):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import (
        FV3DuoColumnModel)
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    dyn = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=KM, n_split=2, **cfg))
    return dyn, FV3DuoColumnModel(dyn)


@pytest.fixture(scope="module")
def dry(grid):
    dyn, col = _model(grid)
    return dyn, col, dyn.dcmip16_initial_state(n_tracers=3)


def _fld(like, data, name):
    return like.replace(data=data, name=name)


def _tend(state, du, dv, dT, tracers=None):
    z = jnp.zeros_like(state.p_s.data)
    return HydrostaticTendencies(
        du_dt=_fld(state.u, du, "du_dt"), dv_dt=_fld(state.v, dv, "dv_dt"),
        dT_dt=_fld(state.T, dT, "dT_dt"), dp_s_dt=_fld(state.p_s, z, "dp_s"),
        dphis_dt=_fld(state.phis, z, "dphis"), tracer_tendencies=tracers)


def _zero_physics(state, mesh, coord, phys_state=None, forcing=None):
    return _tend(state, jnp.zeros_like(state.u.data),
                 jnp.zeros_like(state.v.data), jnp.zeros_like(state.T.data))


def _assert_bundle_equal(a, b, *, rel=0.0):
    """Compute-window equality of two six-face bundles (D-grid winds on
    their (n, n+1)/(n+1, n) windows, everything else on (n, n)) to
    ``rel`` of each field's peak; ``rel=0`` is bitwise.  Two DIFFERENT
    jit programs of the same math fuse differently and differ at
    ~1e-15 of peak (the closed lane's known 2e-12 hook-vs-eager class),
    so cross-program identities use ``rel=1e-12``, the twin gate's."""
    win = {"u": (slice(None), CI, CE), "v": (slice(None), CE, CI)}

    def chk(x, y, nm):
        x, y = np.asarray(x), np.asarray(y)
        np.testing.assert_allclose(x, y, rtol=0, atol=rel * np.abs(y).max(),
                                   err_msg=nm)
    for k in ("u", "v", "pt", "delp"):
        w = win.get(k, (slice(None), CI, CI))
        chk(a["state"][k][w], b["state"][k][w], k)
    for k in a["press"]:
        chk(a["press"][k], b["press"][k], k)
    for i, (qa, qb) in enumerate(zip(a["q"], b["q"])):
        chk(qa[:, CI, CI], qb[:, CI, CI], f"q{i}")


# ---------------------------------------------------------------------
# the view
# ---------------------------------------------------------------------

def test_mesh_and_coordinate_are_the_duo_grid(dry):
    dyn, col, ic = dry
    m = col.mesh
    assert m.nCells == 6 * N * N
    assert float(jnp.abs(m.latCell).max()) <= np.pi / 2
    area = float(m.areaCell.sum())
    # the gridstruct's cell areas sum to the sphere to 3.5e-6 relative at
    # C12 (MEASURED 2026-09-26; the closed lane's own metric, not exact
    # spherical excess) -- a finding for the metric, not this view
    assert np.isclose(area, 4 * np.pi * constants.R_earth ** 2, rtol=1e-5)
    st = col.from_bundle(ic)
    assert st.u.data.shape == st.v.data.shape == st.T.data.shape == (
        m.nCells, KM)
    assert st.p_s.data.shape == (m.nCells,)
    assert set(st.tracers) == {"q_v", "q_c", "q_r"}
    # the hybrid coordinate reproduces the bundle's OWN interface
    # pressures from p_s (the ak/bk -> A/B mapping is exact)
    for b in (ic, dyn.step(ic, DT)):     # at the IC and post-remap
        s_ = col.from_bundle(b)
        pe = np.transpose(np.asarray(b["press"]["pe"])[:, 1:N + 1, :, 1:N + 1],
                          (0, 1, 3, 2)).reshape(m.nCells, KM + 1)
        ph = np.asarray(col.sigma_coord.pressure_at_half(s_.p_s.data))
        np.testing.assert_allclose(ph, pe, rtol=1e-12, atol=0)
    # the column winds ARE the closed lane's order-4 c2l view
    from legoesm.core.fv3_native_physics_coupling import (
        column_view_sixface_jax)
    _, _, ua6, va6 = column_view_sixface_jax(
        ic["state"], col._tab, col._amat6, n=N, ng=NG, km=KM)
    ua = np.asarray(ua6)[:, CI, CI].reshape(-1, KM)
    np.testing.assert_allclose(np.asarray(st.u.data), ua, rtol=0,
                               atol=1e-12 * np.abs(ua).max())
    assert float(jnp.abs(st.v.data).max()) > 0.0


# ---------------------------------------------------------------------
# rung 1: plumbing identity
# ---------------------------------------------------------------------

def test_zero_tendency_physics_is_the_closed_lane_bitwise(dry):
    dyn, col, ic = dry
    closed = dyn.step(ic, DT)
    out = col.step(col.from_bundle(ic), DT, physics_fn=_zero_physics)
    _assert_bundle_equal(out.native, closed)
    # the exchanged halos the column path leaves behind do not change the
    # next dynamics step either
    _assert_bundle_equal(dyn.step(out.native, DT), dyn.step(closed, DT))
    assert col._phys_state is None


# ---------------------------------------------------------------------
# rung 2: Held-Suarez through the contract == the closed lane's twin
# ---------------------------------------------------------------------

def _held_suarez_physics(col):
    from legoesm.core.fv3_native_physics_coupling import (
        column_view_sixface_jax, held_suarez_tend_jax,
        stack_held_suarez_metrics)
    _, lat6, _ = stack_held_suarez_metrics(col.grid.ctx_np)
    lat6 = jnp.asarray(lat6)[:, CI, CI]

    def physics(state, mesh, coord, phys_state=None, forcing=None):
        b = state.native
        # the tendency reads the bundle's own pe/peln/pkz (the twin's
        # inputs); the winds are the state's columns (the SAME view)
        peln6 = jnp.transpose(jnp.asarray(b["press"]["peln"]), (0, 1, 3, 2))
        pe6 = jnp.transpose(
            jnp.asarray(b["press"]["pe"])[:, 1:N + 1, :, 1:N + 1],
            (0, 1, 3, 2))
        ua = col._faces(state.u.data)
        va = col._faces(state.v.data)
        t_dt, u_dt, v_dt = jax.vmap(
            lambda pt, ua_, va_, delp, peln, pkz, pe, lat:
            held_suarez_tend_jax(pt, ua_, va_, delp, peln, pkz, pe, lat, DT,
                                 strat=True))(
            jnp.asarray(b["state"]["pt"])[:, CI, CI], ua, va,
            jnp.asarray(b["state"]["delp"])[:, CI, CI], peln6,
            jnp.asarray(b["press"]["pkz"]), pe6, lat6)
        flat = lambda a: a.reshape(mesh.nCells, KM)  # noqa: E731
        return _tend(state, flat(u_dt), flat(v_dt), flat(t_dt))
    return physics


def test_held_suarez_through_the_contract_is_the_twin_bitwise(dry):
    from legoesm.core.fv3_native_physics_coupling import (
        apply_held_suarez_step, apply_held_suarez_step_sixface_jax,
        stack_held_suarez_metrics)
    dyn, col, ic = dry
    post = dyn.step(ic, DT)
    amat6, lat6, wv6 = stack_held_suarez_metrics(col.grid.ctx_np)
    twin = apply_held_suarez_step_sixface_jax(
        post["state"], post["press"], col._tab, amat6, lat6, wv6, dt=DT,
        n=N, ng=NG, km=KM, strat=True)
    out = col.step(col.from_bundle(ic), DT,
                   physics_fn=_held_suarez_physics(col))
    for k in ("u", "v", "pt"):
        np.testing.assert_array_equal(np.asarray(out.native["state"][k]),
                                      np.asarray(twin[k]), err_msg=k)
        assert np.abs(np.asarray(twin[k]) - np.asarray(post["state"][k])
                      ).max() > 0.0, f"vacuous: HS did not move {k}"
    np.testing.assert_array_equal(np.asarray(out.native["state"]["delp"]),
                                  np.asarray(post["state"]["delp"]))
    # and the closed lane's NumPy authority, at the twin gate's tolerance
    st = [{k: np.array(np.asarray(post["state"][k])[t])
           for k in ("u", "v", "pt", "delp")} for t in range(6)]
    pr = [{k: np.asarray(post["press"][k][t]) for k in ("pe", "peln", "pkz")}
          for t in range(6)]
    apply_held_suarez_step(col.grid.ctx_np, st, pr, dt=DT, n=N, ng=NG,
                           km=KM, strat=True, backend="numpy")
    for k in ("u", "v", "pt"):
        ref = np.stack([f[k] for f in st])
        np.testing.assert_allclose(np.asarray(out.native["state"][k]), ref,
                                   rtol=1e-12, atol=1e-11, err_msg=k)


# ---------------------------------------------------------------------
# rung 3: Kessler through the contract == the closed lane's bridge
# ---------------------------------------------------------------------

def _saturated_rainy(ic, q_c=2e-3, q_r=2e-2):
    from legoesm.thermo import saturation_mixing_ratio
    peln = jnp.transpose(jnp.asarray(ic["press"]["peln"]), (0, 1, 3, 2))
    p_full = (jnp.asarray(ic["state"]["delp"])[:, CI, CI]
              / (peln[..., 1:] - peln[..., :-1]))
    q = [jnp.asarray(a) for a in ic["q"]]
    q[0] = q[0].at[:, CI, CI].set(saturation_mixing_ratio(
        jnp.asarray(ic["state"]["pt"])[:, CI, CI], p_full))
    q[1] = jnp.full_like(q[1], q_c)
    q[2] = jnp.full_like(q[2], q_r)
    return {**ic, "q": q + [q[0] * 0.5]}     # + a passenger


@pytest.fixture(scope="module")
def moist(grid):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import (
        FV3DuoColumnModel)
    dyn, _ = _model(grid, moist=True)
    col = FV3DuoColumnModel(dyn, tracer_names=("q_v", "q_c", "q_r", "q_p"))
    return dyn, col, _saturated_rainy(dyn.dcmip16_initial_state(n_tracers=3))


def _kessler_physics(col):
    from legoesm.atmosphere.forcing.idealized.kessler_forcing import (
        kessler_tendencies_sixface_jax)

    def physics(state, mesh, coord, phys_state=None, forcing=None):
        b = state.native
        t_dt, q_dt = kessler_tendencies_sixface_jax(
            b["state"], b["press"], b["q"], dt=DT, n=N, ng=NG, km=KM)
        flat = lambda a: a.reshape(mesh.nCells, KM)  # noqa: E731
        z = jnp.zeros_like(state.u.data)
        return _tend(state, z, z, flat(t_dt), tracers={
            nm: _fld(state.tracers[nm], flat(d), nm)
            for nm, d in zip(("q_v", "q_c", "q_r"), q_dt)})
    return physics


def test_kessler_through_the_contract_is_the_bridge(moist):
    from legoesm.atmosphere.forcing.idealized.kessler_forcing import (
        apply_kessler_step_sixface_jax)
    from legoesm.grids.fv3_native_gridstruct import FV3_KAPPA
    dyn, col, ic = moist
    post = dyn.step(ic, DT)
    st, pr, q = jax.jit(lambda s, p, q_: apply_kessler_step_sixface_jax(
        s, p, q_, dt=DT, n=N, ng=NG, km=KM, ptop=dyn.ptop, akap=FV3_KAPPA))(
        post["state"], post["press"], list(post["q"]))
    bridge = {**post, "state": st, "press": pr, "q": q}
    out = col.step(col.from_bundle(ic), DT, physics_fn=_kessler_physics(col))
    _assert_bundle_equal(out.native, bridge, rel=1e-12)
    # non-vacuous: rain reached the surface and took layer mass with it
    dps = np.asarray(bridge["press"]["ps"]) - np.asarray(post["press"]["ps"])
    assert dps[:, CI, CI].min() < 0.0
    # winds: exactly the dynamics winds on the compute window (Kessler
    # has no momentum tendency; the halo strips were exchanged for the view)
    np.testing.assert_array_equal(
        np.asarray(out.native["state"]["u"])[:, CI, CE],
        np.asarray(post["state"]["u"])[:, CI, CE])
    np.testing.assert_array_equal(
        np.asarray(out.native["state"]["v"])[:, CE, CI],
        np.asarray(post["state"]["v"])[:, CE, CI])


# ---------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------

def test_refuses_physics_without_the_meridional_tendency(dry):
    dyn, col, ic = dry

    def no_v(state, mesh, coord, phys_state=None, forcing=None):
        t = _zero_physics(state, mesh, coord)
        return t._replace(dv_dt=None)
    with pytest.raises(ValueError, match="dv_dt=None"):
        col.step(col.from_bundle(ic), DT, physics_fn=no_v)


def test_uncarried_tracer_tendency_dropped_and_passenger_refused(moist):
    dyn, col, ic = moist

    def ice(state, mesh, coord, phys_state=None, forcing=None):
        t = _zero_physics(state, mesh, coord)
        return t._replace(tracer_tendencies={
            "q_i": _fld(state.T, jnp.ones_like(state.T.data), "q_i")})
    # a tendency for a tracer the state does not carry is DROPPED, the
    # MPAS model's contract (the integrations emit the full set)
    out = col.step(col.from_bundle(ic), DT, physics_fn=ice)
    _assert_bundle_equal(out.native, dyn.step(ic, DT))

    # a tendency on a PASSENGER slot (nwat = 3 here, q_p is slot 3) is
    # applied and renormalised like FV3's other mass tracers (:324/:352)
    # and moves no layer mass
    def passenger(state, mesh, coord, phys_state=None, forcing=None):
        t = _zero_physics(state, mesh, coord)
        return t._replace(tracer_tendencies={
            "q_p": _fld(state.T, jnp.full_like(state.T.data, 1e-6), "q_p")})
    out = col.step(col.from_bundle(ic), DT, physics_fn=passenger)
    ref = dyn.step(ic, DT)
    np.testing.assert_array_equal(np.asarray(out.native["state"]["delp"]),
                                  np.asarray(ref["state"]["delp"]))
    np.testing.assert_allclose(
        np.asarray(out.native["q"][3])[:, CI, CI],
        (np.asarray(ref["q"][3]) + DT * 1e-6)[:, CI, CI], rtol=1e-13, atol=0)


def test_refuses_water_tendencies_on_the_dry_deck(dry):
    dyn, col, ic = dry

    def wet(state, mesh, coord, phys_state=None, forcing=None):
        t = _zero_physics(state, mesh, coord)
        return t._replace(tracer_tendencies={
            "q_v": _fld(state.T, jnp.zeros_like(state.T.data), "q_v")})
    with pytest.raises(ValueError, match="DRY deck"):
        col.step(col.from_bundle(ic), DT, physics_fn=wet)


def test_refuses_tracer_names_that_do_not_name_the_warm_rain_slots(dry):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import (
        FV3DuoColumnModel)
    dyn = dry[0]
    for names in ((), ("q_c", "q_v", "q_r"), ("q_v", "q_c", "q_r", "q_v")):
        with pytest.raises(ValueError, match="tracer_names"):
            FV3DuoColumnModel(dyn, tracer_names=names)


def test_surface_diagnostics_merge_slot_wise(dry):
    """A held-radiation step (sw/lw None, precip set) keeps the last
    radiation fluxes, as the MPAS model's stash does."""
    dyn, col, ic = dry
    ncell = col.mesh.nCells

    from types import SimpleNamespace

    def _physics(rad, precip):
        def physics(state, mesh, coord, phys_state=None, forcing=None):
            d = _zero_physics(state, mesh, coord)._asdict()
            d.update(sw_net_sfc=jnp.full((ncell,), 100.0) if rad else None,
                     lw_net_sfc=jnp.full((ncell,), -50.0) if rad else None,
                     precip=jnp.full((ncell,), precip))
            return SimpleNamespace(**d)
        return physics
    st = col.step(col.from_bundle(ic), DT, physics_fn=_physics(True, 1.0))
    assert float(col._sfc_diag[0][0]) == 100.0
    col.step(st, DT, physics_fn=_physics(False, 2.0))
    assert col._sfc_diag[0] is not None and float(col._sfc_diag[0][0]) == 100.0
    assert float(col._sfc_diag[2][0]) == 2.0


def test_refuses_the_window_layout_and_nh(grid, dry):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import (
        FV3DuoColumnModel)
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    dyn = dry[0]
    saved = dyn.window_layout
    dyn.window_layout = object()          # stand-in for a window layout
    try:
        with pytest.raises(NotImplementedError, match="rung 7"):
            FV3DuoColumnModel(dyn)
    finally:
        dyn.window_layout = saved
    with pytest.raises(NotImplementedError, match="NH"):
        FV3DuoColumnModel(FV3DuoDynamicsModel(
            grid, FV3DuoConfig(km=KM, n_split=2, hydrostatic=False)))


# ---------------------------------------------------------------------
# a real MPAS-lane physics package through the contract
# ---------------------------------------------------------------------

def test_louis_turbulence_runs_and_moves_both_wind_components(moist):
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    dyn, col, ic = moist
    fn = make_physics(PhysicsConfig(turbulence=TurbulenceConfig(scheme="louis")),
                      model_type="mpas", dt=DT)
    st0 = col.from_bundle(ic)
    forcing = {"T_sfc": st0.T.data[:, -1] + 2.0}
    closed = dyn.step(ic, DT)
    out = col.step(st0, DT, physics_fn=fn, forcing=forcing)
    for k in ("u", "v", "pt"):
        got = np.asarray(out.native["state"][k])
        assert np.isfinite(got).all(), k
        assert np.abs(got - np.asarray(closed["state"][k])).max() > 0.0, k
    # the applied D-wind increment is the dry twin of the returned
    # cell tendencies (both components carried, none dropped)
    post_cols, view = col.column_view(closed)
    t = fn(post_cols, col.mesh, col.sigma_coord, forcing=forcing)
    t = t[0] if isinstance(t, tuple) else t
    _d = lambda x: getattr(x, "data", x)  # noqa: E731
    assert float(jnp.abs(_d(t.dv_dt)).max()) > 0.0
    from legoesm.core.fv3_native_physics_coupling import (
        apply_column_increments_sixface_jax)
    from legoesm.grids.fv3_native_gridstruct import FV3_KAPPA
    q_dt = {col._tracer_index(nm): col._faces(_d(v))
            for nm, v in (t.tracer_tendencies or {}).items()}
    st, pr, q = jax.jit(lambda: apply_column_increments_sixface_jax(
        closed["state"], closed["press"], list(closed["q"]), view, col._tab,
        col._wv6, col._faces(_d(t.du_dt)), col._faces(_d(t.dv_dt)),
        col._faces(_d(t.dT_dt)), q_dt, dt=DT, n=N, ng=NG, km=KM,
        ptop=dyn.ptop, akap=FV3_KAPPA, moist_cp=True))()
    _assert_bundle_equal(out.native, {**closed, "state": st, "press": pr,
                                      "q": q}, rel=1e-12)
    # budgets (GLM 2026-09-26): the physics' water increment in MPAS's
    # fixed-mass convention, dt*sum_k delp*dq (NOT zero: Louis evaporates
    # from the T_sfc forcing at the bottom level), is exactly the water
    # mass the moist block adds, the surface pressure rises by that
    # mass, and the DRY mass delp*(1 - q) of every column is unchanged
    # (Louis diffuses cloud and rain too: every warm-rain tendency counts)
    dq = sum(np.asarray(_d(v)).reshape(-1, KM)
             for nm, v in t.tracer_tendencies.items()
             if nm in ("q_v", "q_c", "q_r"))
    delp0 = np.asarray(closed["state"]["delp"])[:, CI, CI].reshape(-1, KM)
    delp1 = np.asarray(out.native["state"]["delp"])[:, CI, CI].reshape(-1, KM)
    qw0 = sum(np.asarray(closed["q"][i])[:, CI, CI].reshape(-1, KM)
              for i in range(3))
    qw1 = sum(np.asarray(out.native["q"][i])[:, CI, CI].reshape(-1, KM)
              for i in range(3))
    src = DT * (delp0 * dq).sum(axis=1)
    assert np.abs(src).max() > 0.0
    # differences of O(1e5 Pa) fields resolve to ~1e-11: tolerance on
    # the differenced field's scale, not on the (small) difference
    w0, w1 = (delp0 * qw0).sum(axis=1), (delp1 * qw1).sum(axis=1)
    np.testing.assert_allclose(w1 - w0, src, rtol=0,
                               atol=1e-12 * np.abs(w0).max())
    ps0 = np.asarray(closed["press"]["ps"])[:, CI, CI].reshape(-1)
    ps1 = np.asarray(out.native["press"]["ps"])[:, CI, CI].reshape(-1)
    np.testing.assert_allclose(ps1 - ps0, src, rtol=0,
                               atol=1e-12 * np.abs(ps0).max())
    np.testing.assert_allclose((delp1 * (1.0 - qw1)).sum(axis=1),
                               (delp0 * (1.0 - qw0)).sum(axis=1), rtol=1e-12)


# ---------------------------------------------------------------------
# M3: the driver's MPAS lane with the duo as its dynamics operator
# ---------------------------------------------------------------------

def _driver_cfg(tmp_path, **over):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
    grid = GridConfig(grid_type="cubed_sphere", resolution=N, nlev=KM)
    dycore = DycoreConfig(
        model_type="hydrostatic", discretization="fv3_duo",
        dt=over.pop("dt", 1920.0),
        fv3_duo_column_lane=over.pop("column", True))
    base = dict(
        grid=grid, dycore=dycore, days=over.pop("days", 0.25),
        radiation="none", convection="none", microphysics="none",
        turbulence="none", gravity_wave_drag="none", precision="fp64",
        output=OutputConfig(diag_days=over.pop("diag_days", 0),
                            checkpoint_days=0, output_dir=str(tmp_path)))
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


def test_driver_column_lane_is_the_closed_lane_bitwise(tmp_path):
    """Rung 1 at the driver level: ``run()`` through the MPAS lane with the
    duo column model and no physics reproduces the closed duo lane's
    bundle bitwise after the same number of steps (same IC builder, same
    dt), and the state the driver ends on is the column view."""
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import (
        FV3DuoColumnModel, FV3DuoColumnState)
    col = _run_driver(tmp_path / "col")
    assert isinstance(col.model, FV3DuoColumnModel)
    assert isinstance(col.state, FV3DuoColumnState)
    assert col.grid is col.model.mesh and col.sigma is col.model.sigma_coord
    closed = _run_driver(tmp_path / "closed", column=False)
    assert isinstance(closed.state, dict)
    n_steps = int(0.25 * 86400.0 / 1920.0)
    assert n_steps == 11
    _assert_bundle_equal(col.state.native, closed.state)
    assert np.abs(np.asarray(col.state.native["state"]["pt"])
                  - np.asarray(closed.model.dcmip16_initial_state(
                      do_pert=True)["state"]["pt"])).max() > 0.0


def test_driver_column_lane_kessler_equals_the_closed_lane(tmp_path):
    """Kessler through the MPAS lane's own physics package on the column
    model vs the closed lane's bridge, same IC, same dt: the two runs
    agree to 1e-12 of peak.  On the (unsaturated) DCMIP16 columns
    Kessler's tendency is exactly zero on both lanes (MEASURED
    2026-09-26), so this is the PLUMBING identity of the moist deck
    (moist dycore arm, three tracers, the physics package called every
    step, no fixer); the physics identity is the M2 rung.  The dry-mass
    drift is the dycore's own on both lanes (the moist arm creates ~7e-7
    of global water per step under pure advection -- a closed-lane
    finding), so it is asserted EQUAL between the lanes, not zero."""
    col = _run_driver(tmp_path / "col", microphysics="kessler")
    closed = _run_driver(tmp_path / "closed", column=False,
                         microphysics="kessler")
    _assert_bundle_equal(col.state.native, closed.state, rel=1e-12)
    area = np.asarray(col.model.mesh.areaCell).reshape(6, N, N)

    def dry_mass(bb):
        delp = np.asarray(bb["state"]["delp"])[:, CI, CI]
        qw = sum(np.asarray(bb["q"][i])[:, CI, CI] for i in range(3))
        return float(((delp * (1.0 - qw)).sum(axis=-1) * area).sum())
    assert np.isclose(dry_mass(col.state.native), dry_mass(closed.state),
                      rtol=1e-12)
    for k in ("u", "v", "pt", "delp"):
        assert np.isfinite(np.asarray(col.state.native["state"][k])).all(), k


def test_driver_column_lane_refusals(tmp_path):
    from legoesm.driver.model_driver import ModelDriver
    for over, frag in ((dict(sponge_enabled=True), "sponge_enabled"),
                       (dict(mpas_qv_smooth_del4_m4s=1e14),
                        "MPAS-lane knob|smoothing"),
                       (dict(held_suarez_forcing=True), "hswf"),
                       (dict(topography="gaussian"), "only ic='era5'"),
                       (dict(convection="kuo"), "grid operator"),
                       (dict(convection="kain_fritsch"), "grid operator")):
        drv = ModelDriver(_driver_cfg(tmp_path, **over), output_dir=tmp_path)
        with pytest.raises(ValueError, match=frag):
            drv.setup()


def test_column_model_refuses_wind_edits_and_writes_back_T(dry):
    dyn, col, ic = dry
    st = col.from_bundle(ic)
    edited = st._replace(u=st.u.replace(data=st.u.data * 0.5))
    with pytest.raises(ValueError, match="state.u was edited"):
        col.step(edited, DT, physics_fn=_zero_physics)
    # a T + tracer edit (the hard-saturation drain's shape: T and a
    # COPIED tracer dict, whose keys a jitted output leaves sorted)
    # reaches the bundle
    trc = {k: st.tracers[k] for k in sorted(st.tracers)}
    trc["q_v"] = trc["q_v"].replace(data=trc["q_v"].data * 0.5)
    warmed = st._replace(T=st.T.replace(data=st.T.data + 1.0), tracers=trc)
    out = col.step(warmed, DT, physics_fn=_zero_physics)
    ref = dyn.step({**ic, "state": {**ic["state"], "pt": jnp.asarray(
        ic["state"]["pt"]).at[:, CI, CI].add(1.0)},
        "q": [jnp.asarray(ic["q"][0]).at[:, CI, CI].multiply(0.5)]
        + list(ic["q"][1:])}, DT)
    _assert_bundle_equal(out.native, ref)


# ---------------------------------------------------------------------
# M5: checkpoint / restart on the column lane
# ---------------------------------------------------------------------

def _same_bytes(x, y):
    """Bitwise: same shape, same dtype, finite, identical bytes (no
    equal_nan, no signed-zero leniency -- codex 2026-09-30)."""
    x, y = np.asarray(x), np.asarray(y)
    return (x.shape == y.shape and x.dtype == y.dtype
            and np.isfinite(x).all() and np.isfinite(y).all()
            and x.tobytes() == y.tobytes())


def _walk_bundle(b):
    """Every array of a duo bundle, enumerated from the STRUCTURE (not
    the writer's own flattener, which would make the claim circular)."""
    out = {}
    for k, v in b["state"].items():
        out[f"state.{k}"] = np.asarray(v)
    for k, v in b["press"].items():
        out[f"press.{k}"] = np.asarray(v)
    for i, v in enumerate(b["q"]):
        out[f"q.{i}"] = np.asarray(v)
    out["omga"] = np.asarray(b["omga"])
    return out


@pytest.mark.parametrize("phys", [
    dict(),
    dict(microphysics="kessler"),
    dict(microphysics="kessler", turbulence="mynn25"),
])
def test_driver_column_lane_restart_is_bitwise(tmp_path, phys):
    """PRE-REGISTERED acceptance (M5): run A = 2 days straight with a
    checkpoint each day; run B = a fresh driver loading A's day-1
    checkpoint and advancing the remaining day (the MPAS lane's
    days-this-job convention).  The final native bundles must be
    BITWISE identical on EVERY array: the checkpoint carries the full
    duo bundle (not the column view), the same jitted programs run, and
    the fp64 npz round-trip is exact.  With Kessler the MPAS physics
    package runs every step; with TKE turbulence the package carries
    PROGNOSTIC memory (PhysicsState.tke), so the physics carry restore
    is exercised: every PhysicsState field is compared bitwise too, and
    the carry is first shown to EVOLVE between the checkpoint and the
    end of run A (GLM 2026-09-30: a bundle-only gate is blind to a
    silently re-seeded carry when the physics is stateless).  A
    tolerance here would hide state loss."""
    from legoesm.driver.model_driver import ModelDriver
    dt = 1920.0
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir(), dir_b.mkdir()
    mk = dict(dt=dt, **phys)
    cfg_a = _driver_cfg(dir_a, days=2, **mk)
    cfg_a = cfg_a._replace(output=cfg_a.output._replace(checkpoint_days=1))
    drv_a = ModelDriver(cfg_a, output_dir=dir_a)
    drv_a.setup()
    assert drv_a.run() == "COMPLETED"
    dt = drv_a.config.dycore.dt          # post-CFL-clamp effective dt
    mid = dir_a / "checkpoint_day_0001.npz"
    assert mid.is_file(), sorted(p.name for p in dir_a.iterdir())
    with np.load(mid) as d:
        assert str(d["_schema"]) == "fv3duo_ckpt_v1"
        assert int(d["step"]) == int(d["_step"]) == int(86400.0 / dt)
        # the view rides along for the plotters; the bundle for the restart
        for k in ("u", "T", "p_s", "phis", "state_delp", "state_u",
                  "press_ps", "q_0", "q_2", "omga", "fv3duo_hs6",
                  "fv3duo_tracer_names") + (
                      ("physstate_prng_key",) if phys else ()):
            assert k in d.files, k
        assert [str(n) for n in d["fv3duo_tracer_names"]] == \
            list(drv_a.model.tracer_names)
        ps_mid = {k[len("physstate_"):]: np.asarray(d[k]) for k in d.files
                  if k.startswith("physstate_")
                  and k != "physstate_meta_conv_scheme"}

    cfg_b = _driver_cfg(dir_b, days=1, **mk)
    drv_b = ModelDriver(cfg_b, output_dir=dir_b)
    drv_b.setup()
    step, day = drv_b.load_checkpoint(mid)
    assert step == int(86400.0 / dt) and day == pytest.approx(1.0)
    assert drv_b.run(start_step=step, start_day=day) == "COMPLETED"

    got, ref = _walk_bundle(drv_b.state.native), _walk_bundle(drv_a.state.native)
    assert set(got) == set(ref)
    diffs = [k for k in ref if not _same_bytes(got[k], ref[k])]
    assert not diffs, f"column-lane restart is NOT bitwise: {diffs}"
    # and the chain actually moved past the checkpoint (finite both sides)
    with np.load(mid) as d:
        pt_mid = np.asarray(d["state_pt"])
        assert np.isfinite(pt_mid).all() and np.isfinite(ref["state.pt"]).all()
        assert pt_mid.tobytes() != ref["state.pt"].tobytes()
    # the physics carry: bitwise A == B on every persisted field (the
    # carry exists only when a physics package runs) ...
    ps_a, ps_b = drv_a._mpas_phys_state, drv_b._mpas_phys_state
    if not phys:
        assert ps_a is None and ps_b is None and not ps_mid
    else:
        assert ps_a is not None and ps_b is not None and ps_mid
        ps_diffs = [k for k in ps_mid
                    if not _same_bytes(getattr(ps_a, k), getattr(ps_b, k))]
        assert not ps_diffs, \
            f"physics carry restart is NOT bitwise: {ps_diffs}"
    # ... and the CONTROL: with prognostic turbulence the carry must have
    # moved between the checkpoint and the end of run A, or the
    # comparison above has no power
    if phys.get("turbulence") == "mynn25":
        qke_a = np.asarray(ps_a.qke)
        assert np.isfinite(qke_a).all()
        assert ps_mid["qke"].tobytes() != qke_a.tobytes(), \
            "PhysicsState.qke did not evolve; the carry gate is vacuous"
        # MUTATION (codex 2026-09-30): the same restart with the carry
        # STRIPPED from the file (the documented opt-in to a fresh seed)
        # must NOT reproduce run A -- otherwise the carry comparison
        # above could pass on a silently re-seeded restart.
        dir_c = tmp_path / "c"
        dir_c.mkdir()
        with np.load(mid) as d:
            kept = {k: d[k] for k in d.files if not k.startswith("physstate_")}
        stripped = dir_c / mid.name
        np.savez(stripped, **kept)
        drv_c = ModelDriver(_driver_cfg(dir_c, days=1, **mk), output_dir=dir_c)
        drv_c.setup()
        step_c, day_c = drv_c.load_checkpoint(stripped)
        assert drv_c.run(start_step=step_c, start_day=day_c) == "COMPLETED"
        assert not _same_bytes(drv_c._mpas_phys_state.qke, qke_a), \
            "a carry-stripped restart reproduced run A: the carry gate has no power"
        assert not _same_bytes(drv_c.state.native["state"]["pt"],
                               ref["state.pt"])


def test_driver_column_lane_refuses_a_plain_mpas_checkpoint(tmp_path):
    """A checkpoint without the duo bundle (an MPAS-lane file, or a
    column-lane file with the bundle stripped) cannot restart the column
    lane: the view cannot rebuild the D-grid state.  Refused by name."""
    from legoesm.driver.model_driver import ModelDriver
    cfg = _driver_cfg(tmp_path, days=0.25)
    drv = ModelDriver(cfg, output_dir=tmp_path)
    drv.setup()
    bad = tmp_path / "checkpoint_day_0000.npz"
    np.savez(bad, u=np.zeros(1), T=np.zeros(1), p_s=np.zeros(1),
             phis=np.zeros(1), step=np.asarray(3), day=np.asarray(0.1))
    with pytest.raises(ValueError, match="carries no fv3_duo bundle"):
        drv.load_checkpoint(bad)


def test_driver_column_lane_restart_refuses_another_terrain(tmp_path):
    """Decision C: the terrain lives in the GRID; a run on a non-flat terrain
    checkpoints its padded ``hs6`` stack and a restart on the SAME terrain
    continues bitwise, while a fresh driver whose grid carries ANOTHER
    terrain (flat here) is REFUSED at load -- never rebuilt from the file.
    Synthetic bump through the factory path (phis_fn + the del-2 filter),
    so no data file is needed."""
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.grids.factory import create_fv3_duo_grid

    def bump(lon, lat):
        return 500.0 * constants.g * np.exp(
            -((np.asarray(lat) - 0.6) ** 2 + (np.asarray(lon) - 1.0) ** 2)
            / 0.15)

    dir_a, dir_b, dir_c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (dir_a, dir_b, dir_c):
        d.mkdir()
    cfg_a = _driver_cfg(dir_a, days=1, microphysics="kessler")
    cfg_a = cfg_a._replace(output=cfg_a.output._replace(checkpoint_days=0.5))
    drv_a = ModelDriver(cfg_a, output_dir=dir_a)
    drv_a.setup()
    drv_a._fv3_duo_column_rewrap(create_fv3_duo_grid(
        N, NG, phis_fn=bump, phis_filter_iter=2))
    hs6_a = drv_a._fv3_duo_column_hs6(drv_a.model)
    assert float(np.abs(hs6_a).max()) > 1.0e3        # not flat (>100 m)
    assert drv_a.run() == "COMPLETED"
    mid = dir_a / "checkpoint_day_0000.npz"        # day 0.5 rounds to 0
    assert mid.is_file()
    with np.load(mid) as d:
        assert _same_bytes(d["fv3duo_hs6"], hs6_a)

    # the same terrain: bitwise continuation
    drv_b = ModelDriver(_driver_cfg(dir_b, days=0.5, microphysics="kessler"),
                        output_dir=dir_b)
    drv_b.setup()
    drv_b._fv3_duo_column_rewrap(create_fv3_duo_grid(
        N, NG, phis_fn=bump, phis_filter_iter=2))
    step, day = drv_b.load_checkpoint(mid)
    assert _same_bytes(drv_b._fv3_duo_column_hs6(drv_b.model), hs6_a)
    assert drv_b.run(start_step=step, start_day=day) == "COMPLETED"
    got, ref = _walk_bundle(drv_b.state.native), _walk_bundle(drv_a.state.native)
    diffs = [k for k in ref if not _same_bytes(got[k], ref[k])]
    assert not diffs, f"terrain restart is NOT bitwise: {diffs}"

    # another terrain (the factory's flat grid): refused
    drv_c = ModelDriver(_driver_cfg(dir_c, days=0.5, microphysics="kessler"),
                        output_dir=dir_c)
    drv_c.setup()
    assert float(np.abs(drv_c._fv3_duo_column_hs6(drv_c.model)).max()) == 0.0
    with pytest.raises(RuntimeError, match="terrain differs"):
        drv_c.load_checkpoint(mid)


# ---------------------------------------------------------------------
# nwat = 6: ice / snow / graupel + number passengers through the contract
# ---------------------------------------------------------------------

FULL_NAMES = ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g", "N_c", "N_r", "N_i")


@pytest.mark.parametrize("names,ok", [
    (FULL_NAMES, True),
    (("q_v", "q_c", "q_r"), True),
    (("q_v", "q_c", "q_r", "q_p"), True),              # passenger after 3
    (("q_v", "q_c", "q_r", "q_i"), False),             # nwat = 4
    (("q_v", "q_c", "q_r", "N_c", "q_i"), False),      # water after passenger
    (("q_v", "q_r", "q_c"), False),                    # out of FV3 order
    (("q_v", "q_c", "q_r", "q_i", "q_s", "q_g", "q_i"), False),   # dup
])
def test_column_model_tracer_slot_contract(grid, names, ok):
    """The nwat block reads the leading slots by POSITION: the names must
    be the FV3 water species in FV3 order (3 or 6 of them), passengers
    after; anything else would enter the mass sum as the wrong species
    or fall out of it silently."""
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import (
        FV3DuoColumnModel)
    dyn, _ = _model(grid, moist=True)
    if ok:
        col = FV3DuoColumnModel(dyn, tracer_names=names)
        assert col.nwat == (6 if "q_g" in names else 3)
    else:
        with pytest.raises(ValueError, match="water species"):
            FV3DuoColumnModel(dyn, tracer_names=names)


def _ice_ic(dyn):
    """Nine slots: the moist IC's vapour, cloud and rain, seeded ice /
    snow / graupel and number concentrations (positive so the sinks
    below keep every species non-negative)."""
    ic = _saturated_rainy(dyn.dcmip16_initial_state(n_tracers=3))
    q = list(ic["q"][:3])
    q += [jnp.full_like(q[0], v) for v in (1e-3, 2e-3, 5e-4)]
    q += [jnp.full_like(q[0], v) for v in (1e8, 1e5, 1e4)]
    return {**ic, "q": q}


def test_ice_tendencies_move_the_layer_mass_through_the_nwat6_block(grid):
    """A synthetic scheme that sublimates snow into vapour (mass-neutral),
    sediments graupel out of every layer (a mass SINK) and nucleates ice
    while adjusting N_i: the column's dry air is invariant, its total
    water changes by exactly the six-species tendency integral, the
    passengers take their tendency and the renormalisation, and a
    warm-rain accounting (the three leading species only) would NOT
    close -- the defect the nwat=6 block removes."""
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import (
        FV3DuoColumnModel)
    dyn, _ = _model(grid, moist=True)
    col = FV3DuoColumnModel(dyn, tracer_names=FULL_NAMES)
    ic = _ice_ic(dyn)
    dq = {"q_v": 2e-7, "q_s": -2e-7, "q_g": -3e-7, "q_i": 1e-7, "N_i": 5.0}

    def physics(state, mesh, coord, phys_state=None, forcing=None):
        z = jnp.zeros_like(state.u.data)
        return _tend(state, z, z, jnp.zeros_like(state.T.data), tracers={
            nm: _fld(state.tracers[nm], jnp.full_like(state.T.data, v), nm)
            for nm, v in dq.items()})

    st0 = col.from_bundle(ic)
    out = col.step(st0, DT, physics_fn=physics)
    # the dynamics-only step is the reference the physics increment sits on
    post = dyn.step(ic, DT)
    delp0 = np.asarray(post["state"]["delp"])[:, CI, CI]
    q0 = [np.asarray(a)[:, CI, CI] for a in post["q"]]
    delp1 = np.asarray(out.native["state"]["delp"])[:, CI, CI]
    q1 = [np.asarray(a)[:, CI, CI] for a in out.native["q"]]
    w0, w1 = sum(q0[:6]), sum(q1[:6])
    np.testing.assert_allclose(delp1 * (1.0 - w1), delp0 * (1.0 - w0),
                               rtol=1e-12, atol=0)
    dsum = sum(v for k, v in dq.items() if k in FULL_NAMES[:6])
    np.testing.assert_allclose(delp1 * w1 - delp0 * w0, delp0 * DT * dsum,
                               rtol=1e-10, atol=0)
    assert dsum < 0.0                                   # a net sink
    assert np.abs(delp1 - delp0).max() > 0.0
    # warm-rain accounting would see only +2e-7 (vapour): NOT closed
    d3 = sum(v for k, v in dq.items() if k in FULL_NAMES[:3])
    assert not np.allclose(delp1 * w1 - delp0 * w0, delp0 * DT * d3,
                           rtol=1e-6, atol=0)
    # passengers: updated and renormalised on the new layer mass
    ps_dt = delp1 / delp0
    np.testing.assert_allclose(q1[8], (q0[8] + DT * dq["N_i"]) / ps_dt,
                               rtol=1e-13, atol=0)
    np.testing.assert_allclose(q1[6], q0[6] / ps_dt, rtol=1e-13, atol=0)
    # the view carries the nine names, each on its native slot (a jitted
    # dict comes back with sorted keys, so compare by NAME, not order)
    assert set(out.tracers) == set(FULL_NAMES)
    for i, nm in enumerate(FULL_NAMES):
        np.testing.assert_array_equal(
            np.asarray(out.tracers[nm].data).reshape(6, N, N, KM),
            np.asarray(out.native["q"][i])[:, CI, CI], err_msg=nm)
    assert out.tracers["N_i"].units == "1/kg"
    assert out.tracers["q_g"].units == "kg/kg"
    assert all(np.isfinite(np.asarray(a)).all() for a in out.native["q"])


def test_driver_column_lane_accepts_ice_microphysics_with_nine_slots(tmp_path):
    """The factory builds the column model on the driver's own tracer
    registry: an ice scheme gets the six water species + numbers, and
    the run starts with one slot per name (no silent 3-slot bundle)."""
    from legoesm.driver.model_driver import ModelDriver
    cfg = _driver_cfg(tmp_path, days=2 * 600.0 / 86400.0, dt=600.0,
                      microphysics="morrison")
    drv = ModelDriver(cfg, output_dir=tmp_path)
    drv.setup()
    assert drv.model.tracer_names == FULL_NAMES and drv.model.nwat == 6
    assert drv.run() == "COMPLETED"
    assert len(drv.state.native["q"]) == 9
    assert all(np.isfinite(np.asarray(a)).all() for a in drv.state.native["q"])
    assert set(drv.state.tracers) == set(FULL_NAMES)


# ---------------------------------------------------------------------
# M6: the forcings are regridded onto the duo's OWN columns at setup
# ---------------------------------------------------------------------

def _write_sst_file(path):
    """A small monthly SST/SIC file whose SST varies in lat AND lon, so a
    field sampled at the wrong cell centres is measurably wrong."""
    import xarray as xr
    nlat, nlon, nt = 36, 72, 12
    lat = np.linspace(-89, 89, nlat)
    lon = np.linspace(0, 357.5, nlon)
    la, lo = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
    tos = (285.0 + 15.0 * np.cos(la) * np.cos(lo)
           + 5.0 * np.sin(2 * la) * np.sin(lo)) * np.ones((nt, 1, 1))
    sic = np.zeros((nt, nlat, nlon))
    ds = xr.Dataset({"tosbcs": (("time", "lat", "lon"), tos),
                     "siconcbcs": (("time", "lat", "lon"), sic)},
                    coords={"time": np.arange(nt, dtype=float),
                            "lat": lat, "lon": lon})
    ds["tosbcs"].attrs["units"] = "K"
    ds["siconcbcs"].attrs["units"] = "%"
    ds.to_netcdf(path)


def _sst_analytic(lat, lon):
    return (285.0 + 15.0 * np.cos(lat) * np.cos(lon)
            + 5.0 * np.sin(2 * lat) * np.sin(lon))


def _elev_analytic(lat, lon):
    """Elevation [m]: a continent (positive) around (30N, 100E), ocean
    (negative) elsewhere -- lat AND lon dependent."""
    return 2000.0 * np.exp(-((lat - 0.52) ** 2 + (lon - 1.75) ** 2) / 0.3) - 300.0


def _write_elevation_file(path):
    import xarray as xr
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(0.0, 360.0, 2.0)
    la, lo = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
    z = _elev_analytic(la, lo) * np.ones((lat.size, lon.size))
    xr.Dataset({"z": (("lat", "lon"), z)},
               coords={"lat": lat, "lon": lon}).to_netcdf(path)


def test_m6_setup_forcings_land_on_the_duo_columns(tmp_path, monkeypatch):
    """The driver's grid IS the column mesh from grid creation on
    (per-cell, like the Voronoi mesh), so the setup-time regrids -- an
    elevation file, its derived land fraction, a custom SST file -- are
    sampled at the duo's A-grid centres.  The gate: each field matches
    its analytic value at the DUO's cell positions, and the same regrid
    on the standard cubed sphere's centres (the pre-M6 placement,
    MEASURED 1.6 deg off) does NOT.  Decision C: the dynamics terrain IS
    the elevation file's product (cell-mean, masked del-2), the same
    field as the land fraction's source and the CMOR orog.  The run
    then completes through gray radiation, and the driver's grid /
    vertical coordinate / state are the model's own objects."""
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import FV3DuoColumnState
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.grids.factory import create_grid
    from tests.atmosphere.hydrostatic.unit.test_fv3_duo_era5_orography import (
        _synthetic_era5)
    import legoesm.training.era5_to_state as e2s
    era5 = _synthetic_era5(1500.0)
    monkeypatch.setattr(e2s, "load_era5_ic", lambda path, year, **kw: era5)
    sst_path, elev_path = tmp_path / "sst.nc", tmp_path / "elev.nc"
    _write_sst_file(sst_path)
    _write_elevation_file(elev_path)
    cfg = _driver_cfg(
        tmp_path, days=2 * 600.0 / 86400.0, dt=600.0,
        ic="era5", ic_path="synthetic", topo_smoothing=2,
        topography=str(elev_path), dataset="custom",
        forcing_path=str(sst_path), sst_var="tosbcs", sic_var="siconcbcs",
        sst_offset=0.0, sic_scale=0.01, radiation="gray")
    drv = ModelDriver(cfg, output_dir=tmp_path)
    drv.setup()
    mesh = drv.model.mesh
    assert drv.grid is mesh and drv.sigma is drv.model.sigma_coord
    assert mesh.grid_shape_2d == (mesh.nCells,) == (6 * N * N,)
    lat, lon = np.asarray(mesh.latCell), np.asarray(mesh.lonCell)
    cube = create_grid("cubed_sphere", N)
    clat, clon = np.asarray(cube.lat).reshape(-1), np.asarray(cube.lon).reshape(-1)

    # elevation file -> _phis_data at the duo columns (bilinear from a
    # 2 deg file of a smooth field, ocean clipped to zero by the loader
    # default, no smoothing on a cell list; the cube placement is much
    # worse)
    phis = np.asarray(drv._phis_data).reshape(-1)
    assert phis.shape == (mesh.nCells,)
    # decision C: _phis_data IS the dynamics terrain -- the grid's filtered
    # padded stack on the compute window, bitwise
    ng, n = drv.model.ng, drv.model.n
    hs6 = drv._fv3_duo_column_hs6(drv.model)
    assert np.array_equal(phis, hs6[:, ng:ng + n, ng:ng + n].reshape(-1))
    assert float(np.abs(hs6).max()) > 0.0
    # cell-mean of the clipped analytic elevation, 4 masked del-2 passes:
    # within 10 % of the point value at the duo columns (a smooth field),
    # the cube-centre placement is worse
    want = constants.g * np.maximum(_elev_analytic(lat, lon), 0.0)
    err_duo = np.abs(phis - want).max() / (constants.g * 2000.0)
    err_cube = np.abs(phis - constants.g * np.maximum(
        _elev_analytic(clat, clon), 0.0)).max() / (constants.g * 2000.0)
    assert err_duo < 0.1, err_duo
    assert err_cube > err_duo, (err_duo, err_cube)
    assert np.all(phis[np.asarray(drv._f_land).reshape(-1) == 0.0] == 0.0)
    # ... and the land fraction it derives: the continent sits where the
    # analytic elevation is positive AT THE DUO COLUMNS
    f_land = np.asarray(drv._f_land).reshape(-1)
    assert f_land.shape == (mesh.nCells,) and 0.0 < f_land.mean() < 1.0
    land_duo = (f_land > 0.5) == (_elev_analytic(lat, lon) > 0.0)
    land_cube = (f_land > 0.5) == (_elev_analytic(clat, clon) > 0.0)
    assert land_duo.mean() > 0.97, land_duo.mean()
    assert land_cube.mean() < land_duo.mean()

    # SST: the file's analytic field sampled at the duo columns.  Bilinear
    # from a 5x5 deg file of a smooth field: ~0.03 K; a 1.6 deg placement
    # offset is ~0.4 K (15 K/rad * 0.028 rad), so 0.1 K catches the pre-M6
    # placement (GLM: 0.5 K would have let it pass)
    sst = np.asarray(drv._forcing.sst[0]).reshape(-1)
    assert sst.shape == (mesh.nCells,)
    err_duo = np.abs(sst - _sst_analytic(lat, lon)).max()
    err_cube = np.abs(sst - _sst_analytic(clat, clon)).max()
    assert err_duo < 0.1, err_duo
    assert err_cube > 4.0 * err_duo, (err_duo, err_cube)

    # dynamics terrain = the file's product (NOT the ERA5 IC's 1500 m
    # mountain): the state's phis is _phis_data bitwise
    dyn_phis = np.asarray(drv.state.phis.data).reshape(-1)
    assert np.array_equal(dyn_phis, phis)
    assert np.abs(dyn_phis - constants.g * np.maximum(
        _elev_analytic(lat, lon), 0.0)).max() < 0.1 * constants.g * 2000.0
    assert np.abs(hs6).max() == dyn_phis.max()

    assert drv.run() == "COMPLETED"
    assert isinstance(drv.state, FV3DuoColumnState)
    assert np.isfinite(np.asarray(drv.state.native["state"]["pt"])).all()


def _fake_surface_map_by_latitude(path, lat_deg, lon_deg):
    """A CLM map whose plant type is boreal needleleaf (PFT 2) exactly on
    the columns north of 60N and bare soil elsewhere -- at the latitudes
    the LOADER was handed, so the PFT-weighted land parameters (root
    depth) come back in the duo's column order or not at all."""
    n = int(np.asarray(lat_deg).size)
    north = np.asarray(lat_deg) > 60.0
    pft = np.zeros((n, 17)); pft[~north, 0] = 1.0; pft[north, 2] = 1.0
    o = np.ones(n)
    return dict(
        pft_fractions=jnp.asarray(pft),
        theta_wp=jnp.asarray(0.12 * o), theta_fc=jnp.asarray(0.30 * o),
        glacier_frac=jnp.asarray(np.zeros(n)),
        pct_sand=jnp.asarray(40.0 * o), pct_clay=jnp.asarray(20.0 * o),
        theta_r=jnp.asarray(0.05 * o), theta_sat=jnp.asarray(0.45 * o),
        alpha_vg=jnp.asarray(2.0 * o), n_vg=jnp.asarray(1.4 * o),
        K_sat=jnp.asarray(1.0e-5 * o),
    )


def test_m6_multilayer_land_and_rrtmgp_run_on_the_duo_columns(tmp_path, monkeypatch):
    """The two lifted refusals the placement test does not cover: the
    multilayer land model builds its columns on the duo mesh (the
    boreal-forest cells the synthetic map puts north of 60N are exactly
    the columns with latCell > 60N -- loader latitudes in duo column
    order, read back through the PFT-weighted root depth; soil state
    (nCells, n_layers), advancing) and RRTMGP radiation runs
    on it (daily-mean surface SW at the columns finite, latitude-
    structured)."""
    import legoesm.grids.topography as topo
    import legoesm.land.clm_surface_map as clm
    from legoesm.driver.model_driver import ModelDriver
    monkeypatch.setattr(clm, "download_clm_surfdata", lambda *a, **k: "synthetic")
    monkeypatch.setattr(clm, "load_clm_surface", _fake_surface_map_by_latitude)
    monkeypatch.setattr(topo, "load_land_fraction",
                        lambda grid, path, *a, **k: jnp.full(grid.lat.shape, 0.5))
    cfg = _driver_cfg(
        tmp_path, days=3 * 600.0 / 86400.0, dt=600.0,
        radiation="rrtmgp", rad_update_steps=1,
        land_mask_path="synthetic.nc", use_multilayer_land=True,
        multilayer_n_layers=6, multilayer_soil_depth=2.5,
        # the canopy schemes refuse to start without the per-PFT surfdata
        # file; the soil column + placement is what this test is about
        land_surface_scheme="simple_seb")
    drv = ModelDriver(cfg, output_dir=tmp_path)
    drv.setup()
    mesh = drv.model.mesh
    lat_deg = np.degrees(np.asarray(mesh.latCell))
    st0 = drv._land_ml_state
    assert st0 is not None and st0.theta_soil.shape == (mesh.nCells, 6)
    rd = np.asarray(drv.physics.land_ml_params.root_depth).reshape(-1)
    north = lat_deg > 60.0
    assert north.any() and (~north).any()
    assert np.unique(rd[north]).size == 1 and np.unique(rd[~north]).size == 1
    assert rd[north][0] != rd[~north][0], (rd[north][0], rd[~north][0])
    assert drv.run() == "COMPLETED"
    st1 = drv._land_ml_state
    assert np.isfinite(np.asarray(st1.T_soil)).all()
    assert not np.array_equal(np.asarray(st1.T_soil), np.asarray(st0.T_soil))
    sfc = drv.model._sfc_diag
    assert sfc is not None
    from legoesm.core.state import MPAS_SFC_DIAG_BASE_KEYS, MPAS_SFC_DIAG_EXTRA_KEYS
    keys = MPAS_SFC_DIAG_BASE_KEYS + MPAS_SFC_DIAG_EXTRA_KEYS
    f = sfc[keys.index("sw_down_sfc")]
    sw = np.asarray(getattr(f, "data", f)).reshape(-1)
    assert sw.shape == (mesh.nCells,) and np.isfinite(sw).all()
    assert sw.min() >= 0.0 and sw.max() > 100.0
    # daily-mean insolation (no diurnal cycle on this deck): a strong
    # latitude structure on the duo columns, not a uniform value
    corr = np.corrcoef(sw, np.cos(np.asarray(mesh.latCell)))[0, 1]
    assert corr > 0.5, corr


def test_m6_the_model_is_built_on_the_grid_the_forcings_saw(tmp_path):
    """The factory refuses to build its own grid: without the bundle the
    driver built at grid creation, the model's mesh could differ from
    the one the forcings were regridded onto."""
    from legoesm.driver.component_factory import create_atmosphere_dycore
    cfg = _driver_cfg(tmp_path)
    with pytest.raises(ValueError, match="did not build the duo grid"):
        create_atmosphere_dycore(cfg, None, None)


# ---------------------------------------------------------------------
# B3 (2026-10-01): the MPAS lane's post-physics positivity stage
# ---------------------------------------------------------------------

def _plant(ic, where, value):
    qc = jnp.asarray(ic["q"][1]).at[where].set(value)
    return {**ic, "q": [ic["q"][0], qc] + list(ic["q"][2:])}


def _window_mass(col, bundle, i):
    area = np.asarray(col.mesh.areaCell).reshape(6, N, N)[..., None]
    return float((np.asarray(bundle["q"][i])[:, CI, CI]
                  * np.asarray(bundle["state"]["delp"])[:, CI, CI] * area).sum())


def test_column_positivity_stage_is_the_mpas_borrow(moist):
    """A negative planted in q_c survives the dynamics step (the raw
    dycore step is the no-stage control: zero physics tendencies leave
    the increment block an identity); the column model then applies the
    MPAS lane's own end-of-step stage (apply_water_positivity,
    conservative borrow weighted by delp*area): the window comes out
    non-negative and the global tracer MASS is the raw step's to
    roundoff; non-tracer fields are untouched."""
    dyn, col, ic = moist
    ic2 = _plant(ic, (2, NG + 4, NG + 5, 2), -2.0e-3)
    raw = dyn.step(ic2, DT)
    out = col.step(col.from_bundle(ic2), DT, physics_fn=_zero_physics).native
    qc_raw = np.asarray(raw["q"][1])[:, CI, CI]
    qc_on = np.asarray(out["q"][1])[:, CI, CI]
    assert qc_raw.min() < 0.0                     # the dycore left it
    assert qc_on.min() >= 0.0
    for i in range(len(col.tracer_names)):
        assert np.isclose(_window_mass(col, out, i), _window_mass(col, raw, i),
                          rtol=1e-12), i
    # state and pressure windows bitwise the raw step's (the stage touches
    # tracers only); halos are not part of the contract
    _assert_bundle_equal({**out, "q": raw["q"]}, raw)


def test_column_positivity_global_residual_is_area_weighted(moist):
    """A column made WHOLLY negative in q_c (nothing to borrow locally):
    the shared stage floors it and takes the invented mass back from
    every positive cell in proportion to its MASS (delp*area).  The duo
    cells differ 1.4x corner to centre, so the delp-only weight MPAS
    passes on its quasi-uniform mesh would mis-conserve here: the
    area-weighted global q_c mass is kept to 1e-12 by the model, and the
    same routine fed delp alone (the control) does NOT keep it."""
    from legoesm.core.conservation import apply_water_positivity
    dyn, col, ic = moist
    ic2 = _plant(ic, (0, NG + 1, NG + 1, slice(None)), -4.0e-3)   # corner column
    raw = dyn.step(ic2, DT)
    out = col.step(col.from_bundle(ic2), DT, physics_fn=_zero_physics).native
    delp = np.asarray(raw["state"]["delp"])[:, CI, CI]
    area = np.asarray(col.mesh.areaCell).reshape(6, N, N)[..., None]
    q_raw = np.asarray(raw["q"][1])[:, CI, CI]
    q_on = np.asarray(out["q"][1])[:, CI, CI]
    assert (q_raw[0, 1, 1, :] < 0.0).all()         # wholly negative column
    assert q_on.min() >= 0.0
    m = lambda q: float((q * delp * area).sum())   # noqa: E731
    assert np.isclose(m(q_on), m(q_raw), rtol=1e-12)
    fixed_dp, _ = apply_water_positivity(
        {"q_c": jnp.asarray(q_raw)}, None, jnp.asarray(delp),
        conservative=True, energy_consistent=False)
    assert not np.isclose(m(np.asarray(fixed_dp["q_c"])), m(q_raw), rtol=1e-12)


def test_column_positivity_knobs_follow_the_mpas_semantics(moist):
    """conservative_tracer_clamp=False is the MPAS hard floor (max(q, 0):
    negatives deleted, mass CREATED), not 'no stage'; the floor's
    latent-heat T correction (energy_consistent_moisture_clip) is refused
    on this lane rather than silently dropped."""
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import (
        FV3DuoColumnModel)
    dyn, col_on, ic = moist
    col_floor = FV3DuoColumnModel(dyn, tracer_names=col_on.tracer_names,
                                  conservative_tracer_clamp=False)
    ic2 = _plant(ic, (2, NG + 4, NG + 5, 2), -2.0e-3)
    out = col_floor.step(col_floor.from_bundle(ic2), DT, physics_fn=_zero_physics).native
    assert np.asarray(out["q"][1])[:, CI, CI].min() >= 0.0       # floored ...
    raw = dyn.step(ic2, DT)
    assert _window_mass(col_on, out, 1) > _window_mass(col_on, raw, 1)  # ... creating mass
    with pytest.raises(NotImplementedError, match="latent-heat T correction"):
        FV3DuoColumnModel(dyn, tracer_names=col_on.tracer_names,
                          conservative_tracer_clamp=False,
                          energy_consistent_moisture_clip=True)


def test_column_positivity_runs_on_the_dynamics_only_path_too(moist):
    """As on MPAS the stage runs EVERY step: a planted negative is
    repaired by a step with no physics function."""
    dyn, col, ic = moist
    ic2 = _plant(ic, (2, NG + 4, NG + 5, 2), -2.0e-3)
    out = col.step(col.from_bundle(ic2), DT)
    assert np.asarray(out.native["q"][1])[:, CI, CI].min() >= 0.0
    raw = dyn.step(ic2, DT)
    assert np.asarray(raw["q"][1])[:, CI, CI].min() < 0.0   # the dycore alone did not


def test_restart_terrain_check_covers_spectral_phis_hat():
    """Decision C: a spectral checkpoint carries ``phis_hat``; the restart
    check synthesises it and refuses a planted mismatch, accepts the
    product's own coefficients."""
    from types import SimpleNamespace
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis, sh_synthesis
    grid = create_gaussian_grid(10)
    lat = np.asarray(grid.grid_lat)
    phis = jnp.asarray(2000.0 * np.exp(-((lat - 0.5) ** 2) / 0.1))
    product = sh_synthesis(grid, sh_analysis(grid, phis))
    good = SimpleNamespace(_phis_data=np.asarray(product), grid=grid,
                           state=SimpleNamespace(phis=None, phis_hat=SimpleNamespace(
                               data=sh_analysis(grid, phis))))
    ModelDriver._check_restart_terrain(good)
    bad = SimpleNamespace(_phis_data=np.asarray(product), grid=grid,
                          state=SimpleNamespace(phis=None, phis_hat=SimpleNamespace(
                              data=sh_analysis(grid, 0.5 * phis))))
    with pytest.raises(RuntimeError, match="terrain differs"):
        ModelDriver._check_restart_terrain(bad)


def test_column_lane_carries_the_subgrid_orography_through_the_grid_rebuild(tmp_path):
    """The driver attaches the per-column SSO stddev to the grid at
    _create_topography; the column lane then REBUILDS its mesh (terrain in
    the grid) in _init_state, BEFORE _create_physics reads the field.  The
    rebuilt mesh must carry it: the orographic GWD gets the 800 m box, not
    its scalar fallback (the test fails if the rebuild drops the field)."""
    import xarray as xr
    from legoesm.driver.model_driver import ModelDriver
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(1.0, 360.0, 2.0)
    box = ((lat[:, None] >= 25) & (lat[:, None] <= 45)
           & (lon[None, :] >= 70) & (lon[None, :] <= 100))
    path = tmp_path / "sso_stdh_2deg.nc"
    xr.Dataset({"SSO_STDH": (("lat", "lon"), np.where(box, 800.0, 0.0))},
               coords={"lat": lat, "lon": lon}).to_netcdf(path)
    cfg = _driver_cfg(tmp_path, gravity_wave_drag="mcfarlane",
                      subgrid_orography_path=str(path))
    drv = ModelDriver(cfg, output_dir=tmp_path)
    drv.setup()
    assert drv.grid is drv.model.mesh          # the lane's identity contract
    sso_grid = drv.grid.subgrid_topo_stddev
    sso_phys = drv.physics.subgrid_topo_stddev
    assert sso_grid is not None and sso_phys is not None
    assert np.asarray(sso_phys).shape == (drv.grid.nCells,)
    assert np.array_equal(np.asarray(sso_grid), np.asarray(sso_phys))
    assert 700.0 < float(np.asarray(sso_phys).max()) <= 800.0
    assert float(np.asarray(sso_phys).min()) == 0.0
