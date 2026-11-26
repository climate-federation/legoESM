"""Cube (FV3 duo) matrix rows built from the SHARED cube IC builders.

``scripts/matrix/_cube_duo_adapters.shared_ic_bundle`` evaluates the same
``*_init`` the other grids' dispatch names on the duo A-grid cells and the
run's CAM6 L32 coordinate.  These tests pin, per row, that the duo bundle
carries that builder's fields at known points (temperature, hydrostatic
surface pressure over the grid's own terrain, the geographic winds after the
D-grid projection) and that the grid carries the row's terrain and rotation
rate -- plus the refusal when a recipe's terrain disagrees with its builder.
No stepping (the step compile is the matrix run's job).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402

N = 8

A = pytest.importorskip("scripts.matrix._cube_duo_adapters")


def _model(case, ic_kwargs=None):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid
    spec = A.duo_shared_ic_spec(case, ic_kwargs)
    kw = {}
    if spec.phis_fn is not None:
        kw["phis_fn"] = spec.phis_fn
    if spec.omega is not None:
        kw["omega"] = spec.omega
    if spec.radius is not None:
        kw["radius"] = spec.radius
    grid = create_fv3_duo_grid(N, **kw)
    km, eta = ((A.DUO_GW31_KM, A.DUO_GW31_ETA) if spec.eta == A.DUO_GW31_ETA
               else (A.DUO_HYDRO_KM, A.DUO_HYDRO_ETA))
    model = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=km, eta=eta))
    return spec, model


def _fields(model, bundle):
    ctx = model.grid.ctx_np
    sl = slice(ctx["ng"], ctx["ng"] + ctx["n"])
    lon, lat, _ = A.duo_cell_geometry(ctx)
    pt = np.asarray(bundle["state"]["pt"])[:, sl, sl, :]
    ps = np.asarray(bundle["press"]["ps"])[:, sl, sl]
    hs6 = ctx.get("hs6")
    phis = (np.zeros_like(ps) if hs6 is None
            else np.stack([np.asarray(h)[sl, sl] for h in hs6]))
    ue, vn = A._GeoWinds(ctx)(bundle["state"]["u"], bundle["state"]["v"])
    f0 = np.stack([np.asarray(g["f0"])[sl, sl] for g in ctx["gs6"]])
    return SimpleNamespace(lon=np.radians(lon), lat=np.radians(lat), pt=pt,
                           ps=ps, phis=phis, ue=ue, vn=vn, f0=f0)


def _rel_rms(got, ref):
    r = float(np.sqrt(np.mean((got - ref) ** 2) / np.mean(ref ** 2)))
    print(f"rel RMS {r:.4f}")
    return r


@pytest.mark.parametrize("case, kw", [("rest_state_topo", {"h_0": 2000.0}),
                                      ("held_suarez_topo", {"h_0": 2000.0})])
def test_rest_states_over_the_2_0_0_mountain(case, kw):
    spec, model = _model(case, kw)
    f = _fields(model, A.shared_ic_bundle(model, spec))
    # the grid carries the DCMIP 2012 2-0-0 mountain (non-vacuous: high)
    from legoesm.atmosphere.idealized.topography import dcmip_2_0_0_mountain
    from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M
    zs = np.asarray(dcmip_2_0_0_mountain(f.lon, f.lat, FV3_RADIUS_M, h_0=2000.0))
    np.testing.assert_allclose(f.phis, constants.g * zs, rtol=1e-12, atol=1e-9)
    assert f.phis.max() > 0.5 * constants.g * 2000.0
    assert np.all(f.ue == 0.0) and np.all(f.vn == 0.0)
    assert np.all(np.isfinite(f.pt))
    from legoesm.grids.fv3_native_gridstruct import FV3_OMEGA
    if case == "rest_state_topo":
        # DCMIP 2012 2-0-0 is non-rotating (Table XII: Omega = 0)
        assert np.all(f.f0 == 0.0)
    else:   # HS keeps the rotating planet
        np.testing.assert_allclose(f.f0, 2 * FV3_OMEGA * np.sin(f.lat),
                                   rtol=1e-6, atol=1e-12)
    if case == "rest_state_topo":
        # DCMIP 2012 2-0-0 as specified: constant lapse rate 6.5 K/km,
        # p_s = p0 (1 - gamma zs / T0)^(g / (Rd gamma)), T(p) on the run's
        # own L32 full levels (the shared lapse_rate_columns)
        gamma = 0.0065                       # DCMIP 2012 2-0-0 lapse rate [K/m]
        ps_ref = constants.p_ref * (1.0 - gamma * zs / 300.0) ** (
            constants.g / (constants.R_d * gamma))
        np.testing.assert_allclose(f.ps, ps_ref, rtol=1e-6)
        # independent T: the model's own interface table ak/bk, full level =
        # arithmetic mid-point of its interfaces, T = T0 (p/p0)^(Rd gamma/g)
        ak, bk = np.asarray(model.ak), np.asarray(model.bk)
        p_full = (0.5 * (ak[:-1] + ak[1:])
                  + 0.5 * (bk[:-1] + bk[1:]) * ps_ref[..., None])
        T_ref = 300.0 * (p_full / constants.p_ref) ** (
            constants.R_d * gamma / constants.g)
        np.testing.assert_allclose(f.pt, T_ref, rtol=1e-6)
        assert f.pt.max() - f.pt.min() > 50.0     # stratified, not isothermal
    else:   # HS: hydrostatic isothermal 300 K over the terrain, 1 K-scale
        # noise on the lowest level only (FV3 k = km-1); rtol 1e-6:
        # held_suarez_init stores at the fp32 policy dtype
        np.testing.assert_allclose(
            f.ps, constants.p_ref * np.exp(-f.phis / (constants.R_d * 300.0)),
            rtol=1e-6)
        np.testing.assert_allclose(f.pt[..., :-1], 300.0, rtol=0, atol=1e-4)
        assert 0.5 < np.abs(f.pt[..., -1] - 300.0).max() < 6.0
    assert f.ps.min() < 0.85 * constants.p_ref


def test_gravity_wave_3_1_dcmip2012_small_planet():
    """DCMIP 2012 test 3-1 on the duo: a = R_earth/125 in every metric,
    Omega = 0, the App. F.3 L10 table, and the document's p_s / T written
    out independently at the cells (Eqs. 89, 93, 92, 96, 105-109)."""
    from legoesm.grids.vertical import dcmip2012_gw31_eta_table
    spec, model = _model("gravity_wave_3_1")
    ctx = model.grid.ctx_np
    a = float(constants.R_earth) / 125.0
    assert ctx["radius"] == a
    _, _, area = A.duo_cell_geometry(ctx)
    # the radius reached the metrics: the cells tile the small sphere
    np.testing.assert_allclose(np.sum(area), 4 * np.pi * a * a, rtol=1e-12)
    ak, bk, ptop = dcmip2012_gw31_eta_table(10)
    assert np.array_equal(model.ak, ak) and np.array_equal(model.bk, bk)
    f = _fields(model, A.shared_ic_bundle(model, spec))
    assert np.all(f.f0 == 0.0)                       # omega = 0 on the grid
    assert np.all(f.phis == 0.0)
    g, cp, rd, kap = constants.g, constants.c_pd, constants.R_d, constants.kappa
    N2, u0, teq, peq = 1e-4, 20.0, 300.0, 1e5
    G = g * g / (N2 * cp)
    c2 = np.cos(2 * f.lat) - 1.0
    Ts = G + (teq - G) * np.exp(-u0 * u0 * N2 / (4 * g * g) * c2)
    ps = peq * np.exp(u0 * u0 / (4 * G * rd) * c2) * (Ts / teq) ** (1 / kap)
    np.testing.assert_allclose(f.ps, ps, rtol=1e-13)
    p = (0.5 * (ak[:-1] + ak[1:]) + 0.5 * (bk[:-1] + bk[1:]) * ps[..., None])
    q = Ts[..., None] / G * ((p / ps[..., None]) ** kap - 1.0) + 1.0
    Tb = Ts[..., None] * (p / ps[..., None]) ** kap / q
    z = -g / N2 * np.log(q)
    r = a * np.arccos(np.clip(np.cos(f.lat) * np.cos(f.lon - 2 * np.pi / 3),
                              -1.0, 1.0))
    s = 25e6 / (25e6 + r * r)
    T = Tb + s[..., None] * np.sin(2 * np.pi * z / 20000.0) * (p / peq) ** kap
    np.testing.assert_allclose(f.pt, T, rtol=1e-12)
    assert (f.pt - Tb).max() > 0.2   # the bubble is there (C8 cells ~30 km)
    # solid-body 20 cos(lat) through the D-grid projection and c2l lens
    # (MEASURED rel RMS 0.0059 at C8 on the Earth-radius grid)
    ref = 20.0 * np.cos(f.lat)[..., None] * np.ones_like(f.ue)
    assert _rel_rms(f.ue, ref) < 0.02
    assert np.sqrt(np.mean(f.vn ** 2)) < 0.02 * np.sqrt(np.mean(ref ** 2))


def test_build_duo_hydro_admits_exactly_the_gw31_configuration(monkeypatch):
    """The C36-only guard admits the 3-1 small planet with ITS table, radius
    and step (the adapter run to completion, no stepping); a control row
    keeps the Earth deck (FV3 radius, dt 1920 s, CAM6 L32)."""
    from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M
    a = float(constants.R_earth) / 125.0
    duo = A.build_duo_hydro(36, shared_case="gravity_wave_3_1")
    assert duo.dt == A.DUO_GW31_DT_ATMOS == 15.0
    assert (duo.km, duo.eta) == (10, "dcmip2012_gw31_l10")
    assert (duo.model.config.km, duo.model.config.eta) == (duo.km, duo.eta)
    assert duo.grid.ctx_np["radius"] == a
    np.testing.assert_allclose(np.sum(duo.area), 4 * np.pi * a * a,
                               rtol=1e-12)
    assert "radius=" in duo.note and "L10" in duo.note
    for g in duo.grid.ctx_np["gs6"]:           # Omega = 0 reached the grid
        assert np.all(np.asarray(g["f0"]) == 0.0)
        assert np.all(np.asarray(g["fC"]) == 0.0)
    ctl = A.build_duo_hydro(36, shared_case="inertio_gravity_3_2")
    assert ctl.dt == A.DUO_HYDRO_DT_ATMOS == 1920.0
    assert (ctl.km, ctl.eta) == (A.DUO_HYDRO_KM, A.DUO_HYDRO_ETA)
    assert ctl.grid.ctx_np["radius"] == FV3_RADIUS_M
    assert "small-planet step" in duo.note
    assert "small-planet step" not in ctl.note
    # the Courant check binds: a step above 1920 s * a / FV3_RADIUS_M
    from tests.test_cases.dcmip2012.gravity_wave_3_1 import check_step
    check_step(600.0 / 39)       # lands on 600 / 3600 s, but 15.38 > 15.36
    monkeypatch.setattr(A, "DUO_GW31_DT_ATMOS", 600.0 / 39)
    with pytest.raises(ValueError, match="geometric step ceiling"):
        A.build_duo_hydro(36, shared_case="gravity_wave_3_1")
    # ... and so does the end-time check: 15.3 s is under the ceiling but
    # 3600 / 15.3 steps would stop at 3595.5 s
    monkeypatch.setattr(A, "DUO_GW31_DT_ATMOS", 15.3)
    with pytest.raises(ValueError, match="does not divide"):
        A.build_duo_hydro(36, shared_case="gravity_wave_3_1")
    # 14.4 s divides 3600 s (250 steps) but not 600 s (misses 3000 s)
    monkeypatch.setattr(A, "DUO_GW31_DT_ATMOS", 14.4)
    with pytest.raises(ValueError, match="does not divide"):
        A.build_duo_hydro(36, shared_case="gravity_wave_3_1")
    with pytest.raises(ValueError, match="goes with that case only"):
        A.build_duo_hydro(36, km=10, eta="dcmip2012_gw31_l10")
    with pytest.raises(ValueError, match="held_suarez forcing goes with"):
        A.build_duo_hydro(36, shared_case="gravity_wave_3_1",
                          held_suarez=True)
    with pytest.raises(ValueError, match="C36 only"):
        A.build_duo_hydro(24, shared_case="gravity_wave_3_1")
    with pytest.raises(ValueError, match="do not pass km/eta"):
        A.build_duo_hydro(36, shared_case="gravity_wave_3_1", km=32,
                          eta="dcmip2012_gw31_l10")


@pytest.mark.parametrize("oracle_conventions", [True, False])
def test_duo_radius_scales_every_length_metric(oracle_conventions):
    """Every length / area metric of the six gridstructs scales with the
    radius (and inverse metrics inversely); angles do not -- on both
    gridstruct builders (bounded = the duo grid factory's lane, and plain)."""
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M
    a = float(constants.R_earth) / 125.0
    r = a / FV3_RADIUS_M
    kw = dict(use_ext_bundle=oracle_conventions,
              oracle_conventions=oracle_conventions)
    n = 12        # the plain kinked-corner lattice needs n >= 12 at ng 3
    e6 = build_six_face_duo_context(n, 3, **kw)["gs6"]
    s6 = build_six_face_duo_context(n, 3, radius=a, **kw)["gs6"]
    assert len(e6) == len(s6) == 6
    for ge, gs in zip(e6, s6):
        for k, p in (("dx", 1), ("dy", 1), ("dxa", 1), ("dya", 1),
                     ("dxc", 1), ("dyc", 1), ("area", 2), ("area_c", 2),
                     ("rdx", -1), ("rdy", -1), ("rdxa", -1), ("rdya", -1),
                     ("rdxc", -1), ("rdyc", -1), ("rarea", -2),
                     ("rarea_c", -2)):
            x, y = np.asarray(ge[k], float), np.asarray(gs[k], float)
            m = np.isfinite(x) & (np.abs(x) > 0) & (np.abs(x) < 1e20)
            ms = np.isfinite(y) & (np.abs(y) > 0) & (np.abs(y) < 1e20)
            assert m.sum() > 0 and np.array_equal(m, ms), k
            np.testing.assert_allclose(y[m], x[m] * r ** p, rtol=1e-11,
                                       err_msg=k)
        for k in ("cosa", "sina"):
            np.testing.assert_array_equal(np.asarray(gs[k]), np.asarray(ge[k]))


def test_inertio_gravity_keeps_the_rotating_planet():
    spec, model = _model("inertio_gravity_3_2")
    f = _fields(model, A.shared_ic_bundle(model, spec))
    from legoesm.grids.fv3_native_gridstruct import FV3_OMEGA
    np.testing.assert_allclose(f.f0, 2 * FV3_OMEGA * np.sin(f.lat),
                               rtol=1e-6, atol=1e-12)
    # DCMIP 2008 [6-3-0]: at rest, p_s = 1000 hPa, 10 K theta bubble
    # centred at (pi, pi/4)
    assert np.all(f.ue == 0.0) and np.all(f.vn == 0.0)
    np.testing.assert_allclose(f.ps, constants.p_ref, rtol=1e-14)
    d = np.abs(f.pt - 300.0).max(-1)
    assert 5.0 < d.max() < 10.0
    i = np.unravel_index(np.argmax(d), d.shape)
    assert abs(np.degrees(f.lon[i]) - 180.0) < 12
    assert abs(np.degrees(f.lat[i]) - 45.0) < 12
    # signed field: T = 300 + 10 s sin(2 pi z/20 km) (p/p0)^kappa (Eqs.
    # 94-98) at the duo cells and the run's own L32 full levels (ak/bk
    # mid-points), z = (R_d T0/g) ln(p0/p) (Eq. 92)
    from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M
    a, p0 = FV3_RADIUS_M, constants.p_ref
    r = a * np.arccos(np.clip(np.sin(np.pi / 4) * np.sin(f.lat)
                              + np.cos(np.pi / 4) * np.cos(f.lat)
                              * np.cos(f.lon - np.pi), -1, 1))
    s = np.where(r < a / 3, 0.5 * (1 + np.cos(3 * np.pi * r / a)), 0.0)
    ak, bk = np.asarray(model.ak), np.asarray(model.bk)
    p = 0.5 * (ak[:-1] + ak[1:]) + 0.5 * (bk[:-1] + bk[1:]) * p0
    z = constants.R_d * 300.0 / constants.g * np.log(p0 / p)
    T_ref = 300.0 + (10.0 * s[..., None] * np.sin(2 * np.pi * z / 20000.0)
                     * (p / p0) ** constants.kappa)
    # MEASURED max rel diff 2.0e-9 (distance evaluated by a different
    # routine); rtol 1e-6 as the other duo T checks
    np.testing.assert_allclose(f.pt, T_ref, rtol=1e-6)


def test_mountain_rossby_terrain_isothermal_288():
    spec, model = _model("mountain_rossby_5_0")
    f = _fields(model, A.shared_ic_bundle(model, spec))
    i = np.unravel_index(np.argmax(f.phis), f.phis.shape)
    # DCMIP 2008 5-0-0 (Eq. 82): the peak cell sits near (90E, 30N)
    # (C8 cells are ~11 deg)
    assert abs(np.degrees(f.lon[i]) - 90.0) < 12 and abs(np.degrees(f.lat[i]) - 30.0) < 12
    assert f.phis.max() > 0.5 * constants.g * 2000.0
    assert np.all(f.pt == 288.0)
    # Eq. (83) with N^2 = g^2/(c_p T0) (Eq. 80), p_sp = 930 hPa, u0 = 20,
    # a and Omega = the duo's own radius and rotation rate
    from legoesm.grids.fv3_native_gridstruct import FV3_OMEGA, FV3_RADIUS_M
    g, a, u0 = constants.g, FV3_RADIUS_M, 20.0
    n2, k = g ** 2 / (constants.c_pd * 288.0), constants.kappa
    ps_ref = 93000.0 * np.exp(
        -(a * n2 * u0 / (2 * g ** 2 * k)) * (u0 / a + 2 * FV3_OMEGA)
        * (np.sin(f.lat) ** 2 - 1) - n2 / (g ** 2 * k) * f.phis)
    np.testing.assert_allclose(f.ps, ps_ref, rtol=1e-10)
    np.testing.assert_allclose(f.f0, 2 * FV3_OMEGA * np.sin(f.lat),
                               rtol=1e-6, atol=1e-12)


def test_rossby_haurwitz_winds():
    spec, model = _model("rossby_haurwitz_6_0")
    bundle = A.shared_ic_bundle(model, spec)
    f = _fields(model, bundle)
    from legoesm.core.williamson_sw_analytic import (
        rossby_haurwitz_4_geopotential, rossby_haurwitz_4_winds)
    from legoesm.grids.fv3_native_gridstruct import FV3_OMEGA, FV3_RADIUS_M
    # DCMIP 2008 Sect. 1.4 (test 4-0-0): M = K = u0/(n a), u0 = 50 m/s
    mk = 50.0 / (4.0 * FV3_RADIUS_M)
    u, v = rossby_haurwitz_4_winds(f.lon, f.lat, radius=FV3_RADIUS_M,
                                   omega_wave=mk, k_wave=mk)
    # D-grid projection + c2l lens at C8 (~4 cells per RH wave-4
    # wavelength): MEASURED rel RMS u 0.027, v 0.079 (amplitude-free);
    # caps ~1.5-2x
    assert _rel_rms(f.ue, np.asarray(u)[..., None] * np.ones_like(f.ue)) < 0.05
    assert _rel_rms(f.vn, np.asarray(v)[..., None] * np.ones_like(f.vn)) < 0.12
    # Eq. (70) p_s and Eq. (66) T on the run's own L32 (ak/bk mid-points),
    # T0 = 288 K, Gamma = 6.5 K/km, p_ref = 955 hPa, Phi_s = 0
    gamma, t0, p_ref = 0.0065, 288.0, 95500.0
    phi = rossby_haurwitz_4_geopotential(
        f.lon, f.lat, radius=FV3_RADIUS_M, omega=FV3_OMEGA, gh0=0.0,
        omega_wave=mk, k_wave=mk)
    ps_ref = p_ref * (1.0 + gamma * phi / (constants.g * t0)) ** (
        constants.g / (gamma * constants.R_d))
    # rtol 1e-10 (fp64 bundle): resolves the duo's FV3_OMEGA vs
    # constants.Omega (~1e-6 relative in p_s)
    np.testing.assert_allclose(f.ps, ps_ref, rtol=1e-10)
    assert f.ps.max() - f.ps.min() > 5000.0           # not uniform
    ak, bk = np.asarray(model.ak), np.asarray(model.bk)
    p_full = (0.5 * (ak[:-1] + ak[1:])
              + 0.5 * (bk[:-1] + bk[1:]) * ps_ref[..., None])
    T_ref = t0 * (p_full / p_ref) ** (constants.R_d * gamma / constants.g)
    np.testing.assert_allclose(f.pt, T_ref, rtol=1e-6)
    # every layer thickness, not just the column sum
    ctx = model.grid.ctx_np
    sl = slice(ctx["ng"], ctx["ng"] + ctx["n"])
    delp = np.asarray(bundle["state"]["delp"])[:, sl, sl, :]
    np.testing.assert_allclose(
        delp, np.diff(ak) + np.diff(bk) * ps_ref[..., None], rtol=1e-6)


def test_dcmip_omega_override(matrix_mod):
    """Non-rotating cases on the non-cube grids: DCMIP 2012 2-0-0 (Table
    XII, Omega = 0) and the gravity wave; everything else keeps Earth's."""
    f = matrix_mod.dcmip_omega_override
    assert f("rest_state_topo") == 0.0
    assert f("gravity_wave_3_1") == 0.0
    for case in ("baroclinic", "held_suarez_topo", "inertio_gravity_3_2",
                 "mountain_rossby_5_0", "rossby_haurwitz_6_0"):
        assert f(case) is None


@pytest.mark.parametrize("grid_type, res, module, factory", [
    ("latlon", "8x16", "legoesm.grids.latlon", "create_latlon_grid"),
    ("icosahedral", "ico2", "legoesm.grids.voronoi", "create_voronoi_mesh"),
    ("spectral", "T10", "legoesm.grids.gaussian", "create_gaussian_grid"),
])
@pytest.mark.parametrize("case, kw, omega", [
    ("rest_state_topo", {"h_0": 2000.0}, 0.0),
    ("gravity_wave_3_1", {}, 0.0),
    ("mountain_rossby_5_0", {}, None),
])
def test_non_cube_grid_factories_get_the_case_omega(
        matrix_mod, monkeypatch, tmp_path, grid_type, res, module, factory,
        case, kw, omega):
    """run_baroclinic hands each non-cube grid factory the case's rotation
    rate: omega=0.0 for the non-rotating cases, nothing (factory default,
    Earth) otherwise."""
    import importlib
    mod = importlib.import_module(module)
    seen = {}

    def fake(*a, **k):
        seen.update(k)
        raise _Stop

    monkeypatch.setattr(mod, factory, fake)
    tc = matrix_mod.TestCase("hydrostatic", case, grid_type, res, "hybrid",
                             1.0, 0.25, dict(kw))
    with pytest.raises(_Stop):
        matrix_mod.run_baroclinic(tc, tmp_path, 0.25)
    if omega is None:
        assert "omega" not in seen, seen
    else:
        assert seen.get("omega") == omega, seen


@pytest.mark.parametrize("perturbed", [True, False])
def test_baroclinic_duo_uses_the_dcmip2016_truncated_perturbation(
        monkeypatch, perturbed):
    """The matrix cube baroclinic rows build the DCMIP-2016 state with the
    document's Eq. (14) perturbation (truncated at R_p, as the other grids)."""
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoDynamicsModel)
    seen = {}

    def spy(self, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(FV3DuoDynamicsModel, "dcmip16_initial_state", spy)
    with pytest.raises(_Stop):
        A.build_duo_hydro(36, perturbed=perturbed)   # deck dt is C36-only
    assert seen == {"do_pert": perturbed, "use_dcmip2016_perturbation": True}


#: radius exponent of every float field of the solver's gridstruct dicts
#: (DuoStepperContext.gs6); 0 = radius-free (angles, ratios, Coriolis)
_GS_RADIUS_POWER = {
    **{k: 1 for k in ("dx", "dy", "dxa", "dya", "dxc", "dyc")},
    **{k: -1 for k in ("rdx", "rdy", "rdxa", "rdya", "rdxc", "rdyc")},
    "area": 2, "area_c": 2, "rarea": -2, "rarea_c": -2,
    **{k: 0 for k in (
        "agrid_lat", "agrid_lon", "grid_lat", "grid_lon", "cos_sg", "sin_sg",
        "cosa", "cosa_s", "cosa_u", "cosa_v", "sina", "sina_u", "sina_v",
        "rsin2", "rsin_u", "rsin_v", "rsina", "del6_u", "del6_v", "divg_u",
        "divg_v", "edge_e", "edge_n", "edge_s", "edge_w", "f0", "fC")},
}


def _radius_powers(ge, gs, r):
    """{field: p} with gs[field] == ge[field] * r**p on every finite,
    non-sentinel entry (None if no integer p in [-2, 2] fits)."""
    out = {}
    for k in ge:
        x, y = np.asarray(ge[k]), np.asarray(gs[k])
        if x.dtype == object or not np.issubdtype(x.dtype, np.floating):
            continue
        # mask from the Earth array alone: a small-planet entry that went
        # non-finite there is a mismatch, not an excluded point
        m = np.isfinite(x) & (np.abs(x) > 1e-300) & (np.abs(x) < 1e20)
        if not m.any():
            continue
        if not np.all(np.isfinite(y[m])):
            out[k] = None
            continue
        q = y[m] / x[m]
        out[k] = next((p for p in (0, 1, 2, -1, -2)
                       if np.allclose(q, r ** p, rtol=1e-9)), None)
    return out


def test_duo_solver_context_metrics_scale_with_radius():
    """The JAX stepper context the solver consumes (gridstruct arrays frozen
    into DuoStepperContext): every float field carries exactly its expected
    radius exponent on every face -- no Earth-radius length survives."""
    from legoesm.grids.factory import create_fv3_duo_grid
    from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M
    a = float(constants.R_earth) / 125.0
    r = a / FV3_RADIUS_M
    E = create_fv3_duo_grid(N).ctx_jax
    S = create_fv3_duo_grid(N, radius=a).ctx_jax
    assert len(E.gs6) == len(S.gs6) == 6
    for ge, gs in zip(E.gs6, S.gs6):
        assert _radius_powers(ge, gs, r) == _GS_RADIUS_POWER
    # negative control: one length field left at its Earth value is caught
    frozen = {**S.gs6[0], "dx": E.gs6[0]["dx"]}
    assert _radius_powers(E.gs6[0], frozen, r)["dx"] == 0
    # ... and so is one corrupted (NaN) entry
    dx = np.array(S.gs6[0]["dx"], dtype=float)
    i = np.argwhere(np.isfinite(dx) & (np.abs(dx) > 0))[0]
    dx[tuple(i)] = np.nan
    assert _radius_powers(E.gs6[0], {**S.gs6[0], "dx": dx}, r)["dx"] is None


def test_rotating_grid_is_refused_for_a_non_rotating_recipe():
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid
    spec = A.duo_shared_ic_spec("gravity_wave_3_1")
    grid = create_fv3_duo_grid(N, radius=spec.radius)       # Earth's omega
    model = FV3DuoDynamicsModel(
        grid, FV3DuoConfig(km=A.DUO_GW31_KM, eta=A.DUO_GW31_ETA))
    with pytest.raises(ValueError, match="non-rotating"):
        A.shared_ic_bundle(model, spec)


def test_terrain_mismatch_is_refused():
    spec, model = _model("mountain_rossby_5_0")
    bad = SimpleNamespace(**{**vars(spec), "builder": A.duo_shared_ic_spec(
        "rest_state_topo", {"h_0": 2000.0}).builder})
    with pytest.raises(ValueError, match="not the grid's terrain"):
        A.shared_ic_bundle(model, bad)


# ---------------------------------------------------------------------------
# matrix dispatch: the cube branches forward the row's case and IC kwargs
# ---------------------------------------------------------------------------

class _Stop(Exception):
    pass


@pytest.fixture(scope="module")
def matrix_mod():
    import importlib.util
    import pathlib
    import sys
    path = (pathlib.Path(__file__).resolve().parents[2]
            / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
    spec = importlib.util.spec_from_file_location("_atmmatrix_duo", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("case, run_kwargs, runner, expect", [
    ("rest_state_topo", {"h_0": 1234.0}, "run_baroclinic",
     {"shared_case": "rest_state_topo", "ic_kwargs": {"h_0": 1234.0}}),
    ("held_suarez_topo", {"h_0": 1234.0}, "run_held_suarez",
     {"shared_case": "held_suarez_topo", "held_suarez": True,
      "ic_kwargs": {"h_0": 1234.0}}),
    ("gravity_wave_3_1", {}, "run_baroclinic",
     {"shared_case": "gravity_wave_3_1", "ic_kwargs": None}),
    ("inertio_gravity_3_2", {}, "run_baroclinic",
     {"shared_case": "inertio_gravity_3_2", "ic_kwargs": None}),
    ("mountain_rossby_5_0", {}, "run_baroclinic",
     {"shared_case": "mountain_rossby_5_0", "ic_kwargs": None}),
    ("rossby_haurwitz_6_0", {}, "run_baroclinic",
     {"shared_case": "rossby_haurwitz_6_0", "ic_kwargs": None}),
])
def test_cube_branch_forwards_case_and_ic_kwargs(matrix_mod, monkeypatch,
                                                 tmp_path, case, run_kwargs,
                                                 runner, expect):
    seen = {}

    def fake(n, **kw):
        seen.update(kw, n=n)
        raise _Stop

    monkeypatch.setattr(A, "build_duo_hydro", fake)
    monkeypatch.delenv("LEGOESM_TOPO_H0", raising=False)
    monkeypatch.delenv("LEGOESM_TOPO_SEED", raising=False)
    tc = matrix_mod.TestCase("hydrostatic", case, "cubed_sphere", "C8",
                             "hybrid", 1.0, 0.25, dict(run_kwargs))
    with pytest.raises(_Stop):
        getattr(matrix_mod, runner)(tc, tmp_path, 0.25)
    assert seen["n"] == 8
    for k, v in expect.items():
        assert seen.get(k) == v, (k, seen)


def test_matrix_cube_held_suarez_is_plain_hs94(monkeypatch):
    """The matrix cube HS row applies Held & Suarez (1994) like every other
    grid (strat=False), not FV3 solo's modified-stratosphere default."""
    import legoesm.core.fv3_native_physics_coupling as cpl
    seen = []

    def _spy(state, press, *a, strat=True, **k):
        seen.append(strat)
        return state
    monkeypatch.setattr(cpl, "apply_held_suarez_step_sixface_jax", _spy)
    duo = A.build_duo_hydro(36, held_suarez=True)
    duo.step(duo.state, duo.dt)
    assert seen == [False], seen
    assert "Held-Suarez 1994" in duo.note
