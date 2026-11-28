"""DCMIP 2012 test 3-1 (``gravity_wave_3_1``) matrix lanes: the small-planet
radius, the X-scaled step / diffusion and the App. F.3 L10 table reach each
lane's model, and the row's diagnostics land on the document's times.
The cube (FV3 duo) lane is covered in ``test_cube_duo_shared_ic.py``.
No stepping beyond construction.  Needs JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

jax = pytest.importorskip("jax")


@pytest.fixture(scope="module")
def mm():
    path = (pathlib.Path(__file__).resolve().parents[2]
            / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
    spec = importlib.util.spec_from_file_location("_atmmatrix_gw31", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("dt, k_expect", [(15.0, 5), (1.6, 25), (4.8, 5)])
def test_diag_cadence_hits_the_document_times(mm, dt, k_expect):
    """cube 15 s, latlon/ico 1.6 s, spectral 4.8 s"""
    k = mm._gw31_diag_every(dt)
    assert k == k_expect
    assert k * dt <= 100.0 + 1e-9
    for t in (3000.0, 3600.0):
        n = t / (k * dt)
        assert abs(n - round(n)) < 1e-9, (dt, k, t)
    for bad in (7.0, 4.800000000001, 6.8965517241379315):
        with pytest.raises(ValueError, match="does not divide 600"):
            mm._gw31_diag_every(bad)
    with pytest.raises(ValueError, match="not in"):
        mm._gw31_diag_every(120.0)


def test_step_count_lands_on_the_run_end(mm):
    days = 3600.0 / 86400.0
    assert mm._gw31_n_steps(days, 15.0) == 240
    assert mm._gw31_n_steps(days, 1.6) == 2250
    assert mm._gw31_n_steps(days, 4.8) == 750
    dt = 600.0 / 673       # int(3600 / dt) == 4037: one step short
    assert int(days * 86400 / dt) == 4037
    assert mm._gw31_n_steps(days, dt) == 4038
    with pytest.raises(ValueError, match="does not divide"):
        mm._gw31_n_steps(days, 7.0)


class _Stop(Exception):
    pass


def _loop_args(mm, monkeypatch, tmp_path, grid, res):
    """run_baroclinic on the 3-1 row with the time loop stubbed: the (dt,
    n_steps, diag_every) it would integrate with."""
    seen = {}

    def fake(step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
             diag_every, key_array_fn, **kw):
        seen.update(dt=dt, n_steps=n_steps, diag_every=diag_every,
                    state=state, step_fn=step_fn)
        raise _Stop

    monkeypatch.setattr(mm, "_run_timeloop", fake)
    tc = mm.TestCase("hydrostatic", "gravity_wave_3_1", grid, res, "hybrid",
                     3600.0 / 86400.0, 3600.0 / 86400.0, {})
    with pytest.raises(_Stop):
        mm.run_baroclinic(tc, tmp_path, tc.quick_days)
    return seen


def test_cube_lane_integrates_to_3600_s(mm, monkeypatch, tmp_path):
    s = _loop_args(mm, monkeypatch, tmp_path, "cubed_sphere", "C36")
    s.pop("state"), s.pop("step_fn")
    assert s == {"dt": 15.0, "n_steps": 240, "diag_every": 5}


def _spy_model(monkeypatch, module, cls_name):
    """Record the constructor arguments of a lane's model class (the real
    model is still built)."""
    import importlib
    mod = importlib.import_module(module)
    real = getattr(mod, cls_name)
    seen = {}

    class Spy(real):
        def __init__(self, *a, **k):
            seen["args"], seen["kwargs"] = a, k
            super().__init__(*a, **k)
            seen["model"] = self

    monkeypatch.setattr(mod, cls_name, Spy)
    return seen


def _assert_gw31_table(sigma):
    import numpy as np
    from legoesm.grids.vertical import make_dcmip2012_gw31_levels
    ref = make_dcmip2012_gw31_levels()
    assert np.array_equal(np.asarray(sigma.B_half), np.asarray(ref.B_half))
    assert np.array_equal(np.asarray(sigma.A_half), np.asarray(ref.A_half))


def test_latlon_lane_runs_the_small_planet(mm, monkeypatch, tmp_path):
    """72x144: radius a_ref/125 and Omega = 0 on the grid, the L10 table,
    the 2012 IC, dt = 200 s / X = 1.6 s and the del-2 A_h divided by X."""
    import math
    import numpy as np
    from legoesm import constants
    from tests.test_cases.dcmip2012 import gravity_wave_3_1 as gw
    seen = _spy_model(monkeypatch,
                      "legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid",
                      "CGridLatLonPrimitiveEquationModel")
    s = _loop_args(mm, monkeypatch, tmp_path, "latlon", "72x144")
    assert s["dt"] == 1.6 and s["n_steps"] == 2250 and s["diag_every"] == 25
    grid, sigma, cfg = seen["args"][:3]
    assert (grid.n_lat, grid.n_lon) == (72, 144)
    assert seen["kwargs"]["dt"] == 1.6
    assert float(grid.radius) == gw.small_planet_radius()
    assert float(grid.omega) == 0.0
    _assert_gw31_table(sigma)
    # del-2 K_2 / X (k = 1), then the lane's own stability cap
    # 0.4 dx_pole^2 / dt, itself on the small grid
    ah_earth = mm._laplacian_visc_latlon(72)
    dx_pole = float(grid.radius) * grid.dlon * math.cos(
        math.pi / 2 - grid.dlat / 2)
    assert cfg.A_h == min(ah_earth / 125.0, 0.4 * dx_pole ** 2 / 1.6)
    # the polar Fourier filter stays on with the lane's settings
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig as Cfg)
    assert cfg.use_polar_filter is True
    for k in ("polar_filter_cutoff_deg", "polar_filter_max_wave_speed"):
        assert getattr(cfg, k) == Cfg._field_defaults[k]
    # the lane's other settings are the ordinary lat-lon rows'
    assert cfg.A_h_pressure_surfaces == mm._LATLON_A_H_PRESSURE_SURFACES
    assert cfg.sb81_omega_conversion is True
    # the shared-core sb81_theta form with the conservative SB theta
    # transport (the ico lane's vertical numerics); the legacy form's upwind
    # vertical T transport drifted the level means and wiped the packets
    assert cfg.vertical_form == "sb81_theta"
    assert cfg.vert_advection_scheme == "sb"
    # the 2012 IC reached the state (after the lane's C-grid conversion)
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        hydrostatic_to_cgrid)
    ref = hydrostatic_to_cgrid(gw.gravity_wave_init_latlon(grid, sigma), grid)
    assert np.all(np.asarray(s["state"].phis) == 0.0)          # z_s = 0
    for k in ("T", "p_s", "u", "v"):
        np.testing.assert_array_equal(np.asarray(getattr(s["state"], k)),
                                      np.asarray(getattr(ref, k)), err_msg=k)
    lat = np.asarray(grid.lat, dtype=np.float64)
    ps = np.asarray(s["state"].p_s)
    # Eqs. 89-90, 93 (Omega = 0) written out independently
    g, cp, rd, kap = constants.g, constants.c_pd, constants.R_d, constants.kappa
    G = g * g / (1e-4 * cp)
    c2 = np.cos(2 * lat) - 1.0
    Ts = G + (300.0 - G) * np.exp(-400.0 * 1e-4 / (4 * g * g) * c2)
    ps_ref = 1e5 * np.exp(400.0 / (4 * G * rd) * c2) * (Ts / 300.0) ** (1 / kap)
    np.testing.assert_allclose(ps[:, 0], ps_ref, rtol=1e-12)
    assert np.abs(np.asarray(s["state"].u)).max() > 19.0  # u0 cos(lat)
    assert float(constants.R_earth) / float(grid.radius) == pytest.approx(125.0)


def test_latlon_lane_divides_the_del2_viscosity_by_x(mm, monkeypatch,
                                                     tmp_path):
    """Below the lane's stability cap the del-2 coefficient reaches the
    model divided by X (the 72x144 Earth value is capped either way)."""
    seen = _spy_model(monkeypatch,
                      "legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid",
                      "CGridLatLonPrimitiveEquationModel")
    monkeypatch.setattr(mm, "_laplacian_visc_latlon",
                        lambda n_lat, frac=0.1: 1000.0)
    _loop_args(mm, monkeypatch, tmp_path, "latlon", "72x144")
    assert seen["args"][2].A_h == 1000.0 / 125.0


#: the lanes wired to the DCMIP 2012 3-1 setup so far (explicit, not read
#: from the module under test)
WIRED = {"cubed_sphere", "latlon", "icosahedral", "spectral"}


@pytest.mark.parametrize("grid", ["cubed_sphere", "latlon", "icosahedral",
                                  "spectral"])
def test_namelist_table_label_follows_the_wired_lanes(mm, grid):
    tc = mm.TestCase("hydrostatic", "gravity_wave_3_1", grid, "x",
                     "hybrid", 1.0, 1.0, {})
    label = mm._namelist_vertical_table(tc)
    assert label.startswith("dcmip2012_gw31_l10") == (grid in WIRED), label
    # ordinary hybrid rows keep their table label
    other = mm.TestCase("hydrostatic", "baroclinic", grid, "x", "hybrid",
                        1.0, 1.0, {})
    assert mm._namelist_vertical_table(other).startswith("cam6_l32")
    assert mm._vertical_table_id(mm._vertical_table_label(
        "hybrid", __import__("legoesm.grids.vertical", fromlist=["x"])
        .make_dcmip2012_gw31_levels())) == "dcmip2012_gw31_l10"


def test_ico_lane_runs_the_small_planet(mm, monkeypatch, tmp_path):
    """ico5: radius a_ref/125, f = 0 on the mesh, the L10 table, the 2012 IC,
    dt = 200 s / X = 1.6 s and del-4 nu = (Earth-mesh value) / X^3."""
    import numpy as np
    from legoesm.grids.voronoi import create_voronoi_mesh
    from tests.test_cases.dcmip2012 import gravity_wave_3_1 as gw
    monkeypatch.delenv("LEGOESM_MPAS_INTEGRATOR", raising=False)
    seen = _spy_model(monkeypatch,
                      "legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas",
                      "MPASPrimitiveEquationModel")
    s = _loop_args(mm, monkeypatch, tmp_path, "icosahedral", "ico5")
    assert s["dt"] == 1.6 and s["n_steps"] == 2250 and s["diag_every"] == 25
    mesh, sigma, cfg = seen["args"][:3]
    assert float(mesh.radius) == gw.small_planet_radius()
    assert np.asarray(mesh.fEdge).shape == (mesh.nEdges,)
    assert np.asarray(mesh.fVertex).shape == (mesh.nVertices,)
    assert np.all(np.asarray(mesh.fEdge) == 0.0)
    assert np.all(np.asarray(mesh.fVertex) == 0.0)
    _assert_gw31_table(sigma)
    earth = create_voronoi_mesh(5)
    # every length / area the solver reads is the Earth mesh's over X, X^2
    # (the two builds agree to ~1e-7, mesh-construction round-off)
    for k, p in (("dcEdge", 1), ("dvEdge", 1), ("areaCell", 2),
                 ("areaTriangle", 2), ("kiteAreasOnVertex", 2)):
        e = np.asarray(getattr(earth, k), dtype=np.float64)
        sm = np.asarray(getattr(mesh, k), dtype=np.float64)
        assert e.size and np.all(e >= 0) and np.count_nonzero(e) > e.size // 2, k
        np.testing.assert_allclose(sm, e / 125.0 ** p, rtol=2e-6, err_msg=k)
    # del-4: the Earth-mesh dx_min^4 / (48 h), written out, over X^3
    dx = min(float(np.min(np.asarray(earth.dcEdge))),
             float(np.min(np.asarray(earth.dvEdge))))
    nu_earth = dx ** 4 / (48.0 * 3600.0)
    assert nu_earth > 1e13
    assert cfg.nu_del4 == pytest.approx(nu_earth / 125.0 ** 3, rel=1e-5)
    # the lane's other settings: the ordinary ico rows' (named)
    assert cfg.time_integrator == "ssp_rk3"      # the matrix ico default
    assert cfg.vert_advection_scheme == "sb"
    # fp64 matrix: no mass fixer (fp64 ICs conserve to roundoff, commit 4bcfab1aeb)
    assert cfg.nu_del2 == 0.0 and not cfg.fix_mass and not cfg.anchor_mass_to_initial
    T_ref = np.asarray(gw.gravity_wave_init_mpas(mesh, sigma).T.data)
    np.testing.assert_array_equal(np.asarray(s["state"].T.data), T_ref)
    np.testing.assert_array_equal(
        np.asarray(s["state"].p_s.data),
        np.asarray(gw.gravity_wave_init_mpas(mesh, sigma).p_s.data))
    assert np.all(np.asarray(s["state"].phis.data) == 0.0)     # z_s = 0
    # edge-normal winds: the projection of u0 cos(lat) eastward
    un = 20.0 * np.cos(np.asarray(mesh.latEdge, np.float64)) * np.cos(
        np.asarray(mesh.angleEdge, np.float64))
    np.testing.assert_allclose(np.asarray(s["state"].u.data),
                               np.broadcast_to(un[:, None], (un.size, 10)),
                               rtol=0, atol=1e-5)   # the mesh angles are f32
    assert np.abs(un).max() > 19.0

    # the step callback the lane hands the time loop advances the state one
    # 1.6 s step: finite, and the bell's temperature field evolves
    out = s["step_fn"](s["state"], s["dt"])
    for f in (out.T.data, out.u.data, out.p_s.data):
        assert np.all(np.isfinite(np.asarray(f)))
    assert np.max(np.abs(np.asarray(out.T.data)
                         - np.asarray(s["state"].T.data))) > 0.0
    # ... and it is the model's own step at the dt it is given
    ref = seen["model"].step(s["state"], s["dt"])
    for k in ("T", "u", "p_s"):
        np.testing.assert_array_equal(np.asarray(getattr(out, k).data),
                                      np.asarray(getattr(ref, k).data),
                                      err_msg=k)
    half = s["step_fn"](s["state"], 0.8)
    assert not np.array_equal(np.asarray(half.T.data), np.asarray(out.T.data))


def test_ico_control_row_keeps_the_earth_lane(mm, monkeypatch, tmp_path):
    """An ordinary ico5 row (inertio_gravity_3_2) still gets the Earth mesh,
    the matrix's CAM6 L32 table, dt = 200 s and the unscaled del-4."""
    import numpy as np
    from legoesm import constants
    for v in ("LEGOESM_MATRIX_NLEV", "LEGOESM_MATRIX_TRANSITION_EXPONENT"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("LEGOESM_MPAS_INTEGRATOR", raising=False)
    seen = _spy_model(monkeypatch,
                      "legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas",
                      "MPASPrimitiveEquationModel")
    loop = {}

    def fake(step_fn, state, dt, n_steps, *a, **k):
        loop.update(dt=dt, n_steps=n_steps, state=state)
        raise _Stop

    monkeypatch.setattr(mm, "_run_timeloop", fake)
    tc = mm.TestCase("hydrostatic", "inertio_gravity_3_2", "icosahedral",
                     "ico5", "hybrid", 3.0, 0.5, {})
    with pytest.raises(_Stop):
        mm.run_baroclinic(tc, tmp_path, tc.quick_days)
    mesh, sigma, cfg = seen["args"][:3]
    assert float(mesh.radius) == float(constants.R_earth)
    from legoesm.grids.vertical import make_cam6_l32_levels
    ref = make_cam6_l32_levels()
    for k in ("A_half", "B_half", "A_full", "B_full"):
        np.testing.assert_array_equal(np.asarray(getattr(sigma, k)),
                                      np.asarray(getattr(ref, k)), err_msg=k)
    assert loop["dt"] == 200.0 and loop["n_steps"] == int(0.5 * 86400 / 200)
    dx = min(float(np.min(np.asarray(mesh.dcEdge))),
             float(np.min(np.asarray(mesh.dvEdge))))
    assert cfg.nu_del4 == dx ** 4 / (48.0 * 3600.0)
    np.testing.assert_allclose(
        np.asarray(mesh.fEdge),
        2 * float(constants.Omega) * np.sin(np.asarray(mesh.latEdge)),
        rtol=1e-6, atol=1e-12)                     # the rotating Earth
    from tests.test_cases.dcmip2008.inertio_gravity_3_2 import (
        inertio_gravity_init_mpas)
    ref = inertio_gravity_init_mpas(mesh, sigma)
    for k in ("T", "u", "p_s", "phis"):
        np.testing.assert_array_equal(np.asarray(getattr(loop["state"], k).data),
                                      np.asarray(getattr(ref, k).data),
                                      err_msg=k)


_SPEC = "legoesm.atmosphere.dynamics.gcm.spectral_pe"


def test_spectral_lane_runs_the_small_planet(mm, monkeypatch, tmp_path):
    """T21: radius a_ref/125, Omega = 0, the L10 table, the 2012 IC,
    dt = 600 s / X = 4.8 s, DCMIP 2008 Table 4 CAM-EUL del-4 K4 / X^3, no
    exp filter, full band (user decision 2026-11-27)."""
    import numpy as np
    from legoesm.grids.gaussian import sh_synthesis_3d
    from tests.test_cases.dcmip2012 import gravity_wave_3_1 as gw
    seen = _spy_model(monkeypatch, _SPEC, "SpectralPrimitiveEquationModel")
    s = _loop_args(mm, monkeypatch, tmp_path, "spectral", "T21")
    assert s["dt"] == 4.8 and s["n_steps"] == 750 and s["diag_every"] == 5
    grid, sigma, cfg = seen["args"][:3]
    assert grid.n_max == 21
    assert float(grid.radius) == gw.small_planet_radius()
    assert float(grid.omega) == 0.0
    assert np.all(np.asarray(grid.f) == 0.0)
    _assert_gw31_table(sigma)
    assert cfg.hyperdiff_order == 2                     # del-4: k = 2
    assert cfg.hyperdiff_coeff == pytest.approx(2.0e16 / 125.0 ** 3,
                                                rel=1e-15)
    assert cfg.spectral_filter_strength == 0.0
    assert cfg.dealiasing_fraction == 1.0
    # the del-4 operator the step applies, on THIS grid: per degree l the
    # rate is K_4 (l(l+1)/a^2)^2 -- X times the Earth row's rate (so the
    # damping per step, rate * dt with dt / X, is the Earth row's)
    from legoesm.grids.gaussian import (
        create_gaussian_grid, spectral_hyperdiffusion_3d)
    from legoesm import constants
    ones = np.ones((grid.n_sh, 1), dtype=np.complex128)
    rate = -np.real(np.asarray(spectral_hyperdiffusion_3d(
        grid, ones, cfg.hyperdiff_coeff, cfg.hyperdiff_order)))[:, 0]
    ell = np.asarray(grid.ls, dtype=np.float64)
    a = gw.small_planet_radius()
    np.testing.assert_allclose(rate, 2.0e16 / 125.0 ** 3
                               * (ell * (ell + 1) / a ** 2) ** 2, rtol=1e-12)
    earth = create_gaussian_grid(21, omega=0.0)
    rate_e = -np.real(np.asarray(spectral_hyperdiffusion_3d(
        earth, ones, 2.0e16, 2)))[:, 0]
    m = ell > 0
    np.testing.assert_allclose(rate[m] / rate_e[m], 125.0, rtol=1e-6)
    np.testing.assert_allclose(rate[m] * 4.8, rate_e[m] * 600.0, rtol=1e-6)
    assert float(earth.radius) == float(constants.R_earth)
    assert cfg.T_damping_pressure_surfaces is False
    assert cfg.fix_mass and cfg.anchor_mass_to_initial
    ref = gw.gravity_wave_init_spectral(grid, sigma)
    for k in ("vor_hat", "div_hat", "T_hat", "lnps_hat", "phis_hat"):
        np.testing.assert_array_equal(np.asarray(getattr(s["state"], k).data),
                                      np.asarray(getattr(ref, k).data),
                                      err_msg=k)
    T = np.asarray(sh_synthesis_3d(grid, s["state"].T_hat.data))
    assert np.all(np.isfinite(T)) and 225.0 < T.min() and T.max() < 305.0
    # the loop's step callback is the model's own step at the given dt,
    # and it applies the EXPLICIT del-4 on the small grid with the scaled
    # coefficient (recorded at trace time)
    assert cfg.implicit_hyperdiff is False
    import importlib
    spm = importlib.import_module(_SPEC)
    real_hd, hd_calls = spm.spectral_hyperdiffusion_3d, []

    def hd_spy(g, c, nu, order=2):
        hd_calls.append((g, nu, order))
        return real_hd(g, c, nu, order)

    monkeypatch.setattr(spm, "spectral_hyperdiffusion_3d", hd_spy)
    out = s["step_fn"](s["state"], s["dt"])
    assert hd_calls and all(g is grid and nu == cfg.hyperdiff_coeff
                            and order == 2 for g, nu, order in hd_calls)
    # ... and its tendency reaches the state: the same step without it
    # (a fresh model, hyperdiff_coeff = 0) evolves T differently
    monkeypatch.setattr(spm, "spectral_hyperdiffusion_3d", real_hd)
    real_cls = type(seen["model"]).__bases__[0]      # the spy's base
    nohd = real_cls(grid, sigma, cfg._replace(hyperdiff_coeff=0.0)).step(
            s["state"], s["dt"])
    assert not np.array_equal(np.asarray(nohd.T_hat.data),
                              np.asarray(out.T_hat.data))
    mod = seen["model"].step(s["state"], s["dt"])
    for k in ("vor_hat", "div_hat", "T_hat", "lnps_hat"):
        a = np.asarray(getattr(out, k).data)
        assert np.all(np.isfinite(a)), k
        np.testing.assert_array_equal(a, np.asarray(getattr(mod, k).data),
                                      err_msg=k)
    assert not np.array_equal(np.asarray(out.T_hat.data),
                              np.asarray(s["state"].T_hat.data))


def test_spectral_control_row_keeps_the_earth_lane(mm, monkeypatch,
                                                   tmp_path):
    """An ordinary T21 row (inertio_gravity_3_2): Earth radius and rotation,
    CAM6 L32, dt = 600 s, the unscaled hyperdiffusion."""
    import numpy as np
    from legoesm import constants
    from legoesm.grids.vertical import make_cam6_l32_levels
    for v in ("LEGOESM_MATRIX_NLEV", "LEGOESM_MATRIX_TRANSITION_EXPONENT"):
        monkeypatch.delenv(v, raising=False)
    seen = _spy_model(monkeypatch, _SPEC, "SpectralPrimitiveEquationModel")
    loop = {}

    def fake(step_fn, state, dt, n_steps, *a, **k):
        loop.update(dt=dt, n_steps=n_steps, state=state)
        raise _Stop

    monkeypatch.setattr(mm, "_run_timeloop", fake)
    tc = mm.TestCase("hydrostatic", "inertio_gravity_3_2", "spectral", "T21",
                     "hybrid", 3.0, 0.5, {})
    with pytest.raises(_Stop):
        mm.run_baroclinic(tc, tmp_path, tc.quick_days)
    grid, sigma, cfg = seen["args"][:3]
    assert float(grid.radius) == float(constants.R_earth)
    assert float(grid.omega) == float(constants.Omega)
    ref = make_cam6_l32_levels()
    for k in ("A_half", "B_half"):
        np.testing.assert_array_equal(np.asarray(getattr(sigma, k)),
                                      np.asarray(getattr(ref, k)), err_msg=k)
    assert loop["dt"] == 600.0 and loop["n_steps"] == int(0.5 * 86400 / 600)
    # DCMIP-2008 dry row: its document's EUL dissipation (Table 4 K4 at
    # T21, no exp filter; test_spectral_dcmip2008_dissipation.py)
    assert cfg.hyperdiff_coeff == 2.0e16
    assert cfg.spectral_filter_strength == 0.0
    from tests.test_cases.dcmip2008.inertio_gravity_3_2 import (
        inertio_gravity_init_spectral)
    ref_state = inertio_gravity_init_spectral(grid, sigma)
    for k in ("vor_hat", "div_hat", "T_hat", "lnps_hat", "phis_hat"):
        np.testing.assert_array_equal(
            np.asarray(getattr(loop["state"], k).data),
            np.asarray(getattr(ref_state, k).data), err_msg=k)
