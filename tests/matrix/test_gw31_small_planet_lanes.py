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
                    state=state)
        raise _Stop

    monkeypatch.setattr(mm, "_run_timeloop", fake)
    tc = mm.TestCase("hydrostatic", "gravity_wave_3_1", grid, res, "hybrid",
                     3600.0 / 86400.0, 3600.0 / 86400.0, {})
    with pytest.raises(_Stop):
        mm.run_baroclinic(tc, tmp_path, tc.quick_days)
    return seen


def test_cube_lane_integrates_to_3600_s(mm, monkeypatch, tmp_path):
    s = _loop_args(mm, monkeypatch, tmp_path, "cubed_sphere", "C36")
    s.pop("state")
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
    # the 2012 IC reached the state (after the lane's C-grid conversion)
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        hydrostatic_to_cgrid)
    ref = hydrostatic_to_cgrid(gw.gravity_wave_init_latlon(grid, sigma), grid)
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
WIRED = {"cubed_sphere", "latlon"}


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
