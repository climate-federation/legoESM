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
    grid = create_fv3_duo_grid(N, **kw)
    model = FV3DuoDynamicsModel(
        grid, FV3DuoConfig(km=A.DUO_HYDRO_KM, eta=A.DUO_HYDRO_ETA))
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
    # hydrostatic surface pressure over THAT terrain, isothermal 300 K
    # (rtol 1e-6: held_suarez_init stores at the fp32 policy dtype)
    np.testing.assert_allclose(
        f.ps, constants.p_ref * np.exp(-f.phis / (constants.R_d * 300.0)),
        rtol=1e-6)
    assert f.ps.min() < 0.85 * constants.p_ref
    assert np.all(f.ue == 0.0) and np.all(f.vn == 0.0)
    assert np.all(np.isfinite(f.pt))
    if case == "rest_state_topo":
        assert np.all(f.pt == 300.0)
    else:   # HS: 1 K-scale noise on the lowest level only (FV3 k = km-1)
        np.testing.assert_allclose(f.pt[..., :-1], 300.0, rtol=0, atol=1e-4)
        assert 0.5 < np.abs(f.pt[..., -1] - 300.0).max() < 6.0


def test_gravity_wave_3_1_non_rotating_with_theta_bubble():
    spec, model = _model("gravity_wave_3_1")
    f = _fields(model, A.shared_ic_bundle(model, spec))
    assert np.all(f.f0 == 0.0)                       # omega = 0 on the grid
    assert np.all(f.phis == 0.0)
    np.testing.assert_allclose(f.ps, constants.p_ref, rtol=1e-14)
    # the bubble: warm anomaly at the (0E, 0N) centre, ~0 at the antipode
    d = np.abs(f.pt - 300.0).max(-1)
    i_c = np.unravel_index(np.argmax(np.cos(f.lat) * np.cos(f.lon)), d.shape)
    i_a = np.unravel_index(np.argmin(np.cos(f.lat) * np.cos(f.lon)), d.shape)
    assert d[i_c] > 0.5 and d[i_a] < 0.2 * d[i_c]
    # solid-body 20 cos(lat) through the D-grid projection and c2l lens
    # (MEASURED rel RMS 0.0059 at C8)
    ref = 20.0 * np.cos(f.lat)[..., None] * np.ones_like(f.ue)
    assert _rel_rms(f.ue, ref) < 0.02
    assert np.sqrt(np.mean(f.vn ** 2)) < 0.02 * np.sqrt(np.mean(ref ** 2))


def test_inertio_gravity_keeps_the_rotating_planet():
    spec, model = _model("inertio_gravity_3_2")
    f = _fields(model, A.shared_ic_bundle(model, spec))
    from legoesm.grids.fv3_native_gridstruct import FV3_OMEGA
    np.testing.assert_allclose(f.f0, 2 * FV3_OMEGA * np.sin(f.lat),
                               rtol=1e-6, atol=1e-12)
    assert np.abs(f.pt - 300.0).max() > 0.2


def test_mountain_rossby_terrain_isothermal_288():
    spec, model = _model("mountain_rossby_5_0")
    f = _fields(model, A.shared_ic_bundle(model, spec))
    i = np.unravel_index(np.argmax(f.phis), f.phis.shape)
    # the peak cell sits near (270E, 30N) (C8 cells are ~11 deg)
    assert abs(np.degrees(f.lon[i]) - 270.0) < 12 and abs(np.degrees(f.lat[i]) - 30.0) < 12
    assert f.phis.max() > 0.5 * constants.g * 2000.0
    assert np.all(f.pt == 288.0)
    np.testing.assert_allclose(
        f.ps, constants.p_ref * np.exp(-f.phis / (constants.R_d * 288.0)),
        rtol=1e-12)


def test_rossby_haurwitz_winds():
    spec, model = _model("rossby_haurwitz_6_0")
    f = _fields(model, A.shared_ic_bundle(model, spec))
    from legoesm.core.williamson_sw_analytic import rossby_haurwitz_4_winds
    from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M
    u, v = rossby_haurwitz_4_winds(f.lon, f.lat, radius=FV3_RADIUS_M)
    # D-grid projection + c2l lens at C8 (~4 cells per RH wave-4
    # wavelength): MEASURED rel RMS u 0.027, v 0.079; caps ~1.5-2x
    assert _rel_rms(f.ue, np.asarray(u)[..., None] * np.ones_like(f.ue)) < 0.05
    assert _rel_rms(f.vn, np.asarray(v)[..., None] * np.ones_like(f.vn)) < 0.12
    assert np.all(f.pt == 300.0)


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
