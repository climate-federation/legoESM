"""Case-6 duo scoring pipeline: runner + gate direct tests.

Covers the two new scripts (``run_duo_stepper_case6.py``,
``case6_duo_oracle_gate.py``) without touching the Zenodo reference
file:

* the analytic RH4 fields against HAND-REDUCED closed forms at chosen
  points (at lat=0 the A/B/C coefficients collapse to scalar arithmetic
  with no cos powers — an independent factoring, so a wrong exponent or
  coefficient in the module cannot cancel here);
* the gate's pure ``score_day`` math against loop-written hand values,
  plus non-vacuity (zero for identical fields, all metrics move when a
  cell is perturbed);
* the npz input contract (the historical 0..359 canvas MUST be
  rejected — scoring it against the 0.5-based reference canvas would
  be a silent half-cell aliasing);
* the runner's sampling on a real C12 duo context: the nearest-cell gh
  sample must be BIT-EQUAL to the analytic geopotential at the selected
  cell centre (same coordinates, same function), and the c2l_ord2 wind
  lens must reproduce the analytic wind at C12 truncation level.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from legoesm.core.fv3_native_duo_stepper import (
    SW_CFG_CASE8,
    build_six_face_duo_context,
    case6_six_face_state,
)
from legoesm.core.williamson_sw_analytic import (
    RH4_K_HZ,
    RH4_MEAN_DEPTH_M,
    RH4_OMEGA_WAVE_HZ,
    rossby_haurwitz_4_geopotential,
    rossby_haurwitz_4_winds,
)
from legoesm.grids.fv3_native_gridstruct import (
    FV3_GRAV,
    FV3_OMEGA,
    FV3_RADIUS_M,
)

_DIR = (pathlib.Path(__file__).resolve().parents[2]
        / "scripts" / "validate" / "fv3_native")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name,
                                                  _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load("run_duo_stepper_case6")
gate = _load("case6_duo_oracle_gate")


# ---------------------------------------------------------------------
# hand-computed analytic points
# ---------------------------------------------------------------------

def test_hand_computed_analytic_points():
    """At lat=0 (cos=1, sin=0) the case-6 coefficients reduce, BY HAND:

        A(0) = w(2 Om + w)/2 - k^2/4
        B(0) = 2 (Om + w) k / 30 * (26 - 25) = (Om + w) k / 15
        C(0) = k^2 (5 - 6)/4 = -k^2/4
        gh(0, 0)      = gh0 + a^2 (A + B + C)
        u(0, 0)       = a w - a k            (== 0, since w == k)
        u(pi/4, 0)    = a w + a k            (cos(4 lon) = -1)
        v(lon, 0)     = 0                    (sin(lat) factor)
    """
    a = FV3_RADIUS_M
    om = FV3_OMEGA
    w = RH4_OMEGA_WAVE_HZ
    k = RH4_K_HZ
    gh0 = RH4_MEAN_DEPTH_M * FV3_GRAV

    gh_hand = gh0 + a * a * (0.5 * w * (2.0 * om + w) - 0.25 * k * k
                             + (om + w) * k / 15.0
                             - 0.25 * k * k)
    gh_mod = rossby_haurwitz_4_geopotential(0.0, 0.0, radius=a, omega=om,
                                            gh0=gh0)
    assert gh_mod == pytest.approx(gh_hand, rel=1e-12)

    u0, v0 = rossby_haurwitz_4_winds(0.0, 0.0, radius=a)
    assert u0 == 0.0          # a*w - a*k with w == k: exact float zero
    assert v0 == 0.0
    u1, v1 = rossby_haurwitz_4_winds(np.pi / 4.0, 0.0, radius=a)
    assert u1 == pytest.approx(a * w + a * k, rel=1e-14)
    assert v1 == 0.0
    # magnitude sanity: the wave peaks near 100 m/s at the equator
    assert 99.0 < u1 < 101.0


def test_reference_canvas_matches_gate_contract():
    """Runner and gate must agree on the canvas, or the contract check
    rejects every legitimate npz."""
    lat_r, lon_r = runner.reference_canvas()
    lat_g, lon_g = gate._canvas()
    np.testing.assert_allclose(lat_r, lat_g, atol=0)
    np.testing.assert_allclose(lon_r, lon_g, atol=0)
    assert lon_r[0] == 0.5 and lon_r[-1] == 359.5      # T-cell centres
    assert lat_r[0] == -90.0 and lat_r[-1] == 90.0


def test_runner_deck_constants_pin_the_resolved_echo():
    """The deck values this port claims to match (logfile echo)."""
    assert runner.CASE6_D4_BG == 0.0
    assert runner.CASE6_DT_ATMOS_S == 1200.0
    assert runner.CASE6_N_SPLIT == 7
    cfg = {**SW_CFG_CASE8, "d4_bg": runner.CASE6_D4_BG}
    diff = {key for key in cfg if cfg[key] != SW_CFG_CASE8[key]}
    assert diff == {"d4_bg"}, (
        "case-6 config must differ from the certified case-8 block in "
        f"d4_bg ONLY, got extra diffs: {diff}")


# ---------------------------------------------------------------------
# gate: pure scoring math
# ---------------------------------------------------------------------

def _toy(run_gh, ref_gh, run_u, ref_u, run_v, ref_v):
    run = {"gh": np.asarray(run_gh, float),
           "u": np.asarray(run_u, float), "v": np.asarray(run_v, float)}
    ref = {"gh": np.asarray(ref_gh, float),
           "u": np.asarray(ref_u, float), "v": np.asarray(ref_v, float)}
    return run, ref


def test_score_day_hand_math():
    """(2 x 3) toy vs loop-written hand values."""
    lat = np.array([0.0, 60.0])                 # cos = 1, 0.5
    ref_gh = np.full((2, 3), 2.0)
    run_gh = ref_gh + np.array([[1.0, 1.0, 1.0],
                                [0.0, 0.0, 0.0]])
    ref_u = np.full((2, 3), 3.0)
    ref_v = np.full((2, 3), 4.0)
    run_u = ref_u + 1.0
    run_v = ref_v.copy()
    run, ref = _toy(run_gh, ref_gh, run_u, ref_u, run_v, ref_v)
    s = gate.score_day(run, ref, lat)

    # hand: gh rel_l2 = sqrt(3 * 1) / sqrt(6 * 4)
    assert s["gh"]["rel_l2"] == pytest.approx(np.sqrt(3.0 / 24.0),
                                              rel=1e-12)
    # hand cosw: num = sqrt(1*(1+1+1)); den = sqrt((1+0.5)*3*4)
    assert s["gh"]["rel_l2_cosw"] == pytest.approx(
        np.sqrt(3.0 / 18.0), rel=1e-12)
    assert s["gh"]["max_abs"] == pytest.approx(1.0, abs=0)
    # winds: vector norm = sqrt(6*9 + 6*16) = sqrt(150); du = 1 each
    assert s["u"]["rel_l2"] == pytest.approx(np.sqrt(6.0 / 150.0),
                                             rel=1e-12)
    assert s["v"]["rel_l2"] == 0.0
    # cosw winds: num u = sqrt((1+0.5)*3*1) = sqrt(4.5);
    # den = sqrt((1+0.5)*3*(9+16)) = sqrt(112.5)
    assert s["u"]["rel_l2_cosw"] == pytest.approx(
        np.sqrt(4.5 / 112.5), rel=1e-12)


def test_score_day_zero_and_nonvacuous():
    lat = np.array([0.0, 45.0])
    f = np.arange(6.0).reshape(2, 3) + 5.0
    run, ref = _toy(f, f, f + 1.0, f + 1.0, f - 2.0, f - 2.0)
    s = gate.score_day(run, ref, lat)
    for key in ("gh", "u", "v"):
        assert s[key]["rel_l2"] == 0.0
        assert s[key]["rel_l2_cosw"] == 0.0
        assert s[key]["max_abs"] == 0.0
    run["gh"][1, 2] += 0.5                      # one-cell perturbation
    run["v"][0, 0] -= 0.25
    s2 = gate.score_day(run, ref, lat)
    assert s2["gh"]["rel_l2"] > 0 and s2["gh"]["rel_l2_cosw"] > 0
    assert s2["gh"]["max_abs"] == pytest.approx(0.5)
    assert s2["v"]["max_abs"] == pytest.approx(0.25)
    assert s2["u"]["max_abs"] == 0.0


def test_score_day_rejects_zero_reference_norm():
    lat = np.array([0.0])
    z = np.zeros((1, 2))
    run, ref = _toy(z + 1.0, z, z, z, z, z)
    with pytest.raises(ValueError, match="reference norm is zero"):
        gate.score_day(run, ref, lat)


# ---------------------------------------------------------------------
# gate: npz input contract
# ---------------------------------------------------------------------

def _write_npz(tmp_path, **over):
    lat, lon = runner.reference_canvas()
    nt = 2
    base = dict(times_days=np.array([0.0, 1.0]),
                gh=np.full((nt, 181, 360), 7.8e4),
                u=np.zeros((nt, 181, 360)),
                v=np.zeros((nt, 181, 360)),
                lat=lat, lon=lon)
    base.update(over)
    path = tmp_path / "run.npz"
    np.savez_compressed(path, **base)
    return str(path)


def test_contract_accepts_canonical(tmp_path):
    out = gate.load_run(_write_npz(tmp_path))
    assert out["gh"].shape == (2, 181, 360)


def test_contract_rejects_historical_canvas(tmp_path):
    """lon 0..359 (the W2/modon canvas) must be REJECTED: half a cell
    off the reference T-cell centres."""
    path = _write_npz(tmp_path, lon=np.arange(360, dtype=float))
    with pytest.raises(gate.ContractError, match="T-cell centres"):
        gate.load_run(path)


@pytest.mark.parametrize("mutate, match", [
    (dict(times_days=np.array([1.0, 2.0])), "must be 0.0"),
    (dict(times_days=np.array([0.0, 0.0])), "strictly increasing"),
    (dict(lat=np.linspace(-89.0, 89.0, 181)), "linspace"),
    (dict(gh=np.full((2, 181, 360), np.nan)), "non-finite"),
    (dict(u=np.zeros((1, 181, 360))), "aligned with times_days"),
])
def test_contract_rejections(tmp_path, mutate, match):
    with pytest.raises(gate.ContractError, match=match):
        gate.load_run(_write_npz(tmp_path, **mutate))


def test_contract_rejects_missing_key(tmp_path):
    lat, lon = runner.reference_canvas()
    path = tmp_path / "short.npz"
    np.savez_compressed(path, times_days=np.array([0.0]), lat=lat,
                        lon=lon, gh=np.zeros((1, 181, 360)),
                        u=np.zeros((1, 181, 360)))
    with pytest.raises(gate.ContractError, match="missing 'v'"):
        gate.load_run(str(path))


# ---------------------------------------------------------------------
# gate: analytic fields on the canvas
# ---------------------------------------------------------------------

def test_analytic_ic_fields_shapes_and_values():
    f = gate.analytic_ic_fields()
    lat, lon = runner.reference_canvas()
    assert f["gh"].shape == (181, 360)
    # spot value == the shared module at the same point (row 90 is the
    # equator, column j at lon 0.5 + j degrees)
    j = 17
    gh_pt = rossby_haurwitz_4_geopotential(
        np.deg2rad(lon[j]), np.deg2rad(lat[90]), radius=FV3_RADIUS_M,
        omega=FV3_OMEGA, gh0=RH4_MEAN_DEPTH_M * FV3_GRAV)
    assert f["gh"][90, j] == gh_pt
    # non-vacuity: the wave is present, not a constant field
    assert f["gh"].std() > 1e3
    assert 95.0 < np.abs(f["u"]).max() < 105.0


# ---------------------------------------------------------------------
# runner: sampling on a real C12 duo context
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def c12():
    ctx = build_six_face_duo_context(12, 3, use_ext_bundle=True,
                                     oracle_conventions=True)
    states = case6_six_face_state(ctx)
    w2 = _load("run_duo_stepper_w2")
    _, lon_deg = runner.reference_canvas()
    nmap = w2.build_nearest_map(ctx, lon_deg=lon_deg)
    return ctx, states, nmap


def test_sampled_gh_bit_equals_analytic_at_selected_centres(c12):
    """The nearest-cell gh sample IS the analytic geopotential at the
    selected cell centre: same coordinates, same shared function —
    bit-equal, no tolerance."""
    ctx, states, nmap = c12
    gh, _, _ = runner.sample_fields(ctx, states, nmap)
    n, ng = ctx["n"], ctx["ng"]
    sl = slice(ng, ng + n)
    lons = np.concatenate([np.asarray(ctx["gs6"][t]["agrid_lon"])[sl, sl]
                           .ravel() for t in range(6)])
    lats = np.concatenate([np.asarray(ctx["gs6"][t]["agrid_lat"])[sl, sl]
                           .ravel() for t in range(6)])
    rng = np.random.default_rng(0)
    for i, j in zip(rng.integers(0, 181, 8), rng.integers(0, 360, 8)):
        idx = nmap[i, j]
        expect = rossby_haurwitz_4_geopotential(
            lons[idx], lats[idx], radius=FV3_RADIUS_M, omega=FV3_OMEGA,
            gh0=RH4_MEAN_DEPTH_M * FV3_GRAV)
        assert gh[i, j] == expect, (i, j, idx)


def test_sampled_winds_match_analytic_at_c12_truncation(c12):
    """c2l_ord2(analytic D projection) vs the analytic wind at the same
    cell centre: agreement at C12 truncation level certifies the lens
    wiring (a swapped component or sign fails by O(1))."""
    ctx, states, nmap = c12
    _, u, v = runner.sample_fields(ctx, states, nmap)
    n, ng = ctx["n"], ctx["ng"]
    sl = slice(ng, ng + n)
    lons = np.concatenate([np.asarray(ctx["gs6"][t]["agrid_lon"])[sl, sl]
                           .ravel() for t in range(6)])
    lats = np.concatenate([np.asarray(ctx["gs6"][t]["agrid_lat"])[sl, sl]
                           .ravel() for t in range(6)])
    ua, va = rossby_haurwitz_4_winds(lons, lats, radius=FV3_RADIUS_M)
    err = np.hypot(u - ua[nmap], v - va[nmap])
    scale = float(np.abs(np.hypot(ua, va)).max())     # ~100 m/s
    # C12 cells are ~7.5 deg; ord2 truncation measured well under 5%
    # of the peak wind; 10% would catch a component swap (O(100%))
    assert float(err.max()) < 0.10 * scale
    # non-vacuity: the sample carries the wave, not zeros
    assert float(np.abs(u).max()) > 0.5 * scale


def test_sample_fields_finite_and_shaped(c12):
    ctx, states, nmap = c12
    gh, u, v = runner.sample_fields(ctx, states, nmap)
    for f in (gh, u, v):
        assert f.shape == (181, 360)
        assert np.isfinite(f).all()
    # gh is a depth * g: strictly positive, near 8e3 * g
    assert gh.min() > 5e4 and gh.max() < 1.2e5
