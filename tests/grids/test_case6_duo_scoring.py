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

# Oracle literals, hard-coded ON PURPOSE (codex c6 r1 #8: importing the
# module's own constants into its "hand" check is circular).  Sources:
#   test_cases.F90:1215-1218 (gh0 = 8e3*Grav, R=4, omg=rk=7.848e-6)
#   gfs_constants.h:33-36    (radius 6371.2e3, omega 7.2921e-5,
#                             grav 9.80665)
_A_ORACLE = 6371.2e3
_OM_ORACLE = 7.2921e-5
_W_ORACLE = 7.848e-6
_K_ORACLE = 7.848e-6
_GH0_ORACLE = 8.0e3 * 9.80665


def test_oracle_literals_match_the_module_constants():
    """If a shared constant drifts, the hand checks below would drift
    WITH it — so pin the module constants to the F90 literals first."""
    assert FV3_RADIUS_M == _A_ORACLE
    assert FV3_OMEGA == _OM_ORACLE
    assert RH4_OMEGA_WAVE_HZ == _W_ORACLE
    assert RH4_K_HZ == _K_ORACLE
    assert RH4_MEAN_DEPTH_M * FV3_GRAV == _GH0_ORACLE


def test_hand_computed_analytic_points_equator():
    """At lat=0 (cos=1, sin=0) the case-6 coefficients reduce, BY HAND:

        A(0) = w(2 Om + w)/2 - k^2/4
        B(0) = 2 (Om + w) k / 30 * (26 - 25) = (Om + w) k / 15
        C(0) = k^2 (5 - 6)/4 = -k^2/4
        gh(0, 0)      = gh0 + a^2 (A + B + C)
        u(0, 0)       = a w - a k            (== 0, since w == k)
        u(pi/4, 0)    = a w + a k            (cos(4 lon) = -1)
        v(lon, 0)     = 0                    (sin(lat) factor)
    """
    a, om, w, k, gh0 = (_A_ORACLE, _OM_ORACLE, _W_ORACLE, _K_ORACLE,
                        _GH0_ORACLE)
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


def test_hand_computed_analytic_point_off_equator():
    """lat = pi/3, lon = pi/8 — the latitude powers are LIVE here, in a
    DIFFERENT factoring from the module (cos^{2R} collapsed to explicit
    products, the A bracket expanded to c^8 (5 c^2 + 26) - 32 c^6, the
    B square expanded to 25 c^2), so a wrong exponent or coefficient in
    either factoring diverges (codex c6 r1 #8: the equator points alone
    leave every latitude power untested)."""
    a, om, w, k, gh0 = (_A_ORACLE, _OM_ORACLE, _W_ORACLE, _K_ORACLE,
                        _GH0_ORACLE)
    lam, phi = np.pi / 8.0, np.pi / 3.0
    c = np.cos(phi)
    s = np.sin(phi)
    c2 = c * c
    c4 = c2 * c2
    c6 = c4 * c2
    c8 = c4 * c4
    # A = w(2Om+w)/2 c^2 + k^2/4 [c^8 (5 c^2 + 26) - 32 c^6]
    a_t = 0.5 * w * (2.0 * om + w) * c2 \
        + 0.25 * k * k * (c8 * (5.0 * c2 + 26.0) - 32.0 * c6)
    # B = 2(Om+w)k/30 c^4 [26 - 25 c^2]
    b_t = (om + w) * k / 15.0 * c4 * (26.0 - 25.0 * c2)
    # C = k^2/4 c^8 [5 c^2 - 6]
    c_t = 0.25 * k * k * c8 * (5.0 * c2 - 6.0)
    gh_hand = gh0 + a * a * (a_t + b_t * np.cos(4.0 * lam)
                             + c_t * np.cos(8.0 * lam))
    gh_mod = rossby_haurwitz_4_geopotential(lam, phi, radius=a, omega=om,
                                            gh0=gh0)
    assert gh_mod == pytest.approx(gh_hand, rel=1e-13)

    # winds, independent factoring: c^{R-1} written as c*c*c
    u_hand = a * w * c + a * k * (c * c2) * (4.0 * s * s - c2) \
        * np.cos(4.0 * lam)
    v_hand = -a * k * 4.0 * s * (c * c2) * np.sin(4.0 * lam)
    u_mod, v_mod = rossby_haurwitz_4_winds(lam, phi, radius=a)
    assert u_mod == pytest.approx(u_hand, rel=1e-13)
    assert v_mod == pytest.approx(v_hand, rel=1e-13)
    assert abs(v_hand) > 10.0      # the point actually exercises v


def test_reference_canvas_matches_gate_contract():
    """Runner and gate must agree on the canvas, or the contract check
    rejects every legitimate npz."""
    lat_r, lon_r = runner.reference_canvas()
    lat_g, lon_g = gate._canvas()
    np.testing.assert_allclose(lat_r, lat_g, atol=0)
    np.testing.assert_allclose(lon_r, lon_g, atol=0)
    assert lon_r[0] == 0.5 and lon_r[-1] == 359.5      # T-cell centres
    assert lat_r[0] == -90.0 and lat_r[-1] == 90.0


# The case-6 stage configuration the runner must hand the stepper,
# written as LITERALS from the authorities — NOT from SW_CFG_CASE8, so
# a drift in that shared block trips here (codex c6 r1 #7):
#   C48.sw.case6.alpha0.duo.hord8/rundir/logfile.000000.out:365-369
#     (hords all 8), :401-417 (DDDMP=D2_BG=D4_BG=KE_BG=0, NORD=2),
#     input.nml do_vort_damp=.false. -> damp_v=0 (dyn_core.F90:761-765)
#   dyn_core.F90:757  nord_v = min(2, NORD) = 2
_CASE6_SW_CFG_LITERAL = {
    "hord_tr": 8, "hord_vt": 8, "hord_tm": 8, "hord_dp": 8,
    "hord_mt": 8, "nord_v": 2, "damp_v": 0.0,
    "dddmp": 0.0, "d2_bg": 0.0, "d4_bg": 0.0, "nord": 2,
}


def test_runner_deck_constants_pin_the_resolved_echo():
    """The deck values this port claims to match (logfile echo),
    against literals — including the shared case-8 block, whose only
    legitimate difference is d4_bg (0.12 there, 0 here)."""
    assert runner.CASE6_D4_BG == 0.0
    assert runner.CASE6_DT_ATMOS_S == 1200.0
    assert runner.CASE6_N_SPLIT == 7
    assert {**SW_CFG_CASE8, "d4_bg": 0.0} == _CASE6_SW_CFG_LITERAL
    assert SW_CFG_CASE8["d4_bg"] == 0.12       # the case-8 del-6 bg


def test_runner_threads_deck_config_to_stepper(tmp_path, monkeypatch):
    """Run the runner's main() at C12 with a RECORDING stub in place of
    advance_duo_outer_step: proves the deck defaults actually ARRIVE at
    the stepper call (dt_atmos, n_split, d_ext, every sw_cfg key), not
    merely that constants exist (codex c6 r1 #7).  The stub returns the
    states unchanged, so no acoustic step runs."""
    import legoesm.core.fv3_native_duo_stepper as stepper_mod

    seen = []

    def _stub(ctx, states, dt_atmos, n_split, d_ext=None, sw_cfg=None):
        seen.append({"dt_atmos": dt_atmos, "n_split": n_split,
                     "d_ext": d_ext, "sw_cfg": dict(sw_cfg)})
        return states

    monkeypatch.setattr(stepper_mod, "advance_duo_outer_step", _stub)
    out = tmp_path / "thread.npz"
    monkeypatch.setattr("sys.argv", ["run_duo_stepper_case6.py",
                                     "--n", "12", "--days", "1",
                                     "--out", str(out)])
    runner.main()
    assert len(seen) == 72                     # 86400 / 1200 blocks
    for call in seen:
        assert call["dt_atmos"] == 1200.0
        assert call["n_split"] == 7
        assert call["d_ext"] == 0.0
        assert call["sw_cfg"] == _CASE6_SW_CFG_LITERAL
    z = np.load(out, allow_pickle=True)
    assert int(z["requested_days"]) == 1
    assert float(z["dt_atmos"]) == 1200.0
    assert int(z["n_split"]) == 7


# ---------------------------------------------------------------------
# gate: pure scoring math
# ---------------------------------------------------------------------

def _toy(run_gh, ref_gh, run_u, ref_u, run_v, ref_v):
    # np.array (unconditional COPY), not np.asarray: callers pass the
    # SAME array for run and ref in the zero-diff case, and asarray
    # would alias them -- a later in-place perturbation of run would
    # then silently change ref too and the non-vacuity check would
    # assert on a diff of exactly 0 (caught live by job 9356452).
    run = {"gh": np.array(run_gh, dtype=float),
           "u": np.array(run_u, dtype=float),
           "v": np.array(run_v, dtype=float)}
    ref = {"gh": np.array(ref_gh, dtype=float),
           "u": np.array(ref_u, dtype=float),
           "v": np.array(ref_v, dtype=float)}
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


def test_score_day_v_uses_the_vector_norm():
    """||u_ref|| >> ||v_ref|| with a dv-only error: normalising v by its
    OWN tiny norm would report ~1000x; the vector norm keeps it ~1e-2.
    Pins the denominator choice (codex c6 r1 #8)."""
    lat = np.array([0.0, 0.0])
    ref_u = np.full((2, 3), 100.0)
    ref_v = np.full((2, 3), 1.0e-3)
    run, ref = _toy(np.full((2, 3), 2.0), np.full((2, 3), 2.0),
                    ref_u, ref_u, ref_v + 1.0, ref_v)
    s = gate.score_day(run, ref, lat)
    # hand: sqrt(6*1) / sqrt(6*1e4 + 6*1e-6)
    hand = np.sqrt(6.0) / np.sqrt(6.0e4 + 6.0e-6)
    assert s["v"]["rel_l2"] == pytest.approx(hand, rel=1e-12)
    assert s["v"]["rel_l2"] < 0.02          # NOT ~1000 (own-norm bug)
    assert s["u"]["rel_l2"] == 0.0


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
                lat=lat, lon=lon,
                # the runner's config record, deck values
                n=np.array(48), dt_atmos=np.array(1200.0),
                n_split=np.array(7), d_ext=np.array(0.0),
                d4_bg=np.array(0.0), k2e_nord=np.array(2),
                ext_exclude=np.array(""),
                oracle_conventions=np.array(True),
                requested_days=np.array(1))
    base.update(over)
    path = tmp_path / "run.npz"
    np.savez_compressed(path, **base)
    return str(path)


def test_contract_accepts_canonical(tmp_path):
    out = gate.load_run(_write_npz(tmp_path))
    assert out["gh"].shape == (2, 181, 360)
    assert gate.check_deck_record(out["record"]) == []
    gate.check_coverage(out["times_days"], out["record"])  # no raise


def test_contract_rejects_polar_sentinel(tmp_path):
    """The exact codex c6 r1 #3 scenario: a finite BIG_NUMBER-class
    corruption confined to the two pole rows.  cos-lat weighting alone
    would score it ~0; the plausibility band rejects the file before
    any score is formed."""
    gh = np.full((2, 181, 360), 7.8e4)
    gh[:, 0, :] = 1.0e8
    gh[:, -1, :] = 1.0e8
    with pytest.raises(gate.ContractError, match="plausibility band"):
        gate.load_run(_write_npz(tmp_path, gh=gh))
    u = np.zeros((2, 181, 360))
    u[:, 0, :] = 1.0e8
    with pytest.raises(gate.ContractError, match="plausibility bound"):
        gate.load_run(_write_npz(tmp_path, u=u))


def test_deck_record_mismatch_and_missing():
    rec = {key: val for key, val in gate.DECK_RECORD.items()}
    assert gate.check_deck_record(rec) == []
    bad = dict(rec)
    bad["dt_atmos"] = 450.0
    problems = gate.check_deck_record(bad)
    assert len(problems) == 1 and "dt_atmos" in problems[0]
    del bad["dt_atmos"]
    problems = gate.check_deck_record(bad)
    assert len(problems) == 1 and "not recorded" in problems[0]


def test_coverage_rules():
    rec = {"requested_days": 2}
    gate.check_coverage(np.array([0.0, 1.0, 2.0]), rec)      # ok
    with pytest.raises(gate.ContractError, match="truncated or sparse"):
        gate.check_coverage(np.array([0.0, 1.0]), rec)       # timeout cut
    with pytest.raises(gate.ContractError, match="truncated or sparse"):
        gate.check_coverage(np.array([0.0, 2.0]), rec)       # sparse
    with pytest.raises(gate.ContractError, match="requested_days"):
        gate.check_coverage(np.array([0.0]), {})             # no record
    ic = {"requested_days": 0}
    with pytest.raises(gate.ContractError, match="allow-ic-only"):
        gate.check_coverage(np.array([0.0]), ic)
    gate.check_coverage(np.array([0.0]), ic, allow_ic_only=True)


def test_verdict_closed_form():
    """Enforcement is finite-AND-<=: NaN scores or exceeded bounds FAIL;
    NaN/negative bounds refuse outright (codex c6 r1 #4/#5)."""
    def _scores(gh=1e-3, u=1e-2, v=1e-2):
        one = {"gh": {"rel_l2": gh, "rel_l2_cosw": gh, "max_abs": 1.0},
               "u": {"rel_l2": u, "rel_l2_cosw": u, "max_abs": 1.0},
               "v": {"rel_l2": v, "rel_l2_cosw": v, "max_abs": 1.0}}
        return {0.0: one}

    assert gate.verdict(_scores(), 2e-3, 5e-2) is True
    assert gate.verdict(_scores(gh=3e-3), 2e-3, 5e-2) is False
    assert gate.verdict(_scores(v=6e-2), 2e-3, 5e-2) is False
    assert gate.verdict(_scores(gh=np.nan), 2e-3, 5e-2) is False
    with pytest.raises(ValueError, match="finite and > 0"):
        gate.verdict(_scores(), np.nan, 5e-2)
    with pytest.raises(ValueError, match="finite and > 0"):
        gate.verdict(_scores(), 2e-3, -1.0)


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


def test_nearest_map_honours_lon_offset_independently(c12):
    """INDEPENDENT nearest-centre construction (codex c6 r1 #6: using
    the map to build its own expectation is circular).  For a sample of
    canvas points on the 0.5-based canvas, brute-force the nearest cube
    centre from the raw context coordinates with test-local arithmetic
    and require the SAME index; then require the 0.5-based map to
    actually differ from the historical 0-based map somewhere."""
    ctx, _, nmap = c12
    n, ng = ctx["n"], ctx["ng"]
    sl = slice(ng, ng + n)
    lons = np.concatenate([np.asarray(ctx["gs6"][t]["agrid_lon"])[sl, sl]
                           .ravel() for t in range(6)])
    lats = np.concatenate([np.asarray(ctx["gs6"][t]["agrid_lat"])[sl, sl]
                           .ravel() for t in range(6)])
    cx = np.cos(lats) * np.cos(lons)
    cy = np.cos(lats) * np.sin(lons)
    cz = np.sin(lats)
    lat_deg, lon_deg = runner.reference_canvas()
    rng = np.random.default_rng(7)
    for i, j in zip(rng.integers(1, 180, 12), rng.integers(0, 360, 12)):
        la = np.deg2rad(lat_deg[i])
        lo = np.deg2rad(lon_deg[j])
        px = np.cos(la) * np.cos(lo)
        py = np.cos(la) * np.sin(lo)
        pz = np.sin(la)
        dots = cx * px + cy * py + cz * pz
        assert nmap[i, j] == int(np.argmax(dots)), (i, j)
    w2 = _load("run_duo_stepper_w2")
    nmap_hist = w2.build_nearest_map(ctx)           # 0..359 canvas
    frac = float(np.mean(nmap != nmap_hist))
    # a 0.5 deg shift moves points within 0.5 deg of a C12 Voronoi
    # boundary (~0.5/7.5 of the canvas); an ignored lon_deg gives 0.0
    assert frac > 0.02, (
        f"0.5-based map differs from the 0-based map on only "
        f"{frac:.3f} of the canvas — lon_deg is being ignored")


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
