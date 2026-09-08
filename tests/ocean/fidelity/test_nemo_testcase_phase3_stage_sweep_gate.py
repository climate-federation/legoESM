"""Direct non-vacuity tests for the phase-3 WS stage sweep gate."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PATH = Path("scripts/validate/ocean_fidelity/testcases/nemo_testcase_phase3_stage_sweep_gate.py")
SPEC = importlib.util.spec_from_file_location("nemo_stage_sweep_gate", PATH)
GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GATE)


def test_stage_score_planted_control_goes_red():
    oracle = np.zeros((2, 3), dtype=np.float64)
    mask = np.ones_like(oracle, dtype=bool)
    clean = GATE.score("clean", oracle, oracle.copy(), mask)
    planted = GATE.score("plant", oracle, oracle.copy(), mask, plant=True)
    assert clean["status"] == "AT-BAR"
    assert planted["status"] == "DEBT"
    assert planted["absolute_max"] == 1.0


def test_owner_label_requires_scale_and_bar_clearance():
    faithful = {"absolute_max": 4.0, "normalized_max_abs": 4.0}
    control = {"absolute_max": 3.0, "normalized_max_abs": 3.0}
    plausible = GATE.classify_arm(faithful, control, {"absolute_max": 1.0}, improving=True)
    assert plausible["classification"] == "PLAUSIBLE_CONTRIBUTOR_NOT_OWNER"
    control["absolute_max"] = 0.0
    control["normalized_max_abs"] = 0.0
    confirmed = GATE.classify_arm(faithful, control, {"absolute_max": 4.0}, improving=True)
    assert confirmed["classification"] == "CONFIRMED"


def test_operand_planted_control_goes_red_on_a_tracer_field():
    oracle = np.full((2, 3, 4), 10.0)
    mask = np.ones_like(oracle, dtype=bool)
    clean = GATE.score("operand", oracle, oracle.copy(), mask, quantity="T")
    planted = GATE.score("operand", oracle, oracle.copy(), mask, quantity="T", plant=True)
    assert clean["status"] == "AT-BAR" and clean["exact"]
    assert planted["status"] == "DEBT"
    assert planted["absolute_max"] == 1.0


def _synthetic_row_mesh(ni=6, nz=5, dz=10.0, dx=1000.0):
    tmask = np.ones((ni, nz))
    tmask[:, -1] = 0.0                       # NEMO bottom dummy level
    umask = tmask.copy()
    umask[-1] = 0.0
    e3t_0 = np.full((ni, nz), dz)
    gdept_1d = (np.arange(nz) + 0.5) * dz
    e3w_1d = np.concatenate([[2.0 * gdept_1d[0]], np.diff(gdept_1d)])
    ht_0 = np.sum(e3t_0 * tmask, axis=-1)
    return {
        "tmask": tmask, "umask": umask, "e3t_0": e3t_0, "e3u_0": e3t_0.copy(),
        "e3w_1d": e3w_1d, "gdept_1d": gdept_1d, "e1u": np.full(ni, dx),
        "r1_ht_0": 1.0 / ht_0, "hu_0": ht_0.copy(),
    }


def test_hpg_sco_replay_is_zero_for_uniform_density_and_moves_on_a_planted_anomaly():
    mesh = _synthetic_row_mesh()
    ni, nz = mesh["tmask"].shape
    rho0, g = 1026.0, 9.80665
    T = np.full((ni, nz), 10.0)
    S = np.full((ni, nz), 35.0)
    ssh = np.zeros(ni)

    def linear_eos(T_, S_, p):     # rho = rho0*(1 - 2e-4*(T-10))
        return rho0 * (1.0 - 2.0e-4 * (np.asarray(T_) - 10.0))

    rest = GATE.hpg_sco_row(mesh, T, S, ssh, linear_eos, g, rho0, rho0 * g)
    assert np.max(np.abs(rest)) == 0.0
    # Tilted free surface with uniform density: zhpi and zuap telescope to the
    # exact -g*rhd*d(ssh)/dx only when rhd != 0; with rhd == 0 both vanish.
    tilted = GATE.hpg_sco_row(mesh, T, S, np.linspace(0.0, 0.1, ni), linear_eos, g, rho0, rho0 * g)
    assert np.max(np.abs(tilted)) == 0.0
    # Planted anomaly: one warm column -> nonzero trend on its two faces only.
    T_plant = T.copy()
    T_plant[2, :] = 11.0
    moved = GATE.hpg_sco_row(mesh, T_plant, S, ssh, linear_eos, g, rho0, rho0 * g)
    faces = np.nonzero(np.abs(moved).max(axis=1) > 0.0)[0]
    assert set(faces.tolist()) == {1, 2}
    # Sign: warm (light) column 2 lowers pressure there, so the face west of it
    # (1) accelerates toward it (+x) and the face east (2) away (-x): NEMO's
    # trend is -(1/rho0) dp/dx on the eastward-positive u.
    assert moved[1, 0] > 0.0 and moved[2, 0] < 0.0
    # Cumulative-in-depth structure: |trend| grows with k below the anomaly top.
    assert np.all(np.diff(np.abs(moved[1, :nz - 1])) > 0.0)


def test_remove_depth_mean_is_exact_and_thickness_weighted():
    values = np.array([[[1.0, 3.0, 5.0]]])
    thickness = np.array([[[1.0, 1.0, 2.0]]])
    mask = np.ones_like(values, dtype=bool)
    out = GATE.remove_depth_mean(values, thickness, mask)
    mean = (1.0 + 3.0 + 10.0) / 4.0
    np.testing.assert_allclose(out, values - mean, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.sum(out * thickness), 0.0, atol=1e-14)


def test_require_planted_is_fail_closed():
    rows = [{"name": "x", "status": "DEBT", "absolute_max": 1.0},
            {"name": "y", "status": "DEBT", "absolute_max": 3.0e-12}]
    GATE.require_planted(rows, "x")
    with pytest.raises(GATE.GateError, match="did not land"):
        GATE.require_planted(rows, "y")          # DEBT but no +1.0 visible
    with pytest.raises(GATE.GateError, match="missing"):
        GATE.require_planted(rows, "z")


LOCK_ROOT = GATE.ROOTS["LOCK_EXCHANGE-zco"]
needs_oracle = pytest.mark.skipif(
    not (LOCK_ROOT / "oracle_stage_kt00000001_s1.bin").is_file(),
    reason="LOCK stage dumps absent")


@needs_oracle
def test_planted_stage_control_exits_nonzero_end_to_end(tmp_path):
    """The REAL LOCK sweep with --plant-stage: exit 1 (DEBT), the planted
    stage-1 row carries the +1.0 and is listed in failed_rows."""
    out = tmp_path / "planted.json"
    code = GATE.main(["LOCK_EXCHANGE-zco", "--plant-stage", "--allow-dirty",
                      "--output", str(out)])
    assert code == 1
    report = json.loads(out.read_text())
    name = "LOCK_EXCHANGE-zco.kt1.stage1.faithful.instantaneous_u"
    row = next(row for row in report["rows"] if row["name"] == name)
    assert row["status"] == "DEBT" and row["absolute_max"] >= 0.5
    assert name in report["failed_rows"]
    assert report["controls"] == {
        "plant_stage": True, "plant_operand": False, "plant_prediction": False}
    assert report["status"] == ("DEBT" if report["failed_rows"] else "AT-BAR")


@needs_oracle
def test_prediction_plant_is_fail_closed(tmp_path):
    """--plant-prediction inflates the frozen H2 numbers 1000x and REQUIRES the
    stage-3 refutation predicate to flip.  On LOCK that predicate does not
    exist, so the planted run must exit 2 rather than silently pass."""
    assert GATE.main(["LOCK_EXCHANGE-zco", "--plant-prediction", "--allow-dirty",
                      "--output", str(tmp_path / "planted.json")]) == 2
    plain = tmp_path / "plain.json"
    # LOCK reaches the 1e-15 bar at every kt=1 stage and at kt=2 since the
    # UP3 upwind-selector fix (dynadv_up3.F90:166-170); a DEBT exit here is a
    # regression, not the expected state.
    assert GATE.main(["LOCK_EXCHANGE-zco", "--allow-dirty",
                      "--output", str(plain)]) == 0
    check = json.loads(plain.read_text())["preregistered_prediction_check"]
    assert check and all(entry["status"] == "MET" for entry in check.values())


@needs_oracle
def test_undetected_plant_exits_two(monkeypatch):
    """If the scorer stops seeing plants, the planted run must exit 2, not 1."""
    real = GATE.score

    def blind(name, oracle, candidate, mask, *, plant=False, quantity="u"):
        return real(name, oracle, candidate, mask, plant=False, quantity=quantity)

    monkeypatch.setattr(GATE, "score", blind)
    assert GATE.main(["LOCK_EXCHANGE-zco", "--plant-stage", "--allow-dirty"]) == 2


def test_selector_prediction_s1_s2_and_the_plant_flip():
    """The stage-3 baroclinic round's frozen predicates (S1 OVERFLOW, S2 LOCK)
    read MET on the measured numbers and NOT-MET when the faithful stage-3
    residual is planted 1000x (the --plant-prediction control), or when the
    legacy arm stops reproducing the pre-fix debt."""
    def rows_for(case, faithful3, legacy3, faithful2=None, legacy2=None):
        out = []
        for stage, f, l in ((2, faithful2, legacy2), (3, faithful3, legacy3)):
            if f is None:
                continue
            out.append({"name": f"{case}.kt1.stage{stage}.faithful.baroclinic_u",
                        "absolute_max": f})
            out.append({"name": f"{case}.kt1.stage{stage}."
                                "legacy_up3_transport_sign_selector.baroclinic_u",
                        "absolute_max": l})
        return out

    arms = {"legacy_up3_transport_sign_selector": None}
    ov = GATE.preregistered_prediction_check(
        "OVERFLOW-zps", {}, [], rows_for("OVERFLOW-zps", 4.5517e-10, 2.598798e-07),
        None, None, arms)
    assert ov["S1_up3_selector_owns_the_stage3_baroclinic_u"]["status"] == "MET"
    planted = GATE.preregistered_prediction_check(
        "OVERFLOW-zps", {}, [], rows_for("OVERFLOW-zps", 4.5517e-10, 2.598798e-07),
        None, None, arms, plant_selector=True)
    assert planted["S1_up3_selector_owns_the_stage3_baroclinic_u"]["status"] == "NOT-MET"
    stale_arm = GATE.preregistered_prediction_check(
        "OVERFLOW-zps", {}, [], rows_for("OVERFLOW-zps", 4.5517e-10, 1.0e-07),
        None, None, arms)
    assert stale_arm["S1_up3_selector_owns_the_stage3_baroclinic_u"]["status"] == "NOT-MET"
    lock = GATE.preregistered_prediction_check(
        "LOCK_EXCHANGE-zco", {}, [],
        rows_for("LOCK_EXCHANGE-zco", 2.3e-17, 2.138804e-10, 2.9e-17, 9.765645e-11),
        None, None, arms)
    assert lock["S2_up3_selector_owns_the_LOCK_stage_debt"]["status"] == "MET"
    lock_bad = GATE.preregistered_prediction_check(
        "LOCK_EXCHANGE-zco", {}, [],
        rows_for("LOCK_EXCHANGE-zco", 2.3e-17, 2.138804e-10, 9.765645e-11, 9.765645e-11),
        None, None, arms)
    assert lock_bad["S2_up3_selector_owns_the_LOCK_stage_debt"]["status"] == "NOT-MET"
