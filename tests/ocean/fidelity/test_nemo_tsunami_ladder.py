"""TSUNAMI round-3 ladder harness and offline HPG replay: run on kt = 1 of the
real RK3 record, and every substituted value is shown to move its rows when
planted (a dead hook would leave the planted run equal to the clean one)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_D = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
      "testcases")
sys.path.insert(0, str(_D))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _D / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lad = _load("nemo_testcase_l1_tsunami_ladder")
rep = _load("nemo_testcase_l1_tsunami_hpg_replay")
ROOT = lad.DEFAULT_ROOT
EVIDENCE = rep.DEFAULT_ROOT
pytestmark = pytest.mark.skipif(not ROOT.is_dir(), reason="RK3 evidence not mounted")


def _rows(out):
    k = out["per_kt"][0]
    return k.get("rows") or [r for s in k["stages"] for r in s["rows"]]


def _max(out, suffix):
    return max(r["max_abs"] for r in _rows(out) if r["name"].endswith(suffix))


def _run(arm, **kw):
    return lad.run(ROOT, arm=arm, kt_max=1, allow_dirty=True, **kw)


def test_unknown_arm_and_plant_are_refused():
    with pytest.raises(Exception):
        lad.run(ROOT, arm="nope", allow_dirty=True)
    with pytest.raises(Exception):
        lad.run(ROOT, arm="independent", plant="nope", allow_dirty=True)


def test_scaling_arm_refuses_without_the_geometric_arm():
    with pytest.raises(Exception, match="geometric"):
        _run("b6_scaling")


def test_independent_kt1_is_not_bit_identical_and_the_score_plant_moves_it():
    clean = _run("independent")
    planted = _run("independent", plant="score")
    assert clean["label"] == "INDEPENDENT"
    assert clean["per_kt"][0]["first_unequal"]["name"].endswith(".ssh")
    assert _max(clean, ".ssh") < 1e-6 < 1e-4 < _max(planted, ".ssh")


def test_given_entry_label_and_entry_plant():
    clean = _run("given_entry")
    planted = _run("given_entry", plant="entry")
    assert clean["label"].startswith("GIVEN-NEMO-ENTRY")
    assert _max(planted, ".ssh") > 1e-4 > _max(clean, ".ssh")


def test_geometric_arm_moves_the_pressure_gradient_row_and_is_recorded():
    insitu = _run("rhs")
    geo = _run("rhs", eos_depth="geometric")
    assert geo["eos_depth_arm"] == "geometric"
    assert "insitu" in insitu["eos_depth_arm"]
    assert _max(insitu, ".u") > 1e-11 > 1e-15 > _max(geo, ".u")


def test_stage_entry_and_external_plants_move_their_stage_rows():
    clean = _run("stages")
    ent = _run("stages", plant="stage_entry")
    ext = _run("stages", plant="external")
    s = lambda o, n, f, k="max_abs": next(  # noqa: E731
        r[k] for r in o["per_kt"][0]["stages"][n - 1]["rows"]
        if r["name"].endswith("." + f))
    # one wet level: the barotropic correction (stprk3_stg.F90:440-445) makes
    # every stage's u, v the external uu_b, vv_b, so a stage-entry velocity is
    # consumed ONLY by the tracer transport -- the plant must be read on T.
    assert s(ent, 2, "T", "rms") != s(clean, 2, "T", "rms")
    assert s(ent, 2, "u") == s(clean, 2, "u")
    assert s(ext, 1, "ssh") > 1e-4 > s(clean, 1, "ssh")


def test_forcing_plant_moves_the_nemo_forcing_arm():
    clean = _run("spgts")
    planted = _run("spgts", plant="forcing")
    assert clean["per_kt"][0]["own"]["first_unequal"]["name"].endswith("entry.zu_frc")
    assert (planted["per_kt"][0]["nemo"]["n_unequal_boundaries"]
            > clean["per_kt"][0]["nemo"]["n_unequal_boundaries"])


def test_replay_reproduces_nemo_rhs_bitwise_and_fails_when_planted():
    cr, _ = lad._tools()
    g = lad.read_step(cr, EVIDENCE / "p3", 1)
    ok = rep.validate_kt1(EVIDENCE, g)
    assert ok["u_bit_identical"] and ok["v_bit_identical"]
    for key in ("mu2", "rho0", "a0"):
        bad = rep.validate_kt1(EVIDENCE, g, perturb=key)
        assert not (bad["u_bit_identical"] and bad["v_bit_identical"]), key


def test_row_helper_reports_first_unequal_cell_and_stats():
    a = np.zeros((3, 4))
    b = a.copy()
    b[1, 2] = 2.0e-3
    r = lad.row("x", a, b)
    assert (r["first_unequal_cell"], r["n_unequal"], r["bit_identical"]) == ([1, 2], 1, False)
    assert r["max_abs"] == 2.0e-3 and r["status"] == "DEBT"
    assert lad.row("y", a, a)["bit_identical"]
