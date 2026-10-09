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


def test_card_selects_geometric_and_the_insitu_arm_moves_the_rhs_row():
    card = _run("rhs")
    insitu = _run("rhs", eos_depth="insitu")
    assert "geometric" in card["eos_depth_arm"]
    assert insitu["eos_depth_arm"] == "insitu"
    assert _max(insitu, ".u") > 1e-11 > 1e-15 > _max(card, ".u")


def _stage_val(o, n, f, k="max_abs"):
    return next(r[k] for r in o["per_kt"][0]["stages"][n - 1]["rows"]
                if r["name"].endswith("." + f))


def test_stage_entry_has_no_consumer_with_advection_off(monkeypatch):
    # One wet level: the correction (stprk3_stg.f90:413-414) overwrites u, v
    # with uu_b, vv_b, and with ln_traadv_OFF a stage restarts T, S from Kbb
    # (:503-505), so a stage-entry velocity reaches no output.  The FCT2
    # card plant shows the seeding itself still delivers the bump.
    clean = _run("stages")
    ent = _run("stages", plant="stage_entry")
    ext = _run("stages", plant="external")
    assert _stage_val(ent, 2, "T", "rms") == _stage_val(clean, 2, "T", "rms")
    assert _stage_val(ent, 2, "u") == _stage_val(clean, 2, "u")
    assert _stage_val(ext, 1, "ssh") > 1e-4 > _stage_val(clean, 1, "ssh")
    _plant_card(monkeypatch, tracer_advection="fct2")
    assert (_stage_val(_run("stages", plant="stage_entry"), 2, "T", "rms")
            != _stage_val(_run("stages"), 2, "T", "rms"))


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


def _plant_card(monkeypatch, **fields):
    import legoesm.ocean.fidelity.nemo_testcase_recipe as rec
    real = rec.build_tsunami_zco_card

    def planted():
        c = real()
        m = c.recipe.model_config._replace(**fields)
        return c._replace(recipe=c.recipe._replace(model_config=m))

    monkeypatch.setattr(rec, "build_tsunami_zco_card", planted)


def _b6_errors(monkeypatch):
    # A stale (un-awaited) observer read breaks this: the first measurement
    # gave ratios 1.5 and 2.0 before the effects barrier was added.
    monkeypatch.setattr(lad, "B6_KTS", (2,))
    out = lad.run(ROOT, arm="b6_scaling", kt_max=2, allow_dirty=True,
                  eos_depth="geometric")
    k = out["per_kt"][0]
    return {r["lambda"]: r["max_abs_card_minus_prediction"] for r in k["lambdas"]}


def test_b6_off_puts_every_lambda_on_the_floor(monkeypatch):
    assert max(_b6_errors(monkeypatch).values()) < 1e-17


def test_b6_up3_plant_is_quadratic_in_velocity(monkeypatch):
    _plant_card(monkeypatch, momentum_flux_scheme="nemo_up3",
                vertical_momentum_scheme="nemo_up3")
    e = _b6_errors(monkeypatch)
    assert e[0.0] < 1e-15 < 1e-9 < e[1.0]
    assert abs(e[1.0] / e[0.5] - 4.0) < 1e-6


def test_b7_off_keeps_tracers_bitwise_and_the_fct2_plant_moves_them(monkeypatch):
    clean = _run("given_entry")
    assert all(r["bit_identical"] for r in _rows(clean)
               if r["name"].endswith((".T", ".S")))
    _plant_card(monkeypatch, tracer_advection="fct2")
    assert _max(_run("given_entry"), ".T") > 1e-4


def test_record100_leaves_the_bar_at_kt15_on_the_j_seam():
    with pytest.raises(Exception):
        lad.run(ROOT, arm="independent", kt_max=11, allow_dirty=True)
    with pytest.raises(Exception):
        lad.run(ROOT, arm="record100", kt_max=101, allow_dirty=True)
    out = lad.run(ROOT, arm="record100", kt_max=16, allow_dirty=True)
    assert out["first_kt_over_bar"]["ssh"] == 15
    last = {r["name"].split(".")[-1]: r for r in out["per_kt"][-1]["rows"]}
    assert last["vv_b"]["max_abs_cell"][0] in (0, 200)
