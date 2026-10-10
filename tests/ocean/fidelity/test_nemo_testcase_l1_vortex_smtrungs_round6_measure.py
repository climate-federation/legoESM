"""SMT-RUNGS round 6 measurement script: direct tests on the real records.

The records live under the evidence root; the tests that need them skip
(not pass) when the root is absent.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
           / "ocean_fidelity" / "testcases")
sys.path.insert(0, str(SCRIPTS))
m = importlib.import_module(
    "nemo_testcase_l1_vortex_smtrungs_round6_smt6_measure")

needs_records = pytest.mark.skipif(
    not (m.ROOTS["smt6b"] / "kt1_10" / "mesh_mask.nc").is_file(),
    reason="SMT-6b NEMO record not on this machine")


def test_touched_columns_marks_both_sides_of_every_open_face():
    ahu = np.zeros((3, 4), bool)
    ahv = np.zeros((3, 4), bool)
    ahu[1, 1] = True      # face between (1,1) and (1,2)
    ahv[0, 3] = True      # face between (0,3) and (1,3)
    got = m.touched_columns(ahu, ahv)
    assert {tuple(i) for i in np.argwhere(got)} == {
        (1, 1), (1, 2), (0, 3), (1, 3)}


@pytest.fixture(scope="module")
def cards():
    return m._cards()


@needs_records
def test_inputs_are_identical_and_the_anomaly_is_61_bottom_cells(cards):
    rep = m.inputs_section(cards)
    assert rep["status"] == "IDENTICAL"
    assert rep["anomaly"]["n_moved"] == 61
    assert rep["anomaly"]["n_moved_not_bottom"] == 0
    assert rep["anomaly"]["n_per_value"] == {"1.7": 56, "3.4": 5}
    assert rep["smt6b_dump_vs_card_initial_T_wet"]["n_unequal"] == 0
    assert rep["smt6b_dump_vs_smt5_dump_wet"]["n_unequal"] == 61


@needs_records
def test_bbl_replay_of_trabbl_equals_the_card_and_the_zero_sign_reading(cards):
    rep = m.bbl_replay_section(cards)
    b = rep["smt6b"]
    assert (b["replay_open_u"], b["replay_open_v"]) == (24, 24)
    assert (b["lego_open_u"], b["lego_open_v"]) == (24, 24)
    assert b["trend_T_unequal_cells"] == 0
    assert b["trend_T_nonzero_replay"] == b["trend_T_nonzero_lego"] == 64
    assert b["trend_S_nonzero_lego"] == 0
    c = rep["smt6"]
    assert c["trend_T_nonzero_replay"] == c["trend_T_nonzero_lego"] == 0
    # the refuted reading would add increments NEMO's own record excludes
    assert c["signed_zero_open_reading"]["max_step_increment_K"] > 1e-6
    gap = c["nemo_minus_lego_geothermal_only"]
    assert gap["n_signed_zero_open_cells"] > 0
    assert gap["max_abs_at_signed_zero_open_cells"] < 1e-14


@needs_records
def test_bbl_replay_is_not_vacuous_a_zeroed_card_gate_is_caught(
        cards, monkeypatch):
    from legoesm.ocean.physics import bbl_adv

    real = bbl_adv.nemo_bbl_diffusive_coefficients

    def closed(*a, **k):
        ahu, ahv = real(*a, **k)
        return ahu * 0.0, ahv * 0.0

    monkeypatch.setattr(bbl_adv, "nemo_bbl_diffusive_coefficients", closed)
    rep = m.bbl_replay_section({"smt6": cards["smt6"],
                                "smt6b": cards["smt6b"],
                                "smt5": cards["smt5"]})
    assert rep["smt6b"]["trend_T_unequal_cells"] == 64
    assert rep["smt6b"]["lego_open_u"] == 0
