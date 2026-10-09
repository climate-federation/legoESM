"""SMT-RUNGS round 4: the BBL gate-open statistic is non-vacuous and signed right."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "scripts/validate/ocean_fidelity/testcases"))
import nemo_testcase_l1_vortex_smtrungs_round4_smt6_survey as S  # noqa: E402


def _row():
    # one row of 4 wet columns: bottom levels 1,1,2,2 -> one sloped face (1|2)
    mb = np.array([[1, 1, 2, 2]])
    g1 = np.array([250.0, 750.0])
    mgu, mgv = S.face_slopes(mb, g1)
    wet = np.ones((1, 4), bool)
    return mb, mgu, mgv, wet[:, 1:] & wet[:, :-1], np.zeros((0, 4), bool)


def test_slope_sign_is_deeper_neighbour():
    _, mgu, _, _, _ = _row()
    assert mgu.tolist() == [[0, 1, 0]]


def test_stable_stratification_leaves_gate_closed():
    _, mgu, mgv, uw, vw = _row()
    Tb = np.array([[8.0, 8.0, 4.0, 4.0]])      # colder (denser) on deep side
    assert S.open_faces(Tb, mgu, mgv, uw, vw) == (0, 0)


def test_planted_dense_shelf_opens_gate_and_is_counted():
    _, mgu, mgv, uw, vw = _row()
    Tb = np.array([[3.0, 3.0, 4.0, 4.0]])      # shelf colder (denser) than deep
    assert S.open_faces(Tb, mgu, mgv, uw, vw) == (1, 0)


def test_equal_temperature_is_closed_like_fortran_sign_positive_zero():
    _, mgu, mgv, uw, vw = _row()
    Tb = np.full((1, 4), 5.0)
    assert S.open_faces(Tb, mgu, mgv, uw, vw) == (0, 0)


def test_land_columns_get_bottom_index_one():
    assert S.bottom_index(np.array([0, 3])).tolist() == [1, 3]
