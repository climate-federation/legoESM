"""Unit controls for the Round-84 source-ordered RHS walk."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import numpy as np

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases"
    / "nemo_testcase_l2_gyre_round84_rhs_walk.py"
)
SPEC = importlib.util.spec_from_file_location("round84_rhs_walk", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ROUND84 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUND84)


def test_source_order_accumulator_and_after_adv_identity():
    terms = [np.asarray([value], dtype=np.float64) for value in range(1, 11)]
    rows = jax.device_get(jax.jit(ROUND84.source_order_accumulators)(*terms))
    assert rows["after_hpg_u"][0] == 1.0
    assert rows["after_ldf_u"][0] == 4.0
    assert rows["after_vor_u"][0] == 9.0
    assert rows["after_keg_u"][0] == 16.0
    assert rows["after_zad_u"][0] == 25.0
    assert np.array_equal(rows["after_adv_u"], rows["after_zad_u"])


def test_first_nonbit_obeys_compiled_boundary_then_face_order():
    exact = {"bit_exact": True, "absolute_max": 0.0}
    rows = {
        face: {boundary: dict(exact) for boundary in ROUND84.BOUNDARIES}
        for face in ("u", "v")
    }
    rows["v"]["after_hpg"] = {"bit_exact": False, "absolute_max": 2.0}
    rows["u"]["after_ldf"] = {"bit_exact": False, "absolute_max": 3.0}
    assert ROUND84.first_nonbit(rows)["boundary"] == "after_hpg"
    assert ROUND84.first_nonbit(rows)["face"] == "v"


def test_one_ulp_control_moves_max_residual_cell_away():
    candidate = np.asarray([1.0, 2.0], dtype=np.float64)
    reference = np.asarray([1.0, 1.5], dtype=np.float64)
    changed, at = ROUND84._away_one_ulp(
        reference, candidate, np.ones(2, dtype=bool))
    assert at == (1,)
    assert changed[1] < reference[1]
    assert abs(candidate[1] - changed[1]) > abs(candidate[1] - reference[1])
