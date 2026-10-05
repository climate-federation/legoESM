"""Direct controls for the ORCA2 round-17 continuity localization."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
GATE = (REPO / "scripts/validate/ocean_fidelity/orca2_l4"
        / "nemo_testcase_l4_orca2_round17_continuity_du_gate.py")


def _gate():
    spec = importlib.util.spec_from_file_location("_r17_du_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case():
    oracle_full = np.array([
        [1.0, 1.000000000000001, 1.000000000000002],
        [3.0, 3.000000000000001, 3.000000000000002],
    ])
    right = oracle_full[:, 1:]
    oracle_left = oracle_full[:, :-1]
    oracle_du = right - oracle_left
    candidate_full = np.array(oracle_full, copy=True)
    candidate_full[:, 0] = np.nextafter(candidate_full[:, 0], np.inf)
    candidate_du = candidate_full[:, 1:] - candidate_full[:, :-1]
    return candidate_full, candidate_du, right, oracle_du


def test_localization_finds_only_the_unrecorded_west_face():
    gate = _gate()
    result = gate._localize(*_case(), np.ones((2, 2), dtype=bool))

    assert result["all_mismatches_on_west_edge"]
    assert result["column_histogram"] == {"0": 2}
    assert result["right_operand_on_support"]["bit_exact"]
    assert result["left_and_subtraction_support_identical"]
    assert result["mismatch_support_explained_by_left"]
    assert result["candidate_operand_replay"]["bit_exact"]
    assert result["oracle_inversion_replay"]["bit_exact"]


def test_interior_operand_difference_refutes_west_edge_localization():
    gate = _gate()
    candidate_full, _, right, oracle_du = _case()
    candidate_full[0, 1] = np.nextafter(candidate_full[0, 1], np.inf)
    candidate_du = candidate_full[:, 1:] - candidate_full[:, :-1]

    result = gate._localize(
        candidate_full, candidate_du, right, oracle_du,
        np.ones((2, 2), dtype=bool))

    assert not result["all_mismatches_on_west_edge"]


def test_one_ulp_left_face_plant_fires_locally():
    gate = _gate()
    candidate_full = np.array([
        [1.0, 1.000000000000001, 1.000000000000002]])
    candidate_du = candidate_full[:, 1:] - candidate_full[:, :-1]
    result = gate._localize(
        candidate_full, candidate_du, candidate_full[:, 1:], candidate_du,
        np.ones((1, 2), dtype=bool), plant=True)

    assert result["plant"]["fires"]
    assert result["plant"]["subtraction_movement"]["differing_cells"] == 1
