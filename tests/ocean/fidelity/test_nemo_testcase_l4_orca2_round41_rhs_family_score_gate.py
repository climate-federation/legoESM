"""Binding tests for the ORCA2 round-41 family scorer."""

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round41_rhs_family_score_gate as gate,
)


def _row(exact: bool) -> dict:
    return {
        "bit_exact": exact,
        "differing_cells": 0 if exact else 1,
        "absolute_max": 0.0 if exact else 1.0,
    }


def test_first_non_bit_uses_compiled_family_and_face_order():
    rows = {
        face: {family: _row(True) for family in gate.FAMILIES}
        for face in ("u", "v")
    }
    rows["v"]["hpg"] = _row(False)
    rows["u"]["ldf"] = _row(False)
    first = gate.first_non_bit(rows)
    assert first["family"] == "hpg"
    assert first["face"] == "v"


def test_residual_identity_is_bitwise_and_plantable():
    oracle = {
        family: np.arange(6, dtype=np.float64).reshape(2, 3)
        for family in gate.FAMILIES
    }
    residual = np.full((2, 3), 0.25, dtype=np.float64)
    candidate = {family: oracle[family] + residual for family in gate.FAMILIES}
    rows = gate.residual_identity(candidate, oracle)
    assert all(row["bit_exact"] for row in rows.values())
    planted = {name: value.copy() for name, value in oracle.items()}
    planted["ldf"][0, 1] = np.nextafter(planted["ldf"][0, 1], np.inf)
    assert not gate.residual_identity(candidate, planted)["ldf"]["bit_exact"]


def test_family_registry_matches_acquired_record_order():
    assert gate.FAMILIES == ("hpg", "ldf", "vor", "keg", "zad")
    assert tuple(gate.PART_KEYS) == gate.FAMILIES
