from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round168_exit_depth_operands as gate,
)


def _row(*, exact: bool = True, differing: int = 0) -> dict:
    return {"bit_exact": exact, "differing_cells": differing, "count": 41}


def _report() -> dict:
    return {
        "admission": {
            "rank_coverage": "exactly-once",
            "records": [
                {"icycle": 65, "groups": 2106},
                {"icycle": 65, "groups": 2106},
            ],
            "terminal_restart_comparisons": [{} for _ in range(20)],
        },
        "completed_kt": 7,
        "registered_count": 41,
        "source_order": list(gate.SOURCE_ORDER),
        "upstream_order": list(gate.UPSTREAM_ORDER),
        "registered_rows": {
            "raw_plus_oracle_face_ssh": _row(),
            "candidate_sum_replay": _row(),
            "oracle_eta_face_ssh_replay": _row(),
            "candidate_face_ssh": _row(exact=False, differing=41),
            "candidate_after_ssh_stencil": _row(exact=False, differing=41),
        },
        "first_upstream_nonbit": {"name": "transport_v"},
    }


def test_clean_report_passes_and_disposes_every_prediction():
    result = gate.classify(_report())
    assert result["status"] == "PASS_ROUND168_EXIT_DEPTH_OPERAND_WALK"
    assert set(result["prediction_dispositions"]) == {
        "R168-P1", "R168-P2", "R168-P3", "R168-P4", "R168-P5",
    }
    assert set(result["prediction_dispositions"].values()) == {"CONFIRMED"}


def test_failed_prediction_is_retained_not_refused():
    report = _report()
    report["registered_rows"]["oracle_eta_face_ssh_replay"]["bit_exact"] = False
    report["registered_rows"]["oracle_eta_face_ssh_replay"]["differing_cells"] = 7
    result = gate.classify(report)
    assert result["prediction_dispositions"]["R168-P3"] == "REFUTED"
    assert result["status"] == "PASS_ROUND168_EXIT_DEPTH_OPERAND_WALK"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_all_plants_fire(plant: str):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)


def test_registered_row_uses_bits_and_nonzero_values():
    candidate = np.array([[2.0, -0.0]])
    oracle = np.array([[2.0, 0.0]])
    row = gate._registered_row(candidate, oracle, np.ones((1, 2), dtype=bool))
    assert row["differing_cells"] == 1
    assert row["candidate_max"] == 2.0
    assert not row["bit_exact"]


def test_first_nonbit_preserves_source_order():
    rows = [
        {"name": "a", "bit_exact": True},
        {"name": "b", "bit_exact": False},
        {"name": "c", "bit_exact": False},
    ]
    assert gate._first_nonbit(rows)["name"] == "b"


def test_upstream_score_retains_nonfinite_census(monkeypatch):
    shape = (1, 1)
    trace = {
        name: np.zeros((2,) + shape)
        for name in (
            "u_entry", "v_entry", "eta_entry", "inverse_depth_u",
            "inverse_depth_v", "u_mid", "v_mid", "eta_mid",
            "transport_face_depth_u", "transport_face_depth_v",
            "transport_metric_u", "transport_metric_v", "continuity_du",
            "continuity_dv", "continuity_divergence", "eta_continuity",
        )
    }
    trace["transport_metric_u"][1, 0, 0] = np.inf
    oracle = {
        f"{prefix}_{name}": np.zeros(shape)
        for prefix in ("j001", "j002")
        for name in (
            "ua_new", "va_new", "ssha_e", "hur_e", "hvr_e", "ua_ext",
            "va_ext", "sshp2_mid", "hup2_e", "hvp2_e", "zhU", "zhV",
        )
    }
    monkeypatch.setattr(gate.r97, "_native_u", np.asarray)
    monkeypatch.setattr(gate.r97, "_native_v", np.asarray)
    rows = gate._score_upstream(
        trace, oracle, {face: np.ones(shape, dtype=bool)
                        for face in ("t", "u", "v")}, np.ones(shape), 1)
    transport = next(row for row in rows if row["name"] == "transport_u")
    assert transport["candidate_nonfinite"] == 1
    assert not transport["bit_exact"]
