from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round167_exit_depth_walk as gate,
)


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
        "compact_nonfinite_count": 42,
        "source_order": list(gate.SOURCE_ORDER),
        "registered_cells": {
            "count": 41,
            "candidate_depth_finite": True,
            "oracle_depth_finite": True,
            "oracle_inverse_finite": True,
            "depth_differing_cells": 41,
            "depth_replay_nonfinite": 0,
            "reciprocal_only_depth_differing_cells": 41,
        },
    }


def test_clean_report_passes_and_keeps_predictions_explicit():
    result = gate.classify(_report())
    assert result["status"] == "PASS_ROUND167_EXIT_DEPTH_WALK"
    assert set(result["prediction_dispositions"]) == {
        "R167-P1", "R167-P2", "R167-P3", "R167-P4",
    }


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_all_plants_fire(plant: str):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)


def test_nonfinite_score_retains_bit_and_finite_censuses():
    candidate = np.array([[0.0, np.inf], [-0.0, 3.0]])
    oracle = np.array([[0.0, 2.0], [0.0, 4.0]])
    row = gate._score(candidate, oracle, np.ones((2, 2), dtype=bool))
    assert row["candidate_nonfinite"] == 1
    assert row["oracle_nonfinite"] == 0
    assert row["differing_cells"] == 3
    assert row["finite_absolute_max"] == 1.0


def test_registered_depth_replay_is_nonvacuous():
    depth = np.array([[0.0, 2.0]])
    oracle_depth = np.array([[4.0, 2.0]])
    inverse = np.array([[np.inf, 0.5]])
    oracle_inverse = np.array([[0.25, 0.5]])
    summary = gate._registered_summary(
        depth, oracle_depth, inverse, oracle_inverse, np.ones((1, 2)))
    assert summary["count"] == 1
    assert summary["depth_differing_cells"] == 1
    assert summary["depth_replay_nonfinite"] == 0
    assert summary["depth_replay_differing_cells"] == 0


def test_kt8_trace_keeps_round166_graph_and_applies_association_afterward():
    source = gate.Path(gate.__file__).read_text()
    measure = source.split("def measure", 1)[1].split("\ndef main", 1)[0]
    assert "expose_barotropic_substeps=True" in measure
    assert "expose_barotropic_boundary_association=True" not in measure
    assert "_nemo_external_mode_boundary_association(" in measure
    assert "jnp.asarray(value)" in measure
