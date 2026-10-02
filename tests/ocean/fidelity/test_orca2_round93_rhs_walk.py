"""Controls for the round-93 ORCA2 rung-0 RHS walk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round93_rhs_walk as gate,
)


RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round92/"
    "acquisition/orca2_rung0_rhs_ranked_10step_np2"
)


def test_first_nonbit_obeys_compiled_boundary_then_face_order() -> None:
    exact = {"bit_exact": True}
    rows = {
        face: {boundary: dict(exact) for boundary in gate.BOUNDARIES}
        for face in gate.FACES
    }
    rows["v"]["after_hpg"] = {"bit_exact": False, "absolute_max": 1.0}
    rows["u"]["after_ldf"] = {"bit_exact": False, "absolute_max": 2.0}
    first = gate.first_nonbit(rows)
    assert first is not None
    assert (first["boundary"], first["face"]) == ("after_hpg", "v")


def test_score_detects_one_ulp_and_keeps_argmax_location() -> None:
    reference = np.zeros((2, 3, 4), dtype=np.float64)
    candidate = reference.copy()
    candidate[1, 2, 3] = np.nextafter(0.0, np.float64(np.inf))
    row = gate.score(candidate, reference, np.ones_like(reference, dtype=bool))
    assert row["differing_cells"] == 1
    assert row["argmax_jik"] == [1, 2, 3]
    assert not row["bit_exact"]


def test_rank_complete_record_assembles_and_layout_plant_fires() -> None:
    if not RECORD.is_dir():
        pytest.skip("admitted round-92 RHS record is unavailable")
    arrays, census = gate.assemble_rhs(RECORD)
    assert tuple(arrays) == tuple(gate.check_record.NAMES)
    assert all(value.shape == (148, 180, 30) for value in arrays.values())
    assert census["coverage"] == "exactly-once"
    assert census["jpk_nonzero"] == 0
    with pytest.raises(gate.GateError, match="cover the domain exactly once"):
        gate.assemble_rhs(RECORD, plant="layout")
    with pytest.raises(gate.GateError, match="jpk accumulator slot"):
        gate.assemble_rhs(RECORD, plant="bottom-slot")
