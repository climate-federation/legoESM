"""Controls for the round-94 ORCA2 rung-0 slow-boundary walk."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round94_slow_walk as gate,
)


RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round93/"
    "acquisition/orca2_rung0_slow_ranked_10step_np2"
)


def test_first_nonbit_obeys_compiled_source_order() -> None:
    rows = {name: {"bit_exact": True} for name in gate.SOURCE_ORDER}
    rows["depth_v"] = {"bit_exact": False, "absolute_max": 1.0}
    rows["drag_u"] = {"bit_exact": False, "absolute_max": 2.0}
    first = gate.first_nonbit(rows)
    assert first is not None
    assert first["boundary"] == "depth_v"


def test_rank_complete_slow_record_assembles_and_layout_plant_fires() -> None:
    if not RECORD.is_dir():
        pytest.skip("round-93 slow record is unavailable")
    arrays, census = gate.assemble_slow(RECORD)
    assert tuple(arrays) == gate.check_record.NAMES
    assert all(value.shape == (148, 180) for value in arrays.values())
    assert census["coverage"] == "exactly-once"
    with pytest.raises(gate.GateError, match="cover the domain exactly once"):
        gate.assemble_slow(RECORD, plant="layout")


def test_source_order_covers_every_recorded_scientific_boundary() -> None:
    assert set(gate.SOURCE_ORDER) == set(gate.check_record.NAMES) - {"cd_u", "cd_v"}
