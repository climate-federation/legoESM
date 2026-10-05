"""Controls for the round-93 ORCA2 rung-0 RHS walk."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round93_rhs_walk as gate,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round93_slow_acquisition import (
    check_record as slow_record,
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


def test_slow_record_reader_uses_its_self_describing_header(tmp_path: Path) -> None:
    path = tmp_path / "oracle_r93_slow_rank0000_kt00000001.bin"
    payload = bytearray(slow_record.MAGIC)
    payload.extend(struct.pack(
        "=16i", 1, 1, 1, 3, 3, 0, 94, 152, 1, 1,
        3, 3, 92, 150, 64, len(slow_record.NAMES)))
    for index, name in enumerate(reversed(slow_record.NAMES)):
        n1, n2 = ((90, 148) if index % 2 == 0 else (94, 152))
        values = np.zeros(n1 * n2, dtype=np.float64).tobytes()
        payload.extend(name.encode("ascii").ljust(16, b" "))
        payload.extend(struct.pack("=4i", 2, n1, n2, 1))
        payload.extend(values)
    path.write_bytes(payload)
    record = slow_record.read_record(path, include_owned_values=True)
    assert set(record["fields"]) == set(slow_record.NAMES)
    assert record["rank"] == 0
    assert all(value.shape == (148, 90)
               for value in record["owned_values"].values())
    assert set(map(tuple, record["field_shapes"].values())) == {
        (148, 90), (152, 94),
    }
    for plant in (
        "header", "field-name", "field-dims", "truncation", "swapped-rank",
    ):
        if plant == "swapped-rank":
            assert slow_record.read_record(path, plant)["rank"] == 1
        else:
            with pytest.raises(slow_record.Refusal):
                slow_record.read_record(path, plant)
