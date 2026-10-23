from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round328_dynzdf_gate as gate,
)


def _write_record(path: Path, rank: int) -> None:
    nx, ny, nz = 3, 2, 4
    fields = {}
    for index, name in enumerate(gate.EXPECTED_ORDER, 1):
        n3 = 1 if name in {"barotropic_u", "barotropic_v", "rCdU_bot", "rCdU_top", "utauU", "vtauV"} else nz
        fields[name] = np.full((nx, ny, n3), float(index), dtype="=f8", order="F")
    fields["u_bottom_diag_diag"] += 1.0
    fields["v_bottom_diag_diag"] += 1.0
    fields["u_top_diag_diag"] += 2.0
    fields["v_top_diag_diag"] += 2.0
    fields["corrected_u"] += 1.0
    fields["corrected_v"] += 1.0
    with path.open("wb") as stream:
        stream.write(gate.MAGIC.encode().ljust(16))
        stream.write(struct.pack("=15i", 1, 1, 3, 1, 2, 3, 3, rank, nx, ny, nz, 1, 1, 64, len(fields)))
        for name, values in fields.items():
            stream.write(name.encode().ljust(24))
            stream.write(struct.pack("=7i", 2 if values.shape[2] == 1 else 3,
                                     nx, ny, values.shape[2], 1, 1, 1))
            stream.write(values.tobytes(order="F"))


@pytest.fixture()
def record_dir(tmp_path: Path) -> Path:
    (tmp_path / "producer_commit.txt").write_text("a" * 40 + "\n")
    for rank in (0, 1):
        _write_record(tmp_path / gate.record_name(rank), rank)
    return tmp_path


def test_clean_records_pass(record_dir: Path) -> None:
    result = gate.run_gate(record_dir, "a" * 40)
    assert result["status"] == "PASS_R328_DYNZDF_STRUCTURE"


@pytest.mark.parametrize("plant", gate.PLANTS)
def test_every_plant_refuses(record_dir: Path, plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.run_gate(record_dir, "a" * 40, plant)
