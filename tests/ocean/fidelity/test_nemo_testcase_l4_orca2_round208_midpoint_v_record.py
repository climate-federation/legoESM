import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_round208_midpoint_v_acquisition/check_record.py"
)
SPEC = importlib.util.spec_from_file_location("round208_record", MODULE_PATH)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gate)


def _write_record(path, rank):
    nx, ny = 94, 152
    origin_i = 1 if rank == 0 else 91
    header = (1, 1, 1, rank, nx, ny, 1, origin_i, 1,
              3, 3, 92, 150, 64, len(gate.NAMES), 0)
    with path.open("wb") as handle:
        handle.write(gate.MAGIC)
        handle.write(struct.pack("=16i", *header))
        for name in gate.NAMES:
            handle.write(name.encode().ljust(16, b" "))
            handle.write(struct.pack("=4i", 2, nx, ny, 1))
            handle.write(np.zeros((nx, ny), dtype="=f8").tobytes(order="F"))


def _roots(tmp_path):
    root, baseline = tmp_path / "root", tmp_path / "baseline"
    root.mkdir()
    baseline.mkdir()
    for rank in (0, 1):
        _write_record(
            root / f"oracle_r208_midv_rank{rank:04d}_kt00000001.bin", rank)
        name = f"ORCA2_00000010_restart_{rank:04d}.nc"
        payload = f"restart-{rank}".encode()
        (root / name).write_bytes(payload)
        (baseline / name).write_bytes(payload)
    return root, baseline


def test_record_admits_exactly_once_rank_coverage(tmp_path):
    root, baseline = _roots(tmp_path)
    result = gate.run(root, baseline)
    assert result["status"] == "PASS_R208_MIDPOINT_V_RECORD"
    assert result["rank_coverage"] == "exactly-once"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_record_plants_fire(tmp_path, plant):
    root, baseline = _roots(tmp_path)
    with pytest.raises(gate.Refusal):
        gate.run(root, baseline, plant)
