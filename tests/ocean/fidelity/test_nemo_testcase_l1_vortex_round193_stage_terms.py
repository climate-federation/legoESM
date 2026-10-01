"""Fail-closed controls for the VORTEX round-193 stage-term reader."""
from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_round193_stage_terms.py"
SPEC = importlib.util.spec_from_file_location("round193_stage_terms", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _group(name, rank, nx, ny, nz, value):
    count = nx * ny * (nz if rank == 3 else 1)
    return (name.encode().ljust(16) + struct.pack("=4i", rank, nx, ny, nz)
            + np.full(count, value, dtype=np.float64).tobytes())


def _record(root: Path, stage=2, *, omit=None):
    checker_spec = importlib.util.spec_from_file_location(
        "checker", ROOT / "scripts/validate/ocean_fidelity/testcases/"
        "nemo_testcase_l1_vortex/check_records.py")
    checker = importlib.util.module_from_spec(checker_spec)
    checker_spec.loader.exec_module(checker)
    names = list(checker._STAGE_TERM_BY_STAGE[stage])
    if omit:
        names.remove(omit)
    nx, ny, nz = 9, 8, 3
    body = b"".join(_group(name, 2 if name == "ssh_kmm" else 3,
                           nx, ny, 1 if name == "ssh_kmm" else nz, i)
                    for i, name in enumerate(names))
    header = struct.pack("=15i", 1, 1, stage, 1, 2, 3, 4, nx, ny, nz,
                         len(names), 0, 0, 0, 64)
    path = root / f"oracle_stage_terms_kt00000001_s{stage}.bin"
    path.write_bytes(b"NEMO_L1_STGTRM1 " + header + body)
    return path


def test_reader_returns_named_interior_groups(tmp_path):
    _record(tmp_path, 2)
    groups = gate.read_stage_terms(tmp_path, 2)
    assert groups["hpg_u"].shape == (4, 5, 3)
    assert groups["ssh_kmm"].shape == (4, 5)


def test_reader_refuses_a_missing_required_group(tmp_path):
    _record(tmp_path, 2, omit="zad_u")
    with pytest.raises(Exception, match="missing group"):
        gate.read_stage_terms(tmp_path, 2)


def test_reader_preserves_each_cumulative_boundary():
    shape = (2, 3, 1)
    groups = {}
    for index, name in enumerate(("hpg", "vor", "keg", "zad"), start=1):
        groups[f"{name}_u"] = np.full(shape, index, dtype=np.float64)
        groups[f"{name}_v"] = np.full(shape, 10 * index, dtype=np.float64)
    parts = gate.nemo_components(groups)
    assert np.all(parts["hpg"][0] == 1.0)
    assert np.all(parts["vor"][0] == 2.0)
    assert np.all(parts["keg"][1] == 30.0)
    assert np.all(parts["zad"][0] == 4.0)


def test_every_stage_operator_face_has_a_named_plant():
    names = {f"s{s}.{op}.{face}" for s in (2, 3)
             for op in gate.BOUNDARIES for face in ("u", "v")}
    assert len(names) == 16
