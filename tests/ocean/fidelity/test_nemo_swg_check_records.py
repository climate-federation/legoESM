"""Direct synthetic tests for the SWG acquisition admission checker."""
from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

_P = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
      "testcases/nemo_testcase_l1_swg/check_records.py")
_S = importlib.util.spec_from_file_location("swg_check_records", _P)
cr = importlib.util.module_from_spec(_S); _S.loader.exec_module(cr)

NI, NJ = 8, 6
BOUNDS = (3, 3, 6, 4)


def _record(path, kt, stage, groups=None):
    groups = cr.GROUPS if groups is None else groups
    field = np.zeros((NJ, NI), dtype="<f8")
    with path.open("wb") as f:
        f.write(cr.MAGIC.ljust(16).encode())
        f.write(struct.pack(f"<{cr.NINT}i",
                            1, kt, stage, 1, 1, 3, 3, NI, NJ, 2,
                            *BOUNDS, 64))
        for name in sorted(groups):
            f.write(name.ljust(16).encode())
            f.write(struct.pack("<4i", 2, NI, NJ, 1))
            f.write(field.tobytes())


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    run, ref = tmp_path / "run", tmp_path / "ref"
    run.mkdir(); ref.mkdir()
    (tmp_path / "manifest").mkdir()
    (tmp_path / "manifest/legoesm_commit").write_text("a" * 40 + "\n")
    for kt in range(1, 11):
        for stage in (1, 2, 3):
            _record(run / f"oracle_swgstage_kt{kt:08d}_s{stage}.bin", kt, stage)
    (run / "mesh_mask.nc").write_bytes(b"")
    monkeypatch.setattr(cr, "compare_outputs", lambda a, b: {"stub": []})
    return run, ref


def test_complete_ordered_record_set_is_admitted(evidence):
    out = cr.check(*evidence)
    assert len(out["records"]) == 30 and out["shape"] == [NJ, NI]


@pytest.mark.parametrize("plant", ["missing", "header", "group", "magic", "nan"])
def test_record_plants_are_refused(evidence, plant):
    run, ref = evidence
    p = run / "oracle_swgstage_kt00000003_s2.bin"
    if plant == "missing":
        p.unlink()
    elif plant == "header":
        _record(p, 4, 2)
    elif plant == "group":
        _record(p, 3, 2, cr.GROUPS - {"r3f"})
    elif plant == "magic":
        raw = bytearray(p.read_bytes()); raw[0] = ord("X"); p.write_bytes(raw)
    else:
        raw = bytearray(p.read_bytes())
        off = 16 + 4 * cr.NINT + 32
        raw[off + 8 * (3 * NI + 3):off + 8 * (3 * NI + 4)] = np.asarray(np.nan, dtype="<f8").tobytes()
        p.write_bytes(raw)
    with pytest.raises(cr.Refusal):
        cr.check(run, ref)


def test_output_passivity_refuses_one_changed_bit(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    run, ref = tmp_path / "run", tmp_path / "ref"
    run.mkdir(); ref.mkdir()
    for kind in cr.GRID_KINDS:
        for directory, bump in ((run, 0.0), (ref, 2.0 ** -52)):
            with netCDF4.Dataset(directory / f"x_grid_{kind}.nc", "w") as ds:
                ds.createDimension("x", 2)
                ds.createVariable("field", "f8", ("x",))[:] = [1.0, 1.0 + bump]
    assert cr.compare_outputs(run, run)
    with pytest.raises(cr.Refusal, match="instrument changed"):
        cr.compare_outputs(run, ref)
