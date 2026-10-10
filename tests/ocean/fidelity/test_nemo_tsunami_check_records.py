"""TSUNAMI record checker (round-1 MLF and round-2 RK3 programs): admits a
synthetic complete set, refuses plants."""
from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

_PATH = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
         "testcases/nemo_testcase_l1_tsunami/check_records.py")
_spec = importlib.util.spec_from_file_location("tsunami_check_records", _PATH)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

NI, NJ = 8, 6          # includes a 2-point halo: owned is 3..6 x 3..4 (1-based)
BOUNDS = (3, 3, 6, 4)


def _record(path, magic, head, groups):
    with open(path, "wb") as f:
        f.write(magic.ljust(16).encode())
        f.write(struct.pack(f"<{len(head)}i", *head))
        for name, arr in groups.items():
            arr = np.asarray(arr, dtype="<f8")
            n1, n2 = (arr.shape[1], arr.shape[0]) if arr.ndim == 2 else (arr.size, 1)
            f.write(name.ljust(16).encode())
            f.write(struct.pack("<4i", arr.ndim, n1, n2, 1))
            f.write(arr.tobytes())


def _write_set(d: Path, steps: int, program: str = "mlf"):
    field = np.zeros((NJ, NI))
    field[0, 0] = np.nan           # halo garbage must NOT refuse
    full, short = ((cr.FULL, cr.AFTER) if program == "mlf"
                   else (cr.RK3_FULL, cr.RK3_FINAL))
    for kt in range(1, steps + 1):
        names = full if kt <= cr.FULL_STEPS else short
        _record(d / f"oracle_tsustep_kt{kt:08d}.bin", cr.STEP_MAGIC,
                [1, kt, 1, 2, 3, 3, NI, NJ, 2, *BOUNDS, 64],
                {n: field for n in names})
    for kt in range(1, cr.FULL_STEPS + 1):
        _record(d / f"oracle_spgts_kt{kt:08d}.bin", cr.SPGTS_MAGIC,
                [1, kt, 1, 2, 3, 3, NI, NJ, 2, 8, *BOUNDS, 64],
                {n: field for n in sorted(cr.spgts_groups(8))})
    (d / "mesh_mask.nc").write_bytes(b"")


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    run, ref = tmp_path / "run", tmp_path / "ref"
    run.mkdir(), ref.mkdir()
    _write_set(run, 12)
    monkeypatch.setattr(cr, "compare_outputs", lambda a, b: {"stub": True})
    return run, ref


def test_admits_complete_set(evidence):
    run, ref = evidence
    out = cr.check(run, ref, 12, "mlf")
    assert len(out["records"]) == 22 and out["shape"] == [NJ, NI]


@pytest.mark.parametrize("missing", ["i000_entry_sc", "j008_cor_v", "o000_ssh_aa"])
def test_refuses_a_substep_record_missing_one_group(evidence, missing):
    run, ref = evidence
    names = sorted(cr.spgts_groups(8) - {missing})
    _record(run / "oracle_spgts_kt00000002.bin", cr.SPGTS_MAGIC,
            [1, 2, 1, 2, 3, 3, NI, NJ, 2, 8, *BOUNDS, 64],
            {n: np.zeros((NJ, NI)) for n in names})
    with pytest.raises(cr.Refusal, match="missing"):
        cr.check(run, ref, 12, "mlf")


def test_refuses_a_substep_record_with_fewer_substeps_than_declared(evidence):
    run, ref = evidence
    _record(run / "oracle_spgts_kt00000002.bin", cr.SPGTS_MAGIC,
            [1, 2, 1, 2, 3, 3, NI, NJ, 2, 9, *BOUNDS, 64],
            {n: np.zeros((NJ, NI)) for n in sorted(cr.spgts_groups(8))})
    with pytest.raises(cr.Refusal, match="j009"):
        cr.check(run, ref, 12, "mlf")


@pytest.mark.parametrize("plant", ["nan_owned", "missing_step", "bad_magic",
                                   "header_step", "missing_group"])
def test_refuses_plants(evidence, plant):
    run, ref = evidence
    p = run / "oracle_tsustep_kt00000003.bin"
    if plant == "nan_owned":
        bad = np.zeros((NJ, NI)); bad[3, 3] = np.nan
        _record(p, cr.STEP_MAGIC, [1, 3, 1, 2, 3, 3, NI, NJ, 2, *BOUNDS, 64],
                {n: bad for n in cr.FULL})
    elif plant == "missing_step":
        p.unlink()
    elif plant == "bad_magic":
        raw = bytearray(p.read_bytes()); raw[0:1] = b"X"; p.write_bytes(bytes(raw))
    elif plant == "header_step":
        _record(p, cr.STEP_MAGIC, [1, 4, 1, 2, 3, 3, NI, NJ, 2, *BOUNDS, 64],
                {n: np.zeros((NJ, NI)) for n in cr.FULL})
    else:
        _record(p, cr.STEP_MAGIC, [1, 3, 1, 2, 3, 3, NI, NJ, 2, *BOUNDS, 64],
                {n: np.zeros((NJ, NI)) for n in cr.FULL[1:]})
    with pytest.raises(cr.Refusal):
        cr.check(run, ref, 12, "mlf")


def test_output_identity_refuses_a_one_bit_difference(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    for name, bump in (("run", 0.0), ("ref", 2.0 ** -52)):
        d = tmp_path / name
        d.mkdir()
        for field, grid in zip(cr.OUTPUT_FIELDS, "TUV"):
            with netCDF4.Dataset(d / f"T_grid_{grid}.nc", "w") as ds:
                ds.createDimension("x", 2)
                ds.createVariable(field, "f8", ("x",))[:] = [1.0, 1.0 + bump]
    assert cr.compare_outputs(tmp_path / "run", tmp_path / "run")
    with pytest.raises(cr.Refusal):
        cr.compare_outputs(tmp_path / "run", tmp_path / "ref")


@pytest.fixture
def rk3_evidence(tmp_path, monkeypatch):
    run, ref = tmp_path / "run", tmp_path / "ref"
    run.mkdir(), ref.mkdir()
    _write_set(run, 12, "rk3")
    monkeypatch.setattr(cr, "compare_outputs", lambda a, b: {"stub": True})
    return run, ref


def test_rk3_admits_complete_set(rk3_evidence):
    run, ref = rk3_evidence
    out = cr.check(run, ref, 12, "rk3")
    assert len(out["records"]) == 22 and out["program"] == "rk3"
    assert len(cr.RK3_FULL) == 13 + 8 + 3 * 11 + 6


@pytest.mark.parametrize("plant", ["missing_stage_group", "extra_group",
                                   "mlf_shaped", "final_only_at_kt3"])
def test_rk3_refuses_plants(rk3_evidence, plant):
    run, ref = rk3_evidence
    p = run / "oracle_tsustep_kt00000003.bin"
    names = {"missing_stage_group": [n for n in cr.RK3_FULL if n != "2_ssh_aa"],
             "extra_group": cr.RK3_FULL + ["2_bogus"],
             "mlf_shaped": cr.FULL,
             "final_only_at_kt3": cr.RK3_FINAL}[plant]
    _record(p, cr.STEP_MAGIC, [1, 3, 1, 2, 3, 3, NI, NJ, 2, *BOUNDS, 64],
            {n: np.zeros((NJ, NI)) for n in names})
    with pytest.raises(cr.Refusal):
        cr.check(run, ref, 12, "rk3")


def test_rk3_refuses_a_full_record_after_kt10(rk3_evidence):
    run, ref = rk3_evidence
    _record(run / "oracle_tsustep_kt00000011.bin", cr.STEP_MAGIC,
            [1, 11, 1, 2, 3, 3, NI, NJ, 2, *BOUNDS, 64],
            {n: np.zeros((NJ, NI)) for n in cr.RK3_FULL})
    with pytest.raises(cr.Refusal, match="extra"):
        cr.check(run, ref, 12, "rk3")


def test_program_has_no_default(rk3_evidence):
    run, ref = rk3_evidence
    with pytest.raises(TypeError):
        cr.check(run, ref, 12)
    with pytest.raises(ValueError):
        cr.check(run, ref, 12, "leapfrog")


_DIAWRI = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/TSUNAMI/"
               "MY_SRC/diawri.F90")


@pytest.mark.skipif(not _DIAWRI.exists(), reason="NEMO source not present")
def test_output_fields_are_the_case_histwrite_names():
    src = _DIAWRI.read_text()
    written = {line.split('"')[1] for line in src.splitlines()
               if "CALL histwrite" in line and '"' in line}
    assert set(cr.OUTPUT_FIELDS) <= written, sorted(written)
