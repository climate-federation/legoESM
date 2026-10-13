"""Fail-closed controls for the ORCA2G1 acquisition tools."""
from __future__ import annotations

import importlib.util
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
CASE = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_g1_orca2grid"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"g1_{name}", CASE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load("check_records")
bathy = _load("make_bathy_level")
JPI, JPJ = 6, 5
STAGE = {"e": ("ssh_bb",), "b": ("ssh_aa",), "1": ("ssh_aa",), "2": ("ssh_aa",),
         "3": ("ssh_aa",), "f": ("ssh_bb", "uu_b_bb")}


def _stream(step: int, stage: bool) -> bytes:
    blob = b"NEMO_G1_STEP_01 " + struct.pack(
        "<14i", 1, step, 1, 2, 3, 3, JPI, JPJ, 2, 3, 3, JPI - 2, JPJ - 2, 64)
    for frame, names in STAGE.items():
        if not stage and frame != "f":
            continue
        for name in names:
            data = np.arange(JPI * JPJ, dtype="<f8") + step
            blob += struct.pack("<16s4i", f"{frame}_{name}".ljust(16).encode(),
                                2, JPI, JPJ, 1) + data.tobytes()
    return blob


@pytest.fixture
def record(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for step in range(1, 5):
        (a / f"oracle_g1step_kt{step:08d}.bin").write_bytes(_stream(step, step <= 2))
    for step in range(1, 3):
        (b / f"oracle_g1step_kt{step:08d}.bin").write_bytes(_stream(step, True))
    return a, b


def test_clean_record_is_admitted_from_its_own_headers(record):
    a, b = record
    result = checker.check(a, b, steps=4, stage_steps=2)
    assert result["status"] == "PASS_G1_RECORD"
    assert result["geometry"]["jpi"] == JPI and result["geometry"]["jpj"] == JPJ
    assert result["fields_final_only"] == ["f_ssh_bb", "f_uu_b_bb"]


@pytest.mark.parametrize("plant", [p for p in checker.PLANTS if p != "none"])
def test_every_plant_exits_nonzero(record, plant):
    a, b = record
    with pytest.raises(checker.Refuse):
        checker.check(a, b, steps=4, stage_steps=2, plant=plant)
    done = subprocess.run(
        [sys.executable, str(CASE / "check_records.py"), "--evidence", str(a),
         "--twin", str(b), "--steps", "4", "--stage-steps", "2", "--plant", plant],
        capture_output=True, text=True)
    assert done.returncode == 1 and "STATUS PLANT-FIRED" in done.stderr


def test_real_defects_are_refused(record):
    a, b = record
    (b / "oracle_g1step_kt00000002.bin").write_bytes(_stream(3, True))
    with pytest.raises(checker.Refuse, match="twin run differs"):
        checker.check(a, b, steps=4, stage_steps=2)
    (a / "oracle_g1step_kt00000003.bin").write_bytes(_stream(3, True))
    with pytest.raises(checker.Refuse, match="frames"):
        checker.check(a, None, steps=4, stage_steps=2)
    zero = _stream(1, True)
    head = 72
    (a / "oracle_g1step_kt00000001.bin").write_bytes(zero[:head] + bytes(len(zero) - head))
    with pytest.raises(checker.Refuse):
        checker.check(a, None, steps=1, stage_steps=1)


def test_bathy_level_is_the_source_mask(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    bottom = np.array([[0, 3, 30], [1, 0, 12]], dtype="i4")
    with netCDF4.Dataset(tmp_path / "src.nc", "w") as ds:
        ds.createDimension("y", 2)
        ds.createDimension("x", 3)
        ds.createVariable("bottom_level", "i4", ("y", "x"))[:] = bottom
    assert bathy.write_bathy_level(tmp_path / "src.nc", tmp_path / "out.nc") == 4
    with netCDF4.Dataset(tmp_path / "out.nc") as ds:
        level = np.asarray(ds["Bathy_level"][:])
    assert level.dtype == np.float64
    assert np.array_equal(level, [[0, 1, 1], [1, 0, 1]])


@pytest.fixture
def domain_cfg(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    domcheck = _load("check_domain_cfg")
    path = tmp_path / "domain_cfg.nc"
    with netCDF4.Dataset(path, "w") as ds:
        for name, size in {**domcheck.DIMS, "time_counter": 1}.items():
            ds.createDimension(name, size)
        for name in domcheck.REQUIRED:
            if name in ("bottom_level", "top_level"):
                var = ds.createVariable(name, "i4", ("time_counter", "y", "x"))
                var[:] = 1
                var[0, 0, :] = 0
            elif name.startswith(("jp", "ln_")):
                ds.createVariable(name, "i4", ())[...] = domcheck.SCALARS.get(name, 0)
            elif name.endswith("_0"):
                ds.createVariable(name, "f8", ("time_counter", "nav_lev", "y", "x"))[:] = 100.0
            elif name.endswith("_1d"):
                ds.createVariable(name, "f8", ("time_counter", "nav_lev"))[:] = 100.0
            else:
                ds.createVariable(name, "f8", ("time_counter", "y", "x"))[:] = 1.0
    return domcheck, path


def test_complete_domain_cfg_is_admitted_with_its_hash(domain_cfg):
    import hashlib

    domcheck, path = domain_cfg
    result = domcheck.check(path, shipped=path)
    assert result["status"] == "PASS_G1_DOMAIN_CFG"
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["wet_cells"] == 147 * 180
    assert result["vs_shipped"]["mesh_fields_differing"] == []
    with pytest.raises(domcheck.Refuse, match="absent"):
        domcheck.check(path.with_name("none.nc"))
    path.with_name("junk.nc").write_bytes(b"not netcdf")
    with pytest.raises(domcheck.Refuse, match="not a readable"):
        domcheck.check(path.with_name("junk.nc"))


@pytest.mark.parametrize("plant", ["missing-variable", "bad-level", "thin-cell",
                                   "wrong-jperio", "wrong-dims", "nonfinite"])
def test_every_domain_plant_exits_nonzero(domain_cfg, plant):
    domcheck, path = domain_cfg
    assert set(domcheck.PLANTS) - {"none"} == {
        "missing-variable", "bad-level", "thin-cell", "wrong-jperio", "wrong-dims", "nonfinite"}
    with pytest.raises(domcheck.Refuse):
        domcheck.check(path, plant=plant)
    done = subprocess.run(
        [sys.executable, str(CASE / "check_domain_cfg.py"), "--domain-cfg", str(path),
         "--plant", plant], capture_output=True, text=True)
    assert done.returncode == 1 and "STATUS PLANT-FIRED" in done.stderr


def test_run_sh_judges_the_domain_file_not_the_mpirun_status():
    text = (CASE / "run.sh").read_text()
    assert "./make_domain_cfg.exe >run.user.log 2>&1 || dom_status=$?" in text
    assert text.count("domain_cfg_admit ") == 2  # preflight of a resume, and --run
    assert text.index('domain_cfg_admit "$EVIDENCE/manifest"') < text.index("build_one \"$REF_NAME\" 0")
