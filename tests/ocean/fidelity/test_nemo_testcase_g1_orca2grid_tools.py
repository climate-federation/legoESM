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
                                   "wrong-jperio", "wrong-dims", "nonfinite", "flat-e3t"])
def test_every_domain_plant_exits_nonzero(domain_cfg, plant):
    domcheck, path = domain_cfg
    assert set(domcheck.PLANTS) - {"none"} == {
        "missing-variable", "bad-level", "thin-cell", "wrong-jperio", "wrong-dims", "nonfinite",
        "flat-e3t"}
    with pytest.raises(domcheck.Refuse):
        domcheck.check(path, plant=plant)
    done = subprocess.run(
        [sys.executable, str(CASE / "check_domain_cfg.py"), "--domain-cfg", str(path),
         "--plant", plant], capture_output=True, text=True)
    assert done.returncode == 1 and "STATUS PLANT-FIRED" in done.stderr


def domcheck_plants():
    return [p for p in _load("check_domain_cfg").PLANTS if p != "none"]


def test_run_sh_judges_the_domain_file_not_the_mpirun_status():
    text = (CASE / "run.sh").read_text()
    assert "./make_domain_cfg.exe >run.user.log 2>&1 || dom_status=$?" in text
    assert text.count("domain_cfg_admit ") == 3  # --admit-existing, resume preflight, --run
    for plant in domcheck_plants():
        assert f" {plant}" in text
    last_admit = text.rindex('domain_cfg_admit "$EVIDENCE/manifest"')
    assert text.index("# ---- 1. domain_cfg.nc") < last_admit < text.index("# ---- 2. the two NEMO")


def test_round3_acquisition_only_moves_the_stage_frame_gate():
    """run_r3.sh is run.sh under a new target with stage frames to kt 50; its
    patch is the round-1 patch with the writer's gate moved, nothing else."""
    base = (CASE / "stprk3_g1_step_record.patch").read_text()
    moved = (CASE / "stprk3_g1_step_record_stage50.patch").read_text()
    assert base.count("kstp <= nit000 + 9 )") == 5
    assert moved == base.replace("kstp <= nit000 + 9 )", "kstp <= nit000 + 49 )")
    wrapper = (CASE / "run_r3.sh").read_text()
    assert "export RUN_NAME=ORCA2G1_OMIP_L1_P4" in wrapper
    assert "export REF_NAME=ORCA2G1_OMIP_L1_R3REF" in wrapper
    assert "REF_NAME=${REF_NAME:-ORCA2G1_OMIP_L1}" in (CASE / "run.sh").read_text()
    assert "export STAGE_STEPS=50" in wrapper and 'exec "$here/run.sh" "$@"' in wrapper
    assert "makenemo" not in wrapper and "mpirun" not in wrapper
    text = (CASE / "run.sh").read_text()
    # the defaults of the round-1 acquisition are unchanged
    assert "RUN_NAME=${RUN_NAME:-ORCA2G1_OMIP_L1_P3}" in text
    assert "STAGE_STEPS=${STAGE_STEPS:-10}" in text
    assert "STEP_PATCH=${STEP_PATCH:-$here/stprk3_g1_step_record.patch}" in text


# ------------------------------------------------- round 4: substep records
def _substep_blob(step, icycle=2, bad=None):
    import struct
    blob = b"NEMO_G1_SPGTS1  " + struct.pack(
        "<15i", 1, step, 1, 1, 2, 3, 6, 5, 2, icycle, 3, 3, 4, 3, 64)

    def rec(name, a):
        a = np.asarray(a, dtype="<f8")
        shape = (a.shape[1], a.shape[0], 1) if a.ndim == 2 else (a.size, 1, 1)
        return (name.ljust(16).encode() + struct.pack("<4i", a.ndim, *shape)
                + a.tobytes())
    blob += rec("i000_sshn_e", np.ones((5, 6))) + rec("i000_ffu_nw", np.ones((1, 2)))
    for jn in range(1, icycle + 1):
        ssh = np.full((5, 6), float(jn))
        if bad == "nonfinite" and jn == icycle:
            ssh[0, 0] = np.nan
        blob += rec(f"j{jn:03d}_ssha_e", ssh) + rec(f"j{jn:03d}_ext_coef", np.ones(3))
    return blob + rec("o000_ssh_aa", np.zeros((5, 6)))


def _records_checker():
    import importlib.util
    path = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
            "testcases/nemo_testcase_g1_orca2grid/check_records.py")
    spec = importlib.util.spec_from_file_location("g1_check_records_r4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_substep_streams_are_admitted_at_exactly_the_listed_steps(tmp_path):
    cr = _records_checker()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    for step in (1, 17):
        cr.substep_path(a, step).write_bytes(_substep_blob(step))
    cr.substep_path(b, 1).write_bytes(_substep_blob(1))
    out = cr.check_substeps(a, b, [1, 17])
    assert out[17] == {"icycle": 2, "n_fields": 7,
                       "substep_fields": ["ext_coef", "ssha_e"]}
    head, fields = cr.read_substeps(cr.substep_path(a, 1).read_bytes(), "x")
    assert head["icycle"] == 2 and fields["i000_ffu_nw"].shape == (1, 2)
    assert fields["j002_ext_coef"].shape == (3,)
    # each refusal, one at a time
    with pytest.raises(cr.Refuse, match="missing substep record"):
        cr.check_substeps(a, b, [1, 17], plant="missing-substep")
    with pytest.raises(cr.Refuse, match="unlisted steps"):
        cr.check_substeps(a, b, [1])
    cr.substep_path(b, 1).write_bytes(_substep_blob(1, icycle=3))
    with pytest.raises(cr.Refuse, match="twin run differs"):
        cr.check_substeps(a, b, [1, 17])
    cr.substep_path(b, 1).unlink()
    cr.substep_path(a, 17).write_bytes(_substep_blob(17, bad="nonfinite"))
    with pytest.raises(cr.Refuse, match="non-finite"):
        cr.check_substeps(a, b, [1, 17])
    cr.substep_path(a, 17).write_bytes(_substep_blob(17)[:-8])
    with pytest.raises(cr.Refuse, match="truncated payload"):
        cr.check_substeps(a, b, [1, 17])
    blob = _substep_blob(17, icycle=3)
    cut = blob.index(b"j003_ssha_e")
    cr.substep_path(a, 17).write_bytes(blob[:cut] + blob[blob.index(b"o000_ssh_aa"):])
    with pytest.raises(cr.Refuse, match="frames"):
        cr.check_substeps(a, b, [1, 17])


def test_round4_substep_patch_only_adds_writer_calls_and_matches_its_gate():
    here = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
            "testcases/nemo_testcase_g1_orca2grid")
    patch = (here / "dynspg_ts_g1_substep_record.patch").read_text().splitlines()
    added = [ln[1:].strip() for ln in patch if ln.startswith("+") and not ln.startswith("+++")]
    assert not [ln for ln in patch if ln.startswith("-") and not ln.startswith("---")]
    assert all(ln.startswith(("USE g1_spgts_record", "!", "CALL spgts_r12_", "&"))
               for ln in added)
    assert sum("'ff" in ln for ln in added) == 8
    module = (here / "g1_spgts_record.F90").read_text()
    import re
    assert re.findall(r"kstp /= (\d+)", module) == ["1", "2", "17", "31", "41"]
    wrapper = (here / "run_r4.sh").read_text()
    assert "SPGTS_STEPS=1,2,17,31,41" in wrapper and "ORCA2G1_OMIP_L1_P5" in wrapper
