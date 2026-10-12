"""Direct synthetic tests for the SWG acquisition admission checker."""
from __future__ import annotations

import importlib.util
import subprocess
import struct
from pathlib import Path

import numpy as np
import pytest

_P = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
      "testcases/nemo_testcase_l1_swg/check_records.py")
_S = importlib.util.spec_from_file_location("swg_check_records", _P)
cr = importlib.util.module_from_spec(_S); _S.loader.exec_module(cr)
TOOLS = _P.parent
SHIPPED_NAMELIST = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/SWG/"
    "EXPREF/namelist_cfg")
SHIPPED_ZGR = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/SWG/"
    "MY_SRC/usrdef_zgr.F90")

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


def test_daily_stage_one_sequence_is_driven_by_requested_inventory(
        tmp_path, monkeypatch):
    run, ref = tmp_path / "run", tmp_path / "ref"
    run.mkdir(); ref.mkdir()
    (tmp_path / "manifest").mkdir()
    (tmp_path / "manifest/legoesm_commit").write_text("b" * 40 + "\n")
    for kt in (1, 25, 49):
        _record(run / f"oracle_swgstage_kt{kt:08d}_s1.bin", kt, 1)
    (run / "mesh_mask.nc").write_bytes(b"")
    monkeypatch.setattr(cr, "compare_outputs", lambda a, b: {"stub": []})
    out = cr.check(run, ref, record_kts=(1, 25, 49), stages=(1,))
    assert len(out["records"]) == 3
    (run / "oracle_swgstage_kt00000049_s1.bin").unlink()
    with pytest.raises(cr.Refusal, match="stage sequence"):
        cr.check(run, ref, record_kts=(1, 25, 49), stages=(1,))


@pytest.mark.parametrize(
    "plant", ["missing", "header", "group", "magic", "nan", "halo_nan",
              "halo_sentinel"])
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
    elif plant == "nan":
        raw = bytearray(p.read_bytes())
        off = 16 + 4 * cr.NINT + 32
        raw[off + 8 * (3 * NI + 3):off + 8 * (3 * NI + 4)] = np.asarray(np.nan, dtype="<f8").tobytes()
        p.write_bytes(raw)
    else:
        raw = bytearray(p.read_bytes())
        off = 16 + 4 * cr.NINT + 32
        # Row 0, column 0 lies outside the owned 1-based BOUNDS=(3,3,6,4).
        value = np.nan if plant == "halo_nan" else cr.SENTINEL
        raw[off:off + 8] = np.asarray(value, dtype="<f8").tobytes()
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


def test_daily_restart_passivity_refuses_one_changed_bit(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    run, ref = tmp_path / "run", tmp_path / "ref"
    run.mkdir(); ref.mkdir()
    name = "SWG_00000024_restart.nc"
    for directory, value in ((run, 1.0), (ref, 1.0)):
        with netCDF4.Dataset(directory / name, "w") as ds:
            ds.createDimension("x", 1)
            ds.createVariable("sshn", "f8", ("x",))[:] = [value]
    assert cr.compare_restarts(run, ref, (24,))
    with netCDF4.Dataset(ref / name, "a") as ds:
        ds["sshn"][:] = [np.nextafter(np.float64(1.0), np.float64(np.inf))]
    with pytest.raises(cr.Refusal, match="instrument changed"):
        cr.compare_restarts(run, ref, (24,))


@pytest.mark.skipif(not SHIPPED_NAMELIST.is_file(),
                    reason="shipped SWG deck not mounted")
@pytest.mark.parametrize("name,expected", (
    ("namelist_cfg_maddison_kt10.patch", {
        "nn_SWG": "16", "rn_domsiz": "4000000", "rn_tau": "0.10",
        "rn_modified_grav": "9.80665", "rn_rfr": "1.e-7",
        "rn_beta": "2.e-11", "rn_f0": "0.7e-4", "rn_Uv": "2.0",
        "rn_Lv": "10", "rn_Dt": "50.", "nn_itend": "10",
        "nn_stock": "1", "nn_write": "1", "ln_meshmask": ".true.",
        "sn_cfctl%l_runstat": ".TRUE.", "sn_cfctl%ptimincr": "1",
    }),
    ("namelist_cfg_maddison_30d.patch", {
        "nn_SWG": "4", "rn_domsiz": "4000000", "rn_tau": "0.10",
        "rn_modified_grav": "9.80665", "rn_rfr": "1.e-7",
        "rn_beta": "2.e-11", "rn_f0": "0.7e-4", "rn_Uv": "2.0",
        "rn_Lv": "10", "rn_Dt": "200.", "nn_itend": "12960",
        "nn_stock": "432", "nn_write": "432", "ln_meshmask": ".true.",
        "sn_cfctl%l_runstat": ".TRUE.", "sn_cfctl%ptimincr": "1",
    }),
))
def test_maddison_patches_resolve_exact_operator_decks(tmp_path, name, expected):
    target = tmp_path / "namelist_cfg"
    target.write_bytes(SHIPPED_NAMELIST.read_bytes())
    subprocess.run(
        ["patch", "-s", "--fuzz=0", str(target)],
        input=(TOOLS / name).read_bytes(), check=True)
    subprocess.run(
        ["patch", "-s", "--fuzz=0", str(target)],
        input=(TOOLS / "namelist_meshmask.patch").read_bytes(), check=True)
    values = {}
    for raw in target.read_text().splitlines():
        code = raw.split("!", 1)[0]
        if "=" not in code:
            continue
        key, value = (part.strip() for part in code.split("=", 1))
        if key in expected:
            values[key] = value
    assert values == expected


@pytest.mark.skipif(not SHIPPED_ZGR.is_file(),
                    reason="shipped SWG source not mounted")
def test_source_case_patch_changes_only_the_two_upper_mask_limits(tmp_path):
    target = tmp_path / "usrdef_zgr.F90"
    target.write_bytes(SHIPPED_ZGR.read_bytes())
    before = target.read_text().splitlines()
    patch = TOOLS / "usrdef_zgr_domsiz.patch"
    subprocess.run(
        ["patch", "-s", "--fuzz=0", str(target)],
        input=patch.read_bytes(), check=True)
    after = target.read_text().splitlines()
    changed = [(left.strip(), right.strip())
               for left, right in zip(before, after, strict=True)
               if left != right]
    assert changed == [
        ("zylim1 = 2010000._wp    ! 2010km",
         "zylim1 = rn_domsiz + 10000._wp    ! SWG_DG: wet extent follows "
         "rn_domsiz (+10 km)"),
        ("zxlim1 = 2010000._wp    ! 2010km",
         "zxlim1 = rn_domsiz + 10000._wp    ! SWG_DG: wet extent follows "
         "rn_domsiz (+10 km)"),
    ]


def test_daily_writer_derives_cadence_from_resolved_timestep():
    patch = (TOOLS / "stprk3_daily_entry_record.patch").read_text()
    assert "NINT(rday / rn_Dt)" in patch
    assert "MOD(kstp - nit000, 24)" not in patch


def test_daily_clock_refuses_fractional_nint_cadence(tmp_path):
    deck = TOOLS / "deck.sh"

    def check(dt):
        namelist = tmp_path / "namelist_cfg"
        namelist.write_text(
            f" rn_Dt = {dt}\n nn_itend = 12960\n"
            " nn_stock = 432\n nn_write = 432\n")
        return subprocess.run(
            ["bash", "-c",
             f'. "{deck}"; swg_require_daily_clock "$1" || exit $?; '
             'printf "STEPS_PER_DAY=%s\\n" "$STEPS_PER_DAY"',
             "clock", str(namelist)],
            capture_output=True, text=True)

    accepted = check("200.")
    assert accepted.returncode == 0
    assert accepted.stdout == "STEPS_PER_DAY=432\n"

    # Non-vacuity: the removed integer-truncation guard accepted this plant as
    # 86400 / int(50.5) = 1728, while NEMO NINT gives 1711.
    assert 86_400 % int(50.5) == 0
    refused = check("50.5")
    assert refused.returncode == 68
    assert "NINT cadence=1711; exact integer required" in refused.stderr


def test_makenemo_source_guard_accepts_only_the_shipped_test_case(tmp_path):
    tests = tmp_path / "tests"
    (tests / "SWG").mkdir(parents=True)
    (tests / "demo_cfgs.txt").write_text("SWG OCE SWE\n")
    deck = TOOLS / "deck.sh"

    def guard(candidate):
        return subprocess.run(
            ["bash", "-c",
             f'. "{deck}"; swg_verify_makenemo_source_case "$1" "$2"',
             "guard", str(tmp_path), candidate],
            capture_output=True, text=True)

    assert guard("SWG").returncode == 0
    refused = guard("SWG_DG")
    assert refused.returncode == 68
    assert "-a must name shipped test case SWG" in refused.stderr


def test_target_cpp_key_guard_reads_the_created_target(tmp_path):
    run_name = "SWG_DG_OMIP_L1_P3"
    cpp = tmp_path / f"cpp_{run_name}.fcm"
    cpp.write_text("bld::tool::fppkeys key_qco key_vco_3d key_RK3\n")
    deck = TOOLS / "deck.sh"

    def guard():
        return subprocess.run(
            ["bash", "-c",
             f'. "{deck}"; swg_verify_target_cpp_keys "$1" "$2"',
             "guard", str(tmp_path), run_name],
            capture_output=True, text=True)

    assert guard().returncode == 0
    cpp.write_text("bld::tool::fppkeys key_xios key_qco key_vco_3d key_RK3\n")
    refused = guard()
    assert refused.returncode == 70
    assert str(cpp) in refused.stderr


@pytest.mark.parametrize("name", ("run.sh", "run_30d.sh"))
def test_writer_target_is_created_from_shipped_swg_before_deck_overlay(name):
    script = (TOOLS / name).read_text()
    invocation = (
        './makenemo -a "$MAKENEMO_SOURCE_CASE" -n "$RUN_NAME" '
        '-m conda-scalarmath del_key key_xios')
    assert script.count(invocation) == 1
    assert './makenemo -a "$SOURCE_CASE_NAME"' not in script
    assert (script.index(invocation)
            < script.index('swg_verify_target_cpp_keys "$cfg" "$RUN_NAME"')
            < script.index('cp -r "$SRC_CASE/MY_SRC/." "$cfg/MY_SRC/"')
            < script.index('patch "$cfg/MY_SRC/stprk3.F90" <"$STEP_PATCH"'))

    # Non-vacuity: the exact Round-1 regression is rejected by this assertion.
    regressed = script.replace(invocation, invocation.replace(
        '"$MAKENEMO_SOURCE_CASE"', '"$SOURCE_CASE_NAME"'), 1)
    assert invocation not in regressed
