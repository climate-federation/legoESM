import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = (
    ROOT / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_budget"
)
PACKAGER_PATH = TOOL / "package_records.py"
SPEC = importlib.util.spec_from_file_location(
    "gyre_hires_budget_package_records", PACKAGER_PATH)
PACKAGER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PACKAGER)


def write_record(path: Path, *, day: int = 30, plant=None) -> None:
    ni, nj = 126, 86
    mask = np.zeros((ni, nj), dtype=np.float64)
    interior = np.zeros((122, 82), dtype=np.float64)
    # Match the r4 GYRE surface-V mask: 120 zonal columns x 79 rows.
    # Output column 120 is wet and is the last inner column the old companion
    # recorder failed to populate; output columns 0 and 121 are dry.
    interior[1:121, 1:80] = 1.0
    mask[2:-2, 2:-2] = interior
    terms = np.zeros((ni, nj, len(PACKAGER.TERMS)), dtype=np.float64)
    if plant == "non_finite":
        terms[3, 3, 0] = np.nan
    elif plant == "closure":
        terms[3, 3, 0] = 1.0e-8
    elif plant == "east_closure":
        terms[PACKAGER.HALO + 120, PACKAGER.HALO + 1, 0] = 1.0e-8
    header = (
        PACKAGER.MAGIC
        + struct.pack(
            "=9i", 1, day, day * 24, ni, nj, 30, 64,
            len(PACKAGER.TERMS), 24)
        + struct.pack("=2d", 3600.0, PACKAGER.CLOSURE_BAR_ABS_M_S)
    )
    path.write_bytes(
        header + mask.tobytes(order="F") + terms.tobytes(order="F"))


def write_restart(path: Path, value: float) -> None:
    from netCDF4 import Dataset

    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 2)
        for name in sorted(PACKAGER.REQUIRED_RESTART_VARIABLES):
            variable = dataset.createVariable(name, "f8", ("x",))
            first = value if name == "vn" else 1.0
            variable[:] = np.asarray([first, 1.0], dtype=np.float64)


def test_record_reader_closes_and_all_active_plants_refuse(tmp_path):
    good = tmp_path / "good.bin"
    write_record(good)
    record = PACKAGER.read_record(good)
    assert record["day"] == 30
    assert record["terms"].shape == (8, 82, 122)
    assert int(np.count_nonzero(record["mask"])) == 9480
    assert record["closure_max_abs_m_s"] == 0.0

    for plant, match in (
        ("non_finite", "non-finite"),
        ("closure", "closure"),
        ("east_closure", "closure"),
    ):
        path = tmp_path / f"{plant}.bin"
        write_record(path, plant=plant)
        with pytest.raises(ValueError, match=match):
            PACKAGER.read_record(path)

    bad_magic = tmp_path / "bad-magic.bin"
    bad_magic.write_bytes(b"x" * 100)
    with pytest.raises(ValueError, match="magic"):
        PACKAGER.read_record(bad_magic)

    wrong_day = tmp_path / "oracle_gyre_r14_budget_day0030.bin"
    write_record(wrong_day, day=31)
    with pytest.raises(ValueError, match="does not match filename day"):
        PACKAGER.read_record(wrong_day, expected_day=30)


def test_refusal_diagnostic_localizes_east_active_word(tmp_path):
    path = tmp_path / "east.bin"
    write_record(path, plant="east_closure")
    control = tmp_path / "control.nc"
    instrumented = tmp_path / "instrumented.nc"
    write_restart(control, 0.0)
    write_restart(instrumented, 0.0)
    result = subprocess.run(
        [sys.executable, str(TOOL / "diagnose_closure_refusal.py"),
         "--record", str(path),
         "--control-restart", str(control),
         "--instrumented-restart", str(instrumented)],
        check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["failing_words"] == 1
    assert report["failing_output_columns_zero_based"] == [120]
    assert report["eastmost_wet_output_column_zero_based"] == 120
    assert report["all_failing_words_are_eastmost_wet_column"] is True
    assert report["max_abs_residual_without_eastmost_wet_column_m_s"] == 0.0
    assert report["existing_recorder_passive_identity"]["status"] == "PASS"
    assert report["existing_recorder_passive_identity"]["unequal_words"] == 0


def test_restart_passivity_is_word_exact_and_signed_zero_plant_refuses(tmp_path):
    control = tmp_path / "control.nc"
    same = tmp_path / "same.nc"
    moved = tmp_path / "moved.nc"
    write_restart(control, 0.0)
    write_restart(same, 0.0)
    write_restart(moved, -0.0)
    assert PACKAGER.compare_restarts(control, same)["status"] == "PASS"
    planted = PACKAGER.compare_restarts(control, moved)
    assert planted["status"] == "REFUSE"
    assert planted["unequal_words"] == 1

    empty_control = tmp_path / "empty-control.nc"
    empty_instrumented = tmp_path / "empty-instrumented.nc"
    from netCDF4 import Dataset
    Dataset(empty_control, "w").close()
    Dataset(empty_instrumented, "w").close()
    with pytest.raises(ValueError, match="required prognostic providers"):
        PACKAGER.compare_restarts(empty_control, empty_instrumented)


def test_short_verification_closes_passively_and_refuses_extra_record(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    write_record(run_root / "oracle_gyre_r14_budget_day0030.bin")
    control = tmp_path / "control.nc"
    instrumented = tmp_path / "instrumented.nc"
    write_restart(control, 0.0)
    write_restart(instrumented, 0.0)
    report = PACKAGER.verify_short(run_root, control, instrumented)
    assert report["status"] == "PASS"
    assert report["closure_max_abs_m_s"] == 0.0
    assert report["wet_words"] == 9480
    assert (run_root / "momentum_budget_short_verification.json").is_file()

    write_record(run_root / "oracle_gyre_r14_budget_day0031.bin", day=31)
    with pytest.raises(ValueError, match="expected day 30 only"):
        PACKAGER.verify_short(run_root, control, instrumented)


def test_nemo_patches_apply_exactly_and_scripts_fail_closed(tmp_path):
    nemo = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
    sources = {
        "stprk3_stg": nemo / "cfgs/GYRE_OMIP_L2_P3/MY_SRC/stprk3_stg.F90",
        "dynadv": nemo / "src/OCE/DYN/dynadv.F90",
        "dynzdf": nemo / "src/OCE/DYN/dynzdf.F90",
    }
    for stem, source in sources.items():
        target = tmp_path / f"{stem}.F90"
        target.write_bytes(source.read_bytes())
        result = subprocess.run(
            ["patch", "-l", "-s", "--fuzz=0", str(target)],
            input=(TOOL / f"{stem}_round14.patch").read_text(),
            text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "r14_" in target.read_text()

    namelist = tmp_path / "namelist_cfg"
    namelist.write_bytes((
        TOOL.parent / "configs/gyre_omip_l2_phase3_namelist_cfg"
    ).read_bytes())
    for patch in (
        TOOL.parent / "nemo_testcase_l2_gyre_hires/"
        "namelist_cfg_r4_smag_kt10.patch",
        TOOL / "namelist_cfg_r4_smag_30d_verify.patch",
    ):
        result = subprocess.run(
            ["patch", "-s", "--fuzz=0", str(namelist)],
            input=patch.read_text(), text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
    resolved = namelist.read_text()
    assert 'cn_exp      =  "GYRE_R4_SMAG_R15_VERIFY"' in resolved
    assert "nn_itend    =     720" in resolved
    assert "nn_stock    =     720" in resolved
    assert "nn_write    =      24" in resolved

    run = (TOOL / "run.sh").read_text()
    assert "RANKS must be 1" in run
    assert "-d 'OCE'" in run
    assert '-m "$ARCH"' in run
    assert "conda-scalarmath" in run
    assert "scalar-math-conda" in run
    assert "export GYRE_R14_BUDGET=$enabled" in run
    assert "run_nemo control 0" in run
    assert "run_nemo instrumented 1" in run
    assert "rg " not in run
    assert "mpirun -np 1" in run
    assert "--run" in run
    assert "--run-short" in run
    assert "--preflight-scalar" in run
    assert "--run-scalar" in run
    assert "GYRE_OMIP_L2_P3_R15BUDV" in run
    assert "GYRE_OMIP_L2_P3_R15BUD" in run
    assert "GYRE_OMIP_L2_P3_R16BUDSM" in run
    assert "namelist_cfg_r4_smag_30d_verify.patch" in run
    assert "merge-base --is-ancestor" in run
    assert "REFUSE_NEMO_COMMIT" in run
    assert "REFUSE_NEMO_BUILD_TREE" in run
    assert "--untracked-files=all -- src ext mk tools makenemo" in run
    assert "--nemo-engine-commit" in run
    assert '[[ "$actual_my_src" == "$EXPECTED_MY_SRC" ]]' in run
    assert "REFUSE_MY_SRC_REGISTRY" in run
    assert "unexpected_override.F90" not in run
    for override in (
            "dynhpg.F90", "dynspg_ts.F90", "dynvor.F90", "eosbn2.F90",
            "stprk3.F90", "stprk3_stg.F90", "traadv.F90", "namelist_ref"):
        assert f'check_hash "$SOURCE_ROOT/{"EXP00" if override == "namelist_ref" else "MY_SRC"}/{override}"' in run
    assert "output.namelist.dyn" in run
    assert subprocess.run(
        ["bash", "-n", str(TOOL / "run.sh")], check=False).returncode == 0


def test_packager_refuses_an_unregistered_precision_flavour(tmp_path):
    with pytest.raises(ValueError, match="unsupported NEMO precision flavour"):
        PACKAGER.package(
            tmp_path, tmp_path / "control.nc", tmp_path / "instrumented.nc",
            tmp_path / "prereg.md", "a" * 40, "b" * 40,
            tmp_path / "namelist", tmp_path / "sources",
            tmp_path / "nemo", "c" * 40, "invented-math")


def test_nemo_writer_has_independent_total_and_no_stress_companion():
    module = (TOOL / "gyre_r14_budget.F90").read_text()
    zdf = (TOOL / "dynzdf_round14.patch").read_text()

    def require_aligned_companion(source: str) -> None:
        assert "DIMENSION(jpi,jpkm1)" in source
        assert "DIMENSION(T1Di(0),jpkm1)      ::   zr14_vnostress" not in source

    assert "pv_final(:,:) - zr14_entry(:,:)" in module
    assert "pv_no_stress(:,1) - zr14_pre_zdf(:,kj)" in module
    assert "r14_zdf_pre_row" in module
    assert "zr14_entry(:,kj) + prdt" not in module
    assert "zr14_stress(:,kj) = pv_full(:,1) - pv_no_stress(:,1)" in module
    assert "zr14_vnostress" in zdf
    require_aligned_companion(zdf)
    pre_fix = subprocess.run(
        ["git", "-C", str(ROOT), "show",
         "c708a7d2e83232d008cce19df2f42802f15c6f54:"
         "scripts/validate/ocean_fidelity/testcases/"
         "nemo_testcase_l2_gyre_hires_budget/dynzdf_round14.patch"],
        check=True, capture_output=True, text=True).stdout
    with pytest.raises(AssertionError):
        require_aligned_companion(pre_fix)
    assert "CALL r14_zdf_pre_row( jj, pvv(:,jj,:,Kaa) )" in zdf
    assert "CALL r14_zdf_row( jj, zr14_vnostress" in zdf
    # The control would be vacuous if it reused the already-solved production
    # V field; it must run a second forward/back substitution from the saved
    # pre-stress RHS.
    assert "zr14_vnostress(ji,jk) = zr14_vnostress(ji,jk) -" in zdf
    assert "pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) -" in zdf
