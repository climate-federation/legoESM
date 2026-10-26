"""Controls for the operator-only SI3 O8--O16 master acquisition."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "scripts/validate/ocean_fidelity/testcases"
ACQ = TOOLS / "nemo_si3_o8_o16_master_acquisition"
sys.path.insert(0, str(REPO))
SPEC = importlib.util.spec_from_file_location("o8o16_gate", TOOLS / "nemo_si3_o8_o16_master_gate.py")
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)
READSET_SPEC = importlib.util.spec_from_file_location("o8o16_readset", TOOLS / "nemo_si3_o8_o16_readset.py")
assert READSET_SPEC and READSET_SPEC.loader
readset = importlib.util.module_from_spec(READSET_SPEC)
sys.modules[READSET_SPEC.name] = readset
READSET_SPEC.loader.exec_module(readset)


def test_schema_has_every_named_boundary_and_registered_projection() -> None:
    assert set(gate.KINDS) == set(range(1, 22))
    assert gate.PROJECTIONS == {
        "O8": (1, 2), "O9": (21, 3, 4, 5, 6, 17), "O10": (21, 6, 7, 8),
        "O11": (21, 8, 9), "O12": (10, 11), "O13": (11, 12),
        "O14": (12, 13), "O15": (13, 14), "O16": (18, 15, 16, 19, 20),
    }
    assert [gate.KINDS[k].label for k in range(1, 22)] == [
        "frazil_entry", "frazil_exit", "category_entry", "zdf_exit", "dh_exit",
        "temp1_exit_sal_entry", "sal_exit", "temp2_exit_da_entry", "da_exit",
        "pond_entry_category_done", "pond_exit_itd_entry", "itd_exit_do_entry",
        "do_exit_cor_entry", "cor_exit", "update_flx_entry",
        "update_flx_exit_ocean_boundary", "picard_final", "pre_aggregation",
        "tau_entry", "tau_exit",
        "pre_category_state",
    ]


def test_schema_includes_o9_fluxes_o11_inputs_and_o16_boundary() -> None:
    names3 = {field.name for field in gate.KINDS[3].fields}
    assert {field.name for field in gate.DH_FLUXES} <= names3
    names8 = {field.name for field in gate.KINDS[8].fields}
    assert {"at_i", "a_i", "h_i", "h_s", "e_i", "e_s", "sz_i", "sst", "t_bo", "sfx_lam", "hfx_thd", "wfx_lam"} <= names8
    assert {field.name for field in gate.KINDS[16].fields} >= {
        "qsr", "qns", "emp", "sfx", "fr_i", "fwfice", "tn_ice", "alb_ice", "snwice_fmass"
    }


def test_every_reviewed_boundary_input_is_recorded() -> None:
    names = lambda kind: {field.name for field in gate.KINDS[kind].fields}
    assert {"t_su", "qns_ice",
            "zkappa_i", "zkappa_s", "zradab_i", "zradab_s", "zradtr_i", "zradtr_s"} <= names(17)
    for kind in range(3, 10):
        assert "o_i" in names(kind)
        assert not ({"a_i_b", "h_i_b"} & names(kind))
    assert names(21) == {field.name for field in gate.STATE}
    assert names(1) == {"qlead", "smask0", "at_i", "u_ice", "v_ice", "umask_k1",
                         "vmask_k1", "utau_ice", "vtau_ice", "hi_max"}
    assert {"h_i_b", "a_i_b", "hi_max", "rn_amax_2d"} <= names(11) & names(12)
    assert {"t_bo", "ht_i_new", "fraz_frac", "sss_m", "hi_max", "rn_amax_2d"} <= names(12) & names(13)
    assert {"sss_m", "sst_m", "hi_max", "hi_mean", "rn_amax_2d"} <= names(13) & names(14)
    assert {"at_i_b", "a_i_b", "wfx_snw_dyn", "tmask_k1", "smask0", "qemp_ice", "cloud_fra"} <= names(15)
    assert {"uu_kbb", "vv_kbb", "umask_k1", "vmask_k1", "utau", "vtau", "taum"} <= names(19)
    assert {"utau", "vtau", "taum", "rCdU_ice"} == names(20)
    assert names(18) == {field.name for field in gate.STATE}
    assert {"rcnd_s", "rn_kappa_i", "rn_kappa_s", "rn_snwblow", "rn_sinew",
            "nn_icesal", "rn_beta", "rn_dmin"} <= set(gate.SCALAR_NAMES)
    assert {"rn_icesal", "rn_sal_gd", "rn_time_gd", "rn_sal_fl", "rn_time_fl",
            "nn_liquidus", "rn_alpha_GN", "rn_Rc_GN", "rn_alpha_CW", "ln_sal_chk",
            "ln_pnd", "ln_pnd_LEV", "ln_pnd_CST", "ln_pnd_TOPO", "rn_apnd",
            "rn_hpnd", "ln_pnd_alb", "rn_pnd_hl_min", "rn_pnd_hl_max"} <= set(gate.SCALAR_NAMES)
    assert {"rn_alb_sdry", "rn_alb_smlt", "rn_alb_idry", "rn_alb_imlt",
            "rn_alb_dpnd", "rn_alb_lpnd", "rn_alb_hpiv"} <= set(gate.SCALAR_NAMES)
    assert gate.KINDS[17].fields[-5].extent == "GI"
    assert gate.KINDS[17].fields[-2].extent == "GS"
    extents = lambda kind: {field.name: field.extent for field in gate.KINDS[kind].fields}
    assert extents(16)["fr_i"] == "F2"
    assert extents(19)["utau"] == extents(19)["vtau"] == "F2"
    assert extents(20)["utau"] == extents(20)["vtau"] == "F2"
    assert extents(20)["rCdU_ice"] == "F2"
    assert "vm_ibr" not in {field.name for field in gate.TOTALS}


@pytest.mark.parametrize("plant", ["header", "truncation", "swapped-field", "one-bit", "category-dh"])
def test_reader_plants_fire(plant: str) -> None:
    with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
        gate.self_test(plant)


@pytest.mark.parametrize("rung", sorted(gate.PROJECTIONS))
def test_each_rung_reader_accepts_only_its_registered_projection(rung: str) -> None:
    expected = gate._expected_projection(rung)
    assert gate.validate_projection_index(expected, rung)["records"] == len(expected)
    with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
        gate.validate_projection_index(expected, rung, "foreign-kind")


def test_size_and_retention_are_explicit() -> None:
    report = gate.size_report()
    assert report["master_bytes_worst_case"] == 140_714_793_940
    assert report["master_gib_worst_case"] == pytest.approx(131.05086417868733)
    assert "without copying" in report["retention"]


def test_writer_field_counts_and_real_extent_branches_match_schema() -> None:
    text = (ACQ / "o8_o16_master_writer.F90").read_text()
    for count in (2, 4, 10, 12, 13, 16, 23, 67, 107, 113):
        assert f",{count},u)" in text
    for spec in gate.KINDS.values():
        names = [field.name for field in spec.fields]
        assert len(names) == len(set(names))
    assert "SIZE(a,1)==ni" in text
    assert "SIZE(a,1)==jpi" in text
    assert "ERROR STOP 'O8-O16 I2 extent mismatch'" in text
    assert "ACCESS='stream'" in text


def test_bounds_fixture_uses_real_producers_for_reviewed_values() -> None:
    text = (ACQ / "real_bounds_driver.F90").read_text()
    assert "ht_i_new=851._wp" not in text and "fraz_frac=852._wp" not in text
    assert text.index("CALL ice_thd_frazil") < text.index("CALL o8o16_global(21)")
    assert text.index("o_i_1d=ieee_value") < text.index("CALL tab_2d_1d")
    assert text.index("CALL tab_2d_1d") < text.index("CALL o8o16_category(3,1)")


def test_source_derived_readsets_cover_every_rung_and_plants_fire() -> None:
    report = readset.derive_and_gate(readset.DEFAULT_PPSRC)
    assert set(report["rungs"]) == set(gate.PROJECTIONS)
    assert "sst_m" in report["rungs"]["O15"]["array_reads"]
    assert "rn_hinew" in report["rungs"]["O8"]["scalar_reads"]
    assert all(row["scalar_coverage"] for row in report["rungs"].values())
    assert "rCdU_ice" in {field.name for field in gate.KINDS[20].fields}
    for plant in ("drop-sst_m", "drop-qlead", "drop-rn_hinew"):
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            readset.derive_and_gate(readset.DEFAULT_PPSRC, plant)


def test_category_entry_observer_follows_every_diagnostic_zero() -> None:
    patch = (ACQ / "icethd_o8_o16.patch").read_text()
    assert "@@ -146,0 +150,3 @@" in patch
    source = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/ICE/icethd.F90").read_text()
    assert source.index("dh_s_tot  (1:npti) = 0._wp") < source.index("dh_snowice(1:npti) = 0._wp")


def test_every_patch_insertion_is_inside_observer_markers() -> None:
    for patch in ACQ.glob("*.patch"):
        inside = False
        blocks = 0
        for line in patch.read_text().splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            value = line[1:].strip()
            if value == "! O8O16_OBSERVER_BEGIN":
                assert not inside; inside = True; blocks += 1
            elif value == "! O8O16_OBSERVER_END":
                assert inside; inside = False
            else:
                assert inside, f"{patch.name}: insertion outside observer block"
        assert not inside and blocks >= 2


def test_script_is_read_only_by_default_and_retains_refusals() -> None:
    text = (ACQ / "run.sh").read_text()
    assert "mode=${1:---preflight-only}" in text
    assert text.index('if [[ "$mode" == --preflight-only ]]') < text.index('build_target "$REF_CFG"')
    assert 'timeout "$remaining" mpirun -np 1' in text
    assert "--admit-existing" in text and "--move-aside-builds" in text
    assert "REFUSE_RETAINED" in text and "REFUSE_BUILD_RETAINED" in text
    assert "UNMODIFIED_PRODUCTION" in text and "ln_zdf_chkcvg_overlay" in text
    assert "same_program.json" in text
    assert "--expected-ppsrc-dir" in text and "--plant same-program" in text
    assert "compile_real_bounds_fixture" in text and "real_bounds_driver.F90" in text
    assert "stub_driver_fixture" in text and 'NEMO_EXE_OVERRIDE="$STUB_NEMO"' in text
    assert 'O8O16_PATH_PREFIX="$scratch/stub_bin"' in text and 'STUB_MPIRUN' in text
    assert "p.unlink(); p.write_text(text)" in text
    assert "PASS_EXISTING_CFG_DIGESTS_UNCHANGED" in text
    assert "cfg-digest" in text
    assert "here/'stub_mpirun.sh'" in text
    assert "'icethd_zdf_bl99','icealb'" in text
    assert "O8O16_STUB_MODE" not in text and "O8O16_SKIP_DRIVER_TEST" not in text
    assert 'cp "$NEMO_SOURCE_ROOT/src/ICE/$unit.F90"' in text
    assert 'timeout "$remaining" mpirun -np 1' in text
    assert 'O8O16_DRIVER_TRUNCATION_POINT' in text and '"$scratch/run_truncated.sh"' in text
    assert 'syntax_plant.sh' not in text


def test_reused_o7_log_gate_accepts_the_preregistered_100_step_window(tmp_path: Path) -> None:
    subprocess.run([str(ACQ / "stub_nemo.sh")], cwd=tmp_path, check=True, capture_output=True, text=True)
    report = gate.verify_o7_log(tmp_path / "ocean.output", expected_nn_itend=100)
    assert report["o7_production"]["ln_icethd"] is True
    with pytest.raises(ValueError, match="nn_itend is not 2"):
        gate.verify_o7_log(tmp_path / "ocean.output")


def test_preflight_compiles_and_runs_real_patched_source_without_nemo(tmp_path: Path) -> None:
    if not Path("/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran").is_file():
        pytest.skip("NEMO compiler unavailable")
    report = tmp_path / "preflight.json"
    env = os.environ.copy()
    env.update(EVIDENCE=str(tmp_path / "evidence"), REFUSED_DIR=str(tmp_path / "refused"))
    proc = subprocess.run(
        [str(ACQ / "run.sh"), "--preflight-only", str(report)], cwd=REPO, env=env,
        text=True, capture_output=True, check=True, timeout=1200,
    )
    assert json.loads(report.read_text()) == {"status": "PREFLIGHT_OK", "nemo_run": False, "plants": 45}
    assert "PASS_SOURCE_DERIVED_READ_SETS" in proc.stdout
    assert "PASS_REAL_PPSRC_PLACEMENT" in proc.stdout
    assert "PASS_REAL_PATCHED_BOUNDS_FIXTURE" in proc.stdout
    assert "PASS_OBSERVER_FREE_SAME_PROGRAM" in proc.stdout
    assert "PASS_100_STEP_STUB_DRIVER_RETENTION" in proc.stdout
    assert "PASS_EXISTING_CFG_DIGESTS_UNCHANGED" in proc.stdout
    assert "PREFLIGHT_OK: no real NEMO integration was run" in proc.stdout
