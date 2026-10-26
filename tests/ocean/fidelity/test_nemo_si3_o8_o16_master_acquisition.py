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


def test_schema_has_every_named_boundary_and_registered_projection() -> None:
    assert set(gate.KINDS) == set(range(1, 21))
    assert gate.PROJECTIONS == {
        "O8": (1, 2), "O9": (3, 4, 5, 6, 17), "O10": (6, 7, 8),
        "O11": (8, 9), "O12": (10, 11), "O13": (11, 12),
        "O14": (12, 13), "O15": (13, 14), "O16": (18, 15, 16, 19, 20),
    }
    assert [gate.KINDS[k].label for k in range(1, 21)] == [
        "frazil_entry", "frazil_exit", "category_entry", "zdf_exit", "dh_exit",
        "temp1_exit_sal_entry", "sal_exit", "temp2_exit_da_entry", "da_exit",
        "pond_entry_category_done", "pond_exit_itd_entry", "itd_exit_do_entry",
        "do_exit_cor_entry", "cor_exit", "update_flx_entry",
        "update_flx_exit_ocean_boundary", "picard_final", "pre_aggregation",
        "tau_entry", "tau_exit",
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
    assert {"a_i_b", "h_i_b"} <= names(8)
    assert {"h_i_b", "a_i_b", "hi_max", "rn_amax_2d"} <= names(11) & names(12)
    assert {"t_bo", "ht_i_new", "fraz_frac", "sss_m", "hi_max", "rn_amax_2d"} <= names(12) & names(13)
    assert {"sss_m", "hi_max", "hi_mean", "rn_amax_2d"} <= names(13) & names(14)
    assert {"at_i_b", "a_i_b", "wfx_snw_dyn", "tmask_k1", "smask0", "qemp_ice", "cloud_fra"} <= names(15)
    assert {"uu_kbb", "vv_kbb", "umask_k1", "vmask_k1", "utau", "vtau", "taum"} <= names(19)
    assert {"utau", "vtau", "taum"} == names(20)
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


@pytest.mark.parametrize("plant", ["header", "truncation", "swapped-field", "one-bit"])
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
    assert report["master_bytes_worst_case"] == 133_386_394_340
    assert report["master_gib_worst_case"] == pytest.approx(124.22576019540429)
    assert "without copying" in report["retention"]


def test_writer_field_counts_and_real_extent_branches_match_schema() -> None:
    text = (ACQ / "o8_o16_master_writer.F90").read_text()
    for count in (3, 2, 67, 15, 107, 114, 16, 23, 12, 13):
        assert f",{count},u)" in text
    for spec in gate.KINDS.values():
        names = [field.name for field in spec.fields]
        assert len(names) == len(set(names))
    assert "SIZE(a,1)==ni" in text
    assert "SIZE(a,1)==jpi" in text
    assert "ERROR STOP 'O8-O16 I2 extent mismatch'" in text
    assert "ACCESS='stream'" in text


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
        text=True, capture_output=True, check=True, timeout=300,
    )
    assert json.loads(report.read_text()) == {"status": "PREFLIGHT_OK", "nemo_run": False, "plants": 40}
    assert "PASS_REAL_PPSRC_PLACEMENT" in proc.stdout
    assert "PASS_REAL_PATCHED_BOUNDS_FIXTURE" in proc.stdout
    assert "PASS_OBSERVER_FREE_SAME_PROGRAM" in proc.stdout
    assert "PASS_100_STEP_STUB_DRIVER_RETENTION" in proc.stdout
    assert "PREFLIGHT_OK: no real NEMO integration was run" in proc.stdout
