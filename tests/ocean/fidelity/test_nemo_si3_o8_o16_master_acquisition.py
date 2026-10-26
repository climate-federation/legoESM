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
    assert set(gate.KINDS) == set(range(1, 18))
    assert gate.PROJECTIONS == {
        "O8": (1, 2), "O9": (3, 4, 5, 6, 17), "O10": (6, 7, 8),
        "O11": (8, 9), "O12": (10, 11), "O13": (11, 12),
        "O14": (12, 13), "O15": (13, 14), "O16": (15, 16),
    }
    assert [gate.KINDS[k].label for k in range(1, 18)] == [
        "frazil_entry", "frazil_exit", "category_entry", "zdf_exit", "dh_exit",
        "temp1_exit_sal_entry", "sal_exit", "temp2_exit_da_entry", "da_exit",
        "pond_entry_category_done", "pond_exit_itd_entry", "itd_exit_do_entry",
        "do_exit_cor_entry", "cor_exit", "update_flx_entry",
        "update_flx_exit_ocean_boundary", "picard_final",
    ]


def test_schema_includes_o9_fluxes_o11_inputs_and_o16_boundary() -> None:
    names3 = {field.name for field in gate.KINDS[3].fields}
    assert {field.name for field in gate.DH_FLUXES} <= names3
    names8 = {field.name for field in gate.KINDS[8].fields}
    assert {"at_i", "a_i", "h_i", "h_s", "e_i", "e_s", "sz_i", "sst", "t_bo", "sfx_lam", "hfx_thd", "wfx_lam"} <= names8
    assert {field.name for field in gate.KINDS[16].fields} >= {
        "qsr", "qns", "emp", "sfx", "fr_i", "fwfice", "tn_ice", "alb_ice", "snwice_fmass"
    }


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
    assert report["master_bytes_worst_case"] == 114_202_493_300
    assert report["master_gib_worst_case"] == pytest.approx(106.35936008766294)
    assert "without copying" in report["retention"]


def test_writer_field_counts_and_real_extent_branches_match_schema() -> None:
    text = (ACQ / "o8_o16_master_writer.F90").read_text()
    for count in (65, 15, 10, 91, 104, 16, 4):
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
    assert text.index('if [[ "$mode" == --preflight-only ]]') < text.index("./makenemo")
    assert 'timeout "$RUN_TIMEOUT_SEC" mpirun -np 1' in text
    assert "--admit-existing" in text and "--move-aside-builds" in text
    assert "REFUSE_RETAINED" in text and "REFUSE_BUILD_RETAINED" in text
    assert "UNMODIFIED_PRODUCTION" in text and "ln_zdf_chkcvg_overlay" in text
    assert "same_program.json" in text


def test_preflight_compiles_and_runs_real_patched_source_without_nemo(tmp_path: Path) -> None:
    if not Path("/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran").is_file():
        pytest.skip("NEMO compiler unavailable")
    report = tmp_path / "preflight.json"
    env = os.environ.copy()
    env.update(EVIDENCE=str(tmp_path / "evidence"), REFUSED_DIR=str(tmp_path / "refused"))
    proc = subprocess.run(
        [str(ACQ / "run.sh"), "--preflight-only", str(report)], cwd=REPO, env=env,
        text=True, capture_output=True, check=True, timeout=120,
    )
    assert json.loads(report.read_text()) == {"status": "PREFLIGHT_OK", "nemo_run": False, "plants": 36}
    assert "PASS_REAL_PPSRC_PLACEMENT" in proc.stdout
    assert "PASS_PATCHED_O11_BOUNDS_FIXTURE" in proc.stdout
    assert "PREFLIGHT_OK: no NEMO build or integration was run" in proc.stdout
