"""Acquisition-tool controls for SI3 ladder rung O5 (round 12)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
GATE_PATH = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o5_orca2_sas_gate.py"
ACQ = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o5_orca2_sas_acquisition"
SPEC = importlib.util.spec_from_file_location("nemo_si3_o5_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu", PYTHONPATH=str(ROOT))
    return env


def test_binary_reader_self_test_preserves_signed_zero_and_declared_extents():
    report = gate.run_self_test()
    assert report["status"] == "PASS_SELF_TEST"
    assert report["header_bytes"] == 100
    assert report["signed_zero"]["ulp_wet_negzero"] != 0
    assert report["signed_zero"]["ulp_wet_1ulp"] == 1


@pytest.mark.parametrize("plant", ["header", "truncation", "checksum", "time-level", "wet-1ulp"])
def test_binary_and_admission_plants_refuse(plant):
    with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
        gate.run_self_test(plant)


def test_default_operator_script_is_read_only_preflight(tmp_path):
    script = ACQ / "run.sh"
    before = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    result = subprocess.run(
        [str(script), "--preflight-only", str(tmp_path / "preflight.json")],
        cwd=ROOT,
        env=_env(),
        check=True,
        text=True,
        capture_output=True,
    )
    after = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    assert "PREFLIGHT_OK: no NEMO build or integration was run" in result.stdout
    assert "PASS_LANDINGS_AND_12_MISPLACEMENT_PLANTS" in result.stdout
    assert before == after


def test_operator_only_block_has_one_rank_timeout_and_real_flag_census():
    text = (ACQ / "run.sh").read_text()
    operator = text.split("# ---- Operator-only --run mode ----", 1)[1]
    assert 'timeout "$RUN_TIMEOUT_SEC" mpirun -np 1' in operator
    assert "-O0" in text and "PASS_FP_CENSUS_REAL_FLAGS" in text
    assert operator.index('build_target "$REF_CFG" 0') < operator.index("same_program")
    assert operator.index("same_program") < operator.index('stage_run "$REF_CFG"')


def test_writer_uses_stream_records_and_direct_array_writes():
    text = (ACQ / "o5_si3_writer.F90").read_text()
    assert "ACCESS='STREAM'" in text
    assert "STATUS='REPLACE'" in text
    assert "WRITE(iu) a_i; WRITE(iu) v_i" in text
    assert "record_magic = 'O5SI3R01'" in text
    assert "SUM(" not in text and "EXP(" not in text
