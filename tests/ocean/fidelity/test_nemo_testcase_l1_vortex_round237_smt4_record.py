"""Fail-closed controls for round 237's SMT-4 momentum record."""
from __future__ import annotations

import importlib.util
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
TOOLS = (ROOT / "scripts/validate/ocean_fidelity/testcases/"
         "nemo_testcase_l1_vortex")
CHECKER = TOOLS / "check_records.py"
_SPEC = importlib.util.spec_from_file_location("round237_checker", CHECKER)
assert _SPEC and _SPEC.loader
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
JPI, JPJ, JPK = 8, 7, 3


def _apply(path: Path, patch: Path) -> None:
    result = subprocess.run(
        ["patch", "-s", str(path)], input=patch.read_text(), text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def _resolved_smt3_and_smt4(tmp_path: Path) -> tuple[str, str]:
    shipped = NEMO / "tests/VORTEX/EXPREF/namelist_cfg"
    smt3 = tmp_path / "namelist_cfg_smt3"
    shutil.copyfile(shipped, smt3)
    _apply(smt3, TOOLS / "namelist_cfg_smt2_vec_een.patch")
    _apply(smt3, TOOLS / "namelist_cfg_smt3_tracer_diffusion.patch")
    smt4 = tmp_path / "namelist_cfg_smt4"
    shutil.copyfile(smt3, smt4)
    _apply(smt4, TOOLS / "namelist_cfg_smt4_momentum_diffusion.patch")
    return smt3.read_text(), smt4.read_text()


def test_smt4_deck_changes_only_the_momentum_diffusion_block(tmp_path):
    smt3, smt4 = _resolved_smt3_and_smt4(tmp_path)
    # Use Python's stable unified diff so the assertion is independent of a
    # platform-specific diff invocation.
    import difflib
    changed = "".join(difflib.unified_diff(
        smt3.splitlines(True), smt4.splitlines(True)))
    assert "&namdyn_ldf" in changed
    assert "&namtra_ldf" not in changed
    for line in (
        "ln_dynldf_OFF =  .false.", "nn_dynldf_typ = 0",
        "ln_dynldf_lap =  .true.", "ln_dynldf_lev =  .true.",
        "ln_dynldf_hor =  .false.", "nn_ahm_ijk_t  = 20",
        "rn_Uv      = 0.1", "rn_Lv      = 10.e+3", "rn_ahm_b   = 0.0",
    ):
        assert line in smt4


def _f16(text: str) -> bytes:
    return text.ljust(16).encode("ascii")


def _group(name: str) -> bytes:
    rank = 2 if name == "ssh_kmm" else 3
    shape = (JPI, JPJ) if rank == 2 else (JPI, JPJ, JPK)
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(
        shape, order="F")
    n1, n2 = shape[:2]
    n3 = shape[2] if rank == 3 else 1
    return (struct.pack("=16s4i", _f16(name), rank, n1, n2, n3)
            + values.astype("<f8").tobytes(order="F"))


def _write_stage(path: Path, stage: int) -> Path:
    names = checker._STAGE_FLUX_BY_STAGE[stage]
    header = struct.pack(
        "=15i", 1, 1, stage, 1, 2, 3, 4, JPI, JPJ, JPK,
        len(names), 0, 0, 0, 64,
    )
    path.write_bytes(
        _f16("NEMO_L1_STGFLX1") + header
        + b"".join(_group(name) for name in names))
    return path


@pytest.mark.parametrize("plant", ["header", "field-name", "truncated"])
def test_each_smt4_record_plant_refuses(tmp_path, plant):
    path = _write_stage(
        tmp_path / "oracle_stage_flux_terms_kt00000001_s1.bin", 1)
    with pytest.raises(checker.Refusal):
        checker.parse_record(path, corrupt_header=plant == "header",
                             plant=None if plant == "header" else plant)


def test_existing_all_stage_writer_brackets_momentum_ldf():
    patch = (TOOLS / "stprk3_stage123_flux_record.patch").read_text()
    call = "CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )"
    dump = "CALL vortex_r16_stage_rhs( 'ldf', Krhs, uu, vv )"
    assert patch.index(call) < patch.index(dump)
    source = (TOOLS / "vortex_r16_stage_terms.F90").read_text()
    assert "kstg < 1 .OR. kstg > 3" in source
    assert set(checker._STAGE_FLUX_BY_STAGE) == {1, 2, 3}


def test_shared_driver_creates_each_space_check_target_before_df():
    source = (TOOLS / "run.sh").read_text()
    loop = source.index(
        'for mount in /tmp "$(dirname "$EVIDENCE")" "$NEMO_ROOT"; do')
    mkdir = source.index('mkdir -p "$mount"', loop)
    disk_free = source.index('free_kb=$(df -Pk "$mount"', loop)
    end = source.index("done", disk_free)
    assert loop < mkdir < disk_free < end


def _run_wrapper_with_stub_driver(tmp_path: Path, *args: str):
    repo = tmp_path / "repo"
    wrapper_dir = (repo / "scripts/validate/ocean_fidelity/testcases/"
                   "nemo_testcase_l1_vortex_smt_round25_smt4")
    driver_dir = (repo / "scripts/validate/ocean_fidelity/testcases/"
                  "nemo_testcase_l1_vortex")
    wrapper_dir.mkdir(parents=True)
    driver_dir.mkdir(parents=True)
    evidence = tmp_path / "evidence"
    source = (ROOT / "scripts/validate/ocean_fidelity/testcases/"
              "nemo_testcase_l1_vortex_smt_round25_smt4/run.sh").read_text()
    source = source.replace(
        "root=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/"
        "oracle_vortex_smt4", f"root={evidence}")
    wrapper = wrapper_dir / "run.sh"
    wrapper.write_text(source)
    wrapper.chmod(0o755)
    driver = driver_dir / "run.sh"
    driver.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$CALL_LOG"
if [[ " $* " == *" --run "* ]]; then
  mkdir -p "$EVIDENCE"
  case "$*" in
    *smt4vec100d*)
      printf '{"status": "ADMITTED"}\n' > "$EVIDENCE/vortex_round237_smt4_vec_100d_admission.json" ;;
    *smt4vec*)
      printf '{"status": "ADMITTED"}\n' > "$EVIDENCE/vortex_round237_smt4_vec_admission.json"
      : > "$EVIDENCE/binaries.sha256" ;;
  esac
fi
"""
    )
    driver.chmod(0o755)
    subprocess.run(["git", "init", "-q", repo], check=True)
    call_log = tmp_path / "calls.log"
    env = os.environ.copy()
    env["CALL_LOG"] = str(call_log)
    result = subprocess.run(
        [wrapper, *args], check=False, text=True, capture_output=True, env=env)
    return result, call_log.read_text().splitlines(), evidence


def test_operator_default_executes_both_smt4_acquisition_arms(tmp_path):
    result, calls, evidence = _run_wrapper_with_stub_driver(tmp_path)
    assert result.returncode == 0, result.stderr
    assert calls == ["--variant smt4vec --run",
                     "--variant smt4vec100d --run"]
    assert "ROUND237_SMT4_RECORD_READY" in result.stdout
    assert (evidence / "kt1_10/binaries.sha256").exists()


def test_explicit_smt4_preflight_keeps_both_arms_dry(tmp_path):
    result, calls, evidence = _run_wrapper_with_stub_driver(
        tmp_path, "--preflight")
    assert result.returncode == 0, result.stderr
    assert calls == ["--variant smt4vec", "--variant smt4vec100d"]
    assert "ROUND237_SMT4_PREFLIGHT_PASS" in result.stdout
    assert not evidence.exists()
