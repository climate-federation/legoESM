"""Fail-closed controls for round 224's SMT-3 tracer boundary record."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import struct
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
CHECKER = (ROOT / "scripts/validate/ocean_fidelity/testcases/"
           "nemo_testcase_l1_vortex/check_records.py")
_SPEC = importlib.util.spec_from_file_location("round224_checker", CHECKER)
assert _SPEC and _SPEC.loader
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

JPI, JPJ, JPK = 8, 7, 3


def _f16(text: str) -> bytes:
    assert len(text) <= 16
    return text.ljust(16).encode("ascii")


def _group(name: str, rank: int) -> bytes:
    shape = (JPI, JPJ) if rank == 2 else (JPI, JPJ, JPK)
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(
        shape, order="F")
    n1, n2 = shape[:2]
    n3 = shape[2] if rank == 3 else 1
    return (struct.pack("=16s4i", _f16(name), rank, n1, n2, n3)
            + values.astype("<f8").tobytes(order="F"))


def _write(path: Path, stage: int, *, drop: str | None = None,
           declared: int | None = None) -> Path:
    names = list(checker._SMT3_TRACER_TERM_BY_STAGE[stage])
    if drop:
        names.remove(drop)
    body = b"".join(
        _group(name, 2 if name.startswith("r3t_") else 3)
        for name in names
    )
    count = len(names) if declared is None else declared
    header = struct.pack(
        "=15i", 1, 1, stage, 1, 2, 3, 4, JPI, JPJ, JPK,
        count, 0, 0, 0, 64,
    )
    path.write_bytes(_f16("NEMO_L1_TRATRM1") + header + body)
    return path


def test_smt3_record_requires_ldf_only_at_stage_three(tmp_path):
    for stage in (1, 2, 3):
        record = checker.parse_record(
            _write(tmp_path /
                   f"oracle_tracer_terms_kt00000001_s{stage}.bin", stage))
        assert set(record["groups"]) == set(
            checker._SMT3_TRACER_TERM_BY_STAGE[stage])
    assert "ldf_t" not in checker._SMT3_TRACER_TERM_BY_STAGE[2]
    assert "ldf_t" in checker._SMT3_TRACER_TERM_BY_STAGE[3]


def test_smt3_record_refuses_a_missing_ldf_boundary(tmp_path):
    path = _write(tmp_path / "oracle_tracer_terms_kt00000001_s3.bin", 3,
                  drop="ldf_t", declared=17)
    with pytest.raises(checker.Refusal, match="missing group"):
        checker.parse_record(path)


@pytest.mark.parametrize("plant", ["header", "field-name", "truncated"])
def test_each_smt3_record_plant_refuses(tmp_path, plant):
    path = _write(tmp_path / "oracle_tracer_terms_kt00000001_s3.bin", 3)
    with pytest.raises(checker.Refusal):
        checker.parse_record(path, corrupt_header=plant == "header",
                             plant=None if plant == "header" else plant)


def test_instrument_places_the_boundary_after_compiled_tra_ldf():
    patch = (ROOT / "scripts/validate/ocean_fidelity/testcases/"
             "nemo_testcase_l1_vortex/"
             "stprk3_smt3_tracer_terms_record.patch").read_text()
    call = "CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )"
    dump = "CALL vortex_r18_tracer_rhs( 'ldf', Krhs, ts )"
    assert patch.index(call) < patch.index(dump)


def test_writer_declares_the_extra_groups_only_for_stage_three():
    source = (ROOT / "scripts/validate/ocean_fidelity/testcases/"
              "nemo_testcase_l1_vortex/"
              "vortex_r18_tracer_terms.F90").read_text()
    assert "IF( ll_ldf .AND. kstg == 3 ) ingroups = 17" in source
    assert "jpi, jpj, jpk, ingroups" in source


def _run_wrapper_with_stub_driver(tmp_path: Path, *args: str):
    repo = tmp_path / "repo"
    wrapper_dir = (repo / "scripts/validate/ocean_fidelity/testcases/"
                   "nemo_testcase_l1_vortex_smt_round12_smt3")
    driver_dir = (repo / "scripts/validate/ocean_fidelity/testcases/"
                  "nemo_testcase_l1_vortex")
    wrapper_dir.mkdir(parents=True)
    driver_dir.mkdir(parents=True)
    evidence = tmp_path / "evidence"
    source = (ROOT / "scripts/validate/ocean_fidelity/testcases/"
              "nemo_testcase_l1_vortex_smt_round12_smt3/run.sh").read_text()
    source = source.replace(
        "root=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round224/"
        "oracle_vortex_smt3",
        f"root={evidence}",
    )
    wrapper = wrapper_dir / "run.sh"
    wrapper.write_text(source)
    wrapper.chmod(0o755)
    driver = driver_dir / "run.sh"
    driver.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$CALL_LOG"
mkdir -p "$(dirname "$EVIDENCE")"
if [[ " $* " == *" --run "* ]]; then
  mkdir -p "$EVIDENCE"
  case "$*" in
    *smt3vec100d*)
      printf '{"status": "ADMITTED"}\\n' > "$EVIDENCE/vortex_round224_smt3_vec_100d_admission.json" ;;
    *smt3vec*)
      printf '{"status": "ADMITTED"}\\n' > "$EVIDENCE/vortex_round224_smt3_vec_admission.json"
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
        [wrapper, *args], check=False, text=True, capture_output=True, env=env,
    )
    return result, call_log.read_text().splitlines(), evidence


def test_operator_default_executes_both_acquisition_arms(tmp_path):
    result, calls, evidence = _run_wrapper_with_stub_driver(tmp_path)
    assert result.returncode == 0, result.stderr
    assert calls == ["--variant smt3vec --run",
                     "--variant smt3vec100d --run"]
    assert "ROUND224_SMT3_RECORD_READY" in result.stdout
    assert (evidence / "kt1_10/binaries.sha256").exists()


def test_explicit_preflight_keeps_both_arms_dry(tmp_path):
    result, calls, evidence = _run_wrapper_with_stub_driver(
        tmp_path, "--preflight")
    assert result.returncode == 0, result.stderr
    assert calls == ["--variant smt3vec", "--variant smt3vec100d"]
    assert "ROUND224_SMT3_PREFLIGHT_PASS" in result.stdout
    assert evidence.is_dir()
    assert list(evidence.iterdir()) == []
