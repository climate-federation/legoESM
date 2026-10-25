"""Acquisition-tool controls for SI3 ladder rung O5 (round 12)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
    assert report["header_bytes"] == 112
    assert report["signed_zero"]["ulp_wet_negzero"] != 0
    assert report["signed_zero"]["ulp_wet_1ulp"] == 1


@pytest.mark.parametrize(
    "plant", ["header", "truncation", "checksum", "time-level", "wet-1ulp", "swapped-field"]
)
def test_binary_and_admission_plants_refuse(plant):
    with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
        gate.run_self_test(plant)


def test_reader_uses_declared_interior_and_initialized_slice_extents(tmp_path):
    path = tmp_path / "o5_kind11_step0001_iter000_rank0000.bin"
    gate._write_synthetic_record(path, 11)
    fields = gate.read_o5_record(path)["fields"]
    assert fields["stress1"].shape == (8, 7)
    assert fields["divu"].shape == (4, 3)
    assert fields["shear"].shape == (4, 3)
    assert fields["delta"].shape == (4, 3)
    masks_path = tmp_path / "o5_kind05_step0001_iter000_rank0000.bin"
    gate._write_synthetic_record(masks_path, 5)
    masks = gate.read_o5_record(masks_path)["fields"]
    assert masks["fast_umask"].shape == (7, 6)
    assert masks["fast_vmask"].shape == (7, 6)


def test_reader_refuses_nonfinite_payloads(tmp_path):
    path = tmp_path / "o5_kind11_step0001_iter000_rank0000.bin"
    gate._write_synthetic_record(path, 11)
    raw = bytearray(path.read_bytes())
    raw[gate.HEADER_BYTES : gate.HEADER_BYTES + 8] = b"\x00\x00\x00\x00\x00\x00\xf8\x7f"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="non-finite payload"):
        gate.read_o5_record(path)


def test_sas_time_level_is_derived_from_nemo_sources():
    nemo = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
    assert gate.derive_sas_time_level(nemo / "src/SAS/nemogcm.F90", nemo / "src/SAS/step.F90") == 2


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
    assert "PASS_PATCHED_PPSRC_FIXTURE_COVERAGE_AND_12_MISPLACEMENT_PLANTS" in result.stdout
    assert "PASS_REAL_PREPROCESSED_SAME_PROGRAM_AND_INSERTION_PLANT" in result.stdout
    assert before == after


def test_operator_only_block_has_one_rank_timeout_and_real_flag_census():
    text = (ACQ / "run.sh").read_text()
    operator = text.split("# ---- Operator-only --run mode ----", 1)[1]
    assert 'timeout "$RUN_TIMEOUT_SEC" mpirun -np 1' in operator
    assert "-O0" in text and "PASS_FP_CENSUS_REAL_FLAGS" in text
    assert operator.index('build_target "$REF_CFG" 0') < operator.index("same_program")
    assert operator.index("same_program") < operator.index('stage_run "$REF_CFG"')


def test_refusal_retains_real_built_trees_for_diagnosis():
    text = (ACQ / "run.sh").read_text()
    cleanup = text.split("cleanup() {", 1)[1].split("trap cleanup EXIT", 1)[0]
    assert 'REFUSE_RETAINED_BUILD: %s' in cleanup
    assert 'rm -rf -- "$REF_ROOT"' not in cleanup
    assert 'rm -rf -- "$REC_ROOT"' not in cleanup


def test_admit_existing_accepts_retained_run_from_older_ancestor_commit(
    tmp_path, monkeypatch
):
    script = (ACQ / "run.sh").read_text()
    retained = script.split('if [[ "$mode" == --admit-existing ]]', 1)[1].split(
        "# ---- Operator-only --run mode ----", 1
    )[0]
    assert "existing_commit=$(tr -d '[:space:]'" in retained
    assert 'admit "$existing_commit"' in retained
    assert 'git -C "$REPO" rev-parse HEAD' not in retained

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    producer = subprocess.run(
        ["git", "rev-parse", "HEAD^"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    assert producer != head
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", producer, head], cwd=ROOT, check=True
    )

    evidence = tmp_path / "retained"
    manifest = evidence / "manifest"
    ref = evidence / "reference_3step"
    record = evidence / "record_3step"
    for directory in (manifest / "binaries", ref, record):
        directory.mkdir(parents=True, exist_ok=True)
    (manifest / "producer_commit.txt").write_text(producer + "\n")
    (manifest / "same_program.json").write_text(json.dumps({"status": "PASS_SAME_PROGRAM"}))
    (manifest / "fp_census.json").write_text(
        json.dumps({"status": "PASS_FP_CENSUS_REAL_FLAGS"})
    )
    binary_rows = []
    for name, payload, run_path in (
        ("nemo.ref", b"synthetic retained reference binary", ref / "nemo"),
        ("nemo.rec", b"synthetic retained record binary", record / "nemo"),
    ):
        retained_binary = manifest / "binaries" / name
        retained_binary.write_bytes(payload)
        run_path.write_bytes(payload)
        binary_rows.append(
            f"{hashlib.sha256(payload).hexdigest()}  binaries/{name}\n"
        )
    (manifest / "binaries.sha256").write_text("".join(binary_rows))
    raw_record = record / "synthetic-record.bin"
    raw_record.write_bytes(b"synthetic retained record")
    (record / "records.sha256").write_text(
        f"{hashlib.sha256(raw_record.read_bytes()).hexdigest()}  {raw_record.name}\n"
    )
    for name in (
        "ORCA2_SAS_00000003_restart.nc",
        "ORCA2_SAS_00000003_restart_ice.nc",
    ):
        (ref / name).write_bytes(b"synthetic matching restart")
        (record / name).write_bytes(b"synthetic matching restart")

    monkeypatch.setattr(gate, "_verify_run_completion_and_logs", lambda *args, **kwargs: None)
    monkeypatch.setattr(gate, "verify_o5_log", lambda *args, **kwargs: {"status": "PASS"})
    monkeypatch.setattr(gate, "derive_sas_time_level", lambda *args, **kwargs: 2)
    monkeypatch.setattr(gate, "_load_step", lambda *args, **kwargs: {})
    monkeypatch.setattr(gate, "_score_step", lambda *args, **kwargs: {"status": "PASS"})

    admitted = gate.admit_evidence_directory(evidence, expect_commit=producer)
    assert admitted["status"] == "PASS_O5_ADMISSION"
    assert admitted["producer_commit"] == producer
    assert admitted["binaries_checked"] == 2


def test_writer_uses_stream_records_and_direct_array_writes():
    text = (ACQ / "o5_si3_writer.F90").read_text()
    assert "ACCESS='STREAM'" in text
    assert "STATUS='REPLACE'" in text
    assert "WRITE(iu) a_i; WRITE(iu) v_i" in text
    assert "record_magic = 'O5SI3R01'" in text
    assert "SUM(" not in text and "EXP(" not in text
