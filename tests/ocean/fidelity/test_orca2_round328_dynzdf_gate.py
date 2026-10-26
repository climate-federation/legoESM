"""Round 328: exact writer layout, A4/A5, and round-326 program identity."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round328_dynzdf_gate as gate,
    nemo_testcase_l4_orca2_round328_dynzdf_replay as replay,
)

REPO = Path(__file__).resolve().parents[3]
ACQ = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition"
WRITER = ACQ / "dynzdf_round328_writer.F90"
STUBS = ACQ / "dynzdf_round328_fixture_stubs.F90"
HARNESS = ACQ / "dynzdf_round328_layout_fixture.F90"
PATCHES = (ACQ / "dynzdf_round328.patch", ACQ / "stprk3_stg_round328.patch")
GFORTRAN = Path("/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran")
RUN_SH = ACQ / "run.sh"


def test_term_localization_separates_iwm_and_ddm_boundaries() -> None:
    closure = np.array([[[1.0, 2.0, 3.0]]])
    after_ddm = np.array([[[1.0, 2.5, 3.0]]])
    consumed = np.array([[[1.25, 2.75, 4.0]]])
    record = np.array([[[1.0, 2.5, 3.5]]])
    result = replay.term_localization(
        consumed, record, after_ddm, closure,
        np.ones_like(consumed, dtype=bool))
    assert result["unequal_by_level"] == [1, 1, 1]
    assert result["record_equals_after_ddm_without_iwm"] == 2
    assert result["record_equals_closure_without_ddm_or_iwm"] == 1
    assert result["iwm_changes_candidate_at_residual"] == 3
    assert result["ddm_changes_candidate_at_residual"] == 1


def test_zdf_term_localization_names_first_boundary_by_level() -> None:
    shape = (1, 2, 2)
    records = {}
    for rank in (0, 1):
        fields = {name: np.ones((1, 1, 2)) for name in (
            "tke_wmask", "b1_avm", "b2_avm", "b3_avm")}
        records[rank] = {"fields": fields}
    candidates = {
        "post_tke": np.ones(shape),
        "post_ddm": np.ones(shape),
        "post_iwm": np.ones(shape),
    }
    candidates["post_tke"][0, 0, 1] = 2.0
    candidates["post_ddm"][0, 0, 1] = 2.0
    candidates["post_iwm"][0, 0, 0] = 2.0
    candidates["post_iwm"][0, 0, 1] = 2.0
    result = replay.zdf_term_localization(candidates, records, np.ones(shape, bool))
    assert result["first_unequal_boundary_by_level"] == ["post_iwm", "post_tke"]


@pytest.fixture(scope="module")
def record_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile and execute the committed harness against the actual writer."""
    root = tmp_path_factory.mktemp("r328-layout")
    subprocess.run([str(GFORTRAN), "-ffree-line-length-none", "-J", str(root),
                    str(STUBS), str(WRITER),
                    str(HARNESS), "-o", str(root / "write_fixture")], check=True)
    for rank in (0, 1):
        subprocess.run([str(root / "write_fixture"), str(rank)], cwd=root, check=True)
    (root / "producer_commit.txt").write_text("a" * 40 + "\n")
    return root


@pytest.mark.skipif(not GFORTRAN.exists(), reason="NEMO gfortran toolchain absent")
def test_compiled_exact_writer_layout_passes(record_dir: Path) -> None:
    result = gate.run_gate(record_dir, "a" * 40)
    assert result["status"] == "PASS_R328_DYNZDF_STRUCTURE"
    first = gate.read_record(record_dir / gate.record_name(0))
    h, fields = first["header"], first["fields"]
    assert fields["avm_input"].shape == (h["nx"] + 1, h["ny"] + 1, h["nz"] + 1)
    assert fields["rCdU_bot"].shape == (h["nx"] + 1, h["ny"] + 1, 1)
    assert fields["r3u_kmm"].shape == (h["nx"], h["ny"], 1)
    assert fields["r3v_kmm"].shape == (h["nx"], h["ny"], 1)
    assert fields["r3u_kaa"].shape == (h["nx"], h["ny"], 1)
    assert fields["r3v_kaa"].shape == (h["nx"], h["ny"], 1)
    assert np.all(fields["r3u_kmm"] == 0.03)
    assert np.all(fields["r3v_kmm"] == 0.04)
    assert np.all(fields["r3u_kaa"] == 0.01)
    assert np.all(fields["r3v_kaa"] == 0.02)
    assert fields["before_u"].shape == (h["nx"], h["ny"], h["nz"])
    assert "wi_input" not in fields


@pytest.mark.skipif(not GFORTRAN.exists(), reason="NEMO gfortran toolchain absent")
def test_a4_ignores_raw_negative_zero_only_at_frame_dry_cells(record_dir: Path) -> None:
    record = gate.read_record(record_dir / gate.record_name(0))["fields"]
    _, state = gate.read_admitted_state(
        record_dir / "oracle_stage_kt00000001_s3.bin", kt=1, stage=3)
    raw_u, frame_u = record["corrected_u"][2, 1, -1], state["u"][2, 1, -2]
    raw_v, frame_v = record["corrected_v"][1, 1, -1], state["v"][1, 1, -2]
    assert raw_u == frame_u == raw_v == frame_v == 0.0
    assert np.signbit(raw_u) and np.signbit(raw_v)
    assert not np.signbit(frame_u) and not np.signbit(frame_v)
    assert gate.run_gate(record_dir, "a" * 40)["status"] == "PASS_R328_DYNZDF_STRUCTURE"


@pytest.mark.skipif(not GFORTRAN.exists(), reason="NEMO gfortran toolchain absent")
def test_a4_wet_cell_one_ulp_plant_refuses(record_dir: Path) -> None:
    with pytest.raises(gate.GateError, match="corrected_u differs"):
        gate.run_gate(record_dir, "a" * 40, "endpoint")


def _cleanup_function() -> str:
    text = RUN_SH.read_text()
    start = text.index("cleanup() {")
    end = text.index("\n}\ntrap cleanup EXIT", start) + 2
    return text[start:end]


def _transport_compare_function() -> str:
    text = RUN_SH.read_text()
    start = text.index("compare_transport_defined() {")
    end = text.index("\n}\n", start) + 2
    return text[start:end]


def _run_transport_compare(reference: Path, candidate: Path) -> int:
    script = "\n".join((
        _transport_compare_function(),
        f"compare_transport_defined {shlex.quote(str(reference))} "
        f"{shlex.quote(str(candidate))}",
    ))
    return subprocess.run(["bash", "-c", script]).returncode


def test_transport_passivity_ignores_only_undefined_zfw(tmp_path: Path) -> None:
    # Writer layout is a 48-byte header followed by equal zFu, zFv, zFw records.
    reference = tmp_path / "reference.bin"
    candidate = tmp_path / "candidate.bin"
    original = bytes(range(48)) + bytes(range(48, 96))
    reference.write_bytes(original)
    changed = bytearray(original)
    changed[80] ^= 1  # third record (zFw), unset at this writer boundary
    candidate.write_bytes(changed)
    assert _run_transport_compare(reference, candidate) == 0

    changed[48] ^= 1  # first record (zFu), a production-consumed operand
    candidate.write_bytes(changed)
    assert _run_transport_compare(reference, candidate) != 0


@pytest.mark.parametrize("completed", [False, True])
def test_cleanup_keeps_crashed_and_admission_refused_runs(
        tmp_path: Path, completed: bool) -> None:
    evidence = tmp_path / "evidence"
    target_run = evidence / "active_run"
    target_root = tmp_path / "target_cfg"
    scratch = tmp_path / "scratch"
    target_run.mkdir(parents=True)
    target_root.mkdir()
    scratch.mkdir()
    (target_run / "record.bin").write_bytes(b"correct record")
    script = "\n".join((
        _cleanup_function(),
        f"scratch={shlex.quote(str(scratch))}",
        f"TARGET_RUN={shlex.quote(str(target_run))}",
        f"TARGET_ROOT={shlex.quote(str(target_root))}",
        f"EVIDENCE={shlex.quote(str(evidence))}",
        "created_target=1", "created_run=1",
        f"run_completed={int(completed)}",
        "false", "cleanup",
    ))
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
    assert result.returncode == 1
    assert not target_run.exists() and target_root.is_dir() and not scratch.exists()
    assert "R328_FAILED_BUILD_KEPT" in result.stderr
    assert "rerun=move-this-directory-aside-first" in result.stderr
    failure_kind = "admission_refused" if completed else "crashed"
    kept = list((evidence / "refused").glob(
        f"active_run_{failure_kind}_*/record.bin"))
    assert len(kept) == 1
    assert kept[0].read_bytes() == b"correct record"
    assert f"R328_FAILED_OUTPUTS_KEPT kind={failure_kind}" in result.stderr


PLANT_REASONS = {
    "header": "wrong version/kt/stage",
    "field-order": "field order",
    "truncation": "truncated",
    "stamp": "producer commit",
    "avm": "avm_input: payload checksum",
    "bottom": "rCdU_bot: payload checksum",
    "top": "rCdU_top: payload checksum",
    "solve": "solve_u: payload checksum",
    "endpoint": "corrected_u differs from the admitted stage-3 endpoint",
}


def test_plants_cover_frozen_a4_a5() -> None:
    assert set(PLANT_REASONS) == set(gate.PLANTS)
    assert {"avm", "bottom", "top", "solve"} <= set(gate.PLANTS)


@pytest.mark.parametrize("plant", gate.PLANTS)
def test_every_plant_refuses_by_its_registered_check(record_dir: Path, plant: str) -> None:
    with pytest.raises(gate.GateError, match=re.escape(PLANT_REASONS[plant])):
        gate.run_gate(record_dir, "a" * 40, plant)


def test_a5_plants_are_exactly_one_ulp(record_dir: Path) -> None:
    fields = gate.read_record(record_dir / gate.record_name(0))["fields"]
    for name in ("avm_input", "rCdU_bot", "rCdU_top", "solve_u"):
        planted = gate._one_ulp(fields[name])
        before = fields[name].flat[0]
        assert planted.flat[0] == np.nextafter(before, np.float64(np.inf))
        assert np.count_nonzero(planted.view(np.uint64) != fields[name].view(np.uint64)) == 1


def _insertions(path: Path) -> list[str]:
    return [line[1:] for line in path.read_text().splitlines()
            if line.startswith("+") and not line.startswith("+++")]


def _build(root: Path, cfg: str, observer: bool) -> Path:
    (root / "ppsrc/nemo").mkdir(parents=True)
    (root / "obj").mkdir()
    for name in gate.BUILD_CONFIG_FILES:
        (root / name).write_text(f"dir::root /cfgs/{cfg}/BLD\n%FCFLAGS -O3\n")
    (root / "ppsrc/nemo/other.f90").write_text("MODULE other\nEND MODULE other\n")
    (root / "obj/other.o").write_bytes(b"same non-ELF object")
    for unit, patch in zip(gate.OBSERVER_UNITS, PATCHES, strict=True):
        lines = [f"MODULE {unit}", f"END MODULE {unit}"]
        if observer:
            lines[1:1] = _insertions(patch)
        (root / f"ppsrc/nemo/{unit}.f90").write_text("\n".join(lines) + "\n")
        (root / f"obj/{unit}.o").write_bytes(f"{unit}-{observer}".encode())
    if observer:
        (root / f"ppsrc/nemo/{gate.WRITER_UNIT}.f90").write_text("MODULE writer\n")
        (root / f"obj/{gate.WRITER_UNIT}.o").write_bytes(b"writer")
    return root


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    return _build(tmp_path / "ref", "R326", False), _build(tmp_path / "new", "R328", True)


def test_same_program_includes_both_observer_units(tmp_path: Path) -> None:
    ref, new = _pair(tmp_path)
    result = gate.same_program(new, ref, PATCHES, new_cfg="R328", ref_cfg="R326")
    assert result["status"] == "PASS_SAME_PROGRAM_R326_PLUS_OBSERVERS"
    assert result["observer_units"] == ["dynzdf", "stprk3_stg"]


@pytest.mark.parametrize("plant, reason", [
    ("other-source", "ppsrc/nemo/other.f90"),
    ("other-object", "obj/other.o"),
    ("stage-edit", "stprk3_stg.f90 replaces"),
    ("flags", "build config differs"),
])
def test_same_program_refuses_nonobserver_differences(tmp_path: Path, plant: str, reason: str) -> None:
    ref, new = _pair(tmp_path)
    if plant == "other-source":
        (new / "ppsrc/nemo/other.f90").write_text("changed\n")
    elif plant == "other-object":
        (new / "obj/other.o").write_bytes(b"changed")
    elif plant == "stage-edit":
        path = new / "ppsrc/nemo/stprk3_stg.f90"
        path.write_text(path.read_text().replace("MODULE stprk3_stg", "MODULE wrong"))
    else:
        (new / "arch_nemo.fcm").write_text("%FCFLAGS -O0\n")
    with pytest.raises(gate.GateError, match=re.escape(reason)):
        gate.same_program(new, ref, PATCHES, new_cfg="R328", ref_cfg="R326")
