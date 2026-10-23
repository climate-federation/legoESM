"""Round 328: exact writer layout, A4/A5, and round-326 program identity."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round328_dynzdf_gate as gate,
)

REPO = Path(__file__).resolve().parents[3]
ACQ = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition"
WRITER = ACQ / "dynzdf_round328_writer.F90"
STUBS = ACQ / "dynzdf_round328_fixture_stubs.F90"
HARNESS = ACQ / "dynzdf_round328_layout_fixture.F90"
PATCHES = (ACQ / "dynzdf_round328.patch", ACQ / "stprk3_stg_round328.patch")
GFORTRAN = Path("/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran")


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
    assert fields["before_u"].shape == (h["nx"], h["ny"], h["nz"])
    assert "wi_input" not in fields


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
