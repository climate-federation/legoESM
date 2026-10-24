"""Round 331: TKE writer layout, controls, and program identity."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round328_dynzdf_gate as dyn_gate,
    nemo_testcase_l4_orca2_round331_tke_gate as gate,
)

REPO = Path(__file__).resolve().parents[3]
DYN_ACQ = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition"
ZDF_ACQ = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_zdf_terms_acquisition"
ACQ = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round331_tke_acquisition"
STUBS = DYN_ACQ / "dynzdf_round328_fixture_stubs.F90"
DYN_WRITER = DYN_ACQ / "dynzdf_round328_writer.F90"
DYN_FIXTURE = DYN_ACQ / "dynzdf_round328_layout_fixture.F90"
ZDF_WRITER = ZDF_ACQ / "zdfphy_round328_writer.F90"
ZDF_FIXTURE = ZDF_ACQ / "zdfphy_round328_layout_fixture.F90"
WRITER = ACQ / "tke_round331_writer.F90"
FIXTURE = ACQ / "tke_round331_layout_fixture.F90"
PATCHES = {
    "dynzdf": DYN_ACQ / "dynzdf_round328.patch",
    "stprk3_stg": DYN_ACQ / "stprk3_stg_round328.patch",
    "zdfphy": ACQ / "zdfphy_round331.patch",
    "zdftke": ACQ / "zdftke_round331.patch",
}
GFORTRAN = Path("/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran")


@pytest.fixture(scope="module")
def record_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("r331-tke-layout")
    commands = (
        (DYN_WRITER, DYN_FIXTURE, "write_dyn_fixture"),
        (DYN_WRITER, ZDF_WRITER, ZDF_FIXTURE, "write_zdf_fixture"),
        (DYN_WRITER, WRITER, FIXTURE, "write_tke_fixture"),
    )
    for command in commands:
        *sources, output = command
        subprocess.run([
            str(GFORTRAN), "-ffree-line-length-none", "-I", str(root),
            "-J", str(root), str(STUBS), *(str(path) for path in sources),
            "-o", str(root / output),
        ], check=True)
    for rank in (0, 1):
        for executable in ("write_dyn_fixture", "write_zdf_fixture", "write_tke_fixture"):
            subprocess.run([str(root / executable), str(rank)], cwd=root, check=True)
    (root / "producer_commit.txt").write_text("a" * 40 + "\n")
    return root


@pytest.mark.skipif(not GFORTRAN.exists(), reason="NEMO gfortran toolchain absent")
def test_real_writer_layout_and_clean_admission(record_dir: Path) -> None:
    result = gate.run_gate(record_dir, record_dir, "a" * 40)
    assert result["status"] == "PASS_R331_TKE_ADMISSION"
    record = gate.read_record(record_dir / gate.record_name(0))
    h, fields = record["header"], record["fields"]
    assert tuple(fields) == gate.EXPECTED_ORDER
    assert fields["sh2_output"].shape == (h["nx"], h["ny"], h["nz"])
    assert fields["after_bc_en"].shape == (h["nx"], h["ny"], h["nz"] + 1)
    assert np.array_equal(fields["sh2_output"], fields["tke_sh2_input"])
    assert np.array_equal(fields["tke_avm_input"], fields["after_tke_avm"])


def test_fixture_rank_payloads_are_distinct(record_dir: Path) -> None:
    rank0 = gate.read_record(record_dir / gate.record_name(0))["fields"]
    rank1 = gate.read_record(record_dir / gate.record_name(1))["fields"]
    assert not np.array_equal(rank0["after_tke_en"], rank1["after_tke_en"])


PLANT_REASONS = {
    "header": "wrong version/kt/RK indices",
    "field-order": "field order",
    "truncation": "truncated",
    "stamp": "producer commit",
    "sh2": "sh2_output differs at TKE entry",
    "drag": "drag_bottom_output differs at TKE entry",
    "mxl": "mxl_nmln_output differs at TKE entry",
    "boundary": "surface TKE changed between boundary and matrix",
    "matrix": "surface TKE changed between boundary and matrix",
    "solve": "surface TKE changed during the interior solve",
    "final": "after_tke_avm differs from admitted zdf_phy",
}


@pytest.mark.parametrize("plant", gate.PLANTS)
def test_every_required_plant_refuses(record_dir: Path, plant: str) -> None:
    with pytest.raises(gate.GateError, match=re.escape(PLANT_REASONS[plant])):
        gate.run_gate(record_dir, record_dir, "a" * 40, plant)


def test_semantic_plants_are_one_ulp(record_dir: Path) -> None:
    mapping = {
        "sh2": "sh2_output", "drag": "drag_bottom_output",
        "mxl": "mxl_nmln_output", "boundary": "after_bc_en",
        "matrix": "matrix_rhs_en", "solve": "after_solve_en",
        "final": "after_tke_avm",
    }
    for plant, name in mapping.items():
        record = gate.read_record(record_dir / gate.record_name(0))
        before = record["fields"][name].copy()
        gate._plant(record, plant)
        changed = record["fields"][name].view(np.uint64) != before.view(np.uint64)
        assert np.count_nonzero(changed) == 1
        wmask = record["fields"]["tke_wmask"] != 0.0
        if name in gate.TWO_D_FIELDS:
            wet = wmask[:, :, :1]
        elif name in {"after_bc_en", "matrix_rhs_en", "after_solve_en"}:
            wet = np.zeros_like(changed, dtype=bool)
            wet[:, :, :1] = wmask[:, :, :1]
        else:
            wet = wmask[:, :, 1:]
        assert bool(wet[changed][0])


def test_fixture_dry_negative_zero_cannot_hide_plants(record_dir: Path) -> None:
    record = gate.read_record(record_dir / gate.record_name(0))
    dry_column = record["fields"]["tke_wmask"][0, 0, :]
    assert np.all(dry_column == 0.0)
    assert np.all(np.signbit(dry_column))
    for plant in ("sh2", "drag", "mxl", "boundary", "matrix", "solve", "final"):
        name = {
            "sh2": "sh2_output", "drag": "drag_bottom_output",
            "mxl": "mxl_nmln_output", "boundary": "after_bc_en",
            "matrix": "matrix_rhs_en", "solve": "after_solve_en",
            "final": "after_tke_avm",
        }[plant]
        before = record["fields"][name][0, 0, 0].view(np.uint64).copy()
        gate._plant(record, plant)
        assert record["fields"][name][0, 0, 0].view(np.uint64) == before


def _insertions(path: Path) -> list[str]:
    return [line[1:] for line in path.read_text().splitlines()
            if line.startswith("+") and not line.startswith("+++")]


def _build(root: Path, cfg: str, observer: bool) -> Path:
    (root / "ppsrc/nemo").mkdir(parents=True)
    (root / "obj").mkdir()
    for name in dyn_gate.BUILD_CONFIG_FILES:
        (root / name).write_text(f"dir::root /cfgs/{cfg}/BLD\n%FCFLAGS -O3\n")
    (root / "ppsrc/nemo/other.f90").write_text("MODULE other\nEND MODULE other\n")
    (root / "obj/other.o").write_bytes(b"same non-ELF object")
    for unit, patch in PATCHES.items():
        lines = [f"MODULE {unit}", f"END MODULE {unit}"]
        if observer:
            lines[1:1] = _insertions(patch)
        (root / f"ppsrc/nemo/{unit}.f90").write_text("\n".join(lines) + "\n")
        (root / f"obj/{unit}.o").write_bytes(f"{unit}-{observer}".encode())
    if observer:
        for unit in gate.WRITER_UNITS:
            (root / f"ppsrc/nemo/{unit}.f90").write_text("MODULE writer\n")
            (root / f"obj/{unit}.o").write_bytes(b"writer")
    return root


def test_same_program_import_covers_all_observers_and_writers(tmp_path: Path) -> None:
    ref = _build(tmp_path / "ref", "R326", False)
    new = _build(tmp_path / "new", "R331", True)
    result = gate.same_program(new, ref, PATCHES, new_cfg="R331", ref_cfg="R326")
    assert result["observer_units"] == list(gate.OBSERVER_UNITS)
    assert result["writer_units"] == list(gate.WRITER_UNITS)


@pytest.mark.parametrize("plant,reason", [
    ("source", "ppsrc/nemo/other.f90"),
    ("object", "obj/other.o"),
    ("observer", "zdftke.f90 replaces"),
    ("flags", "build config differs"),
])
def test_same_program_refuses_nonobserver_change(
        tmp_path: Path, plant: str, reason: str) -> None:
    ref = _build(tmp_path / "ref", "R326", False)
    new = _build(tmp_path / "new", "R331", True)
    if plant == "source":
        (new / "ppsrc/nemo/other.f90").write_text("changed\n")
    elif plant == "object":
        (new / "obj/other.o").write_bytes(b"changed")
    elif plant == "observer":
        path = new / "ppsrc/nemo/zdftke.f90"
        path.write_text(path.read_text().replace("MODULE zdftke", "MODULE wrong"))
    else:
        (new / "arch_nemo.fcm").write_text("%FCFLAGS -O0\n")
    with pytest.raises(gate.GateError, match=re.escape(reason)):
        gate.same_program(new, ref, PATCHES, new_cfg="R331", ref_cfg="R326")
