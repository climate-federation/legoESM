"""Round 328: zdf_phy exact layout, boundary plants, and program identity."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round328_dynzdf_gate as dyn_gate,
    nemo_testcase_l4_orca2_round328_zdf_terms_gate as gate,
)

REPO = Path(__file__).resolve().parents[3]
DYN_ACQ = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition"
ACQ = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_zdf_terms_acquisition"
WRITER = ACQ / "zdfphy_round328_writer.F90"
STUBS = ACQ / "zdfphy_round328_fixture_stubs.F90"
FIXTURE = ACQ / "zdfphy_round328_layout_fixture.F90"
PATCHES = {
    "dynzdf": DYN_ACQ / "dynzdf_round328.patch",
    "stprk3_stg": DYN_ACQ / "stprk3_stg_round328.patch",
    "zdfphy": ACQ / "zdfphy_round328.patch",
    "zdfddm": ACQ / "zdfddm_round328.patch",
    "zdfiwm": ACQ / "zdfiwm_round328.patch",
}
GFORTRAN = Path("/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran")


@pytest.fixture(scope="module")
def record_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("r328-zdfphy-layout")
    subprocess.run([
        str(GFORTRAN), "-ffree-line-length-none", "-J", str(root),
        str(STUBS), str(WRITER), str(FIXTURE), "-o", str(root / "write_fixture"),
    ], check=True)
    for rank in (0, 1):
        subprocess.run([str(root / "write_fixture"), str(rank)], cwd=root, check=True)
    (root / "producer_commit.txt").write_text("a" * 40 + "\n")
    return root


@pytest.mark.skipif(not GFORTRAN.exists(), reason="NEMO gfortran toolchain absent")
def test_real_writer_layout_and_clean_admission(record_dir: Path) -> None:
    result = gate.run_gate(record_dir, "a" * 40)
    assert result["status"] == "PASS_R328_ZDFPHY_ADMISSION"
    record = gate.read_record(record_dir / gate.record_name(0))
    h, fields = record["header"], record["fields"]
    assert tuple(fields) == gate.EXPECTED_ORDER
    assert fields["b1_avm"].shape == (h["nx"], h["ny"], h["nz"])
    assert fields["tke_ht_0"].shape == (h["nx"], h["ny"], 1)
    assert np.array_equal(fields["b1_avm"], fields["tke_avm_k"])


PLANT_REASONS = {
    "header": "wrong version/kt/RK indices",
    "field-order": "field order",
    "truncation": "truncated",
    "stamp": "producer commit",
    "boundary1": "b1_avm: payload checksum",
    "boundary2": "b2_avm: payload checksum",
    "boundary3": "b3_avm: payload checksum",
}


@pytest.mark.parametrize("plant", gate.PLANTS)
def test_every_required_plant_refuses(record_dir: Path, plant: str) -> None:
    with pytest.raises(gate.GateError, match=re.escape(PLANT_REASONS[plant])):
        gate.run_gate(record_dir, "a" * 40, plant)


def test_boundary_plants_are_one_ulp(record_dir: Path) -> None:
    fields = gate.read_record(record_dir / gate.record_name(0))["fields"]
    for name in ("b1_avm", "b2_avm", "b3_avm"):
        planted = gate._one_ulp(fields[name])
        assert np.count_nonzero(planted.view(np.uint64) != fields[name].view(np.uint64)) == 1


def test_final_check_ignores_only_dry_signed_zero(record_dir: Path) -> None:
    record = gate.read_record(record_dir / gate.record_name(0))
    admitted = gate.read_admitted_final(record_dir / "oracle_zdf_entry_kt00000001.bin")
    dry = ~record["fields"]["tke_wmask"].astype(bool)
    assert np.any(dry)
    assert np.any(np.signbit(record["fields"]["b3_avm"][dry]))
    assert not np.any(np.signbit(admitted["avm"][:, :, 1:4][dry]))
    gate._verify_final(record, admitted, False)
    with pytest.raises(gate.GateError, match="post-IWM avm differs"):
        gate._verify_final(record, admitted, True)


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
    new = _build(tmp_path / "new", "R328ZP", True)
    result = gate.same_program(new, ref, PATCHES, new_cfg="R328ZP", ref_cfg="R326")
    assert result["observer_units"] == list(gate.OBSERVER_UNITS)
    assert result["writer_units"] == list(gate.WRITER_UNITS)


@pytest.mark.parametrize("plant, reason", [
    ("source", "ppsrc/nemo/other.f90"),
    ("object", "obj/other.o"),
    ("observer", "zdfphy.f90 replaces"),
    ("flags", "build config differs"),
])
def test_same_program_refuses_nonobserver_change(
        tmp_path: Path, plant: str, reason: str) -> None:
    ref = _build(tmp_path / "ref", "R326", False)
    new = _build(tmp_path / "new", "R328ZP", True)
    if plant == "source":
        (new / "ppsrc/nemo/other.f90").write_text("changed\n")
    elif plant == "object":
        (new / "obj/other.o").write_bytes(b"changed")
    elif plant == "observer":
        path = new / "ppsrc/nemo/zdfphy.f90"
        path.write_text(path.read_text().replace("MODULE zdfphy", "MODULE wrong"))
    else:
        (new / "arch_nemo.fcm").write_text("%FCFLAGS -O0\n")
    with pytest.raises(gate.GateError, match=re.escape(reason)):
        gate.same_program(new, ref, PATCHES, new_cfg="R328ZP", ref_cfg="R326")
