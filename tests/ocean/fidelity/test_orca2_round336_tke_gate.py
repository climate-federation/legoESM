"""Round 336 kt=2 extensions of the reviewed TKE recorders."""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round331_tke_gate as r331,
    nemo_testcase_l4_orca2_round336_tke_gate as gate,
)

REPO = Path(__file__).resolve().parents[3]
DYN = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition"
TKE = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round331_tke_acquisition"
R332 = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round332_source_order_acquisition"
GFORTRAN = Path("/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran")


@pytest.fixture(scope="module")
def record_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("r336-tke-kt2")
    commands = (
        (TKE / "tke_round331_writer.F90", TKE / "tke_round331_layout_fixture.F90", "tke"),
        (R332 / "round332_writer.F90", R332 / "round332_layout_fixture.F90", "mxl"),
    )
    for writer, fixture, name in commands:
        subprocess.run([
            str(GFORTRAN), "-fcheck=bounds", "-ffree-line-length-none",
            "-I", str(root), "-J", str(root),
            str(DYN / "dynzdf_round328_fixture_stubs.F90"),
            str(DYN / "dynzdf_round328_writer.F90"), str(writer),
            str(fixture), "-o", str(root / name),
        ], check=True)
    for rank in (0, 1):
        for name in ("tke", "mxl"):
            subprocess.run([str(root / name), str(rank)], cwd=root, check=True)
    return root


@pytest.mark.skipif(not GFORTRAN.exists(), reason="NEMO gfortran toolchain absent")
def test_real_writers_emit_complete_kt2_records(record_dir: Path) -> None:
    for rank in (0, 1):
        tke = r331.read_record(record_dir / r331.record_name(rank, kt=2),
                               expected_kt=2)
        r331._verify_checksums(tke)
        mxl = gate.read_record(record_dir / gate.record_name(rank))
        phase = {"fields": {"avt_after_tke": tke["fields"]["after_tke_avt"].copy()}}
        result = gate.verify_records(tke, mxl, phase)
        assert result["final_avt_unequal"] == 0
        assert result["wet_interfaces"] > 0


@pytest.mark.parametrize("plant", ["header", "field-order", "truncation", "checksum", "time-level"])
def test_layout_plants_refuse(record_dir: Path, plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.read_record(record_dir / gate.record_name(0), plant=plant)


@pytest.mark.parametrize("plant", ["final", "mxl", "pdlr"])
def test_semantic_plants_refuse_on_wet_cells(record_dir: Path, plant: str) -> None:
    tke = r331.read_record(record_dir / r331.record_name(0, kt=2), expected_kt=2)
    mxl = gate.read_record(record_dir / gate.record_name(0))
    phase = {"fields": {"avt_after_tke": tke["fields"]["after_tke_avt"].copy()}}
    with pytest.raises(gate.GateError):
        gate.verify_records(copy.deepcopy(tke), copy.deepcopy(mxl), phase, plant=plant)


def test_kt1_record_format_remains_admissible(record_dir: Path) -> None:
    record = r331.read_record(record_dir / r331.record_name(0))
    assert record["header"]["kt"] == 1
    assert np.all(np.isfinite(record["fields"]["after_tke_avt"]))


def test_wrapper_pins_distinct_build_and_record_targets() -> None:
    text = (REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh").read_text()
    assert "ORCA2_ORCA1ICE_OMIP_L4_R336TKE" in text
    assert "orca1ice_tke_internal_kt2_np2" in text
    assert "R328_EXISTING_RUN" in (REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/README.md").read_text()
