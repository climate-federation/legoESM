"""Round 332: compiled writer layout, source-order controls, and passivity."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round332_source_order_gate as gate,
)

REPO = Path(__file__).resolve().parents[3]
DYN_ACQ = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition"
ZDF_ACQ = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_zdf_terms_acquisition"
TKE_ACQ = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round331_tke_acquisition"
ACQ = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round332_source_order_acquisition"
STUBS = DYN_ACQ / "dynzdf_round328_fixture_stubs.F90"
DYN_WRITER = DYN_ACQ / "dynzdf_round328_writer.F90"
DYN_FIXTURE = DYN_ACQ / "dynzdf_round328_layout_fixture.F90"
ZDF_WRITER = ZDF_ACQ / "zdfphy_round328_writer.F90"
ZDF_FIXTURE = ZDF_ACQ / "zdfphy_round328_layout_fixture.F90"
TKE_WRITER = TKE_ACQ / "tke_round331_writer.F90"
TKE_FIXTURE = TKE_ACQ / "tke_round331_layout_fixture.F90"
WRITER = ACQ / "round332_writer.F90"
FIXTURE = ACQ / "round332_layout_fixture.F90"
GFORTRAN = Path("/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran")
ACQ_RUN = DYN_ACQ / "run.sh"


@pytest.fixture(scope="module")
def record_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("r332-source-layout")
    commands = (
        (DYN_WRITER, DYN_FIXTURE, "write_dyn_fixture"),
        (DYN_WRITER, ZDF_WRITER, ZDF_FIXTURE, "write_zdf_fixture"),
        (DYN_WRITER, TKE_WRITER, TKE_FIXTURE, "write_tke_fixture"),
        (DYN_WRITER, WRITER, FIXTURE, "write_round332_fixture"),
    )
    for command in commands:
        *sources, output = command
        subprocess.run([
            str(GFORTRAN), "-ffree-line-length-none", "-I", str(root),
            "-J", str(root), str(STUBS), *(str(path) for path in sources),
            "-o", str(root / output),
        ], check=True)
    for rank in (0, 1):
        for executable in ("write_dyn_fixture", "write_zdf_fixture",
                           "write_tke_fixture", "write_round332_fixture"):
            subprocess.run([str(root / executable), str(rank)], cwd=root, check=True)
    (root / "producer_commit.txt").write_text("a" * 40 + "\n")
    return root


@pytest.mark.skipif(not GFORTRAN.exists(), reason="NEMO gfortran toolchain absent")
def test_real_writer_layout_and_clean_admission(record_dir: Path) -> None:
    result = gate.run_gate(record_dir, record_dir, "a" * 40)
    assert result["status"] == "PASS_R332_SOURCE_ORDER_ADMISSION"
    record = gate.read_record(record_dir / gate.record_name(0))
    assert tuple(record["fields"]) == gate.FIELDS
    assert record["fields"]["tke_pdlr"].shape == (3, 2, 4)
    assert set(result["source_order_counts"]) == {"0", "1"}


def test_fixture_rank_payloads_are_distinct(record_dir: Path) -> None:
    rank0 = gate.read_record(record_dir / gate.record_name(0))["fields"]
    rank1 = gate.read_record(record_dir / gate.record_name(1))["fields"]
    assert not np.array_equal(rank0["stage1_v_after"], rank1["stage1_v_after"])


def test_fixture_preserves_dry_negative_zero_sign_bits(record_dir: Path) -> None:
    fields = gate.read_record(record_dir / gate.record_name(0))["fields"]
    for name in ("tke_zmxlm", "tke_zmxld"):
        assert np.all(fields[name][0, 0, :] == 0.0)
        assert np.all(np.signbit(fields[name][0, 0, :]))


def test_legitimate_no_flip_vorticity_record_is_admissible(record_dir: Path) -> None:
    record = gate.read_record(record_dir / gate.record_name(0))
    record["fields"]["stage1_v_add"].fill(0.0)
    record["fields"]["stage1_v_before"][
        record["fields"]["stage1_v_before"] == 0.0] = 0.0
    record["fields"]["stage1_v_after"][:] = record["fields"]["stage1_v_before"]
    prior = gate.r331.read_record(record_dir / gate.r331.record_name(0))
    counts = gate._verify_semantics(record, prior["fields"]["tke_wmask"] != 0.0)
    assert counts["changed_bits"] == 0
    assert counts["value_equal_bit_changes"] == 0


PLANT_REASONS = {
    "header": "wrong record header",
    "field-order": "field order differs",
    "truncation": "truncated",
    "stamp": "producer commit",
    "vor": "vorticity source-order addition does not replay bitwise",
    "pdlr": "wet inverse Prandtl ratio is outside",
}


@pytest.mark.parametrize("plant", gate.PLANTS)
def test_every_required_plant_refuses(record_dir: Path, plant: str) -> None:
    with pytest.raises(gate.GateError, match=re.escape(PLANT_REASONS[plant])):
        gate.run_gate(record_dir, record_dir, "a" * 40, plant)


@pytest.mark.parametrize("plant,field", [
    ("vor", "stage1_v_after"), ("pdlr", "tke_pdlr"),
])
def test_semantic_plants_are_one_ulp_and_wet(
        record_dir: Path, plant: str, field: str) -> None:
    record = gate.read_record(record_dir / gate.record_name(0))
    prior = gate.r331.read_record(record_dir / gate.r331.record_name(0))
    wet = prior["fields"]["tke_wmask"] != 0.0
    before = record["fields"][field].copy()
    gate._plant(record, plant, wet)
    changed = record["fields"][field].view(np.uint64) != before.view(np.uint64)
    assert np.count_nonzero(changed) == 1
    assert bool(wet[changed][0])
    old_bits = before.view(np.uint64)[changed][0]
    new_bits = record["fields"][field].view(np.uint64)[changed][0]
    assert new_bits == old_bits + 1


@pytest.mark.parametrize("plant", ["vor", "pdlr"])
def test_cli_semantic_plants_reach_refuse(record_dir: Path, plant: str) -> None:
    result = subprocess.run([
        sys.executable, str(Path(gate.__file__)), "--record-dir", str(record_dir),
        "--zdf-reference-dir", str(record_dir), "--expect-commit", "a" * 40,
        "--plant", plant,
    ], check=False, text=True, capture_output=True)
    assert result.returncode == 1
    assert "REFUSE:" in result.stdout


def test_patches_are_context_anchored_and_target_owning_statements() -> None:
    dynvor = (ACQ / "dynvor_round332.patch").read_text()
    tke = (ACQ / "zdftke_round332.patch").read_text()
    for patch in (dynvor, tke):
        assert not re.search(r"^@@ -\d+,0 ", patch, re.MULTILINE)
    assert not any(line.startswith("-") and not line.startswith("---")
                       for line in patch.splitlines())
    assert "CALL vor_een( kt, Kmm, ntot" in dynvor
    assert "CALL r332_vor_add" in dynvor
    assert "pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) + zva" not in dynvor
    assert "END SELECT" in tke
    assert "Vertical eddy viscosity and diffusivity" in tke
    assert "p_pdlr(ntsi:ntei,jj,2:jpkm1)" in tke


def test_tke_landing_check_and_misplacement_plant_are_preflighted() -> None:
    text = ACQ_RUN.read_text()
    assert "check_r332_tke_landing" in text
    assert "tke-observer-misplaced" in text


def test_tke332_retained_admission_requires_commit_and_binary_pins() -> None:
    text = ACQ_RUN.read_text()
    assert "R332_RETAINED_PRODUCER" in text
    assert "R332_RETAINED_BINARY_SHA" in text
    assert "retained acquisition binary" in text


def test_launcher_imports_reviewed_acquisition_driver() -> None:
    launcher = (ACQ / "run.sh").read_text()
    assert "R328_ACQ_TARGET=tke332" in launcher
    assert "round328_dynzdf_acquisition/run.sh" in launcher
