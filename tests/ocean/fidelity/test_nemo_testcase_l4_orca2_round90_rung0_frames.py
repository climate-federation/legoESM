"""Controls for the source-resolved round-90 rung-0 frame repair."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / (
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round90_rung0_frames"
)
PATCH = ARTIFACTS / "traadv_round90_skip_off_runoff_probe.patch"
RUN = ARTIFACTS / "run.sh"
SOURCE = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "ORCA2_OMIP_L4_R87FRAMES/MY_SRC/traadv.F90"
)


def test_repair_is_additions_only_and_scopes_every_legacy_write(
    tmp_path: Path,
) -> None:
    patch_text = PATCH.read_text()
    removed = [
        line
        for line in patch_text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    assert removed == []
    assert patch_text.count("+         IF( ln_rnf ) THEN") == 3

    target = tmp_path / "traadv.F90"
    target.write_bytes(SOURCE.read_bytes())
    result = subprocess.run(
        ["patch", "-s", "--fuzz=0", str(target), str(PATCH)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    stage1 = target.read_text().split(
        "! Lane-4 WRITE-only stage-1 WZV operand record", 1
    )[1].split("! Lane-2 WRITE-only diagnostic", 1)[0]
    assert stage1.count("IF( ln_rnf ) THEN") == 3
    assert stage1.count("WRITE(993)") == 5
    assert stage1.count("CLOSE(993)") == 1
    assert stage1.count("CALL wzv") == 1


def test_missing_guard_plant_fires(tmp_path: Path) -> None:
    target = tmp_path / "traadv.F90"
    target.write_bytes(SOURCE.read_bytes())
    subprocess.run(
        ["patch", "-s", "--fuzz=0", str(target), str(PATCH)], check=True
    )
    text = target.read_text()
    planted = text.replace("         IF( ln_rnf ) THEN\n", "", 1)
    assert planted.count("IF( ln_rnf ) THEN") != 3


def test_launcher_pins_the_source_resolved_fault_and_fresh_target() -> None:
    result = subprocess.run(
        ["bash", "-n", str(RUN)], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    text = RUN.read_text()
    assert "traadv.f90:357" in text
    assert "traadv.f90:287" in text
    assert "TARGET_CFG=ORCA2_OMIP_L4_R90FRAMES" in text
    assert "round90/acquisition" in text
    assert "carry 80 frames" in text
    assert "write-only repair changed" in text
    assert "/usr/bin/time" not in text


def test_launcher_requires_all_firing_record_plants() -> None:
    text = RUN.read_text()
    for plant in ("header", "field-name", "truncation", "nonfinite", "stamp"):
        assert plant in text
    for plant in ("absent-as-zero", "owner-on"):
        assert plant in text
    assert "missing-guard plant stayed green" in text

