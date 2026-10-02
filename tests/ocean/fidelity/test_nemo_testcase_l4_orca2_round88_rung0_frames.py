"""Controls for the round-88 rung-0 frame acquisition repair."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / (
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round88_rung0_frames"
)
PATCH = ARTIFACTS / "traadv_round88_skip_off_runoff_probe.patch"
RUN = ARTIFACTS / "run.sh"
SOURCE = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "ORCA2_OMIP_L4_R87FRAMES/MY_SRC/traadv.F90"
)


def test_repair_is_additions_only_and_scopes_all_stage1_probe_writes(
    tmp_path: Path,
) -> None:
    patch_text = PATCH.read_text()
    removed = [
        line for line in patch_text.splitlines()
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
    patched = target.read_text()
    start = patched.index("! Lane-4 WRITE-only stage-1 WZV operand record")
    end = patched.index("! Lane-2 WRITE-only diagnostic", start)
    stage1_probe = patched[start:end]
    assert stage1_probe.count("IF( ln_rnf ) THEN") == 3
    assert stage1_probe.count("l4_canon_2d(rnf,'T')") == 1
    assert stage1_probe.count("WRITE(993)") == 5
    assert stage1_probe.count("CLOSE(993)") == 1


def test_launcher_uses_a_fresh_target_and_has_no_time_dependency() -> None:
    text = RUN.read_text()
    assert "TARGET_CFG=ORCA2_OMIP_L4_R88FRAMES" in text
    assert "round88/acquisition" in text
    assert "/usr/bin/time" not in text
    assert "--admit-existing" in text
    assert "ORCA2_ROUND88_RUNG0_FRAMES_ACQUISITION_PASS" in text
