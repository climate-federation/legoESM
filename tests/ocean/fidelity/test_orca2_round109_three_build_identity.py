from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round109_three_build_identity as gate,
)


def _root(tmp_path: Path, marker: bytes) -> Path:
    root = tmp_path / marker.decode()
    root.mkdir()
    for rank in range(2):
        (root / f"ORCA2_00000010_restart_{rank:04d}.nc").write_bytes(b"restart" + bytes([rank]))
    for name in ("namelist_cfg", "deck_files.sha256", "input_files.sha256"):
        (root / name).write_bytes(marker if name == "namelist_cfg" else b"shared")
    return root


def test_three_build_gate_requires_recorder_deck_and_restart_identity(tmp_path):
    base = _root(tmp_path, b"base")
    accumulator = _root(tmp_path, b"same")
    per_level = tmp_path / "per-level"
    per_level.mkdir()
    for path in accumulator.iterdir():
        (per_level / path.name).write_bytes(path.read_bytes())
    result = gate.run(base, accumulator, per_level, "none")
    assert result["status"] == "PASS_R109_THREE_BUILD_IDENTITY"
    assert all(row["recorder_builds_equal"] for row in result["deck_identity"])
    assert not result["deck_identity"][0]["base_equal"]
