from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l4_orca2_round101_gyre_year_bisect.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("round101_bisect", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshots(root: Path, value: float) -> None:
    root.mkdir(parents=True)
    for day in range(1, 3):
        arrays = {
            name: np.full((2, 3), value, dtype=np.float64)
            for name in ("T", "S", "u", "v", "ssh")
        }
        np.savez(root / f"day{day:03d}.npz", **arrays)


def test_snapshot_comparator_finds_first_full_array_difference(tmp_path):
    module = _module()
    left = tmp_path / "left"
    right = tmp_path / "right"
    _snapshots(left, 0.0)
    _snapshots(right, 0.0)
    with np.load(right / "day002.npz") as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["v"] = arrays["v"].copy()
    arrays["v"][1, 2] = 1.0
    np.savez(right / "day002.npz", **arrays)

    report = module.compare_snapshots(left, right, days=2)
    assert report["status"] == "DIFFERENT"
    assert report["differing_days"] == 1
    assert report["first_difference"]["day"] == 2
    assert report["first_difference"]["field"] == "v"
    assert report["first_difference"]["index"] == (1, 2)


def test_exact_control_and_bit_flip_plant(tmp_path):
    module = _module()
    left = tmp_path / "left"
    right = tmp_path / "right"
    _snapshots(left, 0.0)
    _snapshots(right, 0.0)
    assert module.compare_snapshots(left, right, days=2)["status"] == "EXACT"
    planted = module.compare_snapshots(
        left, right, days=2, plant="flip-first-bit"
    )
    assert planted["status"] == "DIFFERENT"
    assert planted["first_difference"]["day"] == 1

    with np.load(right / "day001.npz") as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["T"] = arrays["T"].copy()
    arrays["T"][1, 1] = -0.0
    np.savez(right / "day001.npz", **arrays)
    signed_zero = module.compare_snapshots(left, right, days=2)
    assert signed_zero["first_difference"]["field"] == "T"
    assert signed_zero["first_difference"]["index"] == (1, 1)
    assert signed_zero["first_difference"]["max_abs"] == 0.0


def test_source_file_parser_is_fail_closed(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(module, "differing_files", lambda _repo: ("a.py", "b.py"))
    assert module.parse_source_files(tmp_path, "ALL") == ("a.py", "b.py")
    assert module.parse_source_files(tmp_path, "b.py") == ("b.py",)
    with pytest.raises(module.GateError, match="outside the frozen census"):
        module.parse_source_files(tmp_path, "c.py")
    with pytest.raises(module.GateError, match="duplicate"):
        module.parse_source_files(tmp_path, "a.py,a.py")


def test_reverse_commit_parser_is_fail_closed(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(module, "_git", lambda _repo, *_args: b"commit\n")
    assert module.parse_reverse_commits(tmp_path, "abc,def") == ("abc", "def")
    with pytest.raises(module.GateError, match="duplicate"):
        module.parse_reverse_commits(tmp_path, "abc,abc")

    def _missing(_repo, *_args):
        raise module.subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(module, "_git", _missing)
    with pytest.raises(module.GateError, match="unknown reverse commit"):
        module.parse_reverse_commits(tmp_path, "missing")


def test_source_opcode_parser_and_overlay_are_frozen():
    module = _module()
    root = Path(__file__).parents[3]
    selected = module.parse_source_opcodes(root, "0,2-3")
    assert selected == (0, 2, 3)
    payload, records = module.source_opcode_overlay(root, selected)
    assert len(records) == 49
    assert [record["index"] for record in records if record["selected"]] == [
        0, 2, 3
    ]
    assert module.sha256_bytes(payload) == (
        "686f5feaa98176c956b76e6936080b2275cad5e0fb56b44ae536cfae0279dfea"
    )
