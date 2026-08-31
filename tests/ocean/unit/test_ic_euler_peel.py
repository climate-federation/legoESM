from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def peel():
    script_dir = (Path(__file__).parents[3] / "scripts" / "validate" /
                  "ocean_fidelity" / "dino_1226")
    sys.path.insert(0, str(script_dir))
    try:
        yield importlib.import_module("ic_euler_peel")
    finally:
        sys.path.remove(str(script_dir))


def test_planted_controls_fire(peel):
    assert peel.self_test() == {
        "field_mismatch_plant": "FIRED",
        "wrong_core_plant": "FIRED",
    }


def test_diff_row_exact_and_first_mismatch(peel):
    expected = np.arange(6, dtype=np.float64).reshape(2, 3)
    actual = expected.copy()
    active = np.ones_like(actual, dtype=bool)
    assert peel.diff_row("same", actual, expected, active)["status"] == "PASS"
    actual[0, 2] += 2.0 * peel.BAR
    row = peel.diff_row("plant", actual, expected, active)
    assert row["status"] == "OVER_BAR"
    assert row["mismatch_count"] == 1
    assert row["first_over_bar"]["index"] == [0, 2]


def test_diff_row_rejects_shape_and_nonfinite(peel):
    active = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        peel.diff_row("shape", np.zeros((2, 1)), np.zeros((2, 2)), active)
    bad = np.zeros((2, 2))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        peel.diff_row("nan", bad, np.zeros((2, 2)), active)


def test_runtime_dump_mapping(peel, tmp_path):
    raw = np.arange(35 * 203 * 56, dtype=np.float64).reshape(35, 203, 56)
    path = tmp_path / "dump.bin"
    raw.tofile(path)
    core = peel._runtime_dump(path)
    assert core.shape == (195, 48, 35)
    assert core[0, 0, 0] == raw[0, 4, 4]


def test_horizontal_mask_broadcasts_only_over_levels(peel):
    mask = np.array([[True, False], [False, True]])
    expanded = peel._broadcast_mask(mask, (2, 2, 3))
    assert expanded.shape == (2, 2, 3)
    assert np.array_equal(expanded[..., 0], mask)
    assert np.array_equal(expanded[..., 2], mask)


def test_parser_requires_receipt_inputs(peel):
    parser = peel.build_parser()
    args = parser.parse_args([
        "--run-kt2", "/tmp/a", "--run-traj-y1", "/tmp/b",
        "--lego-kt2", "/tmp/c", "--lego-d10", "/tmp/d",
        "--output", "/tmp/e.json",
    ])
    assert args.output == Path("/tmp/e.json")
