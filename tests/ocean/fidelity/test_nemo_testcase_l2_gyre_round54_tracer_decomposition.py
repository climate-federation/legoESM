from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
HARNESS = (
    ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
    / "nemo_testcase_l2_gyre_round54_tracer_decomposition.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("gyre_round54_decomposition", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_field_stats_recovers_exact_synthetic_answer():
    module = _module()
    oracle = np.arange(12.0).reshape(2, 2, 3)
    candidate = oracle.copy()
    candidate[1, 0, 2] += 3.0
    mask = np.ones_like(oracle, dtype=bool)
    row = module.field_stats(candidate, oracle, mask)
    assert row["cells_unequal"] == 1
    assert row["max_abs"] == 3.0
    assert row["rms"] == pytest.approx(3.0 / np.sqrt(12.0))


def test_self_check_and_nonvacuous_plant_exit_codes():
    module = _module()
    assert module.main(["--mode", "self-check"]) == 0
    assert module.main([
        "--mode", "self-check", "--plant", "self-compare"
    ]) == 1
    assert module.main(["--mode", "zdf-score"]) == 1


def test_partitions_cover_vertical_roles_once():
    module = _module()
    wet = np.ones((4, 6, 5), dtype=bool)
    levels, vertical, regions = module._partitions(wet)
    assert len(levels) == 5
    assert sum(int(mask.sum()) for mask in vertical.values()) == int(wet.sum())
    assert set(regions) >= {
        "south_half.west_third", "north_half.middle_third",
        "north_half.east_third",
    }


def test_peak_rows_excludes_larger_dry_residuals():
    module = _module()
    oracle = np.zeros((2, 2, 3))
    candidate = oracle.copy()
    candidate[0, 0, 0] = 100.0
    candidate[1, 1, 2] = 4.0
    candidate[1, 0, 1] = -3.0
    wet = np.ones_like(oracle, dtype=bool)
    wet[0, 0, 0] = False
    rows = module._peak_rows(candidate, oracle, wet)
    assert [(row["j"], row["i"], row["k"]) for row in rows] == [
        (1, 1, 2), (1, 0, 1)
    ]
    assert [row["delta"] for row in rows] == [4.0, -3.0]
