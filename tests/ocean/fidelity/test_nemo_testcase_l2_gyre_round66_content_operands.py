from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
    / "nemo_testcase_l2_gyre_round66_content_operands.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("round66_content", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reciprocal_substitution_names_only_the_changed_operand():
    module = _module()
    shape = (2, 3, 2)
    wet = np.ones(shape, dtype=bool)
    oracle = {
        "tracer_Kbb": np.full(shape, 2.0),
        "e3t_Kbb": np.full(shape, 3.0),
        "p2dt": np.float64(4.0),
        "e3t_Kmm": np.full(shape, 5.0),
        "Krhs": np.full(shape, 0.25),
    }
    live = dict(oracle)
    live["tracer_Kbb"] = np.full(shape, 2.5)
    target = module.content_statement(oracle)
    rows = module.reciprocal_substitutions(live, oracle, target, wet)

    assert rows["live_baseline"]["cells_unequal"] == wet.size
    assert rows["oracle_baseline"]["cells_unequal"] == 0
    assert rows["oracle_into_live"]["tracer_Kbb"]["cells_unequal"] == 0
    assert rows["live_into_oracle"]["tracer_Kbb"]["cells_unequal"] == wet.size
    for name in set(module.OPERANDS) - {"tracer_Kbb"}:
        assert rows["oracle_into_live"][name]["cells_unequal"] == wet.size
        assert rows["live_into_oracle"][name]["cells_unequal"] == 0


def test_one_ulp_operand_plant_changes_exact_content_census():
    module = _module()
    shape = (2, 2, 2)
    operands = {
        "tracer_Kbb": np.full(shape, 2.0),
        "e3t_Kbb": np.full(shape, 3.0),
        "p2dt": np.float64(4.0),
        "e3t_Kmm": np.full(shape, 5.0),
        "Krhs": np.full(shape, 0.25),
    }
    baseline = module.content_statement(operands)
    planted = dict(operands)
    planted["tracer_Kbb"] = operands["tracer_Kbb"].copy()
    planted["tracer_Kbb"][0, 0, 0] = np.nextafter(
        planted["tracer_Kbb"][0, 0, 0], np.float64(np.inf))
    row = module.round54.field_stats(
        module.content_statement(planted), baseline,
        np.ones(shape, dtype=bool))
    assert row["cells_unequal"] == 1
    assert row["max_abs"] > 0.0


def test_distinct_callback_observations_are_not_equal():
    first = {"content_T": np.array([1.0, 2.0])}
    duplicate = {"content_T": np.array([1.0, 2.0])}
    distinct = {"content_T": np.array([1.0, 3.0])}

    assert all(np.array_equal(duplicate[key], first[key]) for key in first)
    assert not all(np.array_equal(distinct[key], first[key]) for key in first)
