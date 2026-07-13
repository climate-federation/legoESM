"""Unit tests for the ``scripts/validate/eval_land.py`` CLI."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

from legoesm.land.evaluation.scorecard import Scorecard, VariableResult

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "validate" / "eval_land.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("_eval_land_cli", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_list_metrics_returns_zero(capsys):
    mod = _load_cli()
    rc = mod.main(["--list-metrics"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "bias_score" in out and "taylor_score" in out


def test_no_recipe_is_usage_error(capsys):
    mod = _load_cli()
    rc = mod.main([])
    assert rc == 2
    assert "--recipe is required" in capsys.readouterr().err


def test_print_scorecard_table_runs(capsys):
    mod = _load_cli()
    v = VariableResult(
        key="flux:shflx", label="Sensible heat", unit="W m-2", group="flux",
        n=100, metrics={"bias": 0.0, "rmse": 0.0, "bias_score": 1.0},
        score=1.0,
    )
    v0 = VariableResult(
        key="flux:gpp", label="GPP", unit="umol", group="flux",
        n=0, metrics={}, score=float("nan"),
    )
    card = Scorecard(case="c", model="m", reference="r", variables=[v, v0])
    mod._print_scorecard_table(card)  # must not raise on n=0 rows / nan
    out = capsys.readouterr().out
    assert "Sensible heat" in out
    assert "overall_score" in out
    assert "(no valid data)" in out  # n=0 row rendered, not crashed


def test_cli_module_exposes_expected_symbols():
    mod = _load_cli()
    for name in ("main", "_print_scorecard_table", "_parse_args"):
        assert hasattr(mod, name)
    # Uses the shared registry, not a private copy.
    assert isinstance(mod.METRIC_REGISTRY, dict)
    assert np.isfinite  # sanity import guard
