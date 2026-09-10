"""Direct test for ``scripts/validate/mpas_ledger_report.py`` (#1354)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "scripts" / "validate" / "mpas_ledger_report.py"

_PROCS = ("turbulence", "convection", "microphysics", "radiation",
          "other_physics", "clips", "dynamics")


def _load():
    spec = importlib.util.spec_from_file_location("_ledrep", _PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ledrep"] = mod
    spec.loader.exec_module(mod)
    return mod


def _artifact(tmp_path, rates):
    path = tmp_path / "budget_ledger_columns.npz"
    np.savez(path, ledger_rates=rates, processes=np.asarray(_PROCS),
             columns=np.asarray(("water_kg_m2_s", "energy_W_m2")),
             n_steps=768, day=3.0)
    return path


def test_global_mean_reproduces_a_planted_row(tmp_path, capsys):
    """A planted 18 W/m2 in one row must come out as exactly that row's mean.

    Two columns with unequal values, so the reduction is exercised rather than
    passed through.
    """
    rates = np.zeros((2, 7, 2))
    rates[0, 1, 1] = 12.0   # convection energy, column 0
    rates[1, 1, 1] = 24.0   # convection energy, column 1 -> mean 18
    mod = _load()
    assert mod.main([str(_artifact(tmp_path, rates))]) == 0
    out = capsys.readouterr().out
    row = next(l for l in out.splitlines() if l.strip().startswith("convection"))
    assert "18.000" in row, row
    total = next(l for l in out.splitlines() if "TOTAL" in l)
    assert "18.000" in total, total


def test_water_column_is_converted_to_mm_per_day(tmp_path, capsys):
    """1 kg/m2/s planted -> 86400 mm/day printed; the unit label is load-bearing."""
    rates = np.zeros((1, 7, 2))
    rates[0, 0, 0] = 1.0
    mod = _load()
    mod.main([str(_artifact(tmp_path, rates))])
    out = capsys.readouterr().out
    row = next(l for l in out.splitlines() if l.strip().startswith("turbulence"))
    assert "86400.0000" in row, row
