"""The twin's per-interface profile reduction: medians per depth, NaN-blind,
cut at depth_max, and names kept in the order given.

Non-vacuity: a column set with one NaN entry must not drop the interface,
and an interface below depth_max must not appear.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_P = _ROOT / "scripts/validate/ocean_fidelity/frozen_column_tke_twin.py"
_SPEC = importlib.util.spec_from_file_location("frz_twin", _P)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["frz_twin"] = _MOD
_SPEC.loader.exec_module(_MOD)


def test_profile_rows_median_nan_and_depth_cut():
    zk = np.array([1.0, 10.0, 50.0, 200.0])
    en = np.array([[1.0, 2.0, 3.0, 4.0],
                   [3.0, np.nan, 5.0, 6.0],
                   [5.0, 4.0, 7.0, 8.0]])
    lk = np.array([[0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, np.nan, 0.4],
                   [0.1, 0.2, 0.3, 0.4]])
    rows = _MOD.state_profile_rows(zk, 100.0, en=en, l_k=lk)
    assert [d for d, _ in rows] == [1.0, 10.0, 50.0]
    assert list(rows[0][1]) == ["en", "l_k"]
    assert rows[0][1]["en"] == 3.0
    assert rows[1][1]["en"] == 3.0          # NaN skipped: median(2, 4)
    assert rows[2][1]["l_k"] == 0.3         # NaN skipped: median(0.3, 0.3)


def test_profile_rows_all_nan_interface_reports_nan_not_drop():
    zk = np.array([1.0, 10.0])
    en = np.array([[1.0, np.nan], [2.0, np.nan]])
    rows = _MOD.state_profile_rows(zk, 1e9, en=en)
    assert len(rows) == 2
    assert np.isnan(rows[1][1]["en"])


def test_print_state_profile_emits_one_line_per_row(capsys):
    rows = [(1.0, {"en": 1e-3}), (10.0, {"en": 2e-4})]
    _MOD.print_state_profile("x", rows)
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 3 and out[1].startswith("[profile:x]")
