"""Direct tests for scripts/concat_amip_output.py (chained-AMIP output concat).

Covers the two pure helpers:
  * ``_sorted_seg_dirs`` orders ``y{YYYY}_s{SS}`` zero-padded segment dirs
    chronologically (by year then segment) and ignores non-matching names;
  * ``_concat_npz`` concatenates per-segment NPZ arrays along the shared time
    axis in segment order.
"""
from __future__ import annotations

import numpy as np

from scripts.run.concat_amip_output import _sorted_seg_dirs, _concat_npz


def test_sorted_seg_dirs_is_chronological(tmp_path):
    # Create out-of-order, zero-padded y{YYYY}_s{SS} dirs.
    for name in ("y0010_s00", "y0002_s01", "y0002_s00", "y0001_s00",
                 "y0010_s01"):
        (tmp_path / name).mkdir()
    (tmp_path / "not_a_segment").mkdir()  # ignored (glob excludes it)
    got = [p.name for p in _sorted_seg_dirs(tmp_path)]
    assert got == ["y0001_s00", "y0002_s00", "y0002_s01",
                   "y0010_s00", "y0010_s01"]


def test_concat_npz_concatenates_along_time(tmp_path):
    p0 = tmp_path / "seg0.npz"
    p1 = tmp_path / "seg1.npz"
    np.savez(p0, days=np.array([0.0, 1.0]), T_atm=np.array([280.0, 281.0]))
    np.savez(p1, days=np.array([2.0, 3.0]), T_atm=np.array([282.0, 283.0]))
    out = _concat_npz([p0, p1], time_key="days")
    assert np.array_equal(out["days"], np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.array_equal(out["T_atm"], np.array([280.0, 281.0, 282.0, 283.0]))
