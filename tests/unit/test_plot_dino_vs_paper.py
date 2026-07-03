"""Smoke test for the DINO model-vs-paper comparison plotter.

Loads without error and reuses the cross-grid regrid/zonal helpers; the heavy
regrid + the netCDF read go through already-tested shared paths and the live run.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "plot_dino_vs_paper",
    Path(__file__).resolve().parents[2] / "scripts" / "plot" / "plot_dino_vs_paper.py")
vp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vp)


def test_reuses_cross_grid_helpers():
    for fn in ("_regrid_TS", "_latlon_coords", "_mpas_coords", "_snap_for_day"):
        assert hasattr(vp.XG, fn), f"missing reused helper plot_dino_cross_grid.{fn}"
    assert callable(vp._paper_surface) and callable(vp.main)
