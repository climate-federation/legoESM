"""Smoke test for the DINO publication-figure plotter.

It must load without error and reuse the sibling plotters' helpers + expose the
figure builders (the heavy regrid/streamfunction work goes through the
already-tested shared diagnostics and is exercised by the live run).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "plot_dino_publication",
    Path(__file__).resolve().parents[2] / "scripts" / "plot" / "plot_dino_publication.py")
pub = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pub)


def test_reuses_sibling_helpers():
    # cross-grid helpers
    for fn in ("_regrid_TS", "_latlon_coords", "_mpas_coords", "_stats",
               "_corr_vs_time", "_snap_for_day"):
        assert hasattr(pub.XG, fn), f"missing reused helper plot_dino_cross_grid.{fn}"
    # ACC helpers
    for fn in ("_grid_and_z", "_snaps"):
        assert hasattr(pub.ACC, fn), f"missing reused helper plot_dino_acc.{fn}"


def test_figure_builders_present():
    for fn in ("_acc_series_and_psi", "_figure_surface_state",
               "_figure_circulation", "main"):
        assert callable(getattr(pub, fn)), f"missing {fn}"
    assert pub.PAPER_ACC_R1_SV == 206.0
