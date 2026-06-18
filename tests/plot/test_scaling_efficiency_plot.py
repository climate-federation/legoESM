"""Direct test for scripts/plot/plot_scaling_efficiency.py — the parallel-
efficiency scaling plotter. Exercises the efficiency normalisation
(E(N)=mc(N)*N0/(N*mc(N0)); E(N0)=1; flat=1 for ideal) and that make_figure
renders without error.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = Path(__file__).resolve().parents[2] / "scripts" / "plot" / \
    "plot_scaling_efficiency.py"
_m = importlib.util.spec_from_file_location("plot_scaling_efficiency", _SPEC)
plot_mod = importlib.util.module_from_spec(_m)
_m.loader.exec_module(plot_mod)


def test_efficiency_base_point_is_one():
    pts = plot_mod.efficiency_curve({2: 7.0, 4: 13.0, 8: 25.0})
    assert pts[0] == (2, 1.0)                      # smallest N -> efficiency 1


def test_efficiency_ideal_flat():
    # perfect scaling: mc proportional to N -> efficiency 1.0 everywhere
    pts = dict(plot_mod.efficiency_curve({1: 10.0, 2: 20.0, 4: 40.0, 8: 80.0}))
    for n, e in pts.items():
        assert e == pytest.approx(1.0), (n, e)


def test_efficiency_subideal():
    # strong scaling falling short: mc(2)=18 (not 20), mc(4)=32 (not 40)
    pts = dict(plot_mod.efficiency_curve({1: 10.0, 2: 18.0, 4: 32.0}))
    assert pts[1] == pytest.approx(1.0)
    assert pts[2] == pytest.approx(0.9)            # 18/(2*10)
    assert pts[4] == pytest.approx(0.8)            # 32/(4*10)


def test_efficiency_anti_scaling_below_one():
    # cube-on-Gloo style: mc DROPS with more devices -> efficiency << 1
    pts = dict(plot_mod.efficiency_curve({1: 16.0, 2: 6.0, 6: 12.0}))
    assert pts[1] == pytest.approx(1.0)
    assert pts[2] == pytest.approx(6.0 / (2 * 16.0))   # 0.1875
    assert pts[6] < 1.0


def test_efficiency_empty():
    assert plot_mod.efficiency_curve({}) == []


def _rows():
    out = []
    # atm strong, two grids, both precisions
    data = {
        ("atm", "latlon", "float64", "strong"): {1: 10, 2: 19, 4: 36, 8: 60},
        ("atm", "latlon", "float32", "strong"): {1: 23, 2: 40, 4: 70, 8: 120},
        ("atm", "icosahedral", "float64", "weak"): {1: 5, 2: 9, 4: 18, 8: 34},
        ("ocean", "latlon", "float64", "strong"): {8: 11, 16: 19, 32: 26},
    }
    for (comp, grid, prec, mode), d in data.items():
        for n, mc in d.items():
            out.append({"component": comp, "grid": grid, "precision": prec,
                        "mode": mode, "backend": "CPU",
                        "n_devices": str(n), "mcells_per_s": str(mc)})
    return out


def test_make_figure_writes_png(tmp_path):
    out = plot_mod.make_figure(_rows(), tmp_path)
    assert out.exists() and out.stat().st_size > 0


def test_make_figure_empty(tmp_path):
    out = plot_mod.make_figure([], tmp_path)
    assert out.exists()
