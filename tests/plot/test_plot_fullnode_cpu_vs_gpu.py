"""Direct test for the CPU-vs-GPU strong-scaling per-grid plotter."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

_PLT = (Path(__file__).resolve().parents[2] / "scripts" / "plot"
        / "plot_fullnode_cpu_vs_gpu.py")
_spec = importlib.util.spec_from_file_location("plot_fullnode_cpu_vs_gpu", _PLT)
plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot)

_FIELDS = ["grid", "backend", "precision", "resolution", "n_resource",
           "n_devices", "sypd", "mcells_per_s"]


def _row(grid, backend, res, n, sypd, mc="100.0"):
    return {f: "" for f in _FIELDS} | {
        "grid": grid, "backend": backend, "precision": "float32",
        "resolution": res, "n_resource": n, "n_devices": n,
        "sypd": sypd, "mcells_per_s": mc}


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _sample_rows():
    rows = []
    # latlon: CPU cores 1,2,4 + GPU 1,2 at res 128 (a scaling curve each).
    for n, s in ((1, 2.0), (2, 3.6), (4, 6.0)):
        rows.append(_row("latlon", "CPU", "128", str(n), str(s), str(10 * n)))
    for n, s in ((1, 20.0), (2, 36.0)):
        rows.append(_row("latlon", "GPU", "128", str(n), str(s), str(100 * n)))
    # cubed_sphere (alias) at res 48, CPU faces 1,2 + GPU 1.
    rows.append(_row("cubed_sphere", "CPU", "48", "1", "5.0", "50"))
    rows.append(_row("cubed_sphere", "CPU", "48", "2", "9.0", "90"))
    rows.append(_row("cubed_sphere", "GPU", "48", "1", "30.0", "300"))
    return rows


def test_group_builds_curves_and_normalises_alias():
    g = plot.group(_sample_rows(), "sypd")
    assert set(g) == {"latlon", "cubed-sphere"}
    # latlon res 128 CPU curve over cores 1,2,4
    assert g["latlon"][128]["CPU"] == [(1, 2.0), (2, 3.6), (4, 6.0)]
    assert g["latlon"][128]["GPU"] == [(1, 20.0), (2, 36.0)]
    assert g["cubed-sphere"][48]["CPU"] == [(1, 5.0), (2, 9.0)]


def test_group_drops_nonpositive_and_unknown_backend():
    rows = _sample_rows() + [
        _row("latlon", "CPU", "128", "8", "0.0", "0"),   # non-positive metric
        _row("latlon", "TPU", "128", "1", "9.0", "9"),   # unknown backend
    ]
    g = plot.group(rows, "sypd")
    assert all(n != 8 for n, _ in g["latlon"][128]["CPU"])
    assert set(g["latlon"][128]) == {"CPU", "GPU"}


def test_peak_table_uses_curve_max():
    tbl = plot.peak_table(_sample_rows())
    d = tbl[("latlon", 128)]
    assert d["cpu"] == 6.0 and d["gpu"] == 36.0          # peak across the curve
    assert d["gpu_over_cpu"] == pytest.approx(6.0)


def test_make_figures_one_png_per_grid(tmp_path):
    csvp = tmp_path / "tidy.csv"
    _write_csv(csvp, _sample_rows())
    written = plot.make_figures(plot._read(csvp), tmp_path / "plots")
    names = sorted(p.name for p in written)
    assert names == ["fullnode_cpu_vs_gpu_cubed-sphere.png",
                     "fullnode_cpu_vs_gpu_latlon.png"]
    assert all(p.exists() for p in written)


def test_make_figures_raises_on_no_usable_rows(tmp_path):
    csvp = tmp_path / "tidy.csv"
    _write_csv(csvp, [_row("latlon", "CPU", "128", "1", "0.0", "0")])
    with pytest.raises(SystemExit):
        plot.make_figures(plot._read(csvp), tmp_path / "plots")
