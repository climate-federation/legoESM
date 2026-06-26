"""Direct test for the full-node CPU-vs-A100 per-grid plotter."""

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

_FIELDS = ["grid", "backend", "precision", "resolution", "sypd", "mcells_per_s"]


def _row(grid, backend, res, sypd, mc="100.0"):
    return {f: "" for f in _FIELDS} | {
        "grid": grid, "backend": backend, "precision": "float32",
        "resolution": res, "sypd": sypd, "mcells_per_s": mc}


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _sample_rows():
    # latlon + cubed_sphere (alias), CPU + GPU, two resolutions each.
    return [
        _row("latlon", "CPU", "128", "5.0", "80"),
        _row("latlon", "GPU", "128", "20.0", "320"),
        _row("latlon", "CPU", "256", "1.0", "70"),
        _row("latlon", "GPU", "256", "8.0", "300"),
        _row("cubed_sphere", "CPU", "48", "6.0", "90"),
        _row("cubed_sphere", "GPU", "48", "30.0", "400"),
    ]


def test_group_by_grid_normalises_alias_and_backend():
    rows = _sample_rows()
    g = plot.group_by_grid(rows, "sypd")
    # cubed_sphere alias collapses to cubed-sphere
    assert set(g) == {"latlon", "cubed-sphere"}
    assert g["latlon"]["CPU"] == [(128, 5.0), (256, 1.0)]
    assert g["latlon"]["GPU"] == [(128, 20.0), (256, 8.0)]
    assert g["cubed-sphere"]["GPU"] == [(48, 30.0)]


def test_group_by_grid_drops_nonpositive_and_unknown_backend():
    rows = _sample_rows() + [
        _row("latlon", "CPU", "512", "0.0", "0"),      # non-positive metric
        _row("latlon", "TPU", "128", "9.0", "9"),      # unknown backend
    ]
    g = plot.group_by_grid(rows, "sypd")
    # 512 (zero sypd) and the TPU row are both excluded
    assert all(res != 512 for res, _ in g["latlon"]["CPU"])
    assert set(g["latlon"]) == {"CPU", "GPU"}


def test_speedup_table_ratio():
    speed = plot.speedup_table(_sample_rows())
    d = speed[("latlon", 128)]
    assert d["cpu"] == 5.0 and d["gpu"] == 20.0
    assert d["gpu_over_cpu"] == pytest.approx(4.0)


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
    _write_csv(csvp, [_row("latlon", "CPU", "128", "0.0", "0")])  # all dropped
    with pytest.raises(SystemExit):
        plot.make_figures(plot._read(csvp), tmp_path / "plots")
