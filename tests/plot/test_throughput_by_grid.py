"""Direct test for the throughput-by-grid plotter (matplotlib path)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

_PLT = Path(__file__).resolve().parents[2] / "scripts" / "plot" / "plot_throughput_by_grid.py"
_spec = importlib.util.spec_from_file_location("plot_throughput_by_grid", _PLT)
plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot)

_FIELDS = ["component", "backend", "grid", "case", "precision", "mode",
           "n_devices", "mcells_per_s"]


def _row(component, backend, grid, prec, mode, n, mc):
    return {f: "" for f in _FIELDS} | {
        "component": component, "backend": backend, "grid": grid,
        "case": "x", "precision": prec, "mode": mode,
        "n_devices": n, "mcells_per_s": mc}


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _sample_rows():
    rows = []
    # atm: two grids x two precisions x strong+weak; ocean: one grid.
    for grid in ("icosahedral", "latlon"):
        for prec, base in (("float64", 50.0), ("float32", 90.0)):
            for mode in ("strong", "weak"):
                for n, f in ((1, 1.0), (2, 1.8), (4, 3.4)):
                    rows.append(_row("atm", "CPU", grid, prec, mode, n, base * f))
    for prec, base in (("float64", 30.0), ("float32", 38.0)):
        for n, f in ((2, 1.0), (8, 3.0), (32, 9.0)):
            rows.append(_row("ocean", "CPU", "latlon", prec, "strong", n, base * f))
    # a GPU point (marker branch) + a weak_band alias (collapses to weak)
    rows.append(_row("ocean", "GPU", "latlon", "float64", "weak_band", 2, 75.0))
    return rows


def test_make_figure_creates_png(tmp_path):
    csvp = tmp_path / "tidy.csv"
    _write_csv(csvp, _sample_rows())
    rows = plot._read(csvp)
    out = plot.make_figure(rows, tmp_path)
    assert out.exists() and out.stat().st_size > 1000  # a real PNG


def test_peak_table_picks_max(tmp_path):
    csvp = tmp_path / "tidy.csv"
    _write_csv(csvp, _sample_rows())
    rows = plot._read(csvp)
    peaks = plot.peak_table(rows)
    # atm icosahedral float32 strong peak = 90.0 * 3.4 = 306.0
    assert abs(peaks[("atm", "strong", "icosahedral", "float32", "CPU")] - 306.0) < 1e-6
    # weak_band collapses into 'weak'
    assert ("ocean", "weak", "latlon", "float64", "GPU") in peaks


def test_skips_nonpositive_and_malformed(tmp_path):
    csvp = tmp_path / "tidy.csv"
    _write_csv(csvp, [
        _row("atm", "CPU", "latlon", "float64", "strong", 1, 0.0),    # mc<=0
        _row("atm", "CPU", "latlon", "float64", "strong", "", 5.0),   # bad n
        _row("atm", "CPU", "latlon", "float64", "strong", 2, 10.0),   # valid
    ])
    rows = plot._read(csvp)
    peaks = plot.peak_table(rows)
    assert peaks == {("atm", "strong", "latlon", "float64", "CPU"): 10.0}
