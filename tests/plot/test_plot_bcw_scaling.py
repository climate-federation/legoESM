"""Direct test for the baroclinic-wave publication plotter (matplotlib path)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

_PLT = Path(__file__).resolve().parents[2] / "scripts" / "plot" / "plot_bcw_scaling.py"
_spec = importlib.util.spec_from_file_location("plot_bcw_scaling", _PLT)
plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot)

_FIELDS = [
    "backend", "grid", "case", "precision", "mode", "n_devices", "resolution",
    "resolution_km", "n_levels", "sypd", "time_per_step_ms", "total_cells",
    "mcells_per_s", "scaling_efficiency", "dt_seconds", "physics_level",
    "compile_time_s", "source",
]


def _row(backend, grid, case, prec, n, res, km, sypd):
    return {f: "" for f in _FIELDS} | {
        "backend": backend, "grid": grid, "case": case, "precision": prec,
        "mode": "strong", "n_devices": n, "resolution": res,
        "resolution_km": km, "sypd": sypd, "physics_level": "none",
    }


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_make_figure_creates_png(tmp_path):
    rows = []
    for grid, km0 in (("icosahedral", 223.0), ("latlon", 156.0)):
        for n, s in ((1, 10.0), (2, 19.0), (4, 36.0)):
            rows.append(_row("GPU", grid, "dry", "float64", n, 5, km0, s))
            rows.append(_row("CPU", grid, "dry", "float64", 16 * n, 5, km0, s * 0.4))
    csvp = tmp_path / "tidy.csv"
    _write_csv(csvp, rows)
    data = plot._read(csvp)
    out = plot.make_figure(data, "dry", "float64", tmp_path)
    assert out is not None and out.exists()
    assert out.stat().st_size > 1000  # a real PNG, not empty


def test_make_figure_none_when_absent(tmp_path):
    csvp = tmp_path / "tidy.csv"
    _write_csv(csvp, [_row("GPU", "icosahedral", "dry", "float64", 1, 5, 223.0, 10.0)])
    data = plot._read(csvp)
    # No moist rows => no moist figure.
    assert plot.make_figure(data, "moist", "float64", tmp_path) is None
