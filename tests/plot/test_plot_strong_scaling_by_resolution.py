"""Direct test for the per-resolution strong-scaling plotter."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

_PLT = (Path(__file__).resolve().parents[2] / "scripts" / "plot"
        / "plot_strong_scaling_by_resolution.py")
_spec = importlib.util.spec_from_file_location("plot_strong_by_resolution", _PLT)
plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot)

_FIELDS = ["grid", "precision", "mode", "n_resource", "n_devices",
           "resolution", "sypd", "mcells_per_s"]


def _row(grid, prec, mode, n, res, sypd, mc="100.0"):
    return {f: "" for f in _FIELDS} | {
        "grid": grid, "precision": prec, "mode": mode,
        "n_resource": n, "n_devices": n, "resolution": res,
        "sypd": sypd, "mcells_per_s": mc}


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _sample_rows():
    rows = []
    # cubed-sphere strong, two resolutions x two precisions x 1/2/3 GPUs.
    for prec, base in (("float64", 1.0), ("float32", 1.6)):
        for res, scale in (("48", 1.0), ("192", 0.25)):
            for n, eff in ((1, 1.0), (2, 1.9), (3, 2.6)):
                rows.append(_row("cubed-sphere", prec, "strong", n, res,
                                 str(base * scale * n * eff)))
    # noise that must be ignored: a weak row, a wrong grid, a non-positive sypd.
    rows.append(_row("cubed-sphere", "float64", "weak", 2, "48", "5.0"))
    rows.append(_row("latlon", "float64", "strong", 2, "64", "9.0"))
    rows.append(_row("cubed-sphere", "float64", "strong", 4, "48", "0.0"))
    return rows


def test_make_figure_creates_png(tmp_path):
    csvp = tmp_path / "tidy.csv"
    _write_csv(csvp, _sample_rows())
    rows = plot._read(csvp)
    out = plot.make_figure(rows, "cubed-sphere", tmp_path)
    assert out.exists() and out.stat().st_size > 1000  # a real PNG


def test_group_series_filters_strong_and_grid():
    rows = _sample_rows()
    series = plot.group_series(rows, "cubed-sphere", "sypd")
    # two precisions, each with the two resolutions; weak/latlon/zero dropped.
    assert set(series) == {"float64", "float32"}
    assert set(series["float64"]) == {"48", "192"}
    # the n=4 sypd=0 row was dropped -> only 1,2,3 survive for res 48.
    ns = [n for n, _ in series["float64"]["48"]]
    assert ns == [1, 2, 3]


def test_group_series_grid_alias():
    # underscore spelling must canonicalise to the same grid.
    rows = [_row("cubed_sphere", "float64", "strong", 1, "48", "1.0"),
            _row("cubed_sphere", "float64", "strong", 2, "48", "1.9")]
    series = plot.group_series(rows, "cubed-sphere", "sypd")
    assert series["float64"]["48"] == [(1, 1.0), (2, 1.9)]


def test_compute_speedup_normalises_to_min_resource():
    by_res = {"48": [(1, 10.0), (2, 18.0), (3, 24.0)]}
    speed = plot.compute_speedup(by_res)
    # baseline N0=1, v0=10 -> speedups 1.0, 1.8, 2.4
    assert speed["48"] == [(1, 1.0), (2, 1.8), (3, 2.4)]


def test_make_figure_raises_on_unknown_grid(tmp_path):
    # typoed --grid -> no matching rows -> fail loud, not a blank PNG (codex).
    csvp = tmp_path / "tidy.csv"
    _write_csv(csvp, _sample_rows())
    rows = plot._read(csvp)
    with pytest.raises(SystemExit):
        plot.make_figure(rows, "no-such-grid", tmp_path)


def test_make_figure_raises_on_unknown_metric(tmp_path):
    # typoed --metric (not a CSV column) -> fail loud, not a blank PNG (codex).
    csvp = tmp_path / "tidy.csv"
    _write_csv(csvp, _sample_rows())
    rows = plot._read(csvp)
    with pytest.raises(SystemExit):
        plot.make_figure(rows, "cubed-sphere", tmp_path, metric="not_a_column")
