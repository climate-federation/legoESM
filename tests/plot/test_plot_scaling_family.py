"""Direct test for the speedup-vs-ideal scaling-family plotter."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

_PLT = (Path(__file__).resolve().parents[2] / "scripts" / "plot"
        / "plot_scaling_family.py")
_spec = importlib.util.spec_from_file_location("plot_scaling_family", _PLT)
plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot)

_FIELDS = ["component", "backend", "grid", "mode", "precision",
           "resolution", "n_resource", "n_devices", "time_per_step_ms"]


def _row(component, backend, grid, mode, prec, res, n, t):
    return {f: "" for f in _FIELDS} | {
        "component": component, "backend": backend, "grid": grid, "mode": mode,
        "precision": prec, "resolution": str(res), "n_resource": str(n),
        "n_devices": str(n), "time_per_step_ms": str(t)}


def _sample_rows():
    rows = []
    # atm strong, latlon, CPU, f32 + f64, cores 1,2,4 (time halves ~ ideal).
    for n, t in ((1, 100.0), (2, 55.0), (4, 32.0)):
        rows.append(_row("atm", "CPU", "latlon", "strong", "float32", 128, n, t))
    for n, t in ((1, 200.0), (2, 120.0), (4, 80.0)):
        rows.append(_row("atm", "CPU", "latlon", "strong", "float64", 128, n, t))
    # atm strong, latlon, GPU, f32 only, 1,2,4.
    for n, t in ((1, 10.0), (2, 6.0), (4, 4.0)):
        rows.append(_row("atm", "CPU", "latlon", "strong", "float32", 128, n, t)
                    if False else
                    _row("atm", "GPU", "latlon", "strong", "float32", 128, n, t))
    # atm strong, spectral, CPU, single point (blocked -> no MPI).
    rows.append(_row("atm", "CPU", "spectral", "strong", "float64", 85, 1, 50.0))
    # atm WEAK, icosahedral, CPU, f32 1,2,4 (constant time = ideal weak).
    for n in (1, 2, 4):
        rows.append(_row("atm", "CPU", "icosahedral", "weak", "float32", 6, n, 40.0))
    # ocean strong, latlon, GPU, f64 1,2.
    for n, t in ((1, 270.0), (2, 150.0)):
        rows.append(_row("ocean", "GPU", "latlon", "strong", "float64", 360, n, t))
    return rows


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_group_nests_and_filters():
    g = plot.group(_sample_rows())
    assert set(g) == {"atm", "ocean"}
    assert set(g["atm"]) == {"strong", "weak"}
    # latlon CPU strong f32 curve over cores 1,2,4
    series = g["atm"]["strong"]["latlon"]["CPU"]["float32"][128]
    assert series == [(1, 100.0), (2, 55.0), (4, 32.0)]


def test_group_drops_unknown_and_nonpositive():
    rows = _sample_rows() + [
        _row("atm", "CPU", "latlon", "strong", "float32", 128, 8, 0.0),  # t<=0
        _row("atm", "TPU", "latlon", "strong", "float32", 128, 1, 5.0),  # backend
        _row("mars", "CPU", "latlon", "strong", "float32", 128, 1, 5.0), # component
        _row("atm", "CPU", "latlon", "medium", "float32", 128, 1, 5.0),  # mode
    ]
    g = plot.group(rows)
    assert all(n != 8 for n, _ in g["atm"]["strong"]["latlon"]["CPU"]["float32"][128])
    assert "ocean" in g and "mars" not in g


def test_speedup_curve_normalises_to_smallest_count():
    # times 100,55,32 over cores 1,2,4 -> speedup 1, 100/55, 100/32
    sp = plot.speedup_curve([(1, 100.0), (2, 55.0), (4, 32.0)])
    assert sp[0] == (1, pytest.approx(1.0))
    assert sp[1] == (2, pytest.approx(100 / 55))
    assert sp[2] == (4, pytest.approx(100 / 32))


def test_speedup_curve_single_point_is_unity():
    assert plot.speedup_curve([(1, 50.0)]) == [(1, pytest.approx(1.0))]
    assert plot.speedup_curve([]) == []


def test_ideal_curve_strong_is_diagonal_weak_is_flat():
    assert plot.ideal_curve([1, 2, 4], "strong") == [(1, 1.0), (2, 2.0), (4, 4.0)]
    assert plot.ideal_curve([1, 2, 4], "weak") == [(1, 1.0), (2, 1.0), (4, 1.0)]


def test_representative_resolution_prefers_longest_ladder():
    by_prec_res = {
        "float32": {128: [(1, 1.0), (2, 1.0)], 256: [(1, 1.0), (2, 1.0), (4, 1.0)]},
    }
    assert plot.representative_resolution(by_prec_res) == 256
    assert plot.representative_resolution({}) is None


def test_make_figures_one_png_per_component_mode(tmp_path):
    csvp = tmp_path / "tidy.csv"
    _write_csv(csvp, _sample_rows())
    written = plot.make_figures(plot._read(csvp), tmp_path / "plots")
    names = sorted(p.name for p in written)
    assert names == ["scaling_atm_strong.png", "scaling_atm_weak.png",
                     "scaling_ocean_strong.png"]
    assert all(p.exists() for p in written)


def test_make_figures_raises_on_missing_columns(tmp_path):
    csvp = tmp_path / "bad.csv"
    with csvp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "backend"])
        w.writeheader()
        w.writerow({"component": "atm", "backend": "CPU"})
    with pytest.raises(SystemExit):
        plot.make_figures(plot._read(csvp), tmp_path / "plots")


def test_make_figures_raises_on_no_usable_rows(tmp_path):
    csvp = tmp_path / "tidy.csv"
    _write_csv(csvp, [_row("atm", "CPU", "latlon", "strong", "float32", 128, 1, 0.0)])
    with pytest.raises(SystemExit):
        plot.make_figures(plot._read(csvp), tmp_path / "plots")
