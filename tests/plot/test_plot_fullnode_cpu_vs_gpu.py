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

_FIELDS = ["grid", "backend", "precision", "resolution", "resolution_km",
           "n_resource", "n_devices", "sypd", "mcells_per_s", "mode"]


def _row(grid, backend, res, n, sypd, mc="100.0", mode="strong", km=""):
    return {f: "" for f in _FIELDS} | {
        "grid": grid, "backend": backend, "precision": "float32",
        "resolution": res, "resolution_km": km, "n_resource": n,
        "n_devices": n, "sypd": sypd, "mcells_per_s": mc, "mode": mode}


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


def test_group_excludes_weak_scaling_rows():
    # weak-scaling rows share grid/backend/resolution but must NOT pollute the
    # strong curves; a blank mode is treated as strong (legacy CSVs).
    rows = _sample_rows() + [
        _row("latlon", "CPU", "128", "8", "9.0", "80", mode="weak"),
        _row("latlon", "GPU", "128", "4", "70.0", "400", mode="weak"),
        _row("latlon", "CPU", "128", "16", "12.0", "120", mode=""),  # blank -> strong
    ]
    g = plot.group(rows, "sypd")
    cpu_ns = [n for n, _ in g["latlon"][128]["CPU"]]
    gpu_ns = [n for n, _ in g["latlon"][128]["GPU"]]
    assert 8 not in cpu_ns          # weak CPU point dropped
    assert gpu_ns == [1, 2]         # weak GPU n=4 point dropped
    assert 16 in cpu_ns             # blank-mode strong point kept


def test_group_drops_nonpositive_and_unknown_backend():
    rows = _sample_rows() + [
        _row("latlon", "CPU", "128", "8", "0.0", "0"),   # non-positive metric
        _row("latlon", "TPU", "128", "1", "9.0", "9"),   # unknown backend
    ]
    g = plot.group(rows, "sypd")
    assert all(n != 8 for n, _ in g["latlon"][128]["CPU"])
    assert set(g["latlon"][128]) == {"CPU", "GPU"}


def test_res_km_maps_resolution_to_km_and_labels():
    rows = [
        _row("cubed_sphere", "GPU", "48", "1", "30.0", km="208.498"),
        _row("latlon", "CPU", "128", "1", "2.0", km="156.373"),
        _row("latlon", "CPU", "256", "1", "0.3", km=""),   # missing km
    ]
    m = plot.res_km(rows)
    assert m[("cubed-sphere", 48)] == pytest.approx(208.498)   # alias canonicalised
    assert m[("latlon", 128)] == pytest.approx(156.373)
    assert ("latlon", 256) not in m                            # blank km not mapped
    assert plot._res_label(m, "latlon", 128) == "156 km"       # rounded km label
    assert plot._res_label(m, "latlon", 256) == "res 256"      # fallback when no km


def test_peak_table_uses_curve_max():
    tbl = plot.peak_table(_sample_rows())
    d = tbl[("latlon", 128)]
    assert d["cpu"] == 6.0 and d["gpu"] == 36.0          # peak across the curve
    assert d["gpu_over_cpu"] == pytest.approx(6.0)


def test_group_builds_mcells_curve():
    g = plot.group(_sample_rows(), "mcells_per_s")
    # latlon res 128 CPU mcells curve over cores 1,2,4 (mc = 10*n in fixture)
    assert g["latlon"][128]["CPU"] == [(1, 10.0), (2, 20.0), (4, 40.0)]


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
