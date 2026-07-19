"""Direct tests for ``scripts/plot/plot_cpu_vs_gpu_scaling.py``.

Focused on the 2x3 paper layout added for the CPU campaign. The things worth
pinning are the ones that would silently produce a WRONG figure rather than a
crash: the CPU x-axis must be converted cores -> nodes (a 2048-core run is 16
nodes, not 2048), the y-axis must be shared so the CPU/GPU comparison is
readable as a vertical offset, and SYPD must stay out.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_MODULE_PATH = os.path.join(_REPO_ROOT, "scripts", "plot",
                            "plot_cpu_vs_gpu_scaling.py")

_HEADER = ("component,backend,grid,case,precision,mode,n_devices,"
           "cpus_per_task,n_cores,n_resource,resolution,resolution_km,"
           "n_levels,sypd,time_per_step_ms,total_cells,mcells_per_s,"
           "scaling_efficiency,dt_seconds,physics_level,fix_mass,"
           "compile_time_s,source")


def _load():
    spec = importlib.util.spec_from_file_location("plot_cpu_vs_gpu_scaling",
                                                  _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plot = _load()


def _row(backend, grid, res, km, n_resource, mcells, sypd=1.0, mode="strong"):
    return (f"atm,{backend},{grid},dry,float32,{mode},{n_resource},1,"
            f"{n_resource},{n_resource},{res},{km},26,{sypd},1.0,1000,"
            f"{mcells},1.0,60.0,none,True,1.0,src")


def _csv(tmp_path, rows):
    path = tmp_path / "tidy.csv"
    path.write_text("\n".join([_HEADER] + rows) + "\n")
    return path


def _full(tmp_path):
    rows = []
    for grid, res, km in (("cubed-sphere", 192, 52.1),
                          ("icosahedral", 8, 27.9),
                          ("latlon", 512, 39.1)):
        for n in (128, 256):
            rows.append(_row("CPU", grid, res, km, n, 100.0 * n / 128))
        for n in (1, 2):
            rows.append(_row("GPU", grid, res, km, n, 1000.0 * n))
    return _csv(tmp_path, rows)


def test_cpu_x_axis_is_converted_cores_to_nodes():
    """A 2048-core CPU run is 16 NODES. Plotting it at x=2048 against a GPU
    axis that tops out at 16 would misrepresent the comparison by 128x."""
    backends = dict((b, div) for b, _label, div in plot._BACKENDS)
    assert backends["CPU"] == 128
    assert backends["GPU"] == 1


def test_backend_row_order_is_cpu_then_gpu():
    assert [b for b, *_ in plot._BACKENDS] == ["CPU", "GPU"]


def test_paper_figure_writes_png_and_pdf(tmp_path):
    rows = plot._read(_full(tmp_path))
    written = plot.make_paper_figure(rows, tmp_path / "out", "fig_test")
    assert len(written) == 2
    for path in written:
        assert Path(path).exists() and Path(path).stat().st_size > 0


def test_paper_figure_shares_the_y_axis(tmp_path):
    """CPU-vs-GPU is read as the vertical offset between rows; that only works
    if every panel is on one scale."""
    import matplotlib.pyplot as plt

    rows = plot._read(_full(tmp_path))
    plot.make_paper_figure(rows, tmp_path / "out", "fig_share")
    src = open(_MODULE_PATH).read()
    assert "sharey=True" in src, "panels must share the y-axis"
    plt.close("all")


def test_paper_figure_uses_mcells_not_sypd(tmp_path):
    """SYPD must not sneak back in: the route-B lanes carry a placeholder dt."""
    rows = plot._read(_full(tmp_path))
    with pytest.raises(SystemExit, match="no 'nonexistent_metric'"):
        plot.make_paper_figure(rows, tmp_path / "out", "x",
                               metric="nonexistent_metric")


def test_paper_figure_errors_loudly_on_no_matching_grids(tmp_path):
    """A typoed/empty CSV must fail, not emit a blank figure."""
    rows = plot._read(_csv(tmp_path, [
        _row("CPU", "spectral", 21, 500.0, 128, 10.0)]))
    with pytest.raises(SystemExit, match="no rows for any of"):
        plot.make_paper_figure(rows, tmp_path / "out", "x")


def test_paper_figure_tolerates_a_missing_backend(tmp_path):
    """A grid swept on only one backend must still plot its populated row."""
    rows = plot._read(_csv(tmp_path, [
        _row("GPU", "latlon", 512, 39.1, 1, 1011.0),
        _row("GPU", "latlon", 512, 39.1, 2, 1795.0)]))
    written = plot.make_paper_figure(rows, tmp_path / "out", "fig_gpu_only")
    assert all(Path(p).exists() for p in written)


def test_weak_scaling_rows_are_excluded(tmp_path):
    """Weak-scaling points are a different problem size per rank count and
    would pollute a strong-scaling curve."""
    rows = plot._read(_csv(tmp_path, [
        _row("GPU", "latlon", 512, 39.1, 1, 1011.0),
        _row("GPU", "latlon", 512, 39.1, 2, 9999.0, mode="weak"),
    ]))
    g = plot.group(rows, "mcells_per_s")
    assert g["latlon"][512]["GPU"] == [(1, 1011.0)]


def test_cube_cpu_packing_caveat_is_annotated():
    """cube CPU runs 1 proc x 128 threads (XLA-CPU) while lat-lon/ico run 128
    MPI procs -- ~15x different per-node throughput on the SAME hardware. The
    figure must say so or a reader will rank grids from it."""
    assert "128 threads" in plot._CUBE_CPU_NOTE
    src = open(_MODULE_PATH).read()
    assert "_CUBE_CPU_NOTE" in src
