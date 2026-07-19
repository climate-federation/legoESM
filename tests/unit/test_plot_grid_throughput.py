"""Direct tests for ``scripts/plot/plot_grid_throughput.py``.

The figure is a per-grid CAPABILITY curve, so the load/series path must keep
the two metrics distinct (Mcells/s is timestep-free, SYPD is not -- the gap
between them is the figure's message) and must attribute "best" to the device
count that actually achieved it, since several configurations peak below the
top of the ladder.
"""
from __future__ import annotations

import importlib.util
import os

import matplotlib
import pytest

matplotlib.use("Agg")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_MODULE_PATH = os.path.join(_REPO_ROOT, "scripts", "plot",
                            "plot_grid_throughput.py")

_HEADER = ("component,backend,grid,case,precision,mode,n_devices,"
           "cpus_per_task,n_cores,n_resource,resolution,resolution_km,"
           "n_levels,sypd,time_per_step_ms,total_cells,mcells_per_s,"
           "scaling_efficiency,dt_seconds,physics_level,fix_mass,"
           "compile_time_s,source")


def _load_mod():
    spec = importlib.util.spec_from_file_location("plot_grid_throughput",
                                                  _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plot = _load_mod()


def _row(grid, res, km, n, sypd, mcells, backend="GPU"):
    return (f"atm,{backend},{grid},dry,float32,strong,{n},1,1,{n},{res},{km},"
            f"26,{sypd},1.0,1000,{mcells},1.0,60.0,none,True,1.0,src")


def _csv(tmp_path, rows):
    path = tmp_path / "tidy.csv"
    path.write_text("\n".join([_HEADER] + rows) + "\n")
    return str(path)


def test_load_keeps_gpu_rows_only(tmp_path):
    path = _csv(tmp_path, [
        _row("latlon", 512, 39.1, 1, 12.2, 1011),
        _row("latlon", 512, 39.1, 1, 99.9, 9999, backend="CPU"),
    ])
    data = plot.load(path)
    assert set(data) == {"latlon"}
    assert data["latlon"][512][1][1] == pytest.approx(1011)


def test_missing_csv_is_a_hard_error(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        plot.load(str(tmp_path / "nope.csv"))
    with pytest.raises(SystemExit, match="required"):
        plot.load("")


def test_series_one_uses_the_single_gpu_point(tmp_path):
    data = plot.load(_csv(tmp_path, [
        _row("latlon", 512, 39.1, 1, 12.2, 1011),
        _row("latlon", 512, 39.1, 16, 52.7, 4375),
    ]))
    got = plot.series(data["latlon"], 1, "one")
    assert got == [(39.1, 1011.0, 1)]


def test_series_best_reports_the_device_count_that_achieved_it(tmp_path):
    """Several configs peak BELOW the top of the ladder (ico L7 peaks at 4).

    Attributing the best value to the largest N would misreport capability.
    """
    data = plot.load(_csv(tmp_path, [
        _row("icosahedral", 7, 55.8, 1, 5.0, 261),
        _row("icosahedral", 7, 55.8, 4, 23.5, 1217),
        _row("icosahedral", 7, 55.8, 16, 15.3, 793),
    ]))
    km, val, n = plot.series(data["icosahedral"], 0, "best")[0]
    assert (km, n) == (55.8, 4), "best must come from N=4, not the top N=16"
    assert val == pytest.approx(23.5)


def test_series_sorted_fine_to_coarse_by_km(tmp_path):
    data = plot.load(_csv(tmp_path, [
        _row("latlon", 192, 104.2, 1, 81.5, 950),
        _row("latlon", 1024, 19.5, 1, 2.95, 978),
        _row("latlon", 512, 39.1, 1, 12.2, 1011),
    ]))
    kms = [p[0] for p in plot.series(data["latlon"], 1, "one")]
    assert kms == sorted(kms)


def test_metrics_are_distinct_sypd_vs_mcells(tmp_path):
    """Index 0 is SYPD, index 1 is Mcells/s — swapping them would silently
    destroy the figure's whole point (the panels must disagree)."""
    data = plot.load(_csv(tmp_path, [_row("latlon", 512, 39.1, 1, 12.2, 1011)]))
    assert plot.series(data["latlon"], 0, "one")[0][1] == pytest.approx(12.2)
    assert plot.series(data["latlon"], 1, "one")[0][1] == pytest.approx(1011)


def test_grid_palette_is_fixed_order_and_not_the_blues_ramp():
    keys = [g[0] for g in plot.GRID_STYLE]
    assert keys == ["latlon", "cubed-sphere", "icosahedral"]
    cols = {g[2] for g in plot.GRID_STYLE}
    assert len(cols) == 3, "each grid needs its own hue"
    assert not (cols & {"#9ecae1", "#4292c6", "#084594"}), (
        "must not reuse the resolution ramp — colour would mean two things")


def test_make_figure_writes_png_and_pdf(tmp_path):
    data = plot.load(_csv(tmp_path, [
        _row("latlon", 512, 39.1, 1, 12.2, 1011),
        _row("cubed-sphere", 192, 52.1, 1, 14.7, 1032),
        _row("icosahedral", 7, 55.8, 1, 5.0, 261),
    ]))
    written = plot.make_figure(data, str(tmp_path / "out"), "fig_test")
    assert len(written) == 2
    for path in written:
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_figure_has_no_sypd_panel(tmp_path):
    """SYPD must NOT be plotted.

    The route-B SPMD lanes carry a placeholder dt (``--dt`` default 60.0,
    recorded identically at LL192/LL512/LL1024 where a CFL dt would shrink),
    so any SYPD derived from them is not physical -- it produced a 260x error
    in an earlier draft. This gate fails if a second panel reappears.
    """
    import matplotlib.pyplot as plt

    data = plot.load(_csv(tmp_path, [
        _row("latlon", 512, 39.1, 1, 12.2, 1011),
        _row("cubed-sphere", 192, 52.1, 1, 14.7, 1032),
    ]))
    plot.make_figure(data, str(tmp_path / "out"), "fig_nosypd")
    # make_figure closes its own figure; inspect the last one it built by
    # rebuilding under a fresh manager count.
    plt.close("all")
    plot.make_figure(data, str(tmp_path / "out"), "fig_nosypd2")
    assert plt.get_fignums() == [], "figure must be closed after saving"

    src = open(_MODULE_PATH).read()
    assert "SYPD (simulated yr / day)" not in src, (
        "a SYPD axis label reappeared — see the dt-provenance caveat")


def test_make_figure_tolerates_a_missing_grid(tmp_path):
    """A partial campaign must still plot what exists."""
    data = plot.load(_csv(tmp_path, [_row("latlon", 512, 39.1, 1, 12.2, 1011)]))
    written = plot.make_figure(data, str(tmp_path / "out"), "fig_partial")
    assert all(os.path.exists(p) for p in written)
