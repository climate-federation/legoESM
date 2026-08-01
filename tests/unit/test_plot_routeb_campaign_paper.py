"""Direct tests for ``scripts/plot/plot_routeb_campaign_paper.py``.

`(grid, resolution, n_devices)` is NOT a unique key in a tidy CSV -- the same
geometry also varies by mode, precision and n_levels.  Filtering on backend
alone let a sibling row silently OVERWRITE its predecessor, resolved by file
order.  In one real campaign that happened twice: a float64 fan-out replaced
float32 across 18 of 39 GPU rows, and an icosahedral rerun at nlev=26 landed
beside stale nlev=8 rows.  Both read as performance regressions; neither was.
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
                            "plot_routeb_campaign_paper.py")

_HEADER = ("component,backend,grid,case,precision,mode,n_devices,"
           "cpus_per_task,n_cores,n_resource,resolution,resolution_km,"
           "n_levels,sypd,time_per_step_ms,total_cells,mcells_per_s,"
           "scaling_efficiency,dt_seconds,physics_level,fix_mass,"
           "compile_time_s,source")


def _load_mod():
    spec = importlib.util.spec_from_file_location(
        "plot_routeb_campaign_paper", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plot = _load_mod()


def _row(grid, res, n, mcells, precision="float32", mode="strong",
         nlev=26, backend="GPU"):
    return (f"atm,{backend},{grid},dry,{precision},{mode},{n},1,1,{n},{res},"
            f"39.1,{nlev},1.0,1.0,1000,{mcells},1.0,60.0,none,True,1.0,src")


def _csv(tmp_path, rows):
    p = tmp_path / "tidy.csv"
    p.write_text("\n".join([_HEADER] + rows) + "\n")
    return str(p)


def test_defaults_to_strong_float32(tmp_path):
    path = _csv(tmp_path, [
        _row("latlon", 512, 1, 1011, precision="float32"),
        _row("latlon", 512, 1, 529, precision="float64"),
        _row("latlon", 512, 1, 777, mode="weak"),
    ])
    data = plot.load(path)
    assert data[("latlon", 512)][1][1] == pytest.approx(1011), \
        "default must select strong/float32, not the last row in file"


def test_can_select_float64(tmp_path):
    path = _csv(tmp_path, [
        _row("latlon", 512, 1, 1011, precision="float32"),
        _row("latlon", 512, 1, 529, precision="float64"),
    ])
    assert plot.load(path, precision="float64")[("latlon", 512)][1][1] \
        == pytest.approx(529)


def test_n_levels_filter_separates_a_rerun_from_stale_rows(tmp_path):
    """The real case: an ico rerun at nlev=26 beside stale nlev=8 rows."""
    path = _csv(tmp_path, [
        _row("icosahedral", 8, 1, 335, nlev=8),
        _row("icosahedral", 8, 1, 474, nlev=26),
    ])
    assert plot.load(path, n_levels=26)[("icosahedral", 8)][1][1] == pytest.approx(474)
    assert plot.load(path, n_levels=8)[("icosahedral", 8)][1][1] == pytest.approx(335)


def test_weak_rows_never_leak_onto_a_strong_curve(tmp_path):
    path = _csv(tmp_path, [
        _row("latlon", 512, 1, 1011, mode="strong"),
        _row("latlon", 512, 2, 4242, mode="weak"),
    ])
    assert set(plot.load(path)[("latlon", 512)]) == {1}


def test_duplicate_after_filtering_is_a_hard_error(tmp_path):
    path = _csv(tmp_path, [
        _row("latlon", 512, 1, 1011, nlev=26),
        _row("latlon", 512, 1, 335, nlev=8),
    ])
    with pytest.raises(SystemExit, match="duplicate rows"):
        plot.load(path, n_levels=None)


def test_cpu_rows_are_excluded(tmp_path):
    path = _csv(tmp_path, [
        _row("latlon", 512, 1, 1011),
        _row("latlon", 512, 1, 9999, backend="CPU"),
    ])
    assert plot.load(path)[("latlon", 512)][1][1] == pytest.approx(1011)


def test_empty_selection_is_an_error_not_an_empty_figure(tmp_path):
    path = _csv(tmp_path, [_row("latlon", 512, 1, 1011)])
    with pytest.raises(SystemExit, match="no GPU rows matched"):
        plot.load(path, precision="float64")
