"""``plot_scm_les_tuning_summary.py`` wraps its panels when the run is wide.

The figure is one panel per case plus a combined one. The five-case campaign
that set the format was six panels across; the eight-case campaign is nine,
which in one row is a 36-inch strip. These pin the wrap and the two things a
wrapped grid gets wrong: spare cells drawn as empty panels, and a y label that
only reaches the first row.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "plot" / "plot_scm_les_tuning_summary.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_plot_summary", _SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("n,expect", [
    (2, (1, 2)), (6, (1, 6)),      # the format the five-case run established
    (7, (2, 4)), (9, (2, 5)), (10, (2, 5)),
])
def test_grid_wraps_only_when_wide(mod, n, expect):
    assert mod._grid(n) == expect


def test_the_grid_always_has_room_for_every_panel(mod):
    for n in range(1, 17):
        nrows, ncols = mod._grid(n)
        assert nrows * ncols >= n, f"{n} panels do not fit in {nrows}x{ncols}"
        assert nrows * ncols - n < ncols, (
            f"{nrows}x{ncols} for {n} panels leaves a whole spare row")


def _payload(cases, schemes=("louis", "tke", "ysu")):
    return {
        "protocol": {"cases": list(cases)},
        "cases": [{"case": c, "window_hours": [4.0, 6.0]} for c in cases],
        "schemes": [
            {
                "scheme": s,
                "score_default": 0.8 + 0.1 * i,
                "score_tuned": 0.6 + 0.1 * i,
                "per_case_default": {c: 0.5 + 0.1 * j
                                     for j, c in enumerate(cases)},
                "per_case_tuned": {c: 0.4 + 0.1 * j
                                   for j, c in enumerate(cases)},
            }
            for i, s in enumerate(schemes)
        ],
    }


@pytest.mark.parametrize("cases", [
    ("bomex", "rico", "gabls1", "cbl", "ekman"),
    ("ekman", "gabls1", "cbl", "wangara", "bomex", "rico", "astex", "dycoms"),
])
def test_the_figure_renders_at_both_campaign_widths(mod, tmp_path, cases):
    """End to end, not just the helper: the wrapped path has to survive the
    spare-cell hiding and the per-row label, and this plot is the LAST thing
    that runs after ~48 h of fitting."""
    (tmp_path / "tuned_parameters.json").write_text(json.dumps(_payload(cases)))
    out = tmp_path / "summary.png"
    assert mod.main([str(tmp_path), "--out", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 10_000


def test_spare_cells_are_hidden_not_left_blank(mod, tmp_path):
    """Nine panels on a 2x5 grid leaves one cell; an empty box with axes on it
    reads as a panel whose data went missing."""
    import matplotlib.pyplot as plt
    cases = ("ekman", "gabls1", "cbl", "wangara", "bomex", "rico", "astex",
             "dycoms")
    (tmp_path / "tuned_parameters.json").write_text(json.dumps(_payload(cases)))
    plt.close("all")
    mod.main([str(tmp_path), "--out", str(tmp_path / "s.png")])
    fig = plt.gcf()
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(fig.axes) == 10, f"expected a 2x5 grid, got {len(fig.axes)}"
    assert len(visible) == 1 + len(cases), (
        f"{len(visible)} visible panels for {1 + len(cases)} of data")
    plt.close("all")


def test_every_row_carries_the_y_label(mod, tmp_path):
    import matplotlib.pyplot as plt
    cases = ("ekman", "gabls1", "cbl", "wangara", "bomex", "rico", "astex",
             "dycoms")
    (tmp_path / "tuned_parameters.json").write_text(json.dumps(_payload(cases)))
    plt.close("all")
    mod.main([str(tmp_path), "--out", str(tmp_path / "s.png")])
    fig = plt.gcf()
    labelled = [ax for ax in fig.axes if ax.get_ylabel()]
    nrows, _ncols = mod._grid(1 + len(cases))
    assert len(labelled) == nrows, (
        f"{len(labelled)} labelled axes on {nrows} rows; a wrapped row without "
        "a label leaves half the figure unitless")
    plt.close("all")


def test_a_run_with_nothing_tuned_is_a_hard_error(mod, tmp_path):
    payload = _payload(("bomex",))
    for s in payload["schemes"]:
        s["score_tuned"] = None
    (tmp_path / "tuned_parameters.json").write_text(json.dumps(payload))
    with pytest.raises(SystemExit):
        mod.main([str(tmp_path), "--out", str(tmp_path / "s.png")])
