"""Unit test for ``scripts/plot/plot_campaign_bias_trajectory.py`` — the campaign
output → bias-trajectory PNG (the done-criterion's visual confirmation).
"""

from __future__ import annotations

import json


def _output(**over):
    """A representative campaign output dict (the build_campaign_output_dict shape)."""
    out = {
        "C_K": [0.4, 0.5],
        "biases": [[5.0, 4.0, True], [4.0, 4.2, False]],   # [baseline, updated, improved]
        "accepted": [True, False],
        "step_fractions": [1.0, 0.5],
        "summary": {
            "n_rounds": 2, "initial_bias": 5.0, "final_bias": 4.0,
            "per_variable_bias": {
                "baseline": {"T_rmse_K": 3.0, "qv_rmse_kg_kg": 1.0e-3, "wind_rmse_m_s": 5.0},
                "final": {"T_rmse_K": 2.0, "qv_rmse_kg_kg": 1.1e-3, "wind_rmse_m_s": 4.0},
                "improved": {"T": True, "qv": False, "wind": True},
            },
        },
        "health": {"status": "improved", "message": "bias fell"},
    }
    out.update(over)
    return out


def test_extract_campaign_trajectory():
    from scripts.plot.plot_campaign_bias_trajectory import extract_campaign_trajectory

    tr = extract_campaign_trajectory(_output())
    assert tr["round_baseline"] == [5.0, 4.0] and tr["round_updated"] == [4.0, 4.2]
    assert tr["accepted"] == [True, False]
    assert tr["initial_bias"] == 5.0 and tr["final_bias"] == 4.0
    assert tr["pv_baseline"]["T_rmse_K"] == 3.0 and tr["pv_final"]["wind_rmse_m_s"] == 4.0
    assert tr["health"]["status"] == "improved"


def test_averaging_caption_distinguishes_snapshot_from_climatology():
    """The plot caption surfaces the iter-267 averaging provenance: a single ERA5 time
    is a SNAPSHOT, N>1 a climatology (the bias means different things); absent → ''."""
    from scripts.plot.plot_campaign_bias_trajectory import (
        _averaging_caption,
        extract_campaign_trajectory,
    )

    assert _averaging_caption({}) == ""                              # pre-267 / mock
    assert _averaging_caption({"era5_n_times": 1, "era5_time_idx": 5}) == \
        "vs ERA5 snapshot @ idx 5"
    assert _averaging_caption({"era5_n_times": 30, "era5_time_idx": 12}) == \
        "vs 30-time ERA5 climatology @ idx 12"
    # the extractor reads the block (or {} when absent — no crash on a pre-267 output).
    tr = extract_campaign_trajectory(_output(averaging={"era5_n_times": 30, "era5_time_idx": 12}))
    assert tr["averaging"] == {"era5_n_times": 30, "era5_time_idx": 12}
    assert extract_campaign_trajectory({})["averaging"] == {}


def test_extract_campaign_trajectory_tolerates_partial():
    """A partial / mock output (no biases / summary / per-variable) yields empties / None,
    not a crash — so a 0-round or diverged run still plots."""
    from scripts.plot.plot_campaign_bias_trajectory import extract_campaign_trajectory

    tr = extract_campaign_trajectory({})
    assert tr["round_baseline"] == [] and tr["pv_baseline"] is None
    assert tr["initial_bias"] is None and tr["accepted"] == []


def test_plot_campaign_bias_trajectory_renders(tmp_path):
    from scripts.plot.plot_campaign_bias_trajectory import plot_campaign_bias_trajectory

    png = tmp_path / "traj.png"
    plot_campaign_bias_trajectory(_output(), str(png))
    assert png.exists() and png.stat().st_size > 0


def test_plot_campaign_bias_trajectory_renders_partial(tmp_path):
    """A 0-round output (no per-variable block) still renders (the per-variable panel
    falls back to a 'no per-variable bias' note instead of crashing)."""
    from scripts.plot.plot_campaign_bias_trajectory import plot_campaign_bias_trajectory

    png = tmp_path / "empty.png"
    plot_campaign_bias_trajectory(
        {"biases": [], "accepted": [], "summary": {}, "health": {"status": "no_change"}},
        str(png))
    assert png.exists() and png.stat().st_size > 0


def test_diverged_run_with_null_bias_extracts_as_nan_and_renders(tmp_path):
    """A DIVERGED round serializes its bias as JSON null (iter 245). The plotter must map
    it to a NaN GAP — not crash on ``float(None)`` — and still render (iter 248).

    ``test_extract_campaign_trajectory_tolerates_partial`` only covers an EMPTY output, so
    the actual null-bias path (the bug) was unexercised."""
    import math

    from scripts.plot.plot_campaign_bias_trajectory import (
        extract_campaign_trajectory,
        plot_campaign_bias_trajectory,
    )

    out = _output(biases=[[5.0, 4.0, True], [4.0, None, False]])   # round 2 diverged -> null
    tr = extract_campaign_trajectory(out)
    assert tr["round_updated"][0] == 4.0
    assert math.isnan(tr["round_updated"][1])                      # null -> NaN gap, not a crash
    png = tmp_path / "diverged.png"
    plot_campaign_bias_trajectory(out, str(png))                   # renders the gap, no crash
    assert png.exists() and png.stat().st_size > 0


def test_plot_campaign_main_reads_json_and_writes_png(tmp_path):
    from scripts.plot.plot_campaign_bias_trajectory import main

    j = tmp_path / "out.json"
    j.write_text(json.dumps(_output()))
    png = tmp_path / "p.png"
    assert main([str(j), "--png", str(png)]) == 0
    assert png.exists() and png.stat().st_size > 0
