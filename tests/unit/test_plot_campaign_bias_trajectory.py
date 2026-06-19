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


def test_plot_campaign_main_reads_json_and_writes_png(tmp_path):
    from scripts.plot.plot_campaign_bias_trajectory import main

    j = tmp_path / "out.json"
    j.write_text(json.dumps(_output()))
    png = tmp_path / "p.png"
    assert main([str(j), "--png", str(png)]) == 0
    assert png.exists() and png.stat().st_size > 0
