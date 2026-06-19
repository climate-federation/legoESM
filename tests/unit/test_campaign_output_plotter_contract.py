"""Producer→consumer contract: a REAL ``build_campaign_output_dict`` output feeds BOTH
campaign plotters.

The per-plotter tests use hand-built dicts, so a format DRIFT in the campaign output
(e.g. renaming ``biases`` or the per-variable block) would silently break the plotters
without failing them.  This drives the actual producer (the campaign's
``build_campaign_output_dict``, after a JSON round-trip) into both plotters' extractors.
"""

from __future__ import annotations

import json

import jax.numpy as jnp


def _real_output():
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.bias_metrics import (
        BiasImprovement,
        PerVariableBias,
        compare_per_variable_bias,
    )
    from legoesm.training.campaign_summary import campaign_health, summarize_campaign
    from legoesm.training.correction_loop import CampaignResult, CorrectionResult

    from scripts.run.run_correction_campaign import build_campaign_output_dict

    field = jnp.array([0.42, 0.55, 0.61, 0.73])
    pvb = compare_per_variable_bias(
        PerVariableBias(jnp.asarray(3.0), jnp.asarray(1.0e-3), jnp.asarray(5.0),
                        jnp.asarray(float("nan"))),
        PerVariableBias(jnp.asarray(2.0), jnp.asarray(1.1e-3), jnp.asarray(4.0),
                        jnp.asarray(float("nan"))))
    bias = BiasImprovement(
        baseline_bias=jnp.asarray(1.0), updated_bias=jnp.asarray(0.6),
        absolute_reduction=jnp.asarray(0.4), fractional_improvement=jnp.asarray(0.4),
        improved=jnp.asarray(True))
    res = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=field),
        iterations=(CorrectionResult(
            updated_config=None, bias=bias, worst_column_change=jnp.asarray(0.0),
            feedback_field=field.reshape(2, 2), n_corrected=2, n_diagnosed=2,
            n_diagnoses_valid=2, per_variable_bias=pvb),),
        final_field=field.reshape(2, 2), accepted=(True,), stop_reason="converged")
    summary = summarize_campaign(res, promotion_key="clubb_lite_C_K")
    health = campaign_health(summary)
    out = build_campaign_output_dict(
        res, grid_provenance={"grid_type": "latlon", "shape_2d": [2, 2], "ncol": 4},
        summary=summary, health=health, corrected_field="C_K")
    return json.loads(json.dumps(out))               # the REAL on-disk JSON round-trip


def test_real_campaign_output_feeds_bias_trajectory_plotter(tmp_path):
    from scripts.plot.plot_campaign_bias_trajectory import (
        extract_campaign_trajectory,
        plot_campaign_bias_trajectory,
    )

    out = _real_output()
    tr = extract_campaign_trajectory(out)
    assert tr["round_updated"] == [0.6]              # the REAL 'biases' key was read
    assert tr["initial_bias"] is not None and tr["final_bias"] is not None
    assert tr["pv_baseline"]["T_rmse_K"] == 3.0      # the REAL per-variable block was read
    png = tmp_path / "t.png"
    plot_campaign_bias_trajectory(out, str(png))
    assert png.exists() and png.stat().st_size > 0


def test_real_campaign_output_feeds_coefficient_map_plotter(tmp_path):
    from scripts.plot.plot_corrected_coefficient_field import (
        extract_coefficient_fields,
        plot_corrected_coefficient_field,
    )

    out = _real_output()
    coeffs = extract_coefficient_fields(out)
    assert "C_K" in coeffs and len(coeffs["C_K"]) == 4   # the REAL top-level C_K array
    png = tmp_path / "c.png"
    plot_corrected_coefficient_field(out, str(png))
    assert png.exists() and png.stat().st_size > 0
