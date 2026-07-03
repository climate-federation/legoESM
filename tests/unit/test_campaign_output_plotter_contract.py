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
import pytest


def _real_output(averaging=None):
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
        summary=summary, health=health, corrected_field="C_K", averaging=averaging)
    return json.loads(json.dumps(out))               # the REAL on-disk JSON round-trip


def test_real_campaign_output_feeds_bias_trajectory_plotter(tmp_path):
    from scripts.plot.plot_campaign_bias_trajectory import (
        extract_campaign_trajectory,
        plot_campaign_bias_trajectory,
    )

    out = _real_output()
    tr = extract_campaign_trajectory(out)
    assert tr["round_updated"] == pytest.approx([0.6])  # REAL 'biases' read (fp32-safe)
    assert tr["initial_bias"] is not None and tr["final_bias"] is not None
    assert tr["pv_baseline"]["T_rmse_K"] == 3.0      # the REAL per-variable block was read
    png = tmp_path / "t.png"
    plot_campaign_bias_trajectory(out, str(png))
    assert png.exists() and png.stat().st_size > 0


def test_stalled_all_rejected_campaign_still_plots(tmp_path):
    """A STALLED run — EVERY round REJECTED by the monotonic gate (the operator's
    experience when the bias is idealization-dominated and the LES C_K cannot move it,
    iter 412) — must still produce a trajectory PNG and a non-'improved' health verdict,
    NOT crash: this is exactly the output the operator inspects to SEE that clause-6 did
    not improve + why (no accepted markers, the start/final lines coincide)."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.bias_metrics import BiasImprovement
    from legoesm.training.campaign_summary import campaign_health, summarize_campaign
    from legoesm.training.correction_loop import CampaignResult, CorrectionResult

    from scripts.plot.plot_campaign_bias_trajectory import (
        extract_campaign_trajectory,
        plot_campaign_bias_trajectory,
    )
    from scripts.run.run_correction_campaign import build_campaign_output_dict

    field = jnp.full((4,), 0.4)
    # The round WORSENED (1.0 -> 1.05) ⇒ improved=False ⇒ the gate rejects it.
    bias = BiasImprovement(
        baseline_bias=jnp.asarray(1.0), updated_bias=jnp.asarray(1.05),
        absolute_reduction=jnp.asarray(-0.05), fractional_improvement=jnp.asarray(-0.05),
        improved=jnp.asarray(False))
    res = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=field),
        iterations=(CorrectionResult(
            updated_config=None, bias=bias, worst_column_change=jnp.asarray(0.0),
            feedback_field=field.reshape(2, 2), n_corrected=1, n_diagnosed=1,
            n_diagnoses_valid=1, per_variable_bias=None),),
        final_field=field.reshape(2, 2), accepted=(False,), stop_reason="max_iterations")
    summary = summarize_campaign(res, promotion_key="clubb_lite_C_K")
    assert summary.final_bias == pytest.approx(1.0)         # no accepted round ⇒ stays at baseline
    health = campaign_health(summary)
    assert not health.ok                                    # clause-6 NOT achieved
    out = json.loads(json.dumps(build_campaign_output_dict(
        res, grid_provenance={"grid_type": "latlon", "shape_2d": [2, 2], "ncol": 4},
        summary=summary, health=health, corrected_field="C_K", averaging=None)))
    tr = extract_campaign_trajectory(out)
    assert tr["accepted"] == [False]                        # no accepted markers to plot
    png = tmp_path / "stalled.png"
    plot_campaign_bias_trajectory(out, str(png))            # must not crash (empty accepted set)
    assert png.exists() and png.stat().st_size > 0


def test_real_averaging_block_flows_producer_to_every_consumer(tmp_path):
    """PRODUCER→CONSUMER contract for the iter-267 ``averaging`` block (iter 278): a REAL
    ``build_campaign_output_dict(averaging=…)`` output, after a JSON round-trip, must feed
    the bias-plotter caption AND surface in the deploy-check — proactively catching the
    drift class that broke ``_maybe_write_env_kernel`` (iter 276) when a new key landed.
    Absent ``averaging`` must NOT add the key (back-compat with the deploy round-trip)."""
    from scripts.plot.plot_campaign_bias_trajectory import (
        _averaging_caption,
        extract_campaign_trajectory,
    )

    # back-compat: no averaging passed ⇒ no key (the deploy/round-trip contract).
    assert "averaging" not in _real_output()

    out = _real_output(averaging={"era5_n_times": 30, "era5_time_idx": 12})
    assert out["averaging"] == {"era5_n_times": 30, "era5_time_idx": 12}
    # bias-plotter reads it through the REAL output → the climatology caption.
    tr = extract_campaign_trajectory(out)
    assert tr["averaging"] == {"era5_n_times": 30, "era5_time_idx": 12}
    assert _averaging_caption(tr["averaging"]) == "vs 30-time ERA5 climatology @ idx 12"
    # a snapshot output drives the snapshot caption (the weather-vs-climate flag).
    snap = _real_output(averaging={"era5_n_times": 1, "era5_time_idx": 5})
    assert _averaging_caption(
        extract_campaign_trajectory(snap)["averaging"]) == "vs ERA5 snapshot @ idx 5"


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
