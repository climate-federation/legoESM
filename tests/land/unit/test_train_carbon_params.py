"""Unit tests for the Stage-B (B2) carbon-parameter calibration trainer
(``scripts/run/train_carbon_params.py``).

Fast: NO full calibration.  Three checks -- the modular loss registry computes a
finite cover-weighted SOC MSE; ``to_overrides()`` -> ``apply_param_overrides``
round-trips a trainable leaf into ``CarbonConfig``; and ONE optimizer step on a
tiny ``--dry-run-synthetic`` world changes the loss finitely and writes a
well-formed ``tuned_carbon_parameters.json``.

Compute-node scale for the ``--quick`` step (JIT-compiles the coupled land+carbon
spin-up + its reverse-mode) -- run via the sbatch/srun wrapper, NOT the login
node.  ``JAX_ENABLE_X64=1``; forced CPU here.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import numpy.testing as npt
import pytest

from scripts.run import train_carbon_params as tcp


def test_cover_weighted_mse_matches_hand_value():
    """area_weighted_mse mapped onto the archetype axis == sum_a w_a e_a^2 / sum_a w_a."""
    import jax.numpy as jnp

    pred = jnp.asarray([1.0, 2.0, 3.0])
    target = jnp.asarray([1.5, 2.0, 5.0])
    weight = jnp.asarray([1.0, 3.0, 0.0])
    # sum w e^2 = 1*0.25 + 3*0 + 0*4 = 0.25 ; sum w = 4 -> 0.0625
    got = float(tcp.cover_weighted_mse(pred, target, weight))
    npt.assert_allclose(got, 0.0625, rtol=1e-12)


def test_loss_registry_sums_weighted_terms():
    """The modular registry sums w_k * cover_weighted_mse without touching the loop."""
    import jax.numpy as jnp

    target = jnp.asarray([1.5, 2.0, 5.0])
    cover = jnp.asarray([1.0, 3.0, 0.0])
    # extractor is identity on a fake "equilibrium" -> reuse the MSE math above.
    fake_eq = jnp.asarray([1.0, 2.0, 3.0])
    losses = {"soc": tcp.LossTerm(target=target, extractor=lambda x: x, weight=2.0)}
    total = float(tcp._total_loss(fake_eq, losses, cover))
    npt.assert_allclose(total, 2.0 * 0.0625, rtol=1e-12)
    assert np.isfinite(total)


def test_nan_target_is_dropped_by_zero_weight():
    """A NaN observed target with weight 0 must NOT poison the (finite) loss."""
    import jax.numpy as jnp

    pred = jnp.asarray([1.0, 2.0])
    target = jnp.asarray([1.5, 0.0])       # 2nd entry sanitised from NaN -> 0
    weight = jnp.asarray([2.0, 0.0])       # ... with zero cover weight
    got = float(tcp.cover_weighted_mse(pred, target, weight))
    npt.assert_allclose(got, 0.25, rtol=1e-12)   # 2*0.25 / 2
    assert np.isfinite(got)


def test_overrides_roundtrip_into_carbon_config():
    """to_overrides() -> apply_param_overrides splices a traced leaf into CarbonConfig,
    and the warm start equals the production defaults (sub-clamp precision)."""
    import jax.numpy as jnp
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.training.param_collector import apply_param_overrides

    params = tcp.build_carbon_trainables(som_only=True)
    names = {c.field for c in params.constraints}
    assert set(tcp.SOM_FIELDS) == names, names

    overrides = tcp._carbon_overrides(params)
    assert set(overrides) == set(tcp.SOM_FIELDS)

    cfg = apply_param_overrides(CarbonConfig(), overrides)
    for field in tcp.SOM_FIELDS:
        v = float(np.asarray(getattr(cfg, field)))
        assert np.isfinite(v)
    # Warm start recovers the production default for som_freeze_floor.
    npt.assert_allclose(
        float(np.asarray(getattr(cfg, "som_freeze_floor"))),
        CarbonConfig().som_freeze_floor, rtol=1e-6)


def test_som_only_filter_selects_exactly_seven_and_excludes_q10():
    """The fast-analytic-valid SOM set is the 7 SOM-pool-only fields; codex fix:
    ``Q10_het_exp`` is deliberately EXCLUDED (its fast-mode gradient would be
    partial -- see the comment on ``SOM_FIELDS``), even though it is a genuine
    tier-2 ``land.carbon`` trainable (it remains available via
    ``--slow-spinup-grad``)."""
    params = tcp.build_carbon_trainables(som_only=True)
    names = {c.field for c in params.constraints}
    assert len(params.constraints) == len(tcp.SOM_FIELDS) == 7
    assert "Q10_het_exp" not in tcp.SOM_FIELDS
    assert "Q10_het_exp" not in names


def test_quick_dry_run_writes_wellformed_json(tmp_path):
    """One optimizer step on a 2-3-archetype synthetic world changes the loss
    finitely and writes a well-formed tuned_carbon_parameters.json + scorecard."""
    outdir = tmp_path / "carbon_calib"
    rc = tcp.main(["--quick", "--output", str(outdir)])
    assert rc == 0

    tuned = json.loads((outdir / "tuned_carbon_parameters.json").read_text())
    assert tuned["scheme_key"] == "land.carbon"
    assert tuned["note"].startswith("RECOMMENDED")
    params = tuned["tuned_carbon_parameters"]
    assert isinstance(params, dict) and len(params) >= 1
    for field, value in params.items():
        assert np.isfinite(value), (field, value)
        assert field in tcp.SOM_FIELDS
    # loss history present; the trainer records at least the initial loss.
    hist = tuned["loss_history"]
    assert len(hist) >= 1 and all(np.isfinite(x) for x in hist)
    assert tuned["training"]["som_only"] is True
    # Each parameter row carries default/tuned/bounds.
    for row in tuned["parameters"]:
        assert {"field", "production_default", "tuned", "lower", "upper"} <= set(row)
        assert row["lower"] <= row["tuned"] <= row["upper"]
        assert np.isfinite(row["tuned"])

    scorecard = json.loads((outdir / "scorecard.json").read_text())
    assert "per_pft_soc" in scorecard
    rmse = scorecard["cover_weighted_soc_rmse_kgC_m2"]
    assert np.isfinite(rmse["default"]) and np.isfinite(rmse["tuned"])
    # A single MUON line-search step must not INCREASE the loss.
    assert hist[-1] <= hist[0] + 1e-9
    # --quick uses the DEFAULT fast-analytic forward: the run records the forward
    # kind + the analytic-vs-spin-up match, and the closed form must be finite.
    training = tuned["training"]
    assert training["forward"] == "fast_analytic"
    match = training["analytic_vs_spinup_match"]
    assert match["n_archetypes"] >= 1
    assert np.isfinite(match["mean_abs_rel_err"]) and match["mean_abs_rel_err"] >= 0.0


def test_fast_analytic_is_default_and_slow_flag_flips_it():
    """--fast-analytic (default True) trains the closed form; --slow-spinup-grad
    selects the B1 grad-through-spin-up forward."""
    p = tcp.build_arg_parser()
    assert p.parse_args(["--dry-run-synthetic"]).fast_analytic is True
    assert p.parse_args(["--dry-run-synthetic", "--fast-analytic"]).fast_analytic is True
    assert p.parse_args(
        ["--dry-run-synthetic", "--slow-spinup-grad"]).fast_analytic is False


def test_fast_analytic_and_all_carbon_params_raises():
    """codex fix: --fast-analytic (default True) + --all-carbon-params must raise
    -- the fast closed form freezes every non-SOM tier-2 input (GPP/phenology/
    allocation) at the DEFAULT params in the one-time precompute, so those fields
    would get a stale/zero gradient, not a real calibration signal.
    --slow-spinup-grad --all-carbon-params (the correct grad-through-spin-up
    forward) is the valid way to train the full tier-2 set and must NOT raise."""
    with pytest.raises(SystemExit, match="not supported"):
        tcp.main(["--dry-run-synthetic", "--fast-analytic", "--all-carbon-params",
                  "--steps", "1"])
    # --fast-analytic defaults True, so omitting it still hits the same guard.
    with pytest.raises(SystemExit, match="not supported"):
        tcp.main(["--dry-run-synthetic", "--all-carbon-params", "--steps", "1"])
    # The slow-path + --all-carbon-params combo remains valid: _finalize_args
    # must NOT raise for it.
    args = tcp._finalize_args(tcp.build_arg_parser().parse_args(
        ["--dry-run-synthetic", "--slow-spinup-grad", "--all-carbon-params"]))
    assert args.fast_analytic is False
    assert args.all_carbon_params is True


def test_quick_slow_spinup_grad_path_still_runs(tmp_path):
    """The slow grad-through-spin-up forward remains available + monotone (one
    optimizer step on a tiny synthetic world)."""
    outdir = tmp_path / "carbon_calib_slow"
    rc = tcp.main(["--quick", "--slow-spinup-grad", "--output", str(outdir)])
    assert rc == 0
    tuned = json.loads((outdir / "tuned_carbon_parameters.json").read_text())
    assert tuned["training"]["forward"] == "slow_spinup_grad"
    hist = tuned["loss_history"]
    assert len(hist) >= 1 and all(np.isfinite(x) for x in hist)
    assert hist[-1] <= hist[0] + 1e-9


def test_resolve_om_to_oc_is_preset_aware():
    """Default om_to_oc: 0.58 (van Bemmelen) for CLM5 organic MATTER, 1.0 for the
    HWSD-carbon legoesm_surfdata; an explicit --om-to-oc overrides either."""
    import argparse

    clm5 = argparse.Namespace(om_to_oc=None, surfdata_preset="clm5_surfdata")
    lego = argparse.Namespace(om_to_oc=None, surfdata_preset="legoesm_surfdata")
    explicit = argparse.Namespace(om_to_oc=0.7, surfdata_preset="clm5_surfdata")
    assert tcp._resolve_om_to_oc(clm5) == tcp._VAN_BEMMELEN_OC_PER_OM == 0.58
    assert tcp._resolve_om_to_oc(lego) == 1.0
    assert tcp._resolve_om_to_oc(explicit) == 0.7


def test_input_mode_required():
    """No input mode -> a clear SystemExit (never a silent empty run)."""
    with pytest.raises(SystemExit):
        tcp.main(["--steps", "1"])
