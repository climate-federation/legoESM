"""Unit test for ``scripts/experiment/check_campaign_deploy.py`` — the deploy-check
preflight (a campaign output + base config -> a grid-verified, validate_strict-sound
deployed config, reported without running the model).
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest


def _base_config(tmp_path, resolution=8, nlev=5):
    from scripts.experiment.write_amip_clubb_lite_config import main as cfg_main

    cfg = tmp_path / "base.json"
    assert cfg_main([str(cfg), "--resolution", str(resolution), "--nlev", str(nlev)]) == 0
    return str(cfg)


def _campaign_output(tmp_path, base_cfg, grid, *, name="out.json", field="C_K",
                     lo=0.3, hi=0.6, grid_for_provenance=None):
    from scripts.run.run_correction_campaign import _grid_provenance

    ncol = int(np.prod(grid.grid_shape_2d))
    out = tmp_path / name
    with open(out, "w") as f:
        json.dump({field: list(np.linspace(lo, hi, ncol)),
                   "grid": _grid_provenance(base_cfg, grid_for_provenance or grid),
                   "biases": [1.0, 0.5], "accepted": [True]}, f)
    return str(out)


def test_check_deploy_validates_and_reports_stats(tmp_path):
    from scripts.experiment.check_campaign_deploy import check_deploy
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    cfg = _base_config(tmp_path)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)
    out = _campaign_output(tmp_path, base_cfg, grid, lo=0.3, hi=0.6)

    stats = check_deploy(cfg, out)
    assert stats["n_columns"] == 128                     # 8 x 16
    assert stats["grid_shape"] == (8, 16)
    assert set(stats["corrected"]) == {"C_K"}            # only C_K was corrected
    assert stats["corrected"]["C_K"]["min"] == pytest.approx(0.3)
    assert stats["corrected"]["C_K"]["max"] == pytest.approx(0.6)


def test_main_returns_zero_and_prints_ok(tmp_path, capsys):
    from scripts.experiment.check_campaign_deploy import main
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    cfg = _base_config(tmp_path)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)
    out = _campaign_output(tmp_path, base_cfg, grid)
    rc = main(["--base-config", cfg, "--campaign-output", out])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "OK" in captured and "C_K" in captured


def test_build_deployed_config_carries_the_override_and_validates(tmp_path):
    """``build_deployed_config`` returns a runnable deployed config: the base config
    with ``turbulence_override`` set to the per-column correction (the canonical way
    to USE a campaign output, since the override does not serialize)."""
    from scripts.experiment.check_campaign_deploy import build_deployed_config
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    cfg = _base_config(tmp_path)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)
    out = _campaign_output(tmp_path, base_cfg, grid, lo=0.31, hi=0.59)

    deployed, dgrid = build_deployed_config(cfg, out)
    # The deployed config carries the per-column override (a real GCM driver would
    # then inject it) — base had a scalar default; deployed has the (128,) field.
    assert deployed.turbulence_override is not None
    ck = np.asarray(deployed.turbulence_override.clubb_lite.C_K)
    assert ck.shape == (128,)
    assert ck.min() == pytest.approx(0.31) and ck.max() == pytest.approx(0.59)
    assert tuple(dgrid.grid_shape_2d) == (8, 16)


def test_allow_unverified_grid_is_the_escape_hatch_but_keeps_the_length_check(tmp_path):
    """``allow_unverified_grid`` is the escape hatch for a campaign output that PREDATES
    grid provenance (no 'grid' block): refused by default (no provenance to verify),
    deployed on the array-LENGTH check alone when the flag is set — but a WRONG-LENGTH
    field still fails even then."""
    from scripts.experiment.check_campaign_deploy import check_deploy

    cfg = _base_config(tmp_path)                         # base 8x16 = 128 cols
    nogrid = tmp_path / "nogrid.json"                    # NO grid-provenance block
    nogrid.write_text(json.dumps({"C_K": list(np.linspace(0.3, 0.6, 128))}))

    with pytest.raises((ValueError, SystemExit)):        # default: provenance required
        check_deploy(cfg, str(nogrid))
    stats = check_deploy(cfg, str(nogrid), allow_unverified_grid=True)   # escape hatch
    assert stats["n_columns"] == 128

    bad_len = tmp_path / "badlen.json"
    bad_len.write_text(json.dumps({"C_K": list(np.linspace(0.3, 0.6, 64))}))  # 64 != 128
    with pytest.raises((ValueError, SystemExit)):        # length check survives the bypass
        check_deploy(cfg, str(bad_len), allow_unverified_grid=True)


def test_deployed_config_drives_a_real_driver_with_the_per_column_correction(tmp_path):
    """The runbook's step-5 deploy snippet must COMPOSE end-to-end: build_deployed_config
    → make_base_driver_builder → a REAL ModelDriver whose turbulence config carries the
    per-column LES-corrected C_K.

    This is what makes the operator's deployed run actually USE the correction (not the
    scalar default).  The AMIP-path analog of the iter-37 CMIP capstone, but through the
    OPERATOR's documented ``build_deployed_config`` + ``make_base_driver_builder``
    composition — no existing test covers that exact path (iter 205-207 wired
    ``corrected_turbulence_override`` → ``build_physics_pipeline`` directly).
    """
    from scripts.experiment.check_campaign_deploy import build_deployed_config
    from scripts.run.run_correction_campaign import (
        load_base_config_and_grid,
        make_base_driver_builder,
    )

    cfg = _base_config(tmp_path)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)
    ck_lo, ck_hi = 0.31, 0.59
    out = _campaign_output(tmp_path, base_cfg, grid, lo=ck_lo, hi=ck_hi)

    deployed, _ = build_deployed_config(cfg, out)
    build_driver, _ = make_base_driver_builder("amip")
    driver = build_driver(deployed)                      # constructs + setup()s the real driver
    drv_ck = np.asarray(driver.physics.turbulence_config.C_K)
    # The per-column correction reached the RUNNING driver's turbulence kernel config.
    assert drv_ck.shape == (128,)
    assert drv_ck.min() == pytest.approx(ck_lo) and drv_ck.max() == pytest.approx(ck_hi)


def test_check_deploy_refuses_a_different_grid_of_the_same_column_count(tmp_path):
    """The grid fingerprint must refuse a correction learned on a DIFFERENT grid even
    when the column count matches (8x16 vs 16x8 are both 128) — else the per-column
    coefficients would silently land on the wrong cells (iter 58)."""
    from legoesm.grids.latlon import create_latlon_grid

    from scripts.experiment.check_campaign_deploy import check_deploy
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    cfg = _base_config(tmp_path)                         # base grid is 8 x 16 (128 cols)
    base_cfg, grid, _ = load_base_config_and_grid(cfg)
    wrong_grid = create_latlon_grid(16, 8, dtype=jnp.float64)   # 16 x 8 — also 128 cols
    out = _campaign_output(tmp_path, base_cfg, grid,
                           grid_for_provenance=wrong_grid)      # provenance from the WRONG grid
    with pytest.raises((ValueError, SystemExit)):
        check_deploy(cfg, out)
