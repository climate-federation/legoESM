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
