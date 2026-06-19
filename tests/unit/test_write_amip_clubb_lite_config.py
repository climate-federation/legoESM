"""Unit test for ``scripts/experiment/write_amip_clubb_lite_config.py`` — the starter
AMIP ``clubb_lite`` base-config generator the correction-campaign launcher consumes.
"""

from __future__ import annotations

import json

import pytest


def test_build_amip_clubb_lite_config_is_clubb_and_parameterized():
    from scripts.experiment.write_amip_clubb_lite_config import (
        build_amip_clubb_lite_config,
    )

    cfg = build_amip_clubb_lite_config(resolution=16, nlev=30, dt=300.0)
    assert cfg.turbulence == "clubb_lite"               # the campaign's hard requirement
    assert cfg.grid.grid_type == "latlon"
    assert cfg.grid.resolution == 16 and cfg.grid.nlev == 30
    assert cfg.dycore.dt == 300.0


def test_config_round_trips_and_launcher_accepts_it(tmp_path):
    """The emitted JSON loads back into an ExperimentConfig AND make_clubb_build_driver
    accepts it — the exact contract the campaign ``--config`` path requires."""
    from legoesm.driver.config import experiment_config_from_dict

    from scripts.experiment.write_amip_clubb_lite_config import main
    from scripts.run.run_correction_campaign import make_clubb_build_driver

    out = tmp_path / "amip_clubb_lite.json"
    assert main([str(out), "--resolution", "8", "--nlev", "10"]) == 0
    cfg = experiment_config_from_dict(json.loads(out.read_text()))
    assert cfg.turbulence == "clubb_lite" and cfg.grid.resolution == 8
    make_clubb_build_driver(cfg, lambda c: object())    # must NOT raise (clubb_lite accepted)


def test_generated_config_loads_through_campaign_loader(tmp_path):
    """End-to-end: the emitted config is loadable by the campaign's OWN loader
    load_base_config_and_grid (config → ModelDriver → grid + sigma), not merely
    round-trippable through experiment_config_from_dict — so it is genuinely runnable.
    A latlon resolution-8, 10-level config builds an (8, 16) grid + a 10-level sigma."""
    from scripts.experiment.write_amip_clubb_lite_config import main
    from scripts.run.run_correction_campaign import load_base_config_and_grid

    out = tmp_path / "cfg.json"
    assert main([str(out), "--resolution", "8", "--nlev", "10"]) == 0
    base_cfg, grid, sigma = load_base_config_and_grid(str(out))
    assert base_cfg.turbulence == "clubb_lite"
    assert tuple(grid.grid_shape_2d) == (8, 16)        # latlon res 8 → 8 lat × 16 lon
    assert len(sigma.sigma_full) == 10                  # nlev levels


def test_non_clubb_config_is_rejected_by_launcher():
    """Non-vacuity: a NON-clubb config is exactly what make_clubb_build_driver rejects,
    so the generator's clubb_lite guarantee is load-bearing (not a no-op)."""
    from types import SimpleNamespace

    from scripts.run.run_correction_campaign import make_clubb_build_driver

    with pytest.raises(ValueError, match="clubb_lite"):
        make_clubb_build_driver(SimpleNamespace(turbulence="louis"), lambda c: object())
