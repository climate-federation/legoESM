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


def test_days_climatology_window_is_exposed_and_defaults_to_a_real_window(tmp_path):
    """--days (iter 311) exposes the run length = the CLIMATOLOGY WINDOW the model time-mean
    is computed over + compared to the matched ERA5 mean, completing the production-scaling
    knobs (resolution/nlev/dt/days). The default (200) is a REAL window (≈40 samples at the
    5-day cadence), not a toy run; a custom value threads through the CLI to the config."""
    from legoesm.driver.config import experiment_config_from_dict

    from scripts.experiment.write_amip_clubb_lite_config import (
        build_amip_clubb_lite_config,
        main,
    )

    assert build_amip_clubb_lite_config().days == 200            # a real default climatology
    out = tmp_path / "long.json"
    assert main([str(out), "--resolution", "8", "--nlev", "5", "--days", "3650"]) == 0
    cfg = experiment_config_from_dict(json.loads(out.read_text()))
    assert cfg.days == 3650          # a 10-year window threads through CLI -> JSON -> config


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


def test_generated_config_drives_cmip_coupled_driver(tmp_path):
    """The done-criterion's SECOND mode: the SAME generated config drives CMIP too.
    Build a REAL CoupledESMDriver from the generator's output via the campaign's
    make_base_driver_builder('cmip', preset), and confirm the fields cmip_column_state
    reads (state / q_v / ocean SST on the atm column shape) are present. So the one
    turnkey config runs AMIP (test above) AND CMIP."""
    from legoesm.driver.config import experiment_config_from_dict
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    from scripts.experiment.write_amip_clubb_lite_config import main
    from scripts.run.run_correction_campaign import make_base_driver_builder

    out = tmp_path / "cfg.json"
    assert main([str(out), "--resolution", "8", "--nlev", "5"]) == 0
    cfg = experiment_config_from_dict(json.loads(out.read_text()))
    build_cmip, _extract = make_base_driver_builder(
        "cmip", coupled_preset=PRESETS["aquaplanet"](), ocean_grid=None)
    driver = build_cmip(cfg)                              # constructs + setup()s the coupled driver
    assert isinstance(driver, CoupledESMDriver)
    assert driver.state is not None and driver.q_v is not None
    assert tuple(driver.ocean_state.T_sfc.data.shape) == (8, 16)  # same-grid: SST on atm shape


def test_starter_resolution_prints_a_scaling_note(tmp_path, capsys):
    """A COARSE starter resolution prints a NOTE reminding the operator to scale up for a
    PRODUCTION ERA5 comparison (so a toy 8x16 campaign is not run by mistake) + to lower
    --dt with resolution (CFL); a production-scale resolution does NOT nag (iter 307)."""
    from scripts.experiment.write_amip_clubb_lite_config import main

    assert main([str(tmp_path / "starter.json"), "--resolution", "8", "--nlev", "10"]) == 0
    note = capsys.readouterr().out
    assert "STARTER" in note and "PRODUCTION" in note and "--dt" in note   # scaling + CFL
    # A production-scale resolution emits NO starter nag (the threshold is meaningful).
    assert main([str(tmp_path / "prod.json"), "--resolution", "96", "--nlev", "30"]) == 0
    assert "STARTER" not in capsys.readouterr().out


def test_non_clubb_config_is_rejected_by_launcher():
    """Non-vacuity: a NON-clubb config is exactly what make_clubb_build_driver rejects,
    so the generator's clubb_lite guarantee is load-bearing (not a no-op)."""
    from types import SimpleNamespace

    from scripts.run.run_correction_campaign import make_clubb_build_driver

    with pytest.raises(ValueError, match="clubb_lite"):
        make_clubb_build_driver(SimpleNamespace(turbulence="louis"), lambda c: object())


def test_bad_radiation_fails_fast_at_generation_not_at_load(tmp_path):
    """A typo'd --radiation must fail LOUD at GENERATION (via the canonical
    ExperimentConfig.validate_strict), NOT write a broken config that only surfaces
    later when the campaign loads it. The realistic HPC radiation (rrtmgp) passes."""
    from scripts.experiment.write_amip_clubb_lite_config import (
        build_amip_clubb_lite_config,
        main,
    )

    with pytest.raises(ValueError, match="radiation must be one of"):
        build_amip_clubb_lite_config(radiation="rrtmpg")          # typo of rrtmgp
    assert build_amip_clubb_lite_config(radiation="rrtmgp").radiation == "rrtmgp"  # HPC config OK
    # Through main(): the raise fires BEFORE the file is written — no broken config on disk.
    out = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="radiation must be one of"):
        main([str(out), "--radiation", "rrtmpg"])
    assert not out.exists()
