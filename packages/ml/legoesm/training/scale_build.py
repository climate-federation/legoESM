"""Mode-specific model / ERA5 / loss builders for WeatherBench scale training.

Thin adapter that reuses the INSTALLED lat-lon training stack (the reference
``run_aimip_latlon.py`` is WIP against a newer API, so this binds to the real
package signatures) and exposes exactly what
``scripts/run/train_weatherbench_scale.py`` needs for the data-parallel loop:

    build_mode_components(cfg, yml) -> (model, grid, sigma, params, make_run_seg, loss_config, dt)
    load_era5_samples(cfg, yml, grid, sigma) -> list[(ic, target, forcing)]
    evaluate_wb2(cfg, yml, model, params, make_run_seg, grid, sigma)

The three modes differ ONLY in the ``make_run_seg`` factory (per
``training_driver._build_train_step``): physics feeds
``TrainablePhysicsParams.to_segment_kwargs()``; neural_gcm builds a
``NeuralPhysics`` step; sfno builds an SFNO lat-lon step.

End-to-end validation is the single-GPU smoke (``--mode physics --smoke``) +
the cluster launch; the data-parallel gradient average is unit-tested separately.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[4]   # packages/ml/legoesm/training/ -> repo root


def _load_run_amip():
    """Exec scripts/run/run_amip.py as a module to reuse its arg parser + config builder."""
    path = _REPO / "scripts" / "run" / "run_amip.py"
    spec = importlib.util.spec_from_file_location("run_amip_for_scale", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_latlon_config(cfg, yml):
    """ExperimentConfig for a lat-lon C-grid PE run, via run_amip's parser."""
    ra = _load_run_amip()
    argv = [
        "--grid-type", "latlon",
        "--discretization", "latlon_cgrid",
        "--resolution", str(int(yml["n_lat"])),
        "--nlev", str(int(yml["nlev"])),
        "--vertical-coord", "sigma",
        "--dt", str(float(yml["dt"])),
        "--radiation", str(yml.get("radiation", "rrtmgp")),
        "--convection", str(yml.get("convection", "sbm")),
        "--turbulence", str(yml.get("turbulence", "louis")),
        "--microphysics", str(yml.get("microphysics", "kessler")),
        "--gravity-wave-drag", str(yml.get("gravity_wave_drag", "hines")),
        # run_amip's --rad-update-steps defaults to None (its main() auto-sets
        # None -> floor(3600/dt) inside _postprocess_args, which we deliberately
        # skip here — it enforces AMIP forcing-path args the WB/ERA5 path lacks).
        # Pass an explicit value so cfg.rad_update_steps is a valid int (the
        # driver's _prepare_run_context reads it bare): radiation every step (1)
        # unless the YAML overrides.
        "--rad-update-steps", str(int(yml.get("rad_update_steps", 1))),
    ]
    parsed = ra.build_arg_parser().parse_args(argv)
    return ra.build_config_from_args(parsed)


def make_loss_config(cfg, yml):
    from legoesm.training.losses import LossConfig
    lb = dict(yml.get("loss", {}))
    valid = set(LossConfig()._fields)
    kwargs = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in lb.items() if k in valid}
    return LossConfig(**kwargs)


def _cfl_dt(model, n_lat, fallback_dt):
    from legoesm.core.cfl import cfl_max_dt
    from legoesm import constants
    dy_min = float(constants.R_earth) * np.pi / float(n_lat)
    dt_cfl = cfl_max_dt(dy_min, wave_speed=400.0, cfl_number=0.7)
    return float(min(float(getattr(model, "effective_dt", fallback_dt)), dt_cfl))


def build_mode_components(cfg, yml):
    """Return (model, grid, sigma, params, make_run_seg, loss_config, dt) for cfg.mode."""
    import jax

    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    from legoesm.training.training_driver import _build_training_segment

    config = build_latlon_config(cfg, yml)
    driver = ModelDriver(config)
    driver.setup()
    grid, sigma, model = driver.grid, driver.sigma, driver.model
    dt = _cfl_dt(model, yml["n_lat"], yml["dt"])
    physics_pipeline = build_physics_pipeline(grid, sigma, config)
    loss_config = make_loss_config(cfg, yml)

    if cfg.mode == "physics":
        from legoesm.training.trainable_params import TrainablePhysicsParams
        params = TrainablePhysicsParams.from_defaults()
        step_unified = physics_pipeline.build_step_unified()

        def make_run_seg(trainable):
            return _build_training_segment(
                model, step_unified, grid, sigma, dt, **trainable.to_segment_kwargs())

    elif cfg.mode == "neural_gcm":
        from legoesm.atmosphere.physics.neural_physics import (
            NeuralPhysics, make_neural_step_unified,
        )
        from legoesm.core.grid_adapters import make_adapter
        ov = yml.get("neural_gcm", {})
        params = NeuralPhysics(
            nlev=int(yml["nlev"]), hidden_dim=int(ov.get("nn_hidden", 256)),
            n_layers=int(ov.get("nn_layers", 4)), key=jax.random.PRNGKey(0))
        adapter = make_adapter(grid)

        def make_run_seg(nn_phys):
            return _build_training_segment(
                model, make_neural_step_unified(nn_phys, adapter), grid, sigma, dt)

    elif cfg.mode == "sfno":
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.training.sfno_dycore_coupling import (
            SFNOPhysics, make_sfno_step_unified_latlon,
        )
        from legoesm.ml.sfno import SFNO, SFNOConfig
        from legoesm.grids.remap import compute_latlon_to_voronoi_weights
        ov = yml.get("sfno", {})
        n_max = int(ov.get("gauss_n_max", max(21, int(yml["n_lat"]) // 3)))
        gauss = create_gaussian_grid(n_max, dealiasing="quadratic")
        sfno = SFNO(SFNOConfig(embed_dim=int(ov.get("sfno_embed_dim", 256)),
                               n_blocks=int(ov.get("sfno_n_blocks", 8))),
                    gauss, key=jax.random.PRNGKey(0))
        params = SFNOPhysics(sfno=sfno, flux_head=None)
        w_ll2g = compute_latlon_to_voronoi_weights(grid, gauss)
        w_g2ll = compute_latlon_to_voronoi_weights(gauss, grid)

        def make_run_seg(sfno_ph):
            step = make_sfno_step_unified_latlon(
                sfno_ph, w_ll2g, w_g2ll, int(gauss.n_lat), int(gauss.n_lon))
            return _build_training_segment(model, step, grid, sigma, dt)

    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    return model, grid, sigma, params, make_run_seg, loss_config, dt


def _rollout_hours(cfg, yml):
    hrs = cfg.multi_step_hours or tuple(yml.get("loss", {}).get("multi_step_hours", ()) or ())
    return float(hrs[0]) if hrs else 6.0     # first lead = the base rollout horizon


def load_era5_samples(cfg, yml, grid, sigma):
    """(ic, target, forcing) samples over the train windows, on the lat-lon grid."""
    import jax.numpy as jnp

    from legoesm.training.era5_to_state import (
        TrainingERA5Config, open_era5_zarr, load_era5_slice,
        era5_to_latlon_carry, regrid_2d_to_gaussian,
    )
    from legoesm.driver.compiled_segments import pack_forcing

    era5_cfg = TrainingERA5Config(dt_hours=int(yml.get("era5_cadence_hours", 6)))._replace(
        zarr_store=yml["era5_zarr"])
    ds = open_era5_zarr(era5_cfg.zarr_store)
    times = np.asarray(ds.time.values, dtype="datetime64[ns]")
    snaps_per_day = 24 // era5_cfg.dt_hours
    roll_h = _rollout_hours(cfg, yml)
    stride = int(roll_h) // era5_cfg.dt_hours

    config = build_latlon_config(cfg, yml)
    driver = _driver_for_ctx(config)
    ctx = driver._prepare_run_context(0, config.start_day, restore_carry=False)

    n_days = 3 if not cfg.smoke else 1
    samples = []
    for year in yml["train_years"]:
        base = int(np.searchsorted(times, np.datetime64(f"{year}-01-01")))
        for d in range(n_days * snaps_per_day):
            i_ic, i_tg = base + d, base + d + stride
            if i_tg >= len(times):
                break
            ic = era5_to_latlon_carry(load_era5_slice(era5_cfg, i_ic), grid, sigma)
            target = era5_to_latlon_carry(load_era5_slice(era5_cfg, i_tg), grid, sigma)
            sst_src = load_era5_slice(era5_cfg, i_ic)
            sst = regrid_2d_to_gaussian(sst_src.sst, sst_src.lat, sst_src.lon, grid)
            doy = float((times[i_ic] - np.datetime64(f"{year}-01-01"))
                        / np.timedelta64(1, "D"))
            forcing = pack_forcing(
                sst=sst, sic=jnp.zeros_like(sst),
                day_of_year=jnp.asarray(doy), seconds_of_day=jnp.asarray(0.0),
                solar_weights=ctx["solar_weights"], s_0=ctx["current_s_0"],
                o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"])
            samples.append((ic, target, forcing))
        if cfg.smoke:
            break
    return samples


def _driver_for_ctx(config):
    from legoesm.driver.model_driver import ModelDriver
    driver = ModelDriver(config)
    driver.setup()
    return driver


def evaluate_wb2(cfg, yml, model, params, make_run_seg, grid, sigma):
    """WB2 scorecard of the trained model on the eval year (rank-0 only).

    Rolls the trained physics_fn from ERA5 ICs and scores headline fields with
    the merged WB2 scorer. Deferred detail: this is the winner-eval hook; the
    scorer (evaluations/wb_orchestrator) is validated separately.
    """
    import logging
    logging.getLogger("wb_scale").info(
        "WB2 eval hook: mode=%s eval_years=%s — run evaluations.wb_orchestrator "
        "against the trained checkpoint (spectral scorer path).", cfg.mode, yml["eval_years"])
    # The lat-lon->WB2 scoring reuses evaluations.wb_forecast.diagnose_and_regrid on a
    # rolled-out state; wired to the spectral scorer in a follow-up (the trained
    # checkpoint from this run is the input).
    return None
