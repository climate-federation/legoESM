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
    # Pure-dycore WB modes (neural_gcm / sfno) replace the physics pipeline with
    # a neural net -- their rollout runs NO boundary-layer scheme, so the
    # turbulence field feeds ONLY _create_friction (fric_decay), nothing else
    # (dt/grid/sigma/dycore are turbulence-independent; physics_pipeline is
    # unused for these modes).  Declare turbulence="none" for them so the #931
    # double-count gate KEEPS the Held-Suarez Rayleigh drag ON -- that
    # fric_decay is their SOLE #797 adjoint dissipation of the otherwise
    # undamped dycore (the epoch-0 zero-init rollout is the bare dycore).  A
    # "louis" label would gate it to a no-op and reintroduce the NaN-gradient
    # blow-up (#797 bug 11).  Verified byte-identical to the pre-#931 always-on
    # drag: turbulence="none" reproduces the exact HS Rayleigh fric_decay while
    # leaving dt/grid/sigma unchanged.  Physics mode genuinely runs louis and
    # keeps it (louis owns BOTH the surface stress and the adjoint damping).
    _mode = getattr(cfg, "mode", None)
    _turbulence = ("none" if _mode in ("neural_gcm", "sfno")
                   else yml.get("turbulence", "louis"))
    argv = [
        "--grid-type", "latlon",
        "--discretization", "latlon_cgrid",
        "--resolution", str(int(yml["n_lat"])),
        "--nlev", str(int(yml["nlev"])),
        "--vertical-coord", "sigma",
        "--dt", str(float(yml["dt"])),
        "--radiation", str(yml.get("radiation", "rrtmgp")),
        "--convection", str(yml.get("convection", "sbm")),
        "--turbulence", str(_turbulence),
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


def _spectral_pe_config(yml):
    """SpectralPEConfig for the WB spectral training core, from ``yml["spectral"]``.

    Defaults mirror the AIMIP spectral trainer's proven values
    (``NeuralGCMSpectralConfig``: hyperdiff 2.5e15 order-2, exponential filter
    0.01/8) with ONE deliberate difference: ``semi_implicit`` defaults **True**
    — the whole point of the spectral core is the Hoskins–Simmons SI step whose
    implicit gravity-wave treatment keeps the training ADJOINT bounded (#817
    blocker 1; the explicit lat-lon core's adjoint grows ~x1.3/step).  Set
    ``spectral: {semi_implicit: false}`` to opt back into the explicit
    integrator (then use an explicit-CFL-safe ``spectral.dt``).
    """
    from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig

    spec = dict(yml.get("spectral", {}))
    return SpectralPEConfig(
        hyperdiff_coeff=float(spec.get("hyperdiff_coeff", 2.5e15)),
        hyperdiff_order=int(spec.get("hyperdiff_order", 2)),
        time_integrator=str(spec.get("time_integrator", "ssp_rk3")),
        semi_implicit=bool(spec.get("semi_implicit", True)),
        si_T_ref=float(spec.get("si_T_ref", 300.0)),
        si_alpha=float(spec.get("si_alpha", 0.5)),
        si_substeps=int(spec.get("si_substeps", 1)),
        sponge_sigma=float(spec.get("sponge_sigma", 0.1)),
        sponge_tau=float(spec.get("sponge_tau", 0.0)),
        spectral_filter_order=int(spec.get("spectral_filter_order", 8)),
        spectral_filter_strength=float(spec.get("spectral_filter_strength", 0.01)),
    )


def _build_mode_components_spectral(cfg, yml):
    """Spectral (Gaussian + semi-implicit) training-core components (#817).

    Same 7-tuple contract as the lat-lon path, with the rollout routed through
    the differentiable spectral PE core (``spectral_rollout`` +
    ``_make_spectral_integrator``, #829): the SI step treats the fast
    gravity-wave terms implicitly, so the training adjoint stays bounded where
    the explicit lat-lon core's grows ~x1.3/step (#817 blocker 1) — and the
    Gaussian grid has no pole-cell CFL clamp, so ``dt`` stays at the configured
    value (1800 s default under SI) instead of collapsing to seconds, which is
    what turned a 6 h lead into thousands of BPTT steps (blocker 2's
    amplifier).

    The returned ``make_run_seg(trainable).raw(ic, n_steps, forcing)`` maps a
    Gaussian-grid SegmentCarry through carry->spectral -> SI rollout ->
    spectral->carry, so the trainer's ``combined_loss(pred, target)`` and the
    data-parallel loop are unchanged.  ``model`` is None (the functional
    spectral rollout has no model object; only the WB2 pointer hook received
    it).
    """
    import jax
    import jax.numpy as jnp

    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.spectral_pe import (
        compute_sponge_factor, compute_spectral_filter,
    )
    from legoesm.training.neural_gcm_spectral import (
        carry_to_spectral_state,
        spectral_state_to_carry,
        spectral_rollout,
        make_sfno_spectral_physics,
        make_column_mlp_spectral_physics,
        make_physics_params_spectral_physics,
    )

    spec = dict(yml.get("spectral", {}))
    n_max = int(spec.get("n_max", max(21, int(yml["n_lat"]) // 3)))
    nlev = int(yml["nlev"])
    grid = create_gaussian_grid(n_max)
    sigma = create_sigma_coordinate(nlev)
    pe_config = _spectral_pe_config(yml)
    # No pole-cell clamp on the Gaussian grid; the SI step is stable at large
    # dt (that is its purpose).  Explicit opt-out gets the AIMIP-proven 600 s.
    dt = float(spec.get("dt", 1800.0 if pe_config.semi_implicit else 600.0))

    sponge_factor = None
    if pe_config.sponge_tau > 0:
        sponge_factor = compute_sponge_factor(
            sigma.sigma_full, pe_config.sponge_sigma, pe_config.sponge_tau, dt)
    spectral_filter = None
    if pe_config.spectral_filter_strength > 0:
        spectral_filter = compute_spectral_filter(
            grid.ls, grid.n_max,
            order=pe_config.spectral_filter_order,
            cutoff_fraction=pe_config.spectral_filter_strength)

    loss_config = make_loss_config(cfg, yml)

    # Mode -> (params pytree, physics_fn factory, does the fn take forcing?).
    # The learned fns (column MLP / SFNO) accept the traced ``forcing`` dict
    # (prescribed-SST pathway); the classical physics-params fn does not (its
    # gray-radiation/SBM stack matches train_physics_params_spectral).
    if cfg.mode == "physics":
        from legoesm.training.trainable_params import TrainablePhysicsParams
        params = TrainablePhysicsParams.from_defaults()

        def make_physics_fn(p):
            return make_physics_params_spectral_physics(p, grid, dt)
        uses_forcing = False

    elif cfg.mode == "neural_gcm":
        from legoesm.atmosphere.physics.neural_physics import NeuralPhysics
        ov = yml.get("neural_gcm", {})
        params = NeuralPhysics(
            nlev=nlev, hidden_dim=int(ov.get("nn_hidden", 256)),
            n_layers=int(ov.get("nn_layers", 4)), key=jax.random.PRNGKey(0))

        def make_physics_fn(p):
            return make_column_mlp_spectral_physics(p, grid)
        uses_forcing = True

    elif cfg.mode == "sfno":
        import equinox as eqx
        from legoesm.ml.sfno import SFNO, SFNOConfig
        from legoesm.training.neural_gcm_spectral import N_SFNO_FORCING_CHANNELS

        ov = yml.get("sfno", {})
        n_ch = 4 * nlev + 2
        sfno = SFNO(
            SFNOConfig(
                in_channels=n_ch + N_SFNO_FORCING_CHANNELS, out_channels=n_ch,
                embed_dim=int(ov.get("sfno_embed_dim", 256)),
                n_blocks=int(ov.get("sfno_n_blocks", 8)),
                residual_prediction=False,
            ),
            grid, key=jax.random.PRNGKey(0))
        # Epoch-0 stability contract (mirrors the lat-lon path): zero-init the
        # decoder so the first rollout is the pure (SI) dycore.
        sfno = eqx.tree_at(
            lambda m: (m.decoder.weight, m.decoder.bias), sfno,
            (jnp.zeros_like(sfno.decoder.weight),
             jnp.zeros_like(sfno.decoder.bias)))
        params = sfno

        def make_physics_fn(p):
            return make_sfno_spectral_physics(p, grid)
        uses_forcing = True

    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    class _SpectralRunSeg:
        """`.raw(ic, n_steps, forcing)` contract of build_segment_fn, on the
        spectral core.  ``raw`` = non-donating, non-jit — safe inside
        eqx.filter_value_and_grad (buffer-donation doctrine)."""

        def __init__(self, physics_fn):
            self._physics_fn = physics_fn

        def raw(self, ic_carry, n_steps, forcing):
            state0 = carry_to_spectral_state(ic_carry, grid)
            forcing_base = forcing if uses_forcing else None
            final = spectral_rollout(
                state0, self._physics_fn, grid, sigma, pe_config,
                dt, int(n_steps),
                sponge_factor, spectral_filter,
                forcing_base=forcing_base,
            )
            return spectral_state_to_carry(final, grid, sigma)

    def make_run_seg(trainable):
        return _SpectralRunSeg(make_physics_fn(trainable))

    return None, grid, sigma, params, make_run_seg, loss_config, dt


def build_mode_components(cfg, yml):
    """Return (model, grid, sigma, params, make_run_seg, loss_config, dt) for cfg.mode."""
    import jax

    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    from legoesm.training.training_driver import build_training_segment

    core = getattr(cfg, "training_core", "latlon")
    if core == "spectral":
        return _build_mode_components_spectral(cfg, yml)
    if core != "latlon":
        raise ValueError(
            f"unknown training_core {core!r}; choose 'latlon' (explicit "
            f"C-grid production core) or 'spectral' (Gaussian semi-implicit "
            f"training core, #817)")

    config = build_latlon_config(cfg, yml)
    driver = ModelDriver(config)
    driver.setup()
    grid, sigma, model = driver.grid, driver.sigma, driver.model
    # The driver's setup applies BOTH stability clamps (the factory's
    # pole-cell advective clamp AND the gravity-wave CFL reduction) and
    # stores the final safe value in config.dycore.dt — use it verbatim.
    # Re-deriving it here is how the old _cfl_dt handed the training
    # segment dt=81.8 s (pole-cell CFL 1.47) at the C32 smoke size: it
    # used the meridional spacing only, and the pure-dycore modes
    # (zero-init neural_gcm / sfno) blew up to loss=nan (#797).
    dt = float(driver.config.dycore.dt)
    # The driver's BL Rayleigh-friction profile: the training rollout needs
    # the same dissipation as production — the adjoint through an undamped
    # dycore returns NaN gradients (#797 bug 11; bites the pure-dycore
    # epoch-0 neural_gcm/sfno modes hardest).
    fric_decay = driver.fric_decay
    physics_pipeline = build_physics_pipeline(grid, sigma, config)
    loss_config = make_loss_config(cfg, yml)

    if cfg.mode == "physics":
        from legoesm.training.trainable_params import TrainablePhysicsParams
        params = TrainablePhysicsParams.from_defaults()
        step_unified = physics_pipeline.build_step_unified()

        def make_run_seg(trainable):
            return build_training_segment(
                model, step_unified, grid, sigma, dt, fric_decay=fric_decay,
                **trainable.to_segment_kwargs())

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
            return build_training_segment(
                model, make_neural_step_unified(nn_phys, adapter), grid, sigma,
                dt, fric_decay=fric_decay)

    elif cfg.mode == "sfno":
        import equinox as eqx
        import jax.numpy as jnp

        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.grids.regridding import compute_latlon_to_voronoi_weights
        from legoesm.training.sfno_dycore_coupling import (
            SFNOPhysics, make_sfno_step_unified_latlon,
        )
        from legoesm.ml.sfno import SFNO, SFNOConfig

        ov = yml.get("sfno", {})
        nlev = int(yml["nlev"])
        n_ch = 4 * nlev + 2   # SFNOPhysics packs [u, v, T, q_v, lnps, phis]
        n_max = int(ov.get("gauss_n_max", max(21, int(yml["n_lat"]) // 3)))
        gauss = create_gaussian_grid(n_max)
        sfno = SFNO(
            SFNOConfig(
                in_channels=n_ch, out_channels=n_ch,
                embed_dim=int(ov.get("sfno_embed_dim", 256)),
                n_blocks=int(ov.get("sfno_n_blocks", 8)),
                # tendencies, NOT state residuals: with the default
                # residual_prediction=True the "tendency" would contain the
                # full state and destroy the rollout in one step.
                residual_prediction=False,
            ),
            gauss, key=jax.random.PRNGKey(0))
        # Epoch-0 stability contract (same as the NeuralPhysics zero-init):
        # the untrained SFNO must emit exactly-zero tendencies so the first
        # rollout is the pure dycore.
        sfno = eqx.tree_at(
            lambda m: (m.decoder.weight, m.decoder.bias), sfno,
            (jnp.zeros_like(sfno.decoder.weight),
             jnp.zeros_like(sfno.decoder.bias)))
        params = SFNOPhysics(sfno=sfno, grid=gauss, nlev=nlev)

        def _flat_points(lat_1d, lon_1d):
            lon2d, lat2d = np.meshgrid(np.asarray(lon_1d), np.asarray(lat_1d))
            return lat2d.ravel(), lon2d.ravel()

        g_lat_f, g_lon_f = _flat_points(gauss.lat, gauss.lon)
        ll_lat_f, ll_lon_f = _flat_points(grid.lat, grid.lon)
        w_ll2g = compute_latlon_to_voronoi_weights(
            np.asarray(grid.lat), np.asarray(grid.lon), g_lat_f, g_lon_f,
        )._replace(target_shape=(int(gauss.n_lat), int(gauss.n_lon)))
        w_g2ll = compute_latlon_to_voronoi_weights(
            np.asarray(gauss.lat), np.asarray(gauss.lon), ll_lat_f, ll_lon_f,
        )._replace(target_shape=(int(grid.n_lat), int(grid.n_lon)))

        def make_run_seg(sfno_ph):
            step = make_sfno_step_unified_latlon(sfno_ph, w_ll2g, w_g2ll)
            return build_training_segment(model, step, grid, sigma, dt,
                                           fric_decay=fric_decay)

    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    return model, grid, sigma, params, make_run_seg, loss_config, dt


def rollout_hours(cfg, yml):
    hrs = cfg.multi_step_hours or tuple(yml.get("loss", {}).get("multi_step_hours", ()) or ())
    return float(hrs[0]) if hrs else 6.0     # first lead = the base rollout horizon


def _era5_time_to_forcing_calendar(time_ns, year):
    """(day_of_year 1-based INTEGER, seconds_of_day) for spectral_rollout.

    Calendar convention of ``spectral_rollout._forcing_at`` (codex #817
    adversarial finding): the rollout advances
    ``doy_eff = day_of_year + (t + seconds_of_day)/86400``, so the base
    ``day_of_year`` must be the integer 1-based day and the intra-day fraction
    must ride ``seconds_of_day`` alone — a 0-based/fractional doy is wrapped a
    full year off (Jan 1 00Z -> day 365), and a fractional doy would
    double-count the hours (which also rules out ``day_to_calendar``'s
    fractional doy here).
    """
    elapsed_s = float((time_ns - np.datetime64(f"{year}-01-01"))
                      / np.timedelta64(1, "s"))
    doy_1based = float(np.floor(elapsed_s / 86400.0)) + 1.0
    sod = float(elapsed_s % 86400.0)
    return doy_1based, sod


def _load_era5_samples_spectral(cfg, yml, grid, sigma):
    """(ic, target, forcing) samples on the Gaussian grid for the spectral core.

    ``ic``/``target`` are SegmentCarry on the Gaussian grid
    (``era5_to_spectral_carry``); ``forcing`` is the traced dict the spectral
    rollout's prescribed-SST pathway consumes (``forcing_base``: flat
    ``T_sfc``/``sic`` (ncol,) + calendar scalars — SegmentForcing doctrine, all
    traced so per-sample values never retrace).  Consumed only inside the
    spectral ``make_run_seg(...).raw`` (opaque to the trainer loop).
    """
    import jax.numpy as jnp

    from legoesm.training.era5_to_state import (
        TrainingERA5Config, open_era5_zarr, load_era5_slice,
        era5_to_spectral_carry, regrid_2d_to_gaussian,
    )

    era5_cfg = TrainingERA5Config(dt_hours=int(yml.get("era5_cadence_hours", 6)))._replace(
        zarr_store=yml["era5_zarr"])
    ds = open_era5_zarr(era5_cfg.zarr_store)
    times = np.asarray(ds.time.values, dtype="datetime64[ns]")
    snaps_per_day = 24 // era5_cfg.dt_hours
    roll_h = rollout_hours(cfg, yml)
    stride = int(roll_h) // era5_cfg.dt_hours

    n_days = 3 if not cfg.smoke else 1
    samples = []
    for year in yml["train_years"]:
        base = int(np.searchsorted(times, np.datetime64(f"{year}-01-01")))
        for d in range(n_days * snaps_per_day):
            i_ic, i_tg = base + d, base + d + stride
            if i_tg >= len(times):
                break
            ic = era5_to_spectral_carry(load_era5_slice(era5_cfg, i_ic), grid, sigma)
            target = era5_to_spectral_carry(load_era5_slice(era5_cfg, i_tg), grid, sigma)
            sst_src = load_era5_slice(era5_cfg, i_ic)
            sst = jnp.asarray(regrid_2d_to_gaussian(
                sst_src.sst, sst_src.lat, sst_src.lon, grid)).reshape(-1)
            doy_1based, sod = _era5_time_to_forcing_calendar(times[i_ic], year)
            forcing = {
                "T_sfc": sst,
                "sic": jnp.zeros_like(sst),
                "day_of_year": jnp.asarray(doy_1based),
                "seconds_of_day": jnp.asarray(sod),
            }
            samples.append((ic, target, forcing))
        if cfg.smoke:
            break
    return samples


def load_era5_samples(cfg, yml, grid, sigma):
    """(ic, target, forcing) samples over the train windows, on the lat-lon grid."""
    import jax.numpy as jnp

    from legoesm.training.era5_to_state import (
        TrainingERA5Config, open_era5_zarr, load_era5_slice,
        era5_to_latlon_carry, regrid_2d_to_gaussian,
    )
    from legoesm.driver.compiled_segments import pack_forcing

    if getattr(cfg, "training_core", "latlon") == "spectral":
        return _load_era5_samples_spectral(cfg, yml, grid, sigma)

    era5_cfg = TrainingERA5Config(dt_hours=int(yml.get("era5_cadence_hours", 6)))._replace(
        zarr_store=yml["era5_zarr"])
    ds = open_era5_zarr(era5_cfg.zarr_store)
    times = np.asarray(ds.time.values, dtype="datetime64[ns]")
    snaps_per_day = 24 // era5_cfg.dt_hours
    roll_h = rollout_hours(cfg, yml)
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
