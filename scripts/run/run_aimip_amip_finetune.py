#!/usr/bin/env python
"""Stability fine-tune of the AIMIP classical model for free-running AMIP.

The ace2loss classical model was trained on 6-12 h forecasts with ``T_sfc=T_air``
(surface fluxes ~0), so run FREE with prescribed SST it drifts to a ~500 K
radiative-runaway equilibrium. This driver fine-tunes the ~58 scheme knobs + 65
spatial-surface coefficients so the model holds a balanced climate under the
EXACT AIMIP forcing:

* the rollout uses the protocol prescribed SST (``training.aimip_amip_forcing``,
  Zenodo 17065758) so surface fluxes are ACTIVE during training (unlike the
  original training);
* rollouts are multi-day, so the radiative drift manifests and is penalised by
  standard ERA5 state matching at the lead (``spectral_state_vs_carry_loss``);
* a modest global-mean near-surface-T drift term directly discourages the runaway
  (energy-balance proxy; ``--w-drift``).

Gradients flow to the physics params via ``make_aimip_classical_spectral_physics``
(built inside the loss so leaves are traced) and the checkpointed
``spectral_amip_rollout(use_checkpoint=True)``. Writes the fine-tuned params for
the AMIP inference driver to consume.

Usage (proof-of-concept)::

    JAX_ENABLE_X64=1 python scripts/run/run_aimip_amip_finetune.py \\
        --init-ckpt results/aimip_ace2loss_assembled/classical/params.eqx \\
        --out-ckpt results/aimip_amip_finetune/classical_stable.eqx \\
        --epochs 8 --rollout-days 3 --n-samples 6 --lr 1e-4 --w-drift 1.0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("aimip-finetune")
_SPD = 86400.0


def _load_yaml(p):
    with Path(p).open() as fh:
        return yaml.safe_load(fh) or {}


def _merged_cfg(suite_path: Path, variant: str = "classical") -> dict:
    suite = _load_yaml(suite_path)
    base = _load_yaml(Path(suite["base"]))
    base.update(suite.get("cfg_overrides", {}) or {})
    base.update(_load_yaml(suite_path.parent / f"variant_{variant}.yaml"))
    base.setdefault("nlev", base["n_levels"])
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-ckpt", type=Path,
                    default=Path("results/aimip_ace2loss_assembled/classical/params.eqx"))
    ap.add_argument("--out-ckpt", type=Path,
                    default=Path("results/aimip_amip_finetune/classical_stable.eqx"))
    ap.add_argument("--variant", type=str, default="classical",
                    choices=["classical", "column_nn", "sfno_physics"],
                    help="What to fine-tune: the classical scheme knobs, or "
                         "the NN weights (column MLP / SFNO physics). NN "
                         "variants free-run to a radiative runaway exactly "
                         "like the pre-finetune classical (sfno r0 sanity: "
                         "435 K year-1) — same multi-day prescribed-SST "
                         "rollout + drift penalty, via "
                         "spectral_rollout(forcing_base=...).")
    ap.add_argument("--suite", type=Path, default=Path("config/aimip/ace2/suite.yaml"))
    ap.add_argument("--forcing-path", type=str, default=None)
    ap.add_argument("--cache-path", type=str,
                    default="results/aimip_forcing/gaussian_forcing.npz")
    ap.add_argument("--curriculum", type=str, default="1:4,3:6,5:8",
                    help="rollout_days:epochs phases, short->long (NeuralGCM-style).")
    def _ema_decay(v):
        f = float(v)
        if not (0.0 <= f < 1.0):
            # d=1.0 would zero the bias-correction denominator (1 - d^t).
            raise argparse.ArgumentTypeError("--ema-decay must be in [0, 1)")
        return f

    ap.add_argument("--ema-decay", type=_ema_decay, default=0.999,
                    help="EMA of the params (ACE2 uses 0.999), in [0,1); "
                         "bias-corrected at save. EMA weights saved.")
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--first-year", type=int, default=1979)
    ap.add_argument("--last-year", type=int, default=2014)  # training period only
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--w-drift", type=float, default=3.0)
    ap.add_argument("--convection-scheme", type=str, default=None,
                    help="Override aimip_convection (attribution 2026-07-02: "
                         "untrained-edmf drives a +3.3 K/day column heating "
                         "runaway; trained tiedtke is near-balanced AND has "
                         "trainable leaves).")
    ap.add_argument("--rad-update-interval", type=int, default=36)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        compute_spectral_filter, compute_sponge_factor, spectral_pe_to_grid,
    )
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.ml.training import load_checkpoint, save_checkpoint
    from legoesm.forcing.surface_utils import blend_surface_temperature
    from legoesm import constants
    from legoesm.training.aimip_amip_forcing import (
        DEFAULT_AIMIP_FORCING, ghg_vmr_at_year, interp_forcing_at,
        open_arco_era5, ozone_vmr_at_date, regrid_monthly_forcing_to_gaussian,
    )
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams, make_aimip_classical_spectral_physics,
    )
    from legoesm.training.campaign_driver import parse_bool_flag
    from legoesm.training.aimip_spatial import grid_with_zm_land_fraction, land_mask_from_phis
    from legoesm.training.era5_to_state import (
        TrainingERA5Config, era5_to_spectral_carry, load_era5_ic,
    )
    from legoesm.training.neural_gcm_spectral import (
        carry_to_spectral_state, spectral_amip_rollout, spectral_rollout,
        spectral_state_vs_carry_loss,
    )

    cfg = _merged_cfg(args.suite, args.variant)

    # Radiation pin, EVERY variant — an AIMIP run uses rrtmgp. Nested under the
    # classical branch it left the NN paths unchecked (my own AST test caught
    # this after codex flagged the same shape in run_aimip and run_aimip_latlon).
    from legoesm.training.campaign_driver import validate_campaign_radiation
    validate_campaign_radiation(
        str(cfg.get("aimip_radiation", "rrtmgp")),
        campaign="aimip",
        smoke=bool(cfg.get("smoke", False)),
    )
    is_nn = args.variant != "classical"
    if args.convection_scheme:
        if is_nn:
            raise SystemExit("--convection-scheme is classical-only.")
        # Single override point: make_aimip_* (loss + warm-up) AND the sizing
        # PhysicsState all read cfg["aimip_convection"].
        cfg["aimip_convection"] = args.convection_scheme
        logger.info(f"convection override: {cfg['aimip_convection']}")
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from legoesm.training.losses import LossConfig
    from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig
    spec_cfg = NeuralGCMSpectralConfig(
        n_max=int(cfg["n_max"]), n_levels=int(cfg["nlev"]), dt=float(cfg["dt"]),
        pe_config=SpectralPEConfig(
            hyperdiff_coeff=2.5e15, hyperdiff_order=2, time_integrator="ssp_rk3",
            spectral_filter_strength=0.01, spectral_filter_order=8,
            # Forwarded so a suite's dry-mass anchor is not silently
            # ignored on the prescribed-SST lane; default False keeps
            # every existing AMIP run byte-identical.
            fix_mass=bool(cfg.get("fix_mass", False)),
            anchor_mass_to_initial=bool(
                cfg.get("anchor_mass_to_initial", False)),
        ),
        n_epochs=1, n_train_days=1, start_year=args.first_year,
        loss_config=LossConfig(),
    )
    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top)
    sigma_full = jnp.asarray(sigma.sigma_full)
    lat_w = jnp.asarray(grid.weights)
    ncol = len(grid.lat) * len(grid.lon)
    nlev = spec_cfg.n_levels
    dt = spec_cfg.dt
    pe = spec_cfg.pe_config  # n_steps is per-curriculum-phase (n_steps_phase)

    times_ns, sst_m, sic_m, land = regrid_monthly_forcing_to_gaussian(
        args.forcing_path or DEFAULT_AIMIP_FORCING, grid, cache_path=args.cache_path,
    )
    ocean = land < 0.5
    T_ice = float(constants.T_freeze_ocean)
    wb2 = TrainingERA5Config().zarr_store
    # Historical ERA5 ozone (ARCO): only the classical RRTMGP consumes it.
    ds_o3 = None if is_nn else open_arco_era5()

    def _override_at(date: _dt.date):
        ns = np.datetime64(date).astype("datetime64[ns]").astype(np.int64)
        sst = interp_forcing_at(times_ns, sst_m, ns)
        sic = np.clip(interp_forcing_at(times_ns, sic_m, ns), 0.0, 1.0)
        tsfc = np.asarray(blend_surface_temperature(sst, sic, T_ice))
        return (np.where(ocean, tsfc, np.nan).astype(np.float64),
                np.where(ocean, sic, 0.0).astype(np.float64))

    # --- sample builder (rollout length varies across the curriculum) ---
    def _build_samples(rollout_days):
        years = np.linspace(args.first_year, args.last_year, args.n_samples).astype(int)
        out, phis = [], None
        for k, yr in enumerate(years):
            d0 = _dt.date(int(yr), 1 + (k % 12), 15)  # spread across seasons
            d1 = d0 + _dt.timedelta(days=rollout_days)
            ic_carry = era5_to_spectral_carry(load_era5_ic(wb2, d0.year, d0.month, d0.day), grid, sigma)
            tgt_carry = era5_to_spectral_carry(load_era5_ic(wb2, d1.year, d1.month, d1.day), grid, sigma)
            mid = d0 + _dt.timedelta(days=rollout_days // 2)
            t_sfc_col, sic_col = _override_at(mid)
            doy = jnp.asarray(float(d0.timetuple().tm_yday))
            ic_state = carry_to_spectral_state(ic_carry, grid)
            if is_nn:
                # NN forcing dict fields (spectral_rollout advances the
                # calendar per step from doy/seconds inside the scan).
                out.append((
                    ic_state, tgt_carry, jnp.asarray(t_sfc_col),
                    jnp.asarray(sic_col), doy,
                ))
            else:
                # Transient historical GHG for this sample's year (traced
                # scalars -> no retrace across samples/years). Physical
                # RRTMGP needs it.
                ghg = {k: jnp.asarray(float(v))
                       for k, v in ghg_vmr_at_year(d0.year).items()}
                # Transient historical ozone at this sample's date
                # (ncol, nlev), interpolated to sigma with the sample's own
                # surface pressure.
                o3 = jnp.asarray(ozone_vmr_at_date(
                    ds_o3, d0, grid, sigma_full, np.asarray(ic_carry.p_s),
                ))
                out.append((
                    ic_state, tgt_carry, jnp.asarray(t_sfc_col), doy, ghg, o3,
                ))
            if phis is None:
                phis = jnp.asarray(ic_carry.phis)
        return out, phis

    sponge = (compute_sponge_factor(sigma.sigma_full, pe.sponge_sigma, pe.sponge_tau, dt)
              if pe.sponge_tau > 0 else None)
    sfilt = (compute_spectral_filter(grid.ls, grid.n_max, order=pe.spectral_filter_order,
                                     cutoff_fraction=pe.spectral_filter_strength)
             if pe.spectral_filter_strength > 0 else None)

    land_mask = None
    sizing_ps = None
    if not is_nn:
        _, phis0 = _build_samples(1)  # phis is rollout-agnostic -> land mask once
        land_mask = land_mask_from_phis(phis0, smooth=True)
        grid = grid_with_zm_land_fraction(grid, str(cfg["aimip_convection"]))
        sizing_ps = init_physics_state(
            ncol, nlev,
            PhysicsConfig(
                radiation=RadiationConfig(scheme="none"),
                convection=ConvectionConfig(scheme=str(cfg["aimip_convection"])),
                turbulence=TurbulenceConfig(scheme=str(cfg["aimip_turbulence"])),
                microphysics=MicrophysicsConfig(scheme=str(cfg["aimip_microphysics"])),
                gravity_wave_drag=GravityWaveDragConfig(scheme=str(cfg["aimip_gwd"])),
            ),
        )

    # --- what gets fine-tuned: classical scheme knobs OR the NN weights ---
    if args.variant == "column_nn":
        from legoesm.atmosphere.physics.neural_physics import (
            build_column_physics,
        )
        from legoesm.training.neural_gcm_spectral import (
            make_column_mlp_spectral_physics,
        )
        params0 = load_checkpoint(
            build_column_physics(
                nlev=nlev,
                hidden_dim=int(cfg.get("nn_hidden_dim", 256)),
                n_layers=int(cfg.get("nn_n_layers", 4)),
                key=jax.random.PRNGKey(int(cfg.get("nn_seed", 0))),
            ),
            args.init_ckpt,
        )
        _make_nn_physics = make_column_mlp_spectral_physics
    elif args.variant == "sfno_physics":
        from legoesm.ml.channel_packing import PE3DChannelSpec
        from legoesm.ml.sfno import SFNO, SFNOConfig
        from legoesm.training.neural_gcm_spectral import (
            N_SFNO_FORCING_CHANNELS, make_sfno_spectral_physics,
        )
        _spec = PE3DChannelSpec(nlev=nlev)
        params0 = load_checkpoint(
            SFNO(
                SFNOConfig(
                    in_channels=_spec.n_channels + N_SFNO_FORCING_CHANNELS,
                    out_channels=_spec.n_channels,
                    embed_dim=int(cfg.get("sfno_embed_dim", 128)),
                    n_blocks=int(cfg.get("sfno_n_blocks", 4)),
                    mlp_expansion=int(cfg.get("sfno_mlp_expansion", 4)),
                    residual_prediction=False,
                ),
                grid, key=jax.random.PRNGKey(int(cfg.get("sfno_seed", 0))),
            ),
            args.init_ckpt,
        )
        _make_nn_physics = make_sfno_spectral_physics
    else:
        params0 = load_checkpoint(
            AIMIPClassicalParams.from_defaults(spatial_surface=True),
            args.init_ckpt,
        )
        _make_nn_physics = None

    def _gm_surf_T_state(s):
        T = spectral_pe_to_grid(s, grid, sigma)["T"][..., -1]  # (n_lat, n_lon)
        return jnp.sum(jnp.mean(T, axis=-1) * lat_w) / jnp.sum(lat_w)

    def _gm_surf_T_carry(c):
        T = jnp.asarray(c.T)[..., -1]
        return jnp.sum(jnp.mean(T, axis=-1) * lat_w) / jnp.sum(lat_w)

    def _sample_loss(params, sample, n_steps):
        # ONE sample per graph -> memory = a single checkpointed rollout (the
        # multi-sample sum OOM'd at the 3-day phase); the epoch loop accumulates
        # gradients across samples instead.
        if is_nn:
            # NN weights are the traced leaves: build the physics fn INSIDE
            # the loss so gradients flow (same doctrine as the classical
            # params below). spectral_rollout's non-gated path jax.checkpoints
            # every step, so reverse-mode through a multi-day rollout fits.
            physics_fn = _make_nn_physics(params, grid)
            ic_state, tgt_carry, t_sfc_col, sic_col, doy = sample
            pred = spectral_rollout(
                ic_state, physics_fn, grid, sigma, pe, dt, n_steps,
                sponge, sfilt,
                forcing_base={
                    "T_sfc": t_sfc_col, "sic": sic_col,
                    "day_of_year": doy,
                    "seconds_of_day": jnp.asarray(0.0, jnp.float64),
                },
            )
        else:
            non_rad_fn, rad_fn = make_aimip_classical_spectral_physics(
                params, grid, dt,
                radiation=str(cfg.get("aimip_radiation", "rrtmgp")),
                rad_update_interval_steps=args.rad_update_interval,
                convection_scheme=str(cfg["aimip_convection"]),
                turbulence_scheme=str(cfg["aimip_turbulence"]),
                surface_bulk_scheme=str(
                    cfg.get("aimip_surface_bulk_scheme", "constant")),
                gwd_scheme=str(cfg["aimip_gwd"]),
                microphysics_scheme=str(cfg["aimip_microphysics"]),
                cloud_scheme=str(cfg.get("aimip_cloud", "xu_randall")),
                # Same suite waiver as the inference driver.
                allow_unfilled_families=parse_bool_flag(
    cfg.get("aimip_allow_unfilled_families", False)),
                land_mask=land_mask, split_rad=True,
            )
            ic_state, tgt_carry, sst_col, doy, ghg, o3 = sample
            pred = spectral_amip_rollout(
                ic_state, non_rad_fn, rad_fn, grid, sigma, pe, dt, n_steps,
                sst_col=sst_col, sizing_phys_state=sizing_ps, day_of_year_base=doy,
                rad_update_interval=args.rad_update_interval,
                sponge_factor=sponge, spectral_filter=sfilt,
                ghg_vmr=ghg, o3_vmr=o3, use_checkpoint=True,
            )
        state_loss = spectral_state_vs_carry_loss(
            pred, tgt_carry, grid, sigma, sigma_full, spec_cfg.loss_config,
        )
        # global-mean near-surface-T drift = energy-imbalance proxy (the
        # ACE2-style stability signal; grows with rollout length so the
        # curriculum's longer phases penalise the radiative runaway harder).
        drift = (_gm_surf_T_state(pred) - _gm_surf_T_carry(tgt_carry)) ** 2
        # aux breakdown so the log shows the STABILITY signal (drift term)
        # separately from forecast skill (state term).
        return state_loss + args.w_drift * drift, (state_loss, drift)

    # Pre-warm the RRTMGP optics-table cache OUTSIDE filter_jit with CONCRETE
    # params: the NetCDF gas-optics load is module-cached by static file paths,
    # but the first (inside-trace) call would hit the tracer state and crash with
    # TracerArrayConversionError. Mirrors _train_spectral_loop's warm-up.
    # NN variants have no RRTMGP -> nothing to warm.
    try:
        _warm = None if is_nn else make_aimip_classical_spectral_physics(
            params0, grid, dt,
            radiation=str(cfg.get("aimip_radiation", "rrtmgp")),
            rad_update_interval_steps=args.rad_update_interval,
            convection_scheme=str(cfg["aimip_convection"]),
            turbulence_scheme=str(cfg["aimip_turbulence"]),
            surface_bulk_scheme=str(
                cfg.get("aimip_surface_bulk_scheme", "constant")),
            gwd_scheme=str(cfg["aimip_gwd"]),
            microphysics_scheme=str(cfg["aimip_microphysics"]),
            cloud_scheme=str(cfg.get("aimip_cloud", "xu_randall")),
            # Same suite waiver (RRTMGP warm-up call).
            allow_unfilled_families=parse_bool_flag(
    cfg.get("aimip_allow_unfilled_families", False)),
            land_mask=land_mask, split_rad=True,
        )
        del _warm
    except Exception as exc:  # noqa: BLE001 - warm-up is best-effort
        logger.warning(f"RRTMGP optics warm-up raised {exc!r}; continuing")

    # Curriculum "rollout_days:epochs,..." (NeuralGCM-style: short rollouts
    # first, gradually extended so the radiative drift is exposed + penalised
    # progressively). One JIT compile per phase (samples + n_steps change).
    phases = []
    for tok in str(args.curriculum).split(","):
        rd, ep = tok.split(":")
        phases.append((int(rd), int(ep)))
    logger.info(f"curriculum: {phases}  lr={args.lr} w_drift={args.w_drift} ema={args.ema_decay}")

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.lr))
    opt_state = opt.init(eqx.filter(params0, eqx.is_inexact_array))
    params = params0
    # EMA on the inexact-array leaves only (static leaves recombined at save).
    # Initialised at params0 -> BIAS-CORRECTED at save (see below); the raw
    # running EMA over a short fine-tune is ~params0 (degenerate).
    p0_arrays = eqx.filter(params0, eqx.is_inexact_array)
    ema_arrays = p0_arrays
    ema_steps = 0
    _, static = eqx.partition(params0, eqx.is_inexact_array)

    broke = False
    best_drift_per_day = float("inf")
    for rd, ep in phases:
        if broke:
            break
        samples, _ = _build_samples(rd)
        n_steps_phase = rd * int(_SPD / dt)
        logger.info(f"phase rollout={rd}d epochs={ep} ({len(samples)} samples, {n_steps_phase} steps)")

        # Per-sample value+grad (one checkpointed rollout in memory at a time).
        @eqx.filter_jit
        def sample_vg(params, sample, _n=n_steps_phase):
            return eqx.filter_value_and_grad(
                lambda p: _sample_loss(p, sample, _n), has_aux=True,
            )(params)

        for epoch in range(ep):
            tot_loss = tot_state = tot_drift = 0.0
            acc = None
            for sample in samples:
                (loss_s, (sl, dr)), g = sample_vg(params, sample)  # 1 sample in mem
                tot_loss += float(loss_s)
                tot_state += float(sl)
                tot_drift += float(dr)
                acc = g if acc is None else jax.tree_util.tree_map(
                    lambda a, b: a + b, acc, g,
                )
            n = len(samples)
            acc = jax.tree_util.tree_map(lambda a: a / n, acc)  # mean grad
            # The measured drift belongs to THESE params (the rollouts above
            # used them). Snapshot before the update so _best_raw is the
            # checkpoint that actually produced the drift (codex MED), not the
            # one-gradient-step-ahead successor.
            params_at_drift = params
            updates, opt_state = opt.update(
                acc, opt_state, eqx.filter(params, eqx.is_inexact_array),
            )
            params = eqx.apply_updates(params, updates)
            cur = eqx.filter(params, eqx.is_inexact_array)
            ema_arrays = jax.tree_util.tree_map(
                lambda e, c: args.ema_decay * e + (1.0 - args.ema_decay) * c,
                ema_arrays, cur,
            )
            ema_steps += 1
            avg_loss = tot_loss / n
            logger.info(
                f"  [rd={rd}d] epoch {epoch:3d}: loss={avg_loss:.6f} "
                f"(state={tot_state / n:.6f} "
                f"drift={tot_drift / n:.6f} [K^2 @lead])"
            )
            # Per-epoch RAW-params checkpoint (atomic): long-rollout phases
            # run hours/epoch, so a SLURM TIMEOUT before the final save must
            # not lose the whole fine-tune. The bias-corrected EMA is only
            # meaningful at the end; the resumable artifact is the raw
            # params. (~10 KB — cheap.)
            _ck = args.out_ckpt.with_name(args.out_ckpt.stem + "_latest_raw.eqx")
            _tmp = _ck.with_suffix(".eqx.tmp")
            _ck.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(params, _tmp)
            os.replace(_tmp, _ck)
            # BEST-drift raw params across the whole fine-tune: a late-phase
            # divergence (colnn v3: 15d drift 4.2 -> 424 K^2) must not
            # overwrite the best stability state. Drift normalised by the
            # phase's rollout length so phases are comparable (K^2 per day
            # at lead ~ drift-rate^2 signal).
            _drift_per_day = (tot_drift / n) / max(rd, 1) ** 2
            if _drift_per_day < best_drift_per_day:
                best_drift_per_day = _drift_per_day
                _bk = args.out_ckpt.with_name(args.out_ckpt.stem + "_best_raw.eqx")
                _btmp = _bk.with_suffix(".eqx.tmp")
                save_checkpoint(params_at_drift, _btmp)  # the params that produced it
                os.replace(_btmp, _bk)
                logger.info(
                    f"  new BEST drift/day^2 {_drift_per_day:.4f} "
                    f"[K^2/d^2 @rd={rd}d] -> {_bk.name}"
                )
            if not np.isfinite(avg_loss):
                logger.error("non-finite loss; stopping")
                broke = True
                break

    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    # Bias-corrected EMA: with EMA_t = d^t p0 + (1-d) sum d^(t-i) p_i, the
    # p0-anchored estimate of the recent-parameter average is
    # (EMA_t - d^t p0) / (1 - d^t). Without this a short fine-tune's raw EMA
    # is ~params0 (0.999^11 -> 99% initial weight — found on the PoC run).
    if ema_steps > 0:
        d_t = float(args.ema_decay) ** ema_steps
        ema_arrays = jax.tree_util.tree_map(
            lambda e, p0: (e - d_t * p0) / (1.0 - d_t), ema_arrays, p0_arrays,
        )
    ema_params = eqx.combine(ema_arrays, static)
    save_checkpoint(ema_params, args.out_ckpt)
    save_checkpoint(params, args.out_ckpt.with_name(args.out_ckpt.stem + "_raw.eqx"))
    logger.info(
        f"Wrote fine-tuned params (bias-corrected EMA over {ema_steps} steps): "
        f"{args.out_ckpt}"
    )


if __name__ == "__main__":
    main()
